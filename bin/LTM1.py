"""Binary relations over explicitly tagged occurrences, symbols and relations."""
import math
import torch
from torch import nn

OCCURRENCE, SYMBOL, RELATION = 1, 2, 3
VERSION = 5
# Reserved outside both address inventories. It never consumes a learnable seat.
NULL_CONCEPT = torch.iinfo(torch.int64).max


class LTM1(nn.Module):
    OCCURRENCE, SYMBOL, RELATION = OCCURRENCE, SYMBOL, RELATION
    columns = ('refs', 'ref_kinds', 'truth_plus', 'truth_minus', 'access_count', 'surprise', 'percept', 'effective_plus', 'effective_minus', 'operator', 'grammar_operation')

    def __init__(self, capacity=8192):
        super().__init__()
        self.capacity = int(capacity)
        if self.capacity < 2:
            raise ValueError('LTM1 capacity must include a reserved zero and a relation seat')
        self.allocate = None
        self.handle_reserved = None
        # One pair and two percept bands (where + when, four coordinates each)
        # per declared relation seat; there is no growth on load.
        self.register_buffer('refs', torch.full((self.capacity, 2), -1, dtype=torch.long))
        self.register_buffer('ref_kinds', torch.zeros(self.capacity, 2, dtype=torch.int8))
        self.register_buffer('percept', torch.zeros(self.capacity, 8))
        self.register_buffer('truth_plus', torch.zeros(self.capacity))
        self.register_buffer('truth_minus', torch.zeros(self.capacity))
        self.register_buffer('effective_plus', torch.zeros(self.capacity))
        self.register_buffer('effective_minus', torch.zeros(self.capacity))
        self.register_buffer('operator', torch.full((self.capacity,), -1, dtype=torch.long))
        self.register_buffer('grammar_operation', torch.full((self.capacity,), -1, dtype=torch.long))
        self.operation_catalog = ()
        from TrustGates import TrustGates
        self.gates = TrustGates()
        self._occupation_owner = None
        self._refreshing_effective = False
        self._primitive_evidence = set()
        self.register_buffer('relation_order', torch.full((self.capacity,), -1, dtype=torch.long))
        self._order_revision = None
        self._effective_source = None
        self._concept_owner = None
        self.register_buffer('access_count', torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer('surprise', torch.full((self.capacity,), -1.))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('layout_version', torch.tensor(VERSION, dtype=torch.long))
        self._index_version = None
        self._symbol_epoch = 0
        self.epoch_start = None
        self.epoch_handles = set()
        self._vacant_epoch_handles = []
        self._pairs, self._parts, self._wholes, self._references = {}, {}, {}, {}

    def _reserved_handle(self, row):
        if self.handle_reserved is not None and self.handle_reserved(row):
            return True
        if self.rows_referencing(SYMBOL, row):
            return True
        store_ref = getattr(self, '_occurrence_store', None)
        store = None if store_ref is None else store_ref()
        if store is None:
            return False
        if row in store._cover_symbols.values() or bool((store.recurrence_symbols == row).any()):
            return True
        names = (('sym', row), ('name', row))
        return any(ref in names for context in store._semantic_rows.values()
                   for ref in (*context.get('role_refs', ()), dict(context.get('bindings', ())).get('_identity')))

    @torch.no_grad()
    def begin_epoch(self):
        """Restore the starting rows, including access counts and both evidence lanes."""
        if self.epoch_start is None:
            self.epoch_start = {name: value.detach().cpu().clone()
                                for name, value in self.state_dict().items()}
            return set()
        before = set(self.indices().tolist())
        state = {name: self.epoch_start[name] for name in self.state_dict()}
        # Learned readiness is monotone across epoch row restoration.
        state['_extra_state'] = self.get_extra_state()
        self.load_state_dict(state)
        self.epoch_start.pop('_initial_observations', None)
        occupied = set(self.indices().tolist())
        # Only handles reserved by this relation allocator are available again.
        # Explicit symbol handles belong to their concepts across epochs.
        self._vacant_epoch_handles = sorted(self.epoch_handles - occupied, reverse=True)
        self._symbol_epoch += 1
        self._index_version = None
        self._index()
        return before - occupied

    def __len__(self):
        return int(self.count)

    @property
    def revision(self):
        return (id(self.refs), self.refs._version, self.ref_kinds._version,
                self.truth_plus._version, self.truth_minus._version, self.access_count._version, len(self))

    @property
    def _topology_revision(self):
        return (id(self.refs), self.refs._version, self.ref_kinds._version, len(self))

    @staticmethod
    def ordered_rows(refs, kinds):
        """Topological order for a live slab or an epoch snapshot."""
        occupied = set(kinds[:, 0].ne(0).nonzero().flatten().tolist())
        visiting, visited, ordered = set(), set(), []
        for root in sorted(occupied):
            pending = [(root, False)]
            while pending:
                row, finished = pending.pop()
                if finished:
                    visiting.remove(row)
                    visited.add(row)
                    ordered.append(row)
                elif row in visiting:
                    raise ValueError('cyclic relation reference')
                elif row not in visited:
                    if row not in occupied:
                        raise ValueError('dangling relation reference')
                    visiting.add(row)
                    pending.append((row, True))
                    pending.extend((value, False) for kind, value in
                                   zip(kinds[row].tolist(), refs[row].tolist()) if kind == RELATION)
        return ordered

    @torch.no_grad()
    def _sort(self):
        """Keep the constructed order slab in dependency order; handles stay stable."""
        if self._order_revision == self._topology_revision:
            return
        ordered = self.ordered_rows(self.refs, self.ref_kinds)
        self.relation_order.fill_(-1)
        if ordered:
            self.relation_order[:len(ordered)] = self.relation_order.new_tensor(ordered)
        self._order_revision = self._topology_revision

    def configure_operations(self, occupations):
        import weakref
        if occupations is None or not occupations.operations:
            return
        catalogue = occupations.operations
        if self.operation_catalog and catalogue != self.operation_catalog:
            raise ValueError('relation operation catalogue cannot change after attachment')
        self.operation_catalog = catalogue
        self._occupation_owner = weakref.ref(occupations)
        self._effective_source = None

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(operation_catalog=self.operation_catalog, gate_open=dict(self.gates.open)))

    @torch.no_grad()
    def clear_operation(self, row):
        """Naming makes a live pair independent of learned-operation authority."""
        if not self.contains(row):
            raise IndexError('unavailable relation')
        self.grammar_operation[row] = -1
        self._effective_source = None

    def _check_extra_state(self, value):
        from IndependentComponents import _unpack_state
        state = _unpack_state(value)
        if not isinstance(state, dict) or 'gate_open' not in state:
            raise ValueError('LTM1 checkpoint requires gate state; regenerate pre-gate memories')
        if tuple(state['operation_catalog']) != self.operation_catalog:
            raise ValueError('relation operation catalogue does not match the loaded grammar')
        return self.gates.validate_state(state['gate_open'])

    def set_extra_state(self, value):
        self.gates.restore(self._check_extra_state(value))
        self._effective_source = None

    def gate_measurements(self):
        """Report readiness; only refresh_effective can open a gate."""
        self.refresh_effective()
        owner = None if self._concept_owner is None else self._concept_owner()
        occupations = None if self._occupation_owner is None else self._occupation_owner()
        values = self.gates.measurements(owner, occupations)
        result = {name: dict(open=self.gates.open[name], measurement=values[name]) for name in self.gates.names}
        if occupations is not None and 'lift' in occupations.operations:
            result['clause']['arguments'] = occupations.operation('lift')['argument_measurements']
        return result

    @torch.no_grad()
    def refresh_effective(self):
        """Propagate changed concept and operation authority in dependency order."""
        if self._refreshing_effective:
            return
        self._sort()
        owner = None if self._concept_owner is None else self._concept_owner()
        trust = None if owner is None else owner.similarity_codebook.trust
        occupations = None if self._occupation_owner is None else self._occupation_owner()
        revision = (self._topology_revision, self.truth_plus._version, self.truth_minus._version,
                    self.operator._version, self.grammar_operation._version,
                    None if occupations is None else occupations.revision,
                    None if trust is None else trust.revision, self.gates.revision)
        if revision == self._effective_source:
            return
        from ConceptTrust import positive_length
        def concept(identity):
            return 1. if owner is None else positive_length(owner, identity)
        def endpoint(kind, value):
            if kind == RELATION:
                return max(float(self.effective_plus[value]), float(self.effective_minus[value]))
            if kind == SYMBOL and owner is not None and value in owner._concept_allocator.placement:
                return concept(value)
            return 1.
        def write(row, operation_trust=1.):
            scale = min(endpoint(*ref) for ref in self.references(row))
            operator = int(self.operator[row])
            if operator != -1:
                scale = min(scale, concept(operator))
            scale = min(scale, operation_trust)
            left, right = self.references(row)
            named = (owner is not None and left[0] == right[0] == SYMBOL
                     and left[1] in owner._concept_allocator.named_identities.get(right[1], ()))
            operation = int(self.grammar_operation[row])
            name = None if operation == -1 else self.operation_catalog[operation]
            scale = min(scale, self.gates.cap(named=named, operation=name))
            self.effective_plus[row] = self.truth_plus[row] * scale
            self.effective_minus[row] = self.truth_minus[row] * scale
        rows = self.relation_order[:len(self)].tolist()
        # Occupation distributions use denotation to select an order. Native
        # named/membership pairs have no grammar factor: make those independent
        # authorities available before asking the categorizer for its factors.
        from PartsOfSpeech import FIXED_OPERATIONS
        self._primitive_evidence = {row for row in rows
            if (int(self.grammar_operation[row]) == -1 or
                self.operation_catalog[int(self.grammar_operation[row])] in FIXED_OPERATIONS)
            and all(kind != RELATION for kind, _ in self.references(row))}
        self._refreshing_effective = True
        try:
            for row in self._primitive_evidence:
                write(row)
            # Empty banks keep their declared start. Opening is an authority
            # read over live rows, never a construction-time side effect.
            if rows and not all(self.gates.open.values()):
                for name in self.gates.names:
                    if self.gates.open[name]:
                        continue
                    measured = self.gates.measurements(owner, occupations)[name]
                    if self.gates.advance(name, measured):
                        self._effective_source = None
                        # A newly admitted name can change the categorizer's
                        # order selection for the next measurement/factor.
                        for row in self._primitive_evidence:
                            write(row)
            factors = {}
            for row in rows:
                operation = int(self.grammar_operation[row])
                if operation != -1 and operation not in factors:
                    if occupations is None:
                        raise ValueError('relation operation has no loaded occupation owner')
                    factors[operation] = occupations.operation(self.operation_catalog[operation])['trust']
            for row in rows:
                if row not in self._primitive_evidence:
                    write(row, factors.get(int(self.grammar_operation[row]), 1.))
        finally:
            self._refreshing_effective = False
        self._effective_source = (*revision[:-2], None if trust is None else trust.revision, self.gates.revision)

    @property
    def effective_revision(self):
        self.refresh_effective()
        return (id(self), self.effective_plus._version, self.effective_minus._version)

    def evidence(self, row):
        """The single authority read: effective support and counter-evidence."""
        if not self.contains(row):
            raise IndexError('unavailable relation')
        if self._refreshing_effective and row not in self._primitive_evidence:
            raise ValueError('operation occupation depends cyclically on learned relation authority')
        self.refresh_effective()
        return self.effective_plus[row], self.effective_minus[row]

    @property
    def symbol_revision(self):
        """Symbol topology and evidence; chain appends cannot expire membership."""
        return (id(self), self._symbol_epoch)

    @staticmethod
    def reference(value, kind=SYMBOL):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError('a relation reference requires its tag and value')
            kind, value = value
        kind, value = int(kind), int(value)
        if kind not in (OCCURRENCE, SYMBOL, RELATION) or value in (-1, 0) or (kind != OCCURRENCE and value < 0):
            raise ValueError('invalid tagged relation reference')
        return kind, value

    def references(self, row):
        if not self.contains(row):
            raise IndexError('unavailable relation')
        return tuple(zip(self.ref_kinds[int(row)].tolist(), self.refs[int(row)].tolist()))

    def _index(self):
        version = (id(self.refs), self.refs._version, self.ref_kinds._version, self.operator._version, len(self))
        if self._index_version == version:
            return
        pairs, parts, wholes, refs = {}, {}, {}, {}
        for row in self.indices().tolist():
            left, right = (self.reference(ref) for ref in self.references(row))
            if (left, right) in pairs:
                raise ValueError('duplicate tagged relation')
            pairs[left, right] = row
            parts.setdefault(right, []).append(left)
            wholes.setdefault(left, []).append(right)
            dependencies = {left, right}
            if int(self.operator[row]) != -1:
                dependencies.add((SYMBOL, int(self.operator[row])))
            for ref in dependencies:
                refs.setdefault(ref, []).append(row)
        self._pairs, self._parts, self._wholes, self._references = pairs, parts, wholes, refs
        self._index_version = version

    def indices(self):
        return self.ref_kinds[:, 0].ne(0).nonzero().flatten()

    def contains(self, row):
        return 0 < int(row) < self.capacity and bool(self.ref_kinds[int(row)].ne(0).all())

    def find(self, part, whole, *, ref_kinds=(SYMBOL, SYMBOL)):
        self._index()
        return self._pairs.get((self.reference(part, ref_kinds[0]), self.reference(whole, ref_kinds[1])))

    def parts(self, whole):
        self._index()
        values = tuple(self._parts.get(self.reference(whole), ()))
        return values if isinstance(whole, tuple) else tuple(v for k, v in values if k == SYMBOL)

    def wholes(self, part):
        self._index()
        values = tuple(self._wholes.get(self.reference(part), ()))
        return values if isinstance(part, tuple) else tuple(v for k, v in values if k == SYMBOL)

    def rows_referencing(self, kind, value):
        self._index()
        return tuple(self._references.get(self.reference(value, kind), ()))

    def stratum(self, row):
        return 2 if any(kind == RELATION for kind, _ in self.references(row)) else 1

    def protected_references(self, rows=None):
        """All dependencies, transitively; freeze this set before forgetting."""
        pending = list(self.indices().tolist() if rows is None else rows)
        seen, references = set(), set()
        while pending:
            row = pending.pop()
            if row in seen:
                continue
            seen.add(row)
            if int(self.operator[row]) != -1:
                references.add((SYMBOL, int(self.operator[row])))
            for reference in self.references(row):
                references.add(reference)
                if reference[0] == RELATION:
                    if not self.contains(reference[1]):
                        raise ValueError('dangling relation reference')
                    pending.append(reference[1])
        return references

    @torch.no_grad()
    def add(self, part, whole, *, ref_kinds=(SYMBOL, SYMBOL), truth_plus=0., truth_minus=0.,
            when=None, where=None, surprise=-1., witness=True, symbol=None,
            operator=None, grammar_operation=None):
        left, right = self.reference(part, ref_kinds[0]), self.reference(whole, ref_kinds[1])
        if (SYMBOL, NULL_CONCEPT) in (left, right):
            raise ValueError('a null-concept cannot be a relation endpoint')
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in (truth_plus, truth_minus)):
            raise ValueError('relation evidence has two independent poles in [0, 1]')
        if operator is not None and (type(operator) is not int or operator == 0 or operator < -1):
            raise ValueError('relation operator must be a word identity or -1')
        if grammar_operation is not None and (type(grammar_operation) is not int
                or not -1 <= grammar_operation < len(self.operation_catalog)):
            raise ValueError('relation operation must index the loaded catalogue or be -1')
        if not math.isfinite(float(surprise)) or not (surprise == -1. or 0. <= surprise <= 1.):
            raise ValueError('relation surprise must be unknown (-1) or in [0, 1]')
        bands = {}
        for name, value in (('when', when), ('where', where)):
            if value is not None:
                value = torch.as_tensor(value, device=self.percept.device, dtype=self.percept.dtype)
                if value.shape != (4,) or not bool(torch.isfinite(value).all()):
                    raise ValueError('percept bands require four finite coordinates')
                bands[name] = value
        for kind, value in (left, right):
            if kind == RELATION and not self.contains(value):
                raise ValueError('relation child must exist before its parent')
        row = self.find(left, right)
        self._sort()
        new_row = row is None
        if row is not None and operator is not None and int(self.operator[row]) != operator:
            raise ValueError('one tagged relation cannot have different lexical operators')
        # The first composition owns this provenance tag. Later closes merge
        # their poles without replacing that origin (including a cleared tag).
        if row is None:
            if symbol is not None:
                row = int(symbol)
            elif self.allocate is None:
                free = self.ref_kinds[:, 0].eq(0).nonzero().flatten()
                free = free[free > 0]
                row = next((cid for cid in free.tolist() if not self._reserved_handle(cid)), None)
                if row is None:
                    raise OverflowError('LTM1 capacity exhausted; forgetting is required')
            else:
                # A name or seat can arrive after the reset or checkpoint.
                # Test its live owners at the actual point of reuse.
                while self._vacant_epoch_handles:
                    candidate = self._vacant_epoch_handles[-1]
                    if not self.contains(candidate) and not self._reserved_handle(candidate):
                        break
                    self._vacant_epoch_handles.pop()
                row = (self._vacant_epoch_handles.pop() if self._vacant_epoch_handles
                       else int(self.allocate()))
                self.epoch_handles.add(row)
            if not 0 < row < self.capacity:
                raise OverflowError('LTM1 handle exceeds its constructed capacity')
            if self.contains(row):
                raise RuntimeError('a live relation cannot be reassigned')
            self.refs[row] = self.refs.new_tensor((left[1], right[1]))
            self.ref_kinds[row] = self.ref_kinds.new_tensor((left[0], right[0]))
            self.relation_order[len(self)] = row
            self.count += 1
            self._order_revision = self._topology_revision
            self.operator[row] = -1 if operator is None else operator
            self.grammar_operation[row] = -1 if grammar_operation is None else grammar_operation
            if right[0] == SYMBOL and (left[0] == SYMBOL or right[1] in getattr(self, '_class_rows', ())):
                self._symbol_epoch += 1
        elif symbol is not None and int(symbol) != row:
            raise ValueError('a tagged pair already has a different handle')
        positive = max(float(self.truth_plus[row]), float(truth_plus))
        negative = max(float(self.truth_minus[row]), float(truth_minus))
        if (right[0] == SYMBOL and (left[0] == SYMBOL or right[1] in getattr(self, '_class_rows', ())) and
                (positive != float(self.truth_plus[row]) or negative != float(self.truth_minus[row]))):
            self._symbol_epoch += 1
        self.truth_plus[row] = positive
        self.truth_minus[row] = negative
        self.access_count[row] += int(witness)
        if surprise >= 0:
            self.surprise[row] = float(surprise)
        for name, value in bands.items():
            self.percept[row, 4:].copy_(value) if name == 'when' else self.percept[row, :4].copy_(value)
        if new_row and self.epoch_start is not None:
            self.epoch_start.setdefault('_initial_observations', {})[row] = (
                int(self.access_count[row]), float(self.surprise[row]))
        return row

    @torch.no_grad()
    def retain(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.refs.device)
        selected = set(indices.tolist())
        if len(selected) != len(indices) or any(not self.contains(i) for i in selected):
            raise ValueError('invalid LTM1 retention selection')
        if any(kind == RELATION and child not in selected for row in selected for kind, child in self.references(row)):
            raise ValueError('cannot discard a retained relation dependency')
        erase = torch.ones(self.capacity, dtype=torch.bool, device=self.refs.device)
        erase[indices] = False
        if bool((erase & self.ref_kinds[:, 1].eq(SYMBOL)).any()):
            self._symbol_epoch += 1
        for name in self.columns:
            getattr(self, name)[erase] = -1 if name in ('refs', 'surprise', 'operator', 'grammar_operation') else 0
        self.count.fill_(len(indices))
        self._sort()
        self.refresh_effective()

    def remove(self, part, whole):
        row = self.find(part, whole)
        if row is not None:
            self.retain([i for i in self.indices().tolist() if i != row])

    def remove_symbol(self, symbol):
        removed = set(self.rows_referencing(SYMBOL, symbol))
        if self.contains(symbol):
            removed.add(int(symbol))
        pending = list(removed)
        while pending:
            for parent in self.rows_referencing(RELATION, pending.pop()):
                if parent not in removed:
                    removed.add(parent); pending.append(parent)
        self.retain([i for i in self.indices().tolist() if i not in removed])

    def row(self, index):
        if not self.contains(index):
            raise IndexError('unavailable relation')
        return {name: getattr(self, name)[int(index)] for name in self.columns}

    def readings(self, index, types):
        left_ref, right_ref = self.references(index)
        left, right = set(types(left_ref)), set(types(right_ref))
        readings = []
        if 'situation' in left and 'situation' in right:
            readings.append('conditional')
        if 'situation' in left and 'speech_individual' in right:
            readings.append('attribution')
        if left_ref != right_ref and 'object' in left and 'word' in right:
            readings.append('definition')
        if 'kind' in left and 'kind' in right:
            readings.append('subsumption')
        if 'singleton' in right and 'definition' not in readings:
            readings.append('equality')
        if 'spatial' in left and 'spatial' in right:
            readings.append('part')
        return tuple(readings)

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        for name in ('layout_version', 'count'):
            value = state.get(prefix + name)
            if value is None or value.shape != () or value.dtype != torch.long:
                raise ValueError('invalid relation checkpoint version or count')
        if int(state.get(prefix + 'layout_version', -1)) != VERSION or prefix + 'ref_kinds' not in state:
            raise ValueError('unsupported LTM1 layout checkpoint version; regenerate pre-layout memories')
        if prefix + '_extra_state' not in state:
            raise ValueError('LTM1 checkpoint requires gate state; regenerate pre-gate memories')
        self._check_extra_state(state[prefix + '_extra_state'])
        refs, kinds = state.get(prefix + 'refs'), state[prefix + 'ref_kinds']
        if refs is None or refs.shape != self.refs.shape or kinds.shape != self.ref_kinds.shape:
            raise ValueError('LTM1 checkpoint capacity differs from construction')
        if kinds.dtype != torch.int8 or refs.dtype != torch.long:
            raise ValueError('invalid relation reference dtypes')
        occupied = kinds.ne(0).any(-1)
        valid = ((kinds >= OCCURRENCE) & (kinds <= RELATION)).all(-1)
        if bool((occupied & ~valid).any()) or bool((~occupied & refs.ne(-1).any(-1)).any()):
            raise ValueError('invalid restored reference tags')
        seen = set()
        for row in occupied.nonzero().flatten().tolist():
            if row == 0:
                raise ValueError('zero is not a relation handle')
            pair = tuple(self.reference((int(k), int(v))) for k, v in zip(kinds[row], refs[row]))
            if pair in seen:
                raise ValueError('duplicate restored tagged relation')
            seen.add(pair)
            if any(k == RELATION and (v >= self.capacity or not bool(occupied[v])) for k, v in pair):
                raise ValueError('dangling restored relation reference')
        if int(occupied.sum()) != int(state.get(prefix + 'count', -1)):
            raise ValueError('relation count disagrees with occupied rows')
        visiting, visited = set(), set()
        for root in occupied.nonzero().flatten().tolist():
            pending = [(root, False)]
            while pending:
                row, finished = pending.pop()
                if finished:
                    visiting.remove(row)
                    visited.add(row)
                elif row in visiting:
                    raise ValueError('cyclic restored relation reference')
                elif row not in visited:
                    visiting.add(row)
                    pending.append((row, True))
                    pending.extend((value, False) for kind, value in
                                   zip(kinds[row].tolist(), refs[row].tolist()) if kind == RELATION)
        for name in self.columns:
            value = state.get(prefix + name)
            if value is None or value.shape != getattr(self, name).shape or value.dtype != getattr(self, name).dtype:
                raise ValueError('relation checkpoint column differs from construction')
        for name in ('truth_plus', 'truth_minus', 'effective_plus', 'effective_minus'):
            values = state[prefix + name]
            if not bool(torch.isfinite(values).all() & (values >= 0).all() & (values <= 1).all()):
                raise ValueError('invalid restored relation evidence')
        if state[prefix + 'percept'].shape != self.percept.shape or not bool(torch.isfinite(state[prefix + 'percept']).all()):
            raise ValueError('invalid restored relation percept')
        surprise = state[prefix + 'surprise']
        if not bool(torch.isfinite(surprise).all() & ((surprise == -1) | ((surprise >= 0) & (surprise <= 1))).all()):
            raise ValueError('invalid restored relation surprise')
        if bool((state[prefix + 'access_count'] < 0).any()):
            raise ValueError('invalid restored relation access count')
        operators = state[prefix + 'operator']
        if bool(((operators < -1) | (operators == 0)).any()):
            raise ValueError('invalid restored relation operator')
        operations = state.get(prefix + 'grammar_operation')
        if (operations is None or operations.shape != self.grammar_operation.shape or operations.dtype != torch.long
                or bool(((operations < -1) | (operations >= len(self.operation_catalog))).any())):
            raise ValueError('invalid restored grammar operation catalogue index')
        order = state.get(prefix + 'relation_order')
        if order is None or order.shape != self.relation_order.shape or order.dtype != torch.long:
            raise ValueError('relation order differs from construction')
        count = int(occupied.sum())
        ordered = order[:count].tolist()
        if set(ordered) != set(occupied.nonzero().flatten().tolist()) or len(set(ordered)) != count or bool(order[count:].ne(-1).any()):
            raise ValueError('invalid restored relation order')
        before = set()
        for row in ordered:
            if any(k == RELATION and v not in before for k, v in zip(kinds[row].tolist(), refs[row].tolist())):
                raise ValueError('restored relation order must put parts before wholes')
            before.add(row)
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
        self._order_revision = self._topology_revision
        self._effective_source = None
        self._index_version = None
        self._symbol_epoch += 1
        self._index()
