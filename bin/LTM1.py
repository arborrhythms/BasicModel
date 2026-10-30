"""Binary relations over explicitly tagged occurrences, symbols and relations."""
import math
import torch
from torch import nn

OCCURRENCE, SYMBOL, RELATION = 1, 2, 3
VERSION = 2


class LTM1(nn.Module):
    OCCURRENCE, SYMBOL, RELATION = OCCURRENCE, SYMBOL, RELATION
    columns = ('refs', 'ref_kinds', 'truth_plus', 'truth_minus', 'access_count', 'surprise', 'percept')

    def __init__(self, capacity=8192):
        super().__init__()
        self.capacity = int(capacity)
        if self.capacity < 2:
            raise ValueError('LTM1 capacity must include a reserved zero and a relation seat')
        self.allocate = None
        # One pair and two percept bands (where + when, four coordinates each)
        # per declared relation seat. No content vector and no growth on load.
        self.register_buffer('refs', torch.full((self.capacity, 2), -1, dtype=torch.long))
        self.register_buffer('ref_kinds', torch.zeros(self.capacity, 2, dtype=torch.int8))
        self.register_buffer('percept', torch.zeros(self.capacity, 8))
        self.register_buffer('truth_plus', torch.zeros(self.capacity))
        self.register_buffer('truth_minus', torch.zeros(self.capacity))
        self.register_buffer('access_count', torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer('surprise', torch.full((self.capacity,), -1.))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('layout_version', torch.tensor(VERSION, dtype=torch.long))
        self._index_version = None
        self._symbol_epoch = 0
        self._pairs, self._parts, self._wholes, self._references = {}, {}, {}, {}

    def __len__(self):
        return int(self.count)

    @property
    def revision(self):
        return (id(self.refs), self.refs._version, self.ref_kinds._version,
                self.truth_plus._version, self.truth_minus._version, self.access_count._version, len(self))

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
        version = (id(self.refs), self.refs._version, self.ref_kinds._version, len(self))
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
            for ref in {left, right}:
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
            for reference in self.references(row):
                references.add(reference)
                if reference[0] == RELATION:
                    if not self.contains(reference[1]):
                        raise ValueError('dangling relation reference')
                    pending.append(reference[1])
        return references

    @torch.no_grad()
    def add(self, part, whole, *, ref_kinds=(SYMBOL, SYMBOL), truth_plus=0., truth_minus=0.,
            when=None, where=None, surprise=-1., witness=True, symbol=None):
        left, right = self.reference(part, ref_kinds[0]), self.reference(whole, ref_kinds[1])
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in (truth_plus, truth_minus)):
            raise ValueError('relation evidence has two independent poles in [0, 1]')
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
        if row is None:
            if symbol is not None:
                row = int(symbol)
            elif self.allocate is None:
                free = self.ref_kinds[:, 0].eq(0).nonzero().flatten()
                free = free[free > 0]
                if not len(free):
                    raise OverflowError('LTM1 capacity exhausted; forgetting is required')
                row = int(free[0])
            else:
                row = int(self.allocate())
            if not 0 < row < self.capacity:
                raise OverflowError('LTM1 handle exceeds its constructed capacity')
            if self.contains(row):
                raise RuntimeError('a live relation cannot be reassigned')
            self.refs[row] = self.refs.new_tensor((left[1], right[1]))
            self.ref_kinds[row] = self.ref_kinds.new_tensor((left[0], right[0]))
            self.count += 1
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
            getattr(self, name)[erase] = -1 if name in ('refs', 'surprise') else 0
        self.count.fill_(len(indices))

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
        for name in ('truth_plus', 'truth_minus'):
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
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
        self._index_version = None
        self._symbol_epoch += 1
        self._index()
