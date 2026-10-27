"""Host-side clause admission and references to LTM occurrences."""
from dataclasses import dataclass, field, replace
from contextlib import contextmanager
from typing import Any

import torch

from Meaning import ConceptualMeaning


def predicate_identity(name):
    """A stable slot address for a grammar operation, never an inventory row.

    Only occurrences in the store give this address content. Its grammar name
    ties those occurrences together without looking at their vector values.
    The tag is disjoint from concept IDs and occurrence-addressed LTM rows.
    """
    import hashlib
    name = 'part' if name in ('whole', 'equal', 'generic') else str(name)
    return (1 << 61) | (int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'big') & ((1 << 61) - 1))


def predicate_relation(reference):
    for name in ('part', 'implies'):
        if reference == predicate_identity(name):
            return name
    return 'operator'


def predicate_point(name, like):
    """The same identity code for a predicate in a closing or a question."""
    from References import symbol_code
    return symbol_code(predicate_identity(name), like.shape[-1],
                       n_where=0, n_when=0).to(like)


@dataclass(frozen=True, eq=False)
class ClausePredicate:
    """An open reading's predicate value and identity, awaiting its row slot."""
    point: Any = field(repr=False)
    name: str

    def __post_init__(self):
        if (not torch.is_tensor(self.point) or self.point.ndim != 1
                or not bool(torch.isfinite(self.point).all())):
            raise ValueError('a clause predicate requires one finite grammar value')
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('a clause predicate requires its grammar operation name')

    @property
    def identity(self):
        return predicate_identity(self.name)


@dataclass(frozen=True, eq=False)
class Composition:
    """One temporary committed operation; its pair is consumed at the close."""
    point: Any = field(repr=False)
    children: tuple = ()
    reference: int = -1
    operation: str | None = None
    head_role: int = 0
    order: int = 0
    leaves: tuple = ()
    binding: Any = field(default=None, repr=False)
    word_id: int = -1
    formal_name: str | None = None


@dataclass(frozen=True, eq=False)
class Clause:
    """A completed field and its temporary pre-fusion prediction target.

    The actual field is ``slots``: one fused point or three relative slots.
    ``meaning`` also carries the pre-fusion target while an idea is being
    written; that target is never stored with the idea. Local child addresses
    name already completed fields, not operations. Equality can supply two
    companion fields before the common writer admits them.
    """
    meaning: ConceptualMeaning
    point: Any = field(default=None, repr=False)
    relation: str | None = None
    refs: tuple = (-1, -1, -1)
    children: tuple = ()
    companions: tuple = ()
    subject_word_id: int = -1
    where: Any = field(default=None, repr=False)
    when: Any = field(default=None, repr=False)
    factored_refs: tuple | None = None
    order: int = 0
    evidence: tuple = (0., 0.)
    # Temporary constituent request. The writer stores only the resolved
    # occurrence reference, never this reading/word-position provenance.
    reference_request: Any = field(default=None, repr=False)
    composition: Composition | None = field(default=None, repr=False)
    minted: bool = False
    identity: Any = field(default=None, repr=False)

    def __post_init__(self):
        import math
        if type(self.order) is not int or self.order < 0:
            raise ValueError('a clause order must be a nonnegative integer')
        if len(self.evidence) != 2 or any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1
                                        for x in self.evidence):
            raise ValueError('clause evidence requires two finite poles in [0, 1]')
        if self.relation not in (None, 'part', 'whole', 'equal', 'implies', 'operator'):
            raise ValueError('unknown clause relation')
        if len(self.refs) != 3:
            raise ValueError('a clause has three reference slots')
        if self.relation is None:
            if (not torch.is_tensor(self.point) or self.point.ndim != 1
                    or self.point.shape != self.meaning.roles[0].shape
                    or not bool(torch.isfinite(self.point).all())):
                raise ValueError('an absolute clause requires its grammar-fused point')
        elif self.point is not None:
            raise ValueError('a relative clause has no fused point')
        for name in ('where', 'when'):
            value = getattr(self, name)
            if value is not None and (not torch.is_tensor(value) or value.shape != (4,)
                                      or not bool(torch.isfinite(value).all())):
                raise ValueError('an ended field has one finite four-coordinate band')
        if self.factored_refs is not None and len(self.factored_refs) != 3:
            raise ValueError('a factored clause has three reference slots')
        for reference in (*self.refs, *(self.factored_refs or ())):
            if isinstance(reference, ClausePredicate):
                continue
            if isinstance(reference, tuple):
                if (len(reference) != 2 or reference[0] != 'clause'
                        or type(reference[1]) is not int
                        or not 0 <= reference[1] < len(self.children)):
                    raise ValueError('unavailable child clause')
            elif type(reference) is not int or reference == 0:
                raise ValueError('clause references are positive shared ids or -1')
        if any(not isinstance(child, Clause) for child in (*self.children, *self.companions)):
            raise TypeError('clause children must be grammatical clauses')
        object.__setattr__(self, 'meaning', replace(self.meaning,
            sentence_kind='idea' if self.relation is None else 'relation'))
    @property
    def slots(self):
        return self.point[None] if self.point is not None else self.meaning.roles


def attach_clause_index(model, store, space):
    """Bind the LTM occurrence owner and the existing concept codebook."""
    from Queries import _basis, _existing_row
    import weakref
    space_ref = weakref.ref(space)
    from Spaces import _concept_alloc_of
    allocator = _concept_alloc_of(space)
    object.__setattr__(store, 'parts', allocator.parts)
    object.__setattr__(store, '_concept_owner', space_ref)
    language = getattr(model, 'languageSpace', None)
    occupations = getattr(language, 'parts_of_speech', None)
    if occupations is not None:
        # Staged word IDs and kept clauses use this registry's allocator;
        # a terminal stage's local IDs need not name the same concepts.
        object.__setattr__(occupations, '_space', space_ref)
    object.__setattr__(store, '_occupation_owner',
                       None if occupations is None else weakref.ref(occupations))
    object.__setattr__(store, '_part_read_rules', tuple(
        getattr(language, '_compose_binary_rules', ())))
    # The declared inverse catalogue is fixed when the owner is attached.
    # Evaluation keeps no reconstructed tree or additional durable payload.
    object.__setattr__(store, '_evaluation_kernels', tuple(
        getattr(language, '_generate_binary_ops', ())))

    store_ref = weakref.ref(store)
    object.__setattr__(space, '_clause_store_ref', store_ref)
    present = getattr(model, 'present', None)
    if callable(present) and hasattr(model, 'when_time'):
        object.__setattr__(store, '_model_present', present)

    def concept_point(cid):
        owner = space_ref()
        try:
            return owner.similarity_codebook.lookup_rows(_existing_row(owner, ('sym', cid)))
        except ValueError:
            return None

    store.configure_clause_index(concept_point=concept_point,
        predicate_kind=predicate_relation, preflight=lambda count: space_ref()._preflight_concept_allocation(count, context='clause taxonomy'))
    def reference_resolved(row, role, reference):
        owner = space_ref()
        current = getattr(owner, '_model', None)
        components = getattr(owner, 'components', None)
        if (current is not None and components is not None
                and getattr(current, '_sentence_training', False)):
            components.reference_close.resolved(row, role, reference, current.errors)
    object.__setattr__(store, '_reference_resolved', reference_resolved)
    from ConceptIndex import rebuild_ltm_taxonomy
    object.__setattr__(store, '_rebuild_taxonomy',
        lambda: rebuild_ltm_taxonomy(space_ref(), store))


class ClauseRows:
    """One writer for completed clauses; vector and row readers share its rows."""

    def configure_clause_index(self, *, allocate=None, concept_point, predicate_kind=None, on_part=None, preflight=None):
        object.__setattr__(self, '_allocate_clause_row', allocate)
        object.__setattr__(self, '_concept_point', concept_point)
        object.__setattr__(self, '_predicate_kind', predicate_kind)
        object.__setattr__(self, '_index_part_row', on_part)
        object.__setattr__(self, '_preflight_clause_rows', preflight)

    @torch.no_grad()
    def part_row(self, symbol, *, reading=None):
        """Build a temporary working view from the two tagged references."""
        from LTM1 import OCCURRENCE, SYMBOL, RELATION
        from ChainRead import envelope, point, head_idea
        from Spaces import _concept_alloc_of
        from ThoughtReferences import with_slots
        symbol = int(symbol)
        part = self.parts.row(symbol)
        owner = self._concept_owner()
        alloc = _concept_alloc_of(owner)
        def types(reference):
            kind, value = reference
            result = set(alloc.symbol_types.get(value, ())) if kind == SYMBOL else set()
            if (value if kind == SYMBOL else reference) in alloc.singletons:
                result.add('singleton')
            if kind == SYMBOL and alloc.order_of(value) > 1:
                result.add('kind')
            if kind == OCCURRENCE:
                row = self.index_of_row(value)
                if row is not None:
                    if bool(self.entry_when(row).ne(0).any()):
                        result.add('situation')
                    if self.when_start is not None and bool(self.slots[row, self.when_start-4:self.when_start].ne(0).any()):
                        result.add('spatial')
            if kind == RELATION:
                result.add('situation')
                if bool(self.parts.percept[value, :4].ne(0).any()):
                    result.add('spatial')
            return result
        readings = self.parts.readings(symbol, types)
        relation = ('implies' if 'conditional' in readings else
                    'equal' if 'equality' in readings else 'part' if readings else None)
        if reading is not None:
            if reading == 'implies' and 'conditional' not in readings:
                raise ValueError('a conditional requires two situation arguments')
            if reading == 'equal' and 'equality' not in readings:
                raise ValueError('equality requires singleton membership')
            relation = reading
        left, right = self.parts.references(symbol)
        roles = self.slots.new_zeros(3, self.nDim)
        mask = torch.zeros(3, dtype=torch.bool, device=roles.device)
        refs = [None, None, None]
        def place(slot, reference):
            value = point(self, reference)
            if value is not None:
                roles[slot] = self._fit(value)
            refs[slot] = envelope(self, reference)
            mask[slot] = True
        place(0, left)
        if relation is not None:
            place(2, right)
            roles[1] = predicate_point(relation, roles[1]); mask[1] = True
            refs[1] = ('sym', predicate_identity(relation))
        elif right[0] == RELATION:
            predicate, obj = self.parts.references(right[1])
            place(1, predicate); place(2, obj)
        else:
            place(1, right)
        marks = {'_equality': True} if relation == 'equal' else {}
        opened, forwards, sites, unresolved = [], [], [], []
        for role, reference in enumerate(refs):
            if reference is None:
                continue
            if reference[0] == 'sym':
                if owner.interpret.no_object(reference[1]):
                    opened.append(('referent', role))
                    forwards.append((role, reference[1]))
                elif self.parts.contains(reference[1]):
                    child = self.part_row(reference[1])['meaning']
                    if any(k != 'evidence' for k, _ in dict(child.bindings).get('_open_references', ())):
                        unresolved.append((role, reference))
                continue
            index = self.location(reference)
            if not isinstance(index, int):
                continue
            child = self.raw_meaning_of(index)
            if child is None:
                continue
            data = dict(child.bindings)
            if any(tag != 'evidence' for tag, _ in data.get('_open_references', ())):
                opened.append(('referent', role))
            forwards.extend((role, name) for _, name in data.get('_forward_references', ()))
            sites.extend((role, dict(site, role=role)) for _, site in data.get('_reference_sites', ()))
        if forwards:
            marks['_forward_references'] = tuple(forwards)
        if sites:
            marks['_reference_sites'] = tuple(sites)
        if unresolved:
            marks['_unresolved_operands'] = tuple(unresolved)
        pair = tuple(float(part[name]) for name in ('truth_plus', 'truth_minus'))
        # A chain view exposes only its actual operands. An absent third
        # grammatical role is not a missing referent manufactured by storage.
        marks['_chain_view'] = True
        meaning = with_slots(ConceptualMeaning(roles, mask, role_refs=tuple(refs),
            sentence_kind='relation', bindings=marks), opened, pair=pair)
        return dict(part, meaning=meaning, head_point=head_idea(self, symbol), occurrence=('sym', symbol), row_id=symbol,
            refs=part['refs'].clone(), readings=readings, evidence=pair,
            trust=pair[0]-pair[1], source_trust=0., when=part['percept'][4:].clone(),
            kind='question' if opened else 'observation' if any(pair) else 'unverified', order=0, stream=(), pending=False,
            metadata_complete=True, expectation=self.expectation_of(('sym', symbol)), text=None, origin=0)

    def location(self, reference):
        """Find either an LTM0 occurrence or a native LTM1 symbol."""
        if isinstance(reference, tuple):
            if len(reference) == 2 and reference[0] == 'sym' and self.parts.contains(reference[1]):
                return reference
            return self._index_occurrences.get(reference)
        row = self.index_of_row(reference)
        return row if row is not None else (('sym', int(reference))
            if self.parts.contains(reference) else None)

    def clause_relation(self, field):
        """Infer the row kind from occupancy and its REL's native identity."""
        if field.slots.shape[0] == 1:
            return None
        if field.slots.shape[0] != 3:
            raise ValueError('a completed row occupies one or three slots')
        resolve = getattr(self, '_predicate_kind', None)
        reference = field.refs[1]
        if isinstance(reference, ClausePredicate):
            return predicate_relation(reference.identity)
        result = resolve(reference) if resolve is not None and type(reference) is int else 'operator'
        if result not in ('part', 'implies', 'operator'):
            raise ValueError('the REL identity has an unknown grammatical kind')
        return result

    @contextmanager
    def clause_assertions(self, *, trust, origin, text):
        """External TruthSet authority, scoped to completed outer clauses only."""
        previous = self.__dict__.get('_clause_assertion')
        context = dict(trust=float(trust), origin=origin, text=text, rows=[])
        object.__setattr__(self, '_clause_assertion', context)
        try:
            yield context['rows']
        finally:
            object.__setattr__(self, '_clause_assertion', previous)

    def _native_reference_indexes(self):
        """Derived identity-to-occurrence caches, invalidated by row edits."""
        count = len(self)
        token = (count, id(self.row_ids), self.row_ids._version)
        if self.__dict__.get('_native_reference_token') != token:
            rows = {}
            for row, identity in enumerate(self.row_ids[:count].tolist()):
                if identity not in (-1, 0):
                    rows.setdefault(identity, row)
            object.__setattr__(self, '_native_row_index', rows)
            object.__setattr__(self, '_native_reference_token', token)
        return self._native_row_index

    def index_of_row(self, reference):
        return self._native_reference_indexes().get(int(reference))

    def _index_slot_metadata(self, address):
        from Occurrence import slot_key
        data = dict(self._semantic_rows.get(address, {}).get('bindings', ()))
        for kind, role in (*data.get('_open_references', ()), *data.get('_bound_roles', ())):
            if kind != 'referent':
                continue
            identity = slot_key(address, role)
            previous = self._slot_index.get(identity)
            if previous is not None and previous != (address, role):
                raise RuntimeError('64-bit slot collision; refusing to merge different slots')
            self._slot_index[identity] = address, role

    def slot_of_reference(self, reference):
        """Derived shared posting: source address and role remain authoritative."""
        if self.__dict__.get('_slot_index_source') is not self._semantic_rows:
            self._slot_index_source, self._slot_index = self._semantic_rows, {}
            for address in self._semantic_rows:
                self._index_slot_metadata(address)
        value = self._slot_index.get(int(reference))
        if value is None:
            return None
        address, role = value
        row = self._index_occurrences.get(address)
        if row is None:
            return None
        if self.index_of_row(reference) is not None:
            raise RuntimeError('slot address collides with an occurrence')
        return row, role

    def semantic_reference(self, reference):
        slot = self.slot_of_reference(reference)
        if slot is not None:
            return ('slot', self.occurrence_of(slot[0]), slot[1])
        row = self.index_of_row(reference)
        return self.occurrence_of(row) if row is not None else ('sym', reference)

    def point_of_row(self, reference):
        """Read a concept/idea point; a relation explicitly has no point."""
        slot = self.slot_of_reference(reference)
        if slot is not None:
            return self.slots[slot[0]]
        index = self.index_of_row(reference)
        if index is not None:
            return self.slots[index]
        resolve = getattr(self, '_concept_point', None)
        if resolve is None:
            raise ValueError('shared concept index is not attached')
        point = resolve(int(reference))
        if point is None:
            raise ValueError(f'unavailable shared row {reference}')
        return self._fit(point)

    def _clause_reference(self, reference, children):
        if isinstance(reference, ClausePredicate):
            return reference.identity
        return int(self.row_ids[children[reference[1]]]) if isinstance(reference, tuple) else reference

    @torch.no_grad()
    def write_clause(self, clause, *, trust=None, origin=None, stream=-1, kind='observation',
                    evidence=None, text=None, expectation=None, document_key=None,
                    sentence_index=0, content_key=None, written_rows=None):
        """Compress resolved semantic journal compositions, retaining only open chains."""
        import math
        from PartClosing import PartClosing
        from LTM1 import OCCURRENCE, SYMBOL, RELATION
        from RowTime import stamp
        from Occurrence import native_content_key
        from ThoughtReferences import bindings
        if not isinstance(clause, Clause):
            raise TypeError('write_clause requires a completed field')
        if document_key is not None:
            with self.at_address(document_key, sentence_index, content_key or native_content_key(clause.meaning)):
                return self.write_clause(clause, trust=trust, origin=origin, stream=stream, kind=kind,
                    evidence=evidence, text=text, expectation=expectation, written_rows=written_rows)
        authority = self.__dict__.get('_clause_assertion')
        if authority is not None:
            if (bindings(clause.meaning).get('_open_references') or
                    (clause.relation is not None and any(ref in (-1, 0) for ref in clause.refs))):
                raise ValueError('a TruthSet cannot assert a clause with open references')
            trust, origin, text = (authority[name] for name in ('trust', 'origin', 'text'))
            kind = 'fact'
        if trust is not None and not math.isfinite(float(trust)):
            raise ValueError('clause provenance must be finite')
        if kind not in self.KINDS or type(stream) is not int or stream < -1:
            raise ValueError('invalid clause evidence kind or stream')
        if stream >= self.stream.shape[1]:
            raise OverflowError('stream exceeds LTM0 construction capacity')
        if text is not None and not isinstance(text, str):
            raise ValueError('truth source text must be a string')
        owner_ref = self.__dict__.get('_concept_owner')
        owner = None if owner_ref is None else owner_ref()
        if owner is None:
            raise ValueError('clause admission requires its native concept owner')
        closing = PartClosing(owner, self)
        nodes, seen, visiting = [], set(), set()
        def visit(value):
            if id(value) in visiting:
                raise ValueError('a temporary derivation cannot contain an object cycle')
            if id(value) in seen:
                return
            visiting.add(id(value))
            for child in (*value.children, *value.companions):
                visit(child)
            visiting.remove(id(value))
            seen.add(id(value))
            nodes.append(value)
            closing.prepare(value)
        visit(clause)
        closing.preflight()
        context = self.__dict__.get('_write_address')
        root_existing = None
        if context is not None:
            from Occurrence import address_key
            root_existing = self._index_occurrences.get(address_key(*context))
        needed, counted = 0, set()
        def reserve(record, sentence=False, force=False):
            nonlocal needed
            term = record['value']
            key = (id(term), sentence)
            if key in counted:
                return
            counted.add(key)
            ref, children = record['reference'], record['children']
            if (not sentence and not force and ref is not None
                    and (not children or term.operation not in (*closing._assertions, 'lift', 'verb'))):
                return
            if (record['resolved'] and term.operation not in closing._assertions) or not children or (force and (term.binding is not None or record.get('mention'))):
                from IdentityRows import carrier
                carrier(owner, term.point)
                needed += int(not sentence or root_existing is None)
            else:
                for child in children:
                    reserve(child)
        for value in nodes:
            sentence = value is clause
            if sentence or value in clause.companions or value.reference_request is not None or value.identity is not None:
                record = closing._planned[id(closing._terms[id(value)])]
                force = not sentence
                reserve(record, sentence, force)
        needed += int(expectation is not None and expectation.estimate is not None and bool(expectation.source_occurrences))
        if len(self) + needed > self.capacity:
            raise OverflowError('LTM0 construction capacity exhausted before close')
        closing.when = stamp(self)
        rows, occurrences = {}, {}
        context = self.__dict__.get('_write_address')
        root_key = native_content_key(clause.meaning) if context is None else context[2]
        closing.content_key = (context[0], context[1], root_key) if context is not None else ('direct', root_key)
        def location(entry):
            tag, value = entry
            return self.index_of_row(value) if tag == OCCURRENCE else ('sym', value)
        # A mention needing a fresh working identity keeps its occurrence.
        # Existing concepts need no second LTM0 copy solely for their role.
        for position, value in enumerate(nodes):
            sentence = value is clause
            if not sentence and value not in clause.companions and value.reference_request is None and value.identity is None:
                continue
            pair = tuple(value.evidence if evidence is None or not sentence else evidence)
            from_trust = sentence and trust is not None and (authority is not None or evidence is None)
            if from_trust:
                pair = max(float(trust), 0.), max(-float(trust), 0.)
            address = {} if value is clause else dict(document_key=('embedded', closing.content_key[0], closing.content_key[1]) if context is not None else ('mention', root_key),
                sentence_index=position, content_key=native_content_key(value.meaning))
            record = closing._planned[id(closing._terms[id(value)])]
            force = not sentence
            entry = closing.close(value, pair=pair, sentence=sentence, kind=kind if sentence else 'unverified',
                address=address, stream=stream, force=force, from_trust=from_trust,
                surprise=-1. if expectation is None or not sentence else expectation.surprise_for(clause.meaning))
            row = location(entry)
            rows[id(value)] = row
            occurrences[id(value)] = self.occurrence_of(row)
            if isinstance(row, int) and sentence:
                if trust is not None:
                    self.source_trust[row] = float(trust)
                if origin is not None or text is not None:
                    self.set_origin(row, self.ORIGIN_CONVERSATION if origin is None else origin, text=text)
        # Entries consumed inside a chain also remain addressable by the live
        # journal. This map ends with the close; it is not durable derivation.
        for value in nodes:
            entry = closing._entries.get(id(closing._terms[id(value)]))
            if entry is not None and id(value) not in rows:
                rows[id(value)] = location(entry)
                occurrences[id(value)] = self.occurrence_of(rows[id(value)])
        result = rows[id(clause)]
        if written_rows is not None:
            written_rows.update(rows)
            written_rows.update({('occurrence', key): value for key, value in occurrences.items()})
            written_rows['root_occurrence'] = occurrences[id(clause)]
            # The live closing keeps its working roles, without saving them
            # beside either stratum. This is also the binder's immediate read.
            for value in nodes:
                refs = []
                for role, ref in enumerate(value.refs):
                    if isinstance(ref, ClausePredicate):
                        ref = ('sym', ref.identity)
                    elif isinstance(ref, tuple) and ref[:1] == ('clause',):
                        ref = occurrences.get(id(value.children[ref[1]]))
                    elif type(ref) is int and ref not in (-1, 0):
                        ref = self.semantic_reference(ref)
                    else:
                        ref = value.meaning.role_refs[role]
                    refs.append(ref if bool(value.meaning.role_mask[role]) else None)
                written_rows['meaning', id(value)] = replace(value.meaning, role_refs=tuple(refs))

            for value in nodes:
                term = closing._terms[id(value)]
                ref = closing._planned[id(term)]['reference']
                written_rows['symbol', id(value)] = ref[1] if ref and ref[0] == SYMBOL and ref[1] in closing.alloc.individuals else None
            from ReferenceParticipants import clause_participants
            _, _, participants = clause_participants(clause, self._part_read_rules)
            written_rows['symbols', id(clause)] = tuple(dict.fromkeys(
                ref[1] for record in closing._planned.values()
                if id(record['value']) in participants and (ref := record['reference'])
                and ref[0] == SYMBOL and ref[1] in closing.alloc.individuals))
            written_rows['activations', id(clause)] = tuple(
                (occurrences[id(value)], written_rows.get(('symbol', id(value))),
                 (value.reference_request or {}).get('role', 0))
                for value in nodes if value.order == 1 and id(value) in occurrences)
        if expectation is not None and expectation.estimate is not None and expectation.source_occurrences:
            prediction = expectation.estimate
            retained = prediction.retained_bindings(observed=expectation.observed,
                occupied=expectation.occupied, activation=expectation.activation, identities=expectation.identities)
            # This is the audited target of the frozen prior, not a second
            # description or a recoverable sentence derivation.
            from Expectation import symbol_activations
            activation = expectation.activation
            if activation is None:
                activation = symbol_activations(prediction.symbol_keys[None],
                    expectation.observed[None], expectation.occupied[None],
                    identities=None if expectation.identities is None else expectation.identities[None],
                    bank_ids=prediction.symbol_ids)[0]
            retained['_expectation_observation'] = (('_observed_symbol_activations', tuple(
                (int(identity), tuple(pair)) for identity, pair in zip(
                    prediction.symbol_ids.tolist(), activation.detach().cpu().tolist()))),)
            estimate = replace(ConceptualMeaning.from_description(prediction.roles[0]),
                bindings=retained, scope=prediction.scope)
            index = self.append_estimate(estimate, presence_logits=prediction.presence_logits,
                source_occurrences=tuple(expectation.source_occurrences), stream=expectation.stream,
                document=expectation.document)
            self.link_estimate_observation(index, result)
        if authority is not None:
            authority['rows'].append(result)
        return result

    def _update_forward_index(self, address):
        """Maintain the shared, derived posting list for unresolved identities."""
        if self.__dict__.get('_slot_index_source') is self._semantic_rows:
            self._index_slot_metadata(address)
        if self.__dict__.get('_forward_index_source') is not self._semantic_rows:
            return  # A restored/replaced store is rebuilt on its next lookup.
        postings = self._forward_postings
        by_address = self._forward_by_address
        for identity in by_address.pop(address, ()):
            postings[identity].discard(address)
            if not postings[identity]:
                del postings[identity]
        metadata = self._semantic_rows.get(address, {})
        identities = {identity for _, identity in
                      dict(metadata.get('bindings', ())).get('_forward_references', ())}
        if identities:
            by_address[address] = identities
            for identity in identities:
                postings.setdefault(identity, set()).add(address)

    def _forward_addresses(self, identities):
        if self.__dict__.get('_forward_index_source') is not self._semantic_rows:
            self._forward_index_source = self._semantic_rows
            self._forward_postings, self._forward_by_address = {}, {}
            for address in self._semantic_rows:
                self._update_forward_index(address)
        return set().union(*(self._forward_postings.get(identity, ()) for identity in identities))

    def read_reference_bindings(self, index, meaning, *, exclude=()):
        """Read an already bound token; equality rows never select a binding.

        The journal's pending names and structural constituent references stay
        immutable. Only ReferenceClose.commit can refine their MTM token.
        """
        from ThoughtReferences import bindings, fill, open_slots, with_slots
        owner_ref = self.__dict__.get('_concept_owner')
        owner = None if owner_ref is None else owner_ref()
        if owner is None:
            return meaning
        resolver, mtm = owner.components.reference_close, owner.mtm
        occurrence = self.occurrence_of(index)
        raw_refs = meaning.role_refs
        if isinstance(index, tuple):
            references = list(meaning.role_refs)
            for role, reference in enumerate(references):
                child = self.location(reference) if reference is not None else None
                if not isinstance(child, int):
                    continue
                if reference not in mtm.names:
                    continue  # an idea's identity binding is not its value's address
                identity = dict(self.raw_meaning_of(child).bindings).get('_identity')
                if identity is None:
                    continue
                token = mtm.token_of(identity)
                if token is not None and not mtm.entries[token].pending:
                    entry = mtm.entries[token]
                    references[role] = ('sym', entry.kind) if entry.kind is not None else token
                elif identity[0] == 'sym':
                    references[role] = identity
            meaning = replace(meaning, role_refs=tuple(references))
        for role, _name in bindings(meaning).get('_forward_references', ()):
            reference = raw_refs[role]
            if reference is not None and reference[0] == 'sym' and owner.interpret.no_object(reference[1]):
                continue
            site = reference if reference is not None and reference[0] == 'ltm' else occurrence
            token = mtm.token_of(site)
            if token is None:
                kind = resolver._derivation_symbol(reference)
                if kind is None:
                    continue
                point = self._concept_point(kind)
                target, uses = reference, (site,)
            else:
                entry = mtm.entries[token]
                kind = entry.kind
                point = None if entry.pending else resolver.token_value(token)
                # A token's own one-slot read exposes its derivation,
                # terminating the address walk. A parent keeps its token.
                target = ('sym', kind) if site == occurrence and kind is not None else token
                uses = mtm.entries[token].occurrences
            # The retained parent already points at its anonymous child. That
            # structural pointer is not the pending role's bound content.
            references = list(meaning.role_refs)
            references[role] = None
            meaning = with_slots(replace(meaning, role_refs=tuple(references)),
                                 (*open_slots(meaning), ('referent', role)))
            if point is None:
                continue
            witnesses = [self.location(address) for address in uses]
            witnesses = [row for row in witnesses if isinstance(row, int)]
            meaning = fill(meaning, dict(reference=target, binding_value=point,
                support_true=max((float(self.c_plus[row]) for row in witnesses), default=0.),
                support_false=max((float(self.c_minus[row]) for row in witnesses), default=0.)), slots=(('referent', role),),
                witnesses=uses, operation='bind')
        return meaning

    def relation_operands(self, symbol):
        """Read the actual operands in their tagged address domains."""
        from LTM1 import RELATION
        from ChainRead import point
        symbol = symbol[1] if isinstance(symbol, tuple) else int(symbol)
        return tuple(self.part_row(ref[1]) if ref[0] == RELATION else point(self, ref)
                     for ref in self.parts.references(symbol))

    def consequents_by_row(self, reference, *, reading=None):
        from LTM1 import SYMBOL
        return [(symbol, right[1] if right[0] == SYMBOL else right,
                 (float(self.parts.truth_plus[symbol]), float(self.parts.truth_minus[symbol])))
                for symbol in self.relations(reading).tolist()
                for left, right in (self.parts.references(symbol),)
                if left == self.parts.reference(reference)]

    def evaluate_rows(self, left, right, *, reading=None):
        result = self.slots.new_zeros(2)
        for symbol in self.relations(reading).tolist():
            if self.parts.references(symbol) == (self.parts.reference(left), self.parts.reference(right)):
                result = torch.maximum(result, result.new_tensor((
                    self.parts.truth_plus[symbol], self.parts.truth_minus[symbol])))
        return result

    def _vector_relations(self, reading=None):
        from ChainRead import point
        result = []
        for symbol in self.relations(reading).tolist():
            a, b = (point(self, ref) for ref in self.parts.references(symbol))
            if a is not None and b is not None:
                result.append((symbol, a, b))
        return result

    @torch.no_grad()
    def consequents(self, state, threshold=.7, *, reading=None):
        from IdentityRows import meet_ratios
        result = []
        for symbol, left, right in self._vector_relations(reading):
            score = float(meet_ratios(state.to(left), left)[1])
            if score > threshold:
                result.append((symbol, score*float(self.parts.truth_plus[symbol]), right.clone()))
        return sorted(result, key=lambda record: -record[1])

    @torch.no_grad()
    def evaluate(self, left, right, *, reading=None):
        from IdentityRows import meet_ratios
        result = self.slots.new_zeros(2)
        for symbol, a, b in self._vector_relations(reading):
            match = meet_ratios(left.to(a), a)[1] * meet_ratios(right.to(b), b)[1]
            result = torch.maximum(result, match * result.new_tensor((
                self.parts.truth_plus[symbol], self.parts.truth_minus[symbol])))
        return result
