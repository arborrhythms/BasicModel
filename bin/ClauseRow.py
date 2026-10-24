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
    object.__setattr__(store, '_occupation_owner',
                       None if occupations is None else weakref.ref(occupations))
    object.__setattr__(store, '_part_read_rules', tuple(
        getattr(language, '_compose_binary_rules', ())))

    store_ref = weakref.ref(store)
    object.__setattr__(space, '_clause_store_ref', store_ref)
    present = getattr(model, 'present', None)
    if callable(present) and hasattr(model, 'when_time'):
        object.__setattr__(store, '_model_present', present)

    def concept_point(cid):
        owner = space_ref()
        try:
            return _basis(owner)[_existing_row(owner, ('sym', cid))]
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
        """Read an LTM1 edge into a temporary grammatical view.

        Only the two arguments and the six Part columns are durable. The
        grammatical operator in this view is chosen from the argument types.
        """
        symbol = int(symbol)
        part = self.parts.row(symbol)
        owner = self._concept_owner()
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(owner)
        def types(identity):
            result = set(alloc.symbol_types.get(identity, ()))
            if identity in alloc.singletons:
                result.add('singleton')
            if alloc.order_of(identity) > 1:
                result.add('kind')
            return result
        readings = self.parts.readings(symbol, types)
        if reading is None:
            reading = ('implies' if 'conditional' in readings else
                       'equal' if 'equality' in readings else 'part')
        if reading == 'implies' and 'conditional' not in readings:
            raise ValueError('a conditional requires two situation arguments')
        if reading == 'equal' and 'equality' not in readings:
            raise ValueError('equality requires singleton membership')
        left, right = part['refs'].tolist()
        roles = self.slots.new_zeros(3, self.nDim)
        for role, reference in ((0, left), (2, right)):
            value = self._concept_point(reference)
            if value is not None:
                roles[role] = self._fit(value)
        # This is an operation request, not an identity coordinate or a
        # third LTM1 column. It disappears with the read.
        roles[1] = predicate_point(reading, roles[1])
        pair = tuple(float(part[name]) for name in ('truth_plus', 'truth_minus'))
        from ThoughtReferences import with_slots
        meaning = with_slots(ConceptualMeaning(roles,
            torch.ones(3, dtype=torch.bool, device=roles.device),
            role_refs=(('sym', left), ('sym', predicate_identity(reading)), ('sym', right)),
            sentence_kind='relation',
            bindings={'_equality': True} if reading == 'equal' else {}),
            (), pair=pair)
        return dict(part, meaning=meaning, occurrence=('sym', symbol),
            row_id=symbol, refs=part['refs'].clone(), readings=readings,
            evidence=pair, trust=pair[0]-pair[1], source_trust=0.,
            kind='observation' if any(pair) else 'unverified', order=alloc.order_of(symbol),
            stream=(), pending=False,
            metadata_complete=True, expectation=self.expectation_of(('sym', symbol)), text=None)

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
            return self.slots[slot]
        index = self.index_of_row(reference)
        if index is not None:
            return self.slots[index, 0] if int(self.role_mask[index].sum()) == 1 else None
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
        """Close concepts into LTM0 and the derivation's argument pairs into LTM1.

        Wide working identities are never stored on occurrence rows. A completed
        relative field keeps its occurrence beside its native Part symbol and
        returns the symbol; an absolute or still-open field returns its occurrence.
        Both have the common row/read interface.
        """
        import math
        from PartClosing import PartClosing
        from ThoughtReferences import bindings, open_slots, with_slots
        if not isinstance(clause, Clause):
            raise TypeError('write_clause requires a completed field')
        if document_key is not None:
            from Occurrence import native_content_key
            with self.at_address(document_key, sentence_index,
                                 content_key or native_content_key(clause.meaning)):
                return self.write_clause(clause, trust=trust, origin=origin, stream=stream,
                    kind=kind, evidence=evidence, text=text, expectation=expectation,
                    written_rows=written_rows)
        authority = self.__dict__.get('_clause_assertion')
        if authority is not None:
            if clause.meaning.mode == 'interrogative':
                raise ValueError('a TruthSet cannot assert an interrogative clause')
            trust, origin, text = authority['trust'], authority['origin'], authority['text']
            kind = 'fact'
        if trust is not None and not math.isfinite(float(trust)):
            raise ValueError('clause provenance must be finite')
        if kind not in self.KINDS or type(stream) is not int or stream < -1:
            raise ValueError('invalid clause evidence kind or stream')
        if text is not None and not isinstance(text, str):
            raise ValueError('truth source text must be a string')
        owner = self.__dict__.get('_concept_owner')
        if owner is None or owner() is None:
            raise ValueError('clause admission requires its native concept owner')
        closing = PartClosing(owner(), self)
        context = self.__dict__.get('_write_address')
        nodes, visiting, visited = [], set(), set()
        def visit(value):
            if id(value) in visiting:
                raise ValueError('a temporary derivation cannot contain an object cycle')
            if id(value) in visited:
                return
            visiting.add(id(value))
            for child in value.children:
                visit(child)
            visiting.remove(id(value)); visited.add(id(value)); nodes.append(value)
            for companion in value.companions:
                visit(companion)
        visit(clause)
        pairs, edge_pairs, addresses, prepared, required = {}, {}, {}, set(), 0
        prediction = None if expectation is None else expectation.estimate
        surprise = -1. if prediction is None else expectation.surprise_for(clause.meaning)
        sources = () if expectation is None else tuple(expectation.source_occurrences)
        estimate, observation_marks = None, {}
        if prediction is not None:
            if expectation.identities is not None:
                observation_marks['_symbol_handles'] = tuple(expectation.identities.cpu().tolist())
            activation = expectation.activation
            if activation is None:
                activation = expectation.residual + prediction.expected_activation
            observation_marks['_observed_symbol_activations'] = tuple(
                (int(identity), tuple(pair)) for identity, pair in zip(
                    prediction.symbol_ids.cpu().tolist(), activation.detach().cpu().tolist()))
            observation_marks.update({key: value for key, value in bindings(clause.meaning).items()
                                      if key == '_withheld_symbols'})
            if sources:
                if any(self.location(source) is None for source in sources):
                    raise ValueError('estimate source occurrence is unavailable')
                retained = prediction.retained_bindings(observed=expectation.observed,
                    occupied=expectation.occupied, activation=activation,
                    identities=expectation.identities)
                # The predictor's frozen activation target is an audit of
                # the arrival, not another stored working identity or an
                # extra column on a Part edge.
                retained['_expectation_observation'] = observation_marks
                estimate = ConceptualMeaning(prediction.roles,
                    torch.ones(3, dtype=torch.bool, device=prediction.roles.device),
                    mode='unspecified', bindings=retained, scope=prediction.scope)
                self._validate_local_references(estimate)
                self._normalise_expectation_provenance(dict(kind='estimate',
                    source_occurrences=sources, intended_occurrence=None,
                    presence_logits=tuple(prediction.presence_logits.detach().cpu().tolist()),
                    stream=expectation.stream, document=expectation.document), self.address_domain)
                previous = self._address_for_write(estimate, kind='estimate',
                    **self._estimate_write_address(estimate, document=expectation.document,
                                                  stream=expectation.stream))[-1]
                required += previous is None
        def pending(value):
            return any(tag != 'evidence' for tag, _ in
                bindings(value.meaning).get('_open_references', ())) or (value.relation is not None
                and any(value.refs[role] in (-1, 0) for role in (0, 2)))
        def incomplete(value):
            return pending(value) or any(incomplete(child) for child in value.children)
        def row_kind(value):
            if value is clause or value in clause.companions:
                return kind
            declared = bindings(value.meaning).get('_open_references', ())
            return 'question' if set(declared).intersection(open_slots(value.meaning)) else 'unverified'
        def retained_pair(value):
            term = closing._terms.get(id(value))
            return (value.relation is not None and term is not None
                and closing._planned[id(term)]['edge'] is not None)
        for position, value in enumerate(nodes):
            asserted = value is clause or value in clause.companions
            pair = tuple(map(float, value.evidence if evidence is None or not asserted else evidence))
            from_trust = asserted and (authority is not None or (trust is not None and evidence is None))
            if from_trust:
                signed = max(-1., min(1., float(trust)))
                pair = (max(signed, 0.), max(-signed, 0.))
            if len(pair) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in pair):
                raise ValueError('clause evidence requires two finite poles in [0, 1]')
            pairs[id(value)] = pair
            # Occurrences retain polarity as a field; the Part edge has
            # no polarity column and needs canonical evidence poles.
            edge_pairs[id(value)] = pair[::-1] if from_trust and not value.meaning.polarity else pair
            if not incomplete(value):
                closing.prepare(value)
                prepared.add(id(value))
            from Occurrence import native_content_key
            kwargs = ({} if value is clause and context is not None else
                dict(document_key=('direct', kind) if context is None else
                     ('embedded', context[0], context[1]), sentence_index=position,
                     content_key=native_content_key(value.meaning)))
            previous = self._address_for_write(value.meaning,
                kind=row_kind(value), **kwargs)[-1]
            required += previous is None
            addresses[id(value)] = kwargs
        if len(self) + required > self.capacity:
            raise OverflowError('LTM1 capacity exhausted before clause admission')
        closing.preflight()
        from RowTime import stamp
        closing.when = stamp(self)
        rows, occurrence_rows = {}, {}
        estimate_index = None
        if estimate is not None:
            estimate_index = self.append_estimate(estimate,
                presence_logits=prediction.presence_logits, source_occurrences=sources,
                stream=expectation.stream, document=expectation.document)
        for value in nodes:
            asserted = value is clause or value in clause.companions
            pair = pairs[id(value)]
            if id(value) in prepared:
                symbol = closing.close(value, pair=edge_pairs[id(value)], when=closing.when,
                                       surprise=surprise if asserted else -1.)
            else:
                symbol = None
            if retained_pair(value):
                if symbol is None or not self.parts.contains(symbol):
                    raise RuntimeError('a completed relation did not produce a Part edge')
            retain_open = pending(value) or (value.relation is not None and incomplete(value))
            unresolved_operands = []
            if retain_open:
                # An open question is retained at the existing occurrence
                # boundary. Filling it later changes only its working read.
                named = []
                for role, reference in enumerate(value.refs):
                    if isinstance(reference, ClausePredicate):
                        reference = ('sym', reference.identity)
                    elif isinstance(reference, tuple) and reference[:1] == ('clause',):
                        child = value.children[reference[1]]
                        term = closing._terms.get(id(child))
                        symbol = None if term is None else closing.symbols.get(id(term))
                        reference = (('sym', symbol) if symbol is not None
                            and closing._point(symbol) is not None else
                            self.occurrence_of(rows[id(child)]))
                        if incomplete(child) and not pending(child):
                            # This operand contains an unresolved dependency;
                            # its fused placeholder is not a completed answer.
                            unresolved_operands.append((role, reference))
                    elif type(reference) is int and reference not in (-1, 0):
                        reference = self.semantic_reference(reference)
                    else:
                        reference = value.meaning.role_refs[role]
                    named.append(reference)
                stored = replace(value.meaning, role_refs=tuple(named))
            else:
                point = value.point
                if point is None:
                    point = closing._terms[id(value)].point
                stored = ConceptualMeaning.from_description(self._fit(point))
                if symbol is not None:
                    # This is the address of the surviving derivation, not
                    # the dissolved working identity. Membership reads the
                    # Part arguments; it never inverts this narrowed point.
                    stored = replace(stored, role_refs=(('sym', symbol), None, None))
            marks = bindings(value.meaning)
            retained = {key: marks[key] for key in ('_withheld_symbols',) if key in marks}
            if retain_open:
                retained.update(marks)
            if unresolved_operands:
                retained['_unresolved_operands'] = tuple(unresolved_operands)
            if value is clause:
                retained.update(observation_marks)
            stored = with_slots(replace(stored, mode=value.meaning.mode,
                polarity=value.meaning.polarity,
                sentence_kind=stored.sentence_kind if retain_open else 'idea',
                bindings=retained), open_slots(stored), pair=pair)
            before_count = len(self)
            row = self.append_meaning(stored, kind=row_kind(value),
                trust=0. if trust is None or not asserted else float(trust), stream=stream,
                surprise=surprise if asserted else -1., evidence=pair, order=value.order,
                **addresses[id(value)])
            if row >= before_count:
                self.when[row].copy_(closing.when)
            if origin is not None:
                self.set_origin(row, origin, text=text if asserted else None)
            occurrence_rows[id(value)] = row
            rows[id(value)] = ('sym', symbol) if retained_pair(value) else row
            if written_rows is not None:
                # The returned handle can be a Part. Identity histories name
                # the completed occurrence beside it, without changing layout.
                written_rows[('occurrence', id(value))] = self.occurrence_of(row)
        result = rows[id(clause)]
        if estimate_index is not None:
            self.link_estimate_observation(estimate_index, result)
        if written_rows is not None:
            written_rows['root_occurrence'] = written_rows[('occurrence', id(clause))]
            written_rows.update(rows)
            written_rows.update({('symbol', id(value)):
                closing.symbols.get(id(closing._terms.get(id(value))))
                for value in nodes})
            written_rows[('symbols', id(clause))] = tuple(dict.fromkeys(
                symbol for symbol in closing.symbols.values()
                if symbol in closing.alloc.individuals))
            written_rows[('activations', id(clause))] = tuple(
                (self.occurrence_of(rows[id(value)]),
                 symbol if symbol in closing.alloc.individuals else None,
                 (value.reference_request or {}).get('role', 0))
                for value in nodes if value.order == 1
                for symbol in (written_rows.get(('symbol', id(value))),))
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
        if owner is None or not isinstance(index, int):
            return meaning
        resolver, mtm = owner.components.reference_close, owner.mtm
        occurrence = self.occurrence_of(index)
        for role, _name in bindings(meaning).get('_forward_references', ()):
            reference = meaning.role_refs[role]
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
                support_true=max(float(self.c_plus[row]) for row in witnesses),
                support_false=max(float(self.c_minus[row]) for row in witnesses)), slots=(('referent', role),),
                witnesses=uses, operation='bind')
        return meaning

    def relation_operands(self, symbol):
        """Read the two symbol arguments without reconstructing an LTM0 row."""
        symbol = symbol[1] if isinstance(symbol, tuple) else int(symbol)
        return tuple(self.part_row(ref) if self.parts.contains(ref) else self.point_of_row(ref).clone()
                     for ref in self.parts.row(symbol)['refs'].tolist())

    def consequents_by_row(self, reference, *, reading=None):
        return [(symbol, int(self.parts.refs[symbol, 1]),
                 (float(self.parts.truth_plus[symbol]), float(self.parts.truth_minus[symbol])))
                for symbol in self.relations(reading).tolist()
                if int(self.parts.refs[symbol, 0]) == int(reference)]

    def evaluate_rows(self, left, right, *, reading=None):
        result = self.slots.new_zeros(2)
        for symbol in self.relations(reading).tolist():
            if tuple(self.parts.refs[symbol].tolist()) == (int(left), int(right)):
                result = torch.maximum(result, result.new_tensor((
                    self.parts.truth_plus[symbol], self.parts.truth_minus[symbol])))
        return result

    def _vector_relations(self, reading=None):
        result = []
        for symbol in self.relations(reading).tolist():
            a, b = (self._concept_point(ref) for ref in self.parts.refs[symbol].tolist())
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
