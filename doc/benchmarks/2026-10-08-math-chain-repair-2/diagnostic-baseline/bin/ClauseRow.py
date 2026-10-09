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
    eternal: bool = False
    factored_refs: tuple | None = None
    order: int = 0
    evidence: tuple = (0., 0.)

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
        if self.eternal and (self.relation is not None or type(self.refs[0]) is not int
                             or self.refs[0] <= 0 or self.when is not None):
            raise ValueError('an eternal NP reuses an unlocated concept identity')
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

    store_ref = weakref.ref(store)
    object.__setattr__(space, '_clause_store_ref', store_ref)

    def concept_point(cid):
        owner = space_ref()
        try:
            return _basis(owner)[_existing_row(owner, ('sym', cid))]
        except ValueError:
            return None

    store.configure_clause_index(concept_point=concept_point,
        predicate_kind=predicate_relation, preflight=lambda count: space_ref()._preflight_concept_allocation(count, context='clause taxonomy'))
    from ConceptIndex import ClauseTaxonomyPlan
    object.__setattr__(store, '_clause_index_plan',
        lambda clause: ClauseTaxonomyPlan(space_ref(), store, clause))


class ClauseRows:
    """One writer for completed clauses; vector and row readers share its rows."""

    def configure_clause_index(self, *, allocate=None, concept_point, predicate_kind=None, on_part=None, preflight=None):
        object.__setattr__(self, '_allocate_clause_row', allocate)
        object.__setattr__(self, '_concept_point', concept_point)
        object.__setattr__(self, '_predicate_kind', predicate_kind)
        object.__setattr__(self, '_index_part_row', on_part)
        object.__setattr__(self, '_preflight_clause_rows', preflight)

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
        token = (count, id(self.row_ids), self.row_ids._version,
                 id(self.refs), self.refs._version, self.rel_type._version)
        if self.__dict__.get('_native_reference_token') != token:
            rows, predicates = {}, {}
            for row, identity in enumerate(self.row_ids[:count].tolist()):
                if identity not in (-1, 0):
                    rows.setdefault(identity, row)
            for row, references in enumerate(self.refs[:count].tolist()):
                for slot, identity in enumerate(references):
                    if (1 << 61) <= identity < (1 << 62):
                        predicates.setdefault(identity, (row, slot))
            object.__setattr__(self, '_native_row_index', rows)
            object.__setattr__(self, '_native_predicate_index', predicates)
            object.__setattr__(self, '_native_reference_token', token)
        return self._native_row_index, self._native_predicate_index

    def index_of_row(self, reference):
        return self._native_reference_indexes()[0].get(int(reference))

    def point_of_row(self, reference):
        """Read a concept/idea point; a relation explicitly has no point."""
        index = self.index_of_row(reference)
        if index is not None:
            return self.slots[index, 0] if int(self.rel_type[index]) == self.REL_NONE else None
        occurrence = self._native_reference_indexes()[1].get(int(reference))
        if occurrence is not None:
            return self.slots[occurrence]
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
                    sentence_index=0, content_key=None):
        """Write embedded S rows first, without granting them assertion authority.

        The caller supplies provenance. Clause operators supply only structure
        and polarity. Repeated relations join independent evidence poles.
        """
        if not isinstance(clause, Clause):
            raise TypeError('write_clause requires a completed field')
        if document_key is not None:
            from Occurrence import native_content_key
            with self.at_address(document_key, sentence_index,
                                 content_key or native_content_key(clause.meaning)):
                return self.write_clause(clause, trust=trust, origin=origin, stream=stream,
                    kind=kind, evidence=evidence, text=text, expectation=expectation)
        assertion = self.__dict__.get('_clause_assertion')
        if assertion is not None:
            if clause.meaning.mode == 'interrogative':
                raise ValueError('a TruthSet cannot assert an interrogative clause')
            trust, origin, text = assertion['trust'], assertion['origin'], assertion['text']
            kind = 'fact'
            clause = replace(clause, meaning=replace(clause.meaning, mode='assertive'))
        import math
        if trust is not None and not math.isfinite(float(trust)):
            raise ValueError('clause provenance must be finite')
        if evidence is not None and (len(evidence) != 2 or any(
                not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in evidence)):
            raise ValueError('clause evidence requires two finite poles in [0, 1]')
        if getattr(self, '_concept_point', None) is None:
            raise ValueError('clause admission requires its reference index')
        if kind not in self.KINDS or (kind == 'fact' and clause.meaning.mode != 'assertive'):
            raise ValueError('invalid clause evidence kind or grammatical mode')
        if type(stream) is not int or stream < -1:
            raise ValueError('index stream must be a row or shared (-1)')
        if text is not None and not isinstance(text, str):
            raise ValueError('truth source text must be a string')
        tags = {None: self.REL_NONE, 'part': self.REL_PARTOF,
                'implies': self.REL_IMPLIES, 'operator': self.REL_OPERATOR}
        make_index_plan = getattr(self, '_clause_index_plan', None)
        index_plan = None if make_index_plan is None else make_index_plan(clause)
        if index_plan is not None:
            clause = index_plan.clause
        keys = {(int(self.rel_type[i]), *self.refs[i].tolist()): int(self.row_ids[i])
                for i in self.relations().tolist() if int(self.row_ids[i]) not in (-1, 0)}
        planned, visiting, eternal_ids, required = {}, set(), set(), 0
        context = self.__dict__.get('_write_address')
        addresses = {}
        def content_of(value):
            # Direct structural callers have no lexical sentence. Include the
            # graph in their content fallback so a parent cannot alias a child
            # whose learned point happens to be identical.
            import hashlib, json
            from Occurrence import native_content_key
            refs = tuple(ref.identity if isinstance(ref, ClausePredicate) else ref
                         for ref in value.refs)
            payload = json.dumps((value.relation, refs), separators=(',', ':')).encode()
            return hashlib.sha256(native_content_key(value.meaning) + payload
                + b''.join(content_of(child) for child in value.children)).digest()
        def source(value):
            if id(value) not in addresses:
                if context is None:
                    from Occurrence import native_content_key
                    kwargs = dict(document_key=('direct-clause', value.eternal),
                                  content_key=content_of(value))
                elif value is clause:
                    kwargs = {}
                else:
                    kwargs = dict(document_key=('embedded', context[0], context[1]),
                                  sentence_index=len(addresses), content_key=context[2])
                identity, _, _, _, old = self._address_for_write(value.meaning,
                    kind=kind if value is clause else 'unverified', **kwargs)
                addresses[id(value)] = (identity, old, kwargs)
            return addresses[id(value)]

        def preflight(value):
            nonlocal required
            if id(value) in visiting:
                raise ValueError('completed clause references contain a cycle')
            if id(value) in planned:
                return planned[id(value)]
            visiting.add(id(value))
            relation = self.clause_relation(value)
            if value.meaning.roles.shape != (3, self.nDim):
                raise ValueError('clause meaning width differs from its truth store')
            self._validate_local_references(replace(value.meaning,
                role_refs=(None, None, None), constituents=()))
            child_ids = tuple(preflight(child) for child in value.children)
            refs = tuple(child_ids[r[1]] if isinstance(r, tuple) else
                         r.identity if isinstance(r, ClausePredicate) else r for r in value.refs)
            for slot, ref in enumerate(value.refs):
                if isinstance(ref, ClausePredicate):
                    if ref.point.shape != (self.nDim,):
                        raise ValueError('predicate point differs from its truth store width')
                elif isinstance(ref, tuple):
                    child = value.children[ref[1]]
                    if relation is None and child.slots.shape[0] == 3:
                        raise ValueError('a clause over a relation cannot fuse')
                elif ref not in (-1, 0):
                    point = (index_plan.points[ref] if index_plan is not None and ref in index_plan.points
                             else self.point_of_row(ref))
                    if relation is None and point is None:
                        raise ValueError('a clause over a relation cannot fuse')
            if relation == 'implies':
                for slot in (0, 2):
                    ref = value.refs[slot]
                    if ref not in (-1, 0) and not isinstance(ref, tuple) and self.index_of_row(ref) is None:
                        raise ValueError('implication operands must be truth rows')
            def reserve(addresses):
                nonlocal required
                key = tags[relation], *addresses
                identity, previous, _ = source(value)
                if previous is not None:
                    return int(self.row_ids[previous])
                if value.eternal and self.index_of_row(value.refs[0]) is not None:
                    return value.refs[0]
                if value.eternal and value.refs[0] in eternal_ids:
                    return value.refs[0]
                if (context is None and relation is not None and key in keys
                        and all(ref not in (-1, 0) for ref in addresses)):
                    return keys[key]
                required += 1
                result = value.refs[0] if value.eternal else -required - 1
                if value.eternal:
                    eternal_ids.add(result)
                if relation is not None:
                    keys[key] = result
                return result
            result = reserve(refs)
            planned[id(value)] = result
            for companion in value.companions:
                preflight(companion)
            visiting.remove(id(value))
            return result

        preflight(clause)
        if self.capacity - len(self) < required:
            raise OverflowError(f'LTM capacity {self.capacity} exhausted; forgetting is required')
        check = getattr(self, '_preflight_clause_rows', None)
        if check is not None and index_plan is not None and index_plan.count:
            check(index_plan.count)
        prediction = None if expectation is None else getattr(expectation, 'estimate', None)
        sources = () if expectation is None else tuple(getattr(expectation, 'source_occurrences', ()))
        surprise = -1.
        estimate = None
        if prediction is not None:
            from Meaning import expectation_surprise
            logits = prediction.presence_logits.detach().reshape(-1)
            namespace = self.address_domain
            self._normalise_expectation_provenance(dict(kind='estimate',
                source_occurrences=sources, intended_occurrence=None,
                presence_logits=tuple(logits.cpu().tolist()),
                kind_logit=None if prediction.kind_logit is None else float(prediction.kind_logit),
                stream=getattr(expectation, 'stream', None),
                document=getattr(expectation, 'document', None)), namespace)
            surprise = expectation_surprise(clause.meaning.roles, prediction.roles,
                clause.meaning.role_mask, prediction.presence_logits.sigmoid())
            if sources:
                if any(source not in self._index_occurrences for source in sources):
                    raise ValueError('estimate source occurrence is unavailable')
                estimate = ConceptualMeaning(prediction.roles,
                    torch.ones(3, dtype=torch.bool, device=prediction.roles.device),
                    mode='unspecified', bindings=prediction.bindings, scope=prediction.scope)
                self._validate_local_references(estimate)
                previous = self._address_for_write(estimate, kind='estimate',
                    **self._estimate_write_address(estimate,
                        document=getattr(expectation, 'document', None),
                        stream=getattr(expectation, 'stream', None)))[-1]
                if self.capacity - len(self) < required + int(previous is None):
                    raise OverflowError(f'LTM capacity {self.capacity} exhausted; forgetting is required')
        if index_plan is not None:
            index_plan.commit()
        rows = {}
        estimate_index = -1

        def write(value, asserted=False, *, primary=True):
            nonlocal estimate_index
            if id(value) in rows:
                return rows[id(value)]
            children = tuple(write(child) for child in value.children)
            if asserted and primary and estimate is not None:
                estimate_index = self.append_estimate(estimate,
                    presence_logits=prediction.presence_logits, kind_logit=prediction.kind_logit,
                    source_occurrences=sources, stream=getattr(expectation, 'stream', None),
                    document=getattr(expectation, 'document', None))
            references = tuple(self._clause_reference(ref, children) for ref in value.refs)
            relation = self.clause_relation(value)
            if relation is None and any(
                    self.point_of_row(ref) is None for ref in references if ref not in (-1, 0)):
                raise ValueError('a clause over a relation cannot fuse')
            if relation in ('part', 'implies', 'operator'):
                if relation == 'implies' and any(
                        ref not in (-1, 0) and self.index_of_row(ref) is None
                        for ref in (references[0], references[2])):
                    raise ValueError('implication operands must be truth rows')
            degree = max(-1., min(1., float(trust))) if asserted and trust is not None else 0.
            positive, negative = map(float, value.evidence if evidence is None or not asserted else evidence)
            if asserted and (assertion is not None or (trust is not None and evidence is None)):
                positive, negative = max(degree, 0.), max(-degree, 0.)
                if not value.meaning.polarity:
                    positive, negative = negative, positive

            def append(refs):
                tag = tags[relation]
                def update(row):
                    self.c_plus[row] = max(float(self.c_plus[row]), positive)
                    self.c_minus[row] = max(float(self.c_minus[row]), negative)
                    if asserted:
                        self.write_timestamp(row)
                        self.record_kind[row] = self.KINDS.index(kind)
                        self.surprise[row] = float(surprise)
                        if origin is not None:
                            self.set_origin(row, origin, text=text)
                    return row
                if relation is None:
                    stored = ConceptualMeaning.from_description(value.point)
                    refs = (refs[0], refs[1], -1)
                    if value.eternal:
                        previous = self.index_of_row(refs[0])
                        if previous is not None:
                            return update(previous)
                else:
                    stored = ConceptualMeaning(value.meaning.roles,
                        value.meaning.role_mask,
                        mode=value.meaning.mode, polarity=value.meaning.polarity)
                    match = ((self.rel_type[:len(self)] == tag)
                             & (self.refs[:len(self)] == self.refs.new_tensor(refs)).all(-1))
                    # The VP address is canonical; its learned vector is not a key.
                    found = match.nonzero().flatten()
                    if context is None and found.numel() and all(ref not in (-1, 0) for ref in refs):
                        return update(int(found[0]))
                role_refs = (None, None, None) if relation is None else tuple(
                    self.occurrence_of(self.index_of_row(ref))
                    if ref not in (-1, 0) and self.index_of_row(ref) is not None else
                    ('sym', ref) if ref not in (-1, 0) else None for ref in refs)
                stored = ConceptualMeaning(stored.roles, stored.role_mask,
                    mode=value.meaning.mode, polarity=True, role_refs=role_refs)
                _, _, address_kwargs = source(value)
                row = self.append_meaning(stored, rel_type=tag, kind=kind if asserted else 'unverified',
                    trust=degree, stream=stream, surprise=surprise if asserted else -1.,
                    evidence=(positive, negative), order=value.order, **address_kwargs)
                if row < 0:
                    raise RuntimeError('clause capacity changed after preflight')
                row_id = value.refs[0] if value.eternal else int(self.address_keys[row])
                self.row_ids[row] = row_id
                self.refs[row] = self.refs.new_tensor(refs)
                if index_plan is not None:
                    index_plan.bind_particular(value, row_id)
                for name in ('where', 'when'):
                    band = getattr(value, name)
                    if band is not None:
                        getattr(self, name)[row] = band.to(self.slots)
                if origin is not None:
                    self.set_origin(row, origin, text=text if asserted else None)
                # An idea's operand references are only the two cached
                # addresses in refs. They never supply hidden factored roles.
                role_refs = (None, None, None) if relation is None else tuple(
                    self.occurrence_of(self.index_of_row(ref))
                    if ref not in (-1, 0) and self.index_of_row(ref) is not None else
                    ('sym', ref) if ref not in (-1, 0) else None for ref in refs)
                canonical = ConceptualMeaning(stored.roles, stored.role_mask,
                    **dict(stored.metadata(), role_refs=role_refs,
                           bindings=value.meaning.bindings, scope=value.meaning.scope))
                from ThoughtReferences import bindings, open_slots, with_slots
                opened = list(open_slots(value.meaning))
                if relation is not None:
                    opened.extend(('relation' if i == 1 else 'referent', i)
                                  for i, ref in enumerate(refs) if ref in (-1, 0))
                if opened:
                    canonical = with_slots(canonical, opened, pair=(positive, negative))
                    data = bindings(canonical)
                    data['_pending'] = True
                    canonical = replace(canonical, bindings=data)
                self._semantic_rows[int(self.address_keys[row])] = {
                    key: canonical.metadata()[key] for key in ('role_refs', 'bindings', 'scope')}
                self._update_forward_index(int(self.address_keys[row]))
                self.metadata_required[row] = True
                self._update_semantic_fingerprint(row)
                if relation == 'part' and not opened:
                    update = getattr(self, '_index_part_row', None)
                    if update is not None:
                        update(refs[0], refs[2], int(self.row_ids[row]))
                return row

            row = append(references)
            rows[id(value)] = row
            for companion in value.companions:
                write(companion, asserted=asserted, primary=False)
            return row

        result = write(clause, asserted=True)
        # The two equality directions supply the same ordinary substitution.
        # Its operands and source references, never source words, determine
        # which pending name is filled by the newly arrived occurrence.
        from ThoughtReferences import bindings
        for value_id, row in tuple(rows.items()):
            meaning = self.meaning_of(row)
            data = bindings(meaning)
            if not data.get('_equality'):
                continue
            left, right = int(self.refs[row, 0]), int(self.refs[row, 2])
            point = None if right in (-1, 0) else self.point_of_row(right)
            if point is None:
                continue
            aliases = {identity for role, identity in data.get('_forward_references', ()) if role == 0}
            if left not in (-1, 0):
                aliases.add(left)
                target = self.index_of_row(left)
                if target is not None:
                    aliases.update(identity for _, identity in bindings(self.meaning_of(target))
                                   .get('_forward_references', ()))
                if index_plan is not None:
                    for identities in index_plan.alloc.word_forms.values():
                        if left in identities:
                            aliases.update(identities)
            self.fill_forward_references(aliases, right, point)
        if estimate_index >= 0:
            self.link_estimate_observation(estimate_index, result)
        if assertion is not None and result >= 0:
            assertion['rows'].append(result)
        return result

    def _update_forward_index(self, address):
        """Maintain the shared, derived posting list for unresolved identities."""
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

    def fill_forward_references(self, identities, reference, point):
        """Fill pending slots by identity at their original occurrence address.

        This is a derived index of the shared truth rows, not stream state.
        A supplied occurrence never matches a forward name by code similarity.
        """
        from ThoughtReferences import bindings, open_slots, fill
        identities = set(identities)
        if not identities:
            return ()
        completed = []
        for address in sorted(self._forward_addresses(identities)):
            metadata = self._semantic_rows[address]
            data = dict(metadata.get('bindings', ()))
            forwards = data.get('_forward_references', ())
            selected = [(role, identity) for role, identity in forwards if identity in identities]
            if not selected:
                continue
            row = self.index_of_row(address)
            if row is None or int(self.row_ids[row]) == reference:
                continue
            meaning = self.meaning_of(row)
            native = self.index_of_row(reference)
            witness = self.occurrence_of(native) if native is not None else ('sym', reference)
            resolved = meaning
            for role, _ in selected:
                resolved = fill(resolved, dict(reference=witness, value=point,
                    support_true=float(self.c_plus[row]), support_false=float(self.c_minus[row])),
                    slots=(('referent', role),), witnesses=(witness,), operation='bind')
            data = bindings(resolved)
            data['_forward_references'] = tuple(pair for pair in forwards if pair not in selected)
            data['_pending'] = bool(open_slots(resolved))
            resolved = replace(resolved, bindings=data)
            # append_meaning's address upsert preserves document, position and
            # occurrence identity; no inferred truth or new address is minted.
            old_refs = self.refs[row].clone()
            previous_context = self.__dict__.get('_write_address')
            object.__setattr__(self, '_write_address', (
                bytes(self.document_keys[row].tolist()), int(self.sentence_index[row]),
                self.content_key(row), float(self.timestamp[row])))
            try:
                self.append_meaning(resolved, kind=self.KINDS[int(self.record_kind[row])],
                    rel_type=int(self.rel_type[row]), trust=float(self.trust[row]),
                    evidence=(float(self.c_plus[row]), float(self.c_minus[row])),
                    order=int(self.order[row]), address=int(self.address_keys[row]))
            finally:
                object.__setattr__(self, '_write_address', previous_context)
            for role, _ in selected:
                old_refs[role] = reference
            self.refs[row].copy_(old_refs)
            self._update_semantic_fingerprint(row)
            completed.append(row)
        return tuple(completed)

    def relation_operands(self, idx):
        """Resolve a relation's shared addresses without inventing fused points."""
        if not 0 <= int(idx) < len(self) or int(self.rel_type[idx]) == self.REL_NONE:
            raise ValueError('row is not a relation')
        result = []
        for reference in (int(self.refs[idx, 0]), int(self.refs[idx, 2])):
            row = self.index_of_row(reference)
            result.append(self.row(row) if row is not None else self.point_of_row(reference).clone())
        return tuple(result)

    def consequents_by_row(self, reference, *, rel_type=None):
        """Native row-indexed relations, including implication over relations."""
        rows = self.relations(rel_type)
        return [(int(i), int(self.refs[i, 2]), (float(self.c_plus[i]), float(self.c_minus[i])))
                for i in rows.tolist() if int(self.refs[i, 0]) == int(reference)]

    def evaluate_rows(self, left, vp, right, *, rel_type=None):
        result = self.slots.new_zeros(2)
        for i in self.relations(rel_type).tolist():
            if tuple(self.refs[i].tolist()) == (int(left), int(vp), int(right)):
                result = torch.maximum(result, result.new_tensor((self.c_plus[i], self.c_minus[i])))
        return result

    def _vector_relations(self, rel_type):
        rows = self.relations(rel_type)
        # A relation reference has no vector. It can only be queried by id.
        return rows[(self.slots[rows, 0].norm(dim=-1) > 0)
                    & (self.slots[rows, 2].norm(dim=-1) > 0)]

    @torch.no_grad()
    def constraint_residuals(self):
        """Measure disagreement between vector-valued relation consequences."""
        from torch.nn import functional as F
        rows = self._vector_relations(None)
        if not len(rows):
            return self.slots.new_zeros(0)
        left, verb, right = F.normalize(self.slots[rows], dim=-1).unbind(1)
        agree = (left @ left.T).clamp_min(0) * (verb @ verb.T).clamp_min(0)
        agree.fill_diagonal_(0)
        return (agree * (1 - right @ right.T)).amax(-1).clamp_min(0)

    def _relation_similarity(self, query, rows, slot):
        from torch.nn import functional as F
        width = min(self.content_width, self.nDim)
        return (F.normalize(self.slots[rows, slot, :width], dim=-1)
                @ F.normalize(self._fit(query)[:width], dim=-1)).clamp_min(0.)

    @torch.no_grad()
    def consequents(self, state, vp=None, threshold=.7, *, rel_type=None):
        """Vector substitution, weighted by independent positive evidence."""
        rows = self._vector_relations(rel_type)
        match = self._relation_similarity(state, rows, 0)
        if vp is not None:
            match = match * self._relation_similarity(vp, rows, 1)
        result = [(int(row), float(score * self.c_plus[row]), self.slots[row, 1].clone(),
                   self.slots[row, 2].clone()) for row, score in zip(rows, match)
                  if float(score) > threshold]
        return sorted(result, key=lambda record: -record[1])

    @torch.no_grad()
    def evaluate(self, np1, vp, np2, *, rel_type=None):
        """Independent support/counterevidence for a vector-addressable relation."""
        rows = self._vector_relations(rel_type)
        if not len(rows):
            return self.slots.new_zeros(2)
        match = (self._relation_similarity(np1, rows, 0)
                 * self._relation_similarity(vp, rows, 1)
                 * self._relation_similarity(np2, rows, 2))
        return torch.stack(((match * self.c_plus[rows]).amax(),
                            (match * self.c_minus[rows]).amax()))
