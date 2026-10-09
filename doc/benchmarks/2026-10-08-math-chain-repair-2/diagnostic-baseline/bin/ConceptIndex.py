"""Derived taxonomy and direct word references over native concept records."""
import torch
from Spaces import _concept_alloc_of


def _inherit_forms(cs, source, target):
    """A grammatical association inherits forms from the definition index."""
    alloc = _concept_alloc_of(cs)
    forms = {form for form, ids in alloc.word_forms.items() if source in ids}
    index = cs._definition_index()
    if index is not None:
        words = (source,) if source in index.word_ids else index.words(source)
        for word in words:
            forms.update(index.description(word)['forms'])
    for form in forms:
        alloc.word_forms.setdefault(form, set()).add(int(target))


def _available(cs, values):
    alloc = _concept_alloc_of(cs)
    values = tuple(dict.fromkeys(map(int, values)))
    if any(cid not in alloc.placement or cid in alloc.retired for cid in values):
        raise ValueError('taxonomy requires existing native concepts')
    return values


def _symbol_at(cs, cid, order):
    """Raise a member by sigma singletons until it reaches the input order."""
    alloc = _concept_alloc_of(cs)
    current = cs._concept_source_order(cid)
    while current < order:
        candidates = [candidate for candidate in alloc.placement
            if candidate not in alloc.retired
            and not alloc.refs(candidate, 'word')
            and cs._concept_source_order(candidate) == current + 1
            and alloc.refs(candidate, 'part') == [('sym', cid)]
            and not alloc.refs(candidate, 'whole')]
        previous = cid
        cid = min(candidates) if candidates else cs.singleton_concept(cid)
        if not candidates:
            basis = cs.similarity_codebook.getW()
            source_row, destination = cs._csw_row_of(previous), cs._csw_row_of(cid)
            if source_row is not None and destination is not None:
                with torch.no_grad():
                    basis[destination].copy_(basis[source_row])
        current += 1
        alloc.reference_orders[cid] = current
        alloc.settle(cid)
    return cid


def index_part_row(cs, part, whole):
    """Testimony adds a sigma edge to the parent at the next native order."""
    alloc = _concept_alloc_of(cs)
    part, whole = _available(cs, (part, whole)) if part != whole else (int(part), int(whole))
    if part == whole:
        return whole  # improper parthood needs no cyclic definition
    index = cs._definition_index()
    words = set(() if index is None else index.word_ids)
    if part in words or whole in words:
        raise ValueError('language taxonomy relates object concepts, never words')
    order = cs._concept_source_order(part) + 1
    if cs._concept_source_order(whole) > order:
        raise ValueError('a taxonomy parent must be exactly one order above its part')
    key = ('part-kind', whole, order)
    parent = whole if cs._concept_source_order(whole) == order else alloc.relate_idx.get(key)
    if parent is None:
        parent = _symbol_at(cs, whole, order)
        alloc.relate_idx[key] = parent
    alloc.add(parent, 'part', ('sym', part))
    alloc.reference_orders[parent] = order
    alloc.singletons.add(parent)
    alloc.raised.add(parent)
    alloc.settle(parent)
    _inherit_forms(cs, whole, parent)
    cs._populate_concept_weights(parent)
    return int(parent)


def taxonomy_children(cs, cid):
    alloc = _concept_alloc_of(cs)
    if cid in alloc.retired:
        return []
    return sorted({ref[1] for ref in alloc.refs(cid, 'part')
        if isinstance(ref, tuple) and ref[0] == 'sym' and ref[1] not in alloc.retired})


def taxonomy_parents(cs, cid):
    alloc = _concept_alloc_of(cs)
    return [parent for parent in alloc.placement if cid in taxonomy_children(cs, parent)]


class ClauseTaxonomyPlan:
    """Validate a closing's taxonomy changes before either owner mutates."""

    def __init__(self, cs, store, clause):
        from dataclasses import replace
        self.cs, self.store = cs, store
        self.alloc = _concept_alloc_of(cs)
        self.definitions, self.edges, self.keys, self.forms = {}, set(), {}, []
        self.points = {}
        self._prepared = {}
        visiting = set()

        def prepare(value):
            if id(value) in self._prepared:
                return self._prepared[id(value)]
            if id(value) in visiting:
                raise ValueError('completed clause references contain a cycle')
            visiting.add(id(value))
            children = tuple(prepare(child) for child in value.children)
            companions = tuple(prepare(child) for child in value.companions)
            from ClauseRow import ClausePredicate
            refs = tuple(self.term(ref) if isinstance(ref, ClausePredicate) else ref for ref in value.refs)
            factored = (None if value.factored_refs is None else tuple(
                self.term(ref) if isinstance(ref, ClausePredicate) else ref for ref in value.factored_refs))
            relation = store.clause_relation(value)
            if (relation == 'part' and all(type(refs[i]) is int and refs[i] > 0
                                           for i in (0, 2))
                    and all(store.index_of_row(refs[i]) is None for i in (0, 2))):
                left, right = refs[0], refs[2]
                _available(cs, (cid for cid in (left, right) if cid not in self.definitions))
                index = cs._definition_index()
                words = set(() if index is None else index.word_ids)
                if set((left, right)) & words:
                    raise ValueError('language taxonomy relates objects, never words')
                if left != right:
                    order = self.order(left) + 1
                    if self.order(right) > order:
                        raise ValueError('taxonomy parent exceeds the next symbolization order')
                    key = ('part-kind', right, order)
                    parent = self.keys.get(key, self.alloc.relate_idx.get(key))
                    if parent is None:
                        parent = self.symbol_at(right, order)
                        self.keys[key] = parent
                    self.edges.add((parent, left))
                    self.forms.append((right, parent))
                    # The index may symbolize the parent, but the completed
                    # row keeps the reference actually held when it ended.
            result = replace(value, relation=relation, refs=refs, children=children, companions=companions,
                             factored_refs=factored)
            self._prepared[id(value)] = result
            visiting.remove(id(value))
            return result
        self.clause = prepare(clause)
        if not self.validate_rows():
            # Optional symbolization is all-or-nothing. The completed field
            # still writes its actual references and the reading goes on.
            self.definitions, self.edges, self.keys, self.forms = {}, set(), {}, []
            self.points = {cid: point for cid, point in self.points.items()
                           if (1 << 61) <= cid < (1 << 62)}

    def validate_rows(self):
        """Reject invalid phrase widths and full order blocks before allocation."""
        from types import SimpleNamespace
        cs, alloc = self.cs, self.alloc
        width = cs.similarity_codebook.getW().shape[-1]
        if any(point is None or point.shape != (width,) for point in self.points.values()):
            raise ValueError('phrase point must fit its native conceptual row')
        if self.definitions:
            _, end, capacity = cs._concept_capacity_window(len(self.definitions))
            if capacity is not None and end > capacity:
                return False
        layer = alloc.layer(0)
        trial = SimpleNamespace(nOutput=layer.nOutput,
            _tensor_rows=dict(layer._tensor_rows),
            _tensor_row_keys=dict(layer._tensor_row_keys), _row_next=dict(layer._row_next))
        caps = cs._order_caps()
        for cid in sorted(set(self.definitions) | {parent for parent, _ in self.edges}):
            order = self.order(cid) if getattr(cs, '_symbolic_order', 0) > 0 else 0
            if getattr(cs, '_concept_binding', 'mixing') == 'aligned':
                key, base, capacity = ('shared', cid), 0, cs.nVectors
            elif order == 0:
                key = ('snap', cid)
                base = 0 if trial._row_next.get(0, 0) < caps[0] else sum(caps)
                capacity = caps[0] if base == 0 else cs.nVectors - base
            elif len(caps) == 2:
                key, base, capacity = ('pool', cid), caps[0], caps[1]
            else:
                order = min(order, len(caps) - 1)
                key, base, capacity = (f'o{order}', cid), sum(caps[:order]), caps[order]
            if type(layer).assign_row(trial, key, capacity=capacity, base=base) is None:
                return False
        return True

    def bind_particular(self, clause, row_id):
        """Associate the subject's form with its ended, order-one occurrence."""
        if clause.slots.shape[0] != 1:
            return
        from ThoughtReferences import open_slots
        if open_slots(clause.meaning):
            return
        sources = {clause.refs[0]} if type(clause.refs[0]) is int else set()
        if clause.subject_word_id > 0:
            sources.add(clause.subject_word_id)
        for source in sources:
            _inherit_forms(self.cs, source, row_id)
        aliases = set(sources)
        for identities in self.alloc.word_forms.values():
            if identities.intersection(sources):
                aliases.update(identities)
        self.store.fill_forward_references(aliases, row_id, clause.point)

    def order(self, cid):
        return self.definitions[cid][1] if cid in self.definitions else self.cs._concept_source_order(cid)

    def term(self, term):
        """Retain a predicate's live slot value until its occurrence is written."""
        cid = term.identity
        self.points[cid] = term.point
        return cid

    def symbol_at(self, source, order):
        alloc, cs = self.alloc, self.cs
        current = self.order(source)
        while current < order:
            key = ('singleton', source)
            candidates = [cid for cid in alloc.placement
                if cid not in alloc.retired and not alloc.refs(cid, 'word')
                and cs._concept_source_order(cid) == current + 1
                and alloc.refs(cid, 'part') == [('sym', source)]
                and not alloc.refs(cid, 'whole')]
            target = self.keys.get(key, alloc.relate_idx.get(key))
            if target is None and candidates:
                target = min(candidates)
            if target is None:
                target = alloc.next_id + len(self.definitions)
                self.definitions[target] = source, current + 1
                self.points[target] = (self.points[source] if source in self.points
                                       else self.store.point_of_row(source))
                self.keys[key] = target
            source, current = target, current + 1
        return source

    @property
    def count(self):
        return len(self.definitions)

    def commit(self):
        cs, alloc = self.cs, self.alloc
        for cid, (source, order) in self.definitions.items():
            if cs.new_concept() != cid:
                raise RuntimeError('native concept allocation changed during clause preflight')
            alloc.reference_orders[cid] = order
            alloc.singletons.add(cid)
            alloc.raised.add(cid)
            for member in (source,):
                alloc.add(cid, 'part', ('sym', member))
        alloc.relate_idx.update(self.keys)
        for parent, part in sorted(self.edges):
            alloc.add(parent, 'part', ('sym', part))
            alloc.singletons.add(parent)
            alloc.raised.add(parent)
        for source, parent in self.forms:
            _inherit_forms(cs, source, parent)
        for cid in sorted(set(self.definitions) | {parent for parent, _ in self.edges}):
            cs._populate_concept_weights(cid)
            if cid in self.definitions:
                row = cs._csw_row_of(cid)
                if row is None:
                    row = cs._csw_concept_row(
                        self.order(cid) if getattr(cs, '_symbolic_order', 0) > 0 else 0, cid)
                basis = cs.similarity_codebook.getW()
                if row is None or self.points[cid].shape != basis[row].shape:
                    raise ValueError('phrase point must fit its native conceptual row')
                basis[row].copy_(self.points[cid].to(basis))
