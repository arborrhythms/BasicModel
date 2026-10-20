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


@torch.no_grad()
def rebuild_ltm_taxonomy(cs, store):
    """Withdraw only edges whose recorded testimony has all been forgotten.

    Native edges (including legacy edges with no provenance) are independent.
    Surviving learned coefficients are preserved, not reinitialized by replay.
    """
    # Part edges are the durable store, so forgetting a separate LTM1
    # occurrence does not erase or re-create its relation evidence.
    meaning = getattr(cs.similarity_codebook, 'mereology', None)
    if meaning is not None:
        meaning.begin_forward()
