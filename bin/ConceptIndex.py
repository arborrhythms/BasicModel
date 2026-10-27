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


def index_part_row(cs, part, whole):
    """Admit a witnessed member to the next-order class of the shared side."""
    from LTM1 import SYMBOL
    alloc = _concept_alloc_of(cs)
    part_ref, whole_ref = alloc.parts.reference(part), alloc.parts.reference(whole)
    if part_ref == whole_ref:
        return whole_ref[1]
    part_order = (alloc.order_of(part_ref[1]) if part_ref[0] == SYMBOL
                  else cs._concept_source_order(part_ref[1]))
    whole_order = (alloc.order_of(whole_ref[1]) if whole_ref[0] == SYMBOL
                   else cs._concept_source_order(whole_ref[1]))
    base = alloc.relate_idx.get(('part-base', whole_ref[1])) if whole_ref[0] == SYMBOL else None
    if base is not None and part_order == whole_order - 1:
        # A later member joins the witnessed equivalence class. Naming its
        # parent after the first raise must not raise that same class again.
        whole_ref, whole_order = base, whole_order - 1
    if part_order < 0 or whole_order < 0:
        raise ValueError('a class requires witnessed member and shared-side orders')
    order = whole_order + 1
    if part_order != whole_order:
        raise ValueError('a class member must have the shared side order')
    key = ('part-kind', whole_ref, order)
    parent = alloc.relate_idx.get(key)
    if parent is None:
        parent = cs.new_concept()
        alloc.relate_idx[key] = parent
        alloc.relate_idx['part-base', parent] = whole_ref
    alloc.parts.add(part_ref, (SYMBOL, parent), truth_plus=1.)
    alloc.reference_orders[parent] = order
    alloc.singletons.add(parent)
    alloc.raised.add(parent)
    alloc.settle(parent)
    if whole_ref[0] == SYMBOL:
        _inherit_forms(cs, whole_ref[1], parent)
    destination = cs._csw_concept_row(order, parent)
    meaning = getattr(cs.similarity_codebook, 'mereology', None)
    if meaning is not None and part_ref[0] == SYMBOL:
        meaning.placement_derived[destination] = True
        meaning.placement_valid[destination] = False
        # Admission can allocate the destination after the relation is written.
        cs._priming_edge_revision = None
    source = cs._csw_row_of(whole_ref[1]) if whole_ref[0] == SYMBOL else None
    if source is not None:
        with torch.no_grad():
            cs.similarity_codebook.W[destination].copy_(cs.similarity_codebook.W[source])
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
    # Part edges are the durable store, so forgetting a separate LTM0
    # occurrence does not erase or re-create its relation evidence.
    meaning = getattr(cs.similarity_codebook, 'mereology', None)
    if meaning is not None:
        meaning.begin_forward()
