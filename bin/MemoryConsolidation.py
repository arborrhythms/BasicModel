"""Promote an active idea once, preserving its selected reference's order."""
import torch


def _content(store, row):
    value = store.slots[row].detach().clone()
    if store.when_start is not None:
        value[store.when_start:store.when_start+4] = 0
    return value


def _fold(owner, store, rows, *, order=1, members=None):
    """Evidence-weighted form centroid; bands and meanings are not folded."""
    from FormCentroids import convex_mean
    cb = owner.similarity_codebook
    derived = getattr(cb, 'mereology', None)
    width = cb.W.shape[-1] if derived is None else derived.percept_width
    if not rows:
        return None
    if members is not None:
        weights = cb.W.new_tensor([members[row] for row in rows])
        codes = cb.lookup_rows(rows)[:, :width]
        value = cb.W.new_zeros(cb.W.shape[-1])
        value[:width] = convex_mean(weights, codes)[0]
        return value
    weights = store.trust[rows].abs()
    if not bool(weights.any()):
        return None
    value = cb.W.new_zeros(cb.W.shape[-1])
    width = min(width, store.slots.shape[-1])
    value[:width] = convex_mean(weights, store.slots[rows, :width])[0]
    return value


def _preflight(owner, plans):
    """Check all seats together, without reserving or changing the inventory."""
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(owner)
    owner._preflight_concept_allocation(len(plans), context='memory consolidation')
    caps = owner._order_caps()
    occupied = set(alloc.layer()._tensor_row_keys)
    for order, _ in plans:
        if getattr(owner, '_concept_binding', 'mixing') == 'aligned':
            pools = ((0, int(owner.nVectors)),)
        elif order == 0:
            pools = ((0, caps[0]), (sum(caps), int(owner.nVectors)-sum(caps)))
        else:
            rung = min(order, len(caps)-1)
            pools = ((sum(caps[:rung]), caps[rung]),
                     (sum(caps), int(owner.nVectors)-sum(caps)))
        row = next((base+i for base, count in pools for i in range(count)
                    if base+i not in occupied), None)
        if row is None:
            raise RuntimeError('memory consolidation exceeds the concept row pool')
        occupied.add(row)


@torch.no_grad()
def consolidate(owner, store, *, context=None, key=None, batch=0, rows=None, written_now=False):
    """Mint each eligible idea and retain the minted concept as its referent."""
    from Spaces import _concept_alloc_of
    from Memory import SYMBOL
    from MemoryActivity import deferred
    if deferred() or store.__dict__.get('_promoting', False):
        return ()
    alloc = _concept_alloc_of(owner)
    selected = range(len(store)) if rows is None else rows
    plans, sources = [], []
    for index in selected:
        index = int(index)
        activity = float(store.activity[index])
        if activity < store.activity_config.promotion:
            continue
        reference = store.referent_of(index)
        # A promoted row has activity one and refers to its minted concept.
        # A just-written one is considered before its activity is made permanent.
        if activity == 1. and not written_now:
            continue
        if reference is not None:
            order = store.order_of_row(index)
        else:
            orders = [alloc.order_of(int(cid)) for cid in store.episodic_concepts[index].tolist() if cid >= 0]
            order = max(orders, default=-1) + 1
        if order < 0:
            raise ValueError('an active idea refers to an unavailable concept')
        point = _content(store, index)
        value = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
        value[:min(len(point),len(value))] = point[:len(value)].to(value)
        plans.append((order, value)); sources.append(index)
    if not plans:
        return ()
    _preflight(owner, plans)
    store._promoting = True
    try:
        promoted = []
        for index,(order,value) in zip(sources,plans):
            symbol = owner.new_concept()
            alloc.reference_orders[symbol] = order
            alloc.symbol_types.setdefault(symbol,set()).update(('concept','object'))
            seat = owner._csw_concept_row(order,symbol)
            if seat is None:
                raise RuntimeError('promotion inventory changed after preflight')
            derived = getattr(owner.similarity_codebook,'mereology',None)
            if derived is not None:
                derived.remember_content(seat,value)
                derived.placement[seat].zero_()
                derived.placement_witnesses[seat] = 0
                derived.placement_dirty[seat] = True
            owner.similarity_codebook.W[seat].copy_(value)
            owner.similarity_codebook.trust.remember(symbol,value)
            from ConceptTrust import observe
            observe(owner,symbol,value)
            alloc.settle(symbol)
            store.referent[index] = store.referent.new_tensor((SYMBOL,symbol))
            store.activity[index] = 1.
            promoted.append(seat)
        derived = getattr(owner.similarity_codebook,'mereology',None)
        if derived is not None:
            if context is None:
                context = owner.placement_context(batch)
            if key is None:
                from Occurrence import sentence_key
                key = sentence_key([derived.content_key(row) for row in promoted])
            derived.place_rows(promoted,context,key)
            owner.similarity_codebook.lookup_rows(promoted)
        return tuple(sources)
    finally:
        store._promoting = False
