"""Promote an active idea once, preserving its selected reference's order."""
import math

import torch


def _content(store, rows):
    """Copy each selected idea's whole point, including .where and .when."""
    if torch.is_tensor(rows) and rows.ndim == 1:
        return store.slots.detach().index_select(0, rows)
    return store.slots[rows].detach().clone()


def eligible_activity(activity, promotion, written_now):
    """Compare stored activity to the binary64 teaching threshold on device."""
    # A scalar PyTorch comparison rounds promotion to the activity dtype.
    # Round it *up* to the first representable eligible value, including in
    # the subnormal range. All three quantities here are configuration data.
    fmt = torch.finfo(activity.dtype)
    exponent = math.frexp(max(fmt.tiny, promotion))[1] - 1
    spacing = math.ldexp(fmt.eps, exponent)
    cutoff = math.ceil(promotion / spacing) * spacing
    eligible = activity >= cutoff
    return eligible if written_now else eligible & (activity != 1.)


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
    occupied = set(alloc.layer()._tensor_row_keys)
    for order, _ in plans:
        start, end = owner.order_slice(order)
        row = next((row for row in range(start, end) if row not in occupied), None)
        if row is None:
            from Spaces import ConceptRowCapacityError
            raise ConceptRowCapacityError(f'concept order {order} capacity exhausted before memory consolidation')
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
    if rows is None:
        selected = torch.arange(len(store), device=store.activity.device)
    else:
        selected = torch.as_tensor(rows if torch.is_tensor(rows) else list(rows),
                                   device=store.activity.device, dtype=torch.long)
    eligible = eligible_activity(store.activity[selected], store.activity_config.promotion, written_now)
    selected = selected[eligible]
    sources = selected.tolist()
    if not sources:
        return ()
    points = _content(store, selected).to(owner.similarity_codebook.W)
    width = owner.similarity_codebook.W.shape[-1]
    points = torch.nn.functional.pad(points[:, :width], (0, max(0, width - points.shape[-1])))
    plans = []
    for index, value in zip(sources, points.unbind(), strict=True):
        reference = store.referent_of(index)
        if reference is not None:
            order = store.order_of_row(index)
        else:
            orders = [alloc.order_of(int(cid)) for cid in store.episodic_concepts[index].tolist() if cid >= 0]
            order = max(orders, default=-1) + 1
        if order < 0:
            raise ValueError('an active idea refers to an unavailable concept')
        plans.append((order, value))
    _preflight(owner, plans)
    store._promoting = True
    try:
        promoted = []
        for index,(order,value) in zip(sources,plans):
            symbol = owner.new_concept(order=order)
            alloc.symbol_types.setdefault(symbol,set()).update(('concept','object'))
            seat = owner._csw_concept_row(order,symbol)
            if seat is None:
                raise RuntimeError('promotion inventory changed after preflight')
            derived = getattr(owner.similarity_codebook,'mereology',None)
            if derived is not None:
                derived.remember_content(seat,value)
                derived.placement_dirty[seat] = True
            owner.similarity_codebook.trust.remember(symbol,value)
            from ConceptTrust import observe
            observe(owner,symbol)
            alloc.settle(symbol)
            store.referent[index] = store.referent.new_tensor((SYMBOL,symbol))
            store.activity[index] = 1.
            promoted.append(seat)
        derived = getattr(owner.similarity_codebook,'mereology',None)
        if derived is not None:
            owner.similarity_codebook.lookup_rows(promoted)
        return tuple(sources)
    finally:
        store._promoting = False
