"""Kept-close recurrence and identity promotion over today's occurrence rows.

Evidence is rebuilt from retained cover segments. Only the association
between a reading and its minted symbol persists; no second value cache.
"""
from dataclasses import replace

import torch


def _fold(owner, store, rows, *, order=1):
    from IdentityRows import carrier, coordinates
    values, weights = [], []
    for row in rows:
        if int(store.role_mask[row].sum()) != 1:
            continue
        value = owner.similarity_codebook.W.new_zeros(owner.similarity_codebook.W.shape[-1])
        point = store.slots[row, 0].to(value)
        value[:point.numel()] = point
        values.append(carrier(owner, value) if order == 0 else coordinates(owner, value))
        weights.append(torch.stack((store.c_plus[row], store.c_minus[row])))
    if not values:
        return None
    values, weights = torch.stack(values), torch.stack(weights).to(values[0])
    # Evidence is not frequency or a union. Each lane has its own weighted
    # centroid; an unwitnessed lane remains unknown.
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if derived is None:
        weight = weights.sum(-1, keepdim=True)
    else:
        weight = weights.sum(-1, keepdim=True).expand_as(values).clone()
        start, size = derived.percept_event_width, derived.reserved_pairs
        weight[:, start:start+size] = weights[:, :1]
        weight[:, start+size:start+2*size] = weights[:, 1:]
    value = (values * weight).sum(0) / weight.sum(0).clamp_min(torch.finfo(values.dtype).tiny)
    return value if bool(weight.any()) else None


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
def consolidate(owner, store):
    """Three source documents for a reading, three sentences for an identity."""
    from Spaces import _concept_alloc_of
    alloc, pool = _concept_alloc_of(owner), owner.mtm
    from DocumentCover import segments
    membership = segments(store)
    retained = owner.__dict__.setdefault('_recurring_readings', {})
    readings = {}
    for row in range(len(store)):
        if (int(store.role_mask[row].sum()) != 1 or
                store.KINDS[int(store.record_kind[row])] in ('estimate', 'question', 'cover')):
            continue
        reference = store.raw_meaning_of(row).role_refs[0]
        if reference is not None and reference[0] == 'sym':
            readings.setdefault(reference[1], []).append(row)
    plans, actions = [], []
    for reading, rows in readings.items():
        if reading in retained and retained[reading] not in alloc.retired:
            continue
        documents = {cover for row in rows for cover in membership.get(row, ())}
        if len(documents) < 3:
            continue
        # A retained non-word concept is already minted. Recurrence must not
        # duplicate that seat just because it was reached by another close.
        if owner._csw_row_of(reading) is not None and 'word' not in alloc.symbol_types.get(reading, ()):
            retained[reading] = reading
            continue
        value = _fold(owner, store, rows, order=int(store.order[rows[0]]))
        if value is not None:
            plans.append((int(store.order[rows[0]]), value))
            actions.append(('reading', reading))
    for key, entry in pool.entries.items():
        if entry.symbol is not None:
            continue
        rows = [store.location(address) for address in entry.occurrences]
        rows = [row for row in rows if isinstance(row, int)]
        sentences = set(rows)
        if len(sentences) < 3:
            continue
        if entry.native is not None:
            pool.entries[key] = replace(entry, symbol=entry.native)
            continue
        # Mention recurrence admits promotion; only actual occurrences give
        # the promoted identity content. Alternative ideas stay references.
        actual = set(entry.occurrences) | {event.occurrence for event in entry.events if event.role == 0}
        rows = [store.location(address) for address in actual]
        rows = [row for row in rows if isinstance(row, int) and int(store.order[row]) == 0]
        value = _fold(owner, store, rows)
        if value is not None:
            plans.append((1, value))
            actions.append(('token', key))
    if plans:
        _preflight(owner, plans)
    for (order, value), (kind, key) in zip(plans, actions):
        symbol = owner.new_concept()
        alloc.reference_orders[symbol] = order
        alloc.symbol_types.setdefault(symbol, set()).update(('concept', 'object'))
        row = owner._csw_concept_row(order, symbol)
        if row is None:
            raise RuntimeError('consolidation inventory changed after preflight')
        owner.similarity_codebook.W[row].copy_(value)
        alloc.settle(symbol)
        if kind == 'token':
            alloc.individuals.add(symbol)
            pool.entries[key] = replace(pool.entries[key], symbol=symbol)
        else:
            retained[key] = symbol
    return tuple((kind, key) for kind, key in actions)
