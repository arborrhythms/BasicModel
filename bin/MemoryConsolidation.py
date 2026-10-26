"""Kept-close recurrence and identity promotion over today's occurrence rows.

Angle recurrence and evidence are rebuilt from retained ideas and cover
segments. Fixed reading seats preserve the first identity and promoted symbol.
"""
from dataclasses import replace

import torch


def _content(store, row):
    """The idea's content, with its sentence time removed for recurrence."""
    value = store.slots[row].detach().clone()
    if store.when_start is not None:
        value[store.when_start:store.when_start+4] = 0
    return value


@torch.no_grad()
def recurrence_groups(store):
    """Angle admission; rebuild evidence once from each retained assigned idea."""
    import math
    from util import TheXMLConfig
    # Fifteen degrees admits modest drift (cos = .966), without exact hashes.
    angle = float(TheXMLConfig.space('ConceptualSpace', 'recurrenceAngle', 15.))
    if not math.isfinite(angle) or not 0 < angle <= 180:
        raise ValueError('recurrenceAngle must lie in (0, 180] degrees')
    threshold = math.cos(math.radians(angle))
    groups, pending = {}, []
    for row in range(len(store)):
        if store.KINDS[int(store.record_kind[row])] in ('estimate', 'question', 'cover'):
            continue
        seat = int(store.recurrence_readings[row])
        (pending if seat < 0 else groups.setdefault(seat, [])).append(row)
    # Forgotten unpromoted groups release their fixed seats. Promoted summaries
    # keep their symbol association even while no occurrence is retained.
    for seat in store.recurrence_used.nonzero().flatten().tolist():
        if seat not in groups and int(store.recurrence_symbols[seat]) < 0:
            store.recurrence_used[seat] = False
            store.recurrence_codes[seat].zero_()
            store.recurrence_weights[seat] = 0
            store._recurrence_identities[seat] = None
    def update(seat, rows):
        values = torch.stack([_content(store, row) for row in rows])
        weights = (store.c_plus[rows] + store.c_minus[rows]).to(values)
        total = weights.sum()
        store.recurrence_codes[seat].copy_((values * weights[:, None]).sum(0) / total.clamp_min(1e-12))
        store.recurrence_weights[seat] = total
    for seat, rows in groups.items():
        update(seat, rows)
    for row in pending:
        value = _content(store, row)
        scores = torch.nn.functional.cosine_similarity(store.recurrence_codes, value[None], dim=-1)
        scores.masked_fill_(~store.recurrence_used, -torch.inf)
        # Strict angle tolerance; equal scores choose the earliest fixed seat.
        seat = int(scores.argmax())
        if not float(scores[seat]) > threshold:
            free = (~store.recurrence_used).nonzero().flatten()
            if not len(free):
                raise RuntimeError('angular recurrence exceeds the constructed recurrenceCapacity')
            seat = int(free[0])
            store.recurrence_used[seat] = True
            store._recurrence_identities[seat] = dict(store.raw_meaning_of(row).bindings).get('_identity')
        store.recurrence_readings[row] = seat
        groups.setdefault(seat, []).append(row)
        update(seat, groups[seat])
    from DocumentCover import segments
    membership = segments(store)
    store.recurrence_documents.zero_()
    for seat, rows in groups.items():
        # Multiple presentations of one cover concept are still one document.
        documents = {bytes(store.document_keys[store.location(cover)].tolist())
                     for row in rows for cover in membership.get(row, ())}
        store.recurrence_documents[seat] = len(documents)
    return groups


def _fold(owner, store, rows, *, order=1):
    """Evidence-weighted form centroid; bands and meanings are not folded."""
    cb = owner.similarity_codebook
    derived = getattr(cb, 'mereology', None)
    width = cb.W.shape[-1] if derived is None else derived.percept_width
    if not rows:
        return None
    weights = store.c_plus[rows] + store.c_minus[rows]
    if not bool(weights.any()):
        return None
    value = cb.W.new_zeros(cb.W.shape[-1])
    width = min(width, store.slots.shape[-1])
    value[:width] = (store.slots[rows, :width] * weights[:, None]).sum(0) / weights.sum()
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
def consolidate(owner, store, *, context=None, key=None, batch=0):
    """Three source documents for a reading, three sentences for an identity."""
    from Spaces import _concept_alloc_of
    alloc, pool = _concept_alloc_of(owner), owner.mtm
    close_key = key
    readings = recurrence_groups(store)
    plans, actions = [], []
    for reading, rows in readings.items():
        retained = int(store.recurrence_symbols[reading])
        if retained > 0 and retained not in alloc.retired:
            continue
        if int(store.recurrence_documents[reading]) < 3:
            continue
        # A retained non-word concept is already minted. Recurrence must not
        # duplicate that seat just because it was reached by another close.
        reference = store._recurrence_identities[reading]
        if reference is not None and reference[0] == 'sym':
            seat = owner._csw_row_of(reference[1])
            if seat is not None and torch.equal(store.slots[rows[0]], owner.similarity_codebook.W[seat]):
                store.recurrence_symbols[reading] = reference[1]
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
            pool.promote(key, entry.native)
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
    promoted = []
    for (order, value), (kind, key) in zip(plans, actions):
        symbol = owner.new_concept()
        alloc.reference_orders[symbol] = order
        alloc.symbol_types.setdefault(symbol, set()).update(('concept', 'object'))
        row = owner._csw_concept_row(order, symbol)
        if row is None:
            raise RuntimeError('consolidation inventory changed after preflight')
        derived = getattr(owner.similarity_codebook, 'mereology', None)
        if derived is not None:
            derived.remember_content(row, value)
            derived.placement[row].zero_()
            derived.placement_witnesses[row] = 0
            derived.placement_dirty[row] = True
        owner.similarity_codebook.W[row].copy_(value)
        promoted.append(row)
        alloc.settle(symbol)
        if kind == 'token':
            alloc.individuals.add(symbol)
            pool.promote(key, symbol)
        else:
            store.recurrence_symbols[key] = symbol
    derived = getattr(owner.similarity_codebook, 'mereology', None)
    if promoted and derived is not None:
        if context is None:
            context = owner.placement_context(batch)
        if close_key is None:
            from Occurrence import sentence_key
            close_key = sentence_key([derived.content_key(row) for row in promoted])
        derived.place_rows(promoted, context, close_key)
        owner.similarity_codebook.lookup_rows(promoted)
    return tuple((kind, key) for kind, key in actions)
