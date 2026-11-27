"""Places: rows of the first Pi with entries on the where wires, nominated by
the cone over the occurrences met at a place.

An occurrence is a reading of a sentence's bank: the where wires of its
bracket in the bank's table (each wire on two lanes, ``c`` and ``1 - c``)
and its code on the what columns. The occurrences met at one band meet under
one key of MTM's count, ``('place', band)``, the count that nominates a
form. Their cone (rule 72) is the least of their coordinates: on the where
wires, the kept value of each; on the what, the parts they all hold and the
least whole that holds them all. A place met by forms that share no part
keeps no part.

A place's row is read as any row of the first Pi: at each attended percept,
the product of its entries there, on the what it keeps and on the where
wires (:meth:`BasicPercepts.BasicPercepts.read_bank`). It brings a concept
of order zero, as any row does (rule 21).
"""
import torch


def readings(bank):
    """The where wires of each reading's bracket ``[B, readings, width]``,
    whether the reading has one, and its code on the what columns ``[B,
    readings, what]``."""
    span = bank.span
    same = ((span[:, None, :, 0] == bank.where[..., :1]) & (span[:, None, :, 1] == bank.where[..., 1:])
            & (span[:, None, :, 1] > span[:, None, :, 0]))                       # [B, K, O]
    has = same.any(-1)
    # A reading with no occurrence (and a bank with none) reads a row of
    # nothing, one past the occurrences.
    O = same.shape[-1]
    at = torch.where(has, torch.nn.functional.pad(same, (0, 1)).int().argmax(-1), O)
    from BasicPercepts import WHERE_WIRES
    columns = torch.nn.functional.pad(bank.table[..., bank.width:bank.width + WHERE_WIRES, 0], (0, 0, 0, 1))
    wires = columns.gather(1, at[..., None].expand(-1, -1, columns.shape[-1]))
    what = torch.nn.functional.pad(bank.table[..., :bank.width, 0], (0, 0, 0, 1))
    what = what.gather(1, at[..., None].expand(-1, -1, what.shape[-1]))
    return wires, has, what


def cone_prefixes(points, predecessors, parts_end, rounds):
    """Meet the observations of each place through the current visit, in parallel."""
    for _ in range(rounds):
        before = points.index_select(0, predecessors)
        points = torch.cat((torch.minimum(points[:, :parts_end], before[:, :parts_end]),
                            torch.maximum(points[:, parts_end:], before[:, parts_end:])), 1)
        predecessors = predecessors.index_select(0, predecessors)
    return points


def definition_keys(plus, minus, what):
    """Existing Pi address keys for the integer membership and rounded where wires."""
    width = plus.shape[-1]
    columns = (plus, minus) if what is None else (plus, minus, what.ne(0))
    return [('place', tuple(round(v, 6) for v in row[:width]),
             tuple(round(v, 6) for v in row[width:2*width]),
             tuple(i for i, live in enumerate(row[2*width:]) if live))
            for row in torch.cat(columns, -1).tolist()]


def _quantize_keys(c):
    """Canonical binary32 where-wire key rows, ``[*, width]`` int64 at 1e-6."""
    return torch.round(c.detach().float() * 1e6).to(torch.int64)


def _host_key(qrow):
    """Host tuple key derived from a quantized int64 row of ``_quantize_keys``."""
    return tuple(int(v) / 1e6 for v in qrow)


def sync_compiled_places(owner, *, promoted=None):
    """Synchronize host place registry and nominations with compiled cone buffers."""
    basic = getattr(owner, 'basic_percepts', None)
    if basic is None or not hasattr(basic, '_place_cone_used'):
        return []
    used = min(int(basic._place_cone_used), int(basic._place_cone_keys.shape[0]))
    if not used:
        return []
    registry = basic._place_cone_rows
    qrows = basic._place_cone_keys[:used].tolist()
    if len(registry) < used:
        for seat in range(used):
            key = _host_key(qrows[seat])
            registry.setdefault(key, seat)
    counts = basic._place_cone_counts[:used].tolist()
    synced = basic.__dict__.setdefault('_place_cone_synced_counts', {})
    delta_seats = [seat for seat in range(used) if counts[seat] > synced.get(seat, 0)]
    if not delta_seats:
        return []
    from BasicPercepts import WHERE_WIRES
    reading = owner.reading(0)
    chosen_seats = []
    for seat in delta_seats:
        key = _host_key(qrows[seat])
        delta = counts[seat] - synced.get(seat, 0)
        synced[seat] = counts[seat]
        nominated = False
        for _ in range(delta):
            if owner.nominates(('place', key), reading):
                nominated = True
        if nominated:
            chosen_seats.append(seat)
    if not chosen_seats:
        return []
    storage = basic._place_cone_bits.view(torch.float32)
    nominated = storage.index_select(0, basic._place_cone_keys.new_tensor(chosen_seats))
    plus, minus, kept = nominated.split((WHERE_WIRES, WHERE_WIRES, basic.width), -1)
    from functools import partial
    publish = admit if promoted is None else partial(admit, promoted=set(promoted))
    iw = basic.identity_width
    made = []
    for i, key in enumerate(definition_keys(plus, minus, kept)):
        has_part = bool(key[-1] and key[-1][0] < iw)
        row = publish(owner, plus[i], minus[i], kept[i] if has_part else None,
                      key=key if has_part else (*key[:3], ()))
        if row is not None:
            made.append(row)
    return made


def _meet_tensor(owner, basic, c, has, what):
    """Compiled fullgraph tensor update of pre-allocated place-cone buffers."""
    B, K, width = c.shape
    iw = int(basic.identity_width)
    c = c.detach().float()
    N = B * K
    live = has.reshape(N)
    qkeys = _quantize_keys(c).reshape(N, width)
    keys_buf = basic._place_cone_keys
    M = keys_buf.shape[0]
    used = basic._place_cone_used
    used_mask = torch.arange(M, device=c.device) < used
    matches_existing = (
        (qkeys[:, None, :] == keys_buf[None, :, :]).all(-1) & used_mask[None, :] & live[:, None])
    has_existing = matches_existing.any(-1)
    existing_row = matches_existing.long().argmax(-1)
    tri = torch.tril(torch.ones(N, N, dtype=torch.bool, device=c.device), diagonal=-1)
    matches_prev = (qkeys[:, None, :] == qkeys[None, :, :]).all(-1) & tri & (live[:, None] & live[None, :])
    has_prev = matches_prev.any(-1)
    novel = live & ~has_existing & ~has_prev
    novel_seat = used + novel.long().cumsum(0) - 1
    first_idx = torch.where(matches_prev, torch.arange(N, device=c.device)[None, :], N).amin(-1)
    first_novel_seat = novel_seat.gather(0, first_idx.clamp(0, max(0, N - 1)))
    seat = torch.where(has_existing, existing_row, torch.where(has_prev, first_novel_seat, novel_seat))
    valid_seat = live & (seat < M)
    safe_seat = seat.clamp(0, M - 1)
    obs = torch.cat((c, 1 - c, what.detach().float()), -1).reshape(N, -1)
    split = 2 * width + iw
    storage = basic._place_cone_bits.view(torch.float32)
    cur_parts = storage[:, :split].clone()
    cur_wholes = storage[:, split:].clone()
    is_new_seat = valid_seat & ~has_existing
    ones = torch.ones_like(cur_parts)
    init_parts = torch.where(
        (torch.arange(M, device=c.device) < used)[:, None], cur_parts, ones)
    zeros = torch.zeros_like(cur_wholes)
    init_wholes = torch.where(
        (torch.arange(M, device=c.device) < used)[:, None], cur_wholes, zeros)
    obs_parts = torch.where(valid_seat[:, None], obs[:, :split], 1.0)
    obs_wholes = torch.where(valid_seat[:, None], obs[:, split:], 0.0)
    new_parts = init_parts.scatter_reduce(0, safe_seat[:, None].expand(-1, split), obs_parts, reduce='amin', include_self=True)
    new_wholes = init_wholes.scatter_reduce(
        0, safe_seat[:, None].expand(-1, obs.shape[1] - split), obs_wholes, reduce='amax', include_self=True)
    updated = torch.cat((new_parts, new_wholes), -1)
    storage.copy_(updated)
    # Deterministic new-seat key write without scatter_ races on duplicate or inactive rows.
    slot_match = (safe_seat[:, None] == torch.arange(M, device=c.device)[None, :]) & is_new_seat[:, None]
    has_write = slot_match.any(0)
    write_src = slot_match.long().argmax(0)
    keys_buf.copy_(torch.where(has_write[:, None], qkeys.index_select(0, write_src), keys_buf))
    basic._place_cone_counts.scatter_add_(0, safe_seat, valid_seat.long())
    basic._place_cone_used.add_(novel.long().sum().clamp_max((M - used).clamp_min(0)))
    return []


def meet(owner, bank, *, promoted=None):
    """Count the occurrences of a sentence's bank at their places and
    nominate a place's row on the count that nominates a form."""
    basic = getattr(owner, 'basic_percepts', None)
    if basic is None or bank.registry is None or not bank.token.shape[1]:
        return []
    c, has, what = readings(bank)
    if torch.compiler.is_compiling():
        return _meet_tensor(owner, basic, c, has, what)
    made = sync_compiled_places(owner, promoted=promoted)
    iw = bank.identity_width
    c = c.detach().float()
    width, readings_per_page = c.shape[-1], c.shape[1]
    # Both host and compiled paths share the same quantized int64 key table.
    qrows = _quantize_keys(c).tolist()
    has_rows = has.tolist()
    visits, positions, prior_rows, predecessors = [], [], [], []
    last, depth = {}, {}
    for b, (occ_row, qpage) in enumerate(zip(has_rows, qrows)):
        for k, (occupied, wires) in enumerate(zip(occ_row, qpage)):
            if not occupied:
                continue
            key = _host_key(wires)
            at = len(visits)
            if key in last:
                before = last[key]
            elif key in basic._place_cone_rows:
                prior_rows.append(basic._place_cone_rows[key])
                before = -len(prior_rows)
            else:
                before = at
            depth[key] = depth.get(key, 0) + int(before != at)
            predecessors.append(before)
            visits.append((b, key))
            positions.append(b * readings_per_page + k)
            last[key] = at
    if not visits:
        return made

    registry = basic._place_cone_rows
    for key in last:
        if key not in registry:
            registry[key] = len(registry)
    storage = basic.place_cone_storage(len(registry))
    synced = basic.__dict__.setdefault('_place_cone_synced_counts', {})
    if registry:
        row_ids = list(registry.values())
        key_rows = torch.tensor([[round(v * 1e6) for v in k] for k in registry], dtype=torch.int64, device=c.device)
        basic._place_cone_keys.index_copy_(0, key_rows.new_tensor(row_ids), key_rows)
        basic._place_cone_used.fill_(len(registry))
    visit_counts = {}
    for _b, key in visits:
        seat = registry[key]
        visit_counts[seat] = visit_counts.get(seat, 0) + 1
    for seat, delta in visit_counts.items():
        basic._place_cone_counts[seat] += delta
        synced[seat] = int(basic._place_cone_counts[seat])
    observations = torch.cat((c, 1 - c, what.detach().float()), -1).flatten(0, 1)
    seats = torch.tensor(positions, dtype=torch.long, device=c.device)
    points = observations.index_select(0, seats)
    if prior_rows:
        points = torch.cat((points, storage.index_select(0, seats.new_tensor(prior_rows))), 0)
    parents = seats.new_tensor([len(visits) - 1 - p if p < 0 else p for p in predecessors]
                               + list(range(len(visits), len(points))))
    points = cone_prefixes(points, parents, 2 * width + iw, max(depth.values()).bit_length())
    storage.index_copy_(0, seats.new_tensor([registry[key] for key in last]),
                        points.index_select(0, seats.new_tensor(list(last.values()))))
    # A nomination sees only its own prefix. Later visits have already been
    # folded into storage, but cannot teach an earlier nominated percept.
    chosen = [at for at, (b, key) in enumerate(visits)
              if owner.nominates(('place', key), owner.reading(b))]
    if not chosen:
        return made
    nominated = points.index_select(0, seats.new_tensor(chosen))
    plus, minus, kept = nominated.split((width, width, basic.width), -1)
    # The sentence lookup already read the promoted rows. Reuse that
    # membership through these ordered admissions, including repeats.
    from functools import partial
    publish = admit if promoted is None else partial(admit, promoted=set(promoted))
    for i, key in enumerate(definition_keys(plus, minus, kept)):
        has_part = bool(key[-1] and key[-1][0] < iw)
        row = publish(owner, plus[i], minus[i], kept[i] if has_part else None,
                    key=key if has_part else (*key[:3], ()))
        if row is not None:
            made.append(row)
    return made


def admit(owner, plus, minus, what, *, key=None, promoted=None):
    """A place's row of the first Pi, promoted, and its concept of order zero
    reading it; ``None`` where there is no room."""
    basic = owner.basic_percepts
    try:
        owner._preflight_concept_row(0, context='place admission')
        row = basic.reserve_place(plus, minus, what, key=key)
    except (RuntimeError, OverflowError):
        return None
    return bring(owner, row, 'place', promoted=promoted)


def nominate(owner, percepts, kind):
    """Each row of ``percepts`` ``[B, A]`` (``-1`` where none) counted under
    ``(kind, row)`` once for each stream that holds it, as a form is, and
    brought with its concept on the count that nominates it."""
    for percept in percepts.unique().tolist():
        if percept >= 0:
            for b in (percepts == percept).any(1).nonzero().flatten().tolist():
                if owner.nominates((kind, percept), owner.reading(b)):
                    bring(owner, percept, kind)


def bring(owner, row, kind, *, promoted=None):
    """A nominated row promoted, and its concept of order zero, of the symbol
    type ``kind``, reading it; ``None`` where order zero has no room."""
    from Spaces import _concept_alloc_of
    try:
        owner._preflight_concept_row(0, context=f'{kind} admission')
    except RuntimeError:
        return None
    owner.basic_percepts.promote(row, promoted=promoted)
    alloc = _concept_alloc_of(owner)
    if any(percept == row for _, percept, _ in alloc.order_zero_members):
        return row
    concept = owner.new_concept(order=0)
    alloc.symbol_types.setdefault(concept, set()).add(kind)
    inventory = owner._csw_concept_row(0, concept)
    if inventory is None:
        return row
    owner.bring_concept(row, inventory)
    return row
