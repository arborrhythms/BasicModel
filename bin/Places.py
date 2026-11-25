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


def meet(owner, bank):
    """Count the occurrences of a sentence's bank at their places and
    nominate a place's row on the count that nominates a form."""
    basic = getattr(owner, 'basic_percepts', None)
    if basic is None or bank.registry is None or not bank.token.shape[1]:
        return []
    c, has, what = readings(bank)
    iw = bank.identity_width
    c = c.detach().float()
    width, readings_per_page = c.shape[-1], c.shape[1]
    # MTM keys use Python's six-decimal rounding of each binary32 wire.
    # Only those keys and their occupancy are needed to link visits.
    keys = torch.cat((has[..., None], c), -1).tolist()
    visits, positions, prior_rows, predecessors = [], [], [], []
    last, depth = {}, {}
    for b, page in enumerate(keys):
        for k, (occupied, *wires) in enumerate(page):
            if not occupied:
                continue
            key = tuple(round(v, 6) for v in wires)
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
        return []

    registry = basic._place_cone_rows
    for key in last:
        if key not in registry:
            registry[key] = len(registry)
    storage = basic.place_cone_storage(len(registry))
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
    made = []
    if not chosen:
        return made
    nominated = points.index_select(0, seats.new_tensor(chosen))
    plus, minus, kept = nominated.split((width, width, basic.width), -1)
    for i, key in enumerate(definition_keys(plus, minus, kept)):
        has_part = bool(key[-1] and key[-1][0] < iw)
        row = admit(owner, plus[i], minus[i], kept[i] if has_part else None,
                    key=key if has_part else (*key[:3], ()))
        if row is not None:
            made.append(row)
    return made


def admit(owner, plus, minus, what, *, key=None):
    """A place's row of the first Pi, promoted, and its concept of order zero
    reading it; ``None`` where there is no room."""
    from Spaces import _concept_alloc_of
    basic = owner.basic_percepts
    try:
        owner._preflight_concept_row(0, context='place admission')
        row = basic.reserve_place(plus, minus, what, key=key)
    except (RuntimeError, OverflowError):
        return None
    basic.promote(row)
    alloc = _concept_alloc_of(owner)
    if any(percept == row for _, percept, _ in alloc.order_zero_members):
        return row
    concept = owner.new_concept(order=0)
    alloc.symbol_types.setdefault(concept, set()).add('place')
    inventory = owner._csw_concept_row(0, concept)
    if inventory is None:
        return row
    owner.bring_concept(row, inventory)
    return row
