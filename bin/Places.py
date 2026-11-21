"""Places: rows of the first Pi with entries on the where wires of their
reading, nominated by the cone over the occurrences met at a place.

An occurrence is a reading of a sentence's bank: the band of the event at
its start (the where wires, each on two lanes, ``c`` and ``1 - c``) and the
members of the percept tower present in it (a member present at a half or
more). The occurrences met at one band meet under one key of MTM's count,
``('place', band)``, the count that nominates a form. Their cone (rule 72)
is the least of their coordinates: weight on what they share, none on what
varies. A place met by many forms shares no member and keeps none: its row
has entries on the where wires alone. A place met by one form keeps that
form's members.

The row reads, on the way up, ``b = exp(sum log m + E+ . log c + E- .
log(1 - c))`` at the event at its reading's start, present to that event's
degree, on its for lane; its against lane is nothing. On the way down a
place's row is passed by its entries: the share on the band of the event at
a reading's start is ``d . b . (E+ c + E- (1 - c))``, and each member it
keeps takes ``d . b`` as a whole of a content row does. A place brings a
concept of order zero, as any row does (rule 21); no concept has a weight on
a where wire.
"""
import torch

#: The lanes of the where wires as keys of the first Pi.
LANES = ('for', 'against')


def where_keys(width):
    return [('where', i, lane) for lane in LANES for i in range(width)]


def starts(bank, readings):
    """The event at each reading's start, ``[B, readings]`` (``-1`` none)."""
    token, place = bank.token, bank.place
    far = torch.iinfo(place.dtype).max
    slots = torch.arange(readings, device=token.device)
    held = token[:, None, :] == slots[None, :, None]                       # [B, K, P]
    at = torch.where(held, place[:, None, :], far)
    first = at.amin(-1, keepdim=True)
    start = held & (at == first)
    events = torch.arange(token.shape[1], device=token.device)
    index = torch.where(start, events[None, None], token.shape[1]).amin(-1)
    return torch.where(held.any(-1), index, -1)


def wires(bank, start):
    """The where wires of each reading's start, ``[B, readings, width]``, and
    whether the reading has one."""
    has = start >= 0
    band = bank.band.gather(1, start.clamp_min(0)[..., None].expand(-1, -1, bank.band.shape[-1]))
    return band, has


def value(basic, bank, native, presence, rows):
    """Each place row of ``rows`` on the way up: ``{index: (lanes [B, K, 2],
    events [B, K, P, 2])}`` and the reading's start events."""
    found = [(i, int(row)) for i, row in enumerate(rows) if int(row) in basic.place_rows]
    if not found or bank.band is None:
        return {}, None
    K = native.values.shape[2]
    start = starts(bank, K)
    c, has = wires(bank, start)
    c = c.to(native.values).clamp(1e-6, 1 - 1e-6)
    lit = presence.gather(1, start.clamp_min(0)) * has.to(presence)          # [B, K]
    order = list(dict.fromkeys(bank.keys))
    position = {key: i for i, key in enumerate(order)}
    selected = native.selected(order) if order else None
    out = {}
    P = bank.token.shape[1]
    at_start = (torch.arange(P, device=start.device)[None, None] == start[..., None]).to(c)   # [B, K, P]
    for index, row in found:
        plus, minus, whats = basic.place_rows[row]
        logs = c.log() @ plus.to(c) + (1 - c).log() @ minus.to(c)
        for key in whats:
            j = position.get(key)
            if j is None:
                logs = logs + torch.log(torch.full_like(logs, 1e-30))
                continue
            m = native.values.index_select(0, selected[j:j + 1])[0, ..., 0]
            logs = logs + m.clamp_min(1e-30).log()
        b = lit * logs.exp()
        lanes = torch.stack((b, torch.zeros_like(b)), -1)
        out[index] = (lanes, torch.stack((b[..., None] * at_start, torch.zeros_like(at_start)), -1))
    return out, start


def down(basic, bank, perception, row_grad, activity=None):
    """The places passed down: the share on the band of each event ``[B,
    events]``, and the shares of the members places keep, each event's
    ``[B, events]`` on its for lane."""
    band = None
    rows = bank.rows
    places = [(i, int(row)) for i, row in enumerate(rows) if int(row) in basic.place_rows]
    events = bank.token.new_zeros(bank.token.shape, dtype=perception.values.dtype)
    if not places or bank.band is None:
        return events, events.clone()
    native = bank.native if activity is None else bank.at(activity)
    K = perception.values.shape[2]
    start = starts(bank, K)
    c, has = wires(bank, start)
    c = c.to(events).clamp(1e-6, 1 - 1e-6)
    band = events.clone()
    whats = events.clone()
    order = list(dict.fromkeys(bank.keys))
    position = {key: i for i, key in enumerate(order)}
    for index, row in places:
        plus, minus, kept = basic.place_rows[row]
        d = row_grad[index, ..., 0].clamp_min(0.)                            # [B, K]
        gate = d * perception.values[index, ..., 0]
        share = gate * (c * plus.to(c) + (1 - c) * minus.to(c)).sum(-1) * has.to(c)
        band = band.scatter_add(1, start.clamp_min(0), torch.where(has, share, 0.))
        for key in kept:
            j = position.get(key)
            if j is None:
                continue
            lit = native.events.index_select(0, native.selected([key]))[0, ..., 0]   # [B, K, P]
            best = lit.amax(-1, keepdim=True)
            at = ((lit >= best - 1e-12) & (lit > 0)).to(gate)
            whats = whats + (at * gate[..., None]).sum(1)
    return band, whats


def _whats(present, b, k, order):
    return {key for i, key in enumerate(order) if present[i][b][k]}


def meet(owner, bank):
    """Count the occurrences of a sentence's bank at their places and
    nominate a place's row on the count that nominates a form."""
    basic = getattr(owner, 'basic_percepts', None)
    if basic is None or bank.band is None or bank.token is None:
        return []
    K = bank.native.values.shape[2]
    start = starts(bank, K)
    c, has = wires(bank, start)
    order = list(dict.fromkeys(bank.keys))
    selected = bank.native.selected(order) if order else None
    present = ((bank.native.values.index_select(0, selected)[..., 0] >= .5).cpu().tolist()
               if order else [])
    cones = basic.place_cones
    made = []
    for b, k in has.nonzero().tolist():
        code = c[b, k].detach().float().cpu()
        key = tuple(round(v, 6) for v in code.tolist())
        whats = _whats(present, b, k, order)
        held = cones.get(key)
        if held is None:
            cones[key] = (code, 1 - code, whats)
        else:
            plus, minus, kept = held
            cones[key] = (torch.minimum(plus, code), torch.minimum(minus, 1 - code), kept & whats)
        if not owner.nominates(('place', key), owner.reading(b)):
            continue
        plus, minus, kept = cones[key]
        row = admit(owner, plus, minus, tuple(sorted(kept, key=repr)))
        if row is not None:
            made.append(row)
    return made


def admit(owner, plus, minus, whats):
    """A place's row of the first Pi, promoted, and its concept of order zero
    reading it; ``None`` where there is no room."""
    from Spaces import _concept_alloc_of
    basic = owner.basic_percepts
    try:
        owner._preflight_concept_row(0, context='place admission')
        row = basic.reserve_place(plus, minus, whats)
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
    owner.add_concept_feature(inventory, 'bp', row, 1.)
    return row
