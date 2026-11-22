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


def meet(owner, bank):
    """Count the occurrences of a sentence's bank at their places and
    nominate a place's row on the count that nominates a form."""
    basic = getattr(owner, 'basic_percepts', None)
    if basic is None or bank.registry is None or not bank.token.shape[1]:
        return []
    c, has, what = readings(bank)
    iw = bank.identity_width
    cones = basic.place_cones
    made = []
    for b, k in has.nonzero().tolist():
        code = c[b, k].detach().float().cpu()
        key = tuple(round(v, 6) for v in code.tolist())
        seen = what[b, k].detach().float().cpu()
        held = cones.get(key)
        if held is None:
            cones[key] = (code, 1 - code, seen)
        else:
            plus, minus, kept = held
            kept = torch.cat((torch.minimum(kept[:iw], seen[:iw]), torch.maximum(kept[iw:], seen[iw:])))
            cones[key] = (torch.minimum(plus, code), torch.minimum(minus, 1 - code), kept)
        if not owner.nominates(('place', key), owner.reading(b)):
            continue
        plus, minus, kept = cones[key]
        row = admit(owner, plus, minus, kept if bool(kept[:iw].any()) else None)
        if row is not None:
            made.append(row)
    return made


def admit(owner, plus, minus, what):
    """A place's row of the first Pi, promoted, and its concept of order zero
    reading it; ``None`` where there is no room."""
    from Spaces import _concept_alloc_of
    basic = owner.basic_percepts
    try:
        owner._preflight_concept_row(0, context='place admission')
        row = basic.reserve_place(plus, minus, what)
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
