"""Places: rows of the first Pi with entries on the where wires and no
what, nominated by the cone over the occurrences met at a place.

An occurrence is a reading of a sentence's bank: the where wires of its
bracket in the bank's table (each wire on two lanes, ``c`` and ``1 - c``)
and the members of the percept tower present in it (a member present at a
half or more). The occurrences met at one band meet under one key of MTM's
count, ``('place', band)``, the count that nominates a form. Their cone
(rule 72) is the least of their coordinates: weight on what they share,
none on what varies. A place met by many forms shares no member and keeps
none; a place met by one form keeps that form's members.

A place's row is read as any row of the first Pi: at each attended percept,
the product of its entries there, on the what it keeps and on the where
wires (:meth:`BasicPercepts.BasicPercepts.read_bank`). It brings a concept
of order zero, as any row does (rule 21).
"""
import torch


def readings(bank):
    """The where wires of each reading's bracket ``[B, readings, width]``,
    whether the reading has one, and each member's greatest presence in
    each reading ``[keys, B, readings]``."""
    K = bank.where.shape[1]
    span = bank.span
    same = ((span[:, None, :, 0] == bank.where[..., :1]) & (span[:, None, :, 1] == bank.where[..., 1:])
            & (span[:, None, :, 1] > span[:, None, :, 0]))                       # [B, K, O]
    has = same.any(-1)
    at = same.int().argmax(-1)
    from BasicPercepts import WHERE_WIRES
    columns = bank.table[..., bank.width:bank.width + WHERE_WIRES, 0]           # [B, O, w]
    wires = columns.gather(1, at[..., None].expand(-1, -1, columns.shape[-1]))
    slot = torch.where(bank.token >= 0, bank.token, K)
    lit = bank.members[..., 0]
    present = lit.new_zeros(lit.shape[0], lit.shape[1], K + 1).scatter_reduce(
        2, slot[None].expand_as(lit), lit, 'amax')[..., :K]
    return wires, has, present


def _whats(present, b, k, order):
    return {key for i, key in enumerate(order) if present[i][b][k]}


def meet(owner, bank):
    """Count the occurrences of a sentence's bank at their places and
    nominate a place's row on the count that nominates a form."""
    basic = getattr(owner, 'basic_percepts', None)
    if basic is None or bank.registry is None or not bank.token.shape[1]:
        return []
    c, has, lit = readings(bank)
    order = list(bank.keys)
    present = (lit >= .5).cpu().tolist() if order else []
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
    owner.bring_concept(row, inventory)
    return row
