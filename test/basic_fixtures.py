"""Explicit basic-level occurrences for isolated native-literal fixtures."""
import torch


def feature(owner, row, tower, value, weight=0., *, negated=False, parts=None):
    """Declare the fixture occurrences as nominated rows, then exercise the
    native selector writer."""
    basic = owner.basic_percepts
    if tower == 'ps':
        group = tuple(value) if isinstance(value, (tuple, list)) else (int(value),)
        basic.promote(basic.reserve(group, ()))
    elif tower == 'ws':
        if parts is None:
            raise ValueError('a property fixture must declare its identity parts')
        groups = tuple((part,) if isinstance(part, int) else tuple(part) for part in parts)
        for group in groups:
            basic.promote(basic.reserve(group, (int(value),)))
    return owner.add_concept_feature(row, tower, value, weight, negated=negated)


def published_presence(owner, batch=0, reading=0):
    """``{basic percept: (for, against)}`` of the last published read."""
    field = owner._percept_field
    values = field.values[:, batch, reading].detach()
    return {key[1]: tuple(map(float, values[index])) for index, key in enumerate(field.keys)}


def order_zero_expected(owner, row, presence):
    """The order-zero concept ``row`` read from ``presence``, computed here
    from the concept's entries: the Not makes each basic percept one signed
    value, ``s = for - against``; the Sigma reads ``tanh(sum w s)`` over its
    entries, a Not entry reading the row exchanged, ``-s``. A percept it has
    no entry on is not read. Returns the signed value and its two lanes."""
    total = 0.
    for percept, weight, negated in owner.order_zero.entries(int(row)):
        sign = -1. if negated else 1.
        values = presence.get(percept)
        if values is not None:
            total += weight * sign * max(-1., min(1., float(values[0]) - float(values[1])))
    y = float(torch.tanh(torch.tensor(total, dtype=torch.float64)))
    return y, torch.tensor([max(y, 0.), max(-y, 0.)])


def with_property_half(cs, width=8):
    """Give the fixture's basic percepts a property half (a whole's code is
    an upper bound that cuts a fixed pattern of coordinates from all ones,
    and is read on that cut, rule 54), so that two percepts with one run of
    parts and different wholes have different vectors and a concept over a
    property and its Not does not cancel. Done at construction, before any
    row."""
    from types import SimpleNamespace
    from BasicPercepts import BasicPercepts
    old = cs.basic_percepts
    basic = BasicPercepts(old.native, old.identity_width, width, old.capacity)
    patterns = torch.zeros(4, width)
    for whole in range(4):
        patterns[whole, whole::4] = 1.

    def lookup_rows(ids):
        ids = torch.as_tensor(ids, dtype=torch.long)
        return torch.cat((torch.zeros(*ids.shape, basic.identity_width), 1 - patterns[ids % 4]), -1)
    basic.bind(SimpleNamespace(lookup_rows=lookup_rows))
    cs._fixture_basic_percepts = basic
    object.__setattr__(cs, '_basic_percepts', basic)
    return cs


def with_own_codes(cs, parts, width=8):
    """Give each of the fixture's ``parts`` a code of its own (a distinct
    set of coordinates of the parts half, one each where they fit, else the
    fewest that tell them apart; every other part's code nothing) and the
    basic percepts a property half (:func:`with_property_half`), so that
    the table's what columns tell the parts and the wholes apart. Done at
    construction, before any row."""
    from itertools import combinations
    with_property_half(cs, width)
    basis = cs.basic_percepts.native._basis
    parts = sorted(dict.fromkeys(map(int, parts)))
    coordinates = cs.basic_percepts.identity_width
    for ones in range(1, coordinates + 1):
        sets = list(combinations(range(coordinates), ones))
        if len(sets) >= len(parts):
            break
    else:
        raise ValueError('more parts than codes of the parts half')
    with torch.no_grad():
        basis.W.zero_()
        for part, chosen in zip(parts, sets):
            basis.W[part, list(chosen)] = 1.
    return cs
