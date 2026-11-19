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
    from the basic percepts' codes and the concept's column of the Sigma:
    ``tanh((x_for - x_against) . u)``, ``x = 1 - prod_p (1 - v_p)^{b_p}`` on
    each lane. The place and time weights of a column start at zero, so
    only the content is read. Returns the signed value and its two lanes."""
    basic = owner.basic_percepts
    sigma = owner.order_zero
    u = sigma.weight[:sigma.content_width, int(row)].detach().cpu().double()
    lanes = []
    for lane in (0, 1):
        gap = torch.ones(sigma.content_width, dtype=torch.float64)
        for percept, values in presence.items():
            code = (basic.codes[percept] if bool(basic.promoted[percept])
                    else basic.code(percept)).detach().cpu().double()
            gap = gap * (1. - code).clamp_min(1e-6) ** float(values[lane])
        lanes.append(1. - gap)
    y = torch.tanh(((lanes[0] - lanes[1]) * u.cpu()).sum())
    return float(y), torch.tensor([max(float(y), 0.), max(-float(y), 0.)])
