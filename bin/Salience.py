"""The salience map: a request over concepts carried down onto the events of
a sentence's percept bank, by the transposes of the matrices of the way up.

The way up is the first Pi (a basic percept's row over its run of parts and
its wholes), the Not and the Sigma of order zero, and above it the symbols'
Pi and the kinds' Sigma (:func:`ConceptSpread.structure`). The way down reads
the same entries:

* a Sigma is passed by its weights, before its squash, so a concept already
  lit still returns its percepts; a Not entry passes the share to the other
  lane, so a request may make salient what tells for a concept or what
  tells against it;
* a Pi is passed by the derivative of its product at the values of the open
  run (the local derivatives kept from the run that read the whole
  sentence): a member's share is the row's share times the row's value;
* at the bank, each member is the greatest over its events, and each event
  at that greatest takes the member's whole share.

What lands on an event is its share, the gradient of the asked concepts by
the event's activity times that activity. It is never below zero. Nothing
here is a parameter of its own, and nothing is read on the host.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SalienceMap:
    events: torch.Tensor      # [B, events, 2] each event's share, on its two lanes
    band: torch.Tensor        # [B, events] the share on each event's band of where wires
    rows: torch.Tensor        # [R, B, readings, 2] the share at each basic percept's row
    members: torch.Tensor     # [keys, B, readings, 2] the share at each member of the first Pi


def _down(graph, request, values, readings, depth=8):
    """The gradient of the asked concepts at every node of the structure
    above the basic percepts' members, ``[N, B, readings, 2]``. ``request``
    ``[V, B, 2]``; ``values`` ``[N, B, readings, 2]`` the open run's values
    (read at the symbols, whose Pi is passed by its derivative). ``depth``
    steps down, enough for the structure's orders; there is no test on the
    host for an earlier end."""
    N = graph.size
    B = request.shape[1]
    grad = request.new_zeros(N, B, readings, 2)
    grad[:graph.concepts] = request[:graph.concepts, :, None, :].clamp_min(0.)
    keep = graph.child < graph.part_start
    parent, child = graph.parent[keep], graph.child[keep]
    weight, negated = graph.weight[keep].detach().to(request), graph.negated[keep]
    symbol = ((parent >= graph.concepts) & (parent < graph.percept_start))[:, None, None, None]
    tiny = torch.finfo(request.dtype).tiny
    source = grad.clone()
    for _ in range(depth):
        lanes = grad.index_select(0, parent)
        lanes = torch.where(negated[:, None, None, None], lanes.flip(-1), lanes)
        # A Pi: d child = d row * w * row / child, at the open run's values.
        by_product = (lanes * values.index_select(0, parent)
                      / values.index_select(0, child).clamp_min(tiny))
        lanes = torch.where(symbol, by_product, lanes) * weight[:, None, None, None]
        grad = source.index_add(0, child, lanes)
    return grad


def first_pi(bank, perception, row_grad, activity=None):
    """The first Pi passed down: each basic percept's row share ``row_grad``
    ``[R, B, readings, 2]`` (a gradient on the row's two lanes, the rows in
    the bank's order) onto its members and their events.

    A row's for lane is its run's presence times the product of its wholes'
    for lanes; its against lane is the presence times the union of its
    wholes' against lanes and its against entries' presence (as the way up,
    :meth:`BasicPercepts._row_values`). Each member's share is the gradient
    of the row's lanes by the member times the member's value. Returns the
    members' shares ``[keys, B, readings, 2]`` and the events' ``[B, events,
    2]``."""
    native = bank.native if activity is None else bank.at(activity)
    order = list(dict.fromkeys(bank.keys))
    position = {key: i for i, key in enumerate(order)}
    selected = native.selected(order)
    members = native.values.index_select(0, selected)          # [keys, B, K, 2]
    happened = native.events.index_select(0, selected)         # [keys, B, K, P, 2]
    rows = perception.values                                   # [R, B, K, 2]
    shares = torch.zeros_like(members)
    events = members.new_zeros(happened.shape[1], happened.shape[3], 2)
    tiny = 1e-12

    def place(share, lit, lane):
        """``share`` ``[B, K]`` to each event at the greatest of ``lit``
        ``[B, K, P]`` on ``lane``."""
        nonlocal events
        best = lit.amax(-1, keepdim=True)
        at = ((lit >= best - tiny) & (lit > 0)).to(share)
        events = events.index_add(2, torch.tensor([lane], device=events.device),
                                   (at * share[..., None]).sum(1)[..., None])

    for index, (group, opposed) in enumerate(zip(bank.groups, bank.opposed)):
        if not group:
            continue
        d_for, d_against = row_grad[index, ..., 0].clamp_min(0.), row_grad[index, ..., 1].clamp_min(0.)
        value_for, value_against = rows[index, ..., 0], rows[index, ..., 1]
        run = position[group[0]]
        presence = members[run, ..., 0]
        at_run = happened[run, ..., 0]                         # [B, K, P]
        # The run: its value is the presence, which multiplies both lanes.
        share = d_for * value_for + d_against * value_against
        shares[run, ..., 0] = shares[run, ..., 0] + share
        place(share, at_run, 0)
        # A whole is a factor of the for lane, read where the run lies.
        for whole in group[1:]:
            j = position[whole]
            share = d_for * value_for
            shares[j, ..., 0] = shares[j, ..., 0] + share
            place(share, at_run * happened[j, ..., 0], 0)
        # The against lane is the union of the wholes' against lanes and the
        # against entries' presence, each read where the run lies.
        sources = [(position[key], 1) for key in group[1:]] + [(position[key], 0) for key in opposed
                                                               if key in position]
        lit = [at_run * happened[j, ..., lane] for j, lane in sources]
        against = [value.amax(-1) if value.shape[-1] else members[j, ..., lane]
                   for value, (j, lane) in zip(lit, sources)]
        for n, (j, lane) in enumerate(sources):
            others = torch.ones_like(presence)
            for m, value in enumerate(against):
                if m != n:
                    others = others * (1. - value.clamp(0., 1.))
            share = d_against * presence * against[n] * others
            shares[j, ..., lane] = shares[j, ..., lane] + share
            place(share, lit[n], lane)
    return shares, events


def salience(owner, bank, request, *, perception=None, field=None):
    """The salience map of ``request`` ``[V, B, 2]`` (each concept's two
    lanes: asked for, asked against) over ``bank`` at the open run.
    ``perception`` is the open run's read of the bank and ``field`` its
    concepts' values ``[V, B, readings, 2]`` (read where not given)."""
    import ConceptSpread
    basic = owner.basic_percepts
    perception = basic.read_bank(bank) if perception is None else perception
    readings = perception.values.shape[2]
    device = perception.values.device
    graph = ConceptSpread.structure(owner, device)
    values = request.new_zeros(graph.size, request.shape[1], readings, 2)
    if field is not None:
        width = min(len(field), graph.concepts)
        values[:width] = field[:width].detach()
    # The symbols' values at the open run: the product of what they hold.
    held = (graph.parent >= graph.concepts) & (graph.parent < graph.percept_start)
    logs = torch.log(values.index_select(0, graph.child[held]).clamp_min(1e-30))
    product = torch.exp(values.new_zeros(graph.size, *values.shape[1:]).index_add(0, graph.parent[held], logs))
    is_symbol = torch.zeros(graph.size, dtype=torch.bool, device=device)
    is_symbol[graph.concepts:graph.percept_start] = True
    values = torch.where(is_symbol[:, None, None, None], product, values)
    grad = _down(graph, request.to(values), values, readings, 2 * (len(owner._order_caps()) + 1))
    rows = torch.tensor(bank.rows, dtype=torch.long, device=device)
    row_grad = grad.index_select(0, graph.percept_start + rows) if len(bank.rows) else grad[:0]
    members, events = first_pi(bank, perception, row_grad)
    # No concept reads a where wire; a place's row of the first Pi does, and
    # is passed by its entries onto the band of its reading's start.
    import Places
    band, kept = Places.down(basic, bank, perception, row_grad)
    events = events + torch.stack((kept, torch.zeros_like(kept)), -1).to(events)
    return SalienceMap(events, band.to(events), row_grad, members)
