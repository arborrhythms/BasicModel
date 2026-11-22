"""The salience map: a request over concepts carried down onto the occurrences
of a sentence's percept bank, by the transposes of the matrices of the way up.

The way up is the first Pi (a basic percept's row read from the what, where
and when columns of each attended percept of a look's table), the Not and the Sigma of order zero, and above it the symbols'
Pi and the kinds' Sigma (:func:`ConceptSpread.structure`). The way down reads
the same entries:

* a Sigma is passed by its weights, before its squash, so a concept already
  lit still returns its percepts; a Not entry passes the share to the other
  lane, so a request may make salient what tells for a concept or what
  tells against it;
* the Not makes a row one value, for less against: a row's reading takes
  a share only where it tells one way more than the other;
* a Pi is passed by the derivative of its product at the values of the open
  run (the local derivatives kept from the run that read the whole
  sentence): a member's share is the row's share times the row's value;
* a reading is the greatest over its attended percepts, and each attended
  percept at that greatest takes the reading's whole share, on its what
  columns (by its presence) and on its where wires (by the row's weights
  there).

What lands on an occurrence of the bank is its share, the gradient of the
asked concepts by its activity times that activity. It is never below zero. Nothing
here is a parameter of its own, and nothing is read on the host.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SalienceMap:
    events: torch.Tensor      # [B, occurrences, 2] each occurrence's share, on its two lanes
    band: torch.Tensor        # [B, occurrences] the share on each occurrence's where wires
    rows: torch.Tensor        # [R, B, readings, 2] the share at each basic percept's row


def _down(graph, request, values, readings, depth=8, nodes=None):
    """The gradient of the asked concepts at every node of the structure
    above the basic percepts' members, ``[N, B, readings, 2]``. ``request``
    ``[V, B, 2]``; ``values`` ``[N, B, readings, 2]`` the open run's values
    (read at the symbols, whose Pi is passed by its derivative). ``depth``
    steps down, enough for the structure's orders; there is no test on the
    host for an earlier end. ``nodes`` (sorted) are the structure's nodes
    that the node axis holds, all of them where not given; a node on no
    edge passes nothing down."""
    B = request.shape[1]
    keep = graph.child < graph.part_start
    parent, child = graph.parent[keep], graph.child[keep]
    weight, negated = graph.weight[keep].detach().to(request), graph.negated[keep]
    symbol = ((parent >= graph.concepts) & (parent < graph.percept_start))[:, None, None, None]
    # Never below zero; at exactly zero the derivative passes, so that a
    # request that starts at nothing can be taught (as signed_lanes).
    asked = torch.where(request >= 0, request, 0.)[:, :, None, :].expand(-1, -1, readings, -1)
    if nodes is None:
        N = graph.size
        grad = request.new_zeros(N, B, readings, 2)
        grad[:graph.concepts] = asked[:graph.concepts]
    else:
        N = len(nodes)
        concept = (nodes < graph.concepts)[:, None, None, None]
        grad = torch.where(concept, asked.index_select(0, nodes.clamp_max(graph.concepts - 1)), 0.)
        parent, child = torch.searchsorted(nodes, parent), torch.searchsorted(nodes, child)
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


def first_pi(bank, perception, row_grad, activity=None, basic=None):
    """The first Pi passed down: each basic percept's row share ``row_grad``
    ``[R, B, readings, 2]`` (a gradient on the row's two lanes, the rows in
    the bank's order) onto the attended percepts of ``perception`` (the read
    of a look's table).

    At an attended percept a row's two lanes are the percept's presence
    times what its what columns give (for, or against) times the row's
    entries on the where and the when wires (:meth:`BasicPercepts.read_bank`);
    a reading's is the greatest over its attended percepts. A reading's share
    reaches the attended percepts at that greatest, each taking the whole of
    it; there the share on the what columns is the gradient of the row's lane
    by the percept's presence times that presence, which is the row's lane
    itself, and the share on the where wires is that times the row's weights
    on them (the derivative of the factor by each wire's two lanes, times
    the lanes). A conjunction passes nothing further down. Returns each
    attended percept's share ``[B, A, 2]`` (on the for lane; nothing reads a
    percept's against lane) and the share on its where wires ``[B, A]``."""
    token = perception.reading
    filled = token >= 0
    rows_at, rows = perception.events, perception.values         # [R, B, A, 2], [R, B, K, 2]
    R, B, K = rows.shape[:3]
    A = token.shape[1]
    events = rows_at.new_zeros(B, A, 2)
    band = rows_at.new_zeros(B, A)
    if not R:
        return events, band
    slot = torch.where(filled, token, K)
    tiny = 1e-12
    conjunctions = getattr(basic, 'conjunctions', {}) if basic is not None else {}
    reads = torch.tensor([row not in conjunctions for row in bank.rows], device=rows.device)[:, None, None]
    wires = (basic.where_weight(bank.rows).detach().to(rows)[:, None, None] if basic is not None
             else rows.new_zeros(R, 1, 1))

    def at_reading(x):
        """``[R, B, K]`` read at each attended percept's reading, ``[R, B, A]``."""
        return torch.nn.functional.pad(x, (0, 1)).gather(2, slot[None].expand(x.shape[0], -1, -1))

    def greatest(v_at, v):
        """Where an attended percept holds its reading's greatest ``v``."""
        return ((v_at >= at_reading(v) - tiny) & (v_at > 0) & filled).to(v_at)

    # Never below zero; at exactly zero the derivative passes (as in
    # ``_down``), so that the share on the where wires teaches a request
    # that starts at nothing.
    d_for = torch.where(row_grad[..., 0] >= 0, row_grad[..., 0], 0.)
    d_against = torch.where(row_grad[..., 1] >= 0, row_grad[..., 1], 0.)
    value_for, value_against = rows[..., 0], rows[..., 1]
    # Through the Not: the share a row's reading takes is what the concept
    # reads there, for less against, and nothing where the two are equal.
    # Each lane's share is scaled to it.
    # The Not's two lanes are those of a signed value: at a row read alike
    # both ways the derivative reaches both (as ``signed_lanes``).
    from Layers import signed_lanes
    net = signed_lanes(value_for - value_against)
    d_for = torch.where(value_for > tiny, d_for * net[..., 0] / value_for.clamp_min(tiny), 0.)
    d_against = torch.where(value_against > tiny, d_against * net[..., 1] / value_against.clamp_min(tiny), 0.)
    for_at, against_at = rows_at[..., 0], rows_at[..., 1]
    whole_for = greatest(for_at, value_for) * reads
    whole_against = greatest(against_at, value_against) * reads
    dF, dA = at_reading(d_for), at_reading(d_against)
    events[..., 0] = (whole_for * dF * for_at + whole_against * dA * against_at).sum(0)
    band = (whole_for * dF * for_at * wires).sum(0)
    return events, band


def onto_bank(bank, index, filled, values):
    """Shares at a look's attended percepts ``[B, A, ...]`` onto the bank's
    occurrences ``[B, occurrences, ...]``."""
    values = values * filled.reshape(*filled.shape, *([1] * (values.ndim - 2))).to(values)
    out = values.new_zeros(bank.token.shape[0], bank.token.shape[1], *values.shape[2:])
    return out.scatter_add(1, index.reshape(*index.shape, *([1] * (values.ndim - 2))).expand_as(values), values)


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
    rows = torch.tensor(bank.rows, dtype=torch.long, device=device)
    # The node axis holds the nodes on an edge above the first Pi's members
    # and the bank's rows: a concept on no edge passes nothing down to a
    # basic percept.
    above = graph.child < graph.part_start
    nodes = torch.unique(torch.cat((graph.parent[above], graph.child[above], graph.percept_start + rows)))
    values = request.new_zeros(len(nodes), request.shape[1], readings, 2)
    width = 0 if field is None else min(len(field), graph.concepts)
    if width:
        values = torch.where((nodes < width)[:, None, None, None],
                             field.detach().to(values).index_select(0, nodes.clamp_max(width - 1)), values)
    # The symbols' values at the open run: the product of what they hold.
    held = (graph.parent >= graph.concepts) & (graph.parent < graph.percept_start)
    parent, child = torch.searchsorted(nodes, graph.parent[held]), torch.searchsorted(nodes, graph.child[held])
    logs = torch.log(values.index_select(0, child).clamp_min(1e-30))
    product = torch.exp(torch.zeros_like(values).index_add(0, parent, logs))
    is_symbol = (nodes >= graph.concepts) & (nodes < graph.percept_start)
    values = torch.where(is_symbol[:, None, None, None], product, values)
    grad = _down(graph, request.to(values), values, readings, 2 * (len(owner._order_caps()) + 1), nodes=nodes)
    row_grad = (grad.index_select(0, torch.searchsorted(nodes, graph.percept_start + rows))
                if len(bank.rows) else grad[:0])
    events, band = first_pi(bank, perception, row_grad, basic=basic)
    filled = perception.reading >= 0
    return SalienceMap(onto_bank(bank, perception.index, filled, events),
                       onto_bank(bank, perception.index, filled, band), row_grad)
