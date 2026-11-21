"""The salience map: a request over concepts carried down onto the occurrences
of a sentence's percept bank, by the transposes of the matrices of the way up.

The way up is the first Pi (a basic percept's row over its run of parts and
its wholes), the Not and the Sigma of order zero, and above it the symbols'
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
  percept at that greatest takes the reading's whole share; the members'
  shares split it among equals, as the framework's backward does;
* a row's entries on the where wires pass the share onto them.

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
    members: torch.Tensor     # [keys, B, readings, 2] the share at each member of the first Pi


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
    the bank's order) onto its entries at each attended percept of
    ``perception`` (the read of a look's table).

    At an attended percept a row's for lane is its run's presence times the
    product of its wholes' for lanes, and its against lane the presence
    times the union of its wholes' against lanes and its against entries'
    presence, both times its entries on the where and the when wires
    (:meth:`BasicPercepts.read_bank`); a reading's is the greatest over its
    attended percepts. A reading's share reaches the attended percepts at
    that greatest, split among equals; there each member's share is the
    gradient of the row's lane by the member times the member's value.
    Returns the members' shares ``[keys, B, readings, 2]``, each attended
    percept's ``[B, A, 2]`` (each at the greatest taking the reading's
    whole share) and the share on its where wires ``[B, A]``."""
    index, token = perception.index, perception.reading
    filled = token >= 0
    own = bank.activity if activity is None else activity
    presence = own.to(bank.members).gather(1, index[..., None].expand(-1, -1, 2)) * filled[..., None]
    keys = bank.members.shape[0]
    members = bank.members.gather(2, index[None, ..., None].expand(keys, -1, -1, 2)) * presence[None]
    order = list(bank.keys)
    position = {key: i for i, key in enumerate(order)}
    rows_at, rows = perception.events, perception.values         # [R, B, A, 2], [R, B, K, 2]
    R, B, K = rows.shape[:3]
    slot = torch.where(filled, token, K)
    tiny = 1e-12
    shares_at = torch.zeros_like(members)                         # [keys, B, A, 2]
    events = members.new_zeros(B, index.shape[1], 2)
    band = members.new_zeros(B, index.shape[1])
    places = getattr(basic, 'place_rows', {}) if basic is not None else {}
    wires = None
    if basic is not None and R:
        factor = basic.place_factor(perception_table(bank, index, filled), bank.rows)   # [R, B, A]
        wires = basic.where_weight(bank.rows).detach().to(members)                    # [R]
    else:
        factor = members.new_ones(R, B, index.shape[1])

    def at_reading(x):
        """``[B, K]`` read at each attended percept's reading, ``[B, A]``."""
        return torch.nn.functional.pad(x, (0, 1)).gather(1, slot)

    def greatest(v_at, v):
        """Where an attended percept holds its reading's greatest ``v``:
        the share each takes (split among equals) and the mask."""
        at = ((v_at >= at_reading(v) - tiny) & (v_at > 0) & filled).to(v_at)
        count = at.new_zeros(B, K + 1).scatter_add(1, slot, at)[:, :K]
        return at / at_reading(count).clamp_min(1.), at

    def put(j, lane, local, split, whole):
        nonlocal shares_at, events
        shares_at[j, ..., lane] = shares_at[j, ..., lane] + split * local
        events[..., lane] = events[..., lane] + whole * local

    for r, (group, opposed) in enumerate(zip(bank.groups, bank.opposed)):
        if not group and bank.rows[r] not in places:
            continue
        d_for, d_against = row_grad[r, ..., 0].clamp_min(0.), row_grad[r, ..., 1].clamp_min(0.)
        value_for, value_against = rows[r, ..., 0], rows[r, ..., 1]
        # Through the Not: the share a row's reading takes is what the
        # concept reads there, for less against, and nothing where the two
        # are equal. Each lane's share is scaled to it.
        net = value_for - value_against
        d_for = torch.where(value_for > tiny, d_for * net.clamp_min(0.) / value_for.clamp_min(tiny), 0.)
        d_against = torch.where(value_against > tiny,
                                d_against * (-net).clamp_min(0.) / value_against.clamp_min(tiny), 0.)
        for_at, against_at = rows_at[r, ..., 0], rows_at[r, ..., 1]
        split_for, whole_for = greatest(for_at, value_for)
        split_against, whole_against = greatest(against_at, value_against)
        dF, dA = at_reading(d_for), at_reading(d_against)
        if wires is not None:
            band = band + whole_for * dF * for_at * wires[r]
        if bank.rows[r] in places:
            # A place reads its whats as wholes: each a factor of its for lane.
            for key in group:
                put(position[key], 0, dF * for_at, split_for, whole_for)
            continue
        run = position[group[0]]
        p = members[run, ..., 0]
        # The run: its value is the presence, which multiplies both lanes.
        shares_at[run, ..., 0] += split_for * dF * for_at + split_against * dA * against_at
        events[..., 0] += whole_for * dF * for_at + whole_against * dA * against_at
        # A whole is a factor of the for lane.
        for whole in group[1:]:
            put(position[whole], 0, dF * for_at, split_for, whole_for)
        # The against lane is the union of the wholes' against lanes and the
        # against entries' presence.
        sources = [(position[key], 1) for key in group[1:]] + [(position[key], 0) for key in opposed
                                                               if key in position]
        values = [members[j, ..., lane].clamp(0., 1.) for j, lane in sources]
        for n, (j, lane) in enumerate(sources):
            others = torch.ones_like(p)
            for m, value in enumerate(values):
                if m != n:
                    others = others * (1. - value)
            put(j, lane, dA * factor[r] * p * values[n] * others, split_against, whole_against)
    shares = shares_at.new_zeros(keys, B, K + 1, 2).scatter_add(2, slot[None, ..., None].expand_as(shares_at), shares_at)[:, :, :K]
    return shares, events, band


def perception_table(bank, index, filled):
    """The bank's muxed rows at a look's attended percepts ``[B, A, C, 2]``."""
    table = bank.table.gather(1, index[..., None, None].expand(-1, -1, *bank.table.shape[2:]))
    return table * filled[..., None, None]


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
    members, events, band = first_pi(bank, perception, row_grad, basic=basic)
    filled = perception.reading >= 0
    return SalienceMap(onto_bank(bank, perception.index, filled, events),
                       onto_bank(bank, perception.index, filled, band), row_grad, members)
