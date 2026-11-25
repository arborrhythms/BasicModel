"""The salience map: a request over concepts carried down onto every
occurrence of a sentence's percept bank by the ways back of the layers of
the way up (rule 79, §19.121).

The way up is the first Pi (a basic percept's row read from the ``.what``,
``.where`` and ``.when`` of each attended percept of a look's table), the
Not and the Sigma of order zero, and above it the symbols' Pi and the kinds'
Sigma (:func:`ConceptSpread.structure`). The way down:

* above order zero the kinds' Sigma has no way back yet, so a request there
  reaches the order-zero concepts by the transposes of the structure above
  them (:func:`_down`): a Sigma passed by its weights, a Not entry passing
  the share to the other lane, a symbol by its weights (by the derivative of
  its product where the open run's values are given);
* the order-zero Sigma is passed by its way back (:meth:`Layers.SparseLDULayer.back`),
  each lane of the concepts' request on its own: what each row of the first
  Pi is asked, a lane's result below nothing asking the row's other lane;
* the first Pi's way back is the percept a row keeps: a row's salience at an
  occurrence is what it is asked times its own read of that occurrence,
  taken back through the reading (the greatest of its occurrences) and the
  Not (for less against)
  (:func:`BasicPercepts.held_at` on the ``.what``,
  :meth:`BasicPercepts.BasicPercepts.place_factor` on the ``.where`` and the
  ``.when``), on each lane, over every occurrence of the sentence's field.

What lands on an occurrence is the sum over the rows, never below zero.
Nothing here is a parameter of its own, and nothing is read on the host.
"""
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class SalienceMap:
    events: torch.Tensor      # [B, occurrences, 2] each occurrence's salience, on its two lanes
    rows: torch.Tensor        # [R, B, 2] what each basic percept's row is asked


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
    parent, child = torch.where(keep, graph.parent, 0), torch.where(keep, graph.child, 0)
    weight = torch.where(keep, graph.weight, 0.).detach().to(request)
    negated = graph.negated & keep
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
    # Each edge's factor is the same at every step: its weight and, at a Pi,
    # d child = d row * row / child at the open run's values. It reads no
    # request, so it is made once.
    with torch.no_grad():
        ratio = values.index_select(0, parent) / values.index_select(0, child).clamp_min(tiny)
        factor = torch.where(symbol, ratio, 1.) * weight[:, None, None, None]
    return _Walk.apply(grad, factor, parent, child, negated, depth)


def _step(grad, factor, parent, child, negated, source):
    """``source`` plus each edge's parent share, carried to its child."""
    lanes = grad.index_select(0, parent)
    lanes = torch.where(negated[:, None, None, None], lanes.flip(-1), lanes)
    return source.index_add(0, child, lanes * factor)


class _Walk(torch.autograd.Function):
    """The way down, ``depth`` steps of :func:`_step` from ``source``. It is
    linear in ``source``, so its backward is the same walk on the transposed
    edges, and the walk keeps for the backward only the edges' factor (no
    step keeps a share of its own)."""

    @staticmethod
    def forward(ctx, source, factor, parent, child, negated, depth):
        ctx.save_for_backward(factor, parent, child, negated)
        ctx.depth = depth
        grad = source
        for _ in range(depth):
            grad = _step(grad, factor, parent, child, negated, source)
        return grad

    @staticmethod
    def backward(ctx, out):
        factor, parent, child, negated = ctx.saved_tensors
        # The transpose of an edge: the child's share, by the factor, flipped
        # where the edge is a Not entry, onto the parent.
        back = lambda h: out.index_add(0, parent, torch.where(
            negated[:, None, None, None], (h.index_select(0, child) * factor).flip(-1),
            h.index_select(0, child) * factor))
        h = out
        for _ in range(ctx.depth):
            h = back(h)
        return h, None, None, None, None, None


def concepts_asked(owner, request, field=None):
    from CandidateGrowth import read_streams
    return read_streams(owner, lambda: _concepts_asked(owner, request, field), axis=1)


def _concepts_asked(owner, request, field=None):
    """Each concept's request with what reaches it from above order zero,
    ``[V, B, 2]``: :func:`_down` on the structure without the order-zero
    Sigma's and the first Pi's edges. ``field`` ``[V, B, readings, 2]``, the
    open run's values, passes a symbol by the derivative of its product
    there, read at the reading where it is greatest; where it is not given a
    symbol is passed by its weights."""
    import ConceptSpread
    device = request.device
    graph = ConceptSpread.structure(owner, device)
    above = (graph.child < graph.percept_start) & (graph.parent < graph.percept_start)
    graph = replace(graph, parent=torch.where(above, graph.parent, 0),
                    child=torch.where(above, graph.child, 0),
                    weight=torch.where(above, graph.weight, 0.), negated=graph.negated & above)
    V = owner.nVectors
    if not len(graph.parent):
        return torch.where(request >= 0, request, 0.)[:V]
    # Every concept already had a seat in the compact graph. Symbols have
    # construction-time seats too; use those addresses directly, without a
    # value-dependent unique() result or a remapping after each look.
    nodes = torch.arange(graph.percept_start, device=device)
    readings = 1 if field is None else field.shape[2]
    values = request.new_ones(len(nodes), request.shape[1], readings, 2)
    if field is not None:
        width = min(len(field), graph.concepts)
        values = torch.where((nodes < width)[:, None, None, None],
                             field.detach().to(values).index_select(0, nodes.clamp_max(width - 1)), values)
        held = (graph.parent >= graph.concepts) & (graph.parent < graph.percept_start)
        parent = torch.where(held, graph.parent, 0)
        child = torch.where(held, graph.child, 0)
        logs = torch.log(values.index_select(0, child).clamp_min(1e-30))
        product = torch.exp(torch.zeros_like(values).index_add(
            0, parent, torch.where(held[:, None, None, None], logs, 0.)))
        is_symbol = (nodes >= graph.concepts) & (nodes < graph.percept_start)
        values = torch.where(is_symbol[:, None, None, None], product, values)
    grad = _down(graph, request.to(values), values, readings, 2 * (len(owner._order_caps()) + 1), nodes=nodes)
    return grad[:V].amax(2)


def rows_asked(sigma, rows, concepts):
    """What each of ``rows`` (basic percepts) is asked, ``[R, B, 2]``, by the
    order-zero Sigma's way back of each lane of the concepts' request
    ``concepts`` ``[V, B, 2]``: a lane's result below nothing asks the row's
    other lane (the Not's). A row no concept reads is asked nothing."""
    S = sigma.slots
    R, B = len(rows), concepts.shape[1]
    if not S or not R:
        return concepts.new_zeros(R, B, 2)
    n = min(len(concepts), sigma.nOutput)
    at = sigma.output_slot[:n].to(concepts.device)
    x = concepts.new_zeros(S + 1, B, 2).index_add(0, torch.where(at >= 0, at, S), concepts[:n])
    back = sigma.back(x[:S])
    back = torch.cat((back, back.new_zeros(1, B, 2)))
    rows = torch.as_tensor(rows, dtype=torch.long, device=concepts.device)
    slot = sigma.input_slot.to(concepts.device).index_select(0, rows.clamp_max(sigma.nInput - 1))
    asked = back.index_select(0, torch.where((rows < sigma.nInput) & (slot >= 0), slot, S))
    from Layers import signed_lanes
    f, a = signed_lanes(asked[..., 0]), signed_lanes(asked[..., 1])
    return torch.stack((f[..., 0] + a[..., 1], f[..., 1] + a[..., 0]), -1)


def rows_read(basic, bank):
    """Each row's read of every occurrence of the bank, on its two lanes,
    ``[R, B, occurrences, 2]``: the percept the row keeps matched against
    the occurrence by shared size (the first Pi's own read: ``held_at`` on
    the ``.what``, ``place_factor`` on the ``.where`` and the ``.when``). A
    conjunction is read through its members' rows and reads nothing here."""
    from BasicPercepts import held_at
    if not bank.rows:
        return bank.table.new_zeros(0, *bank.token.shape, 2)
    if bank.read_basis is None:
        part, whole, added = held_at(bank, bank.table)
        at = basic.place_factor(bank.table, bank.rows)
        outside = basic.outside(bank.table, bank.rows, bank.width)
    else:
        part, whole, added, outside, _, at = bank.read_basis
    # As the way up reads it (:meth:`BasicPercepts.BasicPercepts.read_bank`),
    # each occurrence wholly present: only an occurrence of its own tells
    # against what it holds by what it lacks.
    told = at if bank.holder is None else at * bank.holder.to(at)[None]
    unheld = torch.maximum((1 - whole) * told, outside * at)
    conjunctions = getattr(basic, 'conjunctions', {})
    keep = torch.tensor([row not in conjunctions for row in bank.rows], device=at.device)
    keep = keep[:, None, None].to(at)
    return torch.stack((part * whole * at * keep,
                        part * torch.maximum(unheld, added * told) * keep), -1)


def through_the_not(read, token):
    """A row's read of each occurrence ``[R, B, occurrences, 2]`` taken back
    through the way up's reading and its Not: a reading's lane is the
    greatest of its occurrences' (the way up's), the Not makes the reading one
    value, for less against, and that value's greater lane goes back to the
    occurrences that hold its greatest. So a row read for at a run inside a
    longer word and against at the word, in one reading, lights neither. An
    occurrence of no reading is its own."""
    from Layers import signed_lanes
    R, B, O, _ = read.shape
    K = O + 1
    slot = torch.where(token >= 0, token, O)[None, :, :, None].expand(R, B, O, 2)
    detached = read.detach()
    greatest = detached.new_zeros(R, B, K, 2).scatter_reduce(2, slot, detached, 'amax')
    at = greatest.gather(2, slot)                                             # [R, B, O, 2]
    net = signed_lanes(at[..., 0] - at[..., 1])                               # the reading's one value
    held = ((detached >= at) & (detached > 0)).to(read)
    own = (token < 0)[None, :, :, None]
    return torch.where(own, signed_lanes(read[..., 0] - read[..., 1]), net * held)


def salience(owner, bank, request, *, field=None):
    """The salience map of ``request`` ``[V, B, 2]`` (each concept's two
    lanes: asked for, asked against) over every occurrence of ``bank``: each
    row's ask (:func:`rows_asked`, after :func:`concepts_asked`) times its
    read of the occurrence (:func:`rows_read`) taken back through the
    reading and the Not (:func:`through_the_not`), summed over the rows."""
    concepts = concepts_asked(owner, request, field)
    asked = rows_asked(owner.order_zero, bank.rows, concepts)
    read = through_the_not(rows_read(owner.basic_percepts, bank).to(asked), bank.token)
    events = (asked[:, :, None, :] * read).sum(0)
    return SalienceMap(torch.where(events >= 0, events, 0.), asked)
