"""Expectation: the vertical spread on the structure's own matrices.

What is expected is a fan up and a share down over the Pi, the Not and the
Sigma of every order, read by the size of their weights. It adds no
parameter, writes only the expected field, and is never evidence.

The structure read, as edges from a row to what it reads (its inputs):

* a concept of order zero reads basic percepts (``order_zero``, the Sigma
  over the Not's options), each entry by its weight, a Not entry passing
  to the other lane;
* a basic percept reads its parts (the first Pi's for entries), each at
  its weight;
* a symbol reads the concepts it holds (the symbols' Pi), each at one;
* a kind reads symbols (the store's Sigma over the Not's options), each
  entry by its weight, a Not entry passing to the other lane.

Up: a node's value is split among the rows that read it, each by its
weight's share of all the weights it is read with (its fan). Down: a row's
value is split among its inputs, each by its weight's share of the row.
What a node receives is summed over all its parents and children, at most
one.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Prediction:
    value: torch.Tensor
    rows: torch.Tensor
    active: torch.Tensor
    target: torch.Tensor | None = None


@dataclass(frozen=True)
class Structure:
    """The structure as edges ``parent reads child`` over one node space:
    the concepts at their inventory rows ``[0, V)``, the symbols
    ``[V, V + T)``, the basic percepts ``[V + T, V + T + R)`` and the
    parts of the first Pi ``[V + T + R, N)``."""
    parent: torch.Tensor
    child: torch.Tensor
    weight: torch.Tensor
    negated: torch.Tensor
    concepts: int
    symbols: int
    percepts: int
    parts: int
    part_keys: tuple

    @property
    def size(self):
        return self.concepts + self.symbols + self.percepts + self.parts

    @property
    def percept_start(self):
        return self.concepts + self.symbols

    @property
    def part_start(self):
        return self.concepts + self.symbols + self.percepts


def rule_order(owner, row):
    value = owner._expected_knowing[row].detach().amax(-1)
    return tuple(value.argsort(descending=True, stable=True).tolist())


def _store_edges(owner, V, device):
    """The symbols' Pi and the kinds' Sigma over every inventory row."""
    from Spaces import _concept_alloc_of
    alloc = getattr(owner, '_concept_allocator', None)
    if alloc is None:
        return 0, []
    store = _concept_alloc_of(owner).layer(0)
    if not getattr(store, '_thing_columns', False) or store.values is None or not store.nnz:
        return 0, []
    view = store.bind(torch.arange(V))
    seats = view._thing_seats.to(device)
    T = int(seats.shape[0])
    if not T:
        return 0, []
    edges = []
    complete = view._thing_complete.to(device)
    held = (seats >= 0) & complete[:, None]
    symbol = torch.arange(T, device=device)[:, None].expand_as(seats)[held]
    member = seats[held]
    edges.append((V + symbol, member, torch.ones(len(member), device=device),
                  torch.zeros(len(member), dtype=torch.bool, device=device)))
    rows, cols = view._indices(device)
    value = view.values.to(device)
    keep = (cols < T) & (value != 0)
    missing = getattr(view, '_missing_sources', None)
    if missing is not None and len(missing):
        keep &= ~missing.to(device)
    edges.append((rows[keep], V + cols[keep], value[keep].abs(), value[keep] < 0))
    return T, edges


def structure(owner, device=None):
    """The edges of the structure as it stands now; the weights carry the
    matrices' gradient."""
    V = int(owner.nVectors)
    sigma = getattr(owner, 'order_zero', None)
    device = owner._expected_knowing.device if device is None else device
    T, edges = _store_edges(owner, V, device)
    R, P, keys = 0, 0, ()
    if sigma is not None and sigma.nnz:
        R = int(sigma.nInput)
        n = sigma.nnz
        value = sigma.value[:n].to(device)
        rows = sigma.entry_row[:n].to(device)
        inputs = sigma.entry_input[:n].to(device)
        keep = value != 0
        edges.append((rows[keep], V + T + inputs[keep], value[keep].abs(), value[keep] < 0))
        basic = getattr(owner, 'basic_percepts', None)
        pi = getattr(basic, 'pi', None)
        if pi is not None and pi.n_rows:
            keys = tuple(pi.keys)
            P = len(keys)
            nnz = pi.nnz
            counts = (pi.crow[1:pi.n_rows + 1] - pi.crow[:pi.n_rows]).to(device)
            row_of = torch.repeat_interleave(torch.arange(pi.n_rows, device=device), counts)
            col = pi.col[:nnz].to(device)
            weight = pi.value[0, :nnz].to(device)
            keep = ~pi.against[:nnz].to(device) & (col >= 0)
            edges.append((V + T + row_of[keep], V + T + R + col[keep], weight[keep],
                          torch.zeros(int(keep.sum()), dtype=torch.bool, device=device)))
    if edges:
        parent, child, weight, negated = (torch.cat(part) for part in zip(*edges))
    else:
        parent = child = torch.zeros(0, dtype=torch.long, device=device)
        weight = torch.zeros(0, device=device)
        negated = torch.zeros(0, dtype=torch.bool, device=device)
    return Structure(parent, child, weight, negated, V, T, R, P, keys)


def _carry(values, source, target, share, negated, size):
    """Each edge carries its source's two lanes, by its share, to its
    target, exchanged through a Not entry; summed at each target."""
    lanes = values.index_select(0, source)
    lanes = torch.where(negated[:, None, None], lanes.flip(-1), lanes)
    lanes = lanes * share[:, None, None]
    return values.new_zeros(size, *values.shape[1:]).index_add(0, target, lanes)


def spread(owner, observed, *, percepts=None, structure_=None):
    """The expected value of every node, ``[N, B, 2]``, from the observed
    concepts ``observed`` ``[V, B, 2]`` (and, where given, the present
    parts of the first Pi ``percepts`` ``[P, B, 2]``): up once by the fan,
    then down once by the share, at most one. The observed values are the
    sources and are not returned as expected."""
    graph = structure(owner, observed.device) if structure_ is None else structure_
    N = graph.size
    source = observed.new_zeros(N, *observed.shape[1:])
    source[:min(len(observed), graph.concepts)] = observed[:graph.concepts]
    if percepts is not None and graph.parts:
        source[graph.part_start:graph.part_start + len(percepts)] = percepts[:graph.parts]
    if not len(graph.parent):
        return torch.zeros_like(source), graph
    weight = graph.weight.to(observed)
    fan = weight.new_zeros(N).index_add(0, graph.child, weight)
    row = weight.new_zeros(N).index_add(0, graph.parent, weight)
    up_share = weight / fan.index_select(0, graph.child).clamp_min(torch.finfo(weight.dtype).tiny)
    down_share = weight / row.index_select(0, graph.parent).clamp_min(torch.finfo(weight.dtype).tiny)
    # The structure is layered and acyclic: four steps an order reach its
    # top from its bottom (part, percept, concept, symbol).
    depth = 4 * (len(owner._order_caps()) + 1)
    up = torch.zeros_like(source)
    for _ in range(depth):
        nxt = _carry(source + up, graph.child, graph.parent, up_share, graph.negated, N).clamp(max=1.)
        if torch.equal(nxt, up):
            break
        up = nxt
    down = torch.zeros_like(source)
    for _ in range(depth):
        nxt = _carry(source + up + down, graph.parent, graph.child, down_share, graph.negated, N).clamp(max=1.)
        if torch.equal(nxt, down):
            break
        down = nxt
    return (up + down).clamp(max=1.), graph


def expected(owner, knowing, *, percepts=None):
    """The expected concepts ``[B, W, 2]`` of the observed ``knowing``
    ``[B, W, 2]`` (``W`` the inventory or its first order): the spread's
    concepts, with the gradient of the structure's weights."""
    B, width = knowing.shape[:2]
    V = int(owner.nVectors)
    observed = knowing.new_zeros(V, B, 2)
    observed[:width] = knowing.transpose(0, 1)
    value, graph = spread(owner, observed, percepts=percepts)
    owner._expected_percepts = value[graph.percept_start:graph.part_start].detach()
    owner._expected_parts = (graph.part_keys, value[graph.part_start:].detach())
    return value[:width].transpose(0, 1)


def open_prediction(model, present, *, sid=None):
    owner = model._concept_owner()
    owner._serial_concept_path = bool(getattr(getattr(model, 'inputSpace', None), '_serial_word_major', False)
                                      or getattr(model, 'word_brackets', False) and getattr(model, 'serial_object_meta', False))
    owner._concept_prediction_open = True
    B = len(present)
    if B > owner.nStreams:
        raise RuntimeError('concept prediction exceeds constructed streams')
    rows = torch.arange(owner._order_caps()[0], device=present.device)
    knowing = owner._observed_knowing[:B].detach().clone()
    from IdeaRead import open_read
    open_read(model, knowing)
    prediction = expected(owner, knowing)
    from Omission import concept_teacher
    target, hidden = concept_teacher(model, sid=sid, present=present)
    model._concept_prediction = Prediction(prediction[:, :len(rows)], rows, present & hidden, target)
    with torch.no_grad():
        stored = owner._expected_knowing[:B]
        stored.copy_(torch.where(present[:, None, None], prediction.detach(), stored))


def loss(model):
    prediction = getattr(model, '_concept_prediction', None)
    if prediction is None or prediction.target is None or not bool(prediction.active.any()):
        return None
    width = prediction.target.shape[1]
    error = (prediction.value[:, :width] - prediction.target.detach()).square().mean((1, 2))
    return torch.where(prediction.active, error, 0.)


def complete(model, present, registry):
    """The second stage of the reconstruction: the expected concepts against
    the whole sentence's, where some of it was hidden."""
    value = loss(model)
    if value is not None:
        registry.error('expectation.concepts', value, 1., mask=present, category='expectation')
