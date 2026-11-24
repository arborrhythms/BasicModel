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
What a node receives from several rows is joined as the product on
complements, one less the product of one less each share: it stays below
one and in order, more rows that give, a higher value.
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
        edges = []
        return _pair_edges(owner, V, edges, device), edges
    from CandidateGrowth import visible_store
    store = visible_store(_concept_alloc_of(owner).layer(0))
    if not getattr(store, '_thing_columns', False) or store.values is None or not store.nnz:
        edges = []
        return _pair_edges(owner, V, edges, device), edges
    view = store.bind(range(V))
    seats = view._thing_seats.to(device)
    T = int(seats.shape[0])
    if not T:
        edges = []
        return _pair_edges(owner, V, edges, device), edges
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
    return T + _pair_edges(owner, V + T, edges, device), edges


def _pair_edges(owner, start, edges, device):
    """The rows of the pairs the candidate matrices promoted, as symbols
    from ``start``, each reading its two concepts at one."""
    from CandidateGrowth import visible_pairs
    pairs = visible_pairs(owner)
    rows = [(owner._concept_row_of(a), owner._concept_row_of(b)) for a, b in pairs]
    rows = [pair for pair in rows if None not in pair]
    if not rows:
        return 0
    member = torch.tensor([r for pair in rows for r in pair], dtype=torch.long, device=device)
    parent = start + torch.arange(len(rows), device=device).repeat_interleave(2)
    edges.append((parent, member, torch.ones(len(member), device=device),
                  torch.zeros(len(member), dtype=torch.bool, device=device)))
    return len(rows)


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
        value = sigma.w()[:n].to(device)
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
            row_of = torch.repeat_interleave(torch.arange(pi.n_rows, device=device), counts, output_size=nnz)
            col = pi.col[:nnz].to(device)
            weight = pi.value[0, :nnz].to(device)
            keep = ~pi.against[:nnz].to(device) & (col >= 0)
            source = col[keep]
            edges.append((V + T + row_of[keep], V + T + R + source, weight[keep],
                          torch.zeros_like(source, dtype=torch.bool)))
    if edges:
        parent, child, weight, negated = (torch.cat(part) for part in zip(*edges))
    else:
        parent = child = torch.zeros(0, dtype=torch.long, device=device)
        weight = torch.zeros(0, device=device)
        negated = torch.zeros(0, dtype=torch.bool, device=device)
    return Structure(parent, child, weight, negated, V, T, R, P, keys)


def union(*values):
    """One less the product of one less each value."""
    rest = 1. - values[0]
    for value in values[1:]:
        rest = rest * (1. - value)
    return 1. - rest


def _carry(values, source, target, share, negated, size):
    """Each edge carries its source's two lanes, by its share, to its
    target, exchanged through a Not entry; joined at each target as the
    product on complements."""
    lanes = values.index_select(0, source)
    lanes = torch.where(negated[:, None, None], lanes.flip(-1), lanes)
    lanes = lanes * share[:, None, None]
    index = target.reshape(-1, *([1] * (lanes.ndim - 1))).expand_as(lanes)
    rest = values.new_ones(size, *values.shape[1:]).scatter_reduce(0, index, 1. - lanes, 'prod')
    return 1. - rest


def _flow(graph, observed, percepts, depth):
    """The sources, and the spread's up and down at every node, detached."""
    N = graph.size
    source = observed.new_zeros(N, *observed.shape[1:])
    source[:min(len(observed), graph.concepts)] = observed[:graph.concepts]
    if percepts is not None and graph.parts:
        source[graph.part_start:graph.part_start + len(percepts)] = percepts[:graph.parts]
    source = source.detach().clamp(0., 1.)
    up, down = torch.zeros_like(source), torch.zeros_like(source)
    if not len(graph.parent):
        return source, up, down, None, None
    weight = graph.weight.detach().to(observed)
    tiny = torch.finfo(weight.dtype).tiny
    fan = weight.new_zeros(N).index_add(0, graph.child, weight).clamp_min(tiny)
    row = weight.new_zeros(N).index_add(0, graph.parent, weight).clamp_min(tiny)
    up_share = weight / fan.index_select(0, graph.child)
    down_share = weight / row.index_select(0, graph.parent)
    for _ in range(depth):
        nxt = _carry(union(source, up), graph.child, graph.parent, up_share, graph.negated, N)
        if torch.equal(nxt, up):
            break
        up = nxt
    for _ in range(depth):
        nxt = _carry(union(source, up, down), graph.parent, graph.child, down_share, graph.negated, N)
        if torch.equal(nxt, down):
            break
        down = nxt
    return source, up, down, fan, row


def spread(owner, observed, *, percepts=None, structure_=None):
    """The expected value of every node, ``[N, B, 2]``, from the observed
    concepts ``observed`` ``[V, B, 2]`` (and, where given, the present
    parts of the first Pi ``percepts`` ``[P, B, 2]``): up once by the fan,
    then down once by the share, each joined as the product on complements.
    The observed values are the
    sources and are not returned as expected. It is arithmetic: nothing is
    taught through it."""
    graph = structure(owner, observed.device) if structure_ is None else structure_
    depth = 4 * (len(owner._order_caps()) + 1)
    with torch.no_grad():
        _, up, down, _, _ = _flow(graph, observed, percepts, depth)
    return union(up, down), graph


def _staged(owner, observed, percepts, stage):
    graph = structure(owner, observed.device)
    depth = 4 * (len(owner._order_caps()) + 1)
    source, up, down, fan, _ = _flow(graph, observed, percepts, depth)
    lo, hi = ((0, graph.concepts) if stage == 'concepts' else (graph.percept_start, graph.part_start))
    if fan is None:
        return union(up, down)[lo:hi]
    own = (graph.parent >= lo) & (graph.parent < hi)
    weight = graph.weight.to(observed)[own]
    child, parent = graph.child[own], graph.parent[own]
    share = weight / fan.index_select(0, child)
    reading = _carry(union(source, up), child, parent, share, graph.negated[own], graph.size)
    return union(reading[lo:hi], down[lo:hi])


def staged(owner, observed, *, percepts=None):
    """The expected concepts ``[V, B, 2]`` with the gradient of their own
    stage alone: each concept's reading of the things beneath it (its
    Sigma's entries, by their share of the fan), from what the spread left
    beneath, detached; what reaches it from above is detached too. No
    gradient crosses into another stage or the way back."""
    return _staged(owner, observed, percepts, 'concepts')


def staged_percepts(owner, observed, *, percepts=None):
    """The expected basic percepts ``[R, B, 2]`` with the gradient of their
    own stage alone: each basic percept's reading of its parts (the first
    Pi's for entries, by their share of the fan), from what the spread left
    at the parts, detached; what reaches it from the concepts above is
    detached."""
    return _staged(owner, observed, percepts, 'percepts')


def stage_values(owner, observed, *, percepts=None):
    """What arrives at each stage from the observed concepts: the concepts
    themselves, and each symbol's two lanes, the product of its members'
    (the symbols' Pi). ``dict(concepts=[V, B, 2], symbols=[T, B, 2])``."""
    graph = structure(owner, observed.device)
    V, T = graph.concepts, graph.symbols
    concepts = observed[:V].detach()
    symbols = concepts.new_zeros(T, *concepts.shape[1:])
    if T and V:
        held = (graph.parent >= V) & (graph.parent < V + T)
        children = torch.where(held, graph.child, 0)
        parents = torch.where(held, graph.parent - V, 0)
        logs = concepts.index_select(0, children).clamp_min(1e-30).log()
        products = symbols.index_add(0, parents, torch.where(held[:, None, None], logs, 0.)).exp()
        symbols = torch.where(held.any(), products, 0.)
    return dict(concepts=concepts, symbols=symbols)


def expected(owner, knowing, *, percepts=None):
    """The expected concepts ``[B, W, 2]`` of the observed ``knowing``
    ``[B, W, 2]`` (``W`` the inventory or its first order), detached; the
    expected basic percepts and parts are kept beside them
    (``_expected_percepts``, ``_expected_parts``) and the expected symbols
    (``_expected_symbols``)."""
    B, width = knowing.shape[:2]
    V = int(owner.nVectors)
    observed = knowing.new_zeros(V, B, 2)
    observed[:width] = knowing.transpose(0, 1).detach()
    value, graph = spread(owner, observed, percepts=percepts)
    from CandidateGrowth import speculative
    if not speculative(owner):
        owner._expected_percepts = value[graph.percept_start:graph.part_start]
        owner._expected_parts = (graph.part_keys, value[graph.part_start:])
        owner._expected_symbols = value[graph.concepts:graph.percept_start]
    return value[:V].transpose(0, 1)


def percept_cost(owner, before, arrived):
    """The basic percepts' stage at one look, ``[B]``: what each basic
    percept expected from what the earlier looks left (``before`` ``[B, V,
    2]``), with the gradient of its own entries of the first Pi alone,
    against the basic percepts the look brought (``arrived`` ``[B, R, 2]``
    for the first ``R`` basic percepts)."""
    V = int(owner.nVectors)
    B = before.shape[0]
    observed = before.new_zeros(V, B, 2)
    observed[:before.shape[1]] = before.transpose(0, 1).detach()
    expected = staged_percepts(owner, observed)
    R = min(arrived.shape[1], len(expected))
    if not R:
        return before.new_zeros(B)
    error = (expected[:R] - arrived[:, :R].transpose(0, 1).detach()).square().mean((0, 2))
    stages = dict(getattr(owner, '_expectation_stages', None) or {})
    stages['percepts'] = error.detach().mean()
    owner._expectation_stages = stages
    return error


def look_cost(owner, before, arrived):
    """The stages of expectation at one look. ``before`` ``[B, V, 2]`` is
    what the earlier looks left, ``arrived`` ``[B, V, 2]`` what the look
    brought. The concepts' stage: what each concept expected, with the
    gradient of its own entries alone, against what arrived, ``[B]``. The
    symbols' stage is measured (``_expectation_stages``); the symbols' Pi
    has no learned weight."""
    V = int(owner.nVectors)
    B = before.shape[0]
    observed = before.new_zeros(V, B, 2)
    observed[:before.shape[1]] = before.transpose(0, 1).detach()
    target = arrived.new_zeros(V, B, 2)
    target[:arrived.shape[1]] = arrived.transpose(0, 1).detach()
    concepts = staged(owner, observed)
    error = (concepts - target).square().mean((0, 2))
    with torch.no_grad():
        value, graph = spread(owner, observed)
        symbols = stage_values(owner, target)['symbols']
        expected_symbols = value[graph.concepts:graph.percept_start]
        owner._expectation_stages = dict(
            getattr(owner, '_expectation_stages', None) or {},
            concepts=error.detach().mean(),
            symbols=(expected_symbols - symbols).square().mean() if len(symbols) else error.new_zeros(()))
    return error


def open_prediction(model, present, *, sid=None):
    """What is expected at the sentence's open, from what it has observed so
    far, written to ``_expected_knowing`` for its readers; each look writes
    it anew (:class:`LookForecast.LookForecast`)."""
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
    model._concept_prediction = Prediction(prediction[:, :len(rows)], rows, present.clone())
    model._expectation_looks = []
    model._expectation_percept_looks = []
    with torch.no_grad():
        stored = owner._expected_knowing[:B]
        stored.copy_(torch.where(present[:, None, None], prediction, stored))


def write_look(owner, rows, value):
    """Write what is expected at a look, ``value`` ``[B, W, 2]`` at inventory
    rows ``rows`` ``[W]``, anew: nothing earlier is kept."""
    with torch.no_grad():
        B = value.shape[0]
        owner._expected_knowing[:B].zero_()
        owner._expected_knowing[:B, rows] = value.detach()


def loss(model, looks='_expectation_looks'):
    looks = getattr(model, looks, None)
    if not looks:
        return None
    return torch.stack([error for error, _ in looks]).sum(0)


def complete(model, present, registry):
    """The stages of expectation, each summed over the looks: at each look
    what was expected against what arrived, each look's cost on its own
    stage's entries: the concepts (``expectation.concepts``) and the basic
    percepts (``expectation.percepts``)."""
    for name, looks in (('expectation.concepts', '_expectation_looks'),
                        ('expectation.percepts', '_expectation_percept_looks')):
        held = getattr(model, looks, None)
        if held:
            active = torch.stack([mask for _, mask in held]).any(0)
            registry.error(name, loss(model, looks), 1., mask=active & present, category='expectation')
        setattr(model, looks, [])
