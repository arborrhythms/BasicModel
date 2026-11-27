"""Attention requests concepts from the open field and the preceding look."""
import torch
from torch import nn
from torch.nn import functional as F


class RequestRead(nn.Module):
    """Shared request and stopping reads over two fields and one availability flag."""

    def __init__(self, rows, width, device=None):
        super().__init__()
        self.rows = int(rows)
        given = 4 * self.rows + 1
        self.first = nn.Linear(given, int(width), device=device)
        self.ask = nn.Linear(int(width), self.rows, device=device)
        self.stop = nn.Linear(int(width), 1, device=device)
        from BasicPercepts import WHERE_WIRES, WHEN_WIRES
        # Two native place codes and the present.
        self.state_weight = nn.Parameter(self.first.weight.new_zeros(int(width), 2 * WHERE_WIRES + WHEN_WIRES))
        self.where = nn.Linear(int(width), WHERE_WIRES, device=device)
        self.where_end = nn.Linear(int(width), WHERE_WIRES, device=device)
        for top in (self.ask, self.stop, self.where, self.where_end):
            nn.init.zeros_(top.weight)
            nn.init.zeros_(top.bias)

    def given(self, opened, memory, found):
        return torch.cat((opened.flatten(1), memory.flatten(1), found[:, None].to(opened)), -1)

    def units(self, x, state=None):
        hidden = self.first(x)
        if state is not None:
            hidden = hidden + F.linear(state.to(hidden), self.state_weight)
        return F.relu(hidden)

    def request(self, h):
        """The request ``[V, B, 2]``: asked for by the logistic of its read,
        asked against by nothing."""
        asked = torch.sigmoid(self.ask(h)).transpose(0, 1)
        return torch.stack((asked, torch.zeros_like(asked)), -1)

    def forward(self, units):
        """One query over concept evidence and the native location band."""
        location = torch.cat((self.where(units), self.where_end(units)), dim=-1).T
        return torch.cat((self.request(units), torch.stack((location, torch.zeros_like(location)), -1)))

    def stopping(self, h):
        return self.stop(h)[:, 0]


def install(model):
    """The read of ``attend``, where the model reads by looks; and the mask
    input's tolerance."""
    from util import TheXMLConfig
    from AttentionObjective import tolerate_heterogeneity
    model.het_tolerance = float(TheXMLConfig.get('architecture.hetTolerance', default=1.))
    tolerate_heterogeneity(torch.zeros(1, 2), model.het_tolerance)
    model.request_read = None
    owner = model._concept_owner() if hasattr(model, '_concept_owner') else None
    if owner is None or not hasattr(owner, 'similarity_codebook') or not model.attention_budget:
        return
    width = int(TheXMLConfig.get('architecture.requestReadWidth'))
    device = owner.similarity_codebook.W.device
    # The read's start is drawn from its own fork, so that installing it
    # moves no other draw of the model.
    with torch.random.fork_rng(devices=[]):
        model.request_read = RequestRead(int(owner.nVectors), width, device='cpu').to(device)
    with torch.no_grad():
        model.request_read.ask.bias.fill_(float(TheXMLConfig.get('architecture.requestReadStart')))


def place_axes(bank):
    """Normalize start coordinates across the page and ends among parts sharing a start."""
    if bank.registry is None or not bank.span.shape[1]:
        return None
    def endpoints(spans):
        start, stop = spans.unbind(-1)
        places = torch.stack((start, torch.maximum(start, stop - 1)), -1)
        return bank.registry.encode('text', places).detach().unbind(-2)
    def normalize(value, live):
        center = (value * live[..., None]).sum(1, keepdim=True) / live.sum(1)[:, None, None].clamp_min(1)
        scale = ((value - center) * live[..., None]).abs().amax(1, keepdim=True)
        return center, scale.clamp_min(torch.finfo(value.dtype).eps)
    first, last = endpoints(bank.span)
    live = (bank.span[..., 1] > bank.span[..., 0]) & (bank.activity[..., 0] > 0)
    center, scale = normalize(first, live)
    def basis(spans, first, last, live):
        # Alternatives at this byte share the positional log weight. Their
        # end coordinates distinguish extents without rewarding the position twice.
        start = spans[..., 0]
        index = torch.where(live, start, 0)[..., None]
        count = last.new_zeros(len(spans), max(1, bank.registry._ranges['text'][1]), 1)
        count = count.scatter_add(1, index, live[..., None].to(count))
        index = index.expand_as(last)
        total = torch.zeros_like(count.expand(-1, -1, last.shape[-1])).scatter_add(
            1, index, last * live[..., None])
        relative = last - (total / count.clamp_min(1)).gather(1, index)
        return (first - center) / scale, torch.where(live[..., None], relative / scale, 0.)
    observed = basis(bank.span, first, last, live)
    first, last = endpoints(bank.where)
    readings = basis(bank.where, first, last, bank.where[..., 1] > bank.where[..., 0])
    return observed, readings


def spatial_match(bank, direction, *, readings=None, choose_extent=True):
    """Apply the request to observed parts, or to the first requested reading spans."""
    axes = bank.place_axes
    if axes is None and bank.registry is not None and bank.span.shape[1]:
        axes = place_axes(bank)
        object.__setattr__(bank, 'place_axes', axes)
    if axes is None:
        return direction.new_zeros((len(direction), bank.span.shape[1] if readings is None else readings))
    start, end = axes[0 if readings is None else 1]
    if readings is not None:
        start, end = start[:, :readings], end[:, :readings]
    position, extent = direction.chunk(2, -1)
    score = (start * position[:, None]).sum(-1)
    return score + (end * extent[:, None]).sum(-1) if choose_extent else score


def at_rows(rows, field, V, B, index=None):
    """A field's concepts at their inventory rows ``rows`` (``[S]`` or ``[S,
    B]``), ``[B, V, 2]``: each concept's greatest reading on each lane, in 0
    to 1, detached."""
    evidence = field.evidence.detach()[:, :B]                                 # [S, B, W, 2]
    S = evidence.shape[0]
    rows = rows[:, None].expand(S, B) if rows.ndim == 1 else rows[:, :B]
    if index is None:
        value = evidence.amax(2).clamp(0., 1.)
    else:
        batch = torch.arange(B, device=index.device)
        value = evidence[:, batch, index.clamp_min(0)].clamp(0., 1.) * (index >= 0)[None, :, None]
    index = torch.where(rows >= 0, rows, V)[..., None].expand(-1, -1, 2)
    held = value.new_zeros(V + 1, B, 2).scatter_reduce(0, index, value, 'amax')[:V]
    return held.transpose(0, 1)


def spread_once(owner, held):
    from CandidateGrowth import read_streams
    return read_streams(owner, lambda: _spread_once(owner, held), axis=0)


def _spread_once(owner, held):
    """The last look's field ``[B, V, 2]`` spread once (up then down on the
    structure, :func:`ConceptSpread.spread`), joined with what it held as
    the product on complements; detached."""
    from ConceptSpread import spread, union
    observed = held.transpose(0, 1)
    value, _ = spread(owner, observed)
    return union(observed, value[:len(observed)]).transpose(0, 1).detach()
