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
        for top in (self.ask, self.stop):
            nn.init.zeros_(top.weight)
            nn.init.zeros_(top.bias)

    def given(self, opened, memory, found):
        return torch.cat((opened.flatten(1), memory.flatten(1), found[:, None].to(opened)), -1)

    def units(self, x):
        return F.relu(self.first(x))

    def request(self, h):
        """The request ``[V, B, 2]``: asked for by the logistic of its read,
        asked against by nothing."""
        asked = torch.sigmoid(self.ask(h)).transpose(0, 1)
        return torch.stack((asked, torch.zeros_like(asked)), -1)

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


def at_rows(rows, field, V, B):
    """A field's concepts at their inventory rows ``rows`` (``[S]`` or ``[S,
    B]``), ``[B, V, 2]``: each concept's greatest reading on each lane, in 0
    to 1, detached."""
    evidence = field.evidence.detach()[:, :B]                                 # [S, B, W, 2]
    S = evidence.shape[0]
    rows = rows[:, None].expand(S, B) if rows.ndim == 1 else rows[:, :B]
    value = evidence.amax(2).clamp(0., 1.)                                    # [S, B, 2]
    index = torch.where(rows >= 0, rows, V)[..., None].expand(-1, -1, 2)
    held = value.new_zeros(V + 1, B, 2).scatter_reduce(0, index, value, 'amax')[:V]
    return held.transpose(0, 1)


def spread_once(owner, held):
    """The last look's field ``[B, V, 2]`` spread once (up then down on the
    structure, :func:`ConceptSpread.spread`), joined with what it held as
    the product on complements; detached."""
    from ConceptSpread import spread, union
    observed = held.transpose(0, 1)
    value, _ = spread(owner, observed)
    return union(observed, value[:len(observed)]).transpose(0, 1).detach()
