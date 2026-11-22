"""``attend``'s read of the conceptual field (§21, rule 80).

One network, shared by every use: its input is the sentence's opened field,
the last look's field spread once (:func:`ConceptSpread.spread`, no N:N
map), whether anything is left for a look to find, and the concept of the word
last said on the output, each at the concepts' inventory rows. One layer of
rectified units, and three linear reads on top: the request (a mask over
the concepts in 0 to 1, the logistic of its read), the chance of stopping
before the look, and the word said, read from the same kind of units given
the words the look offers, each by its softmax share of the look. No
concept is lit here: what lights a concept is the input or the output.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RequestRead(nn.Module):
    """The read: ``first`` over ``given = 5 rows + 1`` numbers (two lanes of
    the opened field, two of the spread memory, whether anything is left,
    the word said),
    ``ask`` (the request), ``stop`` and, over ``given`` and the share-weighted
    word the look offers (its concept row and its centred ``.where`` wires),
    ``second`` then ``say`` (the word said, over the concept rows). The tops'
    weights start at nothing; the request's start is set by ``install``."""
    def __init__(self, rows, width, wires, device=None):
        super().__init__()
        self.rows = int(rows)
        given = 5 * self.rows + 1
        self.first = nn.Linear(given, int(width), device=device)
        self.second = nn.Linear(given + self.rows + int(wires), int(width), device=device)
        self.ask = nn.Linear(int(width), self.rows, device=device)
        self.stop = nn.Linear(int(width), 1, device=device)
        self.say = nn.Linear(int(width), self.rows, device=device)
        for top in (self.ask, self.stop, self.say):
            nn.init.zeros_(top.weight)
            nn.init.zeros_(top.bias)

    def given(self, opened, memory, found, said):
        """The input ``[B, given]``: ``opened`` and ``memory`` ``[B, V, 2]``,
        ``found`` ``[B]`` (anything left), ``said`` ``[B, V]``."""
        return torch.cat((opened.flatten(1), memory.flatten(1), found[:, None].to(opened), said.to(opened)), -1)

    def units(self, x):
        return F.relu(self.first(x))

    def request(self, h):
        """The request ``[V, B, 2]``: asked for by the logistic of its read,
        asked against by nothing."""
        asked = torch.sigmoid(self.ask(h)).transpose(0, 1)
        return torch.stack((asked, torch.zeros_like(asked)), -1)

    def stopping(self, h):
        return self.stop(h)[:, 0]

    def said(self, x, seen):
        return self.say(F.relu(self.second(torch.cat((x, seen.to(x)), -1))))


def install(model):
    """The read of ``attend``, where the model reads by looks; and the mask
    input's tolerance."""
    from util import TheXMLConfig
    from AttentionObjective import tolerate_heterogeneity
    model.het_tolerance = float(TheXMLConfig.get('architecture.hetTolerance', default=1.))
    tolerate_heterogeneity(torch.zeros(1, 2), model.het_tolerance)
    model.request_read = None
    model.look_sharpness = float(TheXMLConfig.get('architecture.lookSharpness'))
    owner = model._concept_owner() if hasattr(model, '_concept_owner') else None
    if owner is None or not hasattr(owner, 'similarity_codebook') or not model.attention_budget:
        return
    from BasicPercepts import WHERE_WIRES
    width = int(TheXMLConfig.get('architecture.requestReadWidth'))
    device = owner.similarity_codebook.W.device
    # The read's start is drawn from its own fork, so that installing it
    # moves no other draw of the model.
    with torch.random.fork_rng(devices=[]):
        model.request_read = RequestRead(int(owner.nVectors), width, WHERE_WIRES, device='cpu').to(device)
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
