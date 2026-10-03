"""One immutable trial value shared by reconstruction and answer readers."""
from dataclasses import dataclass, fields
import torch
from torch import nn
from torch.nn import functional as F


def readback_scores(leaf, codes, priming):
    """Signed activation times cosine times the candidate's priming weight.

    For a non-unit code c the recovered activation is the least-squares
    coefficient (leaf.c)/(c.c). Multiplying it by the signed cosine preserves
    an excluded word's identity without assuming a unit code. A zero code has
    no direction and scores zero. The codes and the recovered leaf stay live.
    """
    square = codes.square().sum(-1)
    nonzero = square > 0
    denominator = torch.where(nonzero, square, torch.ones_like(square))
    activation = (leaf[:, None] * codes).sum(-1) / denominator
    cosine = F.cosine_similarity(leaf[:, None], codes, dim=-1)
    return torch.where(nonzero, activation * cosine * priming, 0.)


@dataclass(frozen=True)
class PrimedSymbols:
    rows: torch.Tensor
    codes: torch.Tensor
    weights: torch.Tensor
    own: torch.Tensor
    bytes: torch.Tensor
    byte_valid: torch.Tensor

    @property
    def valid(self):
        return self.rows >= 0


@dataclass(frozen=True)
class SentenceUnderstanding:
    root: torch.Tensor
    end_slots: torch.Tensor
    end_depth: torch.Tensor
    roots: torch.Tensor
    depths: torch.Tensor
    word_values: torch.Tensor
    word_rows: torch.Tensor
    word_valid: torch.Tensor
    primed: PrimedSymbols
    sentence: torch.Tensor

    def reader_features(self):
        """Fixed linear summaries; all evidence is detached at this boundary.

        The numeric reader is affine in the root, end slots and echoic bank.
        Neither the compose journal nor original per-word values are evidence
        for an answer. The code/activation product is fixed at this boundary.
        """
        bank = (self.primed.codes.detach() * self.primed.weights.detach()[..., None]
                * self.primed.valid[..., None]).sum(1)
        positions = torch.arange(self.end_slots.shape[1], device=self.root.device)
        end = self.end_slots * (positions[None] < self.end_depth[:, None])[..., None]
        return torch.cat((self.root, end.flatten(1), bank), -1).detach()

    def detached(self):
        return type(self)(**{f.name: getattr(self, f.name).detach()
                            if torch.is_tensor(getattr(self, f.name)) else getattr(self, f.name)
                            for f in fields(self)})

    @classmethod
    def select(cls, greedy, explore, wins):
        values = {}
        for field in fields(cls):
            left, right = getattr(greedy, field.name), getattr(explore, field.name)
            if field.name == 'primed':
                if left is not right:
                    raise RuntimeError('sentence trials must share their priming snapshot')
                values[field.name] = left
            elif left.ndim == 0:
                values[field.name] = left.detach()
            else:
                gate = wins.reshape((-1,) + (1,) * (left.ndim-1))
                values[field.name] = torch.where(gate, right, left).detach()
        return cls(**values)


class SentenceRecordReader(nn.Module):
    """Answer-owned affine reading of the rest of a concluded record."""
    def __init__(self, dimension, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(int(output_size), 5*int(dimension)))

    def forward(self, record):
        return F.linear(record.reader_features(), self.weight)
