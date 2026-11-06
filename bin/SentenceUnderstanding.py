"""One immutable trial value shared by reconstruction and answer readers."""
from dataclasses import dataclass, fields
import torch
from torch import nn
from torch.nn import functional as F


def readback_scores(leaf, codes, priming, *, percept_width=None, meaning_start=None):
    """Read native surface identity by scale-free perceptual cosine and priming.

    Generic field callers without a perceptual band retain their signed
    activation times cosine score. For a non-unit code c the least-squares
    activation is (leaf.c)/(c.c). Multiplying it by the signed cosine preserves
    an excluded word's identity without assuming a unit code. A zero code has
    no direction and scores zero. The codes and the recovered leaf stay live.
    """
    if meaning_start is not None:
        # Both positive symbols share one signless form. Neither lane is
        # evidence about which word this is, and both/neither are not aliases.
        percept_width = meaning_start if percept_width is None else percept_width
    if percept_width is not None:
        # The sign is the leaf's activation, not its surface identity. On
        # native nonnegative percepts, absolute cosine reads either pole of
        # that identity without using code length or context as evidence.
        return F.cosine_similarity(leaf[:, None, :percept_width],
                                   codes[..., :percept_width], dim=-1).abs() * priming
    square = codes.square().sum(-1)
    nonzero = square > 0
    denominator = torch.where(nonzero, square, torch.ones_like(square))
    activation = (leaf[:, None] * codes).sum(-1) / denominator
    cosine = F.cosine_similarity(leaf[:, None], codes, dim=-1)
    return torch.where(nonzero, activation * cosine * priming, 0.)


@torch.no_grad()
def readback_decisions(words, counts, bank, active):
    """Code, priming, or exact tie, using the same leaves without re-decoding."""
    valid = bank.terminal_valid
    result = []
    for position in range(words.shape[1]):
        code = readback_scores(words[:, position], bank.codes, torch.ones_like(bank.weights),
                               percept_width=bank.percept_width, meaning_start=bank.meaning_start)
        weighted = readback_scores(words[:, position], bank.codes, bank.weights,
                                   percept_width=bank.percept_width, meaning_start=bank.meaning_start)
        code = code.masked_fill(~valid, -torch.inf)
        weighted = weighted.masked_fill(~valid, -torch.inf)
        for b in range(words.shape[0]):
            if not bool(active[b]) or position >= int(counts[b]) or not bool(valid[b].any()):
                continue
            winner, neutral = int(weighted[b].argmax()), int(code[b].argmax())
            code_ties = int(((code[b] == code[b].max()) & valid[b]).sum())
            weighted_ties = int(((weighted[b] == weighted[b].max()) & valid[b]).sum())
            decision = ('tie' if weighted_ties > 1 else
                        'priming' if code_ties > 1 or winner != neutral else 'code')
            result.append(dict(batch_row=b, position=position, decided_by=decision,
                winner_row=int(bank.rows[b, winner]), code_winner_row=int(bank.rows[b, neutral]),
                code_ties=code_ties, weighted_ties=weighted_ties,
                changed_winner=winner != neutral,
                winner_priming=float(bank.weights[b, winner]),
                winner_code_score=float(code[b, winner]), best_code_score=float(code[b].max())))
    return result


@dataclass(frozen=True)
class PrimedSymbols:
    rows: torch.Tensor
    codes: torch.Tensor
    weights: torch.Tensor
    own: torch.Tensor
    bytes: torch.Tensor | None
    byte_valid: torch.Tensor | None
    case_bank: object = None
    percept_width: int | None = None
    forms: torch.Tensor | None = None
    meaning_start: int | None = None
    normalize_reader: bool = False
    type_families: object = None
    form_wholes: object = None
    surface_present: torch.Tensor | None = None
    primitive_support: torch.Tensor | None = None

    def __post_init__(self):
        # Each lexical table belongs to these candidate rows. WholeSpace's
        # structural bank is separate and is rebuilt by walk_bank after joins.
        for name in ('codes', 'weights', 'own', 'bytes', 'byte_valid',
                     'forms', 'surface_present', 'primitive_support'):
            table = getattr(self, name)
            if table is not None and table.shape[:self.rows.ndim] != self.rows.shape:
                raise ValueError(f'PrimedSymbols.{name} leading shape {tuple(table.shape[:self.rows.ndim])} '
                                 f'differs from rows {tuple(self.rows.shape)}')

    @property
    def valid(self):
        return self.rows >= 0

    @property
    def terminal_valid(self):
        """Only a row with a lexical realization can end a word walk."""
        present = self.surface_present
        if present is None:
            present = self.byte_valid.any(-1)
        return self.valid & present

    def reader_value(self, value, *, reference=None):
        """Shared snapshot scales retain an affine read, including the sum control.

        Each block uses one maximum norm over the frozen bank (all streams).
        Never normalize each composed root: that would make a mean nonlinear.
        """
        if not self.normalize_reader or self.meaning_start is None:
            return value
        reference = self.codes * self.valid[..., None] if reference is None else reference
        boundary = self.meaning_start
        result = []
        for start, end in ((0, boundary), (boundary, value.shape[-1])):
            block = reference[..., start:end].detach()
            scale = block.norm(dim=-1).amax().clamp_min(1.)
            result.append(value[..., start:end] / scale)
        return torch.cat(result, -1)


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
    constituents: torch.Tensor | None = None
    constituent_valid: torch.Tensor | None = None
    constituent_families: torch.Tensor | None = None
    percept_expected: torch.Tensor | None = None

    def reader_features(self, *, attention=None, live=False):
        """Fixed linear summaries with the selected training boundary.

        The numeric reader is affine in the root, end slots and echoic bank.
        Neither the compose journal nor original per-word values are evidence
        for an answer. The code/activation product is fixed at this boundary.
        """
        forms = self.primed.codes if self.primed.forms is None else self.primed.forms
        bank = (forms.detach() * self.primed.weights.detach()[..., None]
                * self.primed.valid[..., None]).sum(1)
        positions = torch.arange(self.end_slots.shape[1], device=self.root.device)
        end = self.end_slots * (positions[None] < self.end_depth[:, None])[..., None]
        root = self.primed.reader_value(self.root)
        end = self.primed.reader_value(end)
        bank = self.primed.reader_value(bank, reference=bank)
        from AttentionCredit import fixed_input
        value = torch.cat((root, end.flatten(1), bank), -1)
        return value if live else fixed_input(value, attention)

    def detached(self):
        return type(self)(**{f.name: getattr(self, f.name).detach()
                            if torch.is_tensor(getattr(self, f.name)) else getattr(self, f.name)
                            for f in fields(self)})

    @classmethod
    def select(cls, greedy, explore, wins, *, live=False, replay=None):
        """Keep a complete record. A live mixed record comes from one replay."""
        if greedy.primed is not explore.primed:
            raise RuntimeError('sentence trials must share their priming snapshot')
        if bool(wins.all()) or not bool(wins.any()):
            selected = explore if bool(wins.all()) else greedy
            return selected if live else selected.detached()
        with torch.no_grad():
            values = {}
            for field in fields(cls):
                left, right = getattr(greedy, field.name), getattr(explore, field.name)
                if not torch.is_tensor(left) or left.ndim == 0:
                    values[field.name] = left
                else:
                    gate = wins.reshape((-1,) + (1,) * (left.ndim-1))
                    values[field.name] = torch.where(gate, right, left)
            expected = cls(**values)
        if not live:
            return expected
        if replay is None:
            raise RuntimeError('live mixed sentence records require the selected compose replay')
        selected = replay()
        if selected.primed is not expected.primed:
            raise RuntimeError('kept sentence replay changed its priming snapshot')
        with torch.no_grad():
            for field in fields(cls):
                value, target = getattr(selected, field.name), getattr(expected, field.name)
                if torch.is_tensor(target):
                    torch._assert_async((value == target).all(), 'kept sentence replay changed ' + field.name)
        return selected


class SentenceRecordReader(nn.Module):
    """Answer-owned affine reading of the rest of a concluded record."""
    def __init__(self, dimension, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(int(output_size), 5*int(dimension)))

    def forward(self, record, *, attention=None, live=False):
        return F.linear(record.reader_features(attention=attention, live=live), self.weight)
