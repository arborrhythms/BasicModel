"""Order-zero reads of the shared symbol activation predictor."""
from dataclasses import dataclass
import torch
from Expectation import gain_field, symbol_activations


@dataclass(frozen=True)
class ExpectedWords:
    probabilities: torch.Tensor
    prediction: torch.Tensor
    content: torch.Tensor
    gain: torch.Tensor
    excess: torch.Tensor
    omission: torch.Tensor
    loss: torch.Tensor
    top1: torch.Tensor
    reciprocal_rank: torch.Tensor
    activation: torch.Tensor

    @property
    def surprise(self):
        return self.excess + self.omission


class WordArrivals:
    """One causal forecast state, opened once before any word arrives."""
    def __init__(self, owner, bank, valid, *, gain=1., epsilon=1e-6, saturation='clip'):
        bank = bank.detach()
        B, C, D = bank.shape
        if D != owner.word_dim or not C or valid.shape != (B, C):
            raise ValueError('word expectation needs a nonempty native-width bank')
        self.owner, self.bank, self.valid = owner, bank, valid
        self.gain, self.epsilon, self.saturation = gain, epsilon, saturation
        source = bank.new_zeros(B, C, 2)
        total, count, recent = bank.new_zeros(B, D), bank.new_zeros(B, 1), bank.new_zeros(B, D)
        # The word read begins with the already observed field/STM. Incoming
        # words join this context only after their own causal prediction.
        for row in range(B):
            chain, lane_history, identities, _ = (owner._read_context(row)
                if row < len(owner._inter_context) else ((), (), (), ()))
            if not chain:
                continue
            values = torch.stack([value.to(bank) for _, value, _ in chain]).flatten(0, 1)
            occupied = torch.stack([mask.to(bank.device) for _, _, mask in chain]).flatten()
            pairs = torch.stack([bank.new_tensor((1.,0.) if pair is None else pair).expand(3,2)
                for pair in lane_history]).flatten(0,1)
            handles = torch.stack([torch.full((3,), -1, device=bank.device, dtype=torch.long)
                if ids is None else ids.to(bank.device) for ids in identities]).flatten()
            orders = torch.full_like(handles, -1)
            if len(owner._bank_ids):
                matches = (handles[:, None] == owner._bank_ids.to(bank.device)[None]) & (handles[:, None] >= 0)
                orders = torch.where(matches.any(-1), owner._bank_orders.to(bank.device)[matches.long().argmax(-1)], -1)
            occupied = occupied & (orders == 0)
            source[row] = symbol_activations(bank[row:row+1],values[None],occupied[None],pairs[None],
                key_orders=torch.zeros(C, device=bank.device, dtype=torch.long), value_orders=orders[None])[0]
            total[row] = (values * occupied[:,None]).sum(0)
            count[row] = occupied.sum()
            positions = torch.where(occupied,torch.arange(len(values),device=bank.device)+1,0)
            recent[row] = torch.where(positions.amax() > 0, values[(positions.amax()-1).clamp_min(0)], 0.)
        self.source, self.total, self.count, self.recent = source, total, count, recent

    def step(self, words, targets, *, active=None, arrived=None, lanes=None, forecast=True):
        """Forecast the supplied arrival, then add it to the observed prefix."""
        words = words.detach()
        owner, bank, valid = self.owner, self.bank, self.valid
        B, C, D = bank.shape
        if words.shape != (B, D) or targets.shape != (B,):
            raise ValueError('word arrivals must match their native bank')
        active = torch.ones(B, device=words.device, dtype=torch.bool) if active is None else active
        arrived = active if arrived is None else active & arrived
        gain, epsilon, saturation = self.gain, self.epsilon, self.saturation
        source, total, count, recent = self.source, self.total, self.count, self.recent
        context = torch.cat((total/count.clamp_min(1), recent), -1)
        if forecast:
            self._logits = owner.word_predictor(*owner._prediction_inputs((bank, context, source)))
        logits = self._logits
        expected = logits.sigmoid() * valid[..., None]
        target = targets.clamp(0, C-1)
        observed = ((targets >= 0) & (targets < C)
                    & valid.gather(1, target[:, None])[:, 0] & active)
        pair = torch.stack((observed.to(words), words.new_zeros(B)), -1) if lanes is None else lanes.detach()
        actual = words.new_zeros(B, C, 2).scatter(1, target[:, None, None].expand(-1, 1, 2),
                                                (pair * observed[:, None])[:, None])
        field = gain_field(actual, expected, strength=gain, attention=valid,
                           epsilon=epsilon, saturation=saturation)
        mass = expected[..., 0]
        probability = mass / mass.sum(-1, keepdim=True).clamp_min(epsilon)
        loss = (actual - expected).square().sum((1, 2)) / (2*valid.sum(-1)).clamp_min(1)
        rank = 1. / (1. + (probability > probability.gather(1, target[:, None])).sum(-1))
        self.observe(words, arrived, lanes)
        gains = field.gain.gather(1, target[:, None])[:, 0]
        known = (targets >= 0) & (targets < C) & valid.gather(1, target[:, None])[:, 0]
        gains = torch.where(active, torch.where(known, gains, 1.), 0.)
        return ExpectedWords(probability[:, None],
            torch.einsum('bc,bcd->bd', probability, bank)[:, None],
            (words * gains[:, None])[:, None], gains[:, None],
            field.excess[:, None], field.omission[:, None],
            torch.where(observed, loss, 0.)[:, None],
            ((probability.argmax(-1) == target) & observed)[:, None],
            torch.where(observed, rank, 0.)[:, None], expected[:, None])


    def observe(self, words, seen, lanes=None):
        """Advance this path's prefix, including shared, already forecast words."""
        words = words.detach()
        # An observed prefix need not be a labelled candidate in this read's
        # bank (e.g. a held-out completion). It still supplies factual STM.
        bank, valid = self.bank, self.valid
        B = len(words)
        source, total, count, recent = self.source, self.total, self.count, self.recent
        total = total + words * seen[:, None]
        count = count + seen[:, None]
        recent = torch.where(seen[:, None], words, recent)
        distances = torch.cdist(words[:, None], bank, compute_mode='donot_use_mm_for_euclid_dist')[:, 0]
        matched = (distances <= 1e-5) & valid
        source_pair = (torch.stack((seen.to(words), words.new_zeros(B)), -1)
                       if lanes is None else lanes.detach())
        source_actual = matched[..., None].to(words) * source_pair[:, None]
        source = torch.where(seen[:, None, None], torch.maximum(source, source_actual), source)

        self.source, self.total, self.count, self.recent = source, total, count, recent


def word_distribution(owner, words, bank, valid, targets, *, gain=1., observed_prefix=None, active=None, lanes=None,
                      epsilon=1e-6, saturation='clip'):
    """Batch adapter over the same causal arrival reader used by the looks."""
    from dataclasses import fields
    B, W, D = words.shape
    active = torch.ones(B, W, device=words.device, dtype=torch.bool) if active is None else active
    if active.shape != (B, W) or (observed_prefix is not None and observed_prefix.shape != (B, W)):
        raise ValueError('word expectation masks must match word brackets')
    reader = WordArrivals(owner, bank, valid, gain=gain, epsilon=epsilon, saturation=saturation)
    if not W:
        C = bank.shape[1]
        return ExpectedWords(words.new_empty(B, 0, C), words, words, words.new_empty(B, 0),
            words.new_empty(B, 0, C, 2), words.new_empty(B, 0, C, 2), words.new_empty(B, 0),
            active, words.new_empty(B, 0), words.new_empty(B, 0, C, 2))
    forecast_count = W
    if not torch.is_grad_enabled() and not torch.compiler.is_compiling():
        arrived = active if observed_prefix is None else active & observed_prefix
        last = torch.where(arrived, torch.arange(W, device=words.device)[None]+1, 0).amax()
        forecast_count = min(W, int(last.detach().cpu()) + 1)
    readings = [reader.step(words[:, i], targets[:, i], active=active[:, i],
        arrived=None if observed_prefix is None else observed_prefix[:, i],
        lanes=None if lanes is None else lanes[:, i], forecast=i < forecast_count) for i in range(W)]
    return ExpectedWords(*(torch.cat([getattr(reading, field.name) for reading in readings], 1)
                           for field in fields(ExpectedWords)))
