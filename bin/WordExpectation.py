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


def word_distribution(owner, words, bank, valid, targets, *, gain=1., observed_prefix=None, active=None, lanes=None,
                      epsilon=1e-6, saturation='clip'):
    """Forecast before each arrival, then consume only actual admitted input.

    Candidate identities remain native rows. Forecasts are independent two-lane
    activations; the normalized for-lane distribution is a reporting/readout
    view, not a categorical constraint on evidence or a feedback source.
    """
    words, bank = words.detach(), bank.detach()
    B, W, D = words.shape
    C = bank.shape[1]
    if bank.shape != (B, C, D) or D != owner.word_dim or not C:
        raise ValueError('word expectation needs a nonempty native-width bank')
    active = torch.ones(B, W, device=words.device, dtype=torch.bool) if active is None else active
    if active.shape != (B, W) or (observed_prefix is not None and observed_prefix.shape != (B, W)):
        raise ValueError('word expectation masks must match word brackets')
    arrived = active if observed_prefix is None else active & observed_prefix
    source = words.new_zeros(B, C, 2)
    total, count, recent = words.new_zeros(B, D), words.new_zeros(B, 1), words.new_zeros(B, D)
    # The word read begins with the already observed field/STM. Incoming
    # words join this context only after their own causal prediction.
    for row in range(B):
        chain = owner._inter_context[row] if row < len(owner._inter_context) else ()
        if not chain:
            continue
        values = torch.stack([value.to(words) for _, value, _ in chain]).flatten(0, 1)
        occupied = torch.stack([mask.to(words.device) for _, _, mask in chain]).flatten()
        lane_history = owner._inter_context_lanes[row]
        pairs = torch.stack([words.new_tensor((1.,0.) if pair is None else pair).expand(3,2)
            for pair in lane_history]).flatten(0,1)
        source[row] = symbol_activations(bank[row:row+1],values[None],occupied[None],pairs[None])[0]
        total[row] = (values * occupied[:,None]).sum(0)
        count[row] = occupied.sum()
        positions = torch.where(occupied,torch.arange(len(values),device=words.device)+1,0)
        recent[row] = values[(positions.amax()-1).clamp_min(0)]
    probabilities, predictions, fields, losses, tops, ranks, estimates = [], [], [], [], [], [], []
    forecast_count = W
    if W and not torch.is_grad_enabled() and not torch.compiler.is_compiling():
        # One host read before the loop, never one synchronization per word.
        # After the final arrival one forecast serves the unchanged padding.
        last = torch.where(arrived, torch.arange(W, device=words.device)[None]+1, 0).amax()
        forecast_count = min(W, int(last.detach().cpu()) + 1)
    for index in range(W):
        context = torch.cat((total/count.clamp_min(1), recent), -1)
        if index < forecast_count:
            logits = owner.word_predictor(bank, context, source)
        expected = logits.sigmoid() * valid[..., None]
        target = targets[:, index].clamp(0, C-1)
        observed = ((targets[:, index] >= 0) & (targets[:, index] < C)
                    & valid.gather(1, target[:, None])[:, 0] & active[:, index])
        pair = torch.stack((observed.to(words), words.new_zeros(B)), -1) if lanes is None else lanes[:, index].detach()
        actual = words.new_zeros(B, C, 2).scatter(1, target[:, None, None].expand(-1, 1, 2),
                                                (pair * observed[:, None])[:, None])
        field = gain_field(actual, expected, strength=gain, attention=valid,
                           epsilon=epsilon, saturation=saturation)
        mass = expected[..., 0]
        probability = mass / mass.sum(-1, keepdim=True).clamp_min(epsilon)
        probabilities.append(probability)
        predictions.append(torch.einsum('bc,bcd->bd', probability, bank))
        fields.append(field)
        estimates.append(expected)
        loss = (actual - expected).square().sum((1, 2)) / (2*valid.sum(-1)).clamp_min(1)
        losses.append(torch.where(observed, loss, 0.))
        tops.append((probability.argmax(-1) == target) & observed)
        rank = 1. / (1. + (probability > probability.gather(1, target[:, None])).sum(-1))
        ranks.append(torch.where(observed, rank, 0.))
        # An observed prefix need not be a labelled candidate in this read's
        # bank (e.g. a held-out completion). It still supplies factual STM.
        seen = arrived[:, index]
        total = total + words[:, index] * seen[:, None]
        count = count + seen[:, None]
        recent = torch.where(seen[:, None], words[:, index], recent)
        distances = torch.cdist(words[:, index:index+1], bank, compute_mode='donot_use_mm_for_euclid_dist')[:, 0]
        matched = (distances <= 1e-5) & valid
        source_pair = (torch.stack((seen.to(words), words.new_zeros(B)), -1)
                       if lanes is None else lanes[:, index].detach())
        source_actual = matched[..., None].to(words) * source_pair[:, None]
        source = torch.where(seen[:, None, None], torch.maximum(source, source_actual), source)
    if not W:
        return ExpectedWords(words.new_empty(B, 0, C), words, words, words.new_empty(B, 0),
            words.new_empty(B, 0, C, 2), words.new_empty(B, 0, C, 2), words.new_empty(B, 0),
            active, words.new_empty(B, 0), words.new_empty(B, 0, C, 2))
    gains = torch.stack([field.gain for field in fields], 1).gather(2, targets.clamp(0, C-1)[..., None])[..., 0]
    # An admitted observation outside this prior's bank has no forecast. Its
    # confidence is zero, so it passes whole; only inactive brackets are zero.
    known = (targets >= 0) & (targets < C) & valid.gather(1, targets.clamp(0, C-1))
    gains = torch.where(active, torch.where(known, gains, 1.), 0.)
    return ExpectedWords(torch.stack(probabilities, 1), torch.stack(predictions, 1),
        words * gains[..., None], gains, torch.stack([f.excess for f in fields], 1),
        torch.stack([f.omission for f in fields], 1), torch.stack(losses, 1),
        torch.stack(tops, 1), torch.stack(ranks, 1), torch.stack(estimates, 1))
