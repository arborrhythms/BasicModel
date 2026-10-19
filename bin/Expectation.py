"""Local activation prediction and gain: identity is never the error signal.

Predictor sources/targets are detached; only its local residual trains it.
Content, expected activation, and excess/omission are separate consumers.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def unit(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0. <= value <= 1.:
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return value


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0.:
        raise ValueError(f'{name} must be finite and positive')
    return value


@dataclass(frozen=True)
class GainField:
    observed: torch.Tensor
    expected: torch.Tensor
    content: torch.Tensor
    gain: torch.Tensor
    excess: torch.Tensor
    omission: torch.Tensor

    @property
    def surprise(self):
        return (self.excess + self.omission).square().mean(-1)


def gain_field(observed, expected, *, strength=1., attention=1., confidence=1.,
               exempt=0., withheld=0., epsilon=1e-6, saturation='clip'):
    """Last dimension is evidence lanes; one scalar gain per identity.

    The evidence poles and residual signs are independent axes. ``exempt=1``
    spares an open object or withheld property; it never erases its content.
    Deep callers must supply the region's domain for residual reporting too.
    """
    strength = unit(strength, 'expectation gain')
    epsilon = positive(epsilon, 'expectation epsilon')
    if saturation not in ('clip', 'smooth'):
        raise ValueError('expectation saturation must be clip or smooth')
    expected = expected.to(observed)
    attention = torch.as_tensor(attention, device=observed.device, dtype=observed.dtype)
    exempt = torch.as_tensor(exempt, device=observed.device, dtype=observed.dtype).detach()
    confidence = torch.as_tensor(confidence, device=observed.device, dtype=observed.dtype).detach()
    withheld = torch.as_tensor(withheld, device=observed.device, dtype=observed.dtype).detach()
    for value in (attention, exempt, confidence, withheld):
        torch._assert_async((torch.isfinite(value) & (value >= 0) & (value <= 1)).all(),
                            'attention, confidence and exemptions must be in [0, 1]')
    # Deep expectations have no domain outside the admitted region. A withheld
    # dimension has no prior, including no expected prime or omission.
    expected = expected * (1. - withheld[..., None]) * (attention > 0)[..., None]
    residual = (observed - expected) * (attention > 0)[..., None]
    excess, omission = residual.relu(), (-residual).relu()
    magnitude = (excess + omission).sum(-1)
    denominator = expected.abs().sum(-1) + epsilon
    s = ((magnitude / denominator).clamp(max=1.) if saturation == 'clip'
         else magnitude / (denominator + magnitude))
    # Cut the denominator too. Detaching only r leaves a predictor exploit.
    s = s.detach()
    gain = 1. - strength * (1. - exempt) * confidence * (1. - s)
    gain = gain * attention
    content = gain[..., None] * observed
    # These are content evidence coordinates, not signed identity-key codes.
    torch._assert_async((torch.isfinite(content) & (content >= 0)).all(),
                        'expectation content coordinates must be finite and nonnegative')
    return GainField(observed, expected, content, gain, excess, omission)


def retention_surprise(residual, expected):
    """Bounded MSE over the soft union of observed and expected support.

    Confident omissions count. Truly empty bank padding has zero weight.
    Both evidence lanes contribute under one weight for each symbol.
    """
    residual, expected = residual.detach(), expected.detach()
    actual = (residual + expected).clamp_min(0.)
    support = torch.maximum(actual, expected).amax(-1)
    error = (residual.square().mean(-1) * support).sum() / support.sum().clamp_min(1e-6)
    return error / (1. + error)


class PerceptMoments(nn.Module):
    """Field-wide moments; read before update, commit external evidence once.

    Axes before ``width`` are observations, not independent predictor owners.
    No current target is used to form its own gain. Padding is excluded.
    """
    def __init__(self, width, *, rate=.05, epsilon=1e-6, saturation='clip'):
        super().__init__()
        self.rate = positive(rate, 'percept expectation rate')
        if self.rate > 1.:
            raise ValueError('percept expectation rate must be at most one')
        self.epsilon = positive(epsilon, 'expectation epsilon')
        if saturation not in ('clip', 'smooth'):
            raise ValueError('expectation saturation must be clip or smooth')
        self.saturation = saturation
        self.register_buffer('mean', torch.zeros(int(width)))
        self.register_buffer('variance', torch.ones(int(width)))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))

    def migrate_expectation_checkpoint(self, state_dict, prefix=''):
        added = False
        for name in ('mean', 'variance', 'count'):
            if prefix+name not in state_dict:
                state_dict[prefix+name] = getattr(self, name).detach().clone()
                added = True
        return added

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.migrate_expectation_checkpoint(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def residual(self, observed, *, prior=None):
        if observed.shape[-1] != self.mean.numel():
            raise ValueError('percept statistic width differs from its field')
        mean, variance, count = ((self.mean, self.variance, self.count) if prior is None else prior)
        raw = (observed.detach() - mean.to(observed)).abs()
        scale = variance.to(observed).clamp_min(0.).sqrt() + self.epsilon
        z = ((raw / scale).clamp(max=1.) if self.saturation == 'clip'
             else raw / (scale + raw))
        # A cold coordinate has no basis for attenuation.
        return torch.where(count.to(observed.device) > 0, z, torch.ones_like(z))

    def region_gain(self, observed, valid, *, strength, prior=None, groups=None, regions=None):
        z = self.residual(observed, prior=prior)
        if groups is None:
            mask = valid.to(z)[..., None]
            mass = mask.sum((-2, -1)) * z.shape[-1]
            total = (z * mask).sum((-2, -1))
        else:
            if regions is None or groups.shape != valid.shape:
                raise ValueError('percept regions require aligned bracket membership')
            selected = valid & (groups >= 0) & (groups < regions)
            where = groups.clamp(0, regions-1)
            total = z.new_zeros(len(z), regions).scatter_add(1, where, z.mean(-1)*selected)
            mass = z.new_zeros(len(z), regions).scatter_add(1, where, selected.to(z))
        pooled = torch.where(mass > 0, total/mass.clamp_min(1.), 1.)
        strength = unit(strength, 'percept expectation gain')
        return (1. - strength) + strength * pooled

    @torch.no_grad()
    def observe(self, observed, valid):
        selected = observed.detach()[valid]
        if not selected.numel():
            return
        mean = selected.mean(0).to(self.mean)
        variance = selected.var(0, unbiased=False).to(self.variance)
        delta = mean - self.mean
        rate = self.rate if int(self.count) else 1.
        self.variance.mul_(1. - rate).add_(rate * variance + rate * (1. - rate) * delta.square())
        self.mean.add_(rate * delta)
        self.count.add_(len(selected))


class ActivationPredictor(nn.Module):
    """Shared weights over native symbols of every order, in bounded chunks.

    There is no slot-structured output and no bank-sized parameter matrix.
    A key's two sigmoid outputs are independent activations, not a simplex.
    Context is the observed symbolic field and STM; inputs are always cut.
    """
    def __init__(self, width, hidden=64):
        super().__init__()
        self.width = int(width)
        hidden = int(hidden)
        if hidden < 1:
            raise ValueError('activation predictor hidden width must be positive')
        self.context = nn.Sequential(nn.Linear(4 * self.width, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 2 * self.width))
        self.local = nn.Linear(4, 2)

    def forward(self, keys, context, activation=None):
        keys, context = keys.detach(), context.detach()
        if keys.shape[-1] != self.width or context.shape != (keys.shape[0], 2 * self.width):
            raise ValueError('activation predictor requires native keys and field/STM context')
        keys = F.normalize(keys, dim=-1)
        activation = (keys.new_zeros(*keys.shape[:2], 2) if activation is None else activation.detach())
        if activation.shape != (*keys.shape[:2], 2):
            raise ValueError('symbol activation must retain both evidence lanes')
        # Both evidence lanes across the whole symbolic field are top-down
        # context. A negative source is not interchangeable with its omission.
        field = torch.einsum('bnl,bnd->bld', activation, keys)
        field = field / activation.sum(1).clamp_min(1.)[..., None]
        query = self.context(torch.cat((context, field.flatten(1)), -1)).reshape(keys.shape[0], 2, self.width)
        scores = []
        for start in range(0, keys.shape[1], 1024):
            key = keys[:, start:start + 1024]
            similarity = torch.einsum('bnd,bld->bnl', key, query)
            local = self.local(torch.cat((activation[:, start:start + 1024], similarity), -1))
            scores.append(local)
        return torch.cat(scores, 1) if scores else keys.new_zeros(keys.shape[0], 0, 2) + query.sum() * 0.

    @staticmethod
    def context_from(values, occupied=None):
        """Pool actual activations; chronological current state stays separate."""
        values = values.detach()
        if occupied is None:
            occupied = values.abs().sum(-1) > 0
        mask = occupied.detach().to(values)[..., None]
        flat, weight = values.flatten(1, -2), mask.flatten(1, -2)
        if flat.shape[1] == 0:
            return values.new_zeros(values.shape[0], 2 * values.shape[-1])
        mean = (flat * weight).sum(1) / weight.sum(1).clamp_min(1.)
        # Last occupied source is factual STM, not a recursive forecast.
        positions = torch.arange(flat.shape[1], device=values.device)[None]
        last = torch.where(weight[..., 0] > 0, positions + 1, 0).amax(-1)
        recent = flat.gather(1, (last - 1).clamp_min(0)[:, None, None].expand(-1, 1, flat.shape[-1]))[:, 0]
        recent = torch.where(last[:, None] > 0, recent, 0.)
        return torch.cat((mean, recent), -1)


def alternative_probabilities(logits, *, fraction, temperature):
    """A positive uniform floor over precisely the eligible alternatives."""
    fraction = unit(fraction, 'exploration uniform fraction')
    if fraction == 0.:
        raise ValueError('exploration uniform fraction must be positive')
    temperature = positive(temperature, 'exploration temperature')
    valid = torch.isfinite(logits)
    safe = (logits / temperature).masked_fill(~valid, -torch.inf)
    safe = torch.where(valid.any(-1, keepdim=True), safe, 0.)
    policy = torch.where(valid, safe.softmax(-1), 0.)
    uniform = valid.to(logits) / valid.sum(-1, keepdim=True).clamp_min(1)
    return fraction * uniform + (1. - fraction) * policy


def sample_alternative(probabilities, draw):
    """Inverse CDF with no holes at zero or a rounded-down terminal mass."""
    cumulative = probabilities.cumsum(-1)
    cumulative = cumulative / cumulative[:, -1:].clamp_min(torch.finfo(probabilities.dtype).tiny)
    return (draw >= cumulative).sum(-1).clamp_max(probabilities.shape[-1]-1)


def symbol_activations(keys, values, occupied, lanes=None, *, identities=None, bank_ids=None, key_valid=None):
    """Identify raw observations before applying gain; return two native lanes.

    Native handles identify activated rows. Without a handle, only exact
    native keys (distance <= 1e-5) are observations of the bank.
    Unknown composed values never become targets for their nearest key.
    """
    keys, values = keys.detach(), values.detach()
    B, N, _ = keys.shape
    if not N:
        return values.new_zeros(B, 0, 2)
    indices, matched = match_symbols(keys, values, identities=identities, bank_ids=bank_ids, key_valid=key_valid)
    if lanes is None:
        lanes = torch.stack((occupied.to(values), torch.zeros_like(occupied).to(values)), -1)
    source = torch.where((occupied & matched)[..., None], lanes.detach().to(values), 0.)
    return values.new_zeros(B, N, 2).scatter_reduce(
        1, indices[..., None].expand(-1, -1, 2), source, reduce='amax', include_self=True)


def match_symbols(keys, values, *, identities=None, bank_ids=None, key_valid=None):
    """Native handles identify activated codes; unhandled values require exact keys.

    A missing named handle is unknown, never its nearest neighbour. Direct
    distances avoid cancellation in the 1e-5 fallback for composed values.
    """
    distances = torch.cdist(values.detach(), keys.detach(),
        compute_mode='donot_use_mm_for_euclid_dist')
    if key_valid is not None:
        distances = distances.masked_fill(~key_valid[:, None, :], torch.inf)
    distance, index = distances.min(-1)
    matched = distance <= 1e-5
    if identities is not None:
        if bank_ids is None:
            raise ValueError('native handles require the bank address domain')
        ids = bank_ids[None] if bank_ids.ndim == 1 else bank_ids
        equal = (identities[..., None] == ids[:, None, :]) & (ids[:, None, :] != -1)
        named = identities != -1
        index = torch.where(named, equal.long().argmax(-1), index)
        matched = torch.where(named, equal.any(-1), matched)
    return index, matched


@dataclass(frozen=True)
class SymbolRead:
    keys: torch.Tensor
    expected: torch.Tensor
    strength: float
    epsilon: float
    saturation: str
    identities: object = None
    observed: object = None
    observed_mask: object = None

    def gains(self, values, occupied=None, lanes=None, *, identities=None):
        """A content view of exact raw identities, never a write to their carrier."""
        if self.strength == 0.:
            return values.new_ones(values.shape[:-1])
        index, matched = match_symbols(self.keys, values, identities=identities, bank_ids=self.identities)
        expected = self.expected.detach().gather(1, index[..., None].expand(-1,-1,2))
        if occupied is None:
            occupied = torch.ones_like(matched)
        actual = (torch.stack((occupied.to(values), torch.zeros_like(occupied).to(values)), -1)
                  if lanes is None else lanes.detach().to(values))
        if lanes is None and self.observed is not None:
            known = self.observed_mask.gather(1, index)
            factual = self.observed.gather(1, index[..., None].expand(-1, -1, 2))
            actual = torch.where(known[..., None], factual, actual)
        field = gain_field(actual, expected, strength=self.strength,
            epsilon=self.epsilon, saturation=self.saturation)
        return torch.where(matched & occupied, field.gain, 1.)
