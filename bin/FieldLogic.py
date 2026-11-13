"""Weighted least/greatest on evidence, shared by definitions and fields."""
import torch


def power(value, weight, *, conjunctive):
    """Qualify a membership without moving its exact zero and one rails."""
    value = value.clamp(0, 1)
    eps = torch.finfo(value.dtype).eps
    bounded = value.clamp_min(eps) if conjunctive else value.clamp_max(1-eps)
    charted = value + (bounded-value).detach()
    rail = value == (0 if conjunctive else 1)
    # A written exponent cannot move an exact Boolean rail. The numerical
    # log floor supplies a finite membership derivative, not fictitious
    # evidence that changing an exponent could turn zero into support.
    exponent = torch.where(rail & (weight > 0), weight.detach(), weight)
    result = ((exponent * charted.log()).exp() if conjunctive
              else -torch.expm1(exponent * torch.log1p(-charted)))
    exact = torch.where(rail & (weight > 0), 0. if conjunctive else 1., result)
    exact = torch.where(weight.detach() == 1, value, exact)
    return result + (exact-result).detach()


def reduce(poles, *, operation, dim=-2, weights=None, mask=None):
    """Reduce known operands; padding is absent, an observed zero is zero.

    The final axis is the for/against pair. A missing operand is excluded
    from both folds. Empty reductions withhold both lanes.
    """
    if poles.shape[-1] != 2:
        raise ValueError('field logic requires two evidence lanes')
    if operation == 'not':
        return poles.flip(-1)
    if operation == 'non':
        return torch.zeros_like(poles)
    if operation not in ('intersection', 'union'):
        raise ValueError('field operation must be intersection, union, not or non')
    axis = dim % poles.ndim
    if axis == poles.ndim-1:
        raise ValueError('cannot reduce the evidence lane axis')
    positive, negative = poles.unbind(-1)
    if weights is None:
        weights = torch.ones_like(positive)
    else:
        weights = torch.broadcast_to(torch.as_tensor(weights, device=poles.device, dtype=poles.dtype), positive.shape)
    present = weights > 0
    if mask is not None:
        present = present & torch.broadcast_to(mask, positive.shape)
    answers = []
    for value, conjunctive in ((positive, operation == 'intersection'),
                               (negative, operation == 'union')):
        qualified = power(value, weights, conjunctive=conjunctive)
        qualified = torch.where(present, qualified, 1. if conjunctive else 0.)
        result = qualified.amin(axis) if conjunctive else qualified.amax(axis)
        answers.append(torch.where(present.any(axis), result, 0.))
    return torch.stack(answers, -1)
