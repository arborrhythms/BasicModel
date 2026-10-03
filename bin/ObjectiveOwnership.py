"""One objective writes each optimizer parameter; no gradient projection."""
import torch
from GradientDiagnostics import _sum_gradients


PRIMARY = ('reconstruction', 'expectation', 'output')


def registry_costs(registry, *, reader_rows=None):
    """Read trained terms from Error, retaining lesson and penalty boundaries."""
    result = {}
    for name, term in registry._terms.items():
        if (not term.get('trained', True) or term['weight'] == 0
                or term['category'] in registry._disabled):
            continue
        category = term['category']
        objective = term.get('objective', category)
        if category == 'grammar':
            owners = ('generate_lesson' if 'generate.' in name else 'compose_lesson',)
        elif category == 'reg':
            # A regularizer writes only the owned parameters it actually uses.
            owners = tuple('penalty.' + owner for owner in PRIMARY)
        elif objective in ('expectation', 'intra', 'inter', 'discourse'):
            owners = ('expectation',)
        elif objective in ('output', 'prediction', 'policy'):
            owners = ('output',)
        else:
            owners = ('reconstruction',)
        raw = term['weight'] * term.get('multiplier', 1.) * registry._value(term)
        for owner in owners:
            value = raw
            if reader_rows is not None and owner in ('output', 'penalty.output'):
                # A discarded trial supplies no Adam step, including no
                # momentum-only update from a fabricated zero gradient.
                if not bool(reader_rows.any()):
                    continue
                if value.ndim:
                    value = value * reader_rows.to(value)
                elif not bool(reader_rows.all()):
                    raise ValueError('a trial reader cost must retain its batch rows')
            if value.ndim:
                active = registry.row_mask
                value = (value.mean() if active is None else
                         (value * active.to(value)).sum() / active.sum().clamp_min(1))
            result[owner] = result.get(owner, 0) + value
    return result


def backward_owned(costs, owners, *, pullback=None, scale=1., audit=None):
    """Accumulate exact objective gradients only into that objective's owners.

    The perception pullback is traversed solely for reconstruction. Shared
    operator hosts remain differentiable for generation's reader but never
    receive its cotangent. Saved trial graphs remain alive until their caller
    releases the registry, so both trials use their pre-update forward values.
    """
    for name, cost in costs.items():
        if not torch.is_tensor(cost) or not cost.requires_grad:
            continue
        key = name.removeprefix('penalty.')
        parameters = tuple(owners.get(key, ()))
        if not parameters:
            continue
        objective = ('reconstruction' if key in ('compose_lesson', 'generate_lesson') else key)
        value = cost * scale
        gradients = (pullback.gradients(value, parameters)
                     if pullback is not None and objective == 'reconstruction'
                     else torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True))
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                continue
            parameter.grad = _sum_gradients(parameter.grad, gradient.detach())
            if audit is not None:
                writers = audit.setdefault(id(parameter), set())
                writers.add(objective)
                if len(writers) != 1:
                    raise RuntimeError('an optimizer parameter has more than one objective writer')
