"""Action time scales and realized optimizer displacement receipts."""
import math
import torch


def rate_groups(parameters, language, ratio, lr):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('actionRateRatio must be finite and in (0, 1]')
    lift, adverb = (None, None) if language is None else (language.verb_action(), language.modifier_action())
    ids = {p: power for action, power in ((lift, 1), (adverb, 2)) if action is not None for p in action.parameters()}
    return [dict(params=group, lr=lr * ratio**power, action_rate_power=power)
            for power in (0, 1, 2) if (group := [p for p in parameters if ids.get(p, 0) == power])]


def snapshot(model):
    from GradientProjection import units
    language = getattr(model, 'languageSpace', None)
    if language is None:
        return {}
    groups = {name: tuple(action.parameters()) for name, action in
              (('lift', language.verb_action()), ('adverb', language.modifier_action())) if action is not None}
    groups['codes'] = tuple(p for p, (_, kind, _) in units(model).items() if kind == 'row')
    result = {}
    for name, parameters in groups.items():
        rows = []
        for p in parameters:
            if p.grad is None:
                continue
            indices = p.grad.coalesce().indices()[0] if p.grad.is_sparse else None
            before = (p if indices is None else p[indices]).detach().clone()
            rows.append((p, indices, before))
        result[name] = rows
    return result


def record(model, before):
    epoch = model.__dict__.setdefault('_realized_steps_epoch', {})
    for name, rows in before.items():
        if not rows:
            continue
        change, size = 0., 0.
        for p, indices, old in rows:
            value = p.detach() if indices is None else p.detach()[indices]
            # Receipts accumulate in host float64; MPS has no float64 dtype.
            change += float((value - old).cpu().double().square().sum())
            size += float(old.cpu().double().square().sum())
        stats = epoch.setdefault(name, dict(delta_square=0., parameter_square=0., steps=0))
        stats['delta_square'] += change
        stats['parameter_square'] += size
        stats['steps'] += 1


def receipt(model):
    return {name: dict(stats, relative_step=math.sqrt(stats['delta_square']/stats['parameter_square'])
                      if stats['parameter_square'] else None,
                      convention='root sum squared realized steps / root sum squared pre-step parameters')
            for name, stats in getattr(model, '_realized_steps_epoch', {}).items()}
