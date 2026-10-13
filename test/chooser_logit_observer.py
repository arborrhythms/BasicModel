"""Measure training chooser calls separately from policy-gradient ownership."""
import torch


def record_logits(ranges, key, logits):
    if not torch.is_grad_enabled():
        return
    observed = logits.detach()
    finite = observed[torch.isfinite(observed)]
    if not finite.numel():
        return
    row = ranges.setdefault(key, dict(minimum=float('inf'), maximum=-float('inf'),
        calls=0, gradient_calls=0, nan_count=0, positive_infinity_count=0))
    row['minimum'] = min(row['minimum'], float(finite.min()))
    row['maximum'] = max(row['maximum'], float(finite.max()))
    row['calls'] += 1
    row['gradient_calls'] += int(logits.requires_grad)
    row['nan_count'] += int(torch.isnan(observed).sum())
    row['positive_infinity_count'] += int(torch.isposinf(observed).sum())
