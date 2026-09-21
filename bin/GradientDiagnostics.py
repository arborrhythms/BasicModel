"""Read-only objective agreement on named, optimizer-owned shared operators.

No projection, clipping or gradient replacement happens here. Sparse operator
gradients are compared over touched entries, without a dense capacity slab.
"""
from __future__ import annotations

import math

import torch


def _values(gradient):
    return gradient.coalesce().values() if gradient.is_sparse else gradient


def _norm(gradient):
    if gradient is None:
        return 0.0
    values = _values(gradient).detach()
    if not values.numel():
        return 0.0
    values = values.to(torch.float64 if values.dtype == torch.float64 else torch.float32)
    scale = float(values.abs().max())
    if not math.isfinite(scale):
        raise FloatingPointError("nonfinite operator gradient")
    return scale * float(torch.linalg.vector_norm(values / scale)) if scale else 0.0


def _dot_unit(left, right, left_norm, right_norm):
    """Dot product of unit gradients; never densify a sparse parameter."""
    if left is None or right is None or not left_norm or not right_norm:
        return 0.0
    left, right = left.detach(), right.detach()
    if left.shape != right.shape or left.device != right.device:
        raise ValueError("operator gradients must have matching shape and device")
    if left.is_sparse and right.is_sparse:
        left, right = left.coalesce(), right.coalesce()
        if left.sparse_dim() != right.sparse_dim():
            raise ValueError("sparse operator gradients must have matching dimensions")
        indices, inverse = torch.unique(
            torch.cat((left.indices(), right.indices()), dim=1), dim=1,
            sorted=True, return_inverse=True)
        a = left.values().new_zeros((indices.shape[1],) + left.values().shape[1:])
        b = torch.zeros_like(a)
        a.index_add_(0, inverse[:left._nnz()], left.values())
        b.index_add_(0, inverse[left._nnz():], right.values())
    elif left.is_sparse:
        left = left.coalesce()
        a, b = left.values(), right[tuple(left.indices())]
    elif right.is_sparse:
        right = right.coalesce()
        a, b = left[tuple(right.indices())], right.values()
    else:
        a, b = left, right
    dtype = torch.float64 if a.dtype == torch.float64 else torch.float32
    # Divide by a representable maximum first, then by the scaled norm.
    # Python doubles retain the total norm even when a float32 norm overflows.
    am = float(a.abs().max()) if a.numel() else 0.0
    bm = float(b.abs().max()) if b.numel() else 0.0
    if not am or not bm:
        return 0.0
    a = (a.to(dtype) / am) * (am / left_norm)
    b = (b.to(dtype) / bm) * (bm / right_norm)
    return float((a * b).sum())


def objective_agreement(objectives, groups):
    """Compare reconstruction with output/expectation at one parameter version.

    ``groups`` maps stable operator names to parameter sequences.
    Missing and zero gradients have no cosine (None), rather than agreement.
    ``autograd.grad`` leaves optimizer .grad buffers and parameters unchanged.
    """
    names = ("reconstruction", "output", "expectation")
    parameters = tuple(dict.fromkeys(
        p for group in groups.values() for p in group if p.requires_grad))
    positions = {id(p): i for i, p in enumerate(parameters)}
    gradients = {}
    for name in names:
        cost = objectives.get(name)
        gradients[name] = (torch.autograd.grad(
            cost, parameters, retain_graph=True, allow_unused=True)
            if parameters and torch.is_tensor(cost) and cost.requires_grad
            else (None,) * len(parameters))
    report = {}
    for name, group in groups.items():
        indices = tuple(dict.fromkeys(positions[id(p)] for p in group if id(p) in positions))
        if not indices:
            continue
        norms = {key: tuple(_norm(gradients[key][i]) for i in indices) for key in names}
        totals = {key: math.hypot(*values) for key, values in norms.items()}
        entry = {key + "_norm": totals[key] for key in names}
        for other in ("output", "expectation"):
            nr, no = totals["reconstruction"], totals[other]
            cosine = None
            if nr and no:
                cosine = sum(
                    _dot_unit(gradients["reconstruction"][i], gradients[other][i], a, b)
                    * (a / nr) * (b / no)
                    for i, a, b in zip(indices, norms["reconstruction"], norms[other]))
                cosine = max(-1.0, min(1.0, cosine))
            entry["reconstruction_" + other + "_cosine"] = cosine
        report[name] = entry
    return report


def record_opposition(report, history, *, persistence=3):
    """Name repeated opposition for the run log; never intervene in training."""
    for name, entry in report.items():
        persistent = []
        for other in ("output", "expectation"):
            key = (name, other)
            cosine = entry["reconstruction_" + other + "_cosine"]
            # An unused operator supplies no new agreement evidence.
            streak = history.get(key, 0)
            if cosine is not None:
                streak = streak + 1 if cosine < 0 else 0
            history[key] = streak
            entry[other + "_negative_streak"] = streak
            if streak >= persistence:
                persistent.append(other)
        entry["persistent_opposition"] = persistent
    return report
