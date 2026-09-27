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


_OBJECTIVES = ("reconstruction", "output", "expectation")


def _sum_gradients(left, right):
    if left is None:
        return right
    if right is None:
        return left
    # PyTorch supports dense + sparse, but not sparse + dense.
    total = right + left if left.is_sparse and not right.is_sparse else left + right
    return total.coalesce() if total.is_sparse else total


def accumulate_objective_gradients(destination, objectives, groups, *, gradient_fn=None):
    """Save weighted gradient vectors before a seal's training graph is freed.

    Sum vectors in each parameter's coordinates, not norms or cosines. Values
    are detached snapshots; sparse gradients retain only their touched entries.
    The optional gradient reader includes the cached perception's pullback.
    Neither this collection nor the reader changes optimizer .grad buffers.
    """
    parameters = tuple(dict.fromkeys(
        p for group in groups.values() for p in group if p.requires_grad))
    if not parameters:
        return destination
    for name in _OBJECTIVES:
        cost = objectives.get(name)
        if not torch.is_tensor(cost) or not cost.requires_grad:
            continue
        gradients = (gradient_fn(cost, parameters) if gradient_fn is not None else
            torch.autograd.grad(cost, parameters, retain_graph=True, allow_unused=True))
        saved = destination.setdefault(name, {})
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                saved[parameter] = _sum_gradients(saved.get(parameter), gradient.detach().clone())
    return destination


def objective_agreement(objectives, groups, *, accumulated=None):
    """Compare weighted reconstruction with output/expectation gradients.

    ``groups`` maps stable operator names to parameter sequences.
    Missing and zero gradients have no cosine (None), rather than agreement.
    ``autograd.grad`` leaves optimizer .grad buffers and parameters unchanged.
    When supplied, ``accumulated`` contains actual gradients from earlier seal
    updates. Their sum describes the batch's training directions across those
    versions; it is not a gradient re-evaluated at the batch-end parameters.
    """
    names = _OBJECTIVES
    gradients = {name: dict((accumulated or {}).get(name, {})) for name in names}
    accumulate_objective_gradients(gradients, objectives, groups)
    report = {}
    for name, group in groups.items():
        parameters = tuple(dict.fromkeys(p for p in group if p.requires_grad))
        if not parameters:
            continue
        norms = {key: tuple(_norm(gradients[key].get(p)) for p in parameters) for key in names}
        totals = {key: math.hypot(*values) for key, values in norms.items()}
        entry = {key + "_norm": totals[key] for key in names}
        for other in ("output", "expectation"):
            nr, no = totals["reconstruction"], totals[other]
            cosine = None
            if nr and no:
                cosine = sum(
                    _dot_unit(gradients["reconstruction"].get(p), gradients[other].get(p), a, b)
                    * (a / nr) * (b / no)
                    for p, a, b in zip(parameters, norms["reconstruction"], norms[other]))
                cosine = max(-1.0, min(1.0, cosine))
            entry["reconstruction_" + other + "_cosine"] = cosine
            # Orientation is explicit: >1 means the other weighted objective
            # is larger. A zero reference norm has no finite comparison.
            entry[other + "_reconstruction_norm_ratio"] = no / nr if nr else None
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
