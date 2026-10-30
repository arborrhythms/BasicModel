"""The actual admitted reconstruction, unattended residual, and traversal cost.

Observation targets are immutable. This module supplies no replacement
backward: the reconstructed content and support lanes receive
the ordinary derivative of the numerical objective reported here.
"""
import math

import torch


def validate_floor(floor):
    if not math.isfinite(floor) or not 0. < floor < 1.:
        raise ValueError('attentionFloor must be finite and strictly between zero and one')


def tolerate_heterogeneity(lanes, tolerance):
    """Remove the configured overlap, preserving the two meaning lanes."""
    if not math.isfinite(tolerance) or not 0. <= tolerance <= 1.:
        raise ValueError('hetTolerance must be finite and between zero and one')
    if lanes.shape[-1] != 2:
        raise ValueError('heterogeneity requires paired meaning lanes')
    if tolerance == 1.:
        return lanes
    return lanes - (1. - tolerance) * lanes.amin(-1, keepdim=True)


def field_cost(observed, reconstructed, admission, valid, *, floor,
               defined=None, iterations=None, step_cost=0., expected=None):
    """Return per-row terms, summing every located item exactly once.

    ``admission`` is the hard union of candidates already read.
    ``reconstructed`` is the inverse along the recorded compose derivation,
    aligned to the admitted source supports after inversion.
    ``defined`` is the inverse's coordinate domain: undefined coordinates
    have no expectation and contribute no inside surprise (kappa = 0).
    The outside residual is the actual observation; expectation never
    substitutes evidence in this reconstruction judge.
    """
    validate_floor(floor)
    if (observed.ndim < 3 or reconstructed.shape != observed.shape
            or admission.shape != observed.shape[:2] or valid.shape != admission.shape
            or valid.dtype != torch.bool):
        raise ValueError('whole-field costs require aligned observations, reconstruction and admission')
    target = observed.detach()
    residual = target.abs().flatten(2).sum(-1)
    if defined is not None and (defined.shape != target.shape or defined.dtype != torch.bool):
        raise ValueError('inverse definedness must be a boolean mask over field coordinates')
    difference = target - reconstructed
    if defined is not None:
        difference = torch.where(defined, difference, 0.)
    if expected is None:
        error = difference.abs().flatten(2).sum(-1)
    else:
        from Expectation import percept_reconstruction
        width = expected.shape[-1]
        _, residual_target, _, squared = percept_reconstruction(
            reconstructed[...,:width], target[...,:width], expected)
        if defined is not None:
            squared = torch.where(defined[...,:width], squared, 0.)
        error = squared.sum(-1) + difference[...,width:].abs().flatten(2).sum(-1)
        residual = residual_target.square().sum(-1) + target[...,width:].abs().flatten(2).sum(-1)
    inside = torch.where(valid, admission * error, 0.).sum(-1)
    outside = torch.where(valid, floor * (1. - admission) * residual, 0.).sum(-1)
    work = (torch.zeros_like(inside) if iterations is None else
            iterations.to(inside) * step_cost)
    return dict(inside=inside, outside=outside, work=work, total=inside + outside + work)
