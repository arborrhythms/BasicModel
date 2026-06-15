"""Soft-superposition parser route (doc/Language.md weighted deduction).

Under the two-pass <learning> mode the structured layers drop the
hard-Viterbi + straight-through forward and propagate the PURE sum-product
superposition at a temperature (0 = sharp/deterministic, 1 = flat). The
chooser is then in the gradient path directly -- no argmax, no detach. With
``superposition_temperature`` unset (default) the legacy straight-through
forward is unchanged (covered by the existing signal-router tests).
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import torch.nn as nn

from Language import (
    BinaryStructuredReductionLayer, UnaryStructuredLayer, superposition_scale,
)


class _AddOp(nn.Module):
    def forward(self, a, b=None):
        return a if b is None else a + b


class _MulOp(nn.Module):
    def forward(self, a, b=None):
        return a if b is None else a * b


class _ScaleOp(nn.Module):
    """Unary op that actually TRANSFORMS its input (k*a) so the apply branch
    differs from the copy branch -- otherwise op(x)==x makes apply_anchor's
    true gradient exactly zero and the differentiability test passes only on
    RNG luck. Binary call (b given) stays a real pairwise op."""

    def __init__(self, k):
        super().__init__()
        self.k = float(k)

    def forward(self, a, b=None):
        return self.k * a if b is None else self.k * (a + b)


def test_superposition_scale_endpoints():
    assert superposition_scale(0.0) == 1.0     # sharp: scores pass through
    assert superposition_scale(1.0) == 0.0     # flat: scores zeroed -> uniform
    assert abs(superposition_scale(0.5) - 0.5) < 1e-9


def test_binary_soft_superposition_is_differentiable_into_chooser():
    # The whole point: the soft slab trains the chooser DIRECTLY (no
    # straight-through), so the chooser is in the gradient path.
    layer = BinaryStructuredReductionLayer(
        d_model=4, ops=[_AddOp(), _MulOp()], r_copy=1)
    layer.superposition_temperature = 0.0
    x = torch.randn(1, 6, 4)
    _hard, soft, _routing = layer(x)
    soft.sum().backward()
    assert layer.reduce_anchor.grad is not None
    assert layer.reduce_anchor.grad.abs().sum() > 0
    assert layer.copy_anchor.grad is not None


def test_unary_soft_superposition_is_differentiable_into_chooser():
    # Ops must transform x in the unary (b=None) call: with op(x)==x the apply
    # branch equals copy and apply_anchor's gradient is genuinely 0 (the soft
    # blend is x regardless of routing). Seed for determinism.
    torch.manual_seed(0)
    layer = UnaryStructuredLayer(
        d_model=4, ops=[_ScaleOp(2.0), _ScaleOp(-1.0)], r_copy=1)
    layer.superposition_temperature = 0.0
    x = torch.randn(1, 5, 4)
    _hard, soft, _routing = layer(x)
    soft.sum().backward()
    assert layer.apply_anchor.grad is not None
    assert layer.apply_anchor.grad.abs().sum() > 0


def test_temperature_one_is_flatter_than_zero():
    # The op superposition at temp 1 (flat) must be closer to uniform than at
    # temp 0 (the chooser's own softmax).
    torch.manual_seed(0)
    layer = BinaryStructuredReductionLayer(
        d_model=4, ops=[_AddOp(), _MulOp()], r_copy=1)
    # Push the anchors so the chooser has a clear preference at temp 0.
    with torch.no_grad():
        layer.reduce_anchor.mul_(20.0)
    x = torch.randn(1, 6, 4)
    layer.superposition_temperature = 0.0
    r0 = layer(x)[2]["reduce_marginal_op"]
    layer.superposition_temperature = 1.0
    r1 = layer(x)[2]["reduce_marginal_op"]
    # Entropy of the per-pair op distribution: temp 1 should be higher (flatter).
    def _op_entropy(rm):
        p = rm / rm.sum(-1, keepdim=True).clamp_min(1e-9)
        return -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
    assert _op_entropy(r1) >= _op_entropy(r0) - 1e-6


def test_default_none_keeps_straight_through_path():
    # No superposition_temperature attr -> getattr default None -> legacy
    # path; forward still runs and is differentiable (straight-through).
    layer = BinaryStructuredReductionLayer(
        d_model=4, ops=[_AddOp(), _MulOp()], r_copy=1)
    assert getattr(layer, "superposition_temperature", None) is None
    x = torch.randn(1, 5, 4, requires_grad=True)
    _hard, soft, _routing = layer(x)
    soft.sum().backward()
    assert x.grad is not None
