import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

from Language import BinaryStructuredReductionLayer


class _AddOp(nn.Module):
    """Trivial binary op: y = left + right."""
    def forward(self, left, right):
        return left + right


class _MulOp(nn.Module):
    def forward(self, left, right):
        return left * right


def test_layer_forward_shapes():
    B, N, D = 2, 5, 4
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp(), _MulOp()], r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, routing = layer(x)
    assert hard.shape == (B, N, D)
    assert soft.shape == (B, N, D)
    assert routing["copy_mask"].shape == (B, N, 1)
    assert routing["reduce_mask"].shape == (B, N - 1, 2)
    assert routing["lengths"].shape == (B,)
    assert routing["copy_marginal"].shape == (B, N)
    assert routing["reduce_marginal"].shape == (B, N - 1)


def test_layer_gradient_into_op_and_anchors():
    """Anchor params receive gradient via the slab transformation: the
    hard route's masks (straight-through) and the soft op-mixture both
    flow back into copy_anchor and reduce_anchor.
    """
    B, N, D = 1, 4, 3
    op = _AddOp()
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[op], r_copy=1)
    x = torch.randn(B, N, D, requires_grad=True)
    hard, soft, routing = layer(x)
    loss = hard.sum() + soft.sum() + routing["marginal_slab"].sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert layer.copy_anchor.grad is not None and layer.copy_anchor.grad.abs().sum() > 0
    assert layer.reduce_anchor.grad is not None and layer.reduce_anchor.grad.abs().sum() > 0


def test_layer_keeps_soft_superposition_over_reduce_rules():
    """A tied reduce site keeps both candidate rules in the inside marginal.

    The Viterbi slab may harden to one rule for the forward value, but
    ``reduce_marginal_op`` and ``logZ`` retain the sum-product distribution,
    so gradient reaches every viable rule anchor.
    """
    layer = BinaryStructuredReductionLayer(
        d_model=1, ops=[_AddOp(), _MulOp()], r_copy=1)
    with torch.no_grad():
        layer.copy_anchor.zero_()
        layer.reduce_anchor.zero_()
    x = torch.ones(1, 2, 1, requires_grad=True)
    _hard, _soft, routing = layer(x)
    reduce_op_mass = routing["reduce_marginal_op"][0, 0]
    assert torch.all(reduce_op_mass > 0), reduce_op_mass
    routing["logZ"].sum().backward()
    grad_per_rule = layer.reduce_anchor.grad.abs().sum(dim=1)
    assert torch.all(grad_per_rule > 0), grad_per_rule


def test_layer_n_one_degenerate_pass_through():
    B, N, D = 2, 1, 3
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp()], r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, routing = layer(x)
    assert hard.shape == (B, N, D)
    # No reduction possible at N=1; the row pass-through is the input.
    assert torch.allclose(hard, x)
