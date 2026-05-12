"""Anchor-based scoring tests.

The placement scorer was deleted in favor of per-rule learnable anchors
(BinaryStructuredReductionLayer.{copy_anchor, reduce_anchor};
UnaryStructuredLayer.{copy_anchor, apply_anchor}). The score is a
plain inner product between the rule's output and its anchor. These
tests assert the anchor parameters exist, have the right shape, and
receive gradient via the slab transformation.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

from Language import (
    BinaryStructuredReductionLayer,
    UnaryStructuredLayer,
)


class _AddOp(nn.Module):
    def forward(self, left, right):
        return left + right


class _NegOp(nn.Module):
    def forward(self, x):
        return -x


def test_binary_anchor_shapes():
    D, r_copy, r_reduce = 8, 3, 4
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp() for _ in range(r_reduce)], r_copy=r_copy)
    assert layer.copy_anchor.shape == (r_copy, D)
    assert layer.reduce_anchor.shape == (r_reduce, D)


def test_unary_anchor_shapes():
    D, r_copy = 6, 2
    layer = UnaryStructuredLayer(
        d_model=D, ops=[_NegOp(), _NegOp(), _NegOp()], r_copy=r_copy)
    assert layer.copy_anchor.shape == (r_copy, D)
    assert layer.apply_anchor.shape == (3, D)


def test_binary_anchor_receives_gradient():
    B, N, D = 1, 4, 5
    layer = BinaryStructuredReductionLayer(
        d_model=D, ops=[_AddOp(), _AddOp()], r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, routing = layer(x)
    (hard.sum() + soft.sum() + routing["marginal_slab"].sum()).backward()
    assert layer.copy_anchor.grad is not None
    assert layer.copy_anchor.grad.abs().sum() > 0
    assert layer.reduce_anchor.grad is not None
    assert layer.reduce_anchor.grad.abs().sum() > 0


def test_unary_anchor_receives_gradient():
    B, N, D = 1, 4, 4
    layer = UnaryStructuredLayer(
        d_model=D, ops=[_NegOp()], r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, _ = layer(x)
    (hard.sum() + soft.sum()).backward()
    assert layer.copy_anchor.grad is not None
    assert layer.copy_anchor.grad.abs().sum() > 0
    assert layer.apply_anchor.grad is not None
    assert layer.apply_anchor.grad.abs().sum() > 0
