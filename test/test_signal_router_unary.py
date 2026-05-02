import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
import torch.nn as nn

from SignalRouter import UnaryStructuredLayer


class _NegateOp(nn.Module):
    def forward(self, x):
        return -x


class _AbsOp(nn.Module):
    def forward(self, x):
        return x.abs()


def test_unary_layer_output_shape_unchanged():
    B, N, D = 2, 5, 4
    layer = UnaryStructuredLayer(d_model=D, ops=[_NegateOp(), _AbsOp()],
                                 r_copy=1)
    x = torch.randn(B, N, D)
    hard, soft, routing = layer(x)
    assert hard.shape == (B, N, D)
    assert soft.shape == (B, N, D)
    # action axis = R_copy + R_apply = 1 + 2 = 3
    assert routing["action_logits"].shape == (B, N, 3)
    assert routing["action_probs"].shape == (B, N, 3)
    assert torch.allclose(routing["action_probs"].sum(-1),
                          torch.ones(B, N), atol=1e-5)


def test_unary_layer_hard_one_hot_per_position():
    B, N, D = 1, 4, 3
    layer = UnaryStructuredLayer(d_model=D, ops=[_NegateOp()], r_copy=1)
    x = torch.randn(B, N, D)
    _, _, routing = layer(x)
    cm = routing["copy_mask"]      # [B, N, R_copy]
    am = routing["apply_mask"]     # [B, N, R_apply]
    fired = cm.sum(-1) + am.sum(-1)
    assert torch.all(fired == 1.0)


def test_unary_layer_gradient_into_op_and_input():
    B, N, D = 1, 4, 3
    op = _NegateOp()
    layer = UnaryStructuredLayer(d_model=D, ops=[op], r_copy=1)
    x = torch.randn(B, N, D, requires_grad=True)
    hard, soft, _ = layer(x)
    (hard.sum() + soft.sum()).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    # Anchors carry the placement signal; check their gradients.
    assert layer.copy_anchor.grad is not None and layer.copy_anchor.grad.abs().sum() > 0
    if layer.r_apply > 0:
        assert layer.apply_anchor.grad is not None and layer.apply_anchor.grad.abs().sum() > 0
