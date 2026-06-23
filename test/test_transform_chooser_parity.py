"""TransformChooser parity for the structured grammar layers.

The structured layers delegate placement scoring to a TransformChooser. The
first chooser
(AnchorDotTransformChooser) must reproduce the original inline anchor-dot
scoring EXACTLY -- this is the behavior-preserving foundation before any
more expressive (MLP) chooser. These tests pin that parity: the chooser's
logits equal the raw einsums, and the layers expose a stateless chooser
that adds no state_dict keys (so the pinned basin cannot move).
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

import Language
from Language import (
    TransformChooser, AnchorDotTransformChooser,
    UnaryStructuredLayer, BinaryStructuredReductionLayer,
)


def test_anchordot_is_a_transform_chooser():
    assert issubclass(AnchorDotTransformChooser, TransformChooser)


def test_base_chooser_is_abstract():
    base = TransformChooser()
    try:
        base.score_unary(None, None, None, None)
        assert False, "base score_unary must raise NotImplementedError"
    except NotImplementedError:
        pass


def test_unary_score_matches_raw_einsum():
    torch.manual_seed(0)
    B, N, D, R_copy, R_apply = 2, 5, 4, 1, 3
    x_score = torch.randn(B, N, D)
    applied_score = torch.randn(B, N, R_apply, D)
    copy_anchor = torch.randn(R_copy, D)
    apply_anchor = torch.randn(R_apply, D)

    ch = AnchorDotTransformChooser()
    copy_score, apply_score = ch.score_unary(
        x_score, applied_score, copy_anchor, apply_anchor)

    ref_copy = torch.einsum('bnd,cd->bnc', x_score, copy_anchor)
    ref_apply = torch.einsum('bnad,ad->bna', applied_score, apply_anchor)
    assert torch.equal(copy_score, ref_copy)
    assert torch.equal(apply_score, ref_apply)


def test_unary_score_zero_apply_ops():
    ch = AnchorDotTransformChooser()
    x_score = torch.randn(2, 4, 4)
    applied_score = torch.randn(2, 4, 0, 4)
    copy_anchor = torch.randn(1, 4)
    apply_anchor = torch.randn(0, 4)
    copy_score, apply_score = ch.score_unary(
        x_score, applied_score, copy_anchor, apply_anchor)
    assert apply_score.shape == (2, 4, 0)


def test_binary_score_matches_raw_einsum():
    torch.manual_seed(1)
    B, N, D, R_copy, R_reduce = 2, 6, 4, 1, 3
    x_score = torch.randn(B, N, D)
    reduced_score = torch.randn(B, N - 1, R_reduce, D)
    copy_anchor = torch.randn(R_copy, D)
    reduce_anchor = torch.randn(R_reduce, D)

    ch = AnchorDotTransformChooser()
    copy_score, reduce_score = ch.score_binary(
        x_score, reduced_score, copy_anchor, reduce_anchor)

    ref_copy = torch.einsum('bnd,cd->bnc', x_score, copy_anchor)
    ref_reduce = torch.einsum('bnrd,rd->bnr', reduced_score, reduce_anchor)
    assert torch.equal(copy_score, ref_copy)
    assert torch.equal(reduce_score, ref_reduce)


def test_binary_score_no_pairs():
    ch = AnchorDotTransformChooser()
    x_score = torch.randn(2, 1, 4)               # N=1 -> no adjacent pairs
    reduced_score = torch.randn(2, 0, 3, 4)
    copy_anchor = torch.randn(1, 4)
    reduce_anchor = torch.randn(3, 4)
    _, reduce_score = ch.score_binary(
        x_score, reduced_score, copy_anchor, reduce_anchor)
    assert reduce_score.shape == (2, 0, 3)


def test_layers_use_a_stateless_chooser_no_state_dict_keys():
    u = UnaryStructuredLayer(d_model=4, ops=[], r_copy=1)
    b = BinaryStructuredReductionLayer(d_model=4, ops=[], r_copy=1)
    for layer in (u, b):
        assert isinstance(layer.chooser, AnchorDotTransformChooser)
        keys = list(layer.state_dict().keys())
        assert not any("chooser" in k for k in keys), (
            f"chooser must not add state_dict keys (got {keys})")
        # the anchors remain owned by the layer
        assert any("anchor" in k for k in keys), keys


def test_anchordot_unary_uses_category_role_prior(monkeypatch):
    monkeypatch.setattr(
        Language, "compute_role_vocabulary",
        lambda _grammar: (["negate_I1"], {"negate_I1": 0}, 1))

    class _Neg(nn.Module):
        def forward(self, x):
            return x

    layer = UnaryStructuredLayer(
        d_model=2, ops=[_Neg()], r_copy=1,
        chooser="anchordot", op_names=["negate"])
    with torch.no_grad():
        layer.copy_anchor.zero_()
        layer.apply_anchor.zero_()
    x = torch.zeros(1, 2, 2)
    cat_ctx = torch.tensor([[[4.0], [0.0]]])
    _hard, _soft, routing = layer(x, cat_ctx=cat_ctx)
    expected = torch.tensor([[[0.0, 4.0], [0.0, 0.0]]],
                            dtype=routing["action_logits"].dtype)
    assert torch.equal(
        routing["action_logits"], expected)


def test_anchordot_binary_uses_labelled_left_right_category_prior(monkeypatch):
    monkeypatch.setattr(
        Language, "compute_role_vocabulary",
        lambda _grammar: (
            ["lift_I1", "lift_I2"],
            {"lift_I1": 0, "lift_I2": 1},
            2,
        ))

    class _Add(nn.Module):
        def forward(self, left, right):
            return left + right

    layer = BinaryStructuredReductionLayer(
        d_model=2, ops=[_Add(), _Add()], r_copy=1,
        chooser="anchordot", op_names=["lift", "other"])
    with torch.no_grad():
        layer.copy_anchor.zero_()
        layer.reduce_anchor.zero_()
    x = torch.zeros(1, 3, 2)
    cat_ctx = torch.tensor([[
        [2.0, 0.0],
        [0.0, 4.0],
        [0.0, 0.0],
    ]])
    _hard, _soft, routing = layer(x, cat_ctx=cat_ctx)
    expected = torch.tensor([[[3.0, 0.0], [0.0, 0.0]]],
                            dtype=routing["reduce_score"].dtype)
    assert torch.equal(routing["reduce_score"], expected)
