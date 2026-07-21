"""MLPTransformChooser -- the contextual routing chooser.

A learned MLP over per-candidate context (slot/pair state, candidate
output, tool embedding, position) that produces the same per-(op, location)
logit shapes as the anchor-dot scorer, so it drops into the live structured
grammar layers. UNLIKE anchor-dot it owns parameters -> selecting it changes
the state_dict (a deliberate new basin), behind <transformChooser>mlp
(default anchordot).
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch
import torch.nn as nn

from Language import (
    TransformChooser, AnchorDotTransformChooser, MLPTransformChooser,
    make_transform_chooser, UnaryStructuredLayer,
    BinaryStructuredReductionLayer,
)


def test_is_a_transform_chooser():
    assert issubclass(MLPTransformChooser, TransformChooser)


def test_unary_score_shapes_match_the_contract():
    B, N, D, R_copy, R_apply = 2, 5, 4, 1, 3
    ch = MLPTransformChooser(d_model=D, n_copy=R_copy, n_op=R_apply)
    x = torch.randn(B, N, D)
    applied = torch.randn(B, N, R_apply, D)
    copy_score, apply_score = ch.score_unary(x, applied, None, None)
    assert copy_score.shape == (B, N, R_copy)
    assert apply_score.shape == (B, N, R_apply)


def test_binary_score_shapes_match_the_contract():
    B, N, D, R_copy, R_reduce = 2, 6, 4, 1, 3
    ch = MLPTransformChooser(d_model=D, n_copy=R_copy, n_op=R_reduce)
    x = torch.randn(B, N, D)
    reduced = torch.randn(B, N - 1, R_reduce, D)
    copy_score, reduce_score = ch.score_binary(x, reduced, None, None)
    assert copy_score.shape == (B, N, R_copy)
    assert reduce_score.shape == (B, N - 1, R_reduce)


def test_owns_params_and_scores_are_differentiable():
    ch = MLPTransformChooser(d_model=4, n_copy=1, n_op=2)
    assert sum(p.numel() for p in ch.parameters()) > 0
    x = torch.randn(1, 4, 4)
    reduced = torch.randn(1, 3, 2, 4)
    cs, rs = ch.score_binary(x, reduced, None, None)
    (cs.sum() + rs.sum()).backward()
    assert ch.tool_embedding.grad is not None
    assert any(p.grad is not None for p in ch.mlp.parameters())


def test_feeds_the_binary_router_layer():
    class _AddOp(nn.Module):
        def forward(self, left, right):
            return left + right

    layer = BinaryStructuredReductionLayer(
        d_model=4,
        ops=[_AddOp(), _AddOp(), _AddOp()],
        r_copy=1,
        chooser="mlp",
    )
    x = torch.randn(1, 5, 4)
    hard, soft, routing = layer(x)
    assert hard.shape == x.shape
    assert soft.shape == x.shape
    assert routing["reduce_score"].shape == (1, 4, 3)
    assert torch.isfinite(routing["reduce_score"]).all()


def test_degenerate_shapes():
    ch = MLPTransformChooser(d_model=4, n_copy=1, n_op=2)
    # N=1: no reduce pairs.
    cs, rs = ch.score_binary(torch.randn(1, 1, 4), torch.randn(1, 0, 2, 4),
                             None, None)
    assert cs.shape == (1, 1, 1) and rs.shape == (1, 0, 2)
    # R_apply = 0 (unary).
    cs2, as2 = ch.score_unary(torch.randn(1, 4, 4), torch.randn(1, 4, 0, 4),
                              None, None)
    assert as2.shape == (1, 4, 0)


def test_factory_selects_and_validates():
    assert isinstance(
        make_transform_chooser("anchordot", d_model=4, n_copy=1, n_op=2),
        AnchorDotTransformChooser)
    assert isinstance(
        make_transform_chooser("mlp", d_model=4, n_copy=1, n_op=2),
        MLPTransformChooser)
    with pytest.raises(ValueError):
        make_transform_chooser("bogus", d_model=4, n_copy=1, n_op=2)


def test_layers_select_chooser_and_state_dict_reflects_it():
    # Default (anchordot): no chooser params in the state_dict.
    u_ad = UnaryStructuredLayer(d_model=4, ops=[], r_copy=1)
    assert isinstance(u_ad.chooser, AnchorDotTransformChooser)
    assert not any("chooser" in k for k in u_ad.state_dict())

    # MLP: the chooser owns params -> they appear in the layer state_dict.
    class _AddOp(nn.Module):
        def forward(self, a, b):
            return a + b

    b_mlp = BinaryStructuredReductionLayer(
        d_model=4, ops=[_AddOp(), _AddOp()], r_copy=1, chooser="mlp")
    assert isinstance(b_mlp.chooser, MLPTransformChooser)
    keys = list(b_mlp.state_dict())
    assert any("chooser.tool_embedding" in k for k in keys)
    assert any("chooser.mlp" in k for k in keys)
    # The anchors still exist (the layer keeps them; the MLP ignores them).
    assert any("reduce_anchor" in k for k in keys)


# -- end-to-end: config -> router -> layers ------------------------------

import re
import tempfile
import warnings

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_PROJECT, "data")
_GRAMMAR_CONFIG = os.path.join(_DATA, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build_model(kind):
    import Models, Language
    from util import init_config, init_device
    init_device("cpu")
    torch.manual_seed(0)
    with open(_GRAMMAR_CONFIG) as f:
        text = f.read()
    text = re.sub(
        r"\s*<transformChooser>[^<]*</transformChooser>\s*\n", "\n", text)
    if kind is not None:
        inject = f"<transformChooser>{kind}</transformChooser>"
        text = text.replace("<architecture>", f"<architecture>\n    {inject}", 1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir=_DATA)
    tmp.write(text)
    tmp.close()
    try:
        init_config(path=tmp.name, defaults_path=_DEFAULTS)
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(tmp.name)
        m.eval()
        return m
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _router_choosers(m):
    ll = m.symbolSpace.languageLayer
    layers = list(ll._unary_layers.values()) + list(ll._binary_layers.values())
    assert layers, "no structured layers attached"
    return [type(l.chooser).__name__ for l in layers]


def test_config_mlp_builds_mlp_choosers_in_the_router():
    m = _build_model("mlp")
    names = _router_choosers(m)
    assert all(n == "MLPTransformChooser" for n in names), names


def test_config_default_keeps_anchordot_choosers():
    m = _build_model(None)        # no <transformChooser> -> default anchordot
    names = _router_choosers(m)
    assert all(n == "AnchorDotTransformChooser" for n in names), names


def test_config_mlp_router_parameters_have_one_optimizer_owner():
    """Router Parameters are enlisted after grammar-op ModuleDict wiring."""
    m = _build_model("mlp")
    optimizer = m.getOptimizer(lr=1e-3)
    optimizer_params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    router = m.symbolSpace.languageLayer
    router_params = [p for p in router.parameters() if p.requires_grad]

    assert router_params, "the configured MLP router must own Parameters"
    for parameter in router_params:
        assert sum(candidate is parameter
                   for candidate in optimizer_params) == 1
