"""MLPTransformChooser -- the chooser of acts, and the only one.

A learned MLP over per-candidate context (slot/pair state, candidate
output, tool embedding, position) that gives the per-(op, location) logits
of the structured grammar layers.
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
    MLPTransformChooser, make_transform_chooser, OperationSelectionLayer,
)


def test_the_linear_chooser_and_its_key_are_gone():
    import Language
    assert not hasattr(Language, 'AnchorDotTransformChooser')
    assert not hasattr(Language, 'TransformChooser')
    assert not hasattr(Language, 'SelectedThoughtChooser')


def test_unary_score_shapes_match_the_contract():
    B, N, D, R_copy, R_apply = 2, 5, 4, 1, 3
    ch = MLPTransformChooser(d_model=D, n_copy=R_copy, n_op=R_apply)
    x = torch.randn(B, N, D)
    applied = torch.randn(B, N, R_apply, D)
    copy_score, apply_score = ch.score_unary(x, applied)
    assert copy_score.shape == (B, N, R_copy)
    assert apply_score.shape == (B, N, R_apply)


def test_binary_score_shapes_match_the_contract():
    B, N, D, R_copy, R_reduce = 2, 6, 4, 1, 3
    ch = MLPTransformChooser(d_model=D, n_copy=R_copy, n_op=R_reduce)
    x = torch.randn(B, N, D)
    reduced = torch.randn(B, N - 1, R_reduce, D)
    copy_score, reduce_score = ch.score_binary(x, reduced)
    assert copy_score.shape == (B, N, R_copy)
    assert reduce_score.shape == (B, N - 1, R_reduce)


def test_owns_params_and_scores_are_differentiable():
    ch = MLPTransformChooser(d_model=4, n_copy=1, n_op=2)
    assert sum(p.numel() for p in ch.parameters()) > 0
    x = torch.randn(1, 4, 4)
    reduced = torch.randn(1, 3, 2, 4)
    cs, rs = ch.score_binary(x, reduced)
    (cs.sum() + rs.sum()).backward()
    assert ch.tool_embedding.grad is not None
    assert any(p.grad is not None for p in ch.mlp.parameters())


def test_feeds_the_binary_router_layer():
    class _AddOp(nn.Module):
        def forward(self, left, right):
            return left + right

    layer = OperationSelectionLayer(
        d_model=4,
        ops=[_AddOp(), _AddOp(), _AddOp()],
    )
    x = torch.randn(1, 5, 4)
    hard, soft, routing = layer(x)
    assert hard.shape == x.shape
    assert soft.shape == x.shape
    assert routing["binary_probabilities"].shape == (1, 4, 3)
    assert torch.isfinite(routing["binary_probabilities"]).all()


def test_degenerate_shapes():
    ch = MLPTransformChooser(d_model=4, n_copy=1, n_op=2)
    # N=1: no reduce pairs.
    cs, rs = ch.score_binary(torch.randn(1, 1, 4), torch.randn(1, 0, 2, 4))
    assert cs.shape == (1, 1, 1) and rs.shape == (1, 0, 2)
    # R_apply = 0 (unary).
    cs2, as2 = ch.score_unary(torch.randn(1, 4, 4), torch.randn(1, 4, 0, 4))
    assert as2.shape == (1, 4, 0)


def test_factory_builds_the_network():
    assert isinstance(make_transform_chooser(d_model=4, n_copy=1, n_op=2),
                      MLPTransformChooser)


def test_layers_hold_the_network_and_no_anchor():
    # The chooser owns params -> they appear in the layer state_dict.
    class _AddOp(nn.Module):
        def forward(self, a, b):
            return a + b

    layer = OperationSelectionLayer(d_model=4, ops=[_AddOp(), _AddOp()])
    assert isinstance(layer.chooser, MLPTransformChooser)
    keys = list(layer.state_dict())
    assert any("chooser.tool_embedding" in k for k in keys)
    assert any("chooser.mlp" in k for k in keys)
    assert not any(k.endswith(("stop_anchor", "reduce_anchor", "apply_anchor", "bracket_anchor"))
                   for k in keys)


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
    with open(_GRAMMAR_CONFIG) as f:
        text = f.read()
    text = re.sub(
        r"\s*<transformChooser>[^<]*</transformChooser>\s*\n", "\n", text)
    if kind is not None:
        inject = f"<transformChooser>{kind}</transformChooser>"
        text = text.replace("<architecture>", f"<architecture>\n    {inject}", 1)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False)
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
    layers = [ll.operation_layer]
    assert layers, "no structured layers attached"
    return [type(l.chooser).__name__ for l in layers]


def test_every_configuration_builds_the_network_in_the_router():
    m = _build_model(None)
    names = _router_choosers(m)
    assert all(n == "MLPTransformChooser" for n in names), names


def test_the_retired_key_is_refused():
    with pytest.raises(ValueError, match='transformChooser'):
        _build_model("mlp")


def test_config_mlp_router_parameters_have_one_optimizer_owner():
    """Router Parameters are enlisted after grammar-op ModuleDict wiring."""
    m = _build_model(None)
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
