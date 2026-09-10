"""Configurable chooser capacity without changing the default learned basin.

These are architecture/mechanism tests, not evidence of learned multistep
reasoning. Most exercise tiny standalone heads; checkpoint materialization
uses the real BasicModel methods without constructing a data pipeline.
"""

import os
import sys
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(_ROOT / "bin"))

import pytest
import torch
from torch import nn

import Language
from Language import MLPTransformChooser, WhatStepChooser, make_transform_chooser
from util import XMLConfig


def _linears(module):
    return [layer for layer in module.mlp if isinstance(layer, nn.Linear)]


def _assert_same_state(actual, expected):
    actual, expected = actual.state_dict(), expected.state_dict()
    assert list(actual) == list(expected)
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


@pytest.mark.parametrize("d_model", [4, 12])
def test_default_grammar_head_preserves_legacy_weights_rng_and_outputs(d_model):
    """Pin the pre-capacity-knob construction, not another new constructor."""
    torch.manual_seed(127)
    legacy = nn.Module()
    legacy.tool_embedding = nn.Parameter(torch.randn(3, 8) * 0.02)
    hidden = max(8, d_model)
    legacy.mlp = nn.Sequential(
        nn.Linear(2 * d_model + 16, hidden), nn.GELU(), nn.Linear(hidden, 1))
    with torch.random.fork_rng(devices=[]):
        legacy.what_projection = nn.Linear(29, 3, bias=False)
    nn.init.zeros_(legacy.what_projection.weight)
    expected_rng = torch.random.get_rng_state().clone()

    torch.manual_seed(127)
    chooser = MLPTransformChooser(d_model=d_model, n_copy=1, n_op=2)
    _assert_same_state(chooser, legacy)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)

    x = torch.randn(2, 3, d_model)
    candidates = torch.randn(2, 3, 2, d_model)
    copy_score, op_score = chooser.score_unary(x, candidates, None, None)
    positions = chooser._pos_emb(3, x.device, x.dtype)
    for count, values, tools, actual in (
        (1, x.unsqueeze(2), legacy.tool_embedding[:1], copy_score),
        (2, candidates, legacy.tool_embedding[1:], op_score),
    ):
        features = torch.cat([
            x.unsqueeze(2).expand(2, 3, count, d_model),
            values,
            tools.view(1, 1, count, 8).expand(2, 3, count, 8),
            positions.view(1, 3, 1, 8).expand(2, 3, count, 8),
        ], dim=-1)
        assert torch.equal(actual, legacy.mlp(features).squeeze(-1))


def test_default_step_head_preserves_legacy_weights_rng_and_answer_tie():
    torch.manual_seed(153)
    rng_before = torch.random.get_rng_state().clone()
    legacy = nn.Module()
    with torch.random.fork_rng(devices=[]):
        legacy.mlp = nn.Sequential(nn.Linear(35, 16), nn.GELU(), nn.Linear(16, 1))
    nn.init.zeros_(legacy.mlp[-1].weight)
    nn.init.zeros_(legacy.mlp[-1].bias)

    chooser = WhatStepChooser()
    _assert_same_state(chooser, legacy)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    candidates = [{"kind": "answer", "active": True},
                  {"kind": "open", "position": 0.5}]
    context = torch.randn(29)
    assert torch.equal(chooser.logits(context, candidates), torch.zeros(2))
    choice, log_prob = chooser.choose(context, candidates)
    assert choice == 0
    assert torch.allclose(log_prob, -torch.tensor(2.0).log())


@pytest.mark.parametrize("depth", [1, 3])
def test_custom_grammar_capacity_shapes_and_all_hidden_gradients(depth):
    torch.manual_seed(207)
    chooser = MLPTransformChooser(
        d_model=4, n_copy=1, n_op=2, hidden=13, depth=depth, n_role_cats=3)
    layers = _linears(chooser)
    assert len(layers) == depth + 1
    assert [(layer.in_features, layer.out_features) for layer in layers] == (
        [(27, 13)] + [(13, 13)] * (depth - 1) + [(13, 1)])
    assert sum(isinstance(layer, nn.GELU) for layer in chooser.mlp) == depth

    x = torch.randn(2, 4, 4, requires_grad=True)
    cats = torch.randn(2, 4, 3, requires_grad=True)
    context = torch.randn(2, 29, requires_grad=True)
    unary = chooser.score_unary(
        x, torch.randn(2, 4, 2, 4), None, None, cat_ctx=cats, what_ctx=context)
    binary = chooser.score_binary(
        x, torch.randn(2, 3, 2, 4), None, None, cat_ctx=cats, what_ctx=context)
    assert [tuple(score.shape) for score in unary + binary] == [
        (2, 4, 1), (2, 4, 2), (2, 4, 1), (2, 3, 2)]
    sum(score.square().mean() + score.mean() for score in unary + binary).backward()
    for name, parameter in chooser.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for value in (x, cats):
        assert value.grad is not None and value.grad.abs().sum() > 0


@pytest.mark.parametrize("depth", [1, 3])
def test_custom_step_capacity_gradients_after_output_head_learns(depth):
    torch.manual_seed(251)
    chooser = WhatStepChooser(context_dim=7, hidden=11, depth=depth)
    layers = _linears(chooser)
    assert [(layer.in_features, layer.out_features) for layer in layers] == (
        [(13, 11)] + [(11, 11)] * (depth - 1) + [(11, 1)])
    assert sum(isinstance(layer, nn.GELU) for layer in chooser.mlp) == depth
    assert torch.count_nonzero(layers[-1].weight) == 0
    assert torch.count_nonzero(layers[-1].bias) == 0
    candidates = [{"kind": "answer", "active": True},
                  {"kind": "open", "position": 0.5},
                  {"kind": "open", "position": 1.0, "answered": True}]
    context = torch.randn(7, requires_grad=True)
    logits = chooser.logits(context, candidates, pressure=0.4)
    assert logits.shape == (3,) and torch.count_nonzero(logits) == 0
    # Zero final weights deliberately block the initial hidden-state gradient.
    # Once that head changes, the complete configured depth must receive credit.
    with torch.no_grad():
        layers[-1].weight.fill_(0.1)
    logits = chooser.logits(context, candidates, pressure=0.4)
    (logits.square().mean() + logits.mean()).backward()
    for name, parameter in chooser.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert context.grad is not None and context.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["grammar", "step"])
@pytest.mark.parametrize("parameter,value", [
    ("hidden", 0), ("hidden", -1), ("depth", 0), ("depth", -1),
    ("hidden", True), ("hidden", 1.5), ("depth", True), ("depth", 1.5),
])
def test_direct_heads_reject_nonpositive_capacity(kind, parameter, value):
    constructor, kwargs = ((MLPTransformChooser, {"d_model": 4, "n_copy": 1, "n_op": 2})
                           if kind == "grammar" else (WhatStepChooser, {}))
    with pytest.raises(ValueError):
        constructor(**kwargs, **{parameter: value})


@pytest.mark.parametrize("hidden,depth,expected_hidden", [(0, 1, 8), (19, 3, 19)])
def test_xml_grammar_factory_wires_width_depth(monkeypatch, tmp_path,
                                              hidden, depth, expected_hidden):
    config_path = tmp_path / "chooser.xml"
    config_path.write_text(
        "<model><architecture>"
        f"<transformChooserHidden>{hidden}</transformChooserHidden>"
        f"<transformChooserDepth>{depth}</transformChooserDepth>"
        "</architecture></model>")
    config = XMLConfig(path=str(config_path), defaults_path=str(_ROOT / "data/model.xml"))
    monkeypatch.setattr(Language, "TheXMLConfig", config)
    chooser = make_transform_chooser("mlp", d_model=4, n_copy=1, n_op=2)
    layers = _linears(chooser)
    assert len(layers) == depth + 1
    assert layers[0].out_features == expected_hidden
    assert layers[-1].in_features == expected_hidden
    assert not list(make_transform_chooser(
        "anchordot", d_model=4, n_copy=1, n_op=2).parameters())


def test_canonical_xml_defaults_pin_existing_architectures():
    config = XMLConfig(defaults_path=str(_ROOT / "data/model.xml"))
    for name, expected in {
        "transformChooserHidden": 0, "transformChooserDepth": 1,
        "whatThinkingHidden": 16, "whatThinkingDepth": 1,
    }.items():
        assert config.get(f"architecture.{name}") == expected


@pytest.mark.parametrize("name,bad_value", [
    ("transformChooserHidden", -1), ("transformChooserDepth", 0),
    ("whatThinkingHidden", 0), ("whatThinkingDepth", 0),
])
def test_xml_schema_rejects_invalid_capacity(tmp_path, name, bad_value):
    xsd_path = str(_ROOT / "data/model.xsd")
    good_result = XMLConfig._run_schema_validation(xsd_path, str(_ROOT / "data/model.xml"))
    if good_result is None:
        pytest.skip("No XML schema validation backend is available")
    assert good_result == ""
    document = ET.parse(_ROOT / "data/model.xml")
    element = document.find(f"architecture/{name}")
    assert element is not None
    element.text = str(bad_value)
    path = tmp_path / "invalid_capacity.xml"
    document.write(path)
    result = XMLConfig._run_schema_validation(xsd_path, str(path))
    assert result is not None and result != ""


def test_real_model_xml_wires_both_chooser_architectures(monkeypatch, tmp_path):
    """One small construction checks the config-to-model path, with no training."""
    import Models
    from data import TheData
    from util import TheXMLConfig, init_device

    # Config loading mutates the shared object; give this test disposable data.
    for name in ("_data", "_sources", "_requirements"):
        monkeypatch.setattr(TheXMLConfig, name, copy.deepcopy(getattr(TheXMLConfig, name)))
    document = ET.parse(_ROOT / "data/MM_xor.xml")
    architecture = document.find("architecture")
    for name, value in {
        "transformChooser": "mlp", "transformChooserHidden": 19,
        "transformChooserDepth": 2, "whatThinkingHidden": 23,
        "whatThinkingDepth": 3,
    }.items():
        element = architecture.find(name)
        if element is None:
            element = ET.SubElement(architecture, name)
        element.text = str(value)
    path = tmp_path / "chooser_model.xml"
    document.write(path)
    init_device("cpu")
    Language.TheGrammar._configured = False
    TheData.load("xor")
    model, _ = Models.BaseModel.from_config(str(path), data=TheData)
    assert model.transform_chooser_hidden == 19
    assert model.transform_chooser_depth == 2
    assert model.what_thinking_hidden == 23
    assert model.what_thinking_depth == 3
    grammar_heads = [module for module in model.modules()
                     if isinstance(module, MLPTransformChooser)]
    assert grammar_heads
    for chooser in grammar_heads:
        assert chooser.hidden == 19 and chooser.depth == 2
    step = model._what_step_chooser()
    assert step.hidden == 23 and step.depth == 3


def _model_shell(**capacity):
    from Models import BasicModel

    model = BasicModel()
    model.register_parameter("device_anchor", nn.Parameter(torch.ones(1)))
    for name, value in capacity.items():
        setattr(model, name, value)
    return model


def test_model_lazy_step_head_uses_configured_capacity():
    model = _model_shell(what_thinking_hidden=23, what_thinking_depth=3)
    chooser = model._what_step_chooser(device=torch.device("cpu"), dtype=torch.float64)
    assert len(_linears(chooser)) == 4
    assert _linears(chooser)[0].out_features == 23
    assert next(chooser.parameters()).dtype == torch.float64
    assert model._what_step_chooser() is chooser


def test_deep_step_checkpoint_materializes_saved_architecture_and_loads_strictly():
    """The absent-head restore path infers depth as well as first-layer width."""
    torch.manual_seed(331)
    source = _model_shell(what_thinking_hidden=17, what_thinking_depth=3)
    chooser = source._what_step_chooser()
    with torch.no_grad():
        chooser.mlp[-1].weight.fill_(0.2)
        chooser.mlp[-1].bias.fill_(0.1)
    saved = {key: value.detach().clone() for key, value in source.state_dict().items()}

    restored = _model_shell()
    assert getattr(restored, "what_step_chooser", None) is None
    assert restored._materialize_answer_path_from_checkpoint(saved) == 1
    restored.load_state_dict(saved, strict=True)
    _assert_same_state(restored, source)
    assert len(_linears(restored.what_step_chooser)) == 4
    assert _linears(restored.what_step_chooser)[0].out_features == 17
    candidates = [{"kind": "answer", "active": True}, {"kind": "open"}]
    context = torch.randn(29)
    assert torch.equal(chooser.logits(context, candidates),
                       restored.what_step_chooser.logits(context, candidates))
