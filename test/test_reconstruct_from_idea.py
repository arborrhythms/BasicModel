"""<reconstructFromIdea>: erase parse traces, rebuild reverse rules from idea."""

import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name="MM_mereology_serial.xml"):
    import Language
    import Models
    from util import init_config

    path = os.path.join(_DATA, name)
    init_config(path=path, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model, _ = Models.BasicModel.from_config(path)
    return model


def _forward(model):
    import Models
    from util import TheXMLConfig

    Models.TheData.load(TheXMLConfig.get("data.dataset", default="xor"))
    loader = model.inputSpace.data.data_loader(split="train", num_streams=2)
    items, _ = next(iter(loader))
    x = model.inputSpace.prepInput(items)
    model.eval()
    with torch.no_grad():
        model.forward(x)
    return model


def _concept_dim(model):
    event = model.conceptualSpace.subspace.materialize()
    return int(event.shape[-1])


def test_reconstruct_from_idea_defaults_off():
    model = _build("MM_mereology.xml")
    assert getattr(model, "reconstruct_from_idea", None) is False


def test_clear_grammar_cache_erases_router_route():
    model = _forward(_build())
    ss = model.symbolSpace
    ll = ss.languageLayer

    ss.current_rules = {"SS": [[1]]}
    ss.generate_rules = {"SS": [[2]]}
    ss.recur_pass = 3
    ll._last_space_role_routings = {"SS": {"stale": True}}
    dim = _concept_dim(model)
    ll._last_output = torch.ones(1, 1, dim)
    ll._last_root_state = torch.ones(1, dim)

    ss.clear_grammar_cache()

    assert ss.current_rules == {}
    assert ss.generate_rules == {}
    assert ss.recur_pass == 0
    assert ss.routing_state.rules_by_space_role == {}
    assert ll._last_space_role_routings == {}
    assert ll._last_output is None
    assert ll._last_root_state is None


def test_reconstruct_from_idea_rebuilds_after_clearing():
    model = _forward(_build())
    ss = model.symbolSpace
    ss.current_rules = {"SS": [[1]]}
    ss.generate_rules = {"SS": [[2]]}
    if ss.languageLayer is not None:
        ss.languageLayer._last_space_role_routings = {"SS": {"stale": True}}

    fired = {"n": 0}
    orig = ss.generate

    def _wrapped(target, *args, **kwargs):
        fired["n"] += 1
        assert ss.current_rules == {}
        assert ss.generate_rules == {}
        if ss.languageLayer is not None:
            assert ss.languageLayer._last_space_role_routings == {}
        return orig(target, *args, **kwargs)

    ss.generate = _wrapped
    try:
        model.reconstruct_from_idea = True
        with torch.no_grad():
            model._chart_generate_from_stm(model.conceptualSpace.subspace)
    finally:
        ss.generate = orig

    assert fired["n"] >= 1
    assert isinstance(ss.generate_rules, dict)
    assert len(ss.generate_rules) > 0


def test_reconstruct_from_idea_can_use_reverse_seed_snapshot():
    model = _forward(_build())
    ss = model.symbolSpace
    seed = model.conceptualSpace.subspace
    seed_event = torch.randn(1, 2, _concept_dim(model))
    seed.set_event(seed_event)

    captured = {}
    orig = ss.generate

    def _stub(target, *args, **kwargs):
        captured["target"] = target.detach().clone()
        return {}

    ss.generate = _stub
    try:
        model.reconstruct_from_idea = True
        with torch.no_grad():
            model._chart_generate_from_stm(seed)
    finally:
        ss.generate = orig

    assert "target" in captured
    assert torch.equal(captured["target"], seed_event)
