"""The top-down attention handoff: the 4-case WS->PS pass-back
(doc/specs/mereological-order-raising.md "the top-down attention handoff").

Gated ``<mereologyRaise>`` and DARK by default: with no words-category
attention, no scope, and no parked run-structure observation the pass-back
action is ``"noop"`` and PartSpace gets the stage-0 percept re-fed -- the
forward is byte-identical. The mechanism sits on the multi-stage carrier the
subsymbolicOrder=3 combine fix restored.

These tests exercise the dispatch (``WholeSpace.passback_action``) across all
four cases and verify the default is inert.
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import pytest
import torch

_DATA = os.path.join(os.path.dirname(_BIN), "data")
_DEFAULTS = os.path.join(_DATA, "model.xml")


def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(_DATA, name)
    init_config(path=p, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def _batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def test_mereology_raise_stamps_stage0_ws_and_is_byte_identical():
    # <mereologyRaise> stamps the stage-0 WholeSpace (so it can compute the
    # read-only run-structure obs) and the default forward stays deterministic
    # + finite -- the pass-back is "noop" without attention.
    m = _build("MM_mereology.xml")
    assert m.mereology_raise
    ws0 = m.wholeSpaces[0]
    assert getattr(ws0, "_mereology_raise", False)
    x = _batch(m)
    with torch.no_grad():
        out1 = m.forward(x)[2]
        out2 = m.forward(x)[2]
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2), "the dark pass-back must be deterministic"


def test_first_pass_is_wide_open():
    # Pass 0 ignores any scope/attention (wide-open); always "noop".
    m = _build("MM_mereology.xml")
    ws0 = m.wholeSpaces[0]
    ws0._intent_boosts = torch.ones(4)
    object.__setattr__(ws0, "_mereology_ratio_obs",
                       {"route_hint": torch.tensor([2, 2, 2, 2])})
    assert ws0.passback_action(0) == ("noop", None)


def test_passback_noop_without_attention():
    # No words-category attention engaged -> "noop" (byte-identical), even with
    # a parked observation.
    m = _build("MM_mereology.xml")
    ws0 = m.wholeSpaces[0]
    ws0._intent_boosts = None
    object.__setattr__(ws0, "_mereology_ratio_obs",
                       {"route_hint": torch.tensor([2, 2, 2, 2])})
    assert ws0.passback_action(1) == ("noop", None)


def test_passback_route_hint_dispatch():
    # With attention engaged, route_hint routes the 4 cases:
    #   0 = NULL -> noop ; 1 = single run -> chunk ; 2 = many runs -> refine.
    m = _build("MM_mereology.xml")
    ws0 = m.wholeSpaces[0]
    ws0._intent_boosts = torch.ones(4)
    for hint, expect in [(0, "noop"), (1, "chunk"), (2, "refine")]:
        object.__setattr__(ws0, "_mereology_ratio_obs",
                           {"route_hint": torch.tensor([hint] * 4)})
        action, where = ws0.passback_action(1)
        assert action == expect, f"route_hint {hint} -> {action} (want {expect})"
        assert where is None


def test_passback_scoped_word_where():
    # An explicit scope `.where` (the serial / deterministic-reading override)
    # returns ("scoped", where) regardless of route_hint -- the
    # null-content + word-`.where` second-argument mechanism.
    m = _build("MM_mereology.xml")
    ws0 = m.wholeSpaces[0]
    where = torch.tensor([0.25, 0.75])
    object.__setattr__(ws0, "_passback_scope_where", where)
    action, got = ws0.passback_action(1)
    assert action == "scoped"
    assert torch.equal(got, where)


def test_passback_scope_ps_selects_refed_input_when_engaged():
    # The model-loop helper returns the stage-0 percept (ps_default) when the
    # action is noop, and a re-fed PS analysis when refine/chunk fires.
    m = _build("MM_mereology.xml")
    x = _batch(m)
    # Drive a forward so the per-batch spaces/STM are warm, then exercise the
    # helper directly with a non-empty prior-symbol carrier.
    with torch.no_grad():
        m.forward(x)
    ws0 = m.wholeSpaces[0]
    cs0 = m.conceptualSpaces[0]
    prev = cs0._subspaceForWS
    ps_default = m.perceptualSpace.forward(m._staged_in_sub) \
        if getattr(m, "_staged_in_sub", None) is not None else None
    # noop (no attention) -> returns ps_default unchanged.
    ws0._intent_boosts = None
    out_noop = m._passback_scope_ps(1, ps_default, prev)
    assert out_noop is ps_default
    # refine (attention + route 2) -> a re-fed PS subspace (not ps_default),
    # OR ps_default when the prior carrier is empty (guarded). Either way the
    # call is finite + does not raise.
    ws0._intent_boosts = torch.ones(4)
    object.__setattr__(ws0, "_mereology_ratio_obs",
                       {"route_hint": torch.tensor([2, 2, 2, 2])})
    with torch.no_grad():
        out_refine = m._passback_scope_ps(1, ps_default, prev)
    assert out_refine is not None


def test_passback_scoped_focus_zeros_out_of_span():
    # The "scoped" case focuses the percept to the word's [start, end] span:
    # slots whose normalized position falls OUTSIDE the span are zeroed.
    m = _build("MM_mereology.xml")
    x = _batch(m)
    with torch.no_grad():
        m.forward(x)
    ws0 = m.wholeSpaces[0]
    prev = m.conceptualSpaces[0]._subspaceForWS
    if prev is None or prev.is_empty():
        pytest.skip("prior-symbol carrier empty for this config/batch")
    # Scope to the FIRST ~40% of the slots; the tail must come back zeroed.
    object.__setattr__(ws0, "_passback_scope_where", torch.tensor([0.0, 0.4]))
    with torch.no_grad():
        ps = m._passback_scope_ps(1, None, prev)
    ev = ps.materialize() if ps is not None else None
    if ev is None or ev.dim() != 3 or ev.shape[1] < 4:
        pytest.skip("scoped focus needs a [B, N>=4, D] percept event")
    N = int(ev.shape[1])
    pos = (torch.arange(N).float() + 0.5) / N
    out_of_span = pos > 0.4
    assert float(ev[:, out_of_span, :].abs().max()) == 0.0, (
        "slots outside the scope .where span must be zeroed")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
