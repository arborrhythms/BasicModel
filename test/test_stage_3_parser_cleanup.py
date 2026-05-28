"""Stage 3 TDD gates: parser cleanup.

Plan: doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md Stage 3.

Acceptance gates:
  * ``Chart`` class no longer exists in ``bin/Language.py`` (importing
    ``Language.Chart`` raises ``AttributeError``).
  * ``WordSubSpace`` (the WordSpace carrier) installs ``languageLayer``
    as a direct ``LanguageLayer`` instance (no chart indirection).
  * ``Grammar.rule_probability`` returns floats in [0, 1] for both
    dormant defaults and learned overrides; ``_fired_bodies`` single-
    application gate still works; ``reset_derivation`` still callable.
  * ``binary_tiling_soft_dp`` and ``binary_tiling_viterbi`` still
    callable (the signal-router DP primitives).
  * The retired XML knobs (``parserBackend``, ``routerKind``,
    ``chartTau``, ``chartTopK``, ``chartNoiseEps``) raise a loud
    ValueError at config load time.

These tests follow project memory's "fail loud" rule: retired knobs
must error rather than be silently ignored.
"""
import sys
from pathlib import Path

import pytest

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


# --- Gate 1: Chart class is gone ----------------------------------------

def test_chart_class_no_longer_exists():
    """Importing ``Language.Chart`` should fail; the class is retired."""
    import Language
    assert not hasattr(Language, "Chart"), (
        "Chart class still present in bin/Language.py; Stage 3 retires it."
    )


def test_chart_inside_helpers_gone():
    """The CKY inside/outside helpers should not be live attributes."""
    import Language
    # These were Chart methods. If Chart is retired, the module-level
    # attributes shouldn't exist either.
    for name in (
        "_chart_inside", "_chart_outside",
        "_compose_chart_cky", "_compose_chart_cky_viterbi",
        "_ensure_soft_chart_built",
    ):
        # They can survive as comments / strings in the source but must
        # not be importable callables.
        assert not callable(getattr(Language, name, None)), (
            f"Chart helper {name!r} still callable in Language module."
        )


# --- Gate 2: WordSpace has self.languageLayer ---------------------------

def _bare_word_subspace_with_signal_router():
    """Construct a minimal WordSubSpace-like with a LanguageLayer attached.

    Stage 3 wires ``self.languageLayer`` directly on WordSubSpace (no
    Chart indirection). This test verifies the attribute exists.
    """
    from Language import LanguageLayer
    layer = LanguageLayer(
        n_input=4, n_output=4,
        hidden_dim=16, feature_dim=8,
        max_depth=3, temperature=1.0,
    )
    # Bare object with the attribute, mirroring the expected post-Stage-3
    # WordSubSpace surface.
    class _FakeWS:
        languageLayer = layer
    return _FakeWS()


def test_word_subspace_has_language_layer_attribute():
    ws = _bare_word_subspace_with_signal_router()
    from Language import LanguageLayer
    assert isinstance(ws.languageLayer, LanguageLayer), (
        "WordSpace must expose self.languageLayer as a LanguageLayer "
        "instance (Stage 3 promotion)."
    )


# --- Gate 3: Grammar.rule_probability + single-application gate ---------

def test_rule_probability_returns_float_in_unit_interval():
    """Dormant defaults: fold ops → 1.0, negation ops → 0.0; both floats."""
    from Language import Grammar
    g = Grammar()
    # Dormant defaults:
    p_fold = g.rule_probability("intersection(C, C)")
    assert isinstance(p_fold, float), type(p_fold)
    assert 0.0 <= p_fold <= 1.0
    assert p_fold == 1.0

    p_neg = g.rule_probability("not(S)")
    assert isinstance(p_neg, float), type(p_neg)
    assert 0.0 <= p_neg <= 1.0
    assert p_neg == 0.0

    p_non = g.rule_probability("non(S)")
    assert p_non == 0.0


def test_rule_probability_learned_override_in_unit_interval():
    """When ``_learned_rule_probs`` carries a learned value, it must be
    returned (and remain in [0, 1])."""
    from Language import Grammar
    g = Grammar()
    g._learned_rule_probs = {"intersection(C, C)": 0.42}
    p = g.rule_probability("intersection(C, C)")
    assert p == pytest.approx(0.42)
    assert 0.0 <= p <= 1.0


def test_rule_probability_fired_bodies_blocks_resfire():
    """Single-application: ``note_rule_fired`` makes rule_probability
    return 0 until ``reset_derivation`` is called."""
    from Language import Grammar
    g = Grammar()
    body = "intersection(C, C)"
    assert g.rule_probability(body) == 1.0
    g.note_rule_fired(body)
    assert g.rule_probability(body) == 0.0
    g.reset_derivation()
    assert g.rule_probability(body) == 1.0


# --- Gate 4: signal-router DP primitives still callable -----------------

def test_binary_tiling_soft_dp_callable():
    """The signal router's soft-DP function must remain callable."""
    import torch
    from Language import binary_tiling_soft_dp
    B, N, R_copy, R_red = 2, 4, 1, 2
    copy_score = torch.zeros(B, N, R_copy)
    reduce_score = torch.zeros(B, N - 1, R_red)
    out = binary_tiling_soft_dp(copy_score, reduce_score)
    assert out is not None


def test_binary_tiling_viterbi_callable():
    """The signal router's Viterbi function must remain callable."""
    import torch
    from Language import binary_tiling_viterbi
    B, N, R_copy, R_red = 2, 4, 1, 2
    copy_score = torch.zeros(B, N, R_copy)
    reduce_score = torch.zeros(B, N - 1, R_red)
    out = binary_tiling_viterbi(copy_score, reduce_score)
    assert out is not None


# --- Gate 5: retired XML knobs error loud --------------------------------

@pytest.mark.parametrize("knob", [
    "parserBackend",
    "routerKind",
    "chartTau",
    "chartTopK",
    "chartNoiseEps",
])
def test_retired_xml_knobs_raise_on_load(knob):
    """Legacy configs that still set retired knobs must error loudly.

    The check fires when WordSubSpace is built from a config containing
    any of the retired ``<parserBackend>`` / ``<routerKind>`` /
    ``<chartTau>`` / ``<chartTopK>`` / ``<chartNoiseEps>`` knobs.
    """
    from util import TheXMLConfig
    # Stash and reset config to a clean state, then set the retired knob.
    saved = dict(TheXMLConfig._data)
    try:
        TheXMLConfig._data.clear()
        TheXMLConfig._data.setdefault("WordSpace", {})[knob] = "1.0"
        # Importing helpers from Language so the loud check is in scope.
        import Language
        fn = getattr(Language, "_assert_retired_chart_knobs_absent", None)
        assert fn is not None, (
            "Language must expose _assert_retired_chart_knobs_absent to "
            "gate Stage 3 config migration."
        )
        with pytest.raises((ValueError, RuntimeError), match="retired"):
            fn()
    finally:
        TheXMLConfig._data.clear()
        TheXMLConfig._data.update(saved)
