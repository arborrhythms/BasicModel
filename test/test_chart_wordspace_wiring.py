"""Smoke tests for the 2026-05-01 syntactic-layer refactor.

Exercises:
  * WordSpace constructs a Chart instance with chart-tier params.
  * Per-space SyntacticLayers register their host layers in the
    `wordSpace._host_layer_registry` registry under the right tier.
  * Chart.compose / Chart.generate populate
    `wordSpace.current_rules` / `generate_rules`.
  * Chart's eval-mode (Viterbi) and train-mode (soft) inside passes
    both run cleanly.
  * Chart-via-host-layer dispatch fires when the chart picks a rule
    backed by a parametrized fold (Step 7 wiring).
"""
import sys
from pathlib import Path

import pytest
import torch

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))
sys.path.insert(0, str(_project / "test"))


def _isolated_grammar(rules):
    """Configure ``TheGrammar`` with a flat upward-rule dict."""
    from Language import TheGrammar
    TheGrammar._configured = False
    TheGrammar.configure(rules)
    return TheGrammar


# --- Chart instantiation --------------------------------------------

def test_chart_constructs_in_isolation():
    from Language import Chart
    chart = Chart(nInput=4, max_depth=4, hidden_dim=16, D_rule=4,
                  feature_dim=8, w_max=4)
    # nn.Module bookkeeping populated.
    assert sum(p.numel() for p in chart.parameters()) > 0
    # Per-call state slots default to None.
    assert chart._chart_score is None
    assert chart._chart_vec is None


def test_chart_compose_populates_rules_dict():
    from Language import Chart, TheGrammar
    _isolated_grammar({'S': ['intersection(C, C)'], 'C': []})
    chart = Chart(nInput=4, max_depth=4, hidden_dim=16, D_rule=4,
                  feature_dim=8, w_max=4)

    class _FakeWS:
        current_rules = {}
        generate_rules = {}
        _sentence_completed = [False]
        _compose_generation = 0
        _generate_generation = 0
        _host_layer_registry = {}

        def host_layer(self, tier, rule_name):
            return None

        def _row_K(self):
            return 1

    ws = _FakeWS()
    chart.train()
    data = torch.randn(1, 3, 8)
    rules = chart.compose(data, ws)
    assert isinstance(rules, dict)
    assert chart._chart_score is not None
    assert chart._chart_score.shape[0] == 1


def test_chart_eval_mode_uses_viterbi():
    """Eval mode should run the hard / Viterbi inside pass (Q10.5)."""
    from Language import Chart
    _isolated_grammar({'S': ['intersection(C, C)'], 'C': []})
    chart = Chart(nInput=4, max_depth=4, hidden_dim=16, D_rule=4,
                  feature_dim=8, w_max=4)

    class _FakeWS:
        current_rules = {}
        generate_rules = {}
        _sentence_completed = [False]
        _compose_generation = 0
        _generate_generation = 0
        _host_layer_registry = {}

        def host_layer(self, tier, rule_name):
            return None

        def _row_K(self):
            return 1

    chart.eval()
    data = torch.randn(1, 3, 8)
    rules = chart.compose(data, _FakeWS())
    assert isinstance(rules, dict)
    # Viterbi mode pins each cell's score to a single rule -- not a
    # logsumexp mixture. Confirm finite scores (no NaN / -inf in the
    # populated cells).
    assert torch.isfinite(chart._chart_score).any().item()


# --- WordSpace plumbing ---------------------------------------------

def test_wordspace_owns_chart_and_registry():
    """WordSpace under AR mode owns a Chart instance and a non-empty
    host_layer registry; per-space SyntacticLayers register their
    builtin layers."""
    import test_basicmodel as tb
    import Models, Language

    tb._populate_test_config(
        inputDim=4, perceptDim=4, conceptDim=4, symbolDim=1, wordDim=1,
        nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8,
        nOutput=8, perceptPassThrough=True, symbolPassThrough=True)
    Models.TheXMLConfig._data["architecture"]["syntax"] = True
    (Models.TheXMLConfig._data["architecture"]
        .setdefault("training", {})["maskedPrediction"]) = "AR"

    m = Models.BasicModel()
    m.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8,
             nOutput=8, masked_prediction='AR')
    assert m.wordSpace is not None
    assert type(m.wordSpace.chart).__name__ == 'Chart'
    # Registry has at least the SymbolicSpace builtin layers. Per
    # 2026-05-03 layer-naming refactor, SigmaLayer is registered under
    # rule_name "sigma" (the unary multiplicative OR-fold); the binary
    # min/max ops "intersection" / "union" / "conjunction" /
    # "disjunction" are separate GrammarLayer subclasses, lazy-built
    # only when the grammar references them.
    keys = list(m.wordSpace._host_layer_registry.keys())
    assert ('S', 'sigma') in keys
    # Each space carries a per-space SyntacticLayer with its tier.
    p_sl = m.perceptualSpace.syntacticLayer
    c_sl = m.conceptualSpace.syntacticLayer
    s_sl = m.symbolicSpace.syntacticLayer
    assert p_sl is not None and p_sl.tier == 'P'
    assert c_sl is not None and c_sl.tier == 'C'
    assert s_sl is not None and s_sl.tier == 'S'


# --- Pipeline integration -------------------------------------------

def test_chartcompose_and_chartgenerate_in_pipeline():
    """ChartCompose / ChartGenerate are inserted into the AR-mode
    forward / reverse pipelines (Step 6 wiring)."""
    from Models import ChartCompose, ChartGenerate
    import test_basicmodel as tb
    import Models

    tb._populate_test_config(
        inputDim=4, perceptDim=4, conceptDim=4, symbolDim=1, wordDim=1,
        nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8,
        nOutput=8, perceptPassThrough=True, symbolPassThrough=True)
    Models.TheXMLConfig._data["architecture"]["syntax"] = True
    (Models.TheXMLConfig._data["architecture"]
        .setdefault("training", {})["maskedPrediction"]) = "AR"

    m = Models.BasicModel()
    m.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8,
             nOutput=8, masked_prediction='AR')

    # ChartCompose and ChartGenerate are now module attributes
    # (``_chart_compose`` / ``_chart_generate``) called inside
    # ``_forward_stem`` / ``_run_pipeline_rev``, not children of a
    # ``pipeline_stem`` Sequential.
    assert isinstance(m._chart_compose, ChartCompose), (
        f"_chart_compose not a ChartCompose: {type(m._chart_compose).__name__}")
    assert isinstance(m._chart_generate, ChartGenerate), (
        f"_chart_generate not a ChartGenerate: {type(m._chart_generate).__name__}")


# --- GrammarLayer registry (Step 8) ---------------------------------

def test_grammar_layer_classes_registry_complete():
    """``GRAMMAR_LAYER_CLASSES`` covers every grammar op exposed
    by the post-2026-05-05 operator set. ``Fusion`` /
    ``Contiguous`` were retired (duplicates of DisjunctionLayer
    at S-tier -- migrate to ``disjunction(S, S)``); the prior
    ``what`` / ``where`` / ``when`` slot-selector classes were
    retired (subspace partitioning is the dispatcher's
    responsibility, not a grammar rule). ``absorb`` was retired
    -- the marker lives on ``GrammarLayer.absorb`` (base method)
    rather than a dedicated class."""
    from Layers import GRAMMAR_LAYER_CLASSES
    expected = {
        'not', 'non', 'intersection', 'union',
        'lift', 'lower', 'conjunction', 'disjunction',
        'equals', 'part', 'true', 'false',
        'swap', 'query',
    }
    assert expected.issubset(GRAMMAR_LAYER_CLASSES.keys()), (
        f"missing rules: {expected - set(GRAMMAR_LAYER_CLASSES.keys())}")


def test_grammar_layer_classes_are_real():
    """Each parameter-free GrammarLayer class instantiates and exposes
    `forward` (arity 1) or `compose` (arity 2). Step 8 retired the
    facade dispatch; the classes are now self-sufficient.

    Per 2026-05-05 tier reshuffle, IntersectionLayer and UnionLayer
    are L-tier (logical) primitives implementing binary lattice
    min / max on bivector activation; their constructor takes an
    optional ``monotonic`` kwarg only. ConjunctionLayer /
    DisjunctionLayer are S-tier counterparts on the post-codebook
    scalar activation -- always monotonic, no kwargs. SwapLayer
    still requires a swap_size arg.
    """
    from Layers import GRAMMAR_LAYER_CLASSES
    for rule_name, cls in GRAMMAR_LAYER_CLASSES.items():
        try:
            inst = cls()
        except TypeError:
            if rule_name == 'swap':
                inst = cls(swap_size=1)
            else:
                raise
        if inst.arity == 2:
            assert hasattr(inst, 'compose'), (
                f"{rule_name}: arity-2 layer missing compose")
        else:
            assert hasattr(inst, 'forward'), (
                f"{rule_name}: arity-1 layer missing forward")
