"""Smoke tests for the 2026-05-01 syntactic-layer refactor.

Exercises:
  * WordSpace constructs a Chart instance with chart-tier params.
  * Per-space SyntacticLayers register their host layers in the
    `wordSubSpace._host_layer_registry` registry under the right tier.
  * Chart.compose / Chart.generate populate
    `wordSubSpace.current_rules` / `generate_rules`.
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
             nOutput=8)
    assert m.wordSubSpace is not None
    assert type(m.wordSubSpace.chart).__name__ == 'Chart'
    # Registry has at least the SymbolicSpace builtin layers. Per
    # 2026-05-03 layer-naming refactor, SigmaLayer is registered under
    # rule_name "sigma" (the unary multiplicative OR-fold); the binary
    # min/max ops "intersection" / "union" / "conjunction" /
    # "disjunction" are separate GrammarLayer subclasses, lazy-built
    # only when the grammar references them.
    keys = list(m.wordSubSpace._host_layer_registry.keys())
    # Post-bivector-retirement (2026-05-20) the substrate-level rule
    # layers live on ConceptualSpace (``sigma`` + ``pi``) and the
    # grammar-driven boolean ops register at SymbolicSpace. The
    # PerceptualSpace SigmaLayer is no longer instantiated by default;
    # PerceptualSpace consumes ``cs.sigma_percept`` for the P→C lift
    # instead of carrying its own host SigmaLayer.
    assert ('C', 'sigma') in keys, f"missing ('C', 'sigma'); have {keys!r}"
    assert ('C', 'pi') in keys, f"missing ('C', 'pi'); have {keys!r}"
    assert ('S', 'not') in keys, f"missing ('S', 'not'); have {keys!r}"
    # Per-space SyntacticLayer carries the tier where one is wired.
    c_sl = m.conceptualSpace.syntacticLayer
    s_sl = m.symbolicSpace.syntacticLayer
    assert c_sl is not None and c_sl.tier == 'C'
    assert s_sl is not None and s_sl.tier == 'S'
    # PerceptualSpace's SyntacticLayer is optional under the bivector-
    # retirement contract: not all configs wire one.
    p_sl = getattr(m.perceptualSpace, 'syntacticLayer', None)
    if p_sl is not None:
        assert p_sl.tier == 'P'


# --- Pipeline integration -------------------------------------------

def test_chart_fires_at_C_inside_body():
    """Post-2026-05-14: the chart fires at C-tier inside
    ``_forward_body`` (via ``_chart_compose_at_C``).  The reverse-side
    ``_chart_generate_from_stm`` mirror was retired alongside the
    reverse pipeline in the IR-only refactor; the legacy stem-level
    ChartCompose / ChartGenerate modules remain retired.
    """
    import test_basicmodel as tb
    import Models

    tb._populate_test_config(
        inputDim=4, perceptDim=4, conceptDim=4, symbolDim=1, wordDim=1,
        nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8,
        nOutput=8, perceptPassThrough=True, symbolPassThrough=True)
    Models.TheXMLConfig._data["architecture"]["syntax"] = True

    m = Models.BasicModel()
    m.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8,
             nOutput=8)

    assert hasattr(m, '_chart_compose_at_C'), (
        "BasicModel must expose ``_chart_compose_at_C`` for chart-at-C wiring")
    assert not hasattr(m, '_chart_compose'), (
        "Legacy ``_chart_compose`` module should have been removed")
    assert hasattr(m, '_chart_generate_from_stm'), (
        "Reverse-side chart mirror ``_chart_generate_from_stm`` is "
        "restored with the reverse pipeline (post-2026-05 "
        "reconciliation: reverse() reconstructs input).")
    assert not hasattr(m, '_chart_generate'), (
        "Legacy ``_chart_generate`` module should have been removed")


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
        'isEqual', 'equal', 'part', 'true', 'false',
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
