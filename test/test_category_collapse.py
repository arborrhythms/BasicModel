"""Online category collapse (``<categoryCollapse>``).

``participation.learned_collapse`` — the participation-signature merge that
collapses categories the way the grammatical operations mandate (e.g. two
symbols that fill the same operator roles are one category) — now runs inside
the live model via ``Grammar.category_collapse()``, gated by the default-off
``<categoryCollapse>`` flag. Previously it ran only in the offline D1 gate
(``test_d1_pos_recovery_gate``). Rule choice is unchanged; the collapse is
computed and exposed for a future consuming increment.
"""
import os
import sys

_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project, "bin"))

import Spaces  # noqa: E402
from Language import Grammar  # noqa: E402
from participation import grammar_rules, collapse_conflicts  # noqa: E402
from util import TheXMLConfig  # noqa: E402
from test_basicmodel import _populate_test_config  # noqa: E402

_TRANSITIONAL = os.path.join(_project, "test/fixtures/transitional_pos.grammar")


def _load(grammar_file):
    g = Grammar()
    g.load_from_grammar_file(grammar_file)
    return g


def test_category_collapse_runs_at_runtime_conflict_free():
    """``Grammar.category_collapse()`` runs learned_collapse at runtime and
    keeps every rule distinguishable — the online equivalent of the D1 gate."""
    g = _load(_TRANSITIONAL)
    collapse = g.category_collapse()
    n_sym, n_cat = len(collapse), len(set(collapse.values()))
    assert all(isinstance(c, int) for c in collapse.values())
    assert n_cat < n_sym                       # a real collapse (43 -> 14)
    assert collapse_conflicts(grammar_rules(g), collapse) == 0


def test_category_collapse_is_cached_per_direction():
    g = _load(_TRANSITIONAL)
    assert g.category_collapse() is g.category_collapse()          # cached
    assert g.category_collapse("compose") is not g.category_collapse("both")


def test_live_grammar_already_collapsed_is_a_noop():
    """The live ``complete.grammar`` is already role-collapsed, so the online
    collapse is a conflict-free no-op (no further merges), not an error."""
    g = _load("complete.grammar")
    collapse = g.category_collapse()
    assert collapse_conflicts(grammar_rules(g), collapse) == 0
    assert len(set(collapse.values())) == len(collapse)


def _whole_space(d=8):
    nP, nS = 4, 6
    _populate_test_config(
        inputDim=d, perceptDim=d, conceptDim=d, symbolDim=d,
        wordDim=d, outputDim=d, nInput=nP, nPercepts=nP,
        nConcepts=nS, nSymbols=nS, nWords=nS, nOutput=nS,
        nWhere=0, nWhen=0)
    return Spaces.WholeSpace([nP, d], [nS, d], [nS, d])


def test_gated_off_by_default_stores_no_collapse():
    ws = _whole_space()                       # init_config resets the flag off
    ws.enable_category_codebook(_load(_TRANSITIONAL))
    assert ws._category_collapse is None


def test_gated_on_runs_collapse_at_enable():
    ws = _whole_space()
    TheXMLConfig.set("architecture.categoryCollapse", True)
    try:
        ws.enable_category_codebook(_load(_TRANSITIONAL))
    finally:
        TheXMLConfig.set("architecture.categoryCollapse", False)
    assert isinstance(ws._category_collapse, dict)
    assert len(set(ws._category_collapse.values())) < len(ws._category_collapse)
