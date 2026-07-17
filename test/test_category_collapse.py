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


# -- Consuming the collapse in rule selection ----------------------------------
# The category-role priors pool the category context over collapsed-category
# siblings (Language._collapsed_cat_ctx), so rules whose input category merged
# share routing evidence. Identity on every shipping grammar (all pre-
# collapsed); the pooling is proven here with a synthetic collapse.
import torch  # noqa: E402
from Language import _build_collapse_pool, _collapsed_cat_ctx  # noqa: E402


def test_pooling_matrix_sums_merged_role_columns():
    # a_I1 and b_I1 collapse to one category; c_I1, d_O1 stay singletons.
    role_index = {"a_I1": 0, "b_I1": 1, "c_I1": 2, "d_O1": 3}
    collapse = {"a_I1": 0, "b_I1": 0, "c_I1": 1, "d_O1": 2}
    P = _build_collapse_pool(role_index, collapse, 4)
    x = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]])
    pooled = x @ P
    # merged columns 0,1 -> sum(1,2)=3; singletons unchanged.
    assert pooled.flatten().tolist() == [3.0, 3.0, 4.0, 8.0]


def test_pooling_matrix_is_none_for_identity_collapse():
    role_index = {"a_I1": 0, "b_I1": 1}
    identity = {"a_I1": 0, "b_I1": 1}
    assert _build_collapse_pool(role_index, identity, 2) is None


def test_collapsed_cat_ctx_passthrough_when_flag_off():
    x = torch.randn(2, 3, 5)
    TheXMLConfig.set("architecture.categoryCollapse", False)
    assert torch.equal(_collapsed_cat_ctx(x), x)


def test_consumption_is_clean_noop_in_serial_mode_on_live_grammar():
    """serial=True forward is byte-identical with <categoryCollapse> on vs off
    on the (already role-collapsed) live grammar — the consumption is wired
    into the serial per-word path and correctly a no-op there."""
    from data import TheData
    from Models import BaseModel

    def _build_serial():
        TheData.load("xor")
        m, _ = BaseModel.from_config("data/MM_xor.xml", data=TheData)
        m.serial_mode = True
        if hasattr(m, "perceptualSpace"):
            m.perceptualSpace.serial_mode = True
        if hasattr(m, "conceptualSpace"):
            m.conceptualSpace.serial_mode = True
        return m

    inp = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).float().unsqueeze(1)
    torch.manual_seed(0)
    m_off = _build_serial()
    TheXMLConfig.set("architecture.categoryCollapse", False)
    out_off = m_off.forward(inp)[2].detach().clone()
    torch.manual_seed(0)
    m_on = _build_serial()
    TheXMLConfig.set("architecture.categoryCollapse", True)
    try:
        out_on = m_on.forward(inp)[2].detach().clone()
    finally:
        TheXMLConfig.set("architecture.categoryCollapse", False)
    assert torch.allclose(out_on, out_off, atol=1e-6)
