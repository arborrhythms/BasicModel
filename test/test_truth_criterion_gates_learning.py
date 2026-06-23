"""Task 6c (doc/plans/2026-05-29-stm-serial-parallel-modes.md §7c).

The ``<truthCriterion>`` knob gates learned-relation insertion:

    accept iff  learn_score >= truth_criterion

These tests MOCK the three factor methods via the test seam (the plan's
required design: "Mock the three formula factors directly via test seam
on ``_compute_learn_score``") so the GATING logic is verified
independent of the factor formulas:

  (a) tc == 1  -> no insertion regardless of the (sub-1) score.
  (b) tc == 0  -> insertion regardless of the score.
  (c) tc == 0.3 -> insertion iff learn_score >= 0.3.

For accepted cases that assert WS taxonomy shape, the mocked children factor
is kept at 1.0 so the relation is reducible: ``_maybe_learn_relation`` returns
the predicate position and leaves the two ideas as its taxonomy children. A
reject returns ``None`` and writes nothing.
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_radix_model():
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    m.eval()
    return m


def _mock_factors(cs, children, obvious, resolves):
    """Monkeypatch the three factor seams to fixed constants so the
    product (the learn-score) is deterministic."""
    cs._learn_score_children_in_codebook = (
        lambda i1, i2, _c=children: float(_c))
    cs._learn_score_is_truth_obvious = (
        lambda rel, _o=obvious: float(_o))
    cs._learn_score_resolves_contradiction = (
        lambda rel, _r=resolves: float(_r))


def _ideas(cs):
    D = int(cs.nDim)
    pred = torch.zeros(D)
    pred[0] = 1.0
    a = torch.zeros(D)
    a[1] = 1.0
    b = torch.zeros(D)
    b[2] = 1.0
    return pred, a, b


class TestTruthCriterionGate(unittest.TestCase):

    def test_tc_one_never_learns(self):
        """tc == 1: even a high (but < 1) score is rejected."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        cs.truth_criterion = 1.0
        # learn_score = 0.9 * 0.9 * 0.9 = 0.729 < 1.0 -> reject.
        _mock_factors(cs, 0.9, 0.9, 0.9)
        pred, a, b = _ideas(cs)
        out = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNone(
            out, "tc=1 must reject a sub-1 learn-score (nothing learned)")

    def test_tc_one_rejects_even_a_perfect_score(self):
        """tc == 1 means LEARN NOTHING -- a perfect learn_score of 1.0 must
        NOT slip through (the maximal-bar endpoint is exclusive)."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        cs.truth_criterion = 1.0
        # learn_score = 1.0 * 1.0 * 1.0 = 1.0; must STILL be rejected at tc=1.
        _mock_factors(cs, 1.0, 1.0, 1.0)
        pred, a, b = _ideas(cs)
        out = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNone(
            out, "tc=1 must reject even a perfect 1.0 learn-score")

    def test_tc_zero_always_learns(self):
        """tc == 0: even a near-zero score is accepted."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        cs.truth_criterion = 0.0
        # learn_score = 0.0 (one factor zero) but tc=0 -> 0 >= 0 accept.
        # Keep children=1.0 so the accepted relation is still reducible and
        # lands in the WS taxonomy this test observes.
        _mock_factors(cs, 1.0, 0.0, 1.0)
        pred, a, b = _ideas(cs)
        pred_pos = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNotNone(
            pred_pos, "tc=0 must accept regardless of learn-score")
        children = ws.taxonomy_children(pred_pos)
        self.assertEqual(
            len(children), 2,
            f"accepted relation must leave 2 idea children; got "
            f"{children!r}")

    def test_tc_threshold_accepts_at_or_above(self):
        """tc == 0.3: score 0.343 (>=0.3) is accepted."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        cs.truth_criterion = 0.3
        # 1.0 * 0.7 * 0.49 = 0.343 >= 0.3 -> accept. Keep children=1.0 so
        # the accepted relation is reducible and lands in the WS taxonomy.
        _mock_factors(cs, 1.0, 0.7, 0.49)
        pred, a, b = _ideas(cs)
        pred_pos = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNotNone(
            pred_pos, "learn_score 0.343 >= tc 0.3 must accept")
        self.assertEqual(len(ws.taxonomy_children(pred_pos)), 2)

    def test_tc_threshold_rejects_below(self):
        """tc == 0.3: score 0.125 (<0.3) is rejected."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        cs.truth_criterion = 0.3
        # 0.5^3 = 0.125 < 0.3 -> reject.
        _mock_factors(cs, 0.5, 0.5, 0.5)
        pred, a, b = _ideas(cs)
        out = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNone(out, "learn_score 0.125 < tc 0.3 must reject")

    def test_tc_threshold_exact_boundary_accepts(self):
        """>= is inclusive: a score exactly equal to tc is accepted."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        cs.truth_criterion = 0.5
        # 1.0 * 0.5 * 1.0 = 0.5 == tc -> accept (>= boundary). Keep
        # children=1.0 so the accepted relation is reducible.
        _mock_factors(cs, 1.0, 0.5, 1.0)
        pred, a, b = _ideas(cs)
        pred_pos = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNotNone(
            pred_pos, "score == tc must accept (>= is inclusive)")
        self.assertEqual(len(ws.taxonomy_children(pred_pos)), 2)


if __name__ == "__main__":
    unittest.main()
