"""Task 6c (doc/plans/2026-05-29-stm-serial-parallel-modes.md §7c).

Pin each of the three learn-score factor methods in isolation:

  * ``_learn_score_children_in_codebook`` -> 1.0 when BOTH ideas are
    already-known SS-codebook concepts (nearest-row distance below the
    threshold); 0.0 when neither is.
  * ``_learn_score_is_truth_obvious`` -> 1.0 for a perfectly-agreeing
    TruthSet (all-positive strong truths -> assess support == 1.0).
  * ``_learn_score_resolves_contradiction`` -> 1.0 for a relation over a
    contested region (a truth affirmed AND denied -> assess conflict
    == 1.0).

The factors are the test seam the plan requires: each is a separate
overridable method, so the gating tests can monkeypatch them. These
tests instead exercise the REAL factor computations against a fully-
wired radix model (``terminalSymbolSpace_ref`` + ``symbolSpace.
truth_layer``).
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
    """Build the MM_xor radix-chunking model (fully wired refs)."""
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


def _seed_known_concept(ws, vec):
    """Insert ``vec`` as a fresh SS-codebook row so a later
    nearest-row lookup of the same vector lands distance 0."""
    return ws.insert_whole(init_vec=vec)


class TestChildrenInCodebookFactor(unittest.TestCase):
    """``_learn_score_children_in_codebook`` == 1 when both ideas are
    known concepts, 0 when neither is."""

    def test_known_children_score_one(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        D = int(cs.nDim)
        idea1 = torch.zeros(D)
        idea1[0] = 0.9
        idea2 = torch.zeros(D)
        idea2[1] = -0.7
        # Seed both ideas as known SS rows so nearest-row distance == 0.
        _seed_known_concept(ws, idea1)
        _seed_known_concept(ws, idea2)
        score = cs._learn_score_children_in_codebook(idea1, idea2)
        self.assertEqual(
            score, 1.0,
            f"both ideas are exact codebook rows -> children factor "
            f"must be 1.0; got {score}")

    def test_one_known_one_unknown_score_half(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        D = int(cs.nDim)
        # Tight threshold so a far-away idea is unambiguously "unknown".
        cs._learn_children_dist_threshold = 1e-3
        known = torch.zeros(D)
        known[0] = 0.5
        _seed_known_concept(ws, known)
        far = torch.full((D,), 999.0)
        score = cs._learn_score_children_in_codebook(known, far)
        self.assertEqual(
            score, 0.5,
            f"exactly one of two ideas is known -> 0.5; got {score}")

    def test_unknown_children_score_zero(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        D = int(cs.nDim)
        cs._learn_children_dist_threshold = 1e-3
        far1 = torch.full((D,), 999.0)
        far2 = torch.full((D,), -999.0)
        score = cs._learn_score_children_in_codebook(far1, far2)
        self.assertEqual(
            score, 0.0,
            f"neither idea is near a codebook row -> 0.0; got {score}")


class TestIsTruthObviousFactor(unittest.TestCase):
    """``_learn_score_is_truth_obvious`` == 1 for a perfectly-agreeing
    TruthSet (assess support == 1)."""

    def test_agreeing_truthset_scores_one(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        tl = cs.symbolSpace.truth_layer
        tl.clear()
        D = int(tl.truths.shape[-1])
        # Two strongly-affirming truths -> assess()["support"] == 1.0.
        tl.record(torch.ones(D), 1.0)
        tl.record(torch.ones(D), 1.0)
        self.assertEqual(tl.assess()["support"], 1.0)
        relation = torch.ones(int(cs.nDim))
        score = cs._learn_score_is_truth_obvious(relation)
        self.assertEqual(
            score, 1.0,
            f"perfectly-agreeing TruthSet -> is_truth_obvious 1.0; "
            f"got {score}")

    def test_empty_truthset_scores_zero(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        tl = cs.symbolSpace.truth_layer
        tl.clear()
        # Empty -> support 0 -> nothing to agree with.
        score = cs._learn_score_is_truth_obvious(torch.ones(int(cs.nDim)))
        self.assertEqual(score, 0.0)


class TestResolvesContradictionFactor(unittest.TestCase):
    """``_learn_score_resolves_contradiction`` == 1 for a relation over a
    contested region (assess conflict == 1)."""

    def test_contradicting_truths_score_one(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        tl = cs.symbolSpace.truth_layer
        tl.clear()
        D = int(tl.truths.shape[-1])
        # Affirm AND deny the same concept -> assess()["conflict"] == 1.0.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tl.record(torch.ones(D), 1.0)
            tl.record(torch.ones(D), -1.0)
        self.assertEqual(tl.assess()["conflict"], 1.0)
        relation = torch.ones(int(cs.nDim))
        score = cs._learn_score_resolves_contradiction(relation)
        self.assertEqual(
            score, 1.0,
            f"relation over a contested (affirm+deny) region -> "
            f"resolves_contradiction 1.0; got {score}")

    def test_consistent_truthset_scores_zero(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        tl = cs.symbolSpace.truth_layer
        tl.clear()
        D = int(tl.truths.shape[-1])
        # All-agreeing -> conflict 0 -> nothing to resolve.
        tl.record(torch.ones(D), 1.0)
        tl.record(torch.ones(D), 1.0)
        self.assertEqual(tl.assess()["conflict"], 0.0)
        score = cs._learn_score_resolves_contradiction(
            torch.ones(int(cs.nDim)))
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
