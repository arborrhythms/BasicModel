"""Task 6c (doc/plans/2026-05-29-stm-serial-parallel-modes.md §7c).

Lies / uncertain regions still get learned: a LOW ``is_truth_obvious``
(the relation contradicts the TruthSet) is NOT gated out on its own. If
``children_in_codebook`` and ``resolves_contradiction`` are high, the
PRODUCT can still clear ``truth_criterion`` -- and the accepted relation
lands in the codebook carrying a tetralemma 4-tuple that records the
conflict (high ``b`` / BOTH weight).

Two layers:
  * a SEAM test that fixes the factors (children=1, obvious=0.1,
    resolves=1 -> product 0.1 < 0.3 would reject, so bump children/
    resolves to make 1*0.4*1 = 0.4 >= 0.3 with a still-low obvious),
    proving low-obvious alone does not block; and
  * an END-TO-END test driving the REAL factors + REAL tetralemma trust
    off a contested TruthSet (affirm+deny -> support 0.25 low, conflict
    1.0 high), asserting acceptance AND a high-BOTH trust tuple on the
    inserted META.
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


def _ideas(cs):
    D = int(cs.nDim)
    pred = torch.zeros(D)
    pred[0] = 1.0
    a = torch.zeros(D)
    a[1] = 1.0
    b = torch.zeros(D)
    b[2] = 1.0
    return pred, a, b


class TestLowTruthObviousDoesNotBlockAlone(unittest.TestCase):
    """A low ``is_truth_obvious`` does NOT by itself prevent learning
    when the other two factors are high enough that the product clears
    the criterion."""

    def test_low_obvious_high_others_still_learns(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        cs.truth_criterion = 0.3
        # is_truth_obvious = 0.4 (low-ish, a contested / partly-false
        # claim), children = 1.0, resolves = 1.0 -> product 0.4 >= 0.3.
        cs._learn_score_children_in_codebook = lambda i1, i2: 1.0
        cs._learn_score_is_truth_obvious = lambda rel: 0.4
        cs._learn_score_resolves_contradiction = lambda rel: 1.0
        # The product alone:
        pred, a, b = _ideas(cs)
        score = cs._compute_learn_score(pred, a, b)
        self.assertAlmostEqual(score, 0.4, places=6)
        pred_pos = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNotNone(
            pred_pos,
            "low is_truth_obvious must NOT block when children + "
            "resolves are high (product 0.4 >= tc 0.3)")
        self.assertEqual(len(ws.taxonomy_children(pred_pos)), 2)

    def test_low_obvious_low_others_does_reject(self):
        """Control: when the OTHER factors are also low, the same low
        obvious does reject -- so it is the product, not a special-case
        on obvious, that decides."""
        m = _make_radix_model()
        cs = m.conceptualSpace
        cs.truth_criterion = 0.3
        cs._learn_score_children_in_codebook = lambda i1, i2: 0.5
        cs._learn_score_is_truth_obvious = lambda rel: 0.4
        cs._learn_score_resolves_contradiction = lambda rel: 0.5
        pred, a, b = _ideas(cs)
        # 0.5 * 0.4 * 0.5 = 0.1 < 0.3 -> reject.
        self.assertIsNone(cs._maybe_learn_relation(pred, a, b))


class TestLieCarriesBothTrust(unittest.TestCase):
    """End-to-end with REAL factors + REAL tetralemma trust: a relation
    over a contested TruthSet is learned and records a high-BOTH
    tetralemma tuple."""

    def test_contested_relation_learned_with_high_both(self):
        m = _make_radix_model()
        cs = m.conceptualSpace
        ws = cs.terminalSymbolSpace_ref
        tl = cs.symbolSpace.truth_layer
        cs.truth_criterion = 0.3
        D = int(cs.nDim)

        # Contested TruthSet: affirm AND deny the same concept ->
        # assess support 0.25 (low is_truth_obvious), conflict 1.0
        # (high resolves_contradiction).
        tl.clear()
        Dt = int(tl.truths.shape[-1])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tl.record(torch.ones(Dt), 1.0)
            tl.record(torch.ones(Dt), -1.0)
        assess = tl.assess()
        self.assertLess(assess["support"], 0.5,
                        "contested TruthSet should have low support")
        self.assertEqual(assess["conflict"], 1.0,
                         "affirm+deny -> conflict 1.0")

        # Make both ideas already-known concepts so children == 1.0.
        pred, a, b = _ideas(cs)
        ws.insert_whole(init_vec=a)
        ws.insert_whole(init_vec=b)
        children_factor = cs._learn_score_children_in_codebook(a, b)
        self.assertEqual(children_factor, 1.0)

        # Real learn-score: children(1.0) * obvious(0.25) *
        # resolves(1.0) = 0.25 < 0.3. To exercise the LIE path through
        # the real trust computation while still clearing the gate, drop
        # the criterion just below the real product so the REAL factors
        # (incl. the low support) decide acceptance.
        real_score = cs._compute_learn_score(pred, a, b)
        self.assertAlmostEqual(real_score, 0.25, places=6)
        cs.truth_criterion = 0.2  # 0.25 >= 0.2 -> accept the "lie".

        pred_pos = cs._maybe_learn_relation(pred, a, b)
        self.assertIsNotNone(
            pred_pos,
            "a contested relation grounded in known concepts must be "
            "learnable even with low is_truth_obvious")
        # It landed in the codebook taxonomy.
        self.assertEqual(len(ws.taxonomy_children(pred_pos)), 2)
        # And carries a tetralemma 4-tuple recording the conflict.
        trust = ws.meta_trust.get(pred_pos)
        self.assertIsNotNone(
            trust, "accepted relation must carry a tetralemma trust tuple")
        self.assertEqual(len(trust), 4)
        self.assertAlmostEqual(sum(trust), 1.0, places=5,
                               msg="trust tuple must sum to 1")
        t, f, both, n = trust
        # conflict 1.0 dominates the raw (t, f, b, n) -> b is the
        # largest corner (BOTH: the set both affirms AND denies).
        self.assertEqual(
            both, max(trust),
            f"a contested 'lie' must record BOTH as the dominant "
            f"corner; got (t,f,b,n)={trust}")
        self.assertGreater(both, 0.5,
                           f"BOTH weight should dominate; got {trust}")


if __name__ == "__main__":
    unittest.main()
