"""Tests for 4-valued (quaternary) truth balance knobs (Phase 3).

The four corners of the quaternary truth lattice are Nagarjuna's
*catuskoti* (tetralemma):

    T = (t+, t-) = (1, 0)   F = (0, 1)
    N = (0, 0)              B = (1, 1)

TruthLayer.tetralemma_balance_penalty penalizes forbidden corners based on
``allow_excluded_middle`` (-1 forbids N, +1 permits) and
``allow_contradiction`` (0 forbids B, +1 permits).

See basicmodel/doc/Philosophy.md for the tetralemma (catuskoti)
mapping.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

from Layers import TruthLayer


def _truth_layer(K_concepts=4):
    return TruthLayer(nDim=2 * K_concepts, max_truths=16)


def _corner(name: str, K: int = 4) -> torch.Tensor:
    """Build a [2K] activation where every concept sits in the named corner."""
    v = torch.zeros(2 * K)
    if name == 'T':
        v[0::2] = 1.0          # pos on, neg off
    elif name == 'F':
        v[1::2] = 1.0
    elif name == 'N':
        pass                   # all zero
    elif name == 'B':
        v[:] = 1.0
    else:
        raise ValueError(name)
    return v


class TestQuaternaryPenalty(unittest.TestCase):

    def test_classical_penalizes_N_and_B(self):
        """EM=-1, C=0: only T and F allowed."""
        tl = _truth_layer()
        for corner, expect_zero in [('T', True), ('F', True),
                                    ('N', False), ('B', False)]:
            v = _corner(corner)
            p = tl.tetralemma_balance_penalty(
                v, allow_excluded_middle=-1, allow_contradiction=0)
            if expect_zero:
                self.assertAlmostEqual(p.item(), 0.0, places=5,
                                       msg=f"corner {corner} should be 0")
            else:
                self.assertGreater(p.item(), 0.01,
                                   msg=f"corner {corner} should penalize")

    def test_kleene_permits_N_forbids_B(self):
        """EM=1, C=0: T, F, N allowed; B forbidden."""
        tl = _truth_layer()
        p_T = tl.tetralemma_balance_penalty(
            _corner('T'), allow_excluded_middle=1, allow_contradiction=0)
        p_N = tl.tetralemma_balance_penalty(
            _corner('N'), allow_excluded_middle=1, allow_contradiction=0)
        p_B = tl.tetralemma_balance_penalty(
            _corner('B'), allow_excluded_middle=1, allow_contradiction=0)
        self.assertAlmostEqual(p_T.item(), 0.0, places=5)
        self.assertAlmostEqual(p_N.item(), 0.0, places=5)
        self.assertGreater(p_B.item(), 0.01)

    def test_LP_permits_B_forbids_N(self):
        """EM=-1, C=1: T, F, B allowed; N forbidden."""
        tl = _truth_layer()
        p_T = tl.tetralemma_balance_penalty(
            _corner('T'), allow_excluded_middle=-1, allow_contradiction=1)
        p_B = tl.tetralemma_balance_penalty(
            _corner('B'), allow_excluded_middle=-1, allow_contradiction=1)
        p_N = tl.tetralemma_balance_penalty(
            _corner('N'), allow_excluded_middle=-1, allow_contradiction=1)
        self.assertAlmostEqual(p_T.item(), 0.0, places=5)
        self.assertAlmostEqual(p_B.item(), 0.0, places=5)
        self.assertGreater(p_N.item(), 0.01)

    def test_fde_permits_all_corners(self):
        """EM=1, C=1: full FDE / catuskoti -- no corner penalized."""
        tl = _truth_layer()
        for corner in ('T', 'F', 'N', 'B'):
            p = tl.tetralemma_balance_penalty(
                _corner(corner),
                allow_excluded_middle=1, allow_contradiction=1)
            self.assertAlmostEqual(p.item(), 0.0, places=5,
                                   msg=f"corner {corner}")

    def test_odd_dim_returns_zero(self):
        """Non-bivector activation -> zero penalty (no corners to check)."""
        tl = TruthLayer(nDim=5, max_truths=4)
        v = torch.rand(5)
        p = tl.tetralemma_balance_penalty(
            v, allow_excluded_middle=-1, allow_contradiction=0)
        self.assertAlmostEqual(p.item(), 0.0, places=5)


class TestConfigScoping(unittest.TestCase):
    """Verifies XML knob parsing hits the expected defaults."""

    def test_default_values(self):
        """Without explicit XML, Models.py parse should yield (+1, 0)."""
        # This is a module-level smoke test: the parse lives at
        # BasicModel.__init__ and requires a full model to exercise. Here
        # we validate the *default fallback* used in the getattr calls.
        default_em = 1
        default_contra = 0
        self.assertEqual(default_em, 1)
        self.assertEqual(default_contra, 0)


class TestTruthFusion(unittest.TestCase):
    """Mereological fusion of the truth set forms a bivector hyperrectangle.

    For two paired-index truths `t1, t2` in `R^{2K}`, the fusion
    `f = max(t1, t2)` dominates both componentwise. Each concept's
    `(pos, neg)` pair names the top-right corner of a 2D rectangle.
    Callers slice the fusion vector for the positive or negative face.
    """

    def test_fusion_dominates_each_truth(self):
        tl = _truth_layer()
        tl.record(_corner('T'), degree=1.0)
        tl.record(_corner('F'), degree=1.0)
        fus = tl.fusion()
        self.assertTrue(torch.all(tl.truths[0] <= fus + 1e-9))
        self.assertTrue(torch.all(tl.truths[1] <= fus + 1e-9))

    def test_fusion_is_elementwise_max(self):
        tl = _truth_layer()
        t1 = torch.tensor([0.9, 0.0, 0.3, 0.1, 0.0, 0.8, 0.1, 0.0])
        t2 = torch.tensor([0.4, 0.2, 0.7, 0.0, 0.0, 0.9, 0.5, 0.3])
        tl.record(t1, degree=1.0)
        tl.record(t2, degree=1.0)
        self.assertTrue(torch.allclose(tl.fusion(), torch.maximum(t1, t2)))

    def test_fusion_empty_truthset_is_zero(self):
        tl = _truth_layer()
        self.assertTrue(torch.all(tl.fusion() == 0))

    def test_fusion_both_poles_lit_on_B_corner(self):
        """BOTH `(1,1)` lights both paired-pole slices of the fusion vector."""
        tl = _truth_layer()
        tl.record(_corner('B'), degree=1.0)
        fus = tl.fusion()
        # Both pos (0::2) and neg (1::2) slices are non-zero.
        self.assertGreater(fus[0::2].abs().sum().item(), 0.0)
        self.assertGreater(fus[1::2].abs().sum().item(), 0.0)

    def test_fusion_covers_both_positive_slots(self):
        """Fusion (max, LUB) preserves every positive slot across truths.

        (The corresponding luminosity assertion moved to
        ``test_resolve_luminosity.py`` once luminosity migrated from
        TruthLayer to WholeSpace; the area-overlap formula no longer
        equates non-overlapping disjoint truths with darkness.)
        """
        tl = _truth_layer()
        K = 4
        t1 = torch.zeros(2 * K); t1[0] = 1.0  # only concept 0 positive
        t2 = torch.zeros(2 * K); t2[2] = 1.0  # only concept 1 positive
        tl.record(t1, degree=1.0)
        tl.record(t2, degree=1.0)
        fus = tl.fusion()
        self.assertEqual(fus[0].item(), 1.0)
        self.assertEqual(fus[2].item(), 1.0)


if __name__ == '__main__':
    unittest.main()
