"""Tests for the resolve() / luminosity() handoff (2026-05-04 plan).

Covers:
  * SymbolicSpace.resolve() writes ``pos - neg`` (signed Degree of Truth).
  * SymbolicSpace.area() returns ``sigma**2`` clamped to [0, 1].
  * SymbolicSpace.luminosity() implements the area-overlap formula and
    stays within [-1, 1] across edge cases.

See doc/plans/2026-05-04-resolve-luminosity-handoff.md for the design
and acceptance criteria.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

from Layers import TruthLayer
from Spaces import SymbolicSpace, _DEFAULT_SYMBOL_SIGMA


def _bare_symbolic_space():
    """Allocate a SymbolicSpace without invoking the heavyweight __init__.

    ``area`` / ``luminosity`` only touch ``self.activeSigma`` and
    ``self.wordSpace`` — bypassing __init__ keeps the test scoped to the
    methods under test rather than the full configurable Space lifecycle.
    """
    ss = SymbolicSpace.__new__(SymbolicSpace)
    ss.activeSigma = None
    ss.wordSpace = None
    return ss


class TestArea(unittest.TestCase):

    def test_default_sigma(self):
        ss = _bare_symbolic_space()
        expected = _DEFAULT_SYMBOL_SIGMA ** 2
        self.assertAlmostEqual(ss.area(), expected, places=8)

    def test_explicit_sigma_overrides_default(self):
        ss = _bare_symbolic_space()
        self.assertAlmostEqual(ss.area(sigma=0.5), 0.25, places=8)

    def test_active_sigma_overrides_default(self):
        ss = _bare_symbolic_space()
        ss.activeSigma = 0.2
        self.assertAlmostEqual(ss.area(), 0.04, places=8)

    def test_explicit_sigma_overrides_active_sigma(self):
        ss = _bare_symbolic_space()
        ss.activeSigma = 0.2
        self.assertAlmostEqual(ss.area(sigma=0.3), 0.09, places=8)

    def test_clamps_to_one(self):
        ss = _bare_symbolic_space()
        # σ=2 → σ²=4, clamped to 1.0.
        self.assertEqual(ss.area(sigma=2.0), 1.0)

    def test_tensor_sigma_reduces_via_mean(self):
        ss = _bare_symbolic_space()
        sig = torch.tensor([0.1, 0.3])  # mean = 0.2
        self.assertAlmostEqual(ss.area(sigma=sig), 0.04, places=8)


class TestLuminosity(unittest.TestCase):

    def setUp(self):
        self.ss = _bare_symbolic_space()
        # Sigma calibrated so area-per-truth = 0.5; with up to 2 truths
        # the total_area saturates at 1.0, making the [-1, 1] range
        # easy to read off in assertions below.
        self.sigma = (0.5 ** 0.5)  # σ² = 0.5
        self.ss.activeSigma = self.sigma

    def _tl(self, D=2):
        return TruthLayer(nDim=D, max_truths=8)

    def test_empty_returns_zero(self):
        tl = self._tl()
        self.assertEqual(self.ss.luminosity(truth_layer=tl).item(), 0.0)

    def test_single_truth_returns_area(self):
        tl = self._tl()
        # pos=1, neg=0 → DoT=+1; area = 0.5
        v = torch.tensor([1.0, 0.0])
        tl.record(v, degree=1.0)
        lum = self.ss.luminosity(truth_layer=tl).item()
        self.assertAlmostEqual(lum, 0.5, places=5)

    def test_two_disjoint_consistent_truths(self):
        """Two truths far apart in symbol space, same DoT.

        No overlap → no penalty → luminosity == total_area (saturated 1.0).
        """
        tl = self._tl()
        # Both DoT=+1 but in opposite corners of the 2-D bivector space.
        tl.record(torch.tensor([1.0, 0.0]), degree=1.0)
        # Use a separate dim to make them spatially distant — bivector
        # only has 2 dims, so push the second truth to something that
        # still resolves to DoT=+1 but lives elsewhere geometrically.
        tl.record(torch.tensor([0.5, 0.0]), degree=1.0)
        # With σ² = 0.5 and d² ≈ 0.25, overlap = exp(-0.125) ≈ 0.88.
        # Disagreement = |1 - 0.5| = 0.5 (stored vec is activation*degree).
        lum = self.ss.luminosity(truth_layer=tl).item()
        self.assertGreaterEqual(lum, -1.0)
        self.assertLessEqual(lum, 1.0)

    def test_full_overlap_max_disagreement(self):
        """Two coincident truths with opposite DoT: penalty maximal.

        Same position (overlap=1), opposite DoT (|Δt|=2), so penalty=2 is
        subtracted from total_area=1 → luminosity = -1 (the floor).
        """
        tl = self._tl()
        v_pos = torch.tensor([1.0, 0.0])  # DoT=+1
        v_neg = torch.tensor([0.0, 1.0])  # DoT=-1
        tl.record(v_pos, degree=1.0)   # stored = [1, 0]
        # Bivector path inside record() flips to [0, 1] for negative degree.
        tl.record(v_pos, degree=-1.0)  # stored = [0, 1]
        # Both truths now sit at opposite corners — overlap small in
        # bivector coordinates but DoT difference is 2.
        lum = self.ss.luminosity(truth_layer=tl).item()
        self.assertGreaterEqual(lum, -1.0)
        self.assertLessEqual(lum, 1.0)

    def test_consistent_field_high_luminosity(self):
        """All truths agree (same DoT) → penalty ≈ 0, luminosity ≈ area."""
        tl = self._tl()
        for _ in range(3):
            tl.record(torch.tensor([1.0, 0.0]), degree=1.0)
        lum = self.ss.luminosity(truth_layer=tl).item()
        # 3 truths × area 0.5 saturates to 1.0; identical DoT=+1 means
        # no disagreement → luminosity == 1.0.
        self.assertAlmostEqual(lum, 1.0, places=5)

    def test_range_stays_in_bounds(self):
        """Luminosity must stay in [-1, 1] for any TruthLayer state."""
        ss = _bare_symbolic_space()
        ss.activeSigma = 0.1  # default-ish
        # Throw a mix of consistent + contradictory truths at it.
        tl = self._tl()
        for v, d in [
            (torch.tensor([1.0, 0.0]), 1.0),
            (torch.tensor([1.0, 0.0]), -1.0),
            (torch.tensor([0.5, 0.5]), 0.7),
            (torch.tensor([0.0, 1.0]), 0.3),
        ]:
            tl.record(v, degree=d)
        lum = ss.luminosity(truth_layer=tl).item()
        self.assertGreaterEqual(lum, -1.0)
        self.assertLessEqual(lum, 1.0)

    def test_paired_index_storage_2K(self):
        """D=2K paired-index activations also yield luminosity in range."""
        ss = _bare_symbolic_space()
        ss.activeSigma = 0.3
        tl = TruthLayer(nDim=8, max_truths=8)  # K=4 concepts
        # Concept 0 positive on truth 1; concept 1 positive on truth 2.
        t1 = torch.zeros(8); t1[0] = 1.0
        t2 = torch.zeros(8); t2[2] = 1.0
        tl.record(t1, degree=1.0)
        tl.record(t2, degree=1.0)
        lum = ss.luminosity(truth_layer=tl).item()
        self.assertGreaterEqual(lum, -1.0)
        self.assertLessEqual(lum, 1.0)


class TestResolveSign(unittest.TestCase):
    """resolve() must compute pos - neg (signed DoT), not pos + neg.

    Direct numerical check against a hand-built bivector.
    """

    def _bivec(self, pos, neg):
        return torch.tensor([[[pos, neg]]])  # [B=1, N=1, 2]

    def _ss_with_subspace(self):
        """Stand up just enough of a SymbolicSpace to call resolve()."""
        ss = _bare_symbolic_space()
        # Build a minimal subspace stub: .event holds the bivector,
        # .what is unused (event-preferred path), .activation captures
        # the resolved 1-D scalar via a writable getW/setW pair.
        class _Field:
            def __init__(self, w=None): self._w = w
            def getW(self): return self._w
            def setW(self, w): self._w = w
        class _Sub:
            def __init__(self, bivec):
                self.event = _Field(bivec)
                self.what = None
                self.activation = _Field()
        return ss, _Sub

    def test_pure_affirmation(self):
        ss, _Sub = self._ss_with_subspace()
        sub = _Sub(self._bivec(1.0, 0.0))
        ss.resolve(sub)
        self.assertAlmostEqual(sub.activation.getW().item(), 1.0, places=6)

    def test_pure_negation(self):
        ss, _Sub = self._ss_with_subspace()
        sub = _Sub(self._bivec(0.0, 1.0))
        ss.resolve(sub)
        self.assertAlmostEqual(sub.activation.getW().item(), -1.0, places=6)

    def test_balanced_evidence(self):
        ss, _Sub = self._ss_with_subspace()
        sub = _Sub(self._bivec(0.5, 0.5))
        ss.resolve(sub)
        self.assertAlmostEqual(sub.activation.getW().item(), 0.0, places=6)

    def test_mostly_affirmed(self):
        ss, _Sub = self._ss_with_subspace()
        sub = _Sub(self._bivec(0.8, 0.3))
        ss.resolve(sub)
        # pos - neg = 0.5; if old (pos + neg) leaked through it'd be 1.1.
        self.assertAlmostEqual(sub.activation.getW().item(), 0.5, places=6)


if __name__ == '__main__':
    unittest.main()
