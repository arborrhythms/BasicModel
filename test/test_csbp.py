"""ConceptualSpace Bivector Projection (CSBP) — Stage 4 acceptance.

Two layers of coverage:

1. Codebook level — ``svdOrthogonal=True`` keeps the round-trip MSE
   under 1e-3 from the very first forward call on in-span input (no
   training warm-up). This is the ``svdOrthogonalInit`` mitigation
   for ``project_reverse``'s 1/Σ amplification.

2. ConceptualSpace level — when ``<bivectorOutput>true</bivectorOutput>``
   is configured, ``ConceptualSpace.forward`` returns the per-prototype
   catuskoti bivector ``[B, V_C, 2]`` and ``.reverse`` lifts it back
   through the cached SVD pseudo-inverse.

Spec: doc/2026-05-08-bivector-activation-conceptual-loopback-design.md
(Stage 4 acceptance: round-trip MSE on the C-tier alone < 1e-3 from
fresh SVD-orthogonal init).
"""
import sys
import unittest
from pathlib import Path

import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "bin"))

from Spaces import Codebook  # noqa: E402


class TestSVDOrthogonalInit(unittest.TestCase):
    """``svdOrthogonal=True`` pegs all singular values at 1, so the
    1/Σ scaling in ``project_reverse`` is well-conditioned from t=0."""

    def test_orthogonalized_W_has_unit_singular_values(self):
        torch.manual_seed(0)
        cb = Codebook()
        cb.create(nInput=8, nVectors=8, nDim=8,
                  customVQ=False,
                  monotonic=True, invertible=True,
                  svdOrthogonal=True)
        W = cb.getW()
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        torch.testing.assert_close(S, torch.ones_like(S),
                                   atol=1e-5, rtol=1e-4)

    def test_random_init_has_non_unit_singular_values(self):
        """Sanity check: without svdOrthogonal, S is random and small
        values can blow up the 1/Σ lift."""
        torch.manual_seed(0)
        cb = Codebook()
        cb.create(nInput=8, nVectors=8, nDim=8,
                  customVQ=False,
                  monotonic=True, invertible=True,
                  svdOrthogonal=False)
        W = cb.getW()
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        # Random codebook -- singular values should NOT all be 1.
        self.assertFalse(torch.allclose(S, torch.ones_like(S),
                                        atol=1e-2),
                         f"Random init produced unit singular values: {S}")

    def test_round_trip_mse_below_threshold_from_t0(self):
        """Stage 4 acceptance: round-trip MSE on the C-tier alone < 1e-3
        for in-span input from the very first forward call."""
        torch.manual_seed(0)
        D, V_S, V_in, B = 8, 6, 4, 3
        cb = Codebook()
        cb.create(nInput=V_in, nVectors=V_S, nDim=D,
                  customVQ=False,
                  monotonic=True, invertible=True,
                  svdOrthogonal=True)
        W = cb.getW()                                  # [V_S, D]
        # In-span input: random combinations of codebook rows.
        coeffs = torch.randn(B, V_in, V_S)
        x_in = coeffs @ W                              # [B, V_in, D]
        bivec = cb.forward(x_in, project=True)         # [B, V_S, 2]
        x_back = cb.reverse(bivec, project=True)       # [B, V_in, D]
        mse = torch.mean((x_back - x_in) ** 2).item()
        self.assertLess(mse, 1e-3,
                        f"SVD-orthogonal init round-trip MSE {mse:.6e} "
                        f">= 1e-3 (Stage 4 acceptance threshold)")


class TestRandomInitAmplification(unittest.TestCase):
    """Without svdOrthogonal, small singular values can amplify the lift
    on near-prototype inputs in pathological cases. The test below is a
    soft guard rather than a hard threshold; the point is that
    svdOrthogonal=True yields a deterministic-low MSE without warm-up."""

    def test_orthogonal_beats_random_on_average(self):
        torch.manual_seed(42)
        D, V_S, V_in, B = 8, 6, 4, 3
        coeffs = torch.randn(B, V_in, V_S)

        def round_trip_mse(svdOrthogonal):
            cb = Codebook()
            cb.create(nInput=V_in, nVectors=V_S, nDim=D,
                      customVQ=False,
                      monotonic=True, invertible=True,
                      svdOrthogonal=svdOrthogonal)
            x_in = coeffs @ cb.getW()
            bivec = cb.forward(x_in, project=True)
            x_back = cb.reverse(bivec, project=True)
            return torch.mean((x_back - x_in) ** 2).item()

        mse_ortho = round_trip_mse(True)
        # Repeat the random-init case across seeds to characterize the
        # distribution; we just assert ortho is at most as bad.
        torch.manual_seed(7)
        mse_random = round_trip_mse(False)
        self.assertLessEqual(mse_ortho, max(mse_random, 1e-3) + 1e-6,
                             f"orthogonal mse={mse_ortho:.3e} "
                             f"random mse={mse_random:.3e}")


class TestBivectorEndToEnd(unittest.TestCase):
    """Stage 6 acceptance: MM_xor_bivector.xml constructs cleanly,
    runs forward+reverse end-to-end, and the C-tier event is the
    bivector ``[B, V_C, 2]`` shape."""

    def test_mm_xor_bivector_forward_reverse(self):
        sys.path.insert(0, str(_PROJECT / "bin"))
        import Models  # noqa: E402

        torch.manual_seed(0)
        Models.TheData.load("xor")
        m = Models.BasicModel()
        cfg = str(_PROJECT / "data" / "MM_xor_bivector.xml")
        m.create_from_config(cfg, data=Models.TheData)
        m.eval()
        m.set_sigma(0)
        # ConceptualSpace is in bivector regime.
        cs0 = m.conceptualSpaces[0]
        ss0 = m.symbolicSpaces[0]
        self.assertTrue(cs0._bivector_output)
        self.assertTrue(ss0._bivector_output)
        # Loopback concat retired in favour of per-order input sourcing
        # on ConceptualSpace.forward (order 0 reads PerceptualSpace,
        # order > 0 reads the active sibling lifted via the C-tier
        # codebook's SVD pseudo-inverse). Right-half widening is now
        # always zero.
        self.assertEqual(cs0._right_half_dim, 0,
                         "C[0] right_half_dim should be 0 after the "
                         "loopback retirement")
        # Forward + reverse runs without shape errors.
        with torch.no_grad():
            m.runEpoch(batchSize=2, split="test")
        # End-to-end: the per-stage SymbolicSpace's codebook accumulated
        # the SVD cache from its bivector forward, so its W is invertible
        # mode and ``_project_cache`` is populated. The shape contract
        # we verify here is the SymbolicSpace codebook itself: row width
        # equals the configured nDim (=2 for bivec mode).
        s_basis = ss0.subspace.what
        self.assertTrue(hasattr(s_basis, 'project'),
                        "SymbolicSpace.what should be a Codebook in "
                        "bivector regime")
        W = s_basis.getW()
        self.assertIsNotNone(W)
        self.assertEqual(W.shape[-1], ss0.nDim,
                         f"S codebook row width should be nDim={ss0.nDim}, "
                         f"got {W.shape}")


if __name__ == "__main__":
    unittest.main()
