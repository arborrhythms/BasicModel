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

from Spaces import ProjectionBasis  # noqa: E402

# 2026-05-13: ``svdOrthogonal`` was a Codebook init flag that pegged
# all singular values of W to 1 so the legacy ``project_reverse`` SVD
# 1/Σ scaling was well-conditioned from t=0.  The bivector projection
# surface now lives on ``ProjectionBasis`` whose ``InvertibleLinearLayer``
# parameterization is structurally well-conditioned (LDU triangular
# solve).  Tests below preserved as a sanity check on the new path.


class TestSVDOrthogonalInit(unittest.TestCase):
    """ProjectionBasis default init: LDU with raw_L=0, raw_U=0, d=1
    produces W = rectangular identity, which has unit singular values
    on the square block.  No explicit ``svdOrthogonal`` flag needed."""

    def test_default_W_has_unit_singular_values(self):
        cb = ProjectionBasis()
        cb.create(nInput=8, nVectors=8, nDim=8)
        W = cb.getW()
        _, S, _ = torch.linalg.svd(W, full_matrices=False)
        torch.testing.assert_close(S, torch.ones_like(S),
                                   atol=1e-5, rtol=1e-4)

    def test_round_trip_mse_below_threshold_from_t0(self):
        """Round-trip MSE on identity-init codebook is exactly zero
        for V=1 inputs (the orthographic-decode case)."""
        D, V_S, B = 8, 8, 3
        cb = ProjectionBasis()
        cb.create(nInput=1, nVectors=V_S, nDim=D)
        W = cb.getW()                                  # [V_S, D]
        # Single-vector inputs: in-span by construction (V=1).
        coeffs = torch.randn(B, 1, V_S)
        x_in = coeffs @ W                              # [B, 1, D]
        bivec = cb.forward(x_in)
        x_back = cb.reverse(bivec, V=1)
        mse = torch.mean((x_back - x_in) ** 2).item()
        self.assertLess(mse, 1e-3,
                        f"LDU-parameterized round-trip MSE {mse:.6e} "
                        f">= 1e-3")


class TestRandomInitAmplification(unittest.TestCase):
    """Retired 2026-05-13: ``svdOrthogonal`` flag no longer applies to
    ProjectionBasis (the LDU init is deterministic and always well-
    conditioned).  This class is retained as a placeholder; its
    historical purpose was to compare svdOrthogonal=True vs random
    init for the legacy Codebook.project_reverse path."""

    @unittest.skip("Retired: svdOrthogonal flag does not apply to "
                   "ProjectionBasis (LDU is structurally well-conditioned).")
    def test_orthogonal_beats_random_on_average(self):
        pass

    def _unused_legacy_body(self):  # pragma: no cover -- kept for diff context
        D, V_S, V_in, B = 8, 6, 4, 3
        coeffs = torch.randn(B, V_in, V_S)

        def round_trip_mse(svdOrthogonal):
            return 0.0

        mse_ortho = round_trip_mse(True)
        # Repeat the random-init case across seeds to characterize the
        # distribution; we just assert ortho is at most as bad.

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
        # End-to-end: in the bivector regime, ``SymbolicSpace.subspace.what``
        # is a ``ProjectionBasis`` (added 2026-05-13) whose
        # ``InvertibleLinearLayer`` parameterizes W as an LDU
        # factorization.  The shape contract we verify here is that
        # the codebook's row width equals the configured nDim.
        s_basis = ss0.subspace.what
        self.assertEqual(type(s_basis).__name__, 'ProjectionBasis',
                         "SymbolicSpace.what should be ProjectionBasis "
                         "in bivector regime")
        W = s_basis.getW()
        self.assertIsNotNone(W)
        self.assertEqual(W.shape[-1], ss0.nDim,
                         f"S codebook row width should be nDim={ss0.nDim}, "
                         f"got {W.shape}")


if __name__ == "__main__":
    unittest.main()
