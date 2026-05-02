"""Antipode-aware matmul lookup vs broadcast torus distance.

``_pole_aligned_score`` replaces ``_wrapped_mse_score`` at argmax-
only call sites. The score values differ (matmul abs vs negative
wrapped MSE), but the argmax over the codebook V dim must match
for unit-norm codebooks under torus / antipode-quotient geometry.

This test pins the equivalence on synthetic codebooks where both
metrics agree by construction (unit-norm vectors, well-separated).
"""

import sys
import unittest
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

from embed import _pole_aligned_score, _wrapped_mse_score


def _unit_norm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class TestPoleAlignedShape(unittest.TestCase):
    def test_returns_score_per_codebook_entry(self):
        a = torch.randn(3, 5, 6)
        b = torch.randn(128, 6)
        s = _pole_aligned_score(a, b)
        self.assertEqual(s.shape, (3, 5, 128))
        self.assertTrue(torch.isfinite(s).all())
        self.assertTrue((s >= 0).all())  # |inner_product| >= 0

    def test_handles_width_mismatch(self):
        # Codebook can be wider than query; helper truncates to min D.
        a = torch.randn(2, 4)
        b = torch.randn(64, 6)
        s = _pole_aligned_score(a, b)
        self.assertEqual(s.shape, (2, 64))


class TestPoleAlignedArgmax(unittest.TestCase):
    """Argmax must match _wrapped_mse_score on well-separated unit-norm
    codebooks. Both are monotone-decreasing in angular distance under
    antipode equivalence for normalized vectors."""

    def test_argmax_matches_wrapped_mse_on_unit_codebook(self):
        torch.manual_seed(0)
        # Generate a unit-norm codebook far from the antipode boundary
        # (so wrap doesn't kick in) and queries that are perturbations
        # of codebook entries.
        D, V, N = 8, 256, 32
        cb = _unit_norm(torch.randn(V, D))
        # Each query is a perturbed copy of a random codebook entry.
        target = torch.randint(0, V, (N,))
        noise = 0.05 * torch.randn(N, D)
        queries = _unit_norm(cb[target] + noise)

        s_pole = _pole_aligned_score(queries, cb)
        s_mse = _wrapped_mse_score(queries.unsqueeze(1), cb.unsqueeze(0))
        self.assertEqual(s_pole.shape, s_mse.shape)
        # Argmax under both must recover the target index.
        self.assertTrue(torch.equal(s_pole.argmax(dim=-1), target))
        self.assertTrue(torch.equal(s_mse.argmax(dim=-1), target))

    def test_pole_aware_treats_negation_as_same(self):
        # The negation of an entry should score identically to the
        # entry itself -- that is the defining property of the NEG-
        # quotient (pole-aware) lookup. Distinct from the *antipode*
        # of an entry, which is the furthest point in the manifold
        # and is unrelated to this collapse.
        D, V = 6, 64
        cb = _unit_norm(torch.randn(V, D))
        s_orig = _pole_aligned_score(cb, cb)             # [V, V]
        s_neg = _pole_aligned_score(-cb, cb)             # [V, V]
        self.assertTrue(torch.allclose(s_orig, s_neg, atol=1e-6))


class TestArgmaxParityForLookupShapes(unittest.TestCase):
    """Mirror the actual call-site shapes used by ``Basis._snap_content``
    (``[N_active, V]`` argmax) and ``Basis._nearest_idx`` (``[1, V]``
    argmax). On a perturbed unit-norm codebook the two metrics must
    agree on argmax with high probability."""

    def test_snap_content_call_shape(self):
        torch.manual_seed(0)
        D, V, N_active = 6, 256, 64
        cb = _unit_norm(torch.randn(V, D))
        # ``_snap_content`` truncates to ``nWhat`` columns; mirror that.
        flat_nonzero = _unit_norm(cb[torch.randint(0, V, (N_active,))]
                                  + 0.05 * torch.randn(N_active, D))
        s_pole = _pole_aligned_score(flat_nonzero, cb)
        s_mse = _wrapped_mse_score(
            flat_nonzero.unsqueeze(1), cb.unsqueeze(0))
        idx_pole = s_pole.argmax(dim=1)
        idx_mse = s_mse.argmax(dim=1)
        agree = (idx_pole == idx_mse).float().mean().item()
        self.assertGreater(agree, 0.95,
                           f"argmax agreement {agree:.2f} below threshold")

    def test_nearest_idx_call_shape(self):
        torch.manual_seed(0)
        D, V = 6, 1024
        cb = _unit_norm(torch.randn(V, D))
        target = torch.randint(0, V, (1,))
        vec = _unit_norm(cb[target] + 0.01 * torch.randn(1, D))
        s_pole = _pole_aligned_score(vec, cb)
        s_mse = _wrapped_mse_score(vec.unsqueeze(0), cb)
        self.assertEqual(s_pole.argmax().item(), s_mse.argmax().item())


if __name__ == "__main__":
    unittest.main()
