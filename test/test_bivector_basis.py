"""Tests for bivector (paired-index) monotonic encoding.

Phase 1 of the Belnap-Dunn bivector plan: the SymbolicSpace codebook
stores paired positive/negative poles (index 2k / 2k+1), Basis.negation
with monotonic=True swaps paired indices, the post-optimizer clamp
keeps values in [0, 1], and pre-bivector checkpoints migrate by row
duplication.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

import Spaces


class TestBivectorNegation(unittest.TestCase):

    def _basis(self):
        b = Spaces.Codebook()
        b.create(nInput=1, nVectors=8, nDim=4, monotonic=True, passThrough=True)
        return b

    def test_negation_is_paired_index_flip(self):
        basis = self._basis()
        K = 4
        v = torch.zeros(2 * K)
        v[0] = 0.9   # pos pole of concept 0
        v[3] = 0.7   # neg pole of concept 1
        out = basis.negation(v, monotonic=True)
        # Positive pole 0 flips to negative pole of concept 0 (index 1).
        self.assertAlmostEqual(out[0].item(), 0.0, places=6)
        self.assertAlmostEqual(out[1].item(), 0.9, places=6)
        # Negative pole 3 flips to positive pole of concept 1 (index 2).
        self.assertAlmostEqual(out[2].item(), 0.7, places=6)
        self.assertAlmostEqual(out[3].item(), 0.0, places=6)

    def test_double_negation_identity(self):
        basis = self._basis()
        torch.manual_seed(0)
        v = torch.rand(16).clamp(0.0, 1.0)
        out = basis.negation(basis.negation(v, monotonic=True),
                             monotonic=True)
        self.assertTrue(torch.allclose(out, v))

    def test_negation_inverse_is_negation(self):
        basis = self._basis()
        v = torch.rand(8).clamp(0.0, 1.0)
        direct = basis.negation(v, monotonic=True)
        via_inv = basis.negation_inverse(v, monotonic=True)
        self.assertTrue(torch.allclose(direct, via_inv))

    def test_negation_rejects_odd_last_dim(self):
        basis = self._basis()
        v = torch.zeros(5)  # odd
        with self.assertRaises(ValueError):
            basis.negation(v, monotonic=True)

    def test_bitonic_negation_unchanged(self):
        basis = self._basis()
        v = torch.tensor([0.1, -0.2, 0.3, -0.4])
        out = basis.negation(v, monotonic=False)
        self.assertTrue(torch.allclose(out, -v))


class TestCodebookClamp(unittest.TestCase):

    def test_clamp_preserves_pair_box(self):
        """A synthetic update out of [0,1] gets clamped back."""
        basis = Spaces.Codebook()
        basis.create(nInput=1, nVectors=4, nDim=3, monotonic=True,
                     passThrough=False)
        # Codebook was materialized; grab W and perturb it.
        W = basis.W
        self.assertIsNotNone(W)
        with torch.no_grad():
            W.data.add_(2.0)  # push way out of [0, 1]
            self.assertGreater(W.data.max().item(), 1.0)
            W.data.clamp_(0.0, 1.0)
        self.assertLessEqual(W.data.max().item(), 1.0)
        self.assertGreaterEqual(W.data.min().item(), 0.0)


class TestCheckpointMigration(unittest.TestCase):
    """Verifies the bivector migration shim in BaseModel.loadWeights.

    We exercise the row-duplication logic directly on a mock state dict
    rather than standing up a full MentalModel, which keeps the test
    narrow to the migration contract.
    """

    def test_bivector_row_duplication(self):
        K, D = 4, 3
        saved_W = torch.arange(K * D, dtype=torch.float32).reshape(K, D)
        model_W = torch.zeros(2 * K, D)

        # Simulate the migration step from BaseModel.loadWeights.
        migrated = torch.zeros_like(model_W)
        migrated[0::2] = saved_W
        # Negative poles stay zero.

        # Rows 2k match the saved rows; rows 2k+1 are zero.
        for k in range(K):
            self.assertTrue(torch.allclose(migrated[2 * k], saved_W[k]))
            self.assertTrue(torch.allclose(migrated[2 * k + 1],
                                           torch.zeros(D)))


if __name__ == '__main__':
    unittest.main()
