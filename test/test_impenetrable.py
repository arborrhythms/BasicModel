"""Tests for ImpenetrableLayer -- mereological separation regularizer.

Five-relations design: each ordered pair (i, j) of codebook rows is
classified via clipped-cosine parthood into one of {disjoint, part_ij,
part_ji, equal, overlap}. The penalty is
``overlap_strength(i, j) * |trust(i) - trust(j)|`` where
``overlap_strength = min(P[i,j], P[j,i]) * (1 - max(P[i,j], P[j,i])**k)``
damps to zero as the pair approaches identity (equal). See
basicmodel/doc/BuddhistParallels.md for the 4-valued (quaternary) truth
semantics the codebook underwrites.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

import Spaces
from Layers import ImpenetrableLayer


def _basis(K=8, D=4):
    b = Spaces.Codebook()
    b.create(nInput=1, nVectors=K, nDim=D, monotonic=True)
    return b


class _FakeVQ:
    """Minimal VQ stand-in carrying an EMA `cluster_size` trust signal."""
    def __init__(self, cluster_size):
        self.cluster_size = torch.as_tensor(cluster_size, dtype=torch.float32)


def _attach_vq(basis, cluster_size):
    # Codebook.create installs a real ``VectorQuantize`` nn.Module on
    # ``basis.vq`` (passThrough was retired in Stage 1). Drop the
    # registered submodule first so the plain-attribute ``_FakeVQ`` lands
    # without nn.Module's ``_modules`` slot rejecting a non-Module.
    if "vq" in basis._modules:
        del basis._modules["vq"]
    basis.vq = _FakeVQ(cluster_size)
    return basis


class TestOverlapPenalty(unittest.TestCase):
    """`overlap * |trust-diff|` is the only source of penalty beyond variance."""

    @classmethod
    def setUpClass(cls):
        # ``last_relation_counts`` is gated behind ``util.MODEL_DEBUG``
        # (the per-bucket sum().item() calls are host syncs that would
        # break CUDA-graph capture in non-debug runs). Enable debug for
        # these tests since the counts are what they assert.
        import util
        cls._prev_debug = util.MODEL_DEBUG
        util.MODEL_DEBUG = True

    @classmethod
    def tearDownClass(cls):
        import util
        util.MODEL_DEBUG = cls._prev_debug

    def test_disjoint_codebook_zero_loss(self):
        """Orthogonal rows -- no overlap -> zero overlap loss regardless of trust."""

        K, D = 4, 4
        cb = torch.eye(K, D)
        basis = _attach_vq(_basis(K=K, D=D), [1.0, 2.0, 3.0, 4.0])
        layer = ImpenetrableLayer(
            overlap_weight=1.0, variance_floor=0.0)
        loss = layer(cb, basis)
        self.assertLess(loss.item(), 1e-4)
        counts = layer.last_relation_counts
        self.assertIsNotNone(counts)
        # All off-diagonal pairs disjoint -> K*(K-1) disjoint pairs.
        self.assertEqual(counts["disjoint"], K * (K - 1))
        self.assertEqual(counts["overlap"], 0)

    def test_equal_rows_damped_to_zero_loss(self):
        """Rows (nearly) identical -- damp factor zeroes the overlap penalty."""
        K, D = 4, 4
        cb = torch.zeros(K, D)
        cb[0] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        cb[1] = torch.tensor([1.0, 1.0, 0.0, 0.0])  # identical to row 0
        cb[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        cb[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        # Mismatched trust on the equal pair; damp factor must still zero out.
        basis = _attach_vq(_basis(K=K, D=D), [9.0, 1.0, 1.0, 1.0])
        layer = ImpenetrableLayer(
            overlap_weight=1.0, variance_floor=0.0,
            equal_suppression=4.0)
        loss = layer(cb, basis)
        self.assertLess(loss.item(), 5e-3)

    def test_overlap_matched_trust_zero_loss(self):
        """Overlapping rows with equal VQ counts -> |trust-diff| = 0 -> zero loss."""
        K, D = 3, 4
        cb = torch.zeros(K, D)
        cb[0] = torch.tensor([1.0, 0.5, 0.0, 0.0])
        cb[1] = torch.tensor([0.5, 1.0, 0.0, 0.0])  # partial overlap with row 0
        cb[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        # All trusts equal -> diff 0 everywhere.
        basis = _attach_vq(_basis(K=K, D=D), [1.0, 1.0, 1.0])
        layer = ImpenetrableLayer(
            overlap_weight=1.0, variance_floor=0.0)
        loss = layer(cb, basis)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_overlap_mismatched_trust_penalized(self):
        """Same shape, different VQ counts -> overlap loss > 0."""
        K, D = 3, 4
        cb = torch.zeros(K, D)
        cb[0] = torch.tensor([1.0, 0.5, 0.0, 0.0])
        cb[1] = torch.tensor([0.5, 1.0, 0.0, 0.0])  # partial overlap
        cb[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        # Very different trusts for the overlapping pair.
        basis = _attach_vq(_basis(K=K, D=D), [10.0, 1.0, 1.0])
        layer = ImpenetrableLayer(
            overlap_weight=1.0, variance_floor=0.0)
        loss = layer(cb, basis)
        self.assertGreater(loss.item(), 0.0)
        counts = layer.last_relation_counts
        self.assertIsNotNone(counts)
        self.assertGreaterEqual(counts["overlap"], 2)  # (0,1) and (1,0)

    def test_relation_counts_partition_non_diagonal(self):
        """Sum of the five relation counts equals K*(K-1) (all ordered off-diagonal pairs)."""

        K, D = 5, 4
        cb = torch.randn(K, D)
        basis = _attach_vq(_basis(K=K, D=D), [1.0] * K)
        layer = ImpenetrableLayer(
            overlap_weight=1.0, variance_floor=0.0)
        layer(cb, basis)
        counts = layer.last_relation_counts
        self.assertIsNotNone(counts)
        total = sum(counts.values())
        self.assertEqual(total, K * (K - 1))


class TestVarianceFloor(unittest.TestCase):

    def test_collapse_triggers_variance_penalty(self):
        K, D = 4, 3
        cb = torch.full((K, D), 0.5)  # all rows identical -> std = 0
        basis = _basis(K=K, D=D)
        layer = ImpenetrableLayer(
            overlap_weight=0.0, variance_floor=0.1)
        loss = layer(cb, basis)
        self.assertAlmostEqual(loss.item(), 0.1, places=5)

    def test_well_spread_no_penalty(self):

        K, D = 8, 5
        cb = torch.rand(K, D)
        basis = _basis(K=K, D=D)
        layer = ImpenetrableLayer(
            overlap_weight=0.0, variance_floor=0.01)
        loss = layer(cb, basis)
        # Random rows should easily clear a floor of 0.01.
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestDisabled(unittest.TestCase):

    def test_disabled_short_circuits(self):
        """enabled=False -> loss=0 and basis is never accessed."""
        class _FailBasis:
            def part(self, *a, **kw):
                raise AssertionError("basis.part must not be called when disabled")
            vq = None
        cb = torch.rand(4, 3)
        layer = ImpenetrableLayer(
            overlap_weight=1.0, variance_floor=0.1, enabled=False)
        self.assertAlmostEqual(layer(cb, _FailBasis()).item(), 0.0, places=7)

    def test_all_zero_weights_returns_zero(self):
        basis = _basis(K=4, D=3)
        cb = torch.rand(4, 3)
        layer = ImpenetrableLayer(
            overlap_weight=0.0, variance_floor=0.0)
        self.assertAlmostEqual(layer(cb, basis).item(), 0.0, places=7)


class TestWiring(unittest.TestCase):
    """`ImpenetrableLayer` accessor contract on null inputs."""

    def test_returns_zero_tensor_without_codebook(self):
        class _FakeBasis:
            vq = None
            def getW(self):
                return None
            def part(self, *a, **kw):
                raise AssertionError("should not be called when codebook is None")
        layer = ImpenetrableLayer(overlap_weight=0.5, variance_floor=0.0)
        result = layer(None, _FakeBasis())
        self.assertAlmostEqual(result.item(), 0.0, places=7)


if __name__ == '__main__':
    unittest.main()
