"""Stage 6 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
butterfly cascade mode on the four tetralemma grammar ops.

Stage 5 added butterfly machinery to the ``GrammarLayer`` base class.
Stage 6 adds per-class ``_butterfly_pair_op`` overrides for the four
tetralemma operators so each participates in cross-STM cascade
computation with its own semantics:

  * ``IntersectionLayer`` -- C-tier bivector RadMin / lattice min.
  * ``UnionLayer``        -- C-tier bivector RadMax / lattice max.
  * ``ConjunctionLayer``  -- S-tier monotonic min on post-codebook
                              scalar activation.
  * ``DisjunctionLayer``  -- S-tier monotonic max on post-codebook
                              scalar activation.

These ops are all lossy folds. The butterfly cascade lifts each to
a packed cross-STM aggregation: at each level, adjacent slots are
fed through a per-pair op that (1) weights the packed [a, b] pair
by the per-node ``[2D, 2D]`` weight, (2) computes the op kernel
across the two halves, (3) broadcasts the result back into the
packed 2D form.

Identity notion (lossy ops): a constant input (``x[i] == x[j]``
for all i, j) is preserved -- the op is idempotent on the diagonal.
``forward(x) == x`` is NOT a meaningful contract for general x
when the cascade fold is genuinely lossy.

Roundtrip notion (lossy ops): ``reverse(forward(x))`` returns a
broadcast approximation -- shape matches and values are in the
valid range; exact recovery is not possible.
"""

import math
import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

from Layers import (
    GrammarLayer,
    IntersectionLayer,
    UnionLayer,
    ConjunctionLayer,
    DisjunctionLayer,
    Ops,
)


_TOL = 1e-4


# =====================================================================
# Common helpers
# =====================================================================
def _constant_input(B, N, D, value=0.3, dtype=torch.float32):
    """Build a [B, N, D] tensor whose every slot holds the same value."""
    return torch.full((B, N, D), value, dtype=dtype)


def _random_input(B, N, D, seed=0, scale=0.5):
    """Build a [B, N, D] random tensor scaled into (-1, 1)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(B, N, D, generator=g) * 2 - 1) * scale


# =====================================================================
# IntersectionLayer butterfly
# =====================================================================
class TestIntersectionLayerButterfly(unittest.TestCase):
    """C-tier RadMin/lattice min butterfly cascade."""

    def test_constructs_with_butterfly(self):
        D, N = 3, 8
        layer = IntersectionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), N)

    def test_constructs_monotonic(self):
        D, N = 3, 4
        layer = IntersectionLayer(
            nInput=D, nOutput=D, butterfly=True, N=N, monotonic=True)
        self.assertTrue(layer.butterfly)
        self.assertTrue(layer.monotonic)

    def test_butterfly_W_shape(self):
        D, N = 3, 8
        layer = IntersectionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        expected = (int(math.log2(N)), N // 2, 2 * D, 2 * D)
        self.assertEqual(tuple(layer.butterfly_W.shape), expected)

    def test_identity_on_constant_input(self):
        """Idempotent fold: constant input is preserved."""
        D, N = 3, 8
        layer = IntersectionLayer(nInput=D, nOutput=D, butterfly=True, N=N,
                                  monotonic=True)
        x = _constant_input(2, N, D, value=0.4)
        y = layer.forward(x)
        self.assertEqual(y.shape, x.shape)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL,
                        f"IntersectionLayer butterfly identity-on-const "
                        f"failed: err={err}")

    def test_forward_shape_n8(self):
        D, N = 3, 8
        layer = IntersectionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _random_input(2, N, D, seed=1)
        y = layer.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_reverse_roundtrip_shape(self):
        """Lossy roundtrip: shape preserved, values in range."""
        D, N = 3, 8
        layer = IntersectionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _random_input(2, N, D, seed=2)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        self.assertEqual(x_rec.shape, x.shape)
        # Lossy contract: reverse is a broadcast approximation, not exact.
        # Just check it didn't blow up.
        self.assertFalse(torch.isnan(x_rec).any().item())
        self.assertFalse(torch.isinf(x_rec).any().item())

    def test_pair_op_semantics_monotonic(self):
        """Per-pair op should compute min element-wise on the two halves
        at identity W init."""
        D, N = 3, 2
        layer = IntersectionLayer(nInput=D, nOutput=D, butterfly=True, N=N,
                                  monotonic=True)
        # N=2 -> one level, one node, no permutation effect.
        # Slot 0 = a, slot 1 = b. After identity W and min op,
        # both halves should equal min(a, b) per channel.
        a = torch.tensor([[[0.3, 0.5, -0.1]]])  # [B=1, N=1, D=3] won't fit; use N=2
        x = torch.tensor([[[0.3, 0.5, -0.1],
                           [0.1, 0.7,  0.2]]])  # [B=1, N=2, D=3]
        y = layer.forward(x)
        # min(x[0], x[1]) elementwise = [0.1, 0.5, -0.1]
        expected_min = torch.minimum(x[:, 0, :], x[:, 1, :])
        # Both output positions should hold the min (broadcast).
        self.assertTrue(torch.allclose(y[:, 0, :], expected_min, atol=1e-5),
                        f"position 0 != min(a,b): {y[:, 0, :]} vs {expected_min}")
        self.assertTrue(torch.allclose(y[:, 1, :], expected_min, atol=1e-5),
                        f"position 1 != min(a,b): {y[:, 1, :]} vs {expected_min}")


# =====================================================================
# UnionLayer butterfly
# =====================================================================
class TestUnionLayerButterfly(unittest.TestCase):
    """C-tier RadMax/lattice max butterfly cascade."""

    def test_constructs_with_butterfly(self):
        D, N = 3, 8
        layer = UnionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), N)

    def test_butterfly_W_shape(self):
        D, N = 3, 4
        layer = UnionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        expected = (int(math.log2(N)), N // 2, 2 * D, 2 * D)
        self.assertEqual(tuple(layer.butterfly_W.shape), expected)

    def test_identity_on_constant_input(self):
        """Idempotent fold: constant input is preserved."""
        D, N = 3, 8
        layer = UnionLayer(nInput=D, nOutput=D, butterfly=True, N=N,
                          monotonic=True)
        x = _constant_input(2, N, D, value=0.4)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL,
                        f"UnionLayer butterfly identity-on-const "
                        f"failed: err={err}")

    def test_forward_shape_n8(self):
        D, N = 3, 8
        layer = UnionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _random_input(2, N, D, seed=3)
        y = layer.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_reverse_roundtrip_shape(self):
        D, N = 3, 8
        layer = UnionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _random_input(2, N, D, seed=4)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        self.assertEqual(x_rec.shape, x.shape)
        self.assertFalse(torch.isnan(x_rec).any().item())
        self.assertFalse(torch.isinf(x_rec).any().item())

    def test_pair_op_semantics_monotonic(self):
        """Per-pair op should compute max element-wise on the two halves
        at identity W init."""
        D, N = 3, 2
        layer = UnionLayer(nInput=D, nOutput=D, butterfly=True, N=N,
                          monotonic=True)
        x = torch.tensor([[[0.3, 0.5, -0.1],
                           [0.1, 0.7,  0.2]]])  # [B=1, N=2, D=3]
        y = layer.forward(x)
        expected_max = torch.maximum(x[:, 0, :], x[:, 1, :])
        self.assertTrue(torch.allclose(y[:, 0, :], expected_max, atol=1e-5))
        self.assertTrue(torch.allclose(y[:, 1, :], expected_max, atol=1e-5))


# =====================================================================
# ConjunctionLayer butterfly (S-tier monotonic min)
# =====================================================================
class TestConjunctionLayerButterfly(unittest.TestCase):
    """S-tier monotonic min butterfly cascade."""

    def test_constructs_with_butterfly(self):
        D, N = 3, 8
        layer = ConjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), N)

    def test_butterfly_W_shape(self):
        D, N = 3, 4
        layer = ConjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        expected = (int(math.log2(N)), N // 2, 2 * D, 2 * D)
        self.assertEqual(tuple(layer.butterfly_W.shape), expected)

    def test_identity_on_constant_input(self):
        D, N = 3, 8
        layer = ConjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _constant_input(2, N, D, value=0.4)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL,
                        f"ConjunctionLayer butterfly identity-on-const "
                        f"failed: err={err}")

    def test_forward_shape_n4(self):
        D, N = 3, 4
        layer = ConjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _random_input(2, N, D, seed=5)
        y = layer.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_pair_op_semantics(self):
        """Per-pair op = element-wise min (monotonic) on the two halves."""
        D, N = 3, 2
        layer = ConjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = torch.tensor([[[0.3, 0.5, 0.1],
                           [0.1, 0.7, 0.2]]])  # post-codebook scalar (non-neg)
        y = layer.forward(x)
        expected_min = torch.minimum(x[:, 0, :], x[:, 1, :])
        self.assertTrue(torch.allclose(y[:, 0, :], expected_min, atol=1e-5))
        self.assertTrue(torch.allclose(y[:, 1, :], expected_min, atol=1e-5))


# =====================================================================
# DisjunctionLayer butterfly (S-tier monotonic max)
# =====================================================================
class TestDisjunctionLayerButterfly(unittest.TestCase):
    """S-tier monotonic max butterfly cascade."""

    def test_constructs_with_butterfly(self):
        D, N = 3, 8
        layer = DisjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), N)

    def test_butterfly_W_shape(self):
        D, N = 3, 4
        layer = DisjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        expected = (int(math.log2(N)), N // 2, 2 * D, 2 * D)
        self.assertEqual(tuple(layer.butterfly_W.shape), expected)

    def test_identity_on_constant_input(self):
        D, N = 3, 8
        layer = DisjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _constant_input(2, N, D, value=0.4)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL,
                        f"DisjunctionLayer butterfly identity-on-const "
                        f"failed: err={err}")

    def test_forward_shape_n4(self):
        D, N = 3, 4
        layer = DisjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = _random_input(2, N, D, seed=6)
        y = layer.forward(x)
        self.assertEqual(y.shape, x.shape)

    def test_pair_op_semantics(self):
        """Per-pair op = element-wise max on the two halves."""
        D, N = 3, 2
        layer = DisjunctionLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = torch.tensor([[[0.3, 0.5, 0.1],
                           [0.1, 0.7, 0.2]]])  # post-codebook scalar
        y = layer.forward(x)
        expected_max = torch.maximum(x[:, 0, :], x[:, 1, :])
        self.assertTrue(torch.allclose(y[:, 0, :], expected_max, atol=1e-5))
        self.assertTrue(torch.allclose(y[:, 1, :], expected_max, atol=1e-5))


# =====================================================================
# Cross-op: lossy marker
# =====================================================================
class TestLossyMarker(unittest.TestCase):
    """The four tetralemma ops are lossy by contract -- the class attr
    declares so; the butterfly mode preserves this."""

    def test_intersection_is_lossy(self):
        self.assertTrue(IntersectionLayer.lossy)

    def test_union_is_lossy(self):
        self.assertTrue(UnionLayer.lossy)

    def test_conjunction_is_lossy(self):
        self.assertTrue(ConjunctionLayer.lossy)

    def test_disjunction_is_lossy(self):
        self.assertTrue(DisjunctionLayer.lossy)


# =====================================================================
# Non-butterfly path still works
# =====================================================================
class TestNonButterflyUnchanged(unittest.TestCase):
    """Construct the four ops WITHOUT butterfly; existing (left, right)
    semantics still work."""

    def test_intersection_binary_unchanged(self):
        layer = IntersectionLayer()  # no butterfly
        self.assertFalse(layer.butterfly)
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.forward(left, right)
        self.assertTrue(torch.allclose(y, Ops.intersection(left, right),
                                       atol=1e-6))

    def test_union_binary_unchanged(self):
        layer = UnionLayer()
        self.assertFalse(layer.butterfly)
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.forward(left, right)
        self.assertTrue(torch.allclose(y, Ops.union(left, right), atol=1e-6))

    def test_conjunction_binary_unchanged(self):
        layer = ConjunctionLayer()
        self.assertFalse(layer.butterfly)
        left = torch.rand(2, 5, 4)
        right = torch.rand(2, 5, 4)
        y = layer.forward(left, right)
        self.assertTrue(torch.allclose(y, Ops.intersection(left, right,
                                                          monotonic=True),
                                       atol=1e-6))

    def test_disjunction_binary_unchanged(self):
        layer = DisjunctionLayer()
        self.assertFalse(layer.butterfly)
        left = torch.rand(2, 5, 4)
        right = torch.rand(2, 5, 4)
        y = layer.forward(left, right)
        self.assertTrue(torch.allclose(y, Ops.union(left, right,
                                                   monotonic=True),
                                       atol=1e-6))


if __name__ == "__main__":
    unittest.main()
