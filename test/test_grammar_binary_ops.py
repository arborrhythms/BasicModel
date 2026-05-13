"""Binary tensor ops on GrammarLayer subclasses.

Covers the ``compose`` / ``decompose`` contract added for the CKY
chart parser. Each subclass exposes:
  * arity == 1: ``compose(x) -> y`` and ``decompose(y) -> x``
  * arity == 2: ``compose(left, right) -> y`` and
                ``decompose(y) -> (left, right)``

These are independent of the legacy ``forward`` / ``reverse``
feature-fold path used by ConceptualSpace; the chart parser will
call ``compose`` / ``decompose`` directly.
"""

import os
import sys
import unittest
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

from Layers import (
    GrammarLayer,
    IntersectionLayer,
    NonLayer,
    NotLayer,
    PiLayer,
    SigmaLayer,
    UnionLayer,
)


class TestUnaryGrammarLayers(unittest.TestCase):
    """arity-1 grammar layers route compose -> forward, decompose -> reverse."""

    def test_not_compose_decompose_roundtrip(self):
        layer = NotLayer()
        # NotLayer operates on the materialized muxed event tensor
        # [B, V, nWhat + nWhere + nWhen]; the .what bivector [pos, neg]
        # is at [..., :2] and nWhere / nWhen channels follow. Negation
        # swaps the bivector pair and passes through the rest.
        x = torch.randn(4, 3, 6)
        y = layer.compose(x)
        self.assertEqual(y.shape, x.shape)
        # Bivector swapped, rest preserved.
        self.assertTrue(torch.allclose(y[..., :2], x[..., :2].flip(dims=(-1,)), atol=1e-5))
        self.assertTrue(torch.allclose(y[..., 2:], x[..., 2:], atol=1e-5))
        # Self-inverse: pos/neg swap applied twice is identity.
        self.assertTrue(torch.allclose(layer.decompose(y), x, atol=1e-5))

    def test_non_compose_decompose_self_inverse(self):
        """NonLayer: per-pole bivector complement [1-pos, 1-neg].
        Self-inverse on each pole, where/when channels pass through."""
        layer = NonLayer()
        x = torch.rand(4, 3, 6)        # bivector poles in [0, 1] domain
        y = layer.compose(x)
        self.assertEqual(y.shape, x.shape)
        # Bivector poles complemented at [..., :2]; rest unchanged.
        self.assertTrue(torch.allclose(y[..., :2], 1.0 - x[..., :2], atol=1e-5))
        self.assertTrue(torch.allclose(y[..., 2:], x[..., 2:], atol=1e-5))
        # Self-inverse: non(non(x)) == x.
        self.assertTrue(torch.allclose(layer.decompose(y), x, atol=1e-5))

    # FusionLayer / ContiguousLayer were retired 2026-05-04. The
    # operator was a duplicate of DisjunctionLayer at S-tier --
    # migrate to ``disjunction(S, S)`` (post-codebook scalar max).


class TestPiLayerBinary(unittest.TestCase):
    """PiLayer.compose == AND fold of two operands; decompose
    is a balanced split that round-trips when invertible."""

    def setUp(self):
        pass

    def test_compose_shape_and_range(self):
        layer = PiLayer(nInput=4, nOutput=4, nonlinear=True)
        layer.set_sigma(0.999)
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        self.assertEqual(y.shape, (2, 5, 4))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue((y >= -1).all() and (y <= 1).all())

    def test_compose_decompose_roundtrip_invertible(self):
        layer = PiLayer(nInput=4, nOutput=4, invertible=True, nonlinear=True)
        layer.set_sigma(0.0)
        left = torch.rand(3, 7, 4) * 1.6 - 0.8
        right = torch.rand(3, 7, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        # Balanced inverse: compose(decompose(y)) == y.
        l_rec, r_rec = layer.decompose(y)
        y_rec = layer.compose(l_rec, r_rec)
        err = torch.norm(y - y_rec) / torch.norm(y).clamp_min(1e-8)
        self.assertLess(err.item(), 1e-4,
                        f"PiLayer compose/decompose roundtrip err {err:.2e}")

    def test_decompose_yields_equal_halves(self):
        # Default decompose splits the parent into two equal halves
        # (in log-mult domain). Useful for the chart's outside pass.
        layer = PiLayer(nInput=4, nOutput=4, invertible=True, nonlinear=True)
        layer.set_sigma(0.0)
        parent = torch.rand(2, 3, 4) * 1.6 - 0.8
        left, right = layer.decompose(parent)
        self.assertTrue(torch.allclose(left, right, atol=1e-6))

class TestSigmaLayerBinary(unittest.TestCase):
    """SigmaLayer.compose == OR fold of two operands; decompose
    is a balanced split that round-trips when invertible."""

    def setUp(self):
        pass

    def test_compose_shape_and_range(self):
        layer = SigmaLayer(nInput=4, nOutput=4, nonlinear=True)
        layer.set_sigma(0.999)
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        self.assertEqual(y.shape, (2, 5, 4))
        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue((y >= -1).all() and (y <= 1).all())

    def test_compose_decompose_roundtrip_invertible(self):
        layer = SigmaLayer(nInput=4, nOutput=4, naive=False,
                           invertible=True, nonlinear=True)
        layer.set_sigma(0.0)
        left = torch.rand(3, 7, 4) * 1.6 - 0.8
        right = torch.rand(3, 7, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        l_rec, r_rec = layer.decompose(y)
        y_rec = layer.compose(l_rec, r_rec)
        err = torch.norm(y - y_rec) / torch.norm(y).clamp_min(1e-8)
        self.assertLess(err.item(), 1e-4,
                        f"SigmaLayer compose/decompose roundtrip err {err:.2e}")

    def test_decompose_yields_equal_halves(self):
        layer = SigmaLayer(nInput=4, nOutput=4, naive=False,
                           invertible=True, nonlinear=True)
        layer.set_sigma(0.0)
        parent = torch.rand(2, 3, 4) * 1.6 - 0.8
        left, right = layer.decompose(parent)
        self.assertTrue(torch.allclose(left, right, atol=1e-6))


class TestIntersectionUnionBinary(unittest.TestCase):
    """IntersectionLayer / UnionLayer are L-tier (logical) binary
    lattice min/max on bivector activation. They share kernels
    with ConjunctionLayer / DisjunctionLayer (S-tier counterparts
    on post-codebook scalar activation); the L-vs-S distinction is
    operand domain (bivector vs scalar), not which space holds them.
    """

    def setUp(self):
        pass

    def test_intersection_layer_is_L_tier(self):
        layer = IntersectionLayer()
        self.assertEqual(layer.tier, 'L')

    def test_union_layer_is_L_tier(self):
        layer = UnionLayer()
        self.assertEqual(layer.tier, 'L')

    def test_intersection_compose_is_min_kernel(self):
        from Layers import Ops
        layer = IntersectionLayer()
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        # L-tier intersection is RadMin (same-sign min magnitude,
        # zero passthrough) via Ops.intersection.
        self.assertTrue(torch.allclose(y, Ops.intersection(left, right), atol=1e-6))
        # Idempotent on the diagonal.
        diag = layer.compose(left, left)
        self.assertTrue(torch.allclose(diag, left, atol=1e-6))

    def test_intersection_decompose_pseudo_inverse(self):
        layer = IntersectionLayer()
        parent = torch.rand(2, 5, 4) * 1.6 - 0.8
        l_rec, r_rec = layer.decompose(parent)
        # Lossy fold: pseudo-inverse returns parent for both children.
        self.assertTrue(torch.equal(l_rec, parent))
        self.assertTrue(torch.equal(r_rec, parent))
        # Recomposing the pseudo-inverse with itself is idempotent.
        self.assertTrue(torch.allclose(
            layer.compose(l_rec, r_rec), parent, atol=1e-6))

    def test_union_compose_is_max_kernel(self):
        from Layers import Ops
        layer = UnionLayer()
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        # L-tier union is RadMax (same-sign max magnitude, zero
        # passthrough) via Ops.union.
        self.assertTrue(torch.allclose(y, Ops.union(left, right), atol=1e-6))
        diag = layer.compose(left, left)
        self.assertTrue(torch.allclose(diag, left, atol=1e-6))

    def test_union_decompose_pseudo_inverse(self):
        layer = UnionLayer()
        parent = torch.rand(2, 5, 4) * 1.6 - 0.8
        l_rec, r_rec = layer.decompose(parent)
        self.assertTrue(torch.equal(l_rec, parent))
        self.assertTrue(torch.equal(r_rec, parent))


class TestComposeGradients(unittest.TestCase):
    """Backprop reaches both child operands through compose."""

    def test_intersection_grad_flows_to_both_children(self):
        layer = IntersectionLayer()
        left = (torch.rand(2, 3, 4) * 1.6 - 0.8).requires_grad_(True)
        right = (torch.rand(2, 3, 4) * 1.6 - 0.8).requires_grad_(True)
        y = layer.compose(left, right)
        y.sum().backward()
        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)
        # Lattice min routes gradient to whichever operand was the
        # smaller magnitude per cell; both should receive nonzero
        # gradient on a random batch.
        self.assertGreater(left.grad.abs().sum().item(), 0.0)
        self.assertGreater(right.grad.abs().sum().item(), 0.0)

    def test_union_grad_flows_to_both_children(self):
        layer = UnionLayer()
        left = (torch.rand(2, 3, 4) * 1.6 - 0.8).requires_grad_(True)
        right = (torch.rand(2, 3, 4) * 1.6 - 0.8).requires_grad_(True)
        y = layer.compose(left, right)
        y.sum().backward()
        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)
        self.assertGreater(left.grad.abs().sum().item(), 0.0)
        self.assertGreater(right.grad.abs().sum().item(), 0.0)


class TestArityContract(unittest.TestCase):
    """Default GrammarLayer.compose enforces the arity contract."""

    def test_unary_accepts_one_operand(self):
        layer = NotLayer()
        with self.assertRaises(ValueError):
            layer.compose(torch.zeros(1, 1, 6), torch.zeros(1, 1, 6))

    def test_arity_two_default_raises(self):
        # A bare GrammarLayer subclass with arity=2 but no compose
        # override should raise NotImplementedError (the chart parser
        # surfaces this as "rule has no binary parameterization").
        class _Stub(GrammarLayer):
            arity = 2
        stub = _Stub()
        with self.assertRaises(NotImplementedError):
            stub.compose(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))


if __name__ == "__main__":
    unittest.main()
