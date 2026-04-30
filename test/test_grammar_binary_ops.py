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
    ContiguousLayer,
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
        x = torch.randn(4, 3, 6)
        y = layer.compose(x)
        self.assertEqual(y.shape, x.shape)
        # NotLayer is self-inverse: bivector swap of [pos, neg] dims.
        self.assertTrue(torch.allclose(layer.decompose(y), x, atol=1e-5))

    def test_non_compose_runs_decompose_lossy_identity(self):
        layer = NonLayer()
        x = torch.rand(4, 3, 6) * 2 - 1
        y = layer.compose(x)
        self.assertEqual(y.shape, x.shape)
        # Lossy: decompose returns the parent unchanged (best identity
        # recovery without auxiliary structure).
        recovered = layer.decompose(y)
        self.assertTrue(torch.equal(recovered, y))

    def test_contiguous_compose_decompose_lossy(self):
        layer = ContiguousLayer(nInput=6)
        x = torch.rand(4, 3, 6) * 2 - 1
        y = layer.compose(x)
        self.assertEqual(y.shape, x.shape)
        recovered = layer.decompose(y)
        self.assertTrue(torch.equal(recovered, y))


class TestPiLayerBinary(unittest.TestCase):
    """PiLayer.compose == AND fold of two operands; decompose
    is a balanced split that round-trips when invertible."""

    def setUp(self):
        torch.manual_seed(0)

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

    def test_butterfly_mode_raises(self):
        layer = PiLayer(nInput=4, nOutput=4, invertible=True,
                        nonlinear=True, stage_idx=0, n_t=4, is_last=False)
        x = torch.rand(2, 3, 4) * 1.6 - 0.8
        with self.assertRaises(NotImplementedError):
            layer.compose(x, x)
        with self.assertRaises(NotImplementedError):
            layer.decompose(x)


class TestSigmaLayerBinary(unittest.TestCase):
    """SigmaLayer.compose == OR fold of two operands; decompose
    is a balanced split that round-trips when invertible."""

    def setUp(self):
        torch.manual_seed(0)

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
    """The grammar-facing wrappers delegate compose/decompose to their
    inner Pi/Sigma layers and preserve shape and invertibility."""

    def setUp(self):
        torch.manual_seed(0)

    def test_intersection_compose_delegates_and_roundtrips(self):
        pi = PiLayer(nInput=4, nOutput=4, invertible=True, nonlinear=True)
        pi.set_sigma(0.0)
        layer = IntersectionLayer(pi)
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        # Same as direct PiLayer.compose.
        y_direct = pi.compose(left, right)
        self.assertTrue(torch.allclose(y, y_direct, atol=1e-6))

        l_rec, r_rec = layer.decompose(y)
        y_rec = layer.compose(l_rec, r_rec)
        err = torch.norm(y - y_rec) / torch.norm(y).clamp_min(1e-8)
        self.assertLess(err.item(), 1e-4)

    def test_union_compose_delegates_and_roundtrips(self):
        sig = SigmaLayer(nInput=4, nOutput=4, naive=False,
                         invertible=True, nonlinear=True)
        sig.set_sigma(0.0)
        layer = UnionLayer(sig)
        left = torch.rand(2, 5, 4) * 1.6 - 0.8
        right = torch.rand(2, 5, 4) * 1.6 - 0.8
        y = layer.compose(left, right)
        y_direct = sig.compose(left, right)
        self.assertTrue(torch.allclose(y, y_direct, atol=1e-6))

        l_rec, r_rec = layer.decompose(y)
        y_rec = layer.compose(l_rec, r_rec)
        err = torch.norm(y - y_rec) / torch.norm(y).clamp_min(1e-8)
        self.assertLess(err.item(), 1e-4)


class TestComposeGradients(unittest.TestCase):
    """Backprop reaches both child operands through compose."""

    def test_intersection_grad_flows_to_both_children(self):
        pi = PiLayer(nInput=4, nOutput=4, invertible=True, nonlinear=True)
        pi.set_sigma(0.0)
        layer = IntersectionLayer(pi)
        left = (torch.rand(2, 3, 4) * 1.6 - 0.8).requires_grad_(True)
        right = (torch.rand(2, 3, 4) * 1.6 - 0.8).requires_grad_(True)
        y = layer.compose(left, right)
        y.sum().backward()
        self.assertIsNotNone(left.grad)
        self.assertIsNotNone(right.grad)
        self.assertGreater(left.grad.abs().sum().item(), 0.0)
        self.assertGreater(right.grad.abs().sum().item(), 0.0)

    def test_union_grad_flows_to_both_children(self):
        sig = SigmaLayer(nInput=4, nOutput=4, naive=False,
                         invertible=True, nonlinear=True)
        sig.set_sigma(0.0)
        layer = UnionLayer(sig)
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
