"""Conceptual introspection grammar layers.

Covers ``area(S)``, ``luminosity(S, S)``, ``directPartOf(S, S)`` and
the underlying static op helpers added in 2026-05-12. The plan:
``doc/plans/2026-05-04-conceptual-introspection-handoff.md``.
"""

import os
import sys
import unittest

import torch

_project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project, "bin"))

from Layers import (
    AreaLayer,
    DirectPartOfLayer,
    GRAMMAR_LAYER_CLASSES,
    LuminosityLayer,
    area_op,
    direct_part_of_op,
    luminosity_op,
    ste_answer,
)


class TestIntrospectiveOps(unittest.TestCase):
    """Static op helpers return scalar tensors in the documented range."""

    def test_area_in_unit_interval(self):
        x = torch.randn(2, 3, 4)
        a = area_op(x)
        self.assertIsInstance(a, torch.Tensor)
        self.assertGreaterEqual(float(a.item()), 0.0)
        self.assertLessEqual(float(a.item()), 1.0)

    def test_luminosity_in_signed_unit_interval(self):
        for _ in range(8):
            a = torch.randn(2, 3, 4)
            b = torch.randn(2, 3, 4)
            lum = luminosity_op(a, b)
            self.assertGreaterEqual(float(lum.item()), -1.0 - 1e-6)
            self.assertLessEqual(float(lum.item()), 1.0 + 1e-6)

    def test_direct_part_of_in_unit_interval(self):
        for _ in range(8):
            c = torch.randn(2, 3, 4)
            p = torch.randn(2, 3, 4)
            dpo = direct_part_of_op(c, p)
            self.assertGreater(float(dpo.item()), 0.0)
            self.assertLessEqual(float(dpo.item()), 1.0 + 1e-6)

    def test_direct_part_of_higher_for_closer_points(self):
        """Identical activations score strictly higher than a noisy
        far-away pair under the kernel overlap."""
        v = torch.randn(2, 3, 4)
        far = v + 10.0 * torch.randn(2, 3, 4)
        dpo_same = direct_part_of_op(v, v).item()
        dpo_far = direct_part_of_op(v, far).item()
        self.assertGreater(dpo_same, dpo_far)

    def test_luminosity_consistent_field_near_area(self):
        """When two propositions agree (same DoT, same region), the
        penalty term collapses to zero and luminosity ~= area."""
        x = torch.full((2, 3, 4), 0.5)
        lum = luminosity_op(x, x).item()
        area = area_op(x).item()
        # Disagreement = 0 -> penalty = 0 -> lum == area.
        self.assertAlmostEqual(lum, area, places=4)


class TestIntrospectiveGradient(unittest.TestCase):
    """Backward flows through the introspective ops to their inputs."""

    def test_luminosity_gradient(self):
        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 3, 4, requires_grad=True)
        lum = luminosity_op(x, y)
        lum.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_direct_part_of_gradient(self):
        c = torch.randn(2, 3, 4, requires_grad=True)
        p = torch.randn(2, 3, 4, requires_grad=True)
        dpo = direct_part_of_op(c, p)
        dpo.backward()
        self.assertIsNotNone(c.grad)
        self.assertTrue(torch.isfinite(c.grad).all())


class TestSTEWrapper(unittest.TestCase):
    """ste_answer: forward = answer, backward = grad through query."""

    def test_forward_returns_answer(self):
        q = torch.randn(4, requires_grad=True)
        f = torch.randn(4)
        out = ste_answer(q, f)
        self.assertTrue(torch.allclose(out, f, atol=1e-6))

    def test_backward_routes_to_query(self):
        q = torch.randn(4, requires_grad=True)
        f = torch.randn(4)
        out = ste_answer(q, f)
        out.sum().backward()
        self.assertIsNotNone(q.grad)
        # All-ones gradient flows back through q.
        self.assertTrue(torch.allclose(
            q.grad, torch.ones_like(q), atol=1e-6))


class TestIntrospectionLayers(unittest.TestCase):
    """Layer-wrapper classes plug into the chart via GRAMMAR_LAYER_CLASSES."""

    def test_layer_classes_registered(self):
        for name, cls in [('area', AreaLayer),
                          ('luminosity', LuminosityLayer),
                          ('directPartOf', DirectPartOfLayer)]:
            self.assertIn(name, GRAMMAR_LAYER_CLASSES,
                          f"{name!r} missing from GRAMMAR_LAYER_CLASSES")
            self.assertIs(GRAMMAR_LAYER_CLASSES[name], cls)

    def test_arity_metadata(self):
        self.assertEqual(AreaLayer.arity, 1)
        self.assertEqual(LuminosityLayer.arity, 2)
        self.assertEqual(DirectPartOfLayer.arity, 2)

    def test_layers_lossy(self):
        # Introspective ops produce scalars from vectors; reverse is
        # by definition lossy.
        for cls in (AreaLayer, LuminosityLayer, DirectPartOfLayer):
            self.assertTrue(cls.lossy, f"{cls.__name__} should be lossy")
            self.assertFalse(cls.invertible,
                             f"{cls.__name__} should be non-invertible")

    def test_forward_shapes(self):
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.assertEqual(AreaLayer().forward(x).dim(), 0)
        self.assertEqual(LuminosityLayer().forward(x, y).dim(), 0)
        self.assertEqual(DirectPartOfLayer().forward(x, y).dim(), 0)


if __name__ == "__main__":
    unittest.main()
