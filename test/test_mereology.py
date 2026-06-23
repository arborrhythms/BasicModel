"""Tests for the `Mereology` mixin and its measure family.

Covers the relocation from `BaseModel`:
  * `Contiguous` / `Continuous` / `Peaceful` (mixin parity)
  * `Area`       — sum of leaf hyperrectangle volumes
  * `Luminosity` — totalArea − pairwise(overlap × DoT_disagreement)
  * `Ops.hyperrectangle_volume` / `Ops.hyperrectangle_overlap_volume`

Also re-asserts retained Phase 1b utilities (`CopyLayer`,
`_gaussian_kernel_overlap`, `ste_answer`) that survived the revert.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

from Layers import (  # noqa: E402
    CopyLayer,
    GRAMMAR_LAYER_CLASSES,
    Ops,
    SwapLayer,
    _DEFAULT_SUBSYMBOLIC_SIGMA,
    _gaussian_kernel_overlap,
    ste_answer,
)
from Mereology import Mereology, RuleSpec, StepInfo, HoCShape  # noqa: E402


# ---------------------------------------------------------------------------
# Hyperrectangle Ops kernels
# ---------------------------------------------------------------------------

class TestHyperrectangleVolume(unittest.TestCase):
    def test_two_axes_all_active(self):
        # Two axes with sides (0.7+0.2)=0.9 and (0.5+0.3)=0.8 -> 0.72
        boxes = torch.tensor([[[0.7, 0.2], [0.5, 0.3]]])
        v = Ops.hyperrectangle_volume(boxes)
        self.assertAlmostEqual(float(v.item()), 0.72, places=5)

    def test_zero_box_returns_zero(self):
        v = Ops.hyperrectangle_volume(torch.zeros(1, 2, 2))
        self.assertEqual(float(v.item()), 0.0)

    def test_eps_drops_inactive_axis(self):
        # First axis side 0.9; second side 1e-10 (below eps=1e-6).
        boxes = torch.tensor([[[0.7, 0.2], [1e-10, 1e-10]]])
        v = Ops.hyperrectangle_volume(boxes)
        self.assertAlmostEqual(float(v.item()), 0.9, places=5)

    def test_single_axis(self):
        boxes = torch.tensor([[[0.4, 0.4]]])
        v = Ops.hyperrectangle_volume(boxes)
        self.assertAlmostEqual(float(v.item()), 0.8, places=5)

    def test_batched(self):
        # [B=2, n_axes=2, 2]; per-batch volumes 0.72 and 0.0.
        boxes = torch.tensor([
            [[0.7, 0.2], [0.5, 0.3]],
            [[0.0, 0.0], [0.0, 0.0]],
        ])
        v = Ops.hyperrectangle_volume(boxes)
        self.assertEqual(tuple(v.shape), (2,))
        self.assertAlmostEqual(float(v[0].item()), 0.72, places=5)
        self.assertEqual(float(v[1].item()), 0.0)


class TestHyperrectangleOverlapVolume(unittest.TestCase):
    def test_self_overlap_equals_volume(self):
        boxes = torch.tensor([[[0.7, 0.2], [0.5, 0.3]]])
        vol = Ops.hyperrectangle_volume(boxes)
        ovl = Ops.hyperrectangle_overlap_volume(boxes, boxes)
        self.assertAlmostEqual(float(vol.item()), float(ovl.item()),
                                places=5)

    def test_disjoint_along_one_axis(self):
        # axis interval [0, 0.7] vs [-0.7, -0.3] -> no overlap on this axis
        b1 = torch.tensor([[[0.7, 0.0]]])
        b2 = torch.tensor([[[-0.3, 0.7]]])
        ovl = Ops.hyperrectangle_overlap_volume(b1, b2)
        self.assertEqual(float(ovl.item()), 0.0)

    def test_partial_overlap(self):
        # axis interval [-0.5, 0.5] vs [0, 1] -> shared [0, 0.5] = 0.5
        b1 = torch.tensor([[[0.5, 0.5]]])
        b2 = torch.tensor([[[1.0, 0.0]]])
        ovl = Ops.hyperrectangle_overlap_volume(b1, b2)
        self.assertAlmostEqual(float(ovl.item()), 0.5, places=5)

    def test_zero_boxes_no_overlap(self):
        b = torch.zeros(1, 2, 2)
        ovl = Ops.hyperrectangle_overlap_volume(b, b)
        self.assertEqual(float(ovl.item()), 0.0)


# ---------------------------------------------------------------------------
# Mereology mixin shape and inheritance
# ---------------------------------------------------------------------------

class TestMereologyMixinShape(unittest.TestCase):
    def test_basemodel_inherits_mereology(self):
        from Models import BaseModel, BasicModel
        self.assertTrue(issubclass(BaseModel, Mereology))
        self.assertTrue(issubclass(BasicModel, Mereology))

    def test_measures_callable_on_class(self):
        for name in ('Contiguous', 'Continuous', 'Peaceful',
                      'Area', 'Luminosity'):
            self.assertTrue(callable(getattr(Mereology, name)),
                            f"{name} should be callable on Mereology")

    def test_dataclasses_relocated(self):
        # The shape descriptors moved to Mereology.py.
        self.assertEqual(RuleSpec.__module__, 'Mereology')
        self.assertEqual(StepInfo.__module__, 'Mereology')
        self.assertEqual(HoCShape.__module__, 'Mereology')


# ---------------------------------------------------------------------------
# Direct-volume measure on a stub model
# ---------------------------------------------------------------------------

class _StubSubspace:
    """Minimal subspace with a fixed materialize() result."""

    def __init__(self, event_tensor):
        self._event = event_tensor
        self.knowing = None

    def materialize(self):
        return self._event


class _StubSymbolSpace:
    """Minimal WholeSpace shim for Mereology tests."""

    def __init__(self, event_tensor, threshold=0.0):
        self.subspace = _StubSubspace(event_tensor)
        self._truth_min_magnitude = threshold


class _StubConceptualSpace:
    def __init__(self):
        self.subspace = _StubSubspace(None)


class _StubModel(Mereology):
    """Mereology-mixed stand-in for tests that exercise the measure
    family without the full BasicModel/BasicModel construction.
    """

    def __init__(self, event_tensor, n_stages=1, threshold=0.0):
        self.wholeSpace = _StubSymbolSpace(event_tensor, threshold)
        self.conceptualSpace = _StubConceptualSpace()
        self.symbolSpace = None
        self.subsymbolicOrder = n_stages


class TestArea(unittest.TestCase):
    def test_area_returns_zero_for_empty_event(self):
        m = _StubModel(torch.zeros(1, 1, 4))
        self.assertEqual(m.Area(), 0.0)

    def test_area_returns_zero_for_no_active_positions(self):
        # Bivector all zero -- norm < threshold -> empty hoc_shape leaves.
        m = _StubModel(torch.zeros(1, 2, 4), threshold=0.1)
        self.assertEqual(m.Area(), 0.0)

    def test_area_records_knowing_when_nonempty(self):
        # Active bivector, default-only path (no host layers wired):
        # _walk_reverse falls through to ([parent_tensor], []), which
        # produces a single trustworthy leaf.
        ev = torch.zeros(1, 2, 4)
        ev[..., 0] = 0.7
        ev[..., 1] = 0.2
        m = _StubModel(ev)
        a = m.Area()
        self.assertGreaterEqual(a, 0.0)
        self.assertLessEqual(a, 1.0)
        self.assertIsNotNone(m.conceptualSpace.subspace.knowing)


class TestLuminosity(unittest.TestCase):
    def test_luminosity_zero_for_empty(self):
        m = _StubModel(torch.zeros(1, 2, 4))
        self.assertEqual(m.Luminosity(), 0.0)

    def test_luminosity_within_range(self):
        ev = torch.zeros(1, 2, 4)
        ev[..., 0] = 0.7
        ev[..., 1] = 0.2
        m = _StubModel(ev)
        lum = m.Luminosity()
        self.assertGreaterEqual(lum, -1.0)
        self.assertLessEqual(lum, 1.0)


# ---------------------------------------------------------------------------
# Retained Phase 1b utilities
# ---------------------------------------------------------------------------

class TestCopyLayer(unittest.TestCase):
    def test_copy_layer_forward_returns_left(self):
        layer = CopyLayer()
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        self.assertTrue(torch.equal(layer.forward(a, b), a))

    def test_copy_layer_reverse_pseudo_inverse(self):
        layer = CopyLayer()
        parent = torch.tensor([[1.0, 2.0]])
        left, right = layer.reverse(parent)
        self.assertTrue(torch.equal(left, parent))
        self.assertTrue(torch.equal(right, parent))

    def test_copy_layer_grammar_registration(self):
        self.assertIn('copy', GRAMMAR_LAYER_CLASSES)
        self.assertIsInstance(GRAMMAR_LAYER_CLASSES['copy'](), CopyLayer)

    def test_swap_and_copy_dual(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        self.assertTrue(torch.equal(SwapLayer().forward(a, b), b))
        self.assertTrue(torch.equal(CopyLayer().forward(a, b), a))

    def test_introspection_layers_registered(self):
        # The 2026-05-12 conceptual-introspection refactor restored
        # ``area`` / ``luminosity`` / ``isaPart`` as first-class
        # GrammarLayers (see test_conceptual_introspection.py).  The
        # measures still live on the Mereology mixin; the layers wrap
        # them as chart-dispatch entry points.
        self.assertIn('area', GRAMMAR_LAYER_CLASSES)
        self.assertIn('luminosity', GRAMMAR_LAYER_CLASSES)
        self.assertIn('isaPart', GRAMMAR_LAYER_CLASSES)


class TestSTEAnswer(unittest.TestCase):
    def test_forward_returns_real_answer(self):
        q = torch.tensor([0.3])
        f = torch.tensor([1.0])
        self.assertAlmostEqual(float(ste_answer(q, f).item()), 1.0,
                                places=5)

    def test_backward_routes_through_q(self):
        q = torch.tensor([0.3], requires_grad=True)
        f = torch.tensor([1.0])
        ste_answer(q, f).sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertAlmostEqual(float(q.grad.item()), 1.0, places=5)


class TestGaussianKernelOverlap(unittest.TestCase):
    def test_self_overlap_diagonal_ones(self):
        X = torch.randn(4, 3)
        K = _gaussian_kernel_overlap(X, X, 0.1, 0.1)
        diag = torch.diagonal(K)
        self.assertTrue(torch.allclose(diag, torch.ones_like(diag)))

    def test_overlap_in_unit_interval(self):
        X = torch.randn(2, 3)
        Y = torch.randn(2, 3)
        K = _gaussian_kernel_overlap(X, Y, 0.1, 0.1)
        self.assertTrue(torch.all(K >= 0.0).item())
        self.assertTrue(torch.all(K <= 1.0 + 1e-5).item())

    def test_default_sigma_constant_present(self):
        self.assertGreater(_DEFAULT_SUBSYMBOLIC_SIGMA, 0.0)
        self.assertLessEqual(_DEFAULT_SUBSYMBOLIC_SIGMA, 1.0)


# ---------------------------------------------------------------------------
# Peaceful() -- valence symmetry x luminosity uniformity over the TruthLayer
# ---------------------------------------------------------------------------

class TestPeaceful(unittest.TestCase):
    @staticmethod
    def _model(rows, with_ws=True):
        """A Mereology instance (constructed via __new__ to skip __init__)
        exposing wholeSpace.truth_layer with `rows` recorded -- truths hand-set
        for deterministic control of the per-truth signed DoT."""
        import types
        from Layers import TruthLayer

        class _M(Mereology):
            pass
        m = _M.__new__(_M)
        if not with_ws:
            m.wholeSpace = None
            return m
        tl = TruthLayer(nDim=4, max_truths=16)
        for i, vec in enumerate(rows):
            tl.truths[i] = torch.tensor(vec, dtype=torch.float32)
        tl.count.fill_(len(rows))
        m.wholeSpace = types.SimpleNamespace(truth_layer=tl)
        return m

    def test_empty_truthset_is_unknown(self):
        self.assertEqual(self._model([]).Peaceful(), 0.0)

    def test_balanced_uniform_is_peaceful(self):
        # Balanced valence (+/-), uniform magnitude -> +1 (One Taste).
        m = self._model([[0.6] * 4, [-0.6] * 4, [0.6] * 4, [-0.6] * 4])
        self.assertAlmostEqual(m.Peaceful(), 1.0, places=5)

    def test_valence_bias_is_not_peaceful(self):
        # All-positive valence -> symmetry 0 -> -1 (preferential attention).
        self.assertAlmostEqual(self._model([[0.6] * 4] * 4).Peaceful(),
                               -1.0, places=5)

    def test_uneven_luminosity_lowers_peace(self):
        balanced = self._model([[0.6] * 4, [-0.6] * 4]).Peaceful()
        uneven = self._model([[0.9] * 4, [-0.1] * 4]).Peaceful()
        self.assertLess(uneven, balanced)

    def test_in_range(self):
        for rows in ([[0.6] * 4, [-0.6] * 4], [[0.6] * 4] * 3,
                     [[0.9] * 4, [-0.1] * 4, [0.2] * 4]):
            v = self._model(rows).Peaceful()
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_no_truth_layer_is_unknown(self):
        self.assertEqual(self._model([], with_ws=False).Peaceful(), 0.0)


if __name__ == '__main__':
    unittest.main()
