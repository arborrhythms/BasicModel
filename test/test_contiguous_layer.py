"""Tests for ContiguousLayer (Dakpo Tashi Namgyel one-pointedness fusion).

ContiguousLayer takes a convex hull (Option B: elementwise per-axis amax)
of its concept-axis slots, then broadcasts the hull back across that axis
so shape is preserved. Applied BEFORE NegationLayer in the symbolic
pipeline so hull-then-negate semantics hold.
"""

import os
import sys
import unittest

import torch

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Layers


class TestContiguousLayer(unittest.TestCase):
    def test_single_active_row_is_identity(self):
        """N=1: hull of one row is itself; layer is identity."""
        layer = Layers.ContiguousLayer(4, 4)
        x = torch.tensor([[[1.0, -0.5, 0.25, 0.0]]])  # [B=1, N=1, D=4]
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x))

    def test_multi_row_returns_elementwise_envelope(self):
        """N>1: per-axis amax over the concept axis, broadcast back."""
        layer = Layers.ContiguousLayer(3, 3)
        x = torch.tensor([
            [[0.5, 0.0, 0.2],
             [0.1, 0.9, 0.4]],
        ])  # [B=1, N=2, D=3]
        y = layer(x)
        envelope = torch.tensor([[0.5, 0.9, 0.4]])
        # Output shape preserved; every row carries the hull.
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y[0, 0], envelope[0]))
        self.assertTrue(torch.allclose(y[0, 1], envelope[0]))

    def test_bivector_envelope_preserves_pos_and_neg_poles(self):
        """On a [pos, neg, ...] bivector layout, amax keeps both poles
        from any active row: a positive evidence row contributes its pos
        pole; a negative evidence row contributes its neg pole."""
        layer = Layers.ContiguousLayer(4, 4)
        x = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0],   # pos pole on dim 0
             [0.0, 1.0, 0.0, 0.0]],  # neg pole on dim 1
        ])
        y = layer(x)
        expected_envelope = torch.tensor([1.0, 1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(y[0, 0], expected_envelope))
        self.assertTrue(torch.allclose(y[0, 1], expected_envelope))

    def test_hull_then_negate_well_defined(self):
        """ContiguousLayer applied before NegationLayer produces a stable
        [hull(x), -hull(x)] bivalent expansion -- the order matters."""
        nDim = 3
        contiguous = Layers.ContiguousLayer(nDim, nDim)
        negation = Layers.NegationLayer(nDim)
        x = torch.tensor([
            [[0.4, 0.0, 0.1],
             [0.0, 0.7, 0.2]],
        ])
        hulled = contiguous(x)
        # NegationLayer expects [..., D] last-dim -- flatten via the
        # canonical hull row which is broadcast across the N axis.
        out = negation(hulled[:, 0])
        # Concatenated [pos, neg]; first half = hull, second half = -hull.
        self.assertEqual(out.shape, (1, 2 * nDim))
        envelope = torch.tensor([[0.4, 0.7, 0.2]])
        self.assertTrue(torch.allclose(out[..., :nDim], envelope))
        self.assertTrue(torch.allclose(out[..., nDim:], -envelope))

    def test_reverse_round_trips_for_trivial_case(self):
        """When N=1 the hull is identity; reverse(forward(x)) == x."""
        layer = Layers.ContiguousLayer(3, 3)
        x = torch.tensor([[[0.2, -0.3, 0.7]]])
        y = layer(x)
        recovered = layer.reverse(y)
        self.assertTrue(torch.allclose(recovered, x))

    def test_gradient_flows_through_layer(self):
        """Loss on the hull output produces a gradient on the input."""
        layer = Layers.ContiguousLayer(2, 2)
        x = torch.tensor([
            [[0.3, 0.8],
             [0.5, 0.2]],
        ], requires_grad=True)
        y = layer(x)
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        # Each output cell carries the hull (max over N), so the gradient
        # accumulates onto the argmax row per axis.
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_rule_name_and_arity_metadata(self):
        """Class attributes match the grammar rule operator surface."""
        self.assertEqual(Layers.ContiguousLayer.rule_name, "Contiguous")
        self.assertEqual(Layers.ContiguousLayer.arity, 1)
        self.assertFalse(Layers.ContiguousLayer.invertible)
        self.assertTrue(Layers.ContiguousLayer.lossy)


if __name__ == "__main__":
    unittest.main()
