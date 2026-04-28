"""Tests for the negation layer used by DNF-style Sigma/Pi logic."""

import os
import sys
import unittest

import torch

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Layers
from util import TheDevice


class TestNegationLayer(unittest.TestCase):
    def test_default_materializes_signed_complement(self):
        layer = Layers.NegationLayer(4)
        x = torch.tensor([[-1.0, -0.25, 0.0, 0.75]], device=TheDevice.get())

        y = layer(x)
        expected = torch.cat((x, -x), dim=-1)

        self.assertEqual(y.shape, (1, 8))
        self.assertTrue(torch.allclose(y, expected))
        self.assertTrue(torch.allclose(layer.reverse(y), x))

    def test_ternary_mode_inserts_non_affirming_residual(self):
        layer = Layers.NegationLayer(3, ternary=True)
        x = torch.tensor([[-1.0, 0.0, 1.0]], device=TheDevice.get())

        y = layer(x)
        non = Layers.Ops.non(x, monotonic=False)

        self.assertTrue(torch.allclose(y[..., :3], x))
        self.assertTrue(torch.allclose(y[..., 3:6], non))
        self.assertTrue(torch.allclose(y[..., 6:], -x))
        self.assertTrue(torch.allclose(layer.reverse(y), x))

        positive_channels = torch.stack(
            (torch.relu(y[..., :3]), y[..., 3:6], torch.relu(y[..., 6:])),
            dim=-1,
        )
        self.assertTrue(torch.equal(positive_channels.sum(dim=-1), torch.ones_like(x)))
        self.assertTrue(torch.equal((positive_channels > 0).sum(dim=-1),
                                    torch.ones_like(x, dtype=torch.int64)))

    def test_bivalent_dnf_xor_truth_table(self):
        """A literal bank plus AND-terms plus OR-term expresses canonical DNF.

        XOR is represented as (A and not B) or (not A and B).  This test
        uses positive evidence from the signed literal bank and strict
        min/max gates to pin the Boolean normal form apart from the learned
        Pi/Sigma relaxation.
        """
        layer = Layers.NegationLayer(2)
        x = torch.tensor(
            [[-1.0, -1.0],
             [-1.0,  1.0],
             [ 1.0, -1.0],
             [ 1.0,  1.0]],
            device=TheDevice.get(),
        )

        literals = layer(x)
        evidence = torch.relu(literals)
        a = evidence[..., 0]
        b = evidence[..., 1]
        not_a = evidence[..., 2]
        not_b = evidence[..., 3]
        term_a_not_b = torch.minimum(a, not_b)
        term_not_a_b = torch.minimum(not_a, b)
        y = torch.maximum(term_a_not_b, term_not_a_b).unsqueeze(-1)

        expected = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=TheDevice.get())
        self.assertTrue(torch.equal(y, expected))

    def test_negation_pi_sigma_stack_is_dnf_ordered(self):
        """The no-grammar stack has the DNF order: literals -> Pi -> Sigma."""
        torch.manual_seed(0)
        neg = Layers.NegationLayer(2)
        pi = Layers.PiLayer(4, 3, invertible=True, monotonic=True, nonlinear=True)
        sigma = Layers.SigmaLayer(3, 1, invertible=True, monotonic=True, nonlinear=True)
        pi.set_sigma(0)
        sigma.set_sigma(0)

        x = torch.tensor(
            [[[-0.75, 0.50],
              [ 0.25, -0.90]]],
            device=TheDevice.get(),
        )

        literals = neg(x)
        terms = pi(literals)
        out = sigma(terms)

        self.assertEqual(literals.shape, (1, 2, 4))
        self.assertEqual(terms.shape, (1, 2, 3))
        self.assertEqual(out.shape, (1, 2, 1))
        self.assertTrue(torch.isfinite(terms).all())
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(pi.monotonic)
        self.assertTrue(sigma.monotonic)


if __name__ == "__main__":
    unittest.main()
