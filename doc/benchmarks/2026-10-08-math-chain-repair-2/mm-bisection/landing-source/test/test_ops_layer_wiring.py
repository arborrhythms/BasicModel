"""Tests for PiLayer.forward / SigmaLayer.forward binary=True (Step 5 of
the lift / lower / bivector refactor).

binary=True selects the top-2 input operands (by |x|) via the shared
Ops.top2_select_ste helper, sends the rest to 0 (the operation's
neutral element), and uses straight-through gradient so every input
keeps learning signal.

Linguistically:
    PiLayer.forward(x, binary=True)    -- AND-intersect the top-2 active
                                          concept operands; the long
                                          tail drops to mult-identity (1)
                                          and is irrelevant to the AND.
    SigmaLayer.forward(x, binary=True) -- OR-union the top-2 active
                                          candidates; the long tail drops
                                          to additive zero and is silent
                                          in the OR.

Verifies:
    - default (binary=False) is bit-equivalent to the pre-Step-5 forward.
    - binary=True with nInput <= 2 is a no-op (helper short-circuits).
    - binary=True picks the top-2-by-|x| entries and zeros the rest in
      the effective input that flows into the layer body.
    - binary=True uses STE: gradient w.r.t. unselected entries is
      non-zero (the long tail still learns).
    - Ops.top2_select_ste is itself correct in isolation.

See:
    basicmodel/doc/plans/2026-04-25-step5-ops-wiring-handoff.md
    basicmodel/doc/plans/2026-04-24-lift-lower-bivector-refactor.md (Step 5)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch

from Layers import Ops, PiLayer, SigmaLayer


def _set_seed(seed=0):
    """No-op: seed-defeated determinism was removed from these tests on
    2026-05-12 so the assertions exercise learning robustness, not
    init-pinned reproducibility."""
    return None


class TestTop2SelectSTE(unittest.TestCase):
    """Ops.top2_select_ste behaves as a hard top-2-by-|x| mask with
    straight-through gradient."""

    def test_keeps_top_two_zeros_rest(self):
        x = torch.tensor([0.1, -0.9, 0.4, 0.0, 0.7])
        out = Ops.top2_select_ste(x)
        # Top-2 by |x|: -0.9 (idx 1) and 0.7 (idx 4).
        expected = torch.tensor([0.0, -0.9, 0.0, 0.0, 0.7])
        self.assertTrue(torch.allclose(out, expected))

    def test_no_op_when_n_leq_2(self):
        x = torch.tensor([0.3, -0.5])
        out = Ops.top2_select_ste(x)
        self.assertTrue(torch.equal(out, x))

        x1 = torch.tensor([0.42])
        out1 = Ops.top2_select_ste(x1)
        self.assertTrue(torch.equal(out1, x1))

    def test_batched_last_axis(self):
        x = torch.tensor([
            [0.1, 0.9, -0.5, 0.0],
            [-0.8, 0.2, 0.7, -0.3],
        ])
        out = Ops.top2_select_ste(x)
        expected = torch.tensor([
            [0.0, 0.9, -0.5, 0.0],
            [-0.8, 0.0, 0.7, 0.0],
        ])
        self.assertTrue(torch.allclose(out, expected))

    def test_ste_gradient_flows_to_all_inputs(self):
        """Backward sees identity through the mask, so every input has
        a non-zero gradient even when it was zeroed in the forward."""
        x = torch.tensor([0.1, -0.9, 0.4, 0.0, 0.7], requires_grad=True)
        out = Ops.top2_select_ste(x)
        # Sum so dL/dout = 1 everywhere; STE -> dL/dx = 1 everywhere.
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))


class TestPiLayerBinaryForward(unittest.TestCase):
    """PiLayer.forward(x, binary=True): top-2 selection then the
    log-domain AND fold."""

    def test_default_unchanged(self):
        """binary=False default is bit-equivalent to the pre-Step-5 body."""
        _set_seed(7)
        layer = PiLayer(nInput=5, nOutput=4, invertible=True,
                        monotonic=True, nonlinear=True)
        x = torch.randn(2, 3, 5).clamp(-0.9, 0.9)
        with torch.no_grad():
            y_default = layer.forward(x)
            y_explicit = layer.forward(x, binary=False)
        self.assertTrue(torch.equal(y_default, y_explicit))

    def test_binary_matches_top2_masked_input(self):
        """forward(x, binary=True) == forward(top2_select_ste(x))."""
        _set_seed(11)
        layer = PiLayer(nInput=6, nOutput=4, invertible=True,
                        monotonic=True, nonlinear=True)
        x = torch.randn(1, 2, 6).clamp(-0.9, 0.9)
        with torch.no_grad():
            y_binary = layer.forward(x, binary=True)
            x_masked = Ops.top2_select_ste(x)
            y_via_helper = layer.forward(x_masked)
        self.assertTrue(torch.allclose(y_binary, y_via_helper))

    def test_binary_drops_long_tail_to_identity(self):
        """A near-zero tail ([eps, eps, ..., big1, big2]) should produce
        almost the same output as just [big1, big2] passed through, since
        x=0 is mult-identity in PiLayer's domain."""
        _set_seed(13)
        layer = PiLayer(nInput=5, nOutput=3, invertible=True,
                        monotonic=True, nonlinear=True)
        # Two prominent operands at idx 1 and 3.
        x = torch.tensor([[[1e-4, 0.7, -2e-4, -0.6, 1e-5]]])
        with torch.no_grad():
            y_binary = layer.forward(x, binary=True)
            # Compare to the same input with the small tail explicitly zeroed.
            x_clean = torch.tensor([[[0.0, 0.7, 0.0, -0.6, 0.0]]])
            y_clean = layer.forward(x_clean)
        self.assertTrue(torch.allclose(y_binary, y_clean, atol=1e-6))

    def test_binary_ste_gradient_to_unselected(self):
        """Gradient flows to every input dim, including those zeroed
        out in the forward by top-2 selection."""
        _set_seed(17)
        layer = PiLayer(nInput=5, nOutput=4, invertible=True,
                        monotonic=True, nonlinear=True)
        x = torch.tensor([[[0.1, -0.9, 0.05, 0.7, 0.02]]],
                         requires_grad=True)
        y = layer.forward(x, binary=True)
        y.sum().backward()
        # Every input dim has a non-zero gradient (STE identity through mask).
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue((x.grad.abs() > 0).all())


class TestSigmaLayerBinaryForward(unittest.TestCase):
    """SigmaLayer.forward(x, binary=True): top-2 selection then the
    additive OR fold."""

    def test_default_unchanged(self):
        """binary=False default is bit-equivalent to the pre-Step-5 body."""
        _set_seed(23)
        layer = SigmaLayer(nInput=5, nOutput=4, invertible=True,
                           monotonic=True, nonlinear=True)
        x = torch.randn(2, 3, 5).clamp(-0.9, 0.9)
        with torch.no_grad():
            y_default = layer.forward(x)
            y_explicit = layer.forward(x, binary=False)
        self.assertTrue(torch.equal(y_default, y_explicit))

    def test_binary_matches_top2_masked_input(self):
        """forward(x, binary=True) == forward(top2_select_ste(x))."""
        _set_seed(29)
        layer = SigmaLayer(nInput=6, nOutput=4, invertible=True,
                           monotonic=True, nonlinear=True)
        x = torch.randn(1, 2, 6).clamp(-0.9, 0.9)
        with torch.no_grad():
            y_binary = layer.forward(x, binary=True)
            x_masked = Ops.top2_select_ste(x)
            y_via_helper = layer.forward(x_masked)
        self.assertTrue(torch.allclose(y_binary, y_via_helper))

    def test_binary_drops_long_tail_to_silence(self):
        """A near-zero tail should produce almost the same output as
        just the top-2 entries passed through, since x=0 -> atanh(0)=0
        is the additive identity in SigmaLayer's tanh domain."""
        _set_seed(31)
        layer = SigmaLayer(nInput=5, nOutput=3, invertible=True,
                           monotonic=True, nonlinear=True)
        x = torch.tensor([[[1e-4, 0.7, -2e-4, -0.6, 1e-5]]])
        with torch.no_grad():
            y_binary = layer.forward(x, binary=True)
            x_clean = torch.tensor([[[0.0, 0.7, 0.0, -0.6, 0.0]]])
            y_clean = layer.forward(x_clean)
        self.assertTrue(torch.allclose(y_binary, y_clean, atol=1e-6))

    def test_binary_ste_gradient_to_unselected(self):
        """STE: gradient flows to every input dim."""
        _set_seed(37)
        layer = SigmaLayer(nInput=5, nOutput=4, invertible=True,
                           monotonic=True, nonlinear=True)
        x = torch.tensor([[[0.1, -0.9, 0.05, 0.7, 0.02]]],
                         requires_grad=True)
        y = layer.forward(x, binary=True)
        y.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertTrue((x.grad.abs() > 0).all())


class TestRoundTripUnaffected(unittest.TestCase):
    """Per-layer reverse(forward(x)) round-trip is unchanged by the
    binary=True addition (default path is untouched)."""

    def test_pi_round_trip_default(self):
        _set_seed(41)
        layer = PiLayer(nInput=4, nOutput=4, invertible=True,
                        monotonic=True, nonlinear=True)
        x = torch.randn(1, 2, 4).clamp(-0.9, 0.9)
        with torch.no_grad():
            y = layer.forward(x)
            x_back = layer.reverse(y)
        torch.testing.assert_close(x, x_back, atol=1e-4, rtol=1e-3)

    def test_sigma_round_trip_default(self):
        _set_seed(43)
        layer = SigmaLayer(nInput=4, nOutput=4, invertible=True,
                           monotonic=True, nonlinear=True)
        x = torch.randn(1, 2, 4).clamp(-0.9, 0.9)
        with torch.no_grad():
            y = layer.forward(x)
            x_back = layer.reverse(y)
        torch.testing.assert_close(x, x_back, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
