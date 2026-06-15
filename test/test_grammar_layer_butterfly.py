"""Stage 5 (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):
butterfly cascade mode on ``GrammarLayer``.

Contract verified here:
  * ``GrammarLayer(... , butterfly=True, N=N)`` constructs successfully.
  * At init the cascade is the identity: ``forward(x) == x`` within tolerance.
  * Forward / reverse roundtrip: ``reverse(forward(x)) ~= x`` for N in {2, 4, 8, 16}.
  * Parameter count formula matches ``n_levels * (N//2) * 4D^2``.
  * ``PiLayer(butterfly=True, N=8)`` works; same identity-init / roundtrip
    holds with the pi-style per-pair semantics.
  * ``SigmaLayer(butterfly=True, N=8)`` works similarly with sigma-style pair op.
  * ``LiftLayer`` / ``LowerLayer`` delegate ``_butterfly_pair_op`` to their
    internal sigma / pi instances.

Acceptance signal: this file passes, and the XOR convergence test
``test/test_mm_xor.py::TestMMXorConvergence::test_convergence`` /
``::test_learns_xor_signal`` close their convergence thresholds once
butterfly mode is enabled on PartSpace.pi via the new XML knob.
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
    PiLayer,
    SigmaLayer,
    LiftLayer,
    LowerLayer,
)


_TOL = 1e-4


class TestButterflyConstruction(unittest.TestCase):
    """Constructing a GrammarLayer with butterfly=True works and exposes
    the expected state (flags, packed Parameter, perms buffer)."""

    def test_grammar_layer_constructs_with_butterfly_flag(self):
        layer = GrammarLayer(nInput=4, nOutput=4)
        # The default config does NOT carry butterfly state.
        self.assertFalse(getattr(layer, 'butterfly', False))

    def test_grammar_layer_butterfly_true_n8(self):
        D = 3
        # ``N`` is the total flattened element count (= N_slots * D when
        # the cascade processes an [N_slots, D] payload).
        N = 8
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), N)
        self.assertEqual(int(layer.n_levels), 3)

    def test_butterfly_packed_parameter_shape(self):
        """Per-pair LDU storage: scalars L (sub-diag), d (two diag),
        U (super-diag). Shapes are
            butterfly_L : [n_levels, N//2]
            butterfly_d : [n_levels, N//2, 2]
            butterfly_U : [n_levels, N//2]
        (Replaces the legacy single packed ``butterfly_W`` 2D-block
        scheme; 2x2 nodes invert in closed form so 4 scalars suffice.)"""
        N = 8
        layer = GrammarLayer(nInput=2, nOutput=2, butterfly=True, N=N)
        self.assertIsInstance(layer.butterfly_L, torch.nn.Parameter)
        self.assertIsInstance(layer.butterfly_d, torch.nn.Parameter)
        self.assertIsInstance(layer.butterfly_U, torch.nn.Parameter)
        n_levels = int(math.log2(N))
        self.assertEqual(tuple(layer.butterfly_L.shape),
                         (n_levels, N // 2))
        self.assertEqual(tuple(layer.butterfly_d.shape),
                         (n_levels, N // 2, 2))
        self.assertEqual(tuple(layer.butterfly_U.shape),
                         (n_levels, N // 2))

    def test_butterfly_perms_buffer_shape(self):
        """``self.butterfly_perms`` is a buffer of shape [n_levels, N]."""
        N = 8
        layer = GrammarLayer(nInput=2, nOutput=2, butterfly=True, N=N)
        # Registered as a buffer -> accessible as attribute, contained in
        # state_dict, not a parameter.
        perms = getattr(layer, 'butterfly_perms', None)
        self.assertIsNotNone(perms)
        self.assertIsInstance(perms, torch.Tensor)
        self.assertFalse(isinstance(perms, torch.nn.Parameter))
        self.assertEqual(tuple(perms.shape), (int(math.log2(N)), N))

    def test_butterfly_requires_power_of_two(self):
        """Non-power-of-2 N is auto-padded to the next power of two
        (the cascade structure requires 2^k); ``N`` records the
        nominal value but ``M_total`` is the padded count."""
        layer = GrammarLayer(nInput=2, nOutput=2, butterfly=True, N=7)
        self.assertEqual(int(layer.N), 7)
        self.assertEqual(int(layer.M_total), 8)


class TestButterflyIdentityInit(unittest.TestCase):
    """At init, ``forward(x) == x`` within tolerance for a base
    GrammarLayer with butterfly=True. Per-pair LDU identity init
    (L=0, d=(1,1), U=0) lifts to identity at the cascade level."""

    def test_grammar_layer_identity_forward_n4(self):
        torch.manual_seed(0)
        # Cascade now operates on the flat element axis: N = N_slots * D.
        N_slots, D = 4, 3
        N = N_slots * D
        B = 2
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = torch.randn(B, N_slots, D)
        y = layer.forward(x)
        self.assertEqual(y.shape, x.shape)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL, f"GrammarLayer butterfly identity init forward failed: err={err}")

    def test_grammar_layer_identity_forward_n8(self):
        torch.manual_seed(1)
        N_slots, D = 8, 5
        N = N_slots * D
        B = 4
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = torch.randn(B, N_slots, D)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL, f"GrammarLayer butterfly identity init forward failed: err={err}")

    def test_grammar_layer_identity_reverse_n4(self):
        torch.manual_seed(2)
        N_slots, D = 4, 3
        N = N_slots * D
        B = 2
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        y = torch.randn(B, N_slots, D)
        x = layer.reverse(y)
        err = (x - y).abs().max().item()
        self.assertLess(err, _TOL, f"GrammarLayer butterfly identity init reverse failed: err={err}")


class TestButterflyRoundtrip(unittest.TestCase):
    """``reverse(forward(x)) ~= x`` for every supported N (slot count)."""

    def _roundtrip(self, D, N_slots):
        torch.manual_seed(0)
        B = 2
        N = N_slots * D
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        # Nudge the parameters off identity so the test is non-trivial.
        # Three separate per-pair LDU buffers (L, d, U) replace the
        # legacy single packed ``butterfly_W``.
        with torch.no_grad():
            layer.butterfly_L.add_(0.01 * torch.randn_like(layer.butterfly_L))
            layer.butterfly_d.add_(0.01 * torch.randn_like(layer.butterfly_d))
            layer.butterfly_U.add_(0.01 * torch.randn_like(layer.butterfly_U))
        x = torch.randn(B, N_slots, D) * 0.5
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = (x_rec - x).norm() / (x.norm() + 1e-9)
        return err.item()

    def test_roundtrip_n2(self):
        err = self._roundtrip(D=3, N_slots=2)
        self.assertLess(err, 1e-3, f"N_slots=2 roundtrip err={err}")

    def test_roundtrip_n4(self):
        err = self._roundtrip(D=3, N_slots=4)
        self.assertLess(err, 1e-3, f"N_slots=4 roundtrip err={err}")

    def test_roundtrip_n8(self):
        err = self._roundtrip(D=4, N_slots=8)
        self.assertLess(err, 1e-3, f"N_slots=8 roundtrip err={err}")

    def test_roundtrip_n16(self):
        err = self._roundtrip(D=3, N_slots=16)
        self.assertLess(err, 1e-3, f"N_slots=16 roundtrip err={err}")


class TestButterflyParameterCount(unittest.TestCase):
    """Per-pair LDU storage: each node owns 4 scalars (L, d0, d1, U),
    so total parameter count is ``n_levels * (N//2) * 4``. No D factor:
    nodes operate on scalar element pairs from a flattened axis."""

    def test_param_count_n8(self):
        N = 8
        layer = GrammarLayer(nInput=2, nOutput=2, butterfly=True, N=N)
        n_levels = int(math.log2(N))
        expected_L = n_levels * (N // 2)
        expected_d = n_levels * (N // 2) * 2
        expected_U = n_levels * (N // 2)
        self.assertEqual(layer.butterfly_L.numel(), expected_L)
        self.assertEqual(layer.butterfly_d.numel(), expected_d)
        self.assertEqual(layer.butterfly_U.numel(), expected_U)
        total = (layer.butterfly_L.numel() + layer.butterfly_d.numel() +
                 layer.butterfly_U.numel())
        self.assertEqual(total, n_levels * (N // 2) * 4)

    def test_param_count_formula(self):
        """n_levels * (N/2) * 4 == N * log2(N) * 2."""
        N = 16
        layer = GrammarLayer(nInput=2, nOutput=2, butterfly=True, N=N)
        n_levels = int(math.log2(N))
        f1 = n_levels * (N // 2) * 4
        f2 = N * int(math.log2(N)) * 2
        self.assertEqual(f1, f2)
        total = (layer.butterfly_L.numel() + layer.butterfly_d.numel() +
                 layer.butterfly_U.numel())
        self.assertEqual(total, f1)


class TestPiLayerButterfly(unittest.TestCase):
    """PiLayer inherits butterfly capability and applies the pi pair op."""

    def test_constructs(self):
        N_slots, D = 8, 3
        N = N_slots * D
        layer = PiLayer(nInput=D, nOutput=D, invertible=True,
                        butterfly=True, N=N)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), N)

    def test_identity_init(self):
        torch.manual_seed(3)
        N_slots, D = 8, 3
        N = N_slots * D
        B = 2
        layer = PiLayer(nInput=D, nOutput=D, invertible=True,
                        butterfly=True, N=N)
        # Stay well inside (-1, 1) so atanh doesn't saturate.
        x = torch.tanh(torch.randn(B, N_slots, D) * 0.1)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        # Pi pair op is atanh -> matmul -> tanh; identity init gives
        # tanh(atanh(x)) == x.
        self.assertLess(err, 1e-5,
                        f"PiLayer butterfly identity-init failed: err={err}")

    def test_forward_reverse_roundtrip(self):
        torch.manual_seed(4)
        N_slots, D = 8, 3
        N = N_slots * D
        B = 2
        layer = PiLayer(nInput=D, nOutput=D, invertible=True,
                        butterfly=True, N=N)
        with torch.no_grad():
            layer.butterfly_L.add_(0.005 * torch.randn_like(layer.butterfly_L))
            layer.butterfly_d.add_(0.005 * torch.randn_like(layer.butterfly_d))
            layer.butterfly_U.add_(0.005 * torch.randn_like(layer.butterfly_U))
        # Stay well inside (-1, 1) so the atanh / tanh stack in the
        # cascade doesn't saturate -- the per-pair op composes the
        # nonlinearity 3 times for N=8, so values near the boundary
        # lose precision rapidly.
        x = torch.tanh(torch.randn(B, N_slots, D) * 0.1)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = (x_rec - x).norm() / (x.norm() + 1e-9)
        # The cascade with several nonlinear levels has ~1e-3 - 1e-4
        # achievable; we want substantively-better-than-trivial.
        self.assertLess(err.item(), 1e-2,
                        f"PiLayer butterfly roundtrip failed: err={err.item()}")


class TestSigmaLayerButterfly(unittest.TestCase):
    """SigmaLayer inherits butterfly capability and applies the sigma pair op."""

    def test_constructs(self):
        N_slots, D = 8, 3
        N = N_slots * D
        layer = SigmaLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        self.assertTrue(layer.butterfly)

    def test_identity_init(self):
        torch.manual_seed(5)
        N_slots, D = 8, 3
        N = N_slots * D
        B = 2
        layer = SigmaLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        # Stay well inside (-1, 1) so atanh doesn't saturate.
        x = torch.tanh(torch.randn(B, N_slots, D) * 0.1)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, 1e-5,
                        f"SigmaLayer butterfly identity-init failed: err={err}")

    def test_forward_reverse_roundtrip(self):
        torch.manual_seed(6)
        N_slots, D = 8, 3
        N = N_slots * D
        B = 2
        layer = SigmaLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        with torch.no_grad():
            layer.butterfly_L.add_(0.005 * torch.randn_like(layer.butterfly_L))
            layer.butterfly_d.add_(0.005 * torch.randn_like(layer.butterfly_d))
            layer.butterfly_U.add_(0.005 * torch.randn_like(layer.butterfly_U))
        # Stay well inside (-1, 1) so the cascade atanh / tanh stack
        # doesn't saturate.
        x = torch.tanh(torch.randn(B, N_slots, D) * 0.1)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = (x_rec - x).norm() / (x.norm() + 1e-9)
        self.assertLess(err.item(), 1e-2,
                        f"SigmaLayer butterfly roundtrip failed: err={err.item()}")


class TestLiftLowerButterflyDelegation(unittest.TestCase):
    """LiftLayer's ``_butterfly_pair_op`` delegates to ``self._sigma``;
    LowerLayer's delegates to ``self._pi``. Construction validates the
    delegation exists; identity-init still holds at the cascade level."""

    def test_lift_butterfly_constructs(self):
        N_slots, D = 8, 3
        N = N_slots * D
        layer = LiftLayer(nInput=D, nOutput=D, invertible=True,
                          butterfly=True, N=N)
        self.assertTrue(layer.butterfly)

    def test_lift_butterfly_identity_init(self):
        torch.manual_seed(7)
        N_slots, D = 8, 3
        N = N_slots * D
        B = 2
        layer = LiftLayer(nInput=D, nOutput=D, invertible=True,
                          butterfly=True, N=N)
        x = torch.tanh(torch.randn(B, N_slots, D) * 0.1)
        y = layer.forward_butterfly(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, 1e-5,
                        f"LiftLayer butterfly identity-init failed: err={err}")

    def test_lower_butterfly_constructs(self):
        N_slots, D = 8, 3
        N = N_slots * D
        layer = LowerLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        self.assertTrue(layer.butterfly)

    def test_lower_butterfly_identity_init(self):
        torch.manual_seed(8)
        N_slots, D = 8, 3
        N = N_slots * D
        B = 2
        layer = LowerLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        x = torch.tanh(torch.randn(B, N_slots, D) * 0.1)
        y = layer.forward_butterfly(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, 1e-5,
                        f"LowerLayer butterfly identity-init failed: err={err}")


if __name__ == "__main__":
    unittest.main()
