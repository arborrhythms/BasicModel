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
butterfly mode is enabled on PerceptualSpace.pi via the new XML knob.
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
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=8)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), 8)
        self.assertEqual(int(layer.n_levels), 3)

    def test_butterfly_packed_parameter_shape(self):
        """``self.butterfly_W`` is a single packed nn.Parameter of shape
        [n_levels, N//2, 2D, 2D]."""
        D, N = 3, 8
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        self.assertIsInstance(layer.butterfly_W, torch.nn.Parameter)
        expected = (int(math.log2(N)), N // 2, 2 * D, 2 * D)
        self.assertEqual(tuple(layer.butterfly_W.shape), expected)

    def test_butterfly_perms_buffer_shape(self):
        """``self.butterfly_perms`` is a buffer of shape [n_levels, N]."""
        D, N = 3, 8
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        # Registered as a buffer -> accessible as attribute, contained in
        # state_dict, not a parameter.
        perms = getattr(layer, 'butterfly_perms', None)
        self.assertIsNotNone(perms)
        self.assertIsInstance(perms, torch.Tensor)
        self.assertFalse(isinstance(perms, torch.nn.Parameter))
        self.assertEqual(tuple(perms.shape), (int(math.log2(N)), N))

    def test_butterfly_requires_power_of_two(self):
        """N must be a power of 2; non-pow-2 values raise."""
        with self.assertRaises((ValueError, AssertionError)):
            GrammarLayer(nInput=2, nOutput=2, butterfly=True, N=7)


class TestButterflyIdentityInit(unittest.TestCase):
    """At init, ``forward(x) == x`` within tolerance for a base
    GrammarLayer with butterfly=True. The default ``_butterfly_pair_op``
    is a plain einsum, so identity per-node init (L=I, d=1, U=I) lifts
    to identity at the cascade level."""

    def test_grammar_layer_identity_forward_n4(self):
        torch.manual_seed(0)
        D, N = 3, 4
        B = 2
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = torch.randn(B, N, D)
        y = layer.forward(x)
        self.assertEqual(y.shape, x.shape)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL, f"GrammarLayer butterfly identity init forward failed: err={err}")

    def test_grammar_layer_identity_forward_n8(self):
        torch.manual_seed(1)
        D, N = 5, 8
        B = 4
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        x = torch.randn(B, N, D)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, _TOL, f"GrammarLayer butterfly identity init forward failed: err={err}")

    def test_grammar_layer_identity_reverse_n4(self):
        torch.manual_seed(2)
        D, N = 3, 4
        B = 2
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        y = torch.randn(B, N, D)
        x = layer.reverse(y)
        err = (x - y).abs().max().item()
        self.assertLess(err, _TOL, f"GrammarLayer butterfly identity init reverse failed: err={err}")


class TestButterflyRoundtrip(unittest.TestCase):
    """``reverse(forward(x)) ~= x`` for every supported N."""

    def _roundtrip(self, D, N):
        torch.manual_seed(0)
        B = 2
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        # Nudge the parameters off identity so the test is non-trivial.
        with torch.no_grad():
            layer.butterfly_W.add_(0.01 * torch.randn_like(layer.butterfly_W))
        x = torch.randn(B, N, D) * 0.5
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = (x_rec - x).norm() / (x.norm() + 1e-9)
        return err.item()

    def test_roundtrip_n2(self):
        err = self._roundtrip(D=3, N=2)
        self.assertLess(err, 1e-4, f"N=2 roundtrip err={err}")

    def test_roundtrip_n4(self):
        err = self._roundtrip(D=3, N=4)
        self.assertLess(err, 1e-4, f"N=4 roundtrip err={err}")

    def test_roundtrip_n8(self):
        err = self._roundtrip(D=4, N=8)
        self.assertLess(err, 1e-4, f"N=8 roundtrip err={err}")

    def test_roundtrip_n16(self):
        err = self._roundtrip(D=3, N=16)
        self.assertLess(err, 1e-4, f"N=16 roundtrip err={err}")


class TestButterflyParameterCount(unittest.TestCase):
    """The single packed Parameter holds exactly N log2(N) 2D^2 floats."""

    def test_param_count_n8_d3(self):
        D, N = 3, 8
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        n_levels = int(math.log2(N))
        expected = n_levels * (N // 2) * (2 * D) * (2 * D)
        # ``butterfly_W`` is the packed parameter; LDU bookkeeping (raw_L,
        # d, raw_U) is allocated separately. We only assert the packed-W
        # count here, which is the headline formula on the contract.
        self.assertEqual(layer.butterfly_W.numel(), expected)

    def test_param_count_formula(self):
        """N log2(N) 2D^2 == n_levels (N/2) (2D)^2."""
        D, N = 4, 16
        layer = GrammarLayer(nInput=D, nOutput=D, butterfly=True, N=N)
        n_levels = int(math.log2(N))
        # Two equivalent formulations.
        # 1) n_levels * (N // 2) * (2D)^2
        # 2) N * log2(N) * 2 * D^2
        f1 = n_levels * (N // 2) * (2 * D) ** 2
        f2 = N * int(math.log2(N)) * 2 * D ** 2
        self.assertEqual(f1, f2)
        self.assertEqual(layer.butterfly_W.numel(), f1)


class TestPiLayerButterfly(unittest.TestCase):
    """PiLayer inherits butterfly capability and applies the pi pair op."""

    def test_constructs(self):
        D, N = 3, 8
        layer = PiLayer(nInput=D, nOutput=D, invertible=True,
                        butterfly=True, N=N)
        self.assertTrue(layer.butterfly)
        self.assertEqual(int(layer.N), N)

    def test_identity_init(self):
        torch.manual_seed(3)
        D, N = 3, 8
        B = 2
        layer = PiLayer(nInput=D, nOutput=D, invertible=True,
                        butterfly=True, N=N)
        # Stay well inside (-1, 1) so atanh doesn't saturate.
        x = torch.tanh(torch.randn(B, N, D) * 0.1)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        # Pi pair op is atanh -> matmul -> tanh; identity init gives
        # tanh(atanh(x)) == x.
        self.assertLess(err, 1e-5,
                        f"PiLayer butterfly identity-init failed: err={err}")

    def test_forward_reverse_roundtrip(self):
        torch.manual_seed(4)
        D, N = 3, 8
        B = 2
        layer = PiLayer(nInput=D, nOutput=D, invertible=True,
                        butterfly=True, N=N)
        with torch.no_grad():
            layer.butterfly_W.add_(0.005 * torch.randn_like(layer.butterfly_W))
        # Stay well inside (-1, 1) so the atanh / tanh stack in the
        # cascade doesn't saturate -- the per-pair op composes the
        # nonlinearity 3 times for N=8, so values near the boundary
        # lose precision rapidly.
        x = torch.tanh(torch.randn(B, N, D) * 0.1)
        y = layer.forward(x)
        x_rec = layer.reverse(y)
        err = (x_rec - x).norm() / (x.norm() + 1e-9)
        # The cascade with 3 nonlinear levels has ~1e-3 - 1e-4
        # achievable; we want substantively-better-than-trivial.
        self.assertLess(err.item(), 1e-2,
                        f"PiLayer butterfly roundtrip failed: err={err.item()}")


class TestSigmaLayerButterfly(unittest.TestCase):
    """SigmaLayer inherits butterfly capability and applies the sigma pair op."""

    def test_constructs(self):
        D, N = 3, 8
        layer = SigmaLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        self.assertTrue(layer.butterfly)

    def test_identity_init(self):
        torch.manual_seed(5)
        D, N = 3, 8
        B = 2
        layer = SigmaLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        # Stay well inside (-1, 1) so atanh doesn't saturate.
        x = torch.tanh(torch.randn(B, N, D) * 0.1)
        y = layer.forward(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, 1e-5,
                        f"SigmaLayer butterfly identity-init failed: err={err}")

    def test_forward_reverse_roundtrip(self):
        torch.manual_seed(6)
        D, N = 3, 8
        B = 2
        layer = SigmaLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        with torch.no_grad():
            layer.butterfly_W.add_(0.005 * torch.randn_like(layer.butterfly_W))
        # Stay well inside (-1, 1) so the cascade atanh / tanh stack
        # doesn't saturate.
        x = torch.tanh(torch.randn(B, N, D) * 0.1)
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
        D, N = 3, 8
        layer = LiftLayer(nInput=D, nOutput=D, invertible=True,
                          butterfly=True, N=N)
        self.assertTrue(layer.butterfly)

    def test_lift_butterfly_identity_init(self):
        torch.manual_seed(7)
        D, N = 3, 8
        B = 2
        layer = LiftLayer(nInput=D, nOutput=D, invertible=True,
                          butterfly=True, N=N)
        x = torch.tanh(torch.randn(B, N, D) * 0.1)
        y = layer.forward_butterfly(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, 1e-5,
                        f"LiftLayer butterfly identity-init failed: err={err}")

    def test_lower_butterfly_constructs(self):
        D, N = 3, 8
        layer = LowerLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        self.assertTrue(layer.butterfly)

    def test_lower_butterfly_identity_init(self):
        torch.manual_seed(8)
        D, N = 3, 8
        B = 2
        layer = LowerLayer(nInput=D, nOutput=D, invertible=True,
                           butterfly=True, N=N)
        x = torch.tanh(torch.randn(B, N, D) * 0.1)
        y = layer.forward_butterfly(x)
        err = (y - x).abs().max().item()
        self.assertLess(err, 1e-5,
                        f"LowerLayer butterfly identity-init failed: err={err}")


if __name__ == "__main__":
    unittest.main()
