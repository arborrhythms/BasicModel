"""Lift/Lower factorization acceptance (post-2026-05-12 refactor).

The refactor: LiftLayer and LowerLayer become elementwise-gate-then-
substrate-sigma/pi operators. The gate is ``VP_c * NP_c`` at C-tier;
the substrate sigma (P.sigma, now at concept_dim) runs for lift; the
substrate pi (C.pi) runs for lower.

Tests in this file:
  * Different VPs produce different lift outputs for the same NP --
    proves the gate factorization works (VP determines the operation).
  * Lift vs Lower differ on the same (VP, NP) -- proves the sigma vs
    pi post-gate choice is what gives lift/lower their asymmetry.
  * ``raw_gate`` learnable parameter is gone from both layers.
  * ``sigma_S`` / ``pi_S`` separate layers on SymbolicSpace are gone.
"""
import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor_bivector.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    """Build a fresh BasicModel from MM_xor_bivector.xml."""
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, cfg = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return m


def _random_bivec(shape, seed=0):
    """Random non-negative bivector activation in [0, 1]^shape."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g)


class TestRawGateRetired(unittest.TestCase):
    """LiftLayer / LowerLayer no longer carry a ``raw_gate`` parameter."""

    def test_lift_layer_no_raw_gate(self):
        from Layers import LiftLayer
        layer = LiftLayer(symbolicSpace=None, perceptualSpace=None)
        self.assertFalse(hasattr(layer, 'raw_gate')
                         and getattr(layer, 'raw_gate') is not None,
                         "LiftLayer.raw_gate must be retired after the "
                         "VP-codebook-gate refactor.")
        param_names = {n for n, _ in layer.named_parameters()}
        self.assertNotIn('raw_gate', param_names,
                         "LiftLayer must not register raw_gate as a "
                         "Parameter.")

    def test_lower_layer_no_raw_gate(self):
        from Layers import LowerLayer
        layer = LowerLayer(symbolicSpace=None, conceptualSpace=None)
        self.assertFalse(hasattr(layer, 'raw_gate')
                         and getattr(layer, 'raw_gate') is not None)
        param_names = {n for n, _ in layer.named_parameters()}
        self.assertNotIn('raw_gate', param_names)


class TestSigmaSPiSRetired(unittest.TestCase):
    """``space.sigma_S`` / ``space.pi_S`` separate layers are gone."""

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()

    def test_no_sigma_S_on_symbolic_space(self):
        for sym in self.model.symbolicSpaces:
            self.assertFalse(
                hasattr(sym, 'sigma_S') and getattr(sym, 'sigma_S') is not None,
                f"SymbolicSpace must not retain `sigma_S` after the "
                f"lift/lower refactor; found on {sym}")

    def test_no_pi_S_on_symbolic_space(self):
        for sym in self.model.symbolicSpaces:
            self.assertFalse(
                hasattr(sym, 'pi_S') and getattr(sym, 'pi_S') is not None,
                f"SymbolicSpace must not retain `pi_S` after the "
                f"lift/lower refactor; found on {sym}")


class TestLiftLowerFactorization(unittest.TestCase):
    """The elementwise-gate-then-sigma/pi factorization works as designed."""

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()

    def _make_lift(self):
        from Layers import LiftLayer
        return LiftLayer(symbolicSpace=self.model.symbolicSpace,
                         perceptualSpace=self.model.perceptualSpace)

    def _make_lower(self):
        from Layers import LowerLayer
        return LowerLayer(symbolicSpace=self.model.symbolicSpace,
                          conceptualSpace=self.model.conceptualSpace)

    def _bivec_for_test(self):
        """Build random `[B, V_S, 2]` bivector activations matching S's shape."""
        sym = self.model.symbolicSpace
        V_S = int(sym.subspace.what.nVectors)
        B = 1
        return B, V_S

    def test_different_VPs_give_different_lift_outputs(self):
        """Different VP bivectors → different gates → different lift output for the same NP."""
        lift = self._make_lift()
        B, V_S = self._bivec_for_test()
        NP = _random_bivec((B, V_S, 2), seed=11)
        VP_a = _random_bivec((B, V_S, 2), seed=42)
        VP_b = _random_bivec((B, V_S, 2), seed=43)
        with torch.no_grad():
            out_a = lift.forward(VP_a, NP)
            out_b = lift.forward(VP_b, NP)
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-6),
                         "Different VP gates must produce different "
                         "lift outputs for the same NP -- this is the "
                         "VP-IS-the-mask factorization at work.")

    def test_lift_vs_lower_differ(self):
        """Same (VP, NP) -> lift and lower differ (sigma vs pi asymmetry)."""
        lift = self._make_lift()
        lower = self._make_lower()
        B, V_S = self._bivec_for_test()
        NP = _random_bivec((B, V_S, 2), seed=21)
        VP = _random_bivec((B, V_S, 2), seed=22)
        with torch.no_grad():
            out_lift = lift.forward(VP, NP)
            out_lower = lower.forward(VP, NP)
        self.assertFalse(torch.allclose(out_lift, out_lower, atol=1e-6),
                         "Lift and Lower must differ on the same "
                         "(VP, NP) -- the sigma-vs-pi post-gate "
                         "transform is what gives them asymmetry.")

    def test_lift_no_substrate_falls_back_to_static_kernel(self):
        """LiftLayer without refs → fall back to Ops._lower_kernel."""
        from Layers import LiftLayer, Ops
        layer = LiftLayer(symbolicSpace=None, perceptualSpace=None)
        B, V_S = self._bivec_for_test()
        NP = _random_bivec((B, V_S, 2), seed=31)
        VP = _random_bivec((B, V_S, 2), seed=32)
        with torch.no_grad():
            out = layer.forward(VP, NP)
            expected = Ops._lower_kernel(VP, NP, mode='AND', kind='smooth')
        torch.testing.assert_close(out, expected,
                                   msg="Standalone LiftLayer (no refs) "
                                       "must fall back to static lattice "
                                       "kernel for back-compat with "
                                       "harness tests.")

    def test_lower_no_substrate_falls_back_to_static_kernel(self):
        """LowerLayer without refs → fall back to Ops._lift_kernel."""
        from Layers import LowerLayer, Ops
        layer = LowerLayer(symbolicSpace=None, conceptualSpace=None)
        B, V_S = self._bivec_for_test()
        NP = _random_bivec((B, V_S, 2), seed=41)
        VP = _random_bivec((B, V_S, 2), seed=42)
        with torch.no_grad():
            out = layer.forward(VP, NP)
            expected = Ops._lift_kernel(VP, NP, mode='OR', kind='smooth')
        torch.testing.assert_close(out, expected,
                                   msg="Standalone LowerLayer (no refs) "
                                       "must fall back to static lattice "
                                       "kernel for back-compat.")


if __name__ == "__main__":
    unittest.main()
