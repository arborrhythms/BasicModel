"""Lift/Lower factorization acceptance (Phase 3, 2026-05-18 owner-ratified).

Phase 3 design (doc/plans/2026-05-18-two-loop-pipeline-architecture.md §D7):
LiftLayer and LowerLayer become elementwise-gate-then-internal-sigma/pi
operators at the S tier.  The gate is the elementwise product of the two
operands at the C-tier:

    cb     = symbolicSpace.subspace.what
    left_c  = cb.reverse(left, project=True)
    right_c = cb.reverse(right, project=True)
    gated   = left_c * right_c
    out_c   = self._sigma(gated)    # LiftLayer; own internal SigmaLayer
                                     #   (or self._pi for LowerLayer)
    return cb.forward(out_c)

Lift's gating operand is VP (predication); Lower's gating operand is ADJ
(attribution).  Both layers own their internal SigmaLayer / PiLayer (NOT
borrowed from PartSpace.sigma / ConceptualSpace.pi); the layers
train independently.  When constructed parameter-free
(``LiftLayer()`` / ``LowerLayer()``), forward falls back to the static
lattice kernel (``Ops._lower_kernel`` for lift / ``Ops._lift_kernel`` for
lower) for the standalone-test harness path.

Tests in this file:
  * ``raw_gate`` / ``sigma_S`` / ``pi_S`` retirements still hold.
  * Wired path: different gating operands produce different outputs.
  * Wired path: lift vs lower differ on the same (gate, NP) input.
  * Wired path: the internal sigma / pi expose trainable parameters.
  * Fallback path: parameter-free construction routes to the static
    lattice kernel (unchanged from the post-2026-05-13 contract).
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

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    """Build a fresh BasicModel from MM_xor.xml."""
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, cfg = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return m


def _random_bivec(shape, seed=0):
    """Random bivector activation in [-1, 1]^shape.

    Internal sigma uses ``atanh(x.clamp(-1+eps, 1-eps))`` on its entry
    transform, so operand values are bounded to ``[-1, 1]`` to keep the
    forward in the valid pre-image of tanh.

    Device-agnostic: seed the global RNG (covers every device's
    default generator) and let ``torch.rand`` place the tensor on the
    active device.
    """
    torch.manual_seed(seed)
    # Scale rand from [0, 1) into [-1, 1) with a slight inset so the
    # internal SigmaLayer / PiLayer's atanh entry stays finite.
    return torch.rand(*shape) * 1.8 - 0.9


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
                f"WholeSpace must not retain `sigma_S` after the "
                f"lift/lower refactor; found on {sym}")

    def test_no_pi_S_on_symbolic_space(self):
        for sym in self.model.symbolicSpaces:
            self.assertFalse(
                hasattr(sym, 'pi_S') and getattr(sym, 'pi_S') is not None,
                f"WholeSpace must not retain `pi_S` after the "
                f"lift/lower refactor; found on {sym}")


class TestLiftLowerFactorization(unittest.TestCase):
    """The elementwise-gate-then-internal-sigma/pi factorization works."""

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
        """Build random ``[B, V_S, D]`` bivector activations matching S's shape."""
        sym = self.model.symbolicSpace
        cb = sym.subspace.what
        V_S = int(cb.nVectors)
        D = int(cb.nDim)
        B = 1
        return B, V_S, D

    def test_different_VPs_give_different_lift_outputs(self):
        """Different VP bivectors -> different gates -> different lift outputs."""
        lift = self._make_lift()
        B, V_S, D = self._bivec_for_test()
        NP = _random_bivec((B, V_S, D), seed=11)
        VP_a = _random_bivec((B, V_S, D), seed=42)
        VP_b = _random_bivec((B, V_S, D), seed=43)
        with torch.no_grad():
            out_a = lift.forward(VP_a, NP)
            out_b = lift.forward(VP_b, NP)
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-6),
                         "Different VP gates must produce different "
                         "lift outputs for the same NP -- this is the "
                         "VP-IS-the-mask factorization at work.")

    def test_different_ADJs_give_different_lower_outputs(self):
        """Different ADJ bivectors -> different gates -> different lower outputs."""
        lower = self._make_lower()
        B, V_S, D = self._bivec_for_test()
        NP = _random_bivec((B, V_S, D), seed=51)
        ADJ_a = _random_bivec((B, V_S, D), seed=52)
        ADJ_b = _random_bivec((B, V_S, D), seed=53)
        with torch.no_grad():
            out_a = lower.forward(ADJ_a, NP)
            out_b = lower.forward(ADJ_b, NP)
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-6),
                         "Different ADJ gates must produce different "
                         "lower outputs for the same NP -- ADJ-IS-the-mask.")

    def test_lift_vs_lower_differ_after_perturbation(self):
        """Same (gate, NP) -> lift and lower differ once their internal
        sigma / pi diverge from identity init.

        Phase 3 design note: at fresh construction, both ``_sigma``
        (SigmaLayer) and ``_pi`` (PiLayer) initialize with the
        identity LDU (``W=I, bias=0``).  The underlying numerical
        kernels then collapse to identity at init -- so day-0
        outputs are numerically indistinguishable.  The asymmetry
        between additive sigma and multiplicative pi materialises
        once training (or any explicit weight perturbation) takes
        the diagonals away from 1.  The test perturbs one
        diagonal element of each layer with *different* signs and
        verifies the outputs then diverge -- proving the
        sigma-vs-pi post-gate transforms are structurally
        independent learnable maps.
        """
        lift = self._make_lift()
        lower = self._make_lower()
        # Perturb the LDU diagonals so the two layers move off
        # identity in different directions; this is the smallest
        # change that proves the structural asymmetry without
        # depending on a full training step.
        with torch.no_grad():
            for p in lift._sigma.parameters():
                if p.dim() == 1 and p.shape[0] > 0:
                    p.add_(0.25)
                    break
            for p in lower._pi.parameters():
                if p.dim() == 1 and p.shape[0] > 0:
                    p.add_(-0.25)
                    break
        B, V_S, D = self._bivec_for_test()
        NP = _random_bivec((B, V_S, D), seed=21)
        GATE = _random_bivec((B, V_S, D), seed=22)
        with torch.no_grad():
            out_lift = lift.forward(GATE, NP)
            out_lower = lower.forward(GATE, NP)
        self.assertFalse(torch.allclose(out_lift, out_lower, atol=1e-6),
                         "Lift and Lower must differ once their "
                         "internal sigma / pi diverge from identity "
                         "init -- the sigma-vs-pi post-gate transforms "
                         "are independent learnable maps.")

    def test_lift_has_internal_sigma_trainable(self):
        """Wired LiftLayer owns an internal ``_sigma`` with trainable params."""
        lift = self._make_lift()
        self.assertIsNotNone(lift._sigma,
                             "Wired LiftLayer must own an internal SigmaLayer.")
        # The internal sigma's params must be trainable.
        from Layers import SigmaLayer
        self.assertIsInstance(lift._sigma, SigmaLayer)
        params = list(lift._sigma.parameters())
        self.assertGreater(len(params), 0,
                           "Internal SigmaLayer must expose nn.Parameters.")
        self.assertTrue(all(p.requires_grad for p in params),
                        "Internal SigmaLayer params must be trainable.")

    def test_lower_has_internal_pi_trainable(self):
        """Wired LowerLayer owns an internal ``_pi`` with trainable params."""
        lower = self._make_lower()
        self.assertIsNotNone(lower._pi,
                             "Wired LowerLayer must own an internal PiLayer.")
        from Layers import PiLayer
        self.assertIsInstance(lower._pi, PiLayer)
        params = list(lower._pi.parameters())
        self.assertGreater(len(params), 0,
                           "Internal PiLayer must expose nn.Parameters.")
        self.assertTrue(all(p.requires_grad for p in params),
                        "Internal PiLayer params must be trainable.")

    def test_lift_self_contained_no_substrate_fallback(self):
        """Stage 4 (2026-05-27): the parameter-free static-lattice
        fallback (``Ops._lower_kernel`` for lift) is retired.  Lift
        is now a self-contained binary GrammarLayer with its own
        internal SigmaLayer; ``LiftLayer(nInput=D, nOutput=D)``
        suffices to run forward / reverse.

        Replaces the prior ``test_lift_fallback_uses_static_lattice_kernel``
        which exercised the retired Phase-3 fallback.
        """
        from Layers import LiftLayer
        B = 1
        V_S = 2
        D = 10
        layer = LiftLayer(nInput=D, nOutput=D)
        torch.manual_seed(31)
        NP = torch.rand(B, V_S, D) * 1.8 - 0.9
        torch.manual_seed(32)
        VP = torch.rand(B, V_S, D) * 1.8 - 0.9
        with torch.no_grad():
            out = layer.forward(VP, NP)
        self.assertEqual(out.shape, NP.shape,
                         "Stage 4 LiftLayer.forward returns a single "
                         "tensor matching the operand shape.")
        self.assertTrue(torch.isfinite(out).all(),
                        "Stage 4 LiftLayer.forward must produce finite "
                        "values (fail-loud numerical contract).")

    def test_lower_self_contained_no_substrate_fallback(self):
        """Stage 4 (2026-05-27): the parameter-free static-lattice
        fallback (``Ops._lift_kernel`` for lower) is retired.  Lower
        is now a self-contained binary GrammarLayer with its own
        internal PiLayer.
        """
        from Layers import LowerLayer
        B = 1
        V_S = 2
        D = 10
        layer = LowerLayer(nInput=D, nOutput=D)
        torch.manual_seed(41)
        NP = torch.rand(B, V_S, D) * 1.8 - 0.9
        torch.manual_seed(42)
        ADJ = torch.rand(B, V_S, D) * 1.8 - 0.9
        with torch.no_grad():
            out = layer.forward(ADJ, NP)
        self.assertEqual(out.shape, NP.shape)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
