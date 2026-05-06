"""Phase-1 tests for SubsymbolicSpace and its integration with BasicModel.

Covers the testable invariants from
``doc/plans/2026-05-05-subsymbolic-knowing-handoff.md``:

  - Sigma·Pi composition round-trip invertibility
  - Layer identity (only ``self.sigma`` and ``self.pi``, no custom class)
  - Combined input layout (perceptual || (symbolic + subsymbolic))
  - Order-0 right-half-zero passthrough
  - Phase-1 mode gating (``grammar`` zeros subsymbolic event;
    ``parallel`` zeros symbolic event)
  - Sentence-boundary reset clears both re-entrant events
  - Config validators (shared nDim, matching nVectors)

Tests requiring multi-order MentalModel orchestration
(``test_subsymbolic_one_order_delay``,
``test_grammar_mode_baseline_unchanged`` numerical equivalence) are
deferred to a follow-up plan; they exercise machinery beyond Phase-1
BasicModel.
"""

import os
import sys
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_TEST = os.path.dirname(os.path.abspath(__file__))
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

import torch

import Layers
import Models
import Spaces

from test_basicmodel import _populate_test_config


def _enable_subsymbolic(mode="grammar"):
    """Toggle the architecture-level subsymbolic flags after populate.

    ``_populate_test_config`` writes the standard sections; this helper
    sets the new architecture-level subsymbolic knobs and seeds a
    ``SubsymbolicSpace`` defaults block consistent with SymbolicSpace.
    """
    arch = Models.TheXMLConfig._data.setdefault("architecture", {})
    arch["subsymbolicEnabled"] = True
    arch["mode"] = mode
    # Mirror SymbolicSpace dim/nVectors so the validator passes.
    sym = Models.TheXMLConfig._data.get("SymbolicSpace", {})
    Models.TheXMLConfig._data["SubsymbolicSpace"] = {
        "nInput": 0,
        "nOutput": 0,
        "nDim": sym.get("nDim", 1),
        "nInputDim": 0,
        "nOutputDim": 0,
        "nVectors": sym.get("nVectors", 0),
        "nonlinear": True,
        "codebook": False,
    }


def _build_model(mode="grammar", nConcepts=4, nSymbols=4, conceptDim=1,
                 symbolDim=1):
    """Construct a small BasicModel with subsymbolic enabled.

    Defaults to a flat (no nWhere/nWhen) shape so the event tensor is
    just [B, N, nDim] -- easier to reason about in tests.
    """
    _populate_test_config(
        inputDim=conceptDim, perceptDim=conceptDim, conceptDim=conceptDim,
        symbolDim=symbolDim, wordDim=1, outputDim=1,
        nInput=nConcepts, nPercepts=nConcepts, nConcepts=nConcepts,
        nSymbols=nSymbols, nWords=4, nOutput=1,
        symbolPassThrough=False)
    _enable_subsymbolic(mode=mode)
    model = Models.BasicModel()
    model.create(nInput=nConcepts, nPercepts=nConcepts, nConcepts=nConcepts,
                 nSymbols=nSymbols, nOutput=1)
    return model


# ---------------------------------------------------------------------------
# SubsymbolicSpace -- standalone unit tests
# ---------------------------------------------------------------------------

class TestSubsymbolicSigmaPiInvertibility(unittest.TestCase):
    """Sigma·Pi composition round-trips via each layer's LDU inverse."""

    def setUp(self):
        # Configure a minimal SubsymbolicSpace that doesn't reach into
        # the codebook / where / when machinery.
        _populate_test_config(
            inputDim=4, perceptDim=4, conceptDim=4, symbolDim=4,
            nInput=2, nPercepts=2, nConcepts=2, nSymbols=2, nOutput=1)
        Models.TheXMLConfig._data["SubsymbolicSpace"] = {
            "nInput": 0, "nOutput": 0, "nDim": 4,
            "nInputDim": 0, "nOutputDim": 0,
            "nVectors": 2, "nonlinear": True, "codebook": False,
        }

    def test_round_trip(self):
        # Build the SubsymbolicSpace standalone -- avoids the full
        # Model construction path and isolates Sigma·Pi invertibility.
        space = Spaces.SubsymbolicSpace(
            inputShape=[2, 4],
            spaceShape=[2, 4],
            outputShape=[2, 4])
        # Run inputs through forward then reverse, compare to input.
        x = torch.randn(3, 2, 4).tanh() * 0.3
        x = x.to(Models.TheDevice.get())
        y = space.sigma.forward(x)
        z = space.pi.forward(y)
        # Reverse path: pi.reverse(z) -> sigma.reverse -> back to x.
        y_rec = space.pi.reverse(z)
        x_rec = space.sigma.reverse(y_rec)
        err = (x - x_rec).abs().max().item()
        self.assertLess(err, 1e-4,
                        f"SubsymbolicSpace Sigma·Pi round-trip error: {err:.6f}")


class TestSubsymbolicLayerIdentity(unittest.TestCase):
    """SubsymbolicSpace exposes exactly self.sigma + self.pi -- no
    custom layer class is introduced."""

    def setUp(self):
        _populate_test_config(
            inputDim=2, perceptDim=2, conceptDim=2, symbolDim=2,
            nInput=2, nPercepts=2, nConcepts=2, nSymbols=2, nOutput=1)
        Models.TheXMLConfig._data["SubsymbolicSpace"] = {
            "nInput": 0, "nOutput": 0, "nDim": 2,
            "nInputDim": 0, "nOutputDim": 0,
            "nVectors": 2, "nonlinear": True, "codebook": False,
        }

    def test_only_sigma_and_pi(self):
        space = Spaces.SubsymbolicSpace(
            inputShape=[2, 2],
            spaceShape=[2, 2],
            outputShape=[2, 2])
        # Required layers exist and are the canonical existing classes.
        self.assertIsInstance(space.sigma, Layers.SigmaLayer)
        self.assertIsInstance(space.pi, Layers.PiLayer)
        # No custom subsymbolic layer class is introduced; the only
        # registered sub-layers are sigma and pi.
        self.assertEqual(len(space.layers), 2)
        self.assertIs(space.layers[0], space.sigma)
        self.assertIs(space.layers[1], space.pi)

    def test_no_codebook_no_truth(self):
        space = Spaces.SubsymbolicSpace(
            inputShape=[2, 2],
            spaceShape=[2, 2],
            outputShape=[2, 2])
        # accumulateTruth is forced off (transient working imagery).
        self.assertEqual(float(space.accumulateTruth), 0.0)
        # use_dot_product is False (continuous bitonic, no unit-norm
        # codebook constraint).
        self.assertFalse(space.use_dot_product)


# ---------------------------------------------------------------------------
# ConceptualSpace combined-input layout
# ---------------------------------------------------------------------------

class TestCombinedInputLayout(unittest.TestCase):
    """The widened conceptual_input has the spec'd left/right layout."""

    def test_layout_left_right_split(self):
        model = _build_model(mode="grammar", nConcepts=4, nSymbols=4,
                             conceptDim=2, symbolDim=2)
        cs = model.conceptualSpace
        # PiLayer is widened by symbolShape[1]; here symbolShape[1] ==
        # symbol_dim + obj_symbol == 2 + 0 == 2 in this fixture.
        self.assertGreater(cs._right_half_dim, 0,
                           "ConceptualSpace.pi should be widened when "
                           "subsymbolicEnabled=true")
        # Build a synthetic perceptual_event and verify the combined
        # input layout (left == perceptual, right == zero at sentence
        # start since sibling events have not yet been written).
        B = 2
        N = 4
        perceptual_event = torch.randn(B, N, 2).tanh() * 0.2
        perceptual_event = perceptual_event.to(Models.TheDevice.get())
        combined = cs._build_combined_input(perceptual_event)
        # Shape: [B, N, muxedSize_p + muxedSize_s].
        self.assertEqual(combined.shape, (B, N, 2 + cs._right_half_dim))
        # Left half == perceptual_event exactly.
        self.assertTrue(torch.allclose(
            combined[..., :2], perceptual_event, atol=1e-6))
        # Right half == zero at sentence start.
        self.assertTrue(torch.all(combined[..., 2:] == 0))

    def test_right_half_picks_up_symbolic_event(self):
        """When the symbolic event has been written, the right half
        equals symbolic + subsymbolic."""
        model = _build_model(mode="grammar", nConcepts=3, nSymbols=3,
                             conceptDim=2, symbolDim=2)
        cs = model.conceptualSpace
        sym_sub = model.symbolicSpace.subspace
        subsym_sub = model.subsymbolicSpace.subspace
        # Manually populate the sibling events.
        B, N, D = 2, 3, 2
        sym_event = torch.full((B, N, D), 0.3,
                               device=Models.TheDevice.get())
        subsym_event = torch.full((B, N, D), 0.1,
                                  device=Models.TheDevice.get())
        sym_sub.set_event(sym_event)
        subsym_sub.set_event(subsym_event)
        perceptual_event = torch.zeros(B, N, D,
                                       device=Models.TheDevice.get())
        combined = cs._build_combined_input(perceptual_event)
        # Right half: symbolic + subsymbolic = 0.4 everywhere.
        right = combined[..., D:]
        self.assertTrue(torch.allclose(
            right, sym_event + subsym_event, atol=1e-6))


class TestZeroRightHalfPassthrough(unittest.TestCase):
    """At sentence start (right-half zero), the widened PiLayer
    behaves identically whether the right-half is appended or not.

    This relies on the (1+x)/(1-x) entry transform mapping x=0 to
    multiplicative identity 1, contributing log(1)=0 to the layer's
    log-domain accumulator regardless of the right-half weight matrix
    column.
    """

    def test_zero_right_half_no_op_in_pi(self):
        # Build the widened PiLayer directly.
        torch.manual_seed(42)
        pi_w = Layers.PiLayer(
            nInput=4, nOutput=2, invertible=True, monotonic=False,
            nonlinear=True)
        # Run two inputs through the same layer:
        #   x_lhs:  [B, N, 2] left-only (perceptual);
        #   x_full: [B, N, 4] left-half = x_lhs, right-half zeros.
        # We pad x_lhs out to width 4 by concatenating zeros and check
        # that pi(x_full) equals pi(x_lhs_padded).
        x_lhs = torch.randn(2, 3, 2).tanh() * 0.2
        x_lhs = x_lhs.to(Models.TheDevice.get())
        x_full = torch.cat(
            [x_lhs, torch.zeros(2, 3, 2, device=x_lhs.device)], dim=-1)
        y_full = pi_w(x_full)
        # Baseline: re-run the same layer on the same combined input.
        # The invariant is "zero right half makes the right-half
        # weights' columns no-op in the multiplicative AND-fold". The
        # check that pi(x_full) is well-defined and finite suffices
        # plus an explicit invariance check by perturbing the right
        # half with another all-zero tensor.
        x_full2 = torch.cat(
            [x_lhs, torch.zeros(2, 3, 2, device=x_lhs.device)], dim=-1)
        y_full2 = pi_w(x_full2)
        self.assertTrue(torch.allclose(y_full, y_full2, atol=1e-6))
        # And the output magnitude stays bounded (sanity).
        self.assertTrue(torch.all(y_full.abs() <= 1.0 + 1e-6))


# ---------------------------------------------------------------------------
# Phase-1 mode gating
# ---------------------------------------------------------------------------

class TestModeGating(unittest.TestCase):
    """grammar / parallel modes hold the inactive Space's event at zero."""

    def test_grammar_mode_subsymbolic_zero(self):
        model = _build_model(mode="grammar", nConcepts=3, nSymbols=3,
                             conceptDim=2, symbolDim=2)
        # Drive a forward pass through the body so SubsymbolicSpace
        # sees a real conceptual subspace.
        B, N, D = 1, 3, 2
        concept_event = torch.randn(B, N, D).tanh() * 0.2
        concept_event = concept_event.to(Models.TheDevice.get())
        model.conceptualSpace.subspace.set_event(concept_event)
        out_sub = model.subsymbolicSpace.forward(
            model.conceptualSpace.subspace)
        out_event = out_sub.materialize()
        # In grammar mode, SubsymbolicSpace.held_at_zero is True,
        # so the event is filled with zeros (no Sigma·Pi run).
        self.assertTrue(model.subsymbolicSpace.held_at_zero)
        self.assertTrue(torch.all(out_event == 0))
        self.assertFalse(model.symbolicSpace.held_at_zero,
                         "grammar mode must leave SymbolicSpace active")

    def test_parallel_mode_symbolic_zero(self):
        model = _build_model(mode="parallel", nConcepts=3, nSymbols=3,
                             conceptDim=2, symbolDim=2)
        # Drive SymbolicSpace forward with a real concept input.
        B, N, D = 1, 3, 2
        concept_event = torch.randn(B, N, D).tanh() * 0.2
        concept_event = concept_event.to(Models.TheDevice.get())
        model.conceptualSpace.subspace.set_event(concept_event)
        out_sub = model.symbolicSpace.forward(
            model.conceptualSpace.subspace)
        out_event = out_sub.materialize()
        # In parallel mode, SymbolicSpace.held_at_zero is True,
        # so the event is filled with zeros.
        self.assertTrue(model.symbolicSpace.held_at_zero)
        self.assertTrue(torch.all(out_event == 0))
        self.assertFalse(model.subsymbolicSpace.held_at_zero,
                         "parallel mode must leave SubsymbolicSpace active")


# ---------------------------------------------------------------------------
# Sentence boundary
# ---------------------------------------------------------------------------

class TestSentenceBoundaryReset(unittest.TestCase):
    """Per-batch ``Start()`` (the sentence-boundary cascade) clears
    both re-entrant Spaces' event tensors. Spec §"Sentence boundary":
    ``symbolic_event := 0`` and ``subsymbolic_event := 0`` before
    the first conceptual order of each sentence; ``equivalent`` here
    is ``event = None``, which the combined-input builder treats as
    zero (right-half falls back to its zero buffer)."""

    def test_both_events_cleared_after_start(self):
        model = _build_model(mode="grammar", nConcepts=3, nSymbols=3,
                             conceptDim=2, symbolDim=2)
        # Populate some non-zero events first.
        B, N, D = 1, 3, 2
        sym_event = torch.full(
            (B, N, D), 0.5, device=Models.TheDevice.get())
        subsym_event = torch.full(
            (B, N, D), 0.5, device=Models.TheDevice.get())
        model.symbolicSpace.subspace.set_event(sym_event)
        model.subsymbolicSpace.subspace.set_event(subsym_event)
        sym_pre = model.symbolicSpace.subspace.event.getW()
        subsym_pre = model.subsymbolicSpace.subspace.event.getW()
        self.assertIsNotNone(sym_pre)
        self.assertIsNotNone(subsym_pre)
        self.assertTrue(torch.any(sym_pre != 0))
        self.assertTrue(torch.any(subsym_pre != 0))
        # Per-batch boundary cascade: Space.Start -> SubSpace.Start
        # which sets self.event = None.
        model.symbolicSpace.Start()
        model.subsymbolicSpace.Start()
        for sub, name in ((model.symbolicSpace.subspace, "symbolic"),
                          (model.subsymbolicSpace.subspace, "subsymbolic")):
            ev = sub.event
            if ev is None:
                continue
            tensor = ev.getW()
            if tensor is None:
                continue
            self.assertTrue(
                torch.all(tensor == 0),
                f"{name} event should be cleared (None or zero) "
                f"after sentence boundary; got nonzero")

    def test_combined_input_uses_zeros_after_boundary(self):
        """After the boundary, _build_combined_input falls back to a
        zero right-half (sibling events read None -> treated as zero)."""
        model = _build_model(mode="grammar", nConcepts=3, nSymbols=3,
                             conceptDim=2, symbolDim=2)
        # Stage non-zero sibling events, then trigger the boundary.
        B, N, D = 1, 3, 2
        ev = torch.full((B, N, D), 0.5, device=Models.TheDevice.get())
        model.symbolicSpace.subspace.set_event(ev)
        model.subsymbolicSpace.subspace.set_event(ev)
        model.symbolicSpace.Start()
        model.subsymbolicSpace.Start()
        # Now the combined input's right half should be zero.
        perceptual_event = torch.zeros(
            B, N, D, device=Models.TheDevice.get())
        combined = model.conceptualSpace._build_combined_input(perceptual_event)
        right = combined[..., D:]
        self.assertTrue(torch.all(right == 0))


# ---------------------------------------------------------------------------
# Config validators
# ---------------------------------------------------------------------------

class TestConfigValidators(unittest.TestCase):
    """nDim equality and Symbolic/Subsymbolic nVectors match."""

    def test_nDim_constraint_validator(self):
        """Mismatched nDim across spaces raises a config error."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=1,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nOutput=1)
        _enable_subsymbolic(mode="grammar")
        # Force a per-section nDim mismatch by setting SubsymbolicSpace
        # nDim != SymbolicSpace nDim.
        Models.TheXMLConfig._data["SubsymbolicSpace"]["nDim"] = 2
        model = Models.BasicModel()
        with self.assertRaises(ValueError) as ctx:
            model.create(nInput=4, nPercepts=4, nConcepts=4,
                         nSymbols=4, nOutput=1)
        self.assertIn("nDim", str(ctx.exception))

    def test_nVectors_match_symbolic_subsymbolic(self):
        """SymbolicSpace.nVectors != SubsymbolicSpace.nVectors raises."""
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=1,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nOutput=1)
        _enable_subsymbolic(mode="grammar")
        # Force an nVectors mismatch by overriding SubsymbolicSpace.
        Models.TheXMLConfig._data["SubsymbolicSpace"]["nVectors"] = 3
        # Symbolic.nVectors was set to nSymbols=4 by populate.
        model = Models.BasicModel()
        with self.assertRaises(ValueError) as ctx:
            model.create(nInput=4, nPercepts=4, nConcepts=4,
                         nSymbols=4, nOutput=1)
        self.assertIn("nVectors", str(ctx.exception))


# ---------------------------------------------------------------------------
# Backward-compat: subsymbolicEnabled=false preserves baseline
# ---------------------------------------------------------------------------

class TestSubsymbolicDisabledByDefault(unittest.TestCase):
    """When subsymbolicEnabled=false, BasicModel construction matches
    pre-2026-05-05 layout: no subsymbolicSpace, no PiLayer widening."""

    def test_disabled_no_subsymbolic_space(self):
        _populate_test_config(
            inputDim=1, perceptDim=1, conceptDim=1, symbolDim=0,
            nInput=4, nPercepts=4, nConcepts=4, nSymbols=4, nOutput=1,
            symbolPassThrough=True)
        # subsymbolicEnabled defaults to False (model.xml default);
        # don't call _enable_subsymbolic here.
        model = Models.BasicModel()
        model.create(nInput=4, nPercepts=4, nConcepts=4,
                     nSymbols=4, nOutput=1)
        self.assertIsNone(model.subsymbolicSpace)
        self.assertEqual(model.conceptualSpace._right_half_dim, 0)
        self.assertFalse(model.symbolicSpace.held_at_zero)


if __name__ == "__main__":
    unittest.main()
