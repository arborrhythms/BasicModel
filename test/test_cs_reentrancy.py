"""ConceptualSpace bookkeeping carrier + SymbolicSpace sigma ownership.

2026-06-04 parallel-symbolic-substrate refactor. SUPERSEDES the Stage-10
per-stage CS ``sigma_in`` / ``sigma_cs`` + residual-lift design (that
machinery -- and its ``_prev_cs_event_cache`` roundtrip cache -- is
RETIRED). New ownership / forward contract:

  * PerceptualSpace -- pi-only (PiLayer; no sigma).
  * ConceptualSpace -- a PURE BOOKKEEPING CARRIER: no ``sigma_in`` /
    ``sigma_cs`` / ``sigma``. ``forward`` pushes the perceptual event
    onto the STM; forward and reverse are symmetric (no parameterised
    fold to invert), which is what makes the reconstruction round-trip
    exact.
  * SymbolicSpace   -- OWNS ``self.sigma`` (invertible SigmaLayer; a
    BUTTERFLY cascade when ``<butterfly>true</...>`` so it has cross-slot
    reach). It is the symbolic-loop generalization operator and the
    binding target for the default ``S = sigma(S)`` grammar rule.
  * The non-grammar PARALLEL forward = perception (PS->CS_0) followed by
    ``conceptualOrder`` applications of ``SymbolicSpace.sigma``
    (``BasicModel._symbolic_sigma_step``); the reverse inverts each
    ``sigma`` before ``ConceptualSpace.reverse`` so the round-trip stays
    exact.

The gate builds ``XOR_exact.xml`` (parallel, conceptualOrder=1, invertible
PS/CS/SS passthroughs, butterfly pi AND sigma) -- the fixture this refactor
restored to exact 4/4 reconstruction (see
test_explicit_dimensions.TestXorExactCliReconstruction for the end-to-end
convergence gate).
"""

import os
import sys
import unittest
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from Layers import SigmaLayer, PiLayer
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG = os.path.join(_DATA_DIR, "XOR_exact.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_model():
    """Build XOR_exact (parallel invertible chain, SS butterfly sigma)."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG)
    m.eval()
    return m


class TestPerceptualSigmaOnly(unittest.TestCase):
    def test_ps_has_sigma_no_pi(self):
        # Pi/Sigma swap (analysis/synthesis plan Phase 3, rev. 2026-06-09).
        m = _make_model()
        ps = m.perceptualSpace
        self.assertIsInstance(ps.sigma, SigmaLayer)
        self.assertFalse(hasattr(ps, "pi"),
                         "PerceptualSpace is sigma-only (synthesis); "
                         "pi moved to SymbolicSpace.")


class TestConceptualIsBookkeepingCarrier(unittest.TestCase):
    """CS owns no parameterised fold -- the Stage-10 ``sigma_in`` /
    ``sigma_cs`` + residual-lift cache are RETIRED."""

    def test_cs_has_no_sigma_layers(self):
        m = _make_model()
        for k, cs in enumerate(m.conceptualSpaces):
            for attr in ("sigma_in", "sigma_cs", "sigma"):
                self.assertFalse(
                    hasattr(cs, attr),
                    f"ConceptualSpace[{k}].{attr} must be retired -- CS is "
                    f"a pure bookkeeping carrier.")

    def test_cs_has_no_active_residual_cache(self):
        # The Stage-10 PARALLEL residual-lift roundtrip cache is gone; if
        # the attribute lingers it must never hold a live tensor.
        m = _make_model()
        for k, cs in enumerate(m.conceptualSpaces):
            self.assertIsNone(
                getattr(cs, "_prev_cs_event_cache", None),
                f"ConceptualSpace[{k}]._prev_cs_event_cache must not hold "
                f"a residual-lift tensor (the machinery is retired).")

    def test_conceptualSpaces_is_module_list_len_order(self):
        m = _make_model()
        self.assertIsInstance(m.conceptualSpaces, torch.nn.ModuleList)
        self.assertEqual(
            len(m.conceptualSpaces), max(1, int(m.conceptualOrder)),
            "self.conceptualSpaces length must equal max(1, "
            "conceptualOrder).")


class TestSymbolicOwnsSigma(unittest.TestCase):
    """SymbolicSpace owns the invertible (butterfly) sigma."""

    def test_ss_owns_invertible_pi(self):
        # Pi/Sigma swap (rev. 2026-06-09): SS owns the analysis fold.
        m = _make_model()
        for k, ss in enumerate(m.symbolicSpaces):
            fold = getattr(ss, "pi", None)
            self.assertIsInstance(
                fold, PiLayer,
                f"SymbolicSpace[{k}] must own a pi (PiLayer).")
            self.assertTrue(
                getattr(fold, "invertible", False),
                f"SymbolicSpace[{k}].pi must be invertible so the "
                f"reconstruction reverse can apply pi.reverse exactly.")
            self.assertFalse(
                hasattr(ss, "sigma"),
                f"SymbolicSpace[{k}] must NOT own a sigma (Sigma moved "
                f"to PS -- synthesis).")

    def test_ss_fold_butterfly_wired_from_xml(self):
        # XOR_exact sets <butterfly>true</butterfly> on SymbolicSpace; the
        # flag must reach the fold constructor (cross-slot reach -- a
        # per-slot square fold cannot combine the two word slots for
        # XOR). N is the flattened content count inputShape[0]*nOutputDim.
        m = _make_model()
        ss = m.symbolicSpace
        self.assertTrue(
            getattr(ss.pi, "butterfly", False),
            "SymbolicSpace.<butterfly>true</...> must wire a butterfly "
            "cascade into SymbolicSpace.pi.")
        self.assertEqual(
            int(ss.butterflyN),
            int(ss.inputShape[0]) * int(ss.nOutputDim),
            "SS fold butterfly N must be inputShape[0] * nOutputDim "
            "(flattened content element count).")


class TestSymbolicSigmaStepRoundtrips(unittest.TestCase):
    """The per-order symbolic step is exactly invertible -- the basis of
    the exact reconstruction round-trip."""

    def test_fold_step_forward_then_reverse_recovers_carrier(self):
        # Action C (2026-06-06) removed the ``_symbolic_sigma_step`` wrapper
        # (the parallel carrier advance moved into the ConceptualCombine on
        # the full muxed event). The invariant it guarded -- the SS fold is
        # exactly invertible, the basis of the reconstruction round-trip --
        # is now asserted directly on ``ss.pi`` (the fold post Pi/Sigma
        # swap, rev. 2026-06-09).
        m = _make_model()
        ss = m.symbolicSpace
        fold = getattr(ss, "pi", None)
        self.assertIsNotNone(
            fold, "SymbolicSpace must own a pi to round-trip the carrier.")
        N = int(ss.inputShape[0])
        D = int(ss.nOutputDim)            # content width the fold acts on
        torch.manual_seed(0)
        # Keep values inside the atanh domain so the nonlinear fold
        # round-trips to LDU precision.
        ev = torch.randn(1, N, D).clamp(-0.5, 0.5)
        with torch.no_grad():
            fwd = fold.forward(ev.clone())
            rec = fold.reverse(fwd)
        err = (ev - rec).abs().max().item()
        self.assertLess(
            err, 1e-3,
            f"ss.pi forward->reverse must round-trip the carrier to LDU "
            f"precision; got max abs error {err:g}.")


if __name__ == "__main__":
    unittest.main()
