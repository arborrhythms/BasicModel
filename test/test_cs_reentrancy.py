"""ConceptualSpace bookkeeping carrier + WholeSpace sigma ownership.

2026-06-04 parallel-symbolic-substrate refactor. SUPERSEDES the Stage-10
per-stage CS ``sigma_in`` / ``sigma_cs`` + residual-lift design (that
machinery -- and its ``_prev_cs_event_cache`` roundtrip cache -- is
RETIRED). New ownership / forward contract:

  * PartSpace -- pi-only (PiLayer; no sigma).
  * ConceptualSpace -- a PURE BOOKKEEPING CARRIER: no ``sigma_in`` /
    ``sigma_cs`` / ``sigma``. ``forward`` pushes the perceptual event
    onto the STM; forward and reverse are symmetric (no parameterised
    fold to invert), which is what makes the reconstruction round-trip
    exact.
  * WholeSpace   -- OWNS ``self.sigma`` (invertible SigmaLayer; a
    BUTTERFLY cascade when ``<butterfly>true</...>`` so it has cross-slot
    reach). It is the symbolic-loop generalization operator and the
    binding target for the default ``S = sigma(S)`` grammar rule.
  * The non-grammar PARALLEL forward = perception (PS->CS_0) followed by
    ``subsymbolicOrder`` applications of ``WholeSpace.sigma``
    (``BasicModel._symbolic_sigma_step``); the reverse inverts each
    ``sigma`` before ``ConceptualSpace.reverse`` so the round-trip stays
    exact.

The gate builds ``XOR_exact.xml`` (parallel, subsymbolicOrder=1, invertible
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
from Layers import MeronymicFoldAdapter
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
        # Stage 9 cutover (2026-06-11): under <meronomy>on (the
        # model.xml default) the slot binds the membership kernel via
        # MeronymicFoldAdapter; the ownership contract is unchanged.
        self.assertIsInstance(ps.sigma, (SigmaLayer, MeronymicFoldAdapter))
        if isinstance(ps.sigma, MeronymicFoldAdapter):
            self.assertEqual(ps.sigma.kind, 'sigma')
        self.assertFalse(hasattr(ps, "pi"),
                         "PartSpace is sigma-only (synthesis); "
                         "pi moved to WholeSpace.")


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
            len(m.conceptualSpaces), max(1, int(m.subsymbolicOrder)),
            "self.conceptualSpaces length must equal max(1, "
            "subsymbolicOrder).")


class TestSymbolicOwnsSigma(unittest.TestCase):
    """WholeSpace owns the invertible (butterfly) sigma."""

    def test_ws_owns_invertible_pi(self):
        # Pi/Sigma swap (rev. 2026-06-09): SS owns the analysis fold.
        m = _make_model()
        for k, ws in enumerate(m.wholeSpaces):
            fold = getattr(ws, "pi", None)
            # Stage 9 cutover (2026-06-11): with <meronomy>on (the model.xml default) the meronymic slot binds the membership kernel via MeronymicFoldAdapter; the OWNERSHIP contract is unchanged.
            self.assertIsInstance(
                fold, (PiLayer, MeronymicFoldAdapter),
                f"WholeSpace[{k}] must own a pi.")
            if isinstance(fold, MeronymicFoldAdapter):
                self.assertEqual(fold.kind, 'pi')
            self.assertTrue(
                getattr(fold, "invertible", False),
                f"WholeSpace[{k}].pi must be invertible so the "
                f"reconstruction reverse can apply pi.reverse exactly.")
            self.assertFalse(
                hasattr(ws, "sigma"),
                f"WholeSpace[{k}] must NOT own a sigma (Sigma moved "
                f"to PS -- synthesis).")

    def test_ws_fold_butterfly_wired_from_xml(self):
        # XOR_exact sets <butterfly>true</butterfly> on WholeSpace; the
        # flag must reach the fold constructor (cross-slot reach -- a
        # per-slot square fold cannot combine the two word slots for
        # XOR). N is the flattened content count inputShape[0]*nDim.
        m = _make_model()
        ws = m.wholeSpace
        # Stage 9 cutover (2026-06-11): with <meronomy>on (the model.xml default) the meronymic slot binds the membership kernel via MeronymicFoldAdapter; the OWNERSHIP contract is unchanged. Under the adapter the cascade is replaced by the
        # per-slot membership fold; the cross-slot SIZING contract
        # survives as fold.N (the construction-time flat total).
        # RE-PINNED (unified fold-width law, Alec 2026-07-16): the flat
        # total counts CONTENT columns (nDim), not the muxed event width;
        # the where/when band rides through the dispatch trim.
        if isinstance(ws.pi, MeronymicFoldAdapter):
            self.assertEqual(
                int(ws.pi.N),
                int(ws.inputShape[0]) * int(ws.nDim),
                "adapter must record the legacy flat total as N")
        else:
            self.assertTrue(
                getattr(ws.pi, "butterfly", False),
                "WholeSpace.<butterfly>true</...> must wire a butterfly "
                "cascade into WholeSpace.pi.")
        self.assertEqual(
            int(ws.butterflyN),
            int(ws.inputShape[0]) * int(ws.nDim),
            "SS fold butterfly N must be inputShape[0] * nDim "
            "(flattened content element count).")


class TestSymbolicSigmaStepRoundtrips(unittest.TestCase):
    """The per-order symbolic step is exactly invertible -- the basis of
    the exact reconstruction round-trip."""

    def test_fold_step_forward_then_reverse_recovers_carrier(self):
        # Action C (2026-06-06) removed the ``_symbolic_sigma_step`` wrapper
        # (the parallel carrier advance moved into the ConceptualCombine on
        # the full muxed event). The invariant it guarded -- the SS fold is
        # exactly invertible, the basis of the reconstruction round-trip --
        # is now asserted directly on ``ws.pi`` (the fold post Pi/Sigma
        # swap, rev. 2026-06-09).
        m = _make_model()
        ws = m.wholeSpace
        fold = getattr(ws, "pi", None)
        self.assertIsNotNone(
            fold, "WholeSpace must own a pi to round-trip the carrier.")
        N = int(ws.inputShape[0])
        # Unified fold-width law (2026-07-16): the fold acts on CONTENT
        # columns (fold.nInput == ws.nDim); drive it at its own width.
        D = int(fold.nInput)
        torch.manual_seed(0)
        # PS/WS folds operate directly on percept memberships.
        ev = torch.rand(1, N, D)
        with torch.no_grad():
            fwd = fold.forward(ev.clone())
            rec = fold.reverse(fwd)
        err = (ev - rec).abs().max().item()
        self.assertLess(
            err, 1e-3,
            f"ws.pi forward->reverse must round-trip the membership carrier "
            f"precision; got max abs error {err:g}.")


if __name__ == "__main__":
    unittest.main()
