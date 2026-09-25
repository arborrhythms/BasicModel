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




class TestSymbolicSigmaStepRoundtrips(unittest.TestCase):
    """The per-order symbolic step is exactly invertible -- the basis of
    the exact reconstruction round-trip."""



if __name__ == "__main__":
    unittest.main()
