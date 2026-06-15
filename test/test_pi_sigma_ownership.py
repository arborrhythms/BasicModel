"""Tests for Pi / Sigma layer ownership.

Post-Stage-1.A + Stage-1.C substrate refactor (doc/plans/
2026-05-26-two-loop-pi-sigma-substrate.md). Stage 10 (doc/plans/
2026-05-27-perceptstore-meta-taxonomy-reentrancy.md) revisions noted.

    PartSpace        -- ``pi`` (PiLayer). Stage 10 retired
                              ``self.sigma`` from PS — PS is pi-only.
                              The sigma half migrates to
                              ``ConceptualSpace.sigma_in`` per stage.
    ConceptualSpace        -- NO ``sigma_percept`` and (2026-06-04) NO
                              ``sigma_in`` / ``sigma_cs``: CS is a pure
                              bookkeeping carrier (forward pushes to STM;
                              forward and reverse are symmetric).
    WholeSpace          -- OWNS ``self.sigma`` (invertible SigmaLayer;
                              butterfly when configured) -- the symbolic-
                              loop generalization operator and the
                              ``S = sigma(S)`` binding target. No ``pi``.
    OutputSpace            -- NO `_piLayer`. The ``nonlinear_output``
                              path uses an ``InvertibleLinearLayer``
                              wrapped with ``atanh -> linear -> tanh``.

See:
- basicmodel/doc/Spaces.md (ownership tables)
- basicmodel/doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md
- basicmodel/doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import unittest
import warnings

import torch

import Models
import Language
from Layers import PiLayer, SigmaLayer, InvertibleLinearLayer
from Layers import MeronymicFoldAdapter
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, 'data')
# Switched from MentalModel.xml to MM_xor_loopback.xml because the
# former is broken on ``main`` (WordSubSpace.__init__ reads
# ``self.subspace`` which doesn't exist for that config path).
# MM_xor_loopback.xml exercises the same PS/CS/SS ownership rules
# and is what ``test_perceptual_loopback.py`` already uses cleanly.
_CONFIG_PATH = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS_PATH = os.path.join(_DATA_DIR, "model.xml")


def _make_plain_model():
    """Build a plain-mode model so the ownership asserts hit the
    PiLayer / SigmaLayer instances."""
    init_config(path=_CONFIG_PATH, defaults_path=_DEFAULTS_PATH)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG_PATH)
    Models.TheData.load("xor")
    model.eval()
    return model


class TestOwnership(unittest.TestCase):
    """Each space exposes (or does not expose) the documented layer
    attribute. The architectural rule is: only PS / CS may own
    SigmaLayer / PiLayer instances."""

    def test_perceptual_sigma_folds(self):
        # Pi/Sigma swap (analysis/synthesis plan Phase 3, rev. 2026-06-09):
        # PS owns a single ``sigma`` (SigmaLayer) -- the bottom-up
        # synthesis/union fold. The per-order Ramsified ``pi_input`` /
        # ``pi_concept`` ModuleLists stay retired.
        model = _make_plain_model()
        ps = model.perceptualSpace
        # Stage 9 cutover (2026-06-11): with <meronomy>on (the model.xml default) the meronymic slot binds the membership kernel via MeronymicFoldAdapter; the OWNERSHIP contract is unchanged.
        self.assertIsInstance(ps.sigma, (SigmaLayer, MeronymicFoldAdapter))
        if isinstance(ps.sigma, MeronymicFoldAdapter):
            self.assertEqual(ps.sigma.kind, 'sigma')
        self.assertFalse(
            hasattr(ps, 'pi'),
            "PartSpace.pi moved to WholeSpace (Pi/Sigma swap); "
            "PS is sigma-only (synthesis).")
        # 2026-06-04: ConceptualSpace is a pure bookkeeping carrier now --
        # the per-stage sigma_in / sigma_cs were retired; the symbolic-loop
        # sigma lives on WholeSpace (see test_symbolic_owns_sigma).
        cs = model.conceptualSpaces[0]
        self.assertFalse(
            hasattr(cs, 'sigma_in'),
            "ConceptualSpace.sigma_in is retired; CS is a bookkeeping "
            "carrier.")

    def test_conceptual_no_sigma_percept(self):
        # Post-Stage-1.C: CS no longer owns sigma_percept. The atomic
        # forward C-tier fold is retired; CS.forward is STM bookkeeping
        # (see test_cs_stm_bookkeeping.py for the positive contract).
        model = _make_plain_model()
        self.assertFalse(
            hasattr(model.conceptualSpace, 'sigma_percept'),
            "ConceptualSpace.sigma_percept must be retired by "
            "Stage 1.C.")

    def test_symbolic_owns_pi(self):
        # Pi/Sigma swap (analysis/synthesis plan Phase 3, rev. 2026-06-09):
        # WholeSpace OWNS the pi (the top-down analysis/intersection
        # operator; the binding target for the S-tier fold rule -- bound
        # under BOTH the 'pi' rule name and the legacy 'sigma' alias).
        model = _make_plain_model()
        # Stage 9 cutover (2026-06-11): with <meronomy>on (the model.xml default) the meronymic slot binds the membership kernel via MeronymicFoldAdapter; the OWNERSHIP contract is unchanged.
        fold = getattr(model.symbolicSpace, 'pi', None)
        self.assertIsInstance(
            fold, (PiLayer, MeronymicFoldAdapter),
            "WholeSpace must own a pi under the corrected "
            "analysis/synthesis ownership rule.")
        if isinstance(fold, MeronymicFoldAdapter):
            self.assertEqual(fold.kind, 'pi')

    def test_symbolic_has_no_sigma(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.symbolicSpace, 'sigma'),
                         "WholeSpace.sigma moved to PartSpace "
                         "(Pi/Sigma swap); SS is pi-only (analysis).")

    def test_conceptual_has_no_bare_sigma(self):
        # Post-Stage-1.C: the bare ``sigma`` attribute (and
        # ``sigma_percept``) is retired. Stage 10 reintroduces
        # ``sigma_in`` / ``sigma_cs`` as the per-stage owned sigmas;
        # the BARE ``sigma`` attribute is still retired (it never
        # existed under either contract).
        model = _make_plain_model()
        self.assertFalse(hasattr(model.conceptualSpace, 'sigma'),
                         "ConceptualSpace must NOT own a bare ``sigma`` "
                         "attribute; the atomic C-tier fold is retired "
                         "by Stage 1.C. Stage 10's sigma_in / sigma_cs "
                         "are differently named.")

    def test_perceptual_has_sigma(self):
        # Pi/Sigma swap (rev. 2026-06-09): PS owns a bare ``sigma``
        # (single-layer instance, not ModuleList). The legacy
        # ``pi_input`` / ``pi_concept`` ModuleList interface stays
        # retired.
        model = _make_plain_model()
        self.assertTrue(hasattr(model.perceptualSpace, 'sigma'),
                        "PartSpace must own a bare ``sigma`` "
                        "(SigmaLayer) post Pi/Sigma swap.")
        self.assertFalse(hasattr(model.perceptualSpace, 'pi'),
                         "Pi/Sigma swap: PartSpace.pi moved to "
                         "WholeSpace.")

    def test_output_has_no_pilayer(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.outputSpace, '_piLayer'),
                         "OutputSpace._piLayer is replaced by "
                         "_linearLayer (InvertibleLinearLayer) under "
                         "the ownership rule.")

    def test_output_linear_layer_when_nonlinear_output(self):
        """If the active config exercises the nonlinear_output path,
        the replacement attribute must be an InvertibleLinearLayer."""
        model = _make_plain_model()
        if getattr(model.outputSpace, 'nonlinear_output', False):
            self.assertIsInstance(
                model.outputSpace._linearLayer,
                InvertibleLinearLayer,
            )


class TestForwardReverseAliases(unittest.TestCase):
    """The bare ``forwardPi`` / ``reversePi`` / ``forwardSigma`` /
    ``reverseSigma`` pointer attributes were removed by the
    2026-05-01 syntactic-layer refactor. Likewise, SS no longer has
    ``sigma`` or ``_sigma_reverse`` after the ownership-rule cleanup.
    """

    def test_aliases_removed(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        ss = model.symbolicSpace
        self.assertFalse(hasattr(cs, 'forwardPi'),
                         "ConceptualSpace.forwardPi alias removed")
        self.assertFalse(hasattr(cs, 'reversePi'),
                         "ConceptualSpace.reversePi alias removed")
        self.assertFalse(hasattr(ss, 'forwardSigma'),
                         "WholeSpace.forwardSigma alias removed")
        self.assertFalse(hasattr(ss, 'reverseSigma'),
                         "WholeSpace.reverseSigma alias removed")
        self.assertFalse(hasattr(ss, '_sigma_reverse'),
                         "WholeSpace._sigma_reverse removed with sigma")

    def test_conceptual_no_sigma_percept_forward(self):
        # Post-Stage-1.C: the ``sigma_percept`` SigmaLayer (and the
        # ``_sigma_percept_reverse`` two-pass ergodic helper) are
        # retired with the atomic C-tier fold. CS.forward is now STM
        # bookkeeping (see test_cs_stm_bookkeeping.py).
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertFalse(hasattr(cs, 'sigma_percept'),
                         "CS.sigma_percept retired by Stage 1.C.")
        self.assertFalse(hasattr(cs, '_sigma_percept_reverse'),
                         "CS._sigma_percept_reverse retired with "
                         "``sigma_percept``.")


class TestConceptualSpaceSTMBookkeeping(unittest.TestCase):
    """Stage-1.C contract gate (light mirror of
    test_cs_stm_bookkeeping.py): CS.forward mutates STM rather than
    applying a parameterised fold."""

    def test_cs_forward_uses_stm_not_fold(self):
        """A single CS.forward(PS_sub) call must increase the STM
        depth (per the bookkeeping contract). The retired fold layer
        ``sigma_percept`` is gone so there is no atomic operator the
        forward could dispatch on."""
        model = _make_plain_model()
        cs = model.conceptualSpace
        # Empty the STM at the start of the test so the depth delta
        # is unambiguous.
        cs.stm.ensure_batch(1)
        cs.stm.clear()
        # Build a small concrete PS subspace by running an IS forward.
        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = model.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub, _ = model.inputSpace.forward(x_input)
                ps_sub = model.perceptualSpace.forward(in_sub)
                # Ensure STM is correctly sized for the batch.
                cs.stm.ensure_batch(
                    max(1, int(ps_sub.materialize().shape[0])))
                cs.stm.clear()
                depth_before = cs.stm.size(0)
                cs.forward(ps_sub)
                depth_after = cs.stm.size(0)
        self.assertGreater(
            depth_after, depth_before,
            "CS.forward must push to STM (Stage 1.C bookkeeping).")


if __name__ == "__main__":
    unittest.main()
