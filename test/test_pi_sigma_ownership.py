"""Tests for Pi / Sigma layer ownership.

Post-Stage-1.A + Stage-1.C substrate refactor (doc/plans/
2026-05-26-two-loop-pi-sigma-substrate.md):

    PerceptualSpace        -- ``pi`` (PiLayer) and ``sigma``
                              (SigmaLayer); both percept_dim ->
                              percept_dim. forward composes them as
                              ``pi(x) + sigma(x)`` on the SAME input.
                              (Stage 1.A.)
    ConceptualSpace        -- NO ``sigma_percept`` attribute. Stage 1.C
                              retired the atomic forward C-tier fold;
                              CS.forward is STM bookkeeping (see
                              test_cs_stm_bookkeeping.py for the
                              positive-contract gates).
    SymbolicSpace          -- NO sigma / pi attribute. With
                              ``concept_dim == symbol_dim`` the C->S
                              transform is a dimensional pass-through;
                              the codebook snap (and SyntacticLayer
                              dispatch, if configured) still runs.
    OutputSpace            -- NO `_piLayer`. The ``nonlinear_output``
                              path uses an ``InvertibleLinearLayer``
                              wrapped with ``atanh -> linear -> tanh``.

See:
- basicmodel/doc/Spaces.md (ownership tables)
- basicmodel/doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md
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

    def test_perceptual_pi_folds(self):
        # Post-Stage-1.A: PS owns a single ``pi`` (PiLayer) and a
        # single ``sigma`` (SigmaLayer); the per-order Ramsified
        # ``pi_input`` / ``pi_concept`` ModuleLists are retired.
        model = _make_plain_model()
        ps = model.perceptualSpace
        self.assertIsInstance(ps.pi, PiLayer)
        self.assertIsInstance(ps.sigma, SigmaLayer)

    def test_conceptual_no_sigma_percept(self):
        # Post-Stage-1.C: CS no longer owns sigma_percept. The atomic
        # forward C-tier fold is retired; CS.forward is STM bookkeeping
        # (see test_cs_stm_bookkeeping.py for the positive contract).
        model = _make_plain_model()
        self.assertFalse(
            hasattr(model.conceptualSpace, 'sigma_percept'),
            "ConceptualSpace.sigma_percept must be retired by "
            "Stage 1.C.")

    def test_symbolic_has_no_sigma(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.symbolicSpace, 'sigma'),
                         "SymbolicSpace.sigma is removed; only PS / CS "
                         "may own SigmaLayer / PiLayer.")

    def test_symbolic_has_no_pi(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.symbolicSpace, 'pi'),
                         "SymbolicSpace.pi is gone; only CS owns the "
                         "PiLayer in the P->C->S pipeline.")

    def test_conceptual_has_no_sigma(self):
        # Post-Stage-1.C: CS owns NEITHER ``sigma`` NOR ``sigma_percept``;
        # the atomic forward C-tier fold operator is retired entirely.
        model = _make_plain_model()
        self.assertFalse(hasattr(model.conceptualSpace, 'sigma'),
                         "ConceptualSpace must NOT own a bare ``sigma`` "
                         "attribute; the atomic C-tier fold is retired "
                         "by Stage 1.C.")

    def test_perceptual_has_pi_and_sigma(self):
        # Post-Stage-1.A: PS owns a bare ``pi`` AND a bare ``sigma``
        # (single-layer instances, not ModuleLists). The legacy
        # ``pi_input`` / ``pi_concept`` ModuleList interface is
        # retired -- the grep gate in the refactor task enforces that
        # they don't reappear in ``bin/Spaces.py``.
        model = _make_plain_model()
        self.assertTrue(hasattr(model.perceptualSpace, 'pi'),
                        "PerceptualSpace must own a bare ``pi`` "
                        "(PiLayer) post-Stage-1.A.")
        self.assertTrue(hasattr(model.perceptualSpace, 'sigma'),
                        "PerceptualSpace must own a bare ``sigma`` "
                        "(SigmaLayer) post-Stage-1.A.")

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
                         "SymbolicSpace.forwardSigma alias removed")
        self.assertFalse(hasattr(ss, 'reverseSigma'),
                         "SymbolicSpace.reverseSigma alias removed")
        self.assertFalse(hasattr(ss, '_sigma_reverse'),
                         "SymbolicSpace._sigma_reverse removed with sigma")

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
                in_sub = model.inputSpace.forward(x_input)
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
