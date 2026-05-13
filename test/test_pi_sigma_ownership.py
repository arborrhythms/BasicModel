"""Tests for Pi / Sigma layer ownership.

Post-2026-05 refactor: SigmaLayer / PiLayer instances are owned ONLY
by PerceptualSpace and ConceptualSpace. SymbolicSpace and OutputSpace
hold no SigmaLayer / PiLayer (their previous remap layers were either
moved or replaced):

    PerceptualSpace.sigma  -- SigmaLayer (P -> sub-percept).
    ConceptualSpace.pi     -- PiLayer (P -> C). Two-pass ergodic mode
                              aliases ``self.pi`` to ``self.pi1``;
                              ``self._pi_reverse`` hides the pi2 dispatch.
    SymbolicSpace          -- NO sigma / pi attribute. With
                              ``concept_dim == symbol_dim`` enforced in
                              ``__init__``, the C->S transform is a
                              dimensional pass-through. The codebook
                              snap (and SyntacticLayer dispatch, if
                              configured) still runs.
    OutputSpace            -- NO `_piLayer`. The ``nonlinear_output``
                              path uses an ``InvertibleLinearLayer``
                              wrapped with ``atanh -> linear -> tanh``.

ConceptualSpace.pi still round-trips through its own inverse.

See:
- basicmodel/doc/Spaces.md (ownership tables)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import warnings

import torch

import Models
import Language
from Layers import PiLayer, SigmaLayer, InvertibleLinearLayer
from util import init_config

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _make_plain_model(config='MentalModel.xml'):
    """Build a plain-mode model so the ownership asserts hit the
    PiLayer / SigmaLayer instances."""
    init_config(
        path=os.path.join(_DATA_DIR, config),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(
            os.path.join(_DATA_DIR, config))
    model.eval()
    return model


class TestOwnership(unittest.TestCase):
    """Each space exposes (or does not expose) the documented layer
    attribute. The architectural rule is: only PS / CS may own
    SigmaLayer / PiLayer instances."""

    def test_perceptual_sigma(self):
        model = _make_plain_model()
        self.assertIsInstance(model.perceptualSpace.sigma, SigmaLayer)

    def test_conceptual_pi(self):
        model = _make_plain_model()
        self.assertIsInstance(model.conceptualSpace.pi, PiLayer)

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
        model = _make_plain_model()
        self.assertFalse(hasattr(model.conceptualSpace, 'sigma'),
                         "ConceptualSpace.sigma is gone after the swap; "
                         "use conceptualSpace.pi instead.")

    def test_perceptual_has_no_pi(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.perceptualSpace, 'pi'),
                         "PerceptualSpace.pi is gone after the swap; "
                         "use perceptualSpace.sigma instead.")

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

    def test_conceptual_has_pi_forward(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertTrue(callable(cs.pi.forward))
        self.assertTrue(callable(cs._pi_reverse))


class TestPerLayerRoundTrip(unittest.TestCase):
    """ConceptualSpace.pi round-trips through its own inverse."""

    def test_conceptual_pi_round_trip(self):
        """Pi.reverse(Pi.forward(p)) ~= p on the conceptual space's pi (P->C path)."""
        model = _make_plain_model()
        pi = model.conceptualSpace.pi
        N = 4
        eps = 1e-3
        x = torch.randn(1, N, pi.nInput).clamp(-1 + eps, 1 - eps)
        with torch.no_grad():
            x_back = pi.reverse(pi.forward(x))
        torch.testing.assert_close(x, x_back, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
