"""Tests for Pi / Sigma layer ownership.

Post-swap layout (P->C uses Pi, C->S uses Sigma -- swapped from the
historical assignments; the data flow direction P->C->S forward,
S->C->P reverse is unchanged):

    PerceptualSpace.sigma  -- SigmaLayer (P -> sub-percept; dormant per spec O3)
    ConceptualSpace.pi     -- PiLayer (P -> C). Two-pass ergodic mode
                              aliases ``self.pi`` to ``self.pi1``;
                              ``self._pi_reverse`` hides the pi2 dispatch.
    SymbolicSpace.sigma    -- SigmaLayer (C -> S). Same alias pattern;
                              ``self._sigma_reverse`` hides sigma2.

Each layer round-trips through its own inverse.

The 2026-05-01 syntactic-layer refactor removed the bare
``forwardPi``/``reversePi``/``forwardSigma``/``reverseSigma`` pointer
aliases; callers now use ``self.pi.forward`` / ``self._pi_reverse`` /
``self.sigma.forward`` / ``self._sigma_reverse`` directly.

See:
- basicmodel/doc/specs/2026-04-24-lift-lower-bivector-design.md (§B-summary)
- basicmodel/doc/specs/2026-05-01-syntactic-layer-refactor.md (alias removal)
- basicmodel/doc/Logic.md §8 (the level-crossing axis)
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
from Layers import PiLayer, SigmaLayer
from util import init_config

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _make_plain_model(config='MentalModel.xml'):
    """Build a plain-mode (non-butterfly) model so the ownership
    asserts hit the plain PiLayer / SigmaLayer instances rather than the
    butterfly-aware layers used by butterfly configs."""
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
    """Each space exposes the documented layer attribute and type."""

    def test_perceptual_sigma(self):
        model = _make_plain_model()
        self.assertIsInstance(model.perceptualSpace.sigma, SigmaLayer)

    def test_conceptual_pi(self):
        model = _make_plain_model()
        self.assertIsInstance(model.conceptualSpace.pi, PiLayer)

    def test_symbolic_sigma(self):
        model = _make_plain_model()
        self.assertIsInstance(model.symbolicSpace.sigma, SigmaLayer)

    def test_conceptual_has_no_sigma(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.conceptualSpace, 'sigma'),
                         "ConceptualSpace.sigma is gone after the swap; "
                         "use conceptualSpace.pi instead.")

    def test_symbolic_has_no_pi(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.symbolicSpace, 'pi'),
                         "SymbolicSpace.pi is gone after the swap; "
                         "use symbolicSpace.sigma instead.")

    def test_perceptual_has_no_pi(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.perceptualSpace, 'pi'),
                         "PerceptualSpace.pi is gone after the swap; "
                         "use perceptualSpace.sigma instead.")

    def test_symbolic_layer_alias_is_gone(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.symbolicSpace, 'layer'),
                         "SymbolicSpace.layer (deprecated alias) is gone; "
                         "use symbolicSpace.sigma directly.")


class TestForwardReverseAliases(unittest.TestCase):
    """Post-2026-05-01 refactor: the bare ``forwardPi`` / ``reversePi`` /
    ``forwardSigma`` / ``reverseSigma`` pointer attributes are gone.
    Callers use ``self.pi.forward`` directly (and ``self._pi_reverse``
    in two-pass mode); same for sigma.
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

    def test_conceptual_has_pi_forward(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertTrue(callable(cs.pi.forward))
        self.assertTrue(callable(cs._pi_reverse))

    def test_symbolic_has_sigma_forward(self):
        model = _make_plain_model()
        ss = model.symbolicSpace
        self.assertTrue(callable(ss.sigma.forward))
        self.assertTrue(callable(ss._sigma_reverse))


class TestPerLayerRoundTrip(unittest.TestCase):
    """Each layer round-trips through its own inverse."""

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

    def test_symbolic_sigma_round_trip(self):
        """Sigma.reverse(Sigma.forward(c)) ~= c on the symbolic space's sigma (C->S path)."""
        model = _make_plain_model()
        sigma = model.symbolicSpace.sigma
        N = 4
        eps = 1e-3
        c = torch.randn(1, N, sigma.nInput).clamp(-1 + eps, 1 - eps)
        with torch.no_grad():
            c_back = sigma.reverse(sigma.forward(c))
        torch.testing.assert_close(c, c_back, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
