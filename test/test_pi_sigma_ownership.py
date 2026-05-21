"""Tests for Pi / Sigma layer ownership.

Post-2026-05-13 sigma/pi rebalance (doc/Spaces.md §"Sigma / Pi
ownership"): SigmaLayer / PiLayer instances are owned ONLY by
PerceptualSpace and ConceptualSpace, but the ownership SWAPPED vs the
earlier model -- PS now owns the Pi folds, CS owns the Sigma fold:

    PerceptualSpace        -- ``pi_input`` (input_dim -> percept_dim)
                              and ``pi_concept`` (concept_dim ->
                              percept_dim). Ramsified per conceptualOrder
                              into ``nn.ModuleList``s, so element [0]
                              is the order-0 PiLayer.
    ConceptualSpace.sigma_percept
                           -- SigmaLayer (percept_dim -> concept_dim),
                              the canonical C-tier fold. ``invertible``
                              configs round-trip via
                              ``_sigma_percept_reverse`` /
                              ``sigma_percept.reverse``. (The old
                              ``ConceptualSpace.pi`` was removed by the
                              rebalance.)
    SymbolicSpace          -- NO sigma / pi attribute. With
                              ``concept_dim == symbol_dim`` the C->S
                              transform is a dimensional pass-through;
                              the codebook snap (and SyntacticLayer
                              dispatch, if configured) still runs.
    OutputSpace            -- NO `_piLayer`. The ``nonlinear_output``
                              path uses an ``InvertibleLinearLayer``
                              wrapped with ``atanh -> linear -> tanh``.

ConceptualSpace.sigma_percept round-trips through its own inverse.

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

    def test_perceptual_pi_folds(self):
        # Post-rebalance: PS owns pi_input / pi_concept (PiLayers),
        # ramsified per conceptualOrder into ModuleLists.
        model = _make_plain_model()
        ps = model.perceptualSpace
        self.assertIsInstance(ps.pi_input[0], PiLayer)
        self.assertIsInstance(ps.pi_concept[0], PiLayer)

    def test_conceptual_sigma_percept(self):
        # Post-rebalance: CS owns sigma_percept (SigmaLayer), the
        # canonical percept_dim -> concept_dim fold (was ``pi``).
        model = _make_plain_model()
        self.assertIsInstance(model.conceptualSpace.sigma_percept,
                              SigmaLayer)

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
                         "ConceptualSpace owns ``sigma_percept`` (not a "
                         "bare ``sigma``) post-2026-05-13 rebalance.")

    def test_perceptual_has_no_pi(self):
        model = _make_plain_model()
        self.assertFalse(hasattr(model.perceptualSpace, 'pi'),
                         "PerceptualSpace owns ``pi_input`` / "
                         "``pi_concept`` (not a bare ``pi``) "
                         "post-2026-05-13 rebalance.")

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

    def test_conceptual_has_sigma_percept_forward(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertTrue(callable(cs.sigma_percept.forward))
        self.assertTrue(callable(cs._sigma_percept_reverse))


class TestPerLayerRoundTrip(unittest.TestCase):
    """ConceptualSpace.sigma_percept round-trips through its own inverse."""

    def test_conceptual_sigma_percept_round_trip(self):
        """sigma.reverse(sigma.forward(p)) ~= p on the C-tier fold (P->C).

        ``sigma_percept`` is generally dimension-reducing
        (``nInput=nDim+nWhere+nWhen``, ``nOutput=nDim``) — the
        positional (where/when) suffix is dropped on forward and zeroed
        on reverse. Verify invertibility on the content prefix only;
        the suffix is non-invertible by construction. Inputs are
        narrowly clamped so the linear operator stays inside the tanh
        wrap's linear region."""
        model = _make_plain_model()
        sp = model.conceptualSpace.sigma_percept
        N = 4
        n_in, n_out = int(sp.nInput), int(sp.nOutput)
        x = torch.randn(1, N, n_in).clamp(-0.3, 0.3)
        # Zero the positional suffix so the round-trip is meaningful
        # over the dims that ``sigma_percept`` actually encodes.
        if n_out < n_in:
            x[..., n_out:] = 0.0
        with torch.no_grad():
            x_back = sp.reverse(sp.forward(x))
        # Only the first ``n_out`` dims survive the bottleneck.
        torch.testing.assert_close(
            x[..., :n_out], x_back[..., :n_out], atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
