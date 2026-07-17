"""Stage 1.A substrate refactor: PartSpace single-arg forward.

Post-Stage-1.A contract:

  * ``PartSpace`` owns ``self.pi: PiLayer`` (single layer
    instance, NOT an ``nn.ModuleList``); the per-order Ramsified
    ``pi_input`` / ``pi_concept`` lists are retired. The
    ``subsymbolicOrder`` knob's new role is driving PARALLEL-mode
    forward iteration count over the per-stage CS pipeline.

Stage 10 (doc/plans/2026-05-27-perceptstore-meta-taxonomy-
reentrancy.md) revision:

  * ``PartSpace.sigma`` is RETIRED. PS is pi-only. The sigma
    half migrates to ``ConceptualSpace.sigma_in`` per stage
    (Ramsified across ``self.conceptualSpaces``).

  * ``PartSpace.forward(x_subspace)`` body becomes
    ``return self.pi(x.materialize())`` (drop the ``+ self.sigma(x)``
    term).

  * ``PartSpace.reverse(y_subspace)`` is symmetric (pi-only).

This file is the targeted TDD gate for the refactor. It is independent
of the broader pipeline (loopback test) and uses the same plain config
as ``test_pi_sigma_ownership.py`` so the model boots cheaply.
"""

import inspect
import os
import sys
import unittest
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from Layers import PiLayer, SigmaLayer
from Layers import MeronymicFoldAdapter
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, 'data')
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_plain_model():
    """Build a working model from MM_xor_loopback.xml + xor data so
    the new pi/sigma single-layer instances exist on PartSpace.

    Mirrors ``test/test_perceptual_loopback.py::_fresh_model`` because
    that's a known-good cheap config for PartSpace inspection on
    ``main``.
    """
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


class TestPSOwnsSingleLayers(unittest.TestCase):
    """PartSpace owns ``self.pi`` directly, not a ``ModuleList``
    container. Stage 10 retired ``self.sigma`` from PS."""

    def test_ps_has_sigma_attribute(self):
        # Pi/Sigma swap (analysis/synthesis plan Phase 3, rev. 2026-06-09):
        # PS owns the synthesis fold ``sigma``.
        model = _make_plain_model()
        ps = model.perceptualSpace
        self.assertTrue(hasattr(ps, 'sigma'),
                        "PartSpace must own a ``sigma`` attribute "
                        "post Pi/Sigma swap.")
        # Stage 9 cutover (2026-06-11): with <meronomy>on (the model.xml default) the meronymic slot binds the membership kernel via MeronymicFoldAdapter; the OWNERSHIP contract is unchanged.
        self.assertIsInstance(ps.sigma, (SigmaLayer, MeronymicFoldAdapter),
                              "PartSpace.sigma must be a single "
                              "fold layer (not a ModuleList).")
        if isinstance(ps.sigma, MeronymicFoldAdapter):
            self.assertEqual(ps.sigma.kind, 'sigma')
        self.assertNotIsInstance(ps.sigma, torch.nn.ModuleList,
                                 "PartSpace.sigma must NOT be a "
                                 "ModuleList (single-layer contract).")

    def test_ps_pi_attribute_retired(self):
        """Pi/Sigma swap (rev. 2026-06-09): ``self.pi`` on
        PartSpace moved to WholeSpace (Pi is the top-down
        analysis operator)."""
        model = _make_plain_model()
        ps = model.perceptualSpace
        self.assertFalse(
            hasattr(ps, 'pi'),
            "Pi/Sigma swap: PartSpace.pi must be gone (PS is "
            "sigma-only -- synthesis). The pi lives on WholeSpace.")


class TestPSForwardSingleArg(unittest.TestCase):
    """The new ``forward`` is single positional arg
    (``x_subspace``); the legacy ``CS_subspaceForPS`` second arg is
    gone."""

    def test_forward_signature_is_dual_towers(self):
        # Dual-towers rev 2 (2026-07-10 plan): PS and WS share one
        # symmetric signature -- forward(in_sub, cs_out=None).
        import Spaces
        sig = inspect.signature(Spaces.PartSpace.forward)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(
            params, ["in_sub", "cs_out"],
            f"PartSpace.forward must be (in_sub, cs_out=None); "
            f"got {params}")

    def test_reverse_signature_is_single_arg(self):
        import Spaces
        sig = inspect.signature(Spaces.PartSpace.reverse)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(
            len(params), 1,
            f"PartSpace.reverse must have exactly 1 positional "
            f"arg; got {params}")


class TestPSForwardReturnsValidSubspace(unittest.TestCase):
    """Behavioral smoke: ``forward(x_subspace)`` runs end-to-end and
    returns a valid SubSpace with a 3-D event. A true composition test
    (asserting ``out == pi(primary) + sigma(primary)`` up to forwardEnd
    transforms) lives in ``test_perceptual_loopback.py`` — this gate
    just verifies the single-arg call doesn't crash and produces the
    expected output shape."""

    def test_forward_returns_valid_subspace(self):
        """A standalone PS.forward call produces a non-None subspace
        with a non-empty event."""
        model = _make_plain_model()
        ps = model.perceptualSpace
        # Build a small numeric input subspace by way of the upstream
        # InputSpace forward so all the byte / shape plumbing is sane.
        inp_space = model.inputSpace
        loader = inp_space.data.data_loader(split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = inp_space.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub, _ = inp_space.forward(x_input)
                out = ps.forward(in_sub)
        self.assertIsNotNone(out,
                             "PartSpace.forward must return a "
                             "SubSpace.")
        ev = out.materialize()
        self.assertIsNotNone(ev,
                             "PS output event must materialize.")
        self.assertEqual(ev.dim(), 3,
                         f"PS output event must be 3-D [B, N, D]; "
                         f"got shape {tuple(ev.shape)}.")


class TestPSFoldShapes(unittest.TestCase):
    """The PS fold (``sigma`` post Pi/Sigma swap) is ``content ->
    content`` (unified fold-width law, 2026-07-16): sized at one vector's
    CONTENT width (``nDim``), the where/when band riding through
    application sites via ``fold_content_apply`` -- the same law as
    ``WholeSpace.pi``."""

    def test_fold_input_output_dims(self):
        model = _make_plain_model()
        ps = model.perceptualSpace
        content = int(ps.nDim)
        self.assertEqual(int(ps.sigma.nInput), content,
                         "sigma.nInput must equal PS content width (nDim).")
        self.assertEqual(int(ps.sigma.nOutput), content,
                         "sigma.nOutput must equal PS content width (nDim).")


class TestPSLegacyAttributesGone(unittest.TestCase):
    """The legacy ``pi_input`` / ``pi_concept`` ``ModuleList`` slots
    are no longer the PS-side interface; the single-layer ``pi`` /
    ``sigma`` superseded them."""

    def test_pi_input_module_list_gone(self):
        model = _make_plain_model()
        ps = model.perceptualSpace
        self.assertFalse(
            hasattr(ps, 'pi_input'),
            "PartSpace.pi_input ModuleList must be retired by "
            "the Stage 1.A single-layer refactor.")

    def test_pi_concept_module_list_gone(self):
        model = _make_plain_model()
        ps = model.perceptualSpace
        self.assertFalse(
            hasattr(ps, 'pi_concept'),
            "PartSpace.pi_concept ModuleList must be retired by "
            "the Stage 1.A single-layer refactor.")


if __name__ == "__main__":
    unittest.main()
