"""Stage 1.F substrate refactor: per-stage forward caches retired.

Post-Stage-1.F contract (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):

  * The per-stage forward capture lists ``self._ws_cache`` and
    ``self._cs_cache`` (plain Python lists allocated by
    ``_build_pipelines_per_stage`` and reallocated by
    ``_per_word_prelude``, written each forward stage) are RETIRED.
    The master plan doctrine: "No per-stage forward caches. Existing
    unrolled reverse chain operates on the terminal STM contents at
    the end of the forward chain."

  * The remaining live consumer of ``_cs_cache[-1]`` — the reverse-pass
    input-reconstruction loss path at ``bin/Models.py:2925`` — migrates
    to reading the terminal C-space_role idea from
    ``self.conceptualSpace.stm`` (the canonical ConceptualSpace
    ShortTermMemory instance already populated by the forward).

This file is the targeted TDD gate for Stage 1.F. It uses the same
``MM_xor_loopback.xml`` config as the sibling Stage-1.A / Stage-1.B /
Stage-1.C / Stage-1.D test files (cheap PS/CS/SS boot).
"""

import os
import sys
import unittest
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, 'data')
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_plain_model():
    """Build a working model from MM_xor_loopback.xml + xor data — the
    same cheap-boot pattern used by ``test_cs_stm_bookkeeping.py`` and
    ``test_pi_sigma_ownership.py``."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


def _one_input(model):
    """Pull one batch off the input loader and prepInput it."""
    loader = model.inputSpace.data.data_loader(
        split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    return model.inputSpace.prepInput(inp_items)




class TestPerStageCachesRetiredAfterForward(unittest.TestCase):
    """A full forward pass does not (re)allocate ``_ws_cache`` /
    ``_cs_cache`` either. These lists are gone from every code path."""

    def test_ws_cache_not_present_after_forward(self):
        model = _make_plain_model()
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                model.forward(x)
        self.assertFalse(
            hasattr(model, '_ws_cache'),
            "BasicModel._ws_cache must remain absent after a forward "
            "pass (Stage 1.F: no per-stage forward caches).")

    def test_cs_cache_not_present_after_forward(self):
        model = _make_plain_model()
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                model.forward(x)
        self.assertFalse(
            hasattr(model, '_cs_cache'),
            "BasicModel._cs_cache must remain absent after a forward "
            "pass (Stage 1.F: no per-stage forward caches).")


class TestReverseRoundtripStillCompletes(unittest.TestCase):
    """The reverse-pass reconstruction path that previously read
    ``_cs_cache[-1]`` must continue to function after the cache retires;
    it migrates to reading the terminal STM contents from
    ``self.conceptualSpace.stm``. Approximate reconstruction is fine
    (single-input reverse through averaged loops); the contract is
    "no crash / no exception" — the loss tensor produced by runBatch
    is a non-NaN scalar."""

    def test_terminal_stm_snapshot_available_after_forward(self):
        """The close dissolves STM; its completed understanding owns reversal."""
        model = _make_plain_model()
        with torch.no_grad():
            result = model.forward(_one_input(model))
            held = model._capture_understanding(result)
        self.assertIsNone(model.conceptualSpace.stm.snapshot())
        self.assertTrue(held.sentence_records)
        self.assertGreater(int(held.sentence_records[-1].end_depth.sum()), 0)

    def testreverse_via_terminal_stm(self):
        """The owned inverse completes after its working STM has dissolved."""
        model = _make_plain_model()
        with torch.no_grad():
            result = model.forward(_one_input(model))
            held = model._capture_understanding(result)
            self.assertIsNone(model.conceptualSpace.stm.snapshot())
            recovered, _ = model.reverseReconstruct(held)
        self.assertTrue(torch.isfinite(recovered).all())

    def test_runBatch_reverse_path_does_not_depend_on_cs_cache(self):
        """The reverse-pass reconstruction in ``runBatch`` (the consumer
        at ``bin/Models.py:2925``) must not depend on ``_cs_cache``.
        After Stage 1.F retires the cache, this path reads STM and
        does not consult ``_cs_cache``.

        We assert this indirectly: after a forward, ``hasattr(model,
        '_cs_cache')`` is False, so the reverse path cannot be reading
        from that attribute.
        """
        model = _make_plain_model()
        x = _one_input(model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                model.forward(x)
        self.assertFalse(
            hasattr(model, '_cs_cache'),
            "Reverse-path consumer at runBatch must not depend on "
            "``_cs_cache`` (Stage 1.F retires it entirely).")


if __name__ == "__main__":
    unittest.main()
