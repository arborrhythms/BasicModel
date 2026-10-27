"""Stage 1.C substrate refactor: ConceptualSpace.forward is STM bookkeeping.

Post-Stage-1.C contract (doc/plans/2026-05-26-two-loop-pi-sigma-substrate.md):

  * ``ConceptualSpace`` retires the atomic-fold ``sigma_percept``
    (and ``sigma_percept_1`` / ``sigma_percept_2`` variants). The
    SigmaLayer that previously did the percept→concept lift is gone;
    the C space_role no longer holds a parameterised fold operator.

  * ``ConceptualSpace.forward(PS_subspace, SS_subspace=None)`` performs
    **STM bookkeeping only**: under the newest-at-slot-0 convention it
    shifts the per-batch idea stack one slot RIGHT
    (``STM[1..7] = STM[0..6]``) and writes the incoming idea (the
    materialised PS / SS combination) to ``STM[0]`` (the newest slot),
    dropping the oldest (``STM[7]`` at Miller cap). No atomic fold layer
    is invoked; the signal-router-based grammar dispatch that consumes STM
    is Stage 3, out of scope here.

  * The existing ``ConceptualSpace.stm`` (a ``ShortTermMemory``
    instance) is preserved — its ``push`` / ``peek`` / ``snapshot`` /
    ``clear`` surface is the bookkeeping target.

This file is the targeted TDD gate for Stage 1.C. It uses the same
``MM_xor_loopback.xml`` config as the sibling Stage-1.A / Stage-1.B
test files (cheap PS/CS/SS boot, isolates the bookkeeping behavior).
"""

import os
import sys
import unittest

import pytest
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
from Spaces import SubSpaceView
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, 'data')
_CONFIG = os.path.join(_DATA_DIR, "MM_xor_loopback.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_plain_model():
    """Build a working model from MM_xor_loopback.xml + xor data — the
    same cheap-boot pattern used by ``test_ps_single_arg_refactor.py``
    and ``test_pi_sigma_ownership.py``."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model




class TestSTMInstancePreserved(unittest.TestCase):
    """``ConceptualSpace.stm`` (the ShortTermMemory instance) is
    preserved; Stage 1.C uses it as the bookkeeping target."""

    def test_stm_attribute_exists(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertTrue(
            hasattr(cs, 'stm'),
            "ConceptualSpace.stm (ShortTermMemory) must remain — "
            "Stage 1.C is bookkeeping into it.")

    def test_stm_has_push_peek_snapshot_clear(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertTrue(callable(cs.stm.push))
        self.assertTrue(callable(cs.stm.peek))
        self.assertTrue(callable(cs.stm.snapshot))
        self.assertTrue(callable(cs.stm.clear))


class TestCSForwardMutatesSTM(unittest.TestCase):
    """``CS.forward(PS_sub, SS_sub)`` returns a SubSpace and mutates
    ``cs.stm`` (depth increases or the top slot is written)."""

    def test_forward_pushes_to_stm(self):
        """A single ``cs.forward(PS_sub)`` call increases STM depth
        on the active row."""
        model = _make_plain_model()
        cs = model.conceptualSpace
        ps = model.perceptualSpace
        # Build a real PS subspace by running upstream IS+PS.
        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = model.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub, _ = model.inputSpace.forward(x_input)
                ps_sub = ps.forward(in_sub)
                # Ensure STM batch row 0 is empty before the call.
                cs.stm.ensure_batch(max(1, int(ps_sub.materialize().shape[0])))
                cs.stm.clear()
                depth_before = cs.stm.size(0)
                out = cs.forward(ps_sub)
                depth_after = cs.stm.size(0)
        self.assertIsNotNone(out, "cs.forward must return a SubSpace.")
        self.assertGreater(
            depth_after, depth_before,
            f"cs.forward must push to STM (depth {depth_before} -> "
            f"{depth_after}).")


class TestSTMShiftAtCapacity(unittest.TestCase):
    """The bounded chunk field protects every live stack operand."""

    def test_all_stack_chunks_refuse_shift_and_preserve_every_operand(self):
        model = _make_plain_model()
        try:
            stm = model.conceptualSpace.stm
            stm.ensure_batch(1)
            stm.clear()
            for k in range(stm.capacity):
                stm.push(0, torch.full((stm.concept_dim,), float(k+1)))
            before = tuple(value.clone() for value in stm.state)
            self.assertEqual(stm.size(0), stm.capacity)
            with self.assertRaisesRegex(RuntimeError, 'all chunks are on the stack'):
                stm.push(0, torch.full((stm.concept_dim,), float(stm.capacity+1)))
            for original, retained in zip(before, stm.state):
                torch.testing.assert_close(retained, original, atol=0, rtol=0)
            for position in range(stm.capacity):
                torch.testing.assert_close(stm.peek(0, position),
                    torch.full((stm.concept_dim,), float(stm.capacity-position)), atol=0, rtol=0)
        finally:
            model.End()
            model.symbolSpace.soft_reset()


class TestCSForwardArity(unittest.TestCase):
    """Stage-1.A test_perceptual_loopback.py already asserts CS.forward
    has exactly 2 positional args (PS_subspace, SS_subspace=None).
    Stage 1.C preserves that arity — only the body changes. Mirror the
    same arity check here so this file is self-contained as the Stage-1.C
    regression gate."""

    def test_conceptual_forward_arity(self):
        import inspect
        import Spaces
        sig = inspect.signature(Spaces.ConceptualSpace.forward)
        params = [n for n in sig.parameters if n != "self"]
        self.assertEqual(
            len(params), 2,
            f"ConceptualSpace.forward must be "
            f"(PS_subspace, SS_subspace=None); got {params}")


if __name__ == "__main__":
    unittest.main()
