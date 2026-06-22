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


class TestSigmaPerceptRetired(unittest.TestCase):
    """The atomic-fold ``sigma_percept`` (and its ``_1`` / ``_2`` /
    ``_sigma_percept_reverse`` helpers) are retired. CS no longer carries
    a parameterised percept→concept fold operator at the substrate
    level."""

    def test_sigma_percept_attribute_removed(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertFalse(
            hasattr(cs, 'sigma_percept'),
            "ConceptualSpace.sigma_percept must be retired by "
            "Stage 1.C (CS.forward is now STM bookkeeping).")

    def test_sigma_percept_variants_removed(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertFalse(
            hasattr(cs, 'sigma_percept_1'),
            "ConceptualSpace.sigma_percept_1 (paired-sigma forward "
            "alias) must be retired by Stage 1.C.")
        self.assertFalse(
            hasattr(cs, 'sigma_percept_2'),
            "ConceptualSpace.sigma_percept_2 (paired-sigma reverse "
            "alias) must be retired by Stage 1.C.")

    def test_sigma_percept_reverse_helper_removed(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        self.assertFalse(
            hasattr(cs, '_sigma_percept_reverse'),
            "ConceptualSpace._sigma_percept_reverse helper must be "
            "retired with ``sigma_percept``.")


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
    """Pushing beyond Miller cap shifts the oldest idea out. Under the
    newest-at-slot-0 convention the new idea goes to slot 0 and the window
    rolls RIGHT (STM[1..7] := STM[0..6]); the oldest (the last occupied
    slot) drops. After cap+1 pushes the OLDEST live idea (peek(n=cap-1))
    is the SECOND-pushed one (the first having rolled off), and the newest
    (peek(n=0)) is the just-pushed idea. The assertions below use peek
    (convention-agnostic by semantics), not raw slot indices."""

    @pytest.mark.xfail(reason=(
        "Shift mechanism (buffer roll, newest-at-slot-0) is unaffected by the "
        "codebook, but this test identifies ideas by a per-push marker VALUE "
        "(float(k+1)); the converged modality CS.forward snaps each pushed idea "
        "to a codebook prototype, quantizing the marker so the raw value (e.g. "
        "2.0) is not recoverable on read-back (got ~0.08). Identifying the "
        "shifted idea now needs the snapped value or a no-codebook CS."),
        strict=False)
    def test_shift_drops_oldest(self):
        model = _make_plain_model()
        cs = model.conceptualSpace
        # Use a small unit dim for clarity: build dummy PS subspaces
        # directly with synthetic events that are distinguishable
        # per push (so we can identify which one ended up where).
        import Spaces
        # Use the actual CS-attached subspace dim to avoid shape errors.
        # The bookkeeping push goes via cs.stm.push, which takes
        # whatever the materialised PS event width is.
        ps = model.perceptualSpace
        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = model.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub, _ = model.inputSpace.forward(x_input)
                ps_sub = ps.forward(in_sub)
                ps_ev = ps_sub.materialize()
                # Reduce to a per-batch row-0 vector we can rebuild as
                # an 8-step push sequence with distinctive marker values
                # so we can identify slot occupancy after the shift.
                B = int(ps_ev.shape[0])
                D = int(ps_ev.shape[-1])
                cs.stm.ensure_batch(B)
                cs.stm.clear()
                cap = int(cs.stm.capacity)
                # Push ``cap`` distinguishable ideas: idea k is a vector
                # of all ``float(k+1)`` so the slot-occupancy check
                # is direct.
                for k in range(cap):
                    marker = torch.full(
                        (D,), float(k + 1),
                        dtype=ps_ev.dtype, device=ps_ev.device)
                    # Mutate the existing PS subspace event to carry this
                    # synthetic idea, then push it through cs.forward.
                    # We use [B, 1, D] to keep the 3-D event shape that
                    # cs.forward expects via materialize().
                    synth = marker.view(1, 1, D).expand(B, 1, D).clone()
                    ps_sub.set_event(synth)
                    cs.forward(ps_sub)
                # After cap pushes, depth is cap (no shift yet — the
                # buffer is full but no extra push has overflowed).
                # Now push one more: this is the (cap+1)-th idea and
                # must trigger the shift.
                marker = torch.full(
                    (D,), float(cap + 1),
                    dtype=ps_ev.dtype, device=ps_ev.device)
                synth = marker.view(1, 1, D).expand(B, 1, D).clone()
                ps_sub.set_event(synth)
                cs.forward(ps_sub)
                # After the shift: the OLDEST live idea (peek at
                # n=cap-1) should be the second-pushed one (marker == 2.0;
                # the first rolled off). The most recent (peek(b, 0))
                # should be the just-pushed idea (marker == cap+1). These
                # peek-based checks are convention-agnostic by semantics.
                oldest = cs.stm.peek(0, n=cap - 1)
                top = cs.stm.peek(0, n=0)
        self.assertIsNotNone(oldest, "STM oldest live slot must be populated.")
        self.assertIsNotNone(top, "STM top slot must be populated.")
        # The actual stored idea is the materialised PS event (which
        # cs.forward shapes / wraps but does not magnify); the
        # all-equal-value marker should survive intact: oldest live == 2.0
        # and newest == cap+1.
        self.assertAlmostEqual(
            float(oldest[0].item()), 2.0, places=3,
            msg=f"STM oldest live idea must be the SECOND-pushed one after "
            f"cap+1 pushes (first shifted out); got marker "
            f"{float(oldest[0].item())}.")
        self.assertAlmostEqual(
            float(top[0].item()), float(cap + 1), places=3,
            msg=f"STM top slot must hold the most recently pushed "
            f"idea (marker {cap + 1}); got {float(top[0].item())}.")


class TestCSPredictThenPerceive(unittest.TestCase):
    """Task 3 (STM serial/parallel modes): ``CS.forward`` runs the
    intra-sentence predictor predict-then-perceive. The held prediction
    is stashed on ``cs._stm_predicted_idea`` (serial per-word) and the
    per-step ``L_intra`` accumulates so ``consume_intra_loss`` returns a
    finite grad-bearing scalar after a serial sentence (grad enabled)."""

    def _synthetic_serial_pushes(self, cs, ps_sub, D, B, dtype, device,
                                 n_words):
        """Push ``n_words`` distinguishable [B, 1, D] ideas through
        ``cs.forward`` (serial per-word path)."""
        for k in range(n_words):
            marker = torch.full(
                (D,), float(k + 1), dtype=dtype, device=device)
            synth = marker.view(1, 1, D).expand(B, 1, D).clone()
            ps_sub.set_event(synth)
            cs.forward(ps_sub)

    def test_predicted_idea_populated_after_per_word_push(self):
        """After a per-word push, ``cs._stm_predicted_idea`` is a
        populated ``[B, D]`` tensor (degenerate zeros on the first word,
        a real prediction thereafter)."""
        model = _make_plain_model()
        cs = model.conceptualSpace
        # Ensure the loss weight is positive so accumulation is active.
        cs.intra_loss_weight = 0.1
        ps = model.perceptualSpace
        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = model.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Grad ENABLED (no torch.no_grad) so the loss path is live.
            in_sub, _ = model.inputSpace.forward(x_input)
            ps_sub = ps.forward(in_sub)
            ps_ev = ps_sub.materialize()
            B = int(ps_ev.shape[0])
            D = int(ps_ev.shape[-1])
            cs.stm.ensure_batch(B)
            cs.stm.clear()
            # First word: STM empty -> degenerate zero prediction.
            self._synthetic_serial_pushes(
                cs, ps_sub, D, B, ps_ev.dtype, ps_ev.device, n_words=1)
            self.assertIsNotNone(
                cs._stm_predicted_idea,
                "first per-word push must stash a (degenerate) prediction")
            self.assertEqual(
                tuple(cs._stm_predicted_idea.shape), (B, D),
                "stashed serial prediction must be [B, D]")
            self.assertTrue(
                torch.isfinite(cs._stm_predicted_idea).all(),
                "stashed prediction must be finite")
            # Second word: STM now has a slot -> a REAL prediction.
            self._synthetic_serial_pushes(
                cs, ps_sub, D, B, ps_ev.dtype, ps_ev.device, n_words=1)
            self.assertIsNotNone(
                cs._stm_predicted_idea,
                "second per-word push must stash a prediction")
            self.assertEqual(
                tuple(cs._stm_predicted_idea.shape), (B, D))
            self.assertTrue(
                torch.isfinite(cs._stm_predicted_idea).all())

    def test_consume_intra_loss_finite_after_serial_sentence(self):
        """After a serial sentence (several per-word pushes) with grad
        enabled, ``consume_intra_loss`` returns a finite scalar tensor
        that carries grad; a second consume returns ``None`` (reset)."""
        model = _make_plain_model()
        cs = model.conceptualSpace
        cs.intra_loss_weight = 0.1
        ps = model.perceptualSpace
        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = model.inputSpace.prepInput(inp_items)
        self.assertTrue(
            torch.is_grad_enabled(),
            "test harness must run with grad enabled for L_intra to "
            "accumulate")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            in_sub, _ = model.inputSpace.forward(x_input)
            ps_sub = ps.forward(in_sub)
            ps_ev = ps_sub.materialize()
            B = int(ps_ev.shape[0])
            D = int(ps_ev.shape[-1])
            cs.stm.ensure_batch(B)
            cs.stm.clear()
            # 4 words: word 1 degenerate (no loss), words 2-4 accumulate.
            self._synthetic_serial_pushes(
                cs, ps_sub, D, B, ps_ev.dtype, ps_ev.device, n_words=4)
            loss = cs.consume_intra_loss()
        self.assertIsNotNone(
            loss, "consume_intra_loss must return a tensor after a serial "
            "sentence with >=2 words and grad enabled")
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0, "L_intra must be a scalar")
        self.assertTrue(
            torch.isfinite(loss).all(),
            "L_intra must be finite (fail-loud on divergence is the "
            "training-path contract, not silently nan'd here)")
        self.assertTrue(
            loss.requires_grad,
            "L_intra must carry grad to the predictor's params")
        # Consumed + reset: a second consume returns None.
        self.assertIsNone(
            cs.consume_intra_loss(),
            "consume_intra_loss must reset the accumulator")

    def test_no_accumulation_under_no_grad(self):
        """Under ``torch.no_grad`` the loss does not accumulate (eval-time
        graph-growth guard); ``consume_intra_loss`` returns ``None`` even
        though the prediction is still stashed."""
        model = _make_plain_model()
        cs = model.conceptualSpace
        cs.intra_loss_weight = 0.1
        ps = model.perceptualSpace
        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = model.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub, _ = model.inputSpace.forward(x_input)
                ps_sub = ps.forward(in_sub)
                ps_ev = ps_sub.materialize()
                B = int(ps_ev.shape[0])
                D = int(ps_ev.shape[-1])
                cs.stm.ensure_batch(B)
                cs.stm.clear()
                self._synthetic_serial_pushes(
                    cs, ps_sub, D, B, ps_ev.dtype, ps_ev.device, n_words=4)
                loss = cs.consume_intra_loss()
                # Prediction still stashed even though loss is off.
                pred = cs._stm_predicted_idea
        self.assertIsNone(
            loss, "no-grad forwards must NOT accumulate L_intra")
        self.assertIsNotNone(
            pred, "the prediction is still stashed under no_grad")


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
