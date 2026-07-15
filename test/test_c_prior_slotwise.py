"""Regression gate for the PRODUCTION ``_c_prior`` slot-wise staging block.

Task 8 (plan §9) added a ``[depth, D]`` slot-wise staging path to
``ConceptualSpace._c_prior``. The live block lives in
``ConceptualSpace.forward`` (bin/Spaces.py, the ``if self._c_prior is not
None:`` block). When ``cs._c_prior_slotwise`` is True and ``cs._c_prior``
is a ``[depth, D]`` tensor whose width matches the materialised event, the
block stages one prior row per STM SLOT across the FIRST ``depth`` slots of
``event_for_carrier`` (broadcast over the batch axis), then clears BOTH
``_c_prior`` and ``_c_prior_slotwise``. When the flag is False, a
``[D]`` / ``[1, D]`` / ``[B, D]`` prior takes the legacy path: a single
vector broadcast across ALL slots.

Until now that production block was exercised by ZERO tests -- only a
hand-copied "mirror" (``_apply_prior``) in
``test/test_inter_sentence_prediction_shape.py`` re-implemented it, so a
future edit to the real block would not be caught. This file closes the
gap: it builds a REAL ``ConceptualSpace`` (the same cheap-boot harness as
``test/test_cs_stm_bookkeeping.py``), stages the prior on the real
attributes, and drives the REAL ``cs.forward`` so the production code in
bin/Spaces.py runs. It pins BOTH the slot-wise and the legacy broadcast
behaviors so a regression in either is caught.

The staging block is ADDITIVE onto the materialised event, so each test
feeds a ZERO base event of shape ``[B, N, D]`` (via ``ps_sub.set_event``,
the same override pattern the bookkeeping tests use). With a zero base the
read-back event equals the staged prior exactly, which makes the slot
occupancy assertions direct.
"""

import os
import sys
import unittest
import warnings

import pytest
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
    """Build a working model from MM_xor_loopback.xml + xor data -- the
    same cheap-boot pattern used by ``test_cs_stm_bookkeeping.py`` /
    ``test_ps_single_arg_refactor.py`` / ``test_pi_sigma_ownership.py``."""
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, _ = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    model.eval()
    return model


class _CPriorBase(unittest.TestCase):
    """Shared harness: build a real CS, a real PS subspace, and a helper
    that drives the REAL ``cs.forward`` with a ZERO base event of a chosen
    ``[B, N, D]`` shape after staging a prior. Returns the read-back event
    (``out.materialize()``) plus the post-call ``_c_prior`` /
    ``_c_prior_slotwise`` state so each test can assert on both."""

    def setUp(self):
        self.model = _make_plain_model()
        self.cs = self.model.conceptualSpace
        ps = self.model.perceptualSpace
        loader = self.model.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = self.model.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub, _ = self.model.inputSpace.forward(x_input)
                self.ps_sub = ps.forward(in_sub)
                ps_ev = self.ps_sub.materialize()
        # The materialised PS event width is the carrier D the production
        # block matches the prior against; reuse it verbatim so the
        # slot-wise width guard (prior.shape[-1] == event.shape[-1]) fires
        # against the REAL carrier dimension rather than an invented one.
        self.B = int(ps_ev.shape[0])
        self.D = int(ps_ev.shape[-1])
        self.dtype = ps_ev.dtype
        self.device = ps_ev.device

    def _drive_forward_zero_base(self, N, prior, slotwise):
        """Stage ``prior`` (+ ``slotwise`` flag) on the REAL CS, feed a
        zero ``[B, N, D]`` event through the REAL ``cs.forward``, and
        return ``(event_out, c_prior_after, slotwise_after)``.

        ``N > 1`` keeps the parallel whole-slab path so ``event_for_carrier
        == folded`` is the full ``[B, N, D]`` slot stack the production
        block stages into. A zero base means the read-back equals exactly
        the staged prior."""
        base = torch.zeros(
            self.B, N, self.D, dtype=self.dtype, device=self.device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                self.ps_sub.set_event(base)
                # Stage the prior on the REAL attributes the production
                # block reads -- this is the live injection surface that
                # BasicModel.generate_sentence / InterSentenceLayer.cast
                # populate. The block consumes and clears them.
                self.cs._c_prior = prior
                self.cs._c_prior_slotwise = slotwise
                out = self.cs.forward(self.ps_sub)
                event_out = out.materialize()
        return event_out, self.cs._c_prior, self.cs._c_prior_slotwise


class TestCPriorSlotwiseProductionBlock(_CPriorBase):
    """Slot-wise path: a ``[depth, D]`` prior stages one row per slot
    across the first ``depth`` slots of the real materialised event."""

    def test_slotwise_stages_first_depth_slots_and_clears(self):
        # depth=3 distinct per-slot rows; N=5 slots so slots 3,4 stay zero.
        N, depth = 5, 3
        payload_hat = torch.stack([
            torch.full((self.D,), float(r + 1),
                       dtype=self.dtype, device=self.device)
            for r in range(depth)
        ])  # row r is all (r+1): distinct per slot.
        event_out, c_prior_after, slotwise_after = \
            self._drive_forward_zero_base(N, payload_hat, slotwise=True)

        self.assertEqual(
            tuple(event_out.shape), (self.B, N, self.D),
            "slot-wise staging must preserve the [B, N, D] carrier shape")
        # The distinct per-slot values land in the FIRST ``depth`` slots,
        # broadcast over the batch axis; slots >= depth are untouched (0).
        for b in range(self.B):
            for r in range(depth):
                self.assertTrue(
                    torch.allclose(
                        event_out[b, r],
                        payload_hat[r]),
                    f"slot {r} (batch {b}) must hold prior row {r} "
                    f"(value {r + 1.0}); got {event_out[b, r].tolist()}")
            for r in range(depth, N):
                self.assertTrue(
                    torch.all(event_out[b, r] == 0.0),
                    f"slot {r} (batch {b}) is beyond depth={depth} and "
                    f"must stay zero (untouched by the slot-wise path)")
        # Distinct-per-slot sanity: the production block must NOT collapse
        # all slots to one value (that would be the legacy broadcast bug).
        self.assertFalse(
            torch.allclose(event_out[0, 0], event_out[0, 1]),
            "slot 0 and slot 1 must differ -- the slot-wise path stages "
            "DISTINCT per-slot rows, not one broadcast vector")
        # BOTH staging attributes must be cleared after the block runs.
        self.assertIsNone(
            c_prior_after,
            "the slot-wise block must clear cs._c_prior after staging")
        self.assertFalse(
            slotwise_after,
            "the slot-wise block must clear cs._c_prior_slotwise after "
            "staging")

    def test_slotwise_depth_clamped_to_available_slots(self):
        # depth=3 prior rows but only N=2 slots: stage the first 2, drop
        # the 3rd (clamp depth = min(prior.shape[0], N)).
        N, depth = 2, 3
        payload_hat = torch.stack([
            torch.full((self.D,), float(r + 1),
                       dtype=self.dtype, device=self.device)
            for r in range(depth)
        ])
        event_out, c_prior_after, slotwise_after = \
            self._drive_forward_zero_base(N, payload_hat, slotwise=True)
        self.assertEqual(tuple(event_out.shape), (self.B, N, self.D))
        for b in range(self.B):
            self.assertTrue(torch.allclose(event_out[b, 0], payload_hat[0]))
            self.assertTrue(torch.allclose(event_out[b, 1], payload_hat[1]))
        self.assertIsNone(c_prior_after)
        self.assertFalse(slotwise_after)


class TestCPriorLegacyBroadcastProductionBlock(_CPriorBase):
    """Legacy path (flag False): a ``[D]`` / ``[1, D]`` prior broadcasts
    the SAME vector across ALL slots -- pinned here so the test catches a
    regression in EITHER path, not just the slot-wise one."""

    def test_legacy_1d_prior_broadcasts_all_slots_and_clears(self):
        N = 4
        prior = torch.full(
            (self.D,), 2.0, dtype=self.dtype, device=self.device)  # [D]
        event_out, c_prior_after, slotwise_after = \
            self._drive_forward_zero_base(N, prior, slotwise=False)
        self.assertEqual(tuple(event_out.shape), (self.B, N, self.D))
        # EVERY slot gets the same broadcast vector (== prior on a 0 base).
        for b in range(self.B):
            for n in range(N):
                self.assertTrue(
                    torch.all(event_out[b, n] == 2.0),
                    f"legacy [D] prior must broadcast to ALL slots "
                    f"(batch {b}, slot {n}); got {event_out[b, n].tolist()}")
        self.assertIsNone(
            c_prior_after, "legacy block must clear cs._c_prior")
        self.assertFalse(
            slotwise_after,
            "legacy block must leave cs._c_prior_slotwise False")

    def test_legacy_2d_single_row_prior_broadcasts_all_slots(self):
        N = 3
        prior = torch.full(
            (1, self.D), 5.0, dtype=self.dtype, device=self.device)  # [1, D]
        event_out, c_prior_after, slotwise_after = \
            self._drive_forward_zero_base(N, prior, slotwise=False)
        self.assertEqual(tuple(event_out.shape), (self.B, N, self.D))
        for b in range(self.B):
            for n in range(N):
                self.assertTrue(
                    torch.all(event_out[b, n] == 5.0),
                    f"legacy [1, D] prior must broadcast to ALL slots "
                    f"(batch {b}, slot {n})")
        self.assertIsNone(c_prior_after)
        self.assertFalse(slotwise_after)


if __name__ == "__main__":
    unittest.main()
