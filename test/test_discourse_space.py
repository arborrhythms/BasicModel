"""Tests for DiscourseSpace — inter-sentence [S, W] snapshot substrate.

DiscourseSpace is a service space that sits above WordSpace. It owns a
ring buffer of per-sentence ``[S | W]`` concatenations, a linear
predictor over the last ``context_window`` snapshots, and a resolver
that decodes a predicted tensor back to a concrete sentence via
``OutputSpace.reconstruct_buffer``.

Covers:
  1. Unit-level DiscourseSpace behavior (no model) — snapshot round
     trip, padding/truncation, predict_next shape, loss correctness,
     reset, split, history overflow.
  2. BasicModel integration — that building a MentalModel wires
     DiscourseSpace in, that forward() populates the pending snapshot
     attributes, and that runBatch-equivalent flow pushes snapshots
     into history.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import gc
import unittest
import warnings
import torch
import matplotlib
matplotlib.use('Agg')

from BasicModel import MentalModel, TheData, TheDevice
from Model import DiscourseSpace
from util import init_config, TheXMLConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    from Space import TheGrammar
    TheGrammar._configured = False


def _release_allocator_cache():
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _DiscourseTestBase(unittest.TestCase):
    """Shared tearDown mirroring test_grammar_derivation's pattern so
    MentalModel instances don't accumulate across tests and trip the
    MPS 30 GiB limit.
    """

    _SLOTS = ('model', 'discourse', 'cfg')

    def tearDown(self):
        for slot in self._SLOTS:
            if hasattr(self, slot):
                try:
                    delattr(self, slot)
                except AttributeError:
                    pass
        _release_allocator_cache()


class TestDiscourseSpaceUnit(_DiscourseTestBase):
    """Unit tests for DiscourseSpace in isolation (no model).

    These exercise the class's own contracts: shape fitting, ring
    buffer discipline, predict_next / loss arithmetic, split
    inverse-of-assemble, reset clearing. No BasicModel required.
    """

    N_SYMBOLS = 4
    MAX_DEPTH = 6
    N_DIM = 8
    HISTORY_LEN = 3
    CTX = 2

    def _make(self):
        return DiscourseSpace(
            n_symbols=self.N_SYMBOLS,
            max_depth=self.MAX_DEPTH,
            n_dim=self.N_DIM,
            history_len=self.HISTORY_LEN,
            context_window=self.CTX,
        )

    def test_snapshot_roundtrip(self):
        """snapshot() → latest_snapshot() returns the assembled row."""
        d = self._make()
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        latest = d.latest_snapshot()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.shape,
                          (self.N_SYMBOLS + self.MAX_DEPTH, self.N_DIM))
        # The S half of the snapshot should match the input S.
        s_half, w_half = d.split(latest)
        self.assertTrue(torch.allclose(s_half, s))
        self.assertTrue(torch.allclose(w_half, w))

    def test_empty_history(self):
        """Fresh DiscourseSpace: len=0, predict returns None."""
        d = self._make()
        self.assertEqual(len(d), 0)
        self.assertIsNone(d.latest_snapshot())
        self.assertIsNone(d.predict_next())

    def test_ring_overflow(self):
        """After history_len+1 snapshots, the oldest is evicted."""
        d = self._make()
        # Tag each snapshot with a distinct scalar so we can track which
        # one ended up where.
        tags = []
        for i in range(self.HISTORY_LEN + 2):
            s = torch.full((self.N_SYMBOLS, self.N_DIM), float(i))
            w = torch.zeros(self.MAX_DEPTH, self.N_DIM)
            d.snapshot(s, w)
            tags.append(float(i))
        self.assertEqual(len(d), self.HISTORY_LEN)
        # Latest should be the last tag we pushed.
        latest = d.latest_snapshot()
        s_half, _ = d.split(latest)
        self.assertTrue(torch.allclose(s_half, torch.full_like(s_half, tags[-1])))
        # Oldest should be tags[-HISTORY_LEN] (the first survivor), not tags[0].
        oldest = d._snapshots[0]
        s_oldest, _ = d.split(oldest)
        expected = tags[-self.HISTORY_LEN]
        self.assertTrue(torch.allclose(s_oldest, torch.full_like(s_oldest, expected)))

    def test_predict_next_shape(self):
        """With history, predict_next returns [n_sentence, n_dim]."""
        d = self._make()
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        predicted = d.predict_next()
        self.assertIsNotNone(predicted)
        self.assertEqual(predicted.shape,
                          (self.N_SYMBOLS + self.MAX_DEPTH, self.N_DIM))

    def test_predict_next_short_history(self):
        """context_window > len(history): front-padded with zeros."""
        d = self._make()
        # Only one snapshot but context_window = 2
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.zeros(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        # Build context manually and compare sizes
        ctx = d._build_context()
        self.assertEqual(ctx.shape,
                          (self.CTX, d.n_sentence, self.N_DIM))
        # First row should be the pad (all zeros since pad precedes)
        self.assertTrue(torch.allclose(ctx[0], torch.zeros_like(ctx[0])))

    def test_loss_matches_manual(self):
        """loss(predicted, s, w) == F.mse_loss(predicted, assembled)."""
        d = self._make()
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        predicted = d.predict_next()
        # Provide a different target to get a nonzero loss.
        target_s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        target_w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        loss = d.loss(predicted, target_s, target_w)
        manual = torch.nn.functional.mse_loss(
            predicted, d._assemble(target_s, target_w))
        self.assertTrue(torch.allclose(loss, manual))

    def test_fit_rows_truncate(self):
        """_fit_rows with too many rows truncates, too few pads."""
        d = self._make()
        # Over-wide truncates to n_symbols.
        x = torch.randn(self.N_SYMBOLS + 3, self.N_DIM)
        y = d._fit_rows(x, self.N_SYMBOLS)
        self.assertEqual(y.shape, (self.N_SYMBOLS, self.N_DIM))
        self.assertTrue(torch.allclose(y, x[:self.N_SYMBOLS]))
        # Under-wide pads with zeros.
        x = torch.randn(self.N_SYMBOLS - 2, self.N_DIM)
        y = d._fit_rows(x, self.N_SYMBOLS)
        self.assertEqual(y.shape, (self.N_SYMBOLS, self.N_DIM))
        self.assertTrue(torch.allclose(y[:self.N_SYMBOLS - 2], x))
        self.assertTrue(torch.allclose(
            y[self.N_SYMBOLS - 2:], torch.zeros(2, self.N_DIM)))

    def test_fit_rows_batch_pool(self):
        """3D input is mean-pooled along the batch axis."""
        d = self._make()
        x = torch.randn(5, self.N_SYMBOLS, self.N_DIM)
        y = d._fit_rows(x, self.N_SYMBOLS)
        self.assertEqual(y.shape, (self.N_SYMBOLS, self.N_DIM))
        self.assertTrue(torch.allclose(y, x.mean(dim=0)))

    def test_fit_dim_pad_and_truncate(self):
        """_fit_dim handles column mismatches."""
        d = self._make()
        wider = torch.randn(self.N_SYMBOLS, self.N_DIM + 4)
        y = d._fit_dim(wider)
        self.assertEqual(y.shape[-1], self.N_DIM)
        narrower = torch.randn(self.N_SYMBOLS, self.N_DIM - 3)
        y = d._fit_dim(narrower)
        self.assertEqual(y.shape[-1], self.N_DIM)
        # Padding is trailing zeros.
        self.assertTrue(torch.allclose(y[..., :self.N_DIM - 3], narrower))
        self.assertTrue(torch.allclose(
            y[..., self.N_DIM - 3:], torch.zeros(self.N_SYMBOLS, 3)))

    def test_split_inverse_of_assemble(self):
        """split(assemble(s, w)) == (s, w) for correctly-sized inputs."""
        d = self._make()
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        assembled = d._assemble(s, w)
        s2, w2 = d.split(assembled)
        self.assertTrue(torch.allclose(s2, s))
        self.assertTrue(torch.allclose(w2, w))

    def test_reset_clears_history(self):
        """reset() zeros the buffer and resets the count."""
        d = self._make()
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        self.assertEqual(len(d), 1)
        d.reset()
        self.assertEqual(len(d), 0)
        self.assertIsNone(d.latest_snapshot())
        self.assertIsNone(d.predict_next())

    def test_predictor_gradient_flows(self):
        """The predictor's loss actually produces gradients on its
        weights — confirms the predict → loss → backward path is wired.
        """
        d = self._make()
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.zeros(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        predicted = d.predict_next()
        target_s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        target_w = torch.zeros(self.MAX_DEPTH, self.N_DIM)
        loss = d.loss(predicted, target_s, target_w)
        loss.backward()
        # At least one predictor param should have a nonzero gradient.
        grads = [p.grad for p in d.predictor.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        any_nonzero = any((g.abs().sum() > 0).item() for g in grads)
        self.assertTrue(any_nonzero, "no gradient flowed through predictor")


class TestDiscourseSpaceIntegration(_DiscourseTestBase):
    """Integration tests against a real MentalModel.

    Verifies that create() wires DiscourseSpace, that forward()
    populates the pending snapshot attributes, and that the runBatch
    path pushes snapshots into history correctly.
    """

    def setUp(self):
        _reload_config()

    def _build_model(self):
        _reload_config()
        model, cfg = MentalModel.from_config(
            os.path.join(_DATA_DIR, 'MentalModel.xml'))
        return model, cfg

    def test_create_wires_discourse_space(self):
        """MentalModel.create() should build and attach WordSpace.discourse
        when the ARLM/ARUS/RARLM grammar path is active."""
        self.model, self.cfg = self._build_model()
        self.assertIsNotNone(self.model.wordSpace)
        self.assertIsNotNone(self.model.wordSpace.discourse)
        d = self.model.wordSpace.discourse
        # Sizing should match SymbolicSpace's declared output shape.
        self.assertEqual(d.n_symbols,
                          int(self.model.symbolicSpace.outputShape[0]))
        self.assertEqual(d.n_dim,
                          self.model.symbolicSpace.subspace.muxedSize)

    def test_forward_populates_snapshot_attributes(self):
        """After forward(), the pending snapshot attributes should be
        set so runBatch can pick them up.
        """
        self.model, self.cfg = self._build_model()
        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:1])
            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                try:
                    self.model.forward(x)
                except ValueError:
                    self.skipTest("Untrained model range violation")

        # After forward: the pending snapshot pair should be populated
        # (unless wordSpace.discourse is None, which shouldn't happen
        # in this config).
        self.assertIsNotNone(self.model.wordSpace.discourse)
        self.assertIsNotNone(getattr(self.model, '_current_discourse_s', None))
        self.assertIsNotNone(getattr(self.model, '_current_discourse_w', None))

    def test_discourse_history_grows(self):
        """Running snapshot() directly on the model's wordSpace.discourse
        increments the count — sanity check that the connection is
        live."""
        self.model, self.cfg = self._build_model()
        d = self.model.wordSpace.discourse
        self.assertEqual(len(d), 0)
        # Build correctly-sized dummy tensors and push.
        s = torch.randn(d.n_symbols, d.n_dim, device=TheDevice.get())
        w = torch.zeros(d.max_depth, d.n_dim, device=TheDevice.get())
        d.snapshot(s, w)
        self.assertEqual(len(d), 1)
        d.reset()
        self.assertEqual(len(d), 0)

    def test_epoch_reset_clears_discourse(self):
        """runEpoch's pre-loop reset should clear the discourse
        history. We verify by pushing a snapshot, calling reset on
        the model's wordSpace.discourse directly (which is what
        runEpoch does), and confirming the count goes back to zero."""
        self.model, self.cfg = self._build_model()
        d = self.model.wordSpace.discourse
        s = torch.randn(d.n_symbols, d.n_dim, device=TheDevice.get())
        w = torch.zeros(d.max_depth, d.n_dim, device=TheDevice.get())
        d.snapshot(s, w)
        self.assertEqual(len(d), 1)
        # Mirror runEpoch's behavior
        ws = getattr(self.model, 'wordSpace', None)
        if ws is not None and getattr(ws, 'discourse', None) is not None:
            ws.discourse.reset()
        self.assertEqual(len(d), 0)


if __name__ == '__main__':
    unittest.main()
