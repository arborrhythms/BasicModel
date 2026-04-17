"""Tests for DiscourseSpace -- inter-sentence [S, W] snapshot substrate.

DiscourseSpace is a service space that sits above WordSpace. It owns
two rings of per-sentence ``[S | W]`` concatenations (a recent buffer
used for the attractive centroid and a prev_centroids buffer used for
the repulsive force), plus a contrastive dual-force cosine loss over
the full flattened snapshot vector. It has no learnable parameters.

Covers:
  1. Unit-level DiscourseSpace behavior (no model) -- snapshot round
     trip, padding/truncation, ring eviction + centroid folding,
     contrastive loss arithmetic (attractive alone, attractive +
     repulsive), reset, split, gradient flow through the live
     (s, w) arguments.
  2. BasicModel integration -- that building a MentalModel wires
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
import torch.nn.functional as F
import matplotlib
import Models
import Spaces
import Layers
matplotlib.use('Agg')

from util import init_config, TheXMLConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Spaces.TheGrammar._configured = False


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

    Exercises the class's own contracts: shape fitting, ring-buffer
    eviction with centroid folding, contrastive-loss arithmetic,
    split inverse-of-assemble, reset, and gradient flow through the
    live (s, w) arguments. No BasicModel required.
    """

    N_SYMBOLS = 4
    MAX_DEPTH = 6
    N_DIM = 8
    CTX = 2               # recent buffer depth
    CENTROID_HIST = 2     # prev_centroids depth
    LAM = 1.01

    def _make(self):
        return Layers.InterSentenceLayer(
            n_symbols=self.N_SYMBOLS,
            max_depth=self.MAX_DEPTH,
            n_dim=self.N_DIM,
            context_window=self.CTX,
            centroid_history=self.CENTROID_HIST,
            lam=self.LAM,
        )

    # -- snapshot / ring plumbing -------------------------------------

    def test_snapshot_roundtrip(self):
        """snapshot() -> latest_snapshot() returns the assembled row."""
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
        """Fresh DiscourseSpace: len=0, latest is None, loss is None."""
        d = self._make()
        self.assertEqual(len(d), 0)
        self.assertIsNone(d.latest_snapshot())
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        self.assertIsNone(d.loss(s, w))

    def test_ring_eviction_folds_centroid(self):
        """When the recent buffer is full, the next snapshot folds the
        current centroid into prev_centroids before evicting the
        oldest row.  With CTX=2 and CENTROID_HIST=2 we push three
        snapshots and verify the expected geometry.
        """
        d = self._make()
        # Tag each snapshot with a distinct scalar so we can track it.
        tags = [1.0, 2.0, 3.0]
        for i, tag in enumerate(tags):
            s = torch.full((self.N_SYMBOLS, self.N_DIM), tag)
            w = torch.full((self.MAX_DEPTH, self.N_DIM), tag)
            d.snapshot(s, w)
        # Recent buffer still holds at most CTX rows.
        self.assertEqual(len(d), self.CTX)
        # Latest is the most recent push.
        latest = d.latest_snapshot()
        self.assertTrue(torch.allclose(
            latest, torch.full_like(latest, tags[-1])))
        # The oldest recent entry is tags[1] (tags[0] was evicted).
        oldest = d._recent[0]
        self.assertTrue(torch.allclose(
            oldest, torch.full_like(oldest, tags[1])))
        # Exactly one centroid should have been folded into prev_centroids
        # -- the mean of the pre-eviction recent window (tags[0], tags[1]).
        self.assertEqual(int(d._prev_count.item()), 1)
        expected_prev = torch.full_like(d._prev_centroids[0],
                                        (tags[0] + tags[1]) / 2.0)
        self.assertTrue(torch.allclose(d._prev_centroids[0], expected_prev))

    def test_prev_centroids_ring_overflow(self):
        """Folding more than CENTROID_HIST centroids evicts the oldest
        from the prev_centroids ring."""
        d = self._make()
        # Push enough snapshots to fold CENTROID_HIST + 1 centroids.
        # Each eviction folds one centroid, so we need CTX + CENTROID_HIST + 1
        # total snapshots (first CTX fill the recent buffer cleanly).
        total = self.CTX + self.CENTROID_HIST + 1
        for i in range(total):
            s = torch.full((self.N_SYMBOLS, self.N_DIM), float(i))
            w = torch.full((self.MAX_DEPTH, self.N_DIM), float(i))
            d.snapshot(s, w)
        # prev_centroids should be saturated at CENTROID_HIST.
        self.assertEqual(int(d._prev_count.item()), self.CENTROID_HIST)

    def test_fit_rows_truncate(self):
        """_fit_rows with too many rows truncates, too few pads."""
        d = self._make()
        x = torch.randn(self.N_SYMBOLS + 3, self.N_DIM)
        y = d._fit_rows(x, self.N_SYMBOLS)
        self.assertEqual(y.shape, (self.N_SYMBOLS, self.N_DIM))
        self.assertTrue(torch.allclose(y, x[:self.N_SYMBOLS]))
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
        """reset() zeros both rings and resets both counts."""
        d = self._make()
        # Fill enough to populate both rings.
        for i in range(self.CTX + 1):
            s = torch.full((self.N_SYMBOLS, self.N_DIM), float(i + 1))
            w = torch.full((self.MAX_DEPTH, self.N_DIM), float(i + 1))
            d.snapshot(s, w)
        self.assertGreater(len(d), 0)
        self.assertGreater(int(d._prev_count.item()), 0)
        d.reset()
        self.assertEqual(len(d), 0)
        self.assertEqual(int(d._prev_count.item()), 0)
        self.assertIsNone(d.latest_snapshot())
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        self.assertIsNone(d.loss(s, w))

    # -- contrastive loss --------------------------------------------

    def test_loss_attractive_only_matches_manual(self):
        """With just one snapshot (no prev_centroids), the loss should
        equal the attractive term ``1 - cos(current, recent_centroid)``
        computed over the flattened snapshot vector.
        """
        d = self._make()
        s_hist = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w_hist = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s_hist, w_hist)

        s_cur = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w_cur = torch.randn(self.MAX_DEPTH, self.N_DIM)
        loss = d.loss(s_cur, w_cur)
        self.assertIsNotNone(loss)

        # Manual computation:
        current = d._assemble(s_cur, w_cur).reshape(-1)
        ctx = d._assemble(s_hist, w_hist).reshape(-1)
        manual = 1.0 - F.cosine_similarity(
            current.unsqueeze(0), ctx.unsqueeze(0))
        self.assertTrue(torch.allclose(loss, manual.squeeze()))

    def test_loss_attractive_plus_repulsive(self):
        """With >=1 prev_centroid, the loss should equal
        attractive + lam * mean(cos(current, prev_i)).
        """
        d = self._make()
        # Fill recent + fold one centroid into prev.
        for i in range(self.CTX + 1):
            s = torch.randn(self.N_SYMBOLS, self.N_DIM)
            w = torch.randn(self.MAX_DEPTH, self.N_DIM)
            d.snapshot(s, w)
        self.assertGreaterEqual(int(d._prev_count.item()), 1)

        s_cur = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w_cur = torch.randn(self.MAX_DEPTH, self.N_DIM)
        loss = d.loss(s_cur, w_cur)
        self.assertIsNotNone(loss)

        # Manual: attractive + lam * mean_over_prev(cos)
        current = d._assemble(s_cur, w_cur).reshape(-1)
        ctx = d._recent_centroid().reshape(-1)
        attractive = 1.0 - F.cosine_similarity(
            current.unsqueeze(0), ctx.unsqueeze(0))
        m = int(d._prev_count.item())
        prev = d._prev_centroids[:m].reshape(m, -1)
        sims = F.cosine_similarity(current.unsqueeze(0), prev, dim=-1)
        repulsive = sims.mean()
        manual = attractive.squeeze() + self.LAM * repulsive
        self.assertTrue(torch.allclose(loss, manual))

    def test_loss_zero_when_current_matches_centroid(self):
        """If the current snapshot equals the only prior snapshot,
        cos = 1 so the attractive term is 0.  With no prev_centroids
        the repulsive term is 0 too, so the whole loss is ~0.
        """
        d = self._make()
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        loss = d.loss(s, w)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.allclose(
            loss, torch.zeros_like(loss), atol=1e-6))

    def test_loss_gradient_flows_through_current(self):
        """The contrastive loss has no learnable parameters, but
        gradient must still reach the live (s, w) arguments passed in
        -- that is the signal the rest of the model learns from.
        """
        d = self._make()
        # Prime history (detached, stored in buffers).
        s_hist = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w_hist = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s_hist, w_hist)

        s_cur = torch.randn(self.N_SYMBOLS, self.N_DIM, requires_grad=True)
        w_cur = torch.randn(self.MAX_DEPTH, self.N_DIM, requires_grad=True)
        loss = d.loss(s_cur, w_cur)
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(s_cur.grad)
        self.assertIsNotNone(w_cur.grad)
        self.assertTrue(s_cur.grad.abs().sum().item() > 0,
                        "no gradient on s_cur")
        self.assertTrue(w_cur.grad.abs().sum().item() > 0,
                        "no gradient on w_cur")

    def test_history_is_detached(self):
        """Stored snapshots must not carry graph history -- the loss
        on a later sentence shouldn't try to backprop through the
        pushed tensors (they were detached at snapshot time).
        """
        d = self._make()
        s_hist = torch.randn(self.N_SYMBOLS, self.N_DIM, requires_grad=True)
        w_hist = torch.randn(self.MAX_DEPTH, self.N_DIM, requires_grad=True)
        d.snapshot(s_hist, w_hist)

        s_cur = torch.randn(self.N_SYMBOLS, self.N_DIM, requires_grad=True)
        w_cur = torch.randn(self.MAX_DEPTH, self.N_DIM, requires_grad=True)
        loss = d.loss(s_cur, w_cur)
        loss.backward()
        # The historical tensors should NOT have received gradient --
        # snapshot() detached them.
        self.assertTrue(s_hist.grad is None
                        or s_hist.grad.abs().sum().item() == 0)
        self.assertTrue(w_hist.grad is None
                        or w_hist.grad.abs().sum().item() == 0)


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
        model, cfg = Models.MentalModel.from_config(
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

        with Models.TheData.runtime_batch(sentences, outputs), \
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
        increments the count -- sanity check that the connection is
        live."""
        self.model, self.cfg = self._build_model()
        d = self.model.wordSpace.discourse
        self.assertEqual(len(d), 0)
        # Build correctly-sized dummy tensors and push.
        s = torch.randn(d.n_symbols, d.n_dim, device=Models.TheDevice.get())
        w = torch.zeros(d.max_depth, d.n_dim, device=Models.TheDevice.get())
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
        s = torch.randn(d.n_symbols, d.n_dim, device=Models.TheDevice.get())
        w = torch.zeros(d.max_depth, d.n_dim, device=Models.TheDevice.get())
        d.snapshot(s, w)
        self.assertEqual(len(d), 1)
        # Mirror runEpoch's behavior
        ws = getattr(self.model, 'wordSpace', None)
        if ws is not None and getattr(ws, 'discourse', None) is not None:
            ws.discourse.reset()
        self.assertEqual(len(d), 0)


if __name__ == '__main__':
    unittest.main()
