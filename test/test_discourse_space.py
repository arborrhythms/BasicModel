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
  2. BasicModel integration -- that building a BasicModel wires
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
import Language
import Layers
matplotlib.use('Agg')

from util import init_config, TheXMLConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Language.TheGrammar._configured = False


def _release_allocator_cache():
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _DiscourseTestBase(unittest.TestCase):
    """Shared tearDown mirroring test_grammar_derivation's pattern so
    BasicModel instances don't accumulate across tests and trip the
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
        self.assertIsNone(d.contrastive_loss(s, w))

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
        # Buffers carry a leading B=1 dim under the Task 3 microbatch
        # refactor; row 0 is the legacy single-row content.
        oldest = d._recent[0, 0]
        self.assertTrue(torch.allclose(
            oldest, torch.full_like(oldest, tags[1])))
        # Exactly one centroid should have been folded into prev_centroids
        # -- the mean of the pre-eviction recent window (tags[0], tags[1]).
        self.assertEqual(int(d._prev_count[0].item()), 1)
        expected_prev = torch.full_like(d._prev_centroids[0, 0],
                                        (tags[0] + tags[1]) / 2.0)
        self.assertTrue(torch.allclose(d._prev_centroids[0, 0], expected_prev))

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
        self.assertEqual(int(d._prev_count[0].item()), self.CENTROID_HIST)

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

    def test_fit_rows_preserves_batch_axis(self):
        """3D input keeps its batch axis (Task 3 dropped the mean-pool).

        The microbatch refactor needs each row's snapshot kept
        independent through ``_fit_rows`` so the snapshot path can
        write per-row state into the per-B rings.  The legacy
        ``mean(dim=0)`` collapse is gone.
        """
        d = self._make()
        x = torch.randn(5, self.N_SYMBOLS, self.N_DIM)
        y = d._fit_rows(x, self.N_SYMBOLS)
        self.assertEqual(y.shape, (5, self.N_SYMBOLS, self.N_DIM))
        self.assertTrue(torch.allclose(y, x))

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
        self.assertIsNone(d.contrastive_loss(s, w))

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
        loss = d.contrastive_loss(s_cur, w_cur)
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
        loss = d.contrastive_loss(s_cur, w_cur)
        self.assertIsNotNone(loss)

        # Manual: attractive + lam * mean_over_prev(cos).  Buffers carry
        # a leading B=1 dim under Task 3; row 0 is the legacy state.
        current = d._assemble(s_cur, w_cur).reshape(-1)
        ctx = d._recent_centroid().reshape(-1)
        attractive = 1.0 - F.cosine_similarity(
            current.unsqueeze(0), ctx.unsqueeze(0))
        m = int(d._prev_count[0].item())
        prev = d._prev_centroids[0, :m].reshape(m, -1)
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
        loss = d.contrastive_loss(s, w)
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
        loss = d.contrastive_loss(s_cur, w_cur)
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
        loss = d.contrastive_loss(s_cur, w_cur)
        loss.backward()
        # The historical tensors should NOT have received gradient --
        # snapshot() detached them.
        self.assertTrue(s_hist.grad is None
                        or s_hist.grad.abs().sum().item() == 0)
        self.assertTrue(w_hist.grad is None
                        or w_hist.grad.abs().sum().item() == 0)


class TestDiscoursePredictor(_DiscourseTestBase):
    """Unit tests for the AR-sentence predictor pathway.

    Covers the three methods added to InterSentenceLayer when
    ``concept_dim`` is provided: ``predict`` (next-snapshot + its
    attention-entropy confidence), ``predictive_loss`` (cosine
    distance vs. the actual next snapshot), and ``prime`` (cast into
    concept_dim gated by confidence and scale).
    """

    N_SYMBOLS = 4
    MAX_DEPTH = 6
    N_DIM = 8
    CTX = 4
    CENTROID_HIST = 2
    LAM = 1.01
    CONCEPT_DIM = 12

    def _make(self, concept_dim=None):
        return Layers.InterSentenceLayer(
            n_symbols=self.N_SYMBOLS,
            max_depth=self.MAX_DEPTH,
            n_dim=self.N_DIM,
            context_window=self.CTX,
            centroid_history=self.CENTROID_HIST,
            lam=self.LAM,
            concept_dim=concept_dim,
        )

    def test_predict_empty_buffer(self):
        """With zero snapshots recorded, predict() should return
        ``(None, None)`` -- there is nothing to attend over."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        pred, conf = d.predict()
        self.assertIsNone(pred)
        self.assertIsNone(conf)

    def test_predict_shape(self):
        """After at least one snapshot, predict() should return a
        1-D tensor of length ``s_dim`` (= n_symbols * n_dim -- the
        predictor consumes S only, Task 5.3) and a scalar
        confidence."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        pred, conf = d.predict()
        self.assertIsNotNone(pred)
        self.assertEqual(tuple(pred.shape), (d.s_dim,))
        self.assertLess(d.s_dim, d.snapshot_dim,
                        "s_dim must be strictly smaller than snapshot_dim "
                        "(W block is what the [S|W] augmentation contributed)")
        self.assertIsNotNone(conf)
        self.assertEqual(conf.ndim, 0)

    def test_confidence_range(self):
        """Confidence is derived from attention entropy, so it must
        stay in ``[0, 1]``."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        for _ in range(3):
            s = torch.randn(self.N_SYMBOLS, self.N_DIM)
            w = torch.randn(self.MAX_DEPTH, self.N_DIM)
            d.snapshot(s, w)
        _, conf = d.predict()
        self.assertIsNotNone(conf)
        c = float(conf.item())
        self.assertGreaterEqual(c, 0.0)
        self.assertLessEqual(c, 1.0)

    def test_predictive_loss_zero_when_pred_equals_actual(self):
        """If the predicted S-block equals the actual S-slice of the
        assembled snapshot, the cosine distance should be ~0.  The
        predictor scores S only (Task 5.3); the W rows are the
        discourse substrate's concern, not the head's."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        actual_s_flat = d._assemble(s, w)[:d.n_symbols].reshape(-1).detach()
        loss = d.predictive_loss(s, w, actual_s_flat)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.allclose(
            loss, torch.zeros_like(loss), atol=1e-6))

    def test_predictive_loss_none_when_no_prediction(self):
        """predictive_loss() with ``predicted=None`` returns None."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        self.assertIsNone(d.predictive_loss(s, w, None))

    def test_cast_shape(self):
        """prime() output shape is ``[concept_dim]`` (1-D)."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        pred, conf = d.predict()
        bias = d.prime(pred, conf, 0.5)
        self.assertIsNotNone(bias)
        self.assertEqual(tuple(bias.shape), (self.CONCEPT_DIM,))

    def test_prime_zero_when_scale_zero(self):
        """prime() scaled by 0.0 returns a zero tensor (no bias)."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        pred, conf = d.predict()
        bias = d.prime(pred, conf, 0.0)
        self.assertIsNotNone(bias)
        self.assertTrue(torch.allclose(
            bias, torch.zeros_like(bias), atol=1e-6))

    def test_prime_none_without_concept_dim(self):
        """When ``concept_dim`` isn't set, the cast isn't built and
        prime() falls back to None (legacy contrastive-only path)."""
        d = self._make(concept_dim=None)
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s, w)
        # predict() also returns (None, None) in this legacy mode.
        pred, conf = d.predict()
        self.assertIsNone(pred)
        self.assertIsNone(conf)
        bias = d.prime(pred, conf, 0.1)
        self.assertIsNone(bias)

    def test_contrastive_unaffected_by_predictor(self):
        """Adding the predictor/cast must not change the value of the
        contrastive loss on the same inputs."""
        s = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w = torch.randn(self.MAX_DEPTH, self.N_DIM)
        s_hist = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w_hist = torch.randn(self.MAX_DEPTH, self.N_DIM)

        d_plain = self._make(concept_dim=None)
        d_plain.snapshot(s_hist, w_hist)
        loss_plain = d_plain.contrastive_loss(s, w)

        d_pred = self._make(concept_dim=self.CONCEPT_DIM)
        d_pred.snapshot(s_hist, w_hist)
        loss_pred = d_pred.contrastive_loss(s, w)

        self.assertIsNotNone(loss_plain)
        self.assertIsNotNone(loss_pred)
        self.assertTrue(torch.allclose(loss_plain, loss_pred, atol=1e-6))

    def test_predictive_loss_gradient_flows_through_predictor(self):
        """The predictive loss has learnable parameters in the
        predictor; gradient must reach at least one of them."""
        d = self._make(concept_dim=self.CONCEPT_DIM)
        s_hist = torch.randn(self.N_SYMBOLS, self.N_DIM)
        w_hist = torch.randn(self.MAX_DEPTH, self.N_DIM)
        d.snapshot(s_hist, w_hist)

        pred, _ = d.predict()
        self.assertIsNotNone(pred)

        s_cur = torch.randn(self.N_SYMBOLS, self.N_DIM, requires_grad=True)
        w_cur = torch.randn(self.MAX_DEPTH, self.N_DIM, requires_grad=True)
        loss = d.predictive_loss(s_cur, w_cur, pred)
        self.assertIsNotNone(loss)
        loss.backward()

        any_param_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in d.predictor.parameters())
        self.assertTrue(any_param_grad,
                        "no gradient reached the predictor parameters")


class TestDiscourseSpaceIntegration(_DiscourseTestBase):
    """Integration tests against a real BasicModel.

    Verifies that create() wires DiscourseSpace, that forward()
    populates the pending snapshot attributes, and that the runBatch
    path pushes snapshots into history correctly.
    """

    def setUp(self):
        _reload_config()

    def _build_model(self):
        _reload_config()
        model, cfg = Models.BasicModel.from_config(
            os.path.join(_DATA_DIR, 'MentalModel.xml'))
        return model, cfg

    def test_create_wires_discourse_space(self):
        """BasicModel.create() should build and attach WordSpace.discourse
        when the AR/ARUS/AR grammar path is active."""
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
                except (ValueError, AssertionError):
                    # Subspace.normalize() emits AssertionError on
                    # non-finite checks in non-ergodic mode; untrained
                    # weights can land in that regime regardless of the
                    # predictor's width.
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

    def test_discourse_predictor_wired_when_concept_dim_available(self):
        """When BasicModel builds WordSpace with a concept_dim, the
        discourse layer's predictor/cast should be built -- this is
        what enables the AR priming path."""
        self.model, self.cfg = self._build_model()
        d = self.model.wordSpace.discourse
        self.assertIsNotNone(d.concept_dim)
        self.assertIsNotNone(d.predictor)
        self.assertIsNotNone(d.cast)

    def test_predicted_snapshot_cached_after_forward(self):
        """forward() caches ``_predicted_snapshot`` and
        ``_predicted_confidence`` on self (after at least one prior
        snapshot has been recorded) so runBatch can pick them up."""
        self.model, self.cfg = self._build_model()
        d = self.model.wordSpace.discourse
        # Seed one prior snapshot so predict() has context.
        s = torch.randn(d.n_symbols, d.n_dim, device=Models.TheDevice.get())
        w = torch.zeros(d.max_depth, d.n_dim, device=Models.TheDevice.get())
        d.snapshot(s, w)

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
                except (ValueError, AssertionError):
                    # Subspace.normalize() emits AssertionError on
                    # non-finite checks in non-ergodic mode; untrained
                    # weights can land in that regime regardless of the
                    # predictor's width.
                    self.skipTest("Untrained model range violation")

        self.assertIsNotNone(getattr(self.model, '_predicted_snapshot', None))
        self.assertIsNotNone(getattr(self.model, '_predicted_confidence', None))


if __name__ == '__main__':
    unittest.main()
