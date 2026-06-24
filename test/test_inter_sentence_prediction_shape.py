"""Task 8 (plan §9): inter-sentence prediction of the next STM end-state
SHAPE from the LTM chain.

``InterSentenceLayer`` gains an inter-level ``IntraSentenceLayer`` instance
(``_inter_predictor``) that reads the LTM end-state chain
(``get_stm_chain``) and predicts the SHAPE of the NEXT end-state:
``predict_next_end_state()`` -> ``(depth_hat:int, payload_hat:[depth_hat, D]
tensor)``. The chat-loop's ``generate_sentence`` stages ``payload_hat``
per-slot on ``ConceptualSpace._c_prior`` (extended to accept a ``[depth, D]``
slot-stack), and a per-sentence ``L_inter = MSE(payload_hat_root,
observed_root)`` accumulates live for the training path.

Core plan-§9 assertion (the verbatim new test): observe a relative
(depth=3) sentence followed by an absolute (depth=1) sentence; assert the
predictor after the relative emits a predicted shape ``(depth, payload)``
with ``depth in {1, 3}`` and finite payload.

Also covers:
  * Cold start (empty chain) -> degenerate ``(1, zeros[1, D])``.
  * depth_hat AR prior tracks the most-recent end-state depth.
  * Fail-loud: a NaN-producing prediction RAISES (not silenced).
  * The inter-level predictor's params are exposed (trainable) and it is
    on ``self.layers`` (ergodic / Start / Reset cascade).
  * L_inter accumulates a live grad tensor + ``consume_inter_loss`` mean +
    reset; grad-gated (no accumulation under no-grad); weight gate.
  * ``ConceptualSpace._c_prior`` accepts a ``[depth, D]`` slot-stack AND
    keeps the legacy ``[D]`` / ``[1, D]`` broadcast byte-identical.
  * Reset clears the pending prediction + loss accumulator.
  * Absolute-only / no-discourse configs no-op (the model helper returns
    None).
"""

import os
import sys
import gc
import unittest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import matplotlib
matplotlib.use("Agg")
import Layers


def _release_allocator_cache():
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_layer(D=4, batch=1, ltm_capacity=1024, concept_dim=None):
    """An InterSentenceLayer with the inter-level predictor built
    (``concept_dim`` set, defaulting to D so the predictor width matches
    the end-state payload width)."""
    return Layers.InterSentenceLayer(
        n_symbols=4, max_depth=8, n_dim=D,
        p=5, q=2, batch=batch, ltm_capacity=ltm_capacity,
        concept_dim=(D if concept_dim is None else concept_dim),
    )


class _Base(unittest.TestCase):
    def tearDown(self):
        for attr in ("layer",):
            if hasattr(self, attr):
                setattr(self, attr, None)
        _release_allocator_cache()


class TestPredictNextEndStateShape(_Base):
    """The core Task-8 deliverable: predict the next end-state SHAPE."""

    def setUp(self):
        self.D = 4
        self.layer = _make_layer(D=self.D)

    def test_relative_then_absolute_emits_valid_shape(self):
        """Verbatim plan §9: observe relative (depth 3) then absolute
        (depth 1); the predictor after the relative emits ``(depth,
        payload)`` with ``depth in {1, 3}`` and finite payload."""
        # Observe a RELATIVE end-state (depth 3).
        self.layer.observe_stm_end_state([3], [torch.randn(3, self.D)])
        depth_hat, payload_hat = self.layer.predict_next_end_state()
        self.assertIn(depth_hat, (1, 3),
                      "depth_hat after a relative must be in {1, 3}")
        self.assertEqual(tuple(payload_hat.shape), (depth_hat, self.D),
                         "payload_hat must be [depth_hat, D]")
        self.assertTrue(torch.isfinite(payload_hat).all(),
                        "predicted payload must be finite")

        # Now observe an ABSOLUTE end-state (depth 1) and predict again.
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        depth_hat2, payload_hat2 = self.layer.predict_next_end_state()
        self.assertIn(depth_hat2, (1, 3))
        self.assertEqual(tuple(payload_hat2.shape), (depth_hat2, self.D))
        self.assertTrue(torch.isfinite(payload_hat2).all())

    def test_depth_hat_tracks_most_recent_end_state(self):
        """The AR-prior depth_hat equals the most-recent end-state depth."""
        self.layer.observe_stm_end_state([3], [torch.randn(3, self.D)])
        self.assertEqual(self.layer.predict_next_end_state()[0], 3)
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.assertEqual(self.layer.predict_next_end_state()[0], 1)
        self.layer.observe_stm_end_state([3], [torch.randn(3, self.D)])
        self.assertEqual(self.layer.predict_next_end_state()[0], 3)

    def test_cold_start_returns_degenerate_shape(self):
        """Empty chain -> ``(1, zeros[1, D])`` (well-defined degenerate)."""
        depth_hat, payload_hat = self.layer.predict_next_end_state()
        self.assertEqual(depth_hat, 1)
        self.assertEqual(tuple(payload_hat.shape), (1, self.D))
        self.assertTrue(torch.all(payload_hat == 0.0))

    def test_payload_finite_over_a_longer_chain(self):
        """A mixed run of absolute/relative end-states keeps predictions
        finite and validly shaped throughout."""
        for i in range(10):
            d = 3 if (i % 2 == 0) else 1
            self.layer.observe_stm_end_state([d], [torch.randn(d, self.D)])
            depth_hat, payload_hat = self.layer.predict_next_end_state()
            self.assertIn(depth_hat, (1, 3))
            self.assertEqual(tuple(payload_hat.shape), (depth_hat, self.D))
            self.assertTrue(torch.isfinite(payload_hat).all())


class TestPredictFailLoud(_Base):
    """A non-finite predicted root RAISES (user memory: fail loud)."""

    def test_nan_prediction_raises(self):
        D = 3
        layer = _make_layer(D=D)
        self.layer = layer
        layer.observe_stm_end_state([1], [torch.randn(1, D)])
        # Force the predictor to emit NaN by corrupting its final Sigma
        # weight; the prediction must RAISE rather than return NaN.
        with torch.no_grad():
            for p in layer._inter_predictor.parameters():
                p.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            layer.predict_next_end_state()


class TestInterPredictorParamsExposed(_Base):
    """The inter-level predictor is a trainable submodule on the layer."""

    def setUp(self):
        self.D = 4
        self.layer = _make_layer(D=self.D)

    def test_predictor_is_built(self):
        self.assertIsNotNone(self.layer._inter_predictor)
        self.assertIsInstance(self.layer._inter_predictor,
                              Layers.IntraSentenceLayer)

    def test_predictor_params_are_in_layer_parameters(self):
        layer_params = list(self.layer.parameters())
        pred_params = list(self.layer._inter_predictor.parameters())
        self.assertGreater(len(pred_params), 0)
        for p in pred_params:
            self.assertTrue(any(p is q for q in layer_params),
                            "every inter-predictor param must be reachable "
                            "via InterSentenceLayer.parameters()")

    def test_predictor_is_on_layers_for_ergodic_cascade(self):
        self.assertTrue(
            any(l is self.layer._inter_predictor for l in self.layer.layers),
            "the inter-predictor must be on self.layers (Start/Reset/sigma)")

    def test_chain_window_is_bounded(self):
        # K = min(ltm_capacity, 8); with the default 1024 cap -> 8.
        self.assertEqual(self.layer._inter_chain_window, 8)
        small = _make_layer(D=3, ltm_capacity=3)
        self.assertEqual(small._inter_chain_window, 3)


class TestInterLoss(_Base):
    """L_inter accumulation + consume + grad/weight gating."""

    def setUp(self):
        self.D = 4
        self.layer = _make_layer(D=self.D)
        self.layer.set_inter_loss_weight(0.1)

    def test_predict_then_observe_accumulates_live_loss(self):
        with torch.enable_grad():
            self.layer.predict_next_end_state()                   # predict
            self.layer.observe_stm_end_state(
                [3], [torch.randn(3, self.D)])                    # score
            self.layer.predict_next_end_state()
            self.layer.observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
        loss = self.layer.consume_inter_loss()
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad,
                        "L_inter must be a live grad tensor (trains predictor)")
        self.assertTrue(torch.isfinite(loss).all())
        # consume resets -> a second consume with no new accumulation is None.
        self.assertIsNone(self.layer.consume_inter_loss())

    def test_loss_backprops_into_predictor(self):
        # Prime the chain first: a prediction made from an EMPTY chain is a
        # degenerate zeros root with no grad (correctly not scored), so a
        # real (grad-bearing) prediction needs >=1 prior end-state.
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        with torch.enable_grad():
            self.layer.predict_next_end_state()
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
            loss = self.layer.consume_inter_loss()
        self.assertIsNotNone(loss)
        loss.backward()
        grads = [p.grad for p in self.layer._inter_predictor.parameters()
                 if p.grad is not None]
        self.assertTrue(len(grads) > 0,
                        "backward must populate inter-predictor grads")
        self.assertTrue(any(torch.any(g != 0.0) for g in grads),
                        "at least one predictor grad must be non-zero")

    def test_no_accumulation_under_no_grad(self):
        with torch.no_grad():
            self.layer.predict_next_end_state()
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.assertIsNone(self.layer.consume_inter_loss(),
                          "eval-time forwards must not grow the loss graph")

    def test_weight_off_disables_accumulation(self):
        self.layer.set_inter_loss_weight(0.0)
        with torch.enable_grad():
            self.layer.predict_next_end_state()
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.assertIsNone(self.layer.consume_inter_loss())

    def test_observe_without_prediction_does_not_accumulate(self):
        # No predict_next_end_state call -> nothing pending -> no score.
        with torch.enable_grad():
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.assertIsNone(self.layer.consume_inter_loss())

    def test_cold_start_prediction_is_not_scored(self):
        # A prediction from an EMPTY chain is the degenerate zeros root and
        # carries no grad; the following observe must NOT accumulate a term
        # (there is nothing meaningful to train against on cold start).
        with torch.enable_grad():
            self.layer.predict_next_end_state()                   # cold
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.assertIsNone(self.layer.consume_inter_loss())

    def test_reset_clears_pending_prediction_and_accumulator(self):
        with torch.enable_grad():
            self.layer.predict_next_end_state()
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.layer.Reset()
        self.assertIsNone(self.layer.consume_inter_loss())
        self.assertTrue(all(r is None
                            for r in self.layer._inter_last_pred_root))


class TestTrainingBoundaryStagesPrediction(_Base):
    """Regression for the silent-correctness bug: the TRAINING
    sentence-boundary hook must STAGE a next-end-state prediction before
    scoring it, so ``L_inter`` actually accumulates during training.

    The training boundary (bin/Models.py) historically called bare
    ``observe_stm_end_state`` — which only scores an ALREADY-staged
    prediction. Since nothing staged one on that path,
    ``consume_inter_loss`` always returned ``None`` and the inter-predictor
    never learned. The fix routes the training boundary through the single
    ``predict_and_observe_stm_end_state`` call (predict-from-history THEN
    observe). This test drives THAT method directly (it does NOT call
    ``predict_next_end_state`` itself — that would mask the bug) and asserts
    the loss is cold-None on the 1st boundary but live/finite/grad-bearing
    on the 2nd+.
    """

    def setUp(self):
        self.D = 4
        self.layer = _make_layer(D=self.D)
        self.layer.set_inter_loss_weight(0.1)

    def test_combined_call_accumulates_live_loss_after_first_boundary(self):
        with torch.enable_grad():
            # Boundary 1: cold chain -> the staged prediction is the
            # degenerate zeros root (no grad), so observing it must NOT
            # accumulate a term. (1st sentence has no prior to predict from.)
            self.layer.predict_and_observe_stm_end_state(
                [3], [torch.randn(3, self.D)])
        self.assertIsNone(
            self.layer.consume_inter_loss(),
            "the 1st (cold) training boundary must not accumulate L_inter")

        with torch.enable_grad():
            # Boundary 2: the chain now holds the boundary-1 end-state, so the
            # staged prediction is a REAL grad-bearing root and observing the
            # arriving end-state scores it -> L_inter accumulates.
            self.layer.predict_and_observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
        loss = self.layer.consume_inter_loss()
        self.assertIsNotNone(
            loss,
            "the 2nd training boundary MUST accumulate L_inter (the "
            "regression: bare observe never staged a prediction)")
        self.assertTrue(torch.isfinite(loss).all(),
                        "L_inter must be finite (fail-loud otherwise)")
        self.assertTrue(
            loss.requires_grad,
            "L_inter must carry grad so the inter-predictor trains")

    def test_combined_call_loss_backprops_into_predictor(self):
        # Prime with one boundary, then a second whose staged prediction is
        # grad-bearing; the consumed loss must populate predictor grads.
        with torch.enable_grad():
            self.layer.predict_and_observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
            self.layer.predict_and_observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
            loss = self.layer.consume_inter_loss()
        self.assertIsNotNone(loss)
        loss.backward()
        grads = [p.grad for p in self.layer._inter_predictor.parameters()
                 if p.grad is not None]
        self.assertTrue(len(grads) > 0,
                        "backward must populate inter-predictor grads")
        self.assertTrue(any(torch.any(g != 0.0) for g in grads),
                        "at least one predictor grad must be non-zero")

    def test_combined_call_accumulates_across_several_boundaries(self):
        # A run of consecutive training boundaries: each post-cold boundary
        # contributes a scored term, and the chain grows by exactly one per
        # call (no double-append).
        with torch.enable_grad():
            for i in range(5):
                d = 3 if (i % 2 == 0) else 1
                self.layer.predict_and_observe_stm_end_state(
                    [d], [torch.randn(d, self.D)])
        # 5 boundaries -> chain length 5 (one append each, no double-append).
        self.assertEqual(len(self.layer.get_stm_chain(b=0)), 5)
        loss = self.layer.consume_inter_loss()
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss).all())

    def test_combined_call_no_grad_does_not_accumulate(self):
        with torch.no_grad():
            self.layer.predict_and_observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
            self.layer.predict_and_observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
        self.assertIsNone(
            self.layer.consume_inter_loss(),
            "eval-time combined calls must not grow the loss graph")

    def test_combined_call_no_predictor_is_bare_observe(self):
        # No inter-predictor (concept_dim unset) -> the combined call must
        # degenerate to a bare observe: the chain still grows, no loss, no
        # error (absolute-only / no-discourse no-op safety).
        layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=self.D,
            p=5, q=2, batch=1, ltm_capacity=1024, concept_dim=None)
        self.layer = layer
        self.assertIsNone(layer._inter_predictor)
        with torch.enable_grad():
            layer.predict_and_observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
            layer.predict_and_observe_stm_end_state(
                [1], [torch.randn(1, self.D)])
        self.assertEqual(len(layer.get_stm_chain(b=0)), 2,
                         "chain must still grow when there is no predictor")
        self.assertIsNone(layer.consume_inter_loss(),
                          "no predictor -> no L_inter")

    def test_combined_call_matches_manual_two_call_order(self):
        # The combined call must produce the SAME staging/scoring as the
        # manual predict-then-observe order generate_sentence uses. Build two
        # layers with identical weights, drive one via the combined call and
        # the other via the explicit two-call order over the same inputs, and
        # assert their consumed losses match.
        torch.manual_seed(1234)
        a = _make_layer(D=self.D)
        a.set_inter_loss_weight(0.1)
        b = _make_layer(D=self.D)
        b.set_inter_loss_weight(0.1)
        b.load_state_dict(a.state_dict())
        es1 = torch.randn(3, self.D)
        es2 = torch.randn(1, self.D)
        with torch.enable_grad():
            # Combined path.
            a.predict_and_observe_stm_end_state([3], [es1.clone()])
            a.predict_and_observe_stm_end_state([1], [es2.clone()])
            la = a.consume_inter_loss()
            # Manual two-call path (the generate_sentence order).
            b.predict_next_end_state(0)
            b.observe_stm_end_state([3], [es1.clone()])
            b.predict_next_end_state(0)
            b.observe_stm_end_state([1], [es2.clone()])
            lb = b.consume_inter_loss()
        self.assertIsNotNone(la)
        self.assertIsNotNone(lb)
        self.assertTrue(torch.allclose(la, lb, atol=1e-6),
                        "combined call must match the manual two-call order")
        self.layer = a  # tearDown release


class TestCPriorSlotStack(_Base):
    """``ConceptualSpace._c_prior`` accepts a ``[depth, D]`` slot-stack and
    keeps the legacy ``[D]`` / ``[1, D]`` broadcast byte-identical.

    Replicates the forward-path staging logic on a synthetic
    ``event_for_carrier`` so the test stays a focused unit test (no full
    model build): the production block lives at
    ``ConceptualSpace.forward`` (bin/Spaces.py).
    """

    @staticmethod
    def _apply_prior(event, prior, slotwise):
        """Mirror of the bin/Spaces.py ``_c_prior`` injection block."""
        if (slotwise and prior.dim() == 2 and event.dim() == 3
                and prior.shape[-1] == event.shape[-1]):
            B = event.shape[0]
            N = event.shape[1]
            depth = min(int(prior.shape[0]), int(N))
            if depth > 0:
                add = torch.zeros_like(event)
                add[:, :depth, :] = prior[:depth].unsqueeze(0).expand(B, -1, -1)
                event = event + add
            return event
        if prior.dim() == 1 or prior.dim() == 2:
            if prior.dim() == 1:
                prior = prior.unsqueeze(0)
            if event.dim() == 3 and prior.dim() == 2:
                if prior.shape[0] == 1:
                    prior_b = prior.expand(event.shape[0], -1)
                elif prior.shape[0] == event.shape[0]:
                    prior_b = prior
                else:
                    K = max(1, event.shape[0] // max(1, prior.shape[0]))
                    prior_b = prior.repeat_interleave(K, dim=0)
                if prior_b.shape[-1] == event.shape[-1]:
                    event = event + prior_b.unsqueeze(1)
        return event

    def test_depth3_slotstack_stages_first_three_slots(self):
        B, N, D = 1, 5, 4
        event = torch.zeros(B, N, D)
        payload_hat = torch.arange(1, 3 * D + 1, dtype=torch.float).reshape(3, D)
        out = self._apply_prior(event, payload_hat, slotwise=True)
        # First 3 slots get the payload rows; slots 3,4 stay zero.
        self.assertTrue(torch.equal(out[0, 0], payload_hat[0]))
        self.assertTrue(torch.equal(out[0, 1], payload_hat[1]))
        self.assertTrue(torch.equal(out[0, 2], payload_hat[2]))
        self.assertTrue(torch.all(out[0, 3] == 0.0))
        self.assertTrue(torch.all(out[0, 4] == 0.0))

    def test_depth1_slotstack_stages_only_slot0(self):
        B, N, D = 1, 4, 4
        event = torch.zeros(B, N, D)
        payload_hat = torch.full((1, D), 7.0)
        out = self._apply_prior(event, payload_hat, slotwise=True)
        self.assertTrue(torch.all(out[0, 0] == 7.0))
        self.assertTrue(torch.all(out[0, 1:] == 0.0),
                        "a depth-1 slot-stack must touch ONLY slot 0")

    def test_slotstack_clamped_to_available_slots(self):
        # depth 3 but only N=2 slots -> stage the first 2, drop the 3rd.
        B, N, D = 1, 2, 4
        event = torch.zeros(B, N, D)
        payload_hat = torch.ones(3, D)
        out = self._apply_prior(event, payload_hat, slotwise=True)
        self.assertEqual(tuple(out.shape), (B, N, D))
        self.assertTrue(torch.all(out == 1.0))

    def test_slotstack_broadcasts_over_batch(self):
        B, N, D = 3, 4, 4
        event = torch.zeros(B, N, D)
        payload_hat = torch.full((2, D), 5.0)
        out = self._apply_prior(event, payload_hat, slotwise=True)
        for b in range(B):
            self.assertTrue(torch.all(out[b, 0] == 5.0))
            self.assertTrue(torch.all(out[b, 1] == 5.0))
            self.assertTrue(torch.all(out[b, 2] == 0.0))

    def test_legacy_1d_prior_broadcasts_all_slots_byte_identical(self):
        B, N, D = 2, 3, 4
        event = torch.zeros(B, N, D)
        prior = torch.full((D,), 2.0)               # [D]
        out_new = self._apply_prior(event, prior, slotwise=False)
        # Legacy reference: unsqueeze to [1, D], broadcast to [B, D], add
        # across ALL slots.
        ref = event + prior.unsqueeze(0).expand(B, -1).unsqueeze(1)
        self.assertTrue(torch.equal(out_new, ref))
        self.assertTrue(torch.all(out_new == 2.0),
                        "[D] prior must hit every slot (legacy broadcast)")

    def test_legacy_1xD_prior_broadcasts_all_slots(self):
        B, N, D = 2, 3, 4
        event = torch.zeros(B, N, D)
        prior = torch.full((1, D), 3.0)             # [1, D]
        out = self._apply_prior(event, prior, slotwise=False)
        self.assertTrue(torch.all(out == 3.0))

    def test_legacy_BxD_prior_per_row(self):
        B, N, D = 2, 3, 4
        event = torch.zeros(B, N, D)
        prior = torch.stack([torch.full((D,), 1.0),
                             torch.full((D,), 2.0)], dim=0)   # [B, D]
        out = self._apply_prior(event, prior, slotwise=False)
        self.assertTrue(torch.all(out[0] == 1.0))
        self.assertTrue(torch.all(out[1] == 2.0))


class TestInterContrastive(_Base):
    """InfoNCE next-idea contrastive accumulation (Step 1): ranks the actual
    next root above the chain's past roots; trains the predictor; gradient never
    touches the detached targets; off (weight 0) -> no accumulation."""

    def setUp(self):
        self.D = 4
        self.layer = _make_layer(D=self.D)
        self.layer.set_inter_loss_weight(0.0)         # isolate the contrastive
        self.layer.set_inter_contrastive(0.5, temp=0.1)

    def _prime_and_score(self):
        # >=2 prior end-states so the scoring observe has negatives to rank.
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        with torch.enable_grad():
            self.layer.predict_next_end_state()
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])

    def test_contrastive_accumulates_live_loss(self):
        self._prime_and_score()
        loss = self.layer.consume_inter_contrastive_loss()
        self.assertIsNotNone(loss)
        self.assertTrue(loss.requires_grad,
                        "InfoNCE term must be a live grad tensor")
        self.assertTrue(torch.isfinite(loss).all())
        # consume resets.
        self.assertIsNone(self.layer.consume_inter_contrastive_loss())

    def test_contrastive_trains_predictor(self):
        self._prime_and_score()
        loss = self.layer.consume_inter_contrastive_loss()
        self.assertIsNotNone(loss)
        loss.backward()
        grads = [p.grad for p in self.layer._inter_predictor.parameters()
                 if p.grad is not None]
        self.assertTrue(len(grads) > 0,
                        "InfoNCE backward must populate predictor grads")
        self.assertTrue(any(torch.any(g != 0.0) for g in grads),
                        "at least one predictor grad must be non-zero")

    def test_weight_off_disables_contrastive(self):
        self.layer.set_inter_contrastive(0.0)
        self._prime_and_score()
        self.assertIsNone(self.layer.consume_inter_contrastive_loss())

    def test_no_accumulation_under_no_grad(self):
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        with torch.no_grad():
            self.layer.predict_next_end_state()
            self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.assertIsNone(self.layer.consume_inter_contrastive_loss())


if __name__ == "__main__":
    unittest.main()
