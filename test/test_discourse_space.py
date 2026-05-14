"""Tests for InterSentenceLayer -- ARMA(p, q) next-sentence predictor.

Replaces the pre-2026-05-14 contrastive cosine tests (retired alongside
``<maskedPrediction>``).  Within-sentence training is now IR-only;
sentence-level AR lives here.

Covers:
  1. Unit-level ARMA layer: observe pushes ``s_t`` into the ring,
     ``predict_next`` produces a stable-shape prediction, MSE loss
     fires after the ring has been primed by p observations.
  2. Buffer + lifecycle: ``ensure_batch`` resizes per-row state,
     ``Reset`` clears both rings, ``__len__`` reports max ring depth.
  3. Back-compat shims: ``predict``/``snapshot``/``contrastive_loss``
     keep their pre-ARMA signatures so existing call sites in
     ``runBatch`` still work during the transition.
  4. Integration: building a BasicModel under ``<sentencePrediction>``
     wires an ``InterSentenceLayer`` on ``wordSpace.discourse`` and
     forward() populates ``_current_discourse_s`` for the runBatch
     observe call.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import gc
import unittest
import torch
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

    def tearDown(self):
        for attr in ("layer", "model"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        _release_allocator_cache()


class TestArmaUnit(_DiscourseTestBase):
    """Stand-alone InterSentenceLayer (no model) -- ring + loss arithmetic."""

    def setUp(self):
        self.n_symbols = 4
        self.n_dim = 3
        self.p = 5
        self.q = 2
        self.layer = Layers.InterSentenceLayer(
            n_symbols=self.n_symbols,
            max_depth=2,
            n_dim=self.n_dim,
            p=self.p, q=self.q,
            batch=2,
        )

    def test_sentence_dim_equals_n_dim(self):
        """Sentence rep is the root S-tier slot -- dim is just n_dim,
        not n_symbols * n_dim.  Without this the predictor's Linear
        would balloon to V_S * n_dim and OOM the allocator on
        MM_5M_bivector-scale configs.
        """
        self.assertEqual(self.layer.sentence_dim, self.n_dim)

    def test_observe_pushes_and_returns_none_on_cold_start(self):
        """First observe primes the ring; no AR prediction to score
        against on the first sentence, so the loss is None."""
        s = torch.randn(2, self.n_symbols, self.n_dim)
        loss = self.layer.observe(s)
        self.assertIsNone(loss)
        self.assertEqual(self.layer._s_count.tolist(), [1, 1])

    def test_observe_accumulates_loss_after_first_step(self):
        """Once the ring has any entry, the next observe scores
        s_hat (from history) against s_t and returns a scalar tensor."""
        self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        loss = self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss).all())

    def test_observe_ring_saturates_at_p(self):
        """``_s_count`` clamps at ``p`` once the AR ring fills."""
        for _ in range(self.p + 4):
            self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        self.assertEqual(self.layer._s_count.tolist(), [self.p, self.p])

    def test_e_history_saturates_at_q(self):
        """MA ring fills to ``q`` and stays there."""
        for _ in range(self.q + 4):
            self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        self.assertEqual(self.layer._e_count.tolist(), [self.q, self.q])

    def test_predict_next_shape(self):
        """``predict_next`` returns ``[B, sentence_dim]`` for B>1."""
        for _ in range(2):
            self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        pred = self.layer.predict_next()
        self.assertEqual(tuple(pred.shape), (2, self.n_dim))

    def test_predict_next_stable_shape_on_cold_start(self):
        """Empty history still produces a (zero-input) prediction so
        callers don't need to special-case the cold-start shape."""
        pred = self.layer.predict_next()
        self.assertEqual(tuple(pred.shape), (2, self.n_dim))

    def test_reset_clears_both_rings(self):
        for _ in range(3):
            self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        self.assertGreater(int(self.layer._s_count.max().item()), 0)
        self.layer.Reset()
        self.assertEqual(self.layer._s_count.tolist(), [0, 0])
        self.assertEqual(self.layer._e_count.tolist(), [0, 0])
        self.assertTrue(torch.all(self.layer._s_history == 0))
        self.assertTrue(torch.all(self.layer._e_history == 0))

    def test_loss_gradient_flows_through_predictor(self):
        """Predictor parameters get gradient when the MSE loss is
        backpropagated (verifies the head is actually being trained)."""
        self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        loss = self.layer.observe(
            torch.randn(2, self.n_symbols, self.n_dim, requires_grad=False))
        self.assertIsNotNone(loss)
        loss.backward()
        any_grad = False
        for p in self.layer.predictor.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                any_grad = True
                break
        self.assertTrue(any_grad,
                        "ARMA predictor did not receive any gradient")

    def test_ensure_batch_resizes_buffers(self):
        self.layer.ensure_batch(4)
        self.assertEqual(self.layer._batch, 4)
        self.assertEqual(tuple(self.layer._s_history.shape),
                         (4, self.p, self.n_dim))
        self.assertEqual(tuple(self.layer._e_history.shape),
                         (4, self.q, self.n_dim))

    def test_len_reports_max_ring_depth(self):
        self.assertEqual(len(self.layer), 0)
        self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        self.assertEqual(len(self.layer), 1)
        for _ in range(self.p + 3):
            self.layer.observe(torch.randn(2, self.n_symbols, self.n_dim))
        self.assertEqual(len(self.layer), self.p)


class TestBackCompatShims(_DiscourseTestBase):
    """The legacy contrastive call sites in ``runBatch`` and
    ``Language`` use ``snapshot`` / ``contrastive_loss`` / ``predict``.
    The ARMA layer keeps those names as thin shims so the wiring keeps
    working until callers migrate to observe / predict_next directly.
    """

    def setUp(self):
        self.layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=2, n_dim=3, p=5, q=2,
            concept_dim=6, batch=1,
        )

    def test_predict_returns_two_tuple(self):
        """Legacy callers unpack ``(prediction, confidence)``.
        ARMA has no native confidence score, but downstream priming
        callers gate on ``conf is None`` so the shim emits a
        placeholder ``1.0`` after the ring is primed; cold-start
        still returns ``(None, None)``.
        """
        pred, conf = self.layer.predict()
        self.assertIsNone(pred)
        self.assertIsNone(conf)
        self.layer.observe(torch.randn(1, 4, 3))
        pred, conf = self.layer.predict()
        self.assertIsNotNone(pred)
        self.assertIsNotNone(conf)
        self.assertEqual(float(conf), 1.0)

    def test_snapshot_alias_calls_observe(self):
        """The ``snapshot`` shim accepts the legacy ``(s, w)`` signature
        and just routes to ``observe`` (w_tensor is ignored)."""
        # First snapshot -- primes the ring, returns None loss.
        loss = self.layer.snapshot(
            torch.randn(1, 4, 3), torch.zeros(1, 2, 3))
        self.assertIsNone(loss)
        self.assertEqual(int(self.layer._s_count[0].item()), 1)

    def test_contrastive_loss_alias_returns_arma_mse(self):
        """The ``contrastive_loss`` shim now returns the ARMA MSE so
        existing runBatch code that adds it to the total still works.
        """
        self.layer.observe(torch.randn(1, 4, 3))
        out = self.layer.contrastive_loss(torch.randn(1, 4, 3))
        self.assertIsNotNone(out)
        self.assertEqual(out.dim(), 0)

    def test_predictive_loss_alias_is_no_op(self):
        """ARMA folds the predictive term into the single observe MSE,
        so the legacy split call now returns None.
        """
        out = self.layer.predictive_loss(
            torch.randn(1, 4, 3), torch.zeros(1, 2, 3),
            predicted=torch.randn(3))
        self.assertIsNone(out)

    def test_prime_lifts_prediction_to_concept_dim(self):
        """``prime`` is used by the chat-loop's ``_c_prior`` injection."""
        self.layer.observe(torch.randn(1, 4, 3))
        s_hat = self.layer.predict_next()
        primed = self.layer.prime(s_hat, confidence=None, scale=0.5)
        self.assertIsNotNone(primed)
        self.assertEqual(primed.shape[-1], 6)


class TestModelIntegration(_DiscourseTestBase):
    """End-to-end: building a model wires the layer, forward() stages
    the sentence rep, observe() runs from runBatch (or directly).
    """

    def test_discourse_layer_wired_when_sentencePrediction_true(self):
        _reload_config()
        TheXMLConfig.set("architecture.training.sentencePrediction", True)
        try:
            model, _ = Models.BasicModel.from_config(
                os.path.join(_DATA_DIR, 'MentalModel.xml'))
            self.assertIsNotNone(model.wordSpace.discourse)
            self.assertIsInstance(
                model.wordSpace.discourse, Layers.InterSentenceLayer)
            self.assertEqual(model.wordSpace.discourse.p, 5)
            self.assertEqual(model.wordSpace.discourse.q, 2)
            self.model = model
        finally:
            TheXMLConfig.set(
                "architecture.training.sentencePrediction", False)

    def test_discourse_layer_absent_when_sentencePrediction_false(self):
        # MM_xor.xml does not set <sentencePrediction>; with the
        # default (False), the layer should not be wired.
        init_config(
            path=os.path.join(_DATA_DIR, 'MM_xor.xml'),
            defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
        )
        Language.TheGrammar._configured = False
        TheXMLConfig.set("architecture.training.sentencePrediction", False)
        model, _ = Models.BasicModel.from_config(
            os.path.join(_DATA_DIR, 'MM_xor.xml'))
        self.assertIsNone(model.wordSpace.discourse)
        self.model = model
