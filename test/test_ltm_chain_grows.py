"""Task 7 (plan §8): LTM as a chain of STM end-states on the
InterSentenceLayer.

The long-term-memory (LTM) chain is the AR sequence the inter-sentence
predictor consumes. It lives on the existing ``InterSentenceLayer`` as a
bounded per-row ``collections.deque`` of time-ordered tuples
``(depth:int, payload:[depth,D] tensor, tetralemma:tuple|None)``.

Covers:
  1. Growth: processing N sentences yields ``get_stm_chain(N)`` of N
     tuples with matching depths (the plan's required assertion).
  2. Ragged depths: absolute (depth 1) and relative (depth 3)
     end-states coexist in one chain with correct payload shapes.
  3. Time order + ``n``/``b`` accessor semantics.
  4. Bounded eviction at ``ltm_capacity``.
  5. Fail-loud: a NaN/Inf payload RAISES rather than being stored.
  6. Reset clears the chain; the existing ARMA rings are untouched by
     the LTM additions.
  7. Knob: ``ltmCapacity`` flows from config to the constructor.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import gc
import unittest
import torch
import matplotlib
import Layers
matplotlib.use('Agg')


def _release_allocator_cache():
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _LtmTestBase(unittest.TestCase):

    def tearDown(self):
        for attr in ('layer',):
            if hasattr(self, attr):
                setattr(self, attr, None)
        _release_allocator_cache()


class TestLtmChainGrows(_LtmTestBase):
    """The core Task-7 deliverable: the per-row LTM end-state chain."""

    def setUp(self):
        self.D = 3
        # B=1 / row 0 (the plan's stated test shape).
        self.layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=self.D,
            p=5, q=2, batch=1,
        )

    def _absolute_end_state(self, B=1):
        """One depth-1 end-state per row (the collapsed absolute root)."""
        depths = [1] * B
        payloads = [torch.randn(1, self.D) for _ in range(B)]
        return depths, payloads

    def test_processing_n_sentences_grows_chain_to_n(self):
        """Process N sentences -> ``get_stm_chain(N)`` returns N tuples
        whose depths match the recorded end-state depths (verbatim plan
        §8 assertion)."""
        N = 7
        recorded_depths = []
        for _ in range(N):
            depths, payloads = self._absolute_end_state(B=1)
            recorded_depths.append(depths[0])
            self.layer.observe_stm_end_state(depths, payloads)
        chain = self.layer.get_stm_chain(N)
        self.assertEqual(len(chain), N)
        self.assertEqual([entry[0] for entry in chain], recorded_depths)
        # Each tuple is (depth, payload[depth, D], tetralemma).
        for depth, payload, tet in chain:
            self.assertIsInstance(depth, int)
            self.assertEqual(tuple(payload.shape), (depth, self.D))
            self.assertIsNone(tet)

    def test_get_stm_chain_none_returns_all(self):
        """``n=None`` returns the entire chain for the row."""
        N = 4
        for _ in range(N):
            depths, payloads = self._absolute_end_state(B=1)
            self.layer.observe_stm_end_state(depths, payloads)
        self.assertEqual(len(self.layer.get_stm_chain()), N)
        self.assertEqual(len(self.layer.get_stm_chain(None)), N)

    def test_ragged_absolute_and_relative_depths_coexist(self):
        """Absolute (depth 1) and relative (depth 3) end-states live in
        the same chain; each payload has the right ``[depth, D]`` shape.
        """
        # Sentence 1: absolute (depth 1). Sentence 2: relative (depth 3).
        self.layer.observe_stm_end_state([1], [torch.randn(1, self.D)])
        self.layer.observe_stm_end_state([3], [torch.randn(3, self.D)])
        chain = self.layer.get_stm_chain()
        self.assertEqual([e[0] for e in chain], [1, 3])
        self.assertEqual(tuple(chain[0][1].shape), (1, self.D))
        self.assertEqual(tuple(chain[1][1].shape), (3, self.D))

    def test_chain_is_time_ordered_oldest_first(self):
        """The accessor returns oldest-first; the last ``n`` preserves
        order."""
        marks = []
        for i in range(5):
            payload = torch.full((1, self.D), float(i))
            marks.append(float(i))
            self.layer.observe_stm_end_state([1], [payload])
        full = self.layer.get_stm_chain()
        self.assertEqual([float(e[1][0, 0]) for e in full], marks)
        last3 = self.layer.get_stm_chain(3)
        self.assertEqual([float(e[1][0, 0]) for e in last3], marks[-3:])

    def test_payload_is_snapshot_decoupled_from_live_buffer(self):
        """The stored payload is a detached clone, so mutating the
        source tensor after observe does NOT corrupt the chain (the live
        STM buffer is overwritten by the next sentence in the real
        model)."""
        src = torch.zeros(1, self.D)
        self.layer.observe_stm_end_state([1], [src])
        src.add_(99.0)  # overwrite the source in place
        stored = self.layer.get_stm_chain()[0][1]
        self.assertTrue(torch.all(stored == 0.0))

    def test_tetralemma_field_records_when_provided(self):
        """The optional tetralemma is parked verbatim when passed."""
        tet = (0.1, 0.2, 0.3, 0.4)
        self.layer.observe_stm_end_state(
            [1], [torch.randn(1, self.D)], tetralemmas=[tet])
        self.assertEqual(self.layer.get_stm_chain()[0][2], tet)


class TestLtmBounded(_LtmTestBase):
    """The chain is bounded at ``ltm_capacity`` (deque maxlen)."""

    def test_chain_evicts_oldest_at_capacity(self):
        layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=2,
            p=5, q=2, batch=1, ltm_capacity=3,
        )
        self.layer = layer
        for i in range(6):
            payload = torch.full((1, 2), float(i))
            layer.observe_stm_end_state([1], [payload])
        chain = layer.get_stm_chain()
        # maxlen=3 -> only the 3 most recent survive (marks 3, 4, 5).
        self.assertEqual(len(chain), 3)
        self.assertEqual([float(e[1][0, 0]) for e in chain], [3.0, 4.0, 5.0])


class TestLtmFailLoud(_LtmTestBase):
    """A non-finite payload must RAISE, never be silently stored
    (user memory: fail loud on numerical divergence)."""

    def setUp(self):
        self.layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=3, p=5, q=2, batch=1,
        )

    def test_nan_payload_raises(self):
        bad = torch.randn(1, 3)
        bad[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            self.layer.observe_stm_end_state([1], [bad])
        # Nothing was stored.
        self.assertEqual(len(self.layer.get_stm_chain()), 0)

    def test_inf_payload_raises(self):
        bad = torch.randn(1, 3)
        bad[0, 2] = float('inf')
        with self.assertRaises(FloatingPointError):
            self.layer.observe_stm_end_state([1], [bad])


class TestLtmPerRow(_LtmTestBase):
    """Per-row (parallel document streams) chains stay independent."""

    def setUp(self):
        self.layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=2, p=5, q=2, batch=2,
        )

    def test_rows_are_independent(self):
        # Row 0 gets 3 absolute sentences; row 1 gets 1 relative.
        for i in range(3):
            self.layer.observe_stm_end_state(
                [1, 3] if i == 0 else [1, 1],
                [torch.randn(1, 2),
                 torch.randn(3, 2) if i == 0 else torch.randn(1, 2)])
        self.assertEqual(len(self.layer.get_stm_chain(b=0)), 3)
        self.assertEqual(len(self.layer.get_stm_chain(b=1)), 3)
        # Row 1's first end-state is the depth-3 relative one.
        self.assertEqual(self.layer.get_stm_chain(b=1)[0][0], 3)
        self.assertEqual(self.layer.get_stm_chain(b=0)[0][0], 1)

    def test_out_of_range_row_returns_empty(self):
        self.layer.observe_stm_end_state([1, 1],
                                         [torch.randn(1, 2),
                                          torch.randn(1, 2)])
        self.assertEqual(self.layer.get_stm_chain(b=5), [])


class TestLtmResetAndArmaUntouched(_LtmTestBase):
    """Reset clears the LTM chain; the ARMA rings/observe path are not
    perturbed by the LTM additions."""

    def setUp(self):
        self.layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=3, p=5, q=2, batch=2,
        )

    def test_reset_all_rows_clears_chain(self):
        self.layer.observe_stm_end_state([1, 1],
                                         [torch.randn(1, 3),
                                          torch.randn(1, 3)])
        self.assertEqual(len(self.layer.get_stm_chain(b=0)), 1)
        self.layer.Reset()
        self.assertEqual(len(self.layer.get_stm_chain(b=0)), 0)
        self.assertEqual(len(self.layer.get_stm_chain(b=1)), 0)

    def test_reset_single_row_clears_only_that_row(self):
        self.layer.observe_stm_end_state([1, 1],
                                         [torch.randn(1, 3),
                                          torch.randn(1, 3)])
        self.layer.Reset(batch=0)
        self.assertEqual(len(self.layer.get_stm_chain(b=0)), 0)
        self.assertEqual(len(self.layer.get_stm_chain(b=1)), 1)

    def test_arma_observe_path_unbroken_alongside_ltm(self):
        """The ARMA ``observe`` + ``predict_next`` path still works with
        the LTM additions present (they are orthogonal)."""
        s = torch.randn(2, 4, 3)
        loss0 = self.layer.observe(s)             # cold-start prime
        self.assertIsNotNone(loss0)
        loss1 = self.layer.observe(torch.randn(2, 4, 3))
        self.assertIsNotNone(loss1)
        self.assertTrue(torch.isfinite(loss1).all())
        pred = self.layer.predict_next()
        self.assertEqual(tuple(pred.shape), (2, 3))
        # The ARMA rings are unaffected by LTM observes.
        self.layer.observe_stm_end_state([1, 1],
                                         [torch.randn(1, 3),
                                          torch.randn(1, 3)])
        self.assertEqual(self.layer._s_count.tolist(), [2, 2])

    def test_ensure_batch_resizes_chain(self):
        self.layer.observe_stm_end_state([1, 1],
                                         [torch.randn(1, 3),
                                          torch.randn(1, 3)])
        self.layer.ensure_batch(4)
        self.assertEqual(len(self.layer._stm_end_states), 4)
        # Fresh per-row chains after a batch reshape (mirrors the ARMA
        # rings zeroing).
        self.assertEqual(len(self.layer.get_stm_chain(b=0)), 0)


class TestLtmCapacityKnob(_LtmTestBase):
    """``ltmCapacity`` knob: default + explicit pass-through."""

    def test_default_capacity_is_1024(self):
        layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=3, p=5, q=2, batch=1,
        )
        self.layer = layer
        self.assertEqual(layer.ltm_capacity, 1024)
        self.assertEqual(layer._stm_end_states[0].maxlen, 1024)

    def test_explicit_capacity_is_honored(self):
        layer = Layers.InterSentenceLayer(
            n_symbols=4, max_depth=8, n_dim=3, p=5, q=2,
            batch=1, ltm_capacity=42,
        )
        self.layer = layer
        self.assertEqual(layer.ltm_capacity, 42)
        self.assertEqual(layer._stm_end_states[0].maxlen, 42)


if __name__ == '__main__':
    unittest.main()
