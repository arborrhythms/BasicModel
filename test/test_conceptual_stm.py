"""ShortTermMemory on ConceptualSpace -- acceptance tests.

The STM is a per-batch stack of unquantized C-tier "ideas". The
serial / shift-reduce parser (deferred work) will push and pop
here as it reduces concepts into ideas. The current batched-CKY
chart doesn't consume the STM yet -- this is the structural slot.

Distinct from ``WordSpace._stm_fired`` (a once-per-sentence
discourse-priming flag, not a working-memory buffer).
"""
import os
import sys
import unittest

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    """Build a fresh BasicModel from MM_xor.xml."""
    import Models
    import Language
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, cfg = Models.BasicModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return m


class TestShortTermMemoryUnit(unittest.TestCase):
    """Direct unit tests of the ShortTermMemory class."""

    def setUp(self):
        from Spaces import ShortTermMemory
        self.STM = ShortTermMemory

    def test_default_capacity_is_8(self):
        stm = self.STM(batch=1, capacity=None, concept_dim=4)
        self.assertEqual(stm.capacity, 8,
                         "Default STM capacity sits within the 7±2 "
                         "linguistic limit and matches wMax fallback.")

    def test_custom_capacity(self):
        stm = self.STM(batch=1, capacity=64, concept_dim=4)
        self.assertEqual(stm.capacity, 64,
                         "Custom capacity (subsymbolic-wider) overrides "
                         "the linguistic default.")

    def test_push_pop_round_trip(self):
        stm = self.STM(batch=1, capacity=4, concept_dim=3)
        v1 = torch.tensor([0.1, 0.2, 0.3])
        v2 = torch.tensor([0.4, 0.5, 0.6])
        stm.push(0, v1)
        stm.push(0, v2)
        # LIFO: pop returns v2 first (most recent)
        torch.testing.assert_close(stm.pop(0), v2)
        torch.testing.assert_close(stm.pop(0), v1)
        # Empty after both pops
        self.assertIsNone(stm.pop(0))

    def test_peek_does_not_remove(self):
        stm = self.STM(batch=1, capacity=4, concept_dim=2)
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        stm.push(0, a)
        stm.push(0, b)
        torch.testing.assert_close(stm.peek(0, 0), b,
                                   msg="peek(0, 0) returns the top")
        torch.testing.assert_close(stm.peek(0, 1), a,
                                   msg="peek(0, 1) returns the next-from-top")
        # Stack is unchanged
        self.assertEqual(stm.size(0), 2)

    def test_size_and_is_empty(self):
        stm = self.STM(batch=1, capacity=4, concept_dim=2)
        self.assertEqual(stm.size(0), 0)
        self.assertTrue(stm.is_empty(0))
        stm.push(0, torch.zeros(2))
        self.assertEqual(stm.size(0), 1)
        self.assertFalse(stm.is_empty(0))

    def test_is_full_at_capacity(self):
        stm = self.STM(batch=1, capacity=3, concept_dim=2)
        self.assertFalse(stm.is_full(0))
        stm.push(0, torch.zeros(2))
        stm.push(0, torch.zeros(2))
        stm.push(0, torch.zeros(2))
        self.assertTrue(stm.is_full(0))

    def test_push_when_full_raises(self):
        stm = self.STM(batch=1, capacity=2, concept_dim=2)
        stm.push(0, torch.zeros(2))
        stm.push(0, torch.zeros(2))
        with self.assertRaises(RuntimeError):
            stm.push(0, torch.zeros(2))

    def test_clear_all_rows(self):
        stm = self.STM(batch=3, capacity=4, concept_dim=2)
        stm.push(0, torch.tensor([1.0, 2.0]))
        stm.push(1, torch.tensor([3.0, 4.0]))
        stm.push(2, torch.tensor([5.0, 6.0]))
        stm.clear()
        for b in range(3):
            self.assertEqual(stm.size(b), 0)
            self.assertTrue(stm.is_empty(b))

    def test_clear_single_row(self):
        stm = self.STM(batch=2, capacity=4, concept_dim=2)
        stm.push(0, torch.tensor([1.0, 2.0]))
        stm.push(1, torch.tensor([3.0, 4.0]))
        stm.clear(b=0)
        self.assertEqual(stm.size(0), 0)
        self.assertEqual(stm.size(1), 1,
                         "clear(b=0) must not affect other rows")

    def test_independent_rows(self):
        """Each batch row has its own stack."""
        stm = self.STM(batch=2, capacity=4, concept_dim=2)
        stm.push(0, torch.tensor([1.0, 0.0]))
        stm.push(0, torch.tensor([2.0, 0.0]))
        stm.push(1, torch.tensor([0.0, 1.0]))
        self.assertEqual(stm.size(0), 2)
        self.assertEqual(stm.size(1), 1)
        torch.testing.assert_close(stm.peek(0, 0), torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(stm.peek(1, 0), torch.tensor([0.0, 1.0]))

    def test_ensure_batch_resizes(self):
        stm = self.STM(batch=1, capacity=4, concept_dim=2)
        stm.ensure_batch(3)
        self.assertEqual(stm._buffer.shape[0], 3)
        self.assertEqual(stm._depth.shape[0], 3)
        # New rows start empty
        for b in range(3):
            self.assertEqual(stm.size(b), 0)


class TestConceptualSpaceHasSTM(unittest.TestCase):
    """ConceptualSpace exposes a ``stm`` attribute named as such."""

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()

    def test_conceptualSpace_has_stm_attribute(self):
        from Spaces import ShortTermMemory
        for c in self.model.conceptualSpaces:
            self.assertTrue(hasattr(c, 'stm'),
                            f"ConceptualSpace must expose a 'stm' "
                            f"attribute; missing on {c}")
            self.assertIsInstance(c.stm, ShortTermMemory)

    def test_stm_is_buffer_not_parameter(self):
        """STM contents are runtime state, not learned weights."""
        c = self.model.conceptualSpace
        param_names = {n for n, _ in c.stm.named_parameters()}
        self.assertEqual(len(param_names), 0,
                         "ShortTermMemory must not register any "
                         "nn.Parameter; STM contents are working state.")

    def test_stm_initial_state_is_empty(self):
        for c in self.model.conceptualSpaces:
            for b in range(c.stm._buffer.shape[0]):
                self.assertEqual(c.stm.size(b), 0,
                                 f"STM should start empty on row {b} "
                                 f"of {c}")


class TestSTMClearedOnReset(unittest.TestCase):
    """Sentence-boundary semantics: hard Reset clears the STM."""

    @classmethod
    def setUpClass(cls):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cls.model = _fresh_model()

    def test_hard_reset_clears_stm(self):
        c = self.model.conceptualSpace
        concept_dim = int(c.stm.concept_dim)
        # Push something so STM is non-empty
        c.stm.push(0, torch.ones(concept_dim))
        self.assertGreater(c.stm.size(0), 0,
                           "Setup: stm should have an item after push")
        # Hard reset -> STM cleared
        c.Reset(hard=True)
        self.assertEqual(c.stm.size(0), 0,
                         "Hard Reset (sentence boundary) must "
                         "clear the STM.")

    def test_soft_reset_does_not_clear_stm(self):
        c = self.model.conceptualSpace
        concept_dim = int(c.stm.concept_dim)
        c.stm.clear()  # known empty
        c.stm.push(0, torch.ones(concept_dim))
        c.Reset(hard=False)
        # Soft reset is not a sentence boundary; STM persists
        self.assertEqual(c.stm.size(0), 1,
                         "Soft Reset must NOT clear the STM "
                         "(only hard reset / sentence boundary does).")
        # Clean up so other tests start with empty STM
        c.stm.clear()


if __name__ == "__main__":
    unittest.main()
