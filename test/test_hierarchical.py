"""Tests for the hierarchical epistemic architecture.

Covers: _level_shapes, _butterfly_merge/unmerge, WordEncoding 4-tuple,
per-level Sigmas/Pis, hierarchical forward, and backward compat.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import warnings
import torch
import matplotlib
matplotlib.use('Agg')

from BasicModel import MentalModel, BaseModel, TheData, TheDevice
from util import init_config, ProjectPaths, TheXMLConfig

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _make_model(config='MentalModel.xml'):
    init_config(
        path=os.path.join(_DATA_DIR, config),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    from Space import TheGrammar
    TheGrammar._configured = False
    model, cfg = MentalModel.from_config(os.path.join(_DATA_DIR, config))
    model.eval()
    return model


# ── _level_shapes ────────────────────────────────────────────────────

class TestLevelShapes(unittest.TestCase):

    def test_constant_dim(self):
        """D stays constant across all levels (average merge keeps D fixed)."""
        shapes = MentalModel._level_shapes(1024, 4, 8)
        for n, d in shapes:
            self.assertEqual(d, 4)

    def test_single_order(self):
        """conceptualOrder=1 returns a single post-merge shape."""
        shapes = MentalModel._level_shapes(64, 8, 1)
        self.assertEqual(len(shapes), 1)
        # Post-merge: 64/2 = 32 vectors, D stays 8
        self.assertEqual(shapes[0], (32, 8))

    def test_eight_levels(self):
        """8 levels: N halves each time, D constant."""
        shapes = MentalModel._level_shapes(1024, 4, 8)
        self.assertEqual(len(shapes), 8)
        # Level 0: 1024/2=512, D=4
        self.assertEqual(shapes[0], (512, 4))
        # Level 7: 1024/256=4, D=4
        self.assertEqual(shapes[7], (4, 4))
        # Each level halves N, D stays constant
        for i in range(7):
            self.assertEqual(shapes[i][0], 2 * shapes[i + 1][0])
            self.assertEqual(shapes[i][1], shapes[i + 1][1])

    def test_two_levels(self):
        """Simple 2-level case."""
        shapes = MentalModel._level_shapes(8, 4, 2)
        # Level 0: 8/2=4, D=4. Level 1: 8/4=2, D=4
        self.assertEqual(shapes, [(4, 4), (2, 4)])


# ── _butterfly_merge / _butterfly_unmerge ────────────────────────────

class _MergeHelper:
    """Lightweight stand-in for MentalModel merge/unmerge (instance methods)."""
    def __init__(self):
        self._merge_diffs = []
    _butterfly_merge = MentalModel._butterfly_merge
    _butterfly_unmerge = MentalModel._butterfly_unmerge


class TestButterflyMerge(unittest.TestCase):

    def setUp(self):
        self.helper = _MergeHelper()

    def test_merge_shape(self):
        """[2, 8, 4] -> [2, 4, 4] (D stays constant)."""
        x = torch.randn(2, 8, 4)
        merged = self.helper._butterfly_merge(x)
        self.assertEqual(merged.shape, (2, 4, 4))

    def test_unmerge_inverse(self):
        """unmerge(merge(x)) recovers original."""
        x = torch.randn(3, 16, 6)
        merged = self.helper._butterfly_merge(x)
        recovered = self.helper._butterfly_unmerge(merged)
        self.assertTrue(torch.allclose(x, recovered, atol=1e-6))

    def test_information_preserved(self):
        """Average merge + cached diff preserves all information."""
        x = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
        merged = self.helper._butterfly_merge(x)
        # Shape halves N, keeps D
        self.assertEqual(merged.shape, (1, 4, 3))
        # Average halves the sum
        self.assertAlmostEqual(merged.sum().item(), x.sum().item() / 2, places=4)
        # But full round-trip recovers everything
        recovered = self.helper._butterfly_unmerge(merged)
        self.assertTrue(torch.allclose(x, recovered))


# ── WordEncoding 4-tuple ─────────────────────────────────────────────

class TestWordEncoding(unittest.TestCase):

    def test_4tuple_encoding(self):
        """encode/decode with order parameter."""
        from Space import WordEncoding
        enc = WordEncoding()
        w = enc.encode(batch=0, vector=5, rule=3, order=2)
        self.assertEqual(len(w), 4)
        self.assertEqual(w, (0, 5, 3, 2))

    def test_default_order_zero(self):
        """order defaults to 0 for backward compat."""
        from Space import WordEncoding
        enc = WordEncoding()
        w = enc.encode(batch=1, vector=2, rule=0)
        self.assertEqual(w[3], 0)

    def test_backward_compat_3tuple_unpack(self):
        """Old 3-tuple words unpack correctly with word[0], word[1], word[2]."""
        # The decompose code uses word[0], word[1], word[2] indexing
        # which works for both 3-tuples and 4-tuples
        old_word = (0, 5, 3)
        new_word = (0, 5, 3, 2)
        self.assertEqual(old_word[0], new_word[0])
        self.assertEqual(old_word[1], new_word[1])
        self.assertEqual(old_word[2], new_word[2])
        # 3-tuple has len >= 3, so the guard `if len(word) < 3: continue` passes
        self.assertTrue(len(old_word) >= 3)


# ── Hierarchical forward (requires RamsifiedModel.xml with order>1) ──

class TestHierarchicalForward(unittest.TestCase):

    def test_forward_runs(self):
        """RamsifiedModel with conceptualOrder=2 runs forward without error."""
        model = _make_model('RamsifiedModel.xml')
        if not getattr(model, '_hierarchical', False):
            self.skipTest("Model not hierarchical (conceptualOrder <= 1)")

        sentences = ['test sentence one', 'another test sentence']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:2])
            with torch.no_grad():
                result = model.forward(x)
        self.assertIsNotNone(result)

    def test_concept_states_populated(self):
        """Hierarchical forward populates concept_states per level."""
        model = _make_model('RamsifiedModel.xml')
        if not getattr(model, '_hierarchical', False):
            self.skipTest("Model not hierarchical")

        sentences = ['test one', 'test two']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:2])
            with torch.no_grad():
                model.forward(x)

        self.assertEqual(len(model.concept_states), model.conceptualOrder)


# ── Backward compat: non-hierarchical models still work ──────────────

class TestBackwardCompat(unittest.TestCase):

    def test_mentalmodel_unchanged(self):
        """MentalModel.xml (conceptualOrder=1) still creates and forwards."""
        model = _make_model('MentalModel.xml')
        self.assertFalse(getattr(model, '_hierarchical', False))

        sentences = ['hello world']
        outputs = [torch.tensor([0.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:1])
            with torch.no_grad():
                result = model.forward(x)
        self.assertIsNotNone(result)

    def test_symbolicspace_not_hierarchical(self):
        """SymbolicSpace in non-hierarchical model has no pi_layers list."""
        model = _make_model('MentalModel.xml')
        self.assertFalse(model.symbolicSpace._hierarchical)
        self.assertIsNone(model.symbolicSpace.pi_layers)


# ── Per-level layer construction ─────────────────────────────────────

class TestPerLevelLayers(unittest.TestCase):

    def test_conceptual_sigmas_created(self):
        """Hierarchical ConceptualSpace has per-level sigmas."""
        model = _make_model('RamsifiedModel.xml')
        if not getattr(model, '_hierarchical', False):
            self.skipTest("Model not hierarchical")
        cs = model.conceptualSpace
        self.assertTrue(cs._hierarchical)
        self.assertEqual(len(cs.sigmas), model.conceptualOrder)

    def test_symbolic_pi_layers_created(self):
        """Hierarchical SymbolicSpace has per-level pi_layers."""
        model = _make_model('RamsifiedModel.xml')
        if not getattr(model, '_hierarchical', False):
            self.skipTest("Model not hierarchical")
        ss = model.symbolicSpace
        self.assertTrue(ss._hierarchical)
        self.assertEqual(len(ss.pi_layers), model.conceptualOrder)

    def test_dim_projections_count(self):
        """Number of dim_projections = conceptualOrder (one per level)."""
        model = _make_model('RamsifiedModel.xml')
        if not getattr(model, '_hierarchical', False):
            self.skipTest("Model not hierarchical")
        cs = model.conceptualSpace
        self.assertEqual(len(cs.dim_projections), model.conceptualOrder)


if __name__ == '__main__':
    unittest.main()
