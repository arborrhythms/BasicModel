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
import Models
import Spaces
import Language
matplotlib.use('Agg')

from util import init_config, ProjectPaths, TheXMLConfig

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _make_model(config='MentalModel.xml'):
    init_config(
        path=os.path.join(_DATA_DIR, config),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Language.TheGrammar._configured = False
    model, cfg = Models.BasicModel.from_config(os.path.join(_DATA_DIR, config))
    model.eval()
    return model


# -- _level_shapes ----------------------------------------------------

class TestLevelShapes(unittest.TestCase):

    def test_constant_dim(self):
        """D stays constant across all levels (average merge keeps D fixed)."""
        shapes = Models.BasicModel._level_shapes(1024, 4, 8)
        for n, d in shapes:
            self.assertEqual(d, 4)

    def test_single_order(self):
        """conceptualOrder=1 returns a single post-merge shape."""
        shapes = Models.BasicModel._level_shapes(64, 8, 1)
        self.assertEqual(len(shapes), 1)
        # Post-merge: 64/2 = 32 vectors, D stays 8
        self.assertEqual(shapes[0], (32, 8))

    def test_eight_levels(self):
        """8 levels: N halves each time, D constant."""
        shapes = Models.BasicModel._level_shapes(1024, 4, 8)
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
        shapes = Models.BasicModel._level_shapes(8, 4, 2)
        # Level 0: 8/2=4, D=4. Level 1: 8/4=2, D=4
        self.assertEqual(shapes, [(4, 4), (2, 4)])


# -- _butterfly_merge / _butterfly_unmerge ----------------------------

class _MergeHelper:
    """Lightweight stand-in for BasicModel merge/unmerge (instance methods)."""
    def __init__(self):
        self._merge_diffs = []
    _butterfly_merge = Models.BasicModel._butterfly_merge
    _butterfly_unmerge = Models.BasicModel._butterfly_unmerge


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


# -- WordEncoding 4-tuple ---------------------------------------------

class TestWordEncoding(unittest.TestCase):

    def setUp(self):
        # WordEncoding.encode validates ``rule`` against ``len(TheGrammar)``.
        # These tests poke rule IDs 0-3, so give the grammar enough rules
        # for them to fit. Other tests in this process may have left
        # ``TheGrammar`` in a minimal-configuration state; force a
        # reconfigure with four upward rules so validation passes.
        Language.TheGrammar._configured = False
        Language.TheGrammar.configure({'compose': {
            'S': ['not(S)', 'intersection(S, S)',
                  'union(S, S)', 'lower(S, S)']
        }})

    def test_7tuple_encoding(self):
        """encode produces 7-tuple with correct layout."""
        enc = Spaces.WordEncoding()
        w = enc.encode(batch=0, vector=5, rule=3, order=2)
        self.assertEqual(len(w), 7)
        self.assertEqual(w, (0, 5, 2, 3, -1, -1, -1))

    def test_default_order_zero(self):
        """order defaults to 0."""
        enc = Spaces.WordEncoding()
        w = enc.encode(batch=1, vector=2, rule=0)
        self.assertEqual(w[Spaces.WordEncoding.ORDER], 0)

    def test_7tuple_layout(self):
        """Word tuple is (batch, vector, order, rule, leaf1, leaf2, leaf3)."""
        enc = Spaces.WordEncoding()
        w = enc.encode(batch=0, vector=5, rule=3, order=2, leaf1=10, leaf2=20)
        self.assertEqual(len(w), 7)
        self.assertEqual(w[Spaces.WordEncoding.BATCH], 0)
        self.assertEqual(w[Spaces.WordEncoding.VECTOR], 5)
        self.assertEqual(w[Spaces.WordEncoding.ORDER], 2)
        self.assertEqual(w[Spaces.WordEncoding.RULE], 3)
        self.assertEqual(w[Spaces.WordEncoding.LEAF1], 10)
        self.assertEqual(w[Spaces.WordEncoding.LEAF2], 20)
        self.assertEqual(w[Spaces.WordEncoding.LEAF3], -1)


# -- Hierarchical forward (requires RamsifiedModel.xml with order>1) --

class TestHierarchicalForward(unittest.TestCase):

    def test_forward_runs(self):
        """`RamsifiedModel.xml` with conceptualOrder=2 runs forward without error."""
        model = _make_model('RamsifiedModel.xml')
        if not model.useButterflies:
            self.skipTest("Model not butterfly-enabled")

        sentences = ['test sentence one', 'another test sentence']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:2])
            with torch.no_grad():
                result = model.forward(x)
        self.assertIsNotNone(result)

    def test_symbol_states_populated(self):
        """Hierarchical forward populates symbol_states per level."""
        model = _make_model('RamsifiedModel.xml')
        if not model.useButterflies:
            self.skipTest("Model not butterfly-enabled")

        sentences = ['test one', 'test two']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:2])
            with torch.no_grad():
                model.forward(x)

        self.assertEqual(len(model.symbol_states), model.conceptualOrder)


# -- Backward compat: non-hierarchical models still work --------------

class TestBackwardCompat(unittest.TestCase):

    def test_mentalmodel_unchanged(self):
        """MentalModel.xml (conceptualOrder=1) still creates and forwards."""
        model = _make_model('MentalModel.xml')
        self.assertFalse(model.useButterflies)

        sentences = ['hello world']
        outputs = [torch.tensor([0.0])]

        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:1])
            with torch.no_grad():
                result = model.forward(x)
        self.assertIsNotNone(result)

    def test_symbolicspace_single_instance(self):
        """Non-butterfly model: BasicModel builds T independent SymbolicSpace
        instances (T = conceptualOrder) in symbolicSpaces ModuleList."""
        model = _make_model('MentalModel.xml')
        self.assertEqual(len(model.symbolicSpaces), model.conceptualOrder)


# -- Per-level layer construction -------------------------------------

class TestPerLevelLayers(unittest.TestCase):

    def test_pair_sigmas_created(self):
        """Butterfly path: T independent ConceptualSpace instances, each
        carrying a butterfly-mode PiLayer on pair-dim inputs."""
        model = _make_model('RamsifiedModel.xml')
        if not model.useButterflies:
            self.skipTest("Model not butterfly-enabled")
        pair_dim = 2 * model._butterfly_state_dim
        self.assertEqual(len(model.conceptualSpaces), model.conceptualOrder)
        stage0 = model.conceptualSpaces[0].pi
        self.assertTrue(getattr(stage0, "butterfly", False),
                        "ConceptualSpace.pi should be in butterfly mode")
        self.assertEqual(stage0.nInput, pair_dim)
        self.assertEqual(stage0.nOutput, pair_dim)

    def test_symbolic_spaces_have_no_sigma(self):
        """Butterfly path: post-2026-05 ownership rule, SymbolicSpace
        owns no SigmaLayer. The per-stage butterfly cascade lives
        entirely on ConceptualSpace.pi (see test_per_level_pi_layers_created
        above). This test asserts the absence so a regression that
        re-introduces SS-tier butterfly layers is caught."""
        model = _make_model('RamsifiedModel.xml')
        if not model.useButterflies:
            self.skipTest("Model not butterfly-enabled")
        self.assertEqual(len(model.symbolicSpaces), model.conceptualOrder)
        for s in model.symbolicSpaces:
            self.assertFalse(hasattr(s, 'sigma'),
                             "SymbolicSpace must not own a sigma layer.")
            self.assertFalse(hasattr(s, 'pi'),
                             "SymbolicSpace must not own a pi layer.")

    def test_symbol_factor_matches_configured_volume(self):
        """Pair head reshaping matches the configured symbol volume exactly."""
        model = _make_model('RamsifiedModel.xml')
        if not model.useButterflies:
            self.skipTest("Model not butterfly-enabled")
        self.assertEqual(
            model._butterfly_state_vectors * model._butterfly_state_dim,
            model.nSymbols * model._butterfly_symbol_width,
        )
        self.assertEqual(
            model._butterfly_state_vectors * model._butterfly_symbol_factor,
            model.nSymbols,
        )


if __name__ == '__main__':
    unittest.main()
