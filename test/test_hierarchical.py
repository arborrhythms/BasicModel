"""Tests for the hierarchical epistemic architecture.

Covers native pass geometry, WordEncoding identities, and model forward paths.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import warnings
import gc
import random
import numpy as np
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
        """Each attention pass retains the native code dimension."""
        shapes = Models.BasicModel._level_shapes(1024, 4, 8)
        for n, d in shapes:
            self.assertEqual(d, 4)

    def test_single_order(self):
        """One subsymbolic pass retains the native occurrence count."""
        shapes = Models.BasicModel._level_shapes(64, 8, 1)
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0], (64, 8))

    def test_eight_levels(self):
        """Attention can revisit native geometry without halving its positions."""
        self.assertEqual(Models.BasicModel._level_shapes(1024, 4, 8),
                         [(1024, 4)] * 8)

    def test_two_levels(self):
        self.assertEqual(Models.BasicModel._level_shapes(8, 4, 2),
                         [(8, 4), (8, 4)])


# -- WordEncoding 7-tuple ---------------------------------------------

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
                  'join(S, S)', 'lower(S, S)']
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


# -- Backward compat: non-hierarchical models still work --------------

class TestBackwardCompat(unittest.TestCase):

    def _mentalmodel_forward(self):
        model = _make_model('MentalModel.xml')
        try:
            with Models.TheData.runtime_batch(['hello world'], [torch.tensor([0.0])]), \
                 warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                train_input, _ = model.inputSpace.getTrainData()
                x = model.inputSpace.prepInput(train_input[:1])
                with torch.no_grad():
                    return model.forward(x)
        finally:
            model.End()
            model.symbolSpace.soft_reset()
            del model
            gc.collect()

    def test_mentalmodel_unchanged(self):
        """MentalModel.xml (subsymbolicOrder=1) still creates and forwards."""
        result = self._mentalmodel_forward()
        self.assertIsNotNone(result)
        self.assertTrue(all(bool(torch.isfinite(value).all())
                            for value in result if torch.is_tensor(value)))

    def test_mentalmodel_compaction_overflow_regression(self):
        """Reproduce the known failing initialization, never a selected pass.

        The ordinary compatibility assertion above uses ambient RNG. This
        separate regression retains the seed that exposed reused operands.
        """
        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(3)
                random.seed(3)
                np.random.seed(3)
                result = self._mentalmodel_forward()
            self.assertTrue(all(bool(torch.isfinite(value).all())
                                for value in result if torch.is_tensor(value)))
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def test_symbolicspace_per_stage_instances(self):
        """BasicModel builds T independent WholeSpace instances
        (T = subsymbolicOrder) in the wholeSpaces ModuleList."""
        model = _make_model('MentalModel.xml')
        self.assertEqual(len(model.wholeSpaces), model.subsymbolicOrder)


class TestNativePasses(unittest.TestCase):

    def test_ramsified_passes_have_native_reads_and_no_perceptual_folds(self):
        model = _make_model('RamsifiedModel.xml')
        try:
            self.assertEqual(len(model.wholeSpaces), model.subsymbolicOrder)
            self.assertTrue(callable(model.perceptualSpace.synthesize_word_parts))
            self.assertFalse(hasattr(model.perceptualSpace, 'sigmas'))
            for space in model.wholeSpaces:
                self.assertTrue(callable(space.compute_word_property_event))
                self.assertFalse(hasattr(space, 'pi'))
                self.assertFalse(hasattr(space, 'pis'))
        finally:
            model.End()
            model.symbolSpace.soft_reset()


if __name__ == '__main__':
    unittest.main()
