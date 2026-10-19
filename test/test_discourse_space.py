"""Shared activation owner: local error, causal context and model lifecycle."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import gc
from pathlib import Path
import tempfile
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


class TestActivationOwner(_DiscourseTestBase):
    def setUp(self):
        self.layer = Layers.SymbolExpectation(4, 2, 3, batch=2)
        self.layer.set_symbol_bank(torch.eye(3))

    def test_cold_then_warm_local_error_and_detached_history(self):
        values = torch.eye(3, requires_grad=True)
        self.layer.predict_and_observe_stm_end_state([1, 1], [values[:1], values[1:2]])
        self.assertIsNone(self.layer.consume_inter_loss())
        self.layer.predict_and_observe_stm_end_state([1, 1], [values[1:2], values[2:]])
        loss = self.layer.consume_inter_loss()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNone(values.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() for p in self.layer.parameters()))
        self.assertIsNone(self.layer.consume_inter_loss())

    def test_bounded_context_and_row_reset(self):
        for _ in range(12):
            self.layer.observe_stm_end_state([1, 1], [torch.eye(3)[:1], torch.eye(3)[1:2]])
        self.assertEqual(len(self.layer), self.layer._inter_chain_window)
        self.layer.Reset(batch=0)
        self.assertFalse(self.layer._inter_context[0])
        self.assertTrue(self.layer._inter_context[1])
        self.assertFalse(self.layer._inter_context_lanes[0])
        self.layer.ensure_batch(4)
        self.assertEqual(len(self.layer._inter_context), 4)
        self.assertEqual(len(self.layer), 0)

    def test_one_shared_head_without_legacy_comprehension_cast(self):
        self.assertIs(self.layer.word_predictor, self.layer._inter_predictor)
        for name in ('predictor', '_s_history', '_e_history', 'observe', 'predict_next', 'predict'):
            self.assertFalse(hasattr(self.layer, name), name)


class TestModelIntegration(_DiscourseTestBase):
    """End-to-end: building a model wires the layer, forward() stages
    the sentence rep, observe() runs from runBatch (or directly).
    """

    def test_discourse_layer_wired_when_sentenceExpectation_true(self):
        _reload_config()
        TheXMLConfig.set("architecture.training.sentenceExpectation", True)
        try:
            model, _ = Models.BasicModel.from_config(
                os.path.join(_DATA_DIR, 'MentalModel.xml'))
            self.assertIsNotNone(model.symbolSpace.expectation)
            self.assertIsInstance(
                model.symbolSpace.expectation, Layers.SymbolExpectation)
            self.assertIs(model.symbolSpace.expectation.word_predictor,
                          model.symbolSpace.expectation._inter_predictor)
            self.model = model
        finally:
            TheXMLConfig.set(
                "architecture.training.sentenceExpectation", False)

    def test_discourse_layer_absent_when_sentenceExpectation_false(self):
        # from_config reloads XML: put the explicit off value in the file,
        # rather than relying on a transient override of the former default.
        source = (Path(_DATA_DIR) / 'MM_xor.xml').read_text()
        source = source.replace('<training>',
                                '<training><sentenceExpectation>false</sentenceExpectation>', 1)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'expectation_off.xml'
            config.write_text(source)
            Language.TheGrammar._configured = False
            model, _ = Models.BasicModel.from_config(str(config))
        self.assertFalse(model.symbolSpace.expectation.expectation_enabled)
        self.assertIn("word", model.symbolSpace.expectation.enabled_levels)
        self.assertIsNotNone(model._what_memory())
        self.model = model
