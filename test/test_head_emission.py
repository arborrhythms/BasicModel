"""Downward S -> C emits the codebook atom that best matches the deep state."""

# ---------------------------------------------------------------------
# Skipped pending migration to the post-2026-05-01 chart / GrammarLayer
# surface. The tests in this module exercised the legacy SyntacticLayer
# dispatch tables (`_RULE_METHODS`, `*Forward` / `*Reverse`, `project`,
# `compose(data, subspace, grammar)`, etc.) which were removed by the
# 2026-05-01 syntactic-layer refactor. Rewrite to use the new
# `Chart` class and the `GRAMMAR_LAYER_CLASSES` GrammarLayer kernels.
# ---------------------------------------------------------------------
import pytest
pytestmark = pytest.mark.skip(
    reason="Pending migration to chart + GRAMMAR_LAYER_CLASSES surface; "
           "see doc/specs/2026-05-01-syntactic-layer-refactor.md")

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import torch
from util import TheXMLConfig, init_config
import Language
from Language import Grammar, SyntacticLayer, TheGrammar


class TestHeadEmission(unittest.TestCase):
    def setUp(self):
        try:
            self._prior_chart = TheXMLConfig.get('WordSpace.chartCompose')
        except KeyError:
            self._prior_chart = None
        TheXMLConfig.set('WordSpace.chartCompose', True)
        self._saved_grammar_state = TheGrammar._configured
        TheGrammar._configured = False

    def tearDown(self):
        if self._prior_chart is None:
            TheXMLConfig.set('WordSpace.chartCompose', False)
        else:
            TheXMLConfig.set('WordSpace.chartCompose', self._prior_chart)
        TheGrammar._configured = self._saved_grammar_state

    def test_emit_head_returns_best_codebook_idx(self):
        # Craft a codebook of 3 known atoms; the deep state exactly matches
        # atom 1 plus some noise. emit_head should return idx=1.
        from Spaces import Codebook
        cb = Codebook()
        cb.create(nInput=0, nVectors=3, nDim=4, customVQ=True,
                  monotonic=True, passThrough=False)
        # Replace the random codebook with three known atoms.
        with torch.no_grad():
            cb.getW().copy_(torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]))
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'S': ['not(S)']},
                              'generate': {'S': ['C']}})
        g = TheGrammar
        layer = SyntacticLayer(nInput=4, nOutput=4, rules=g.symbolic(),
                               max_depth=2, hidden_dim=16, grammar=g)
        # The deep state is close to atom 1 + a small residual.
        state = torch.tensor([[0.05, 0.9, 0.02, 0.01]])  # [B=1, D=4]
        best_idx, contained, residual = layer.emit_head(state, cb)
        self.assertEqual(best_idx.item(), 1)
        # Contained contribution should be a scalar multiple of atom 1.
        self.assertTrue(torch.allclose(contained[0, 1:2], torch.tensor([0.9]),
                                       atol=1e-4))
        # Residual must be smaller in norm than the original state.
        self.assertLess(residual.norm().item(), state.norm().item())

    def test_reconstruct_one_word_sentence(self):
        # End-to-end: a one-leaf state → emit_head gives the matching atom.
        from Spaces import Codebook
        cb = Codebook()
        cb.create(nInput=0, nVectors=2, nDim=3, customVQ=True,
                  monotonic=True, passThrough=False)
        with torch.no_grad():
            cb.getW().copy_(torch.tensor([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0]]))
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'S': ['not(S)']},
                              'generate': {'S': ['C']}})
        g = TheGrammar
        layer = SyntacticLayer(nInput=3, nOutput=3, rules=g.symbolic(),
                               max_depth=2, hidden_dim=16, grammar=g)
        state = torch.tensor([[0.2, 0.8, 0.0]])
        idx, _, _ = layer.emit_head(state, cb)
        self.assertEqual(idx.item(), 1)


class TestHeadPredictionCorpus(unittest.TestCase):
    """MVP: feed inline sentences; model predicts each sentence's head word.

    Untrained model correctness is unknown -- what the test fixes is the
    PLUMBING: forward runs, emit_head fires on the deep state, and
    ``MentalModel._predicted_head`` exposes a per-batch prediction the
    loss fn can reach.
    """

    def test_head_prediction_path_is_wired(self):
        import warnings
        import Models
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        init_config(path=os.path.join(data_dir, 'HeadEmission.xml'),
                    defaults_path=os.path.join(data_dir, 'model.xml'))
        Language.TheGrammar._configured = False
        model, _ = Models.MentalModel.from_config(
            os.path.join(data_dir, 'HeadEmission.xml'))
        sentences = ['the teacher helped the student']
        outputs = [torch.tensor([0.0])] * len(sentences)
        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:1])
            model.eval()
            with torch.no_grad():
                model.forward(x)
        head = getattr(model, '_predicted_head', None)
        self.assertIsNotNone(head)
        self.assertIsInstance(head, list)
        self.assertEqual(len(head), 1)


if __name__ == '__main__':
    unittest.main()
