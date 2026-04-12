"""
Head Divergence Test — sentence vector vs. head word
=====================================================

Verifies that the sentence vector (top-of-stack after composition) does
not diverge wildly from the head word's pre-composition embedding.

The cosine angle between the composed sentence vector and the
pre-composition vector at the same slot should be less than 30 degrees
(cos > 0.866) for simple sentences with unambiguous heads.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import math
import warnings
import torch
import matplotlib
matplotlib.use('Agg')

from BasicModel import MentalModel, TheData, TheDevice
from util import init_config, TheXMLConfig

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# 30-degree threshold in radians and cosine
_MAX_ANGLE_DEG = 30
_MIN_COS = math.cos(math.radians(_MAX_ANGLE_DEG))  # ~0.866


def _reload_config():
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )


def _make_model():
    """Create a MentalModel with grammar enabled."""
    _reload_config()
    from Space import TheGrammar
    TheGrammar._configured = False
    model, _ = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
    return model


class TestHeadDivergence(unittest.TestCase):
    """Sentence vector should not diverge wildly from head word embedding."""

    def setUp(self):
        self.model = _make_model()
        self.sentences = [
            "Dogs run",
            "Birds fly",
            "Fish swim",
            "Cats sleep",
            "Rain falls",
        ]

    def _run_sentence(self, sentence):
        """Run a sentence through the model and return (pre_compose, post_compose, top_pos).

        Returns:
            pre_compose: [N, D] tensor of concept vectors before composition
            post_compose: [N, D] tensor of concept vectors after composition
            top_pos: int, top-of-stack position (sentence head)
        """
        # Stage the sentence as a runtime batch (outputs must be tensors)
        outputs = [torch.tensor([0.0])]
        with TheData.runtime_batch([sentence], outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")

            batch, _ = self.model.inputSpace.getBatch(0, 1, "runtime")
            if batch is None:
                return None, None, -1
            inputTensor, _ = batch

            # Forward through Input → Percept → Concept
            with torch.no_grad():
                inputs = self.model.inputSpace.forward(inputTensor)
                percepts = self.model.perceptualSpace.forward(inputs)
                percept_vecs = percepts.materialize()
                B = percept_vecs.shape[0]

                # Build concept input (same as forward() default path)
                symbols = torch.zeros(B, self.model._symbol_shape[0],
                                      self.model._symbol_shape[1],
                                      device=percept_vecs.device)
                concept_input = torch.cat([percept_vecs, symbols], dim=1)

                # Run ConceptualSpace forward (which saves _pre_compose)
                concepts = self.model.conceptualSpace.forward(
                    self.model._wrap_reverse(self.model.conceptualSpace, concept_input))

                post_compose = concepts.materialize()[0]  # [N, D] for batch 0
                pre_compose = getattr(self.model.conceptualSpace, '_pre_compose', None)
                if pre_compose is not None:
                    pre_compose = pre_compose[0]  # [N, D] for batch 0

                # Find top-of-stack
                tops = self.model.conceptualSpace.subspace.top_of_stack(
                    concepts.materialize())
                top_pos = tops[0] if tops else -1

        return pre_compose, post_compose, top_pos

    def test_head_divergence_within_threshold(self):
        """Sentence vector should be within 30 degrees of head word vector."""
        for sentence in self.sentences:
            with self.subTest(sentence=sentence):
                pre, post, top = self._run_sentence(sentence)
                if pre is None or top < 0:
                    self.skipTest(f"Could not process: {sentence}")

                pre_head = pre[top]
                post_head = post[top]

                # Skip zero vectors
                if pre_head.norm() < 1e-8 or post_head.norm() < 1e-8:
                    self.skipTest(f"Zero vector for: {sentence}")

                cos_sim = torch.nn.functional.cosine_similarity(
                    pre_head.unsqueeze(0), post_head.unsqueeze(0)).item()
                angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_sim))))

                self.assertGreater(
                    cos_sim, _MIN_COS,
                    f"'{sentence}': angle={angle_deg:.1f}° (cos={cos_sim:.4f}) "
                    f"exceeds {_MAX_ANGLE_DEG}° threshold")

    def test_pre_compose_snapshot_exists(self):
        """ConceptualSpace._pre_compose should be set after forward()."""
        self._run_sentence("Dogs run")
        self.assertTrue(
            hasattr(self.model.conceptualSpace, '_pre_compose'),
            "_pre_compose snapshot not saved by ConceptualSpace.forward()")


if __name__ == '__main__':
    unittest.main()
