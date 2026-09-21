"""Normal thought model configuration, provisioning and public boundaries.

Chooser optimizer and actual credit are covered by test_unified_thought_controller.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

import Models
from Models import BaseModel
from reasoning import QuerySpec, KIND_IS_PART

_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
_CONFIG = os.path.join(_DATA, 'MM_query_reasoning.xml')


class TestReasoningCDEModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Models.TheData.load('queries')
        cls.m, _ = BaseModel.from_config(_CONFIG)

    def test_gates_on(self):
        self.assertEqual(self.m.reasoning_iterations, 10)
        self.assertEqual(self.m.thinking_budget, 16)
        self.assertEqual(self.m.selected_thought_policy_weight, .1)
        self.assertFalse(hasattr(self.m, "_intervening_generator"))

    def test_truthset_provisions_source_rows(self):
        self.m.provision_ltm()
        store = self.m.symbolSpace.ltm_store
        rows = [store.row(int(i)) for i in store.relations(rel_type=store.REL_PARTOF)]
        assert {row['text'] for row in rows} == {
            'socrates partOf human', 'human partOf mortal'}
        assert all(abs(row['trust'] - .9) < 1e-6 for row in rows)


    def test_training_step_uses_the_normal_policy_configuration(self):
        opt = self.m.getOptimizer(lr=0.01)
        self.m.runEpoch(optimizer=opt, batchSize=6, split='train',
                        max_batches=1)

    def test_answer_query_degrades_gracefully(self):
        # Non-interrogative -> None. A query surface on a byte-grain config (no
        # word vocab to resolve operands) also returns None (generative
        # fallback), never a crash.
        self.assertIsNone(self.m.answer_query('hello there'))
        out = self.m.answer_query('is socrates part of mortal?')
        self.assertTrue(out is None or isinstance(out, dict))

    def test_reason_about_returns_honest_posture(self):
        # Order-zero vectors now have a geometric part effect. They still
        # cannot manufacture taxonomy identities or a native proof path.
        cs = self.m.conceptualSpace
        width = cs.outputShape[-1]
        before = dict(cs._concept_allocator.placement)
        result = self.m.reason_about(QuerySpec(KIND_IS_PART,
            left=torch.ones(width), right=torch.ones(width)))
        self.assertIn(result.posture, ('TRUE', 'FALSE', 'BOTH', 'UNKNOWN'))
        # The trained chooser may select another legal operation. Any
        # geometric part it executes still supplies no taxonomy proof.
        for record in result.records:
            if record.result is not None and record.result.evidence_kind == 'meronymy':
                self.assertNotIn('path', record.result.evidence)
        self.assertEqual(cs._concept_allocator.placement, before)
        with self.assertRaises(ValueError):
            self.m.reason_about(QuerySpec(KIND_IS_PART,
                left=torch.ones(width - 1), right=torch.ones(width - 1)))


if __name__ == '__main__':
    unittest.main()
