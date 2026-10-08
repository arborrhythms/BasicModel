"""Normal thought model configuration, provisioning and public boundaries.

Chooser optimizer and actual credit are covered by test_unified_thought_controller.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch
import pytest
from fineweb_artifacts import fineweb_checkpoint, fineweb_trained_model

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
        # Public-boundary and optimizer mechanisms use a supplied reading.
        # The separate provisioning learning assertion requires its checkpoint.
        from reading_fixtures import force_absolute_reading
        force_absolute_reading(cls.m)

    @pytest.fixture(autouse=True)
    def trained_provisioning(self, request):
        if request.node.name == 'test_truthset_provisions_source_rows':
            self.m = request.getfixturevalue('fineweb_trained_model')

    def test_gates_on(self):
        self.assertEqual(self.m.attention_budget, 16)
        self.assertEqual(self.m.selected_thought_policy_weight, 0.)
        self.assertFalse(hasattr(self.m, 'selected_thought_choosers'))
        self.assertFalse(hasattr(self.m, "_intervening_generator"))

    def test_truthset_provisions_source_rows(self):
        store = self.m.symbolSpace.ltm_store
        # This is a learning assertion, with no supplied relation annotation.
        written = self.m._ltm_ingest_truth_texts(store,
            ['socrates is a human', 'humans are mortal'], trusts=[.9, .9],
            origin=store.ORIGIN_PROVISIONED)
        selected = {i for group in written for i in group}
        rows = [store.row(int(i)) for i in store.relations(rel_type=store.REL_PARTOF)
                if int(i) in selected]
        assert {row['text'] for row in rows} == {
            'socrates is a human', 'humans are mortal'}
        assert all(abs(row['trust'] - .9) < 1e-6 for row in rows)


    @pytest.mark.slow
    @pytest.mark.usefixtures("eager_reading")
    def test_training_step_uses_the_normal_policy_configuration(self):
        opt = self.m.getOptimizer(lr=0.01)
        self.m.runEpoch(optimizer=opt, batchSize=6, split='train',
                        max_batches=1)

    @pytest.mark.usefixtures('eager_reading')
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
        # The symbolic face requires native names; the conceptual face is
        # available as part over codes and cannot manufacture a taxonomy path.
        with self.assertRaisesRegex(TypeError, 'reference'):
            self.m.reason_about(QuerySpec(KIND_IS_PART,
                left=torch.ones(width), right=torch.ones(width)))
        self.assertEqual(cs._concept_allocator.placement, before)
        with self.assertRaises((ValueError, TypeError)):
            self.m.reason_about(QuerySpec(KIND_IS_PART,
                left=torch.ones(width - 1), right=torch.ones(width - 1)))


if __name__ == '__main__':
    unittest.main()
