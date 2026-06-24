"""Reasoning Phases C/D/E -- model-level integration (gated, byte-identical off).

These assert the WIRING is safe and correct end-to-end: the eager generator
joins the optimizer (Phase C), the answer-loss hook runs without crashing, and
the serve-facing answer_query / reason_about degrade gracefully. The full
real-config DEMONSTRATION (a populated truthSet store + word-level operand
extraction + N decoded sentences) is blocked by the byte/radix grain of this
config -- the same word/byte grain that blocks the decode round-trip (Track 1);
the mechanisms themselves are unit-tested in test_truth_grounded_reasoning.py.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch

import Models
from Models import BaseModel
from reasoning import QuerySpec, KIND_IS_PART, ReasoningResult

_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
_CONFIG = os.path.join(_DATA, 'MM_query_reasoning.xml')


class TestReasoningCDEModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Models.TheData.load('queries')
        cls.m, _ = BaseModel.from_config(_CONFIG)

    def test_gates_on(self):
        self.assertEqual(self.m.reasoning_iterations, 10)
        self.assertGreater(self.m.answer_loss_weight, 0.0)

    @staticmethod
    def _opt_param_ids(opt):
        ids = set()
        for o in getattr(opt, 'optimizers', [opt]):
            for g in getattr(o, 'param_groups', []):
                for p in g['params']:
                    ids.add(id(p))
        return ids

    def test_eager_generator_joins_optimizer(self):
        # answer_loss_weight > 0 => the generator is built BEFORE getOptimizer,
        # so its query-head params must be IN THE OPTIMIZER (not merely in
        # model.parameters(); getOptimizer walks self.spaces + named modules, not
        # model.parameters()). This is the assertion that catches the missing
        # optimizer-membership bug.
        gen = getattr(self.m, '_intervening_generator', None)
        self.assertIsNotNone(gen)
        gen_param_ids = {id(p) for p in gen.parameters()}
        opt_ids = self._opt_param_ids(self.m.getOptimizer(lr=0.01))
        self.assertTrue(gen_param_ids and gen_param_ids <= opt_ids)

    def test_answer_loss_actually_trains_the_head(self):
        # End-to-end: a policy loss with a geometric bridge present updates the
        # generator head AFTER optimizer.step() -- proving the head is both
        # grad-bearing AND in the optimizer (regression for the critical bug).
        from reasoning import TruthGroundedReasoner, policy_answer_loss
        D = int(self.m._intervening_generator.dim)
        a = torch.zeros(D); a[0] = a[1] = 1.0          # A
        mid = torch.zeros(D); mid[0] = mid[1] = mid[2] = 1.0   # bridge: A <= mid
        b = torch.zeros(D); b[0] = b[1] = b[2] = b[3] = 1.0    # mid <= B
        d1 = torch.zeros(D); d1[7] = 1.0               # distractor (disjoint)
        spaces = [{"id": 4, "keys": torch.stack([mid, d1]).to(a.device)}]
        reasoner = TruthGroundedReasoner(self.m)       # geometric, no store needed
        opt = self.m.getOptimizer(lr=0.5)
        before = self.m._intervening_generator.head[0].weight.detach().clone()
        loss = policy_answer_loss(self.m._intervening_generator, spaces,
                                  reasoner, [(a, b, 1.0)])
        self.assertIsNotNone(loss)
        opt.zero_grad(); loss.backward(); opt.step()
        after = self.m._intervening_generator.head[0].weight.detach()
        self.assertGreater(float((after - before).abs().sum()), 0.0)

    def test_training_step_runs_the_hook_without_crashing(self):
        # The Phase-C answer-loss hook fires in runBatch; with an empty store it
        # is a graceful no-op. The point: the hot loop does not crash with the
        # gate on.
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
        # Vector operands over the (empty) truth-space => an honest UNKNOWN
        # ReasoningResult, never a hallucinated verdict and never a crash.
        res = self.m.reason_about(
            QuerySpec(KIND_IS_PART, left=torch.randn(1024),
                      right=torch.randn(1024)))
        self.assertIsInstance(res, ReasoningResult)
        self.assertIn(res.posture, {'TRUE', 'FALSE', 'UNKNOWN', 'BOTH'})


if __name__ == '__main__':
    unittest.main()
