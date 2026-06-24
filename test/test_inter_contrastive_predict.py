"""End-to-end: the next-idea predictor (MSE L_inter + InfoNCE contrastive)
trains on MM_sequence_predict.xml. Guards the load-bearing precondition found
2026-06-23: the discourse end-state chain only spans sentences when (a) the BYTE
cursor keeps the document in one stream and (b) each document EXCEEDS the byte
slab width (InputSpace nObj ~1024) so it is walked over multiple ticks (one
end-state per tick). A short document collapses to one tick / one end-state and
the next-idea loss never fires.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest

import Models
from Models import BaseModel

_CONFIG = os.path.join(os.path.dirname(__file__), '..', 'data',
                       'MM_sequence_predict.xml')


class TestInterContrastivePredict(unittest.TestCase):
    def test_predictor_trains_end_to_end(self):
        Models.TheData.load('sequences')
        m, _ = BaseModel.from_config(_CONFIG)
        disc = m.symbolSpace.discourse
        self.assertIsNotNone(disc, "config must build the discourse predictor")
        self.assertGreater(disc._inter_contrastive_weight, 0.0)
        # Precondition (b): documents must exceed the byte slab width so the
        # chain spans multiple ticks (else the next-idea loss never fires).
        self.assertGreater(len(Models.TheData.train_input[0]), 1024)

        opt = m.getOptimizer(lr=0.5)
        before = [p.detach().clone()
                  for p in disc._inter_predictor.parameters()]
        m.runEpoch(optimizer=opt, batchSize=4, split='train', max_batches=2)
        after = list(disc._inter_predictor.parameters())
        changed = any(float((a - b).abs().sum()) > 0.0
                      for a, b in zip(after, before))
        self.assertTrue(
            changed, "the next-idea predictor must train end-to-end "
                     "(MSE L_inter + InfoNCE contrastive over the spanned chain)")


if __name__ == '__main__':
    unittest.main()
