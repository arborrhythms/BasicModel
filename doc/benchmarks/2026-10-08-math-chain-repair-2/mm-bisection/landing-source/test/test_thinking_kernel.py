"""Evidence values and grammar depth after retirement of the frame kernel."""
import os
import unittest
import torch
import pytest
from fineweb_artifacts import fineweb_checkpoint, fineweb_trained_model



class TestDepth3RelativeEndState(unittest.TestCase):
    """Learning campaign over a mature checkpoint, never an untrained grammar.

    The original depth-3 assertion remains the gate. Missing item-9 training
    artifacts are an explicit prerequisite, not an instruction to train 1M
    sentences as part of this test.
    """

    @pytest.fixture(autouse=True)
    def mature_checkpoint(self, fineweb_trained_model):
        self.m = fineweb_trained_model

    def test_first_trained_read_reaches_depth3_end_state(self):
        m = self.m
        depths = []
        original = m._clause_end_state

        def spy(state, sid, clauses, row_ids):
            out = original(state, sid, clauses, row_ids)
            depths.append(out[0][1].detach().reshape(-1).tolist())
            return out

        m._clause_end_state = spy
        try:
            m._ltm_ingest_truth_texts(m.symbolSpace.ltm_store,
                ['socrates is a human', 'humans are mortal'], trusts=[.9, .9],
                origin=m.symbolSpace.ltm_store.ORIGIN_PROVISIONED)
        finally:
            del m._clause_end_state
        flat = [x for row in depths for x in row]
        self.assertIn(3, flat,
                      f"no depth-3 relative end-state in sweeps: {flat}")
        # ConceptualSpace owns the per-position pid grid for conditioning.
        self.assertIsNotNone(
            getattr(m._concept_owner(), '_category_last_pid', None),
            "word-grain autobind must stash the per-position pid grid")


if __name__ == "__main__":
    unittest.main()
