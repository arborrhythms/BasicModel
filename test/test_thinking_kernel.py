"""Evidence values and grammar depth after retirement of the frame kernel."""
import os
import unittest
import torch



class TestDepth3RelativeEndState(unittest.TestCase):
    """Depth-3 routing campaign (2026-07-13): with the syntactic anchors,
    the category conditioning (autobind pid stash on the word-whole arm),
    the CS-role ANCHOR-GATED relative mask, the O1 prior term, and the
    mid-read relative protection all in force, a relative truth text read
    AFTER the anchors registered (they land at the provisioning texts'
    boundary Resets — engagement starts from the next read) must stop its
    reduce sweep at the depth-3 end-state instead of the depth-1 absolute
    collapse."""

    def test_first_trained_read_reaches_depth3_end_state(self):
        import random

        import numpy as np

        import Models
        from Models import BaseModel

        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        _DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
        Models.TheData.load('queries')
        m, _ = BaseModel.from_config(
            os.path.join(_DATA, 'MM_query_reasoning.xml'))

        depths = []
        orig = Models.BasicModel._stm_reduce_to_single_S

        def spy(self, *a, **k):
            out = orig(self, *a, **k)
            d = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            if torch.is_tensor(d):
                depths.append(d.detach().reshape(-1).tolist())
            return out

        Models.BasicModel._stm_reduce_to_single_S = spy
        try:
            m.provision_ltm()          # anchors register at the boundaries
            opt = m.getOptimizer(lr=0.01)
            m.runEpoch(optimizer=opt, batchSize=6, split="train")
        finally:
            Models.BasicModel._stm_reduce_to_single_S = orig
        flat = [x for row in depths for x in row]
        self.assertIn(3, flat,
                      f"no depth-3 relative end-state in sweeps: {flat}")
        # The mechanism behind the flip: the word-whole autobind arm must
        # stash the per-position pid grid (the chooser conditioning's
        # precondition -- pre-fix it returned before the category block).
        self.assertIsNotNone(
            getattr(m.wholeSpace, '_category_last_pid', None),
            "word-grain autobind must stash the per-position pid grid")


if __name__ == "__main__":
    unittest.main()
