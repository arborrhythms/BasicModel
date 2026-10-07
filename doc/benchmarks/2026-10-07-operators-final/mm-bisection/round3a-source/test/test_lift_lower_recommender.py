"""G1 (decode round-trip): lift/lower reverse routes the ``.what`` split through
the nearest-prototype recommender when a codebook basis is present -- recovering
real, DISTINCT constituents by recognition -- while preserving the partition-blind
balanced split (``L == R``) as the fallback when no basis is supplied.
"""
import os
import sys
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bin")
sys.path.insert(0, _BIN)

import unittest
import torch

from util import init_config
_DATA = os.path.join(os.path.dirname(_BIN), "data")
init_config(path=os.path.join(_DATA, "model.xml"),
            defaults_path=os.path.join(_DATA, "model.xml"))

from Language import LiftLayer, LowerLayer


class _StubBasis:
    """Minimal stand-in for a ``.what`` Basis: exposes ``getW()``."""
    def __init__(self, W):
        self._W = W

    def getW(self):
        return self._W


class TestLiftLowerRecommender(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.D = 8
        self.W = torch.tanh(torch.randn(6, self.D))   # distinct codebook rows

    def _check(self, layer):
        basis = _StubBasis(self.W)
        left = self.W[1].view(1, 1, self.D)
        right = self.W[4].view(1, 1, self.D)
        parent = layer.forward(left, right)
        # WITH basis: the recommender produces a split DIFFERENT from the
        # balanced fallback (it draws real constituents from the codebook).
        l, r = layer.reverse(parent, basis=basis)
        l0, r0 = layer.reverse(parent)                # no basis -> balanced split
        self.assertGreater(
            float((l - l0).abs().max() + (r - r0).abs().max()), 1e-3,
            "recommender path should differ from the balanced split")
        # The balanced fallback is UNCHANGED: L == R (byte-identical legacy).
        self.assertLess(float((l0 - r0).abs().max()), 1e-5)

    def test_lift_reverse_uses_recommender_with_basis(self):
        lift = LiftLayer(nInput=self.D)
        lift.eval()
        self._check(lift)

    def test_lower_reverse_uses_recommender_with_basis(self):
        lower = LowerLayer(nInput=self.D)
        lower.eval()
        self._check(lower)


if __name__ == "__main__":
    unittest.main()
