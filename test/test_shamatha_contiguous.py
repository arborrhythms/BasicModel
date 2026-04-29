"""Tests for the Contiguous() predicate on MentalModel.

Contiguous() returns True iff the active SymbolicSpace codebook rows
form a single connected mereological component under the
``ImpenetrableLayer._classify`` relations (anything other than
``disjoint`` counts as a connecting edge). Active rows are determined
by L2 norm > ``truthMinMagnitude`` (the activity floor on
SymbolicSpace).
"""

import os
import sys
import unittest

import torch

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from util import init_config

_CONFIG = os.path.join(_PROJECT, "data", "MM_shamatha.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _fresh_model():
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    m, _ = Models.MentalModel.from_config(_CONFIG)
    return m


def _set_codebook(m, W):
    """Overwrite SymbolicSpace's `what` codebook with W."""
    m.symbolicSpace.subspace.what.setW(W)


class TestContiguousPredicate(unittest.TestCase):
    def setUp(self):
        self.model = _fresh_model()
        # Pull the codebook width from the live basis to avoid hardcoding.
        W = self.model.symbolicSpace.subspace.what.getW()
        self.D = W.shape[-1]
        self.K = W.shape[0]
        # Strong activations so each row clears truthMinMagnitude.
        self.scale = 1.0

    def test_zero_active_rows_returns_true(self):
        """No active rows -> trivially one-pointed (degenerate True)."""
        zeros = torch.zeros(self.K, self.D)
        _set_codebook(self.model, zeros)
        self.assertTrue(self.model.Contiguous())

    def test_single_active_row_returns_true(self):
        """Exactly one active row -> trivially one component."""
        W = torch.zeros(self.K, self.D)
        W[0] = self.scale * torch.ones(self.D)
        _set_codebook(self.model, W)
        self.assertTrue(self.model.Contiguous())

    def test_two_disjoint_rows_returns_false(self):
        """Two rows on orthogonal supports -> two disjoint clusters -> False."""
        W = torch.zeros(self.K, self.D)
        # Row 0 lights only the first half; row 1 lights only the second
        # half. With monotonic parthood they are disjoint.
        half = self.D // 2
        W[0, :half] = self.scale
        W[1, half:] = self.scale
        _set_codebook(self.model, W)
        self.assertFalse(self.model.Contiguous())

    def test_two_overlapping_rows_returns_true(self):
        """Two rows whose supports overlap -> connected -> True."""
        W = torch.zeros(self.K, self.D)
        # Both rows light the leading dim; one extends into the next dim.
        W[0, 0] = self.scale
        W[0, 1] = self.scale
        W[1, 1] = self.scale
        W[1, 2] = self.scale
        _set_codebook(self.model, W)
        self.assertTrue(self.model.Contiguous())

    def test_inactive_rows_excluded(self):
        """Rows below the activity floor do not break connectivity."""
        W = torch.zeros(self.K, self.D)
        # Row 0 is fully active.
        W[0] = self.scale * torch.ones(self.D)
        # Row 1 is below the floor (norm ~0.001) -- should be excluded.
        W[1] = 1e-3 * torch.ones(self.D)
        _set_codebook(self.model, W)
        self.assertTrue(self.model.Contiguous())

    def test_chain_of_overlap_is_one_component(self):
        """A chain A-B-C where A overlaps B and B overlaps C (but A and C
        are disjoint) is still one connected component."""
        W = torch.zeros(self.K, self.D)
        # Pairwise overlap by sharing one dim with each neighbor.
        # Row 0: dims 0, 1. Row 1: dims 1, 2. Row 2: dims 2, 3.
        W[0, 0] = self.scale
        W[0, 1] = self.scale
        W[1, 1] = self.scale
        W[1, 2] = self.scale
        W[2, 2] = self.scale
        W[2, 3] = self.scale
        _set_codebook(self.model, W)
        self.assertTrue(self.model.Contiguous())


if __name__ == "__main__":
    unittest.main()
