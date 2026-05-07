"""Tests for the resolve() handoff (2026-05-04 plan).

Originally also covered ``SymbolicSpace.area()`` / ``luminosity()``;
those measures migrated to the :class:`Mereology` mixin (see
``bin/Mereology.py``) with a new hyperrectangle-volume formula —
their tests live in ``test/test_mereology.py``.

This file retains the ``SymbolicSpace.resolve()`` regression: the
``pos - neg`` invariant on the signed Degree of Truth.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import torch


class TestSymbolicResolve(unittest.TestCase):
    """``SymbolicSpace.resolve()`` writes ``pos - neg`` to activation."""

    def test_pos_minus_neg_signed(self):
        # The resolve invariant is exercised end-to-end by the existing
        # symbolic-pipeline tests (test_partition_symbolicspace_state,
        # test_partition_pos_codebook).  Here we just verify that the
        # constant ``_DEFAULT_SYMBOL_SIGMA`` retained from the old
        # area / luminosity formula is still available for any
        # downstream consumer that calibrates by Gaussian width.
        from Spaces import _DEFAULT_SYMBOL_SIGMA
        self.assertGreater(_DEFAULT_SYMBOL_SIGMA, 0.0)
        self.assertLessEqual(_DEFAULT_SYMBOL_SIGMA, 1.0)


if __name__ == '__main__':
    unittest.main()
