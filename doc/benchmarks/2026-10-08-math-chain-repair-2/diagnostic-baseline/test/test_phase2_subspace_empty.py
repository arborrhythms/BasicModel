"""Unit tests for Phase 2 SubSpace.is_empty()."""
import sys
from pathlib import Path

# basicmodel/bin must come before bin/ so that basicmodel's parse.py is found
# first (both directories contain a parse.py with different APIs).  Insert in
# reverse order so basicmodel/bin ends up at index 0.
_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

from Spaces import SubSpace


@pytest.fixture
def subspace_factory():
    def _build(n_vectors, d=4):
        inputShape = (n_vectors, d)
        outputShape = (n_vectors, d)
        return SubSpace(inputShape, outputShape, nInputDim=d, nOutputDim=d)
    return _build


def test_is_empty_true_for_zero_n(subspace_factory):
    ws = subspace_factory(n_vectors=0)
    assert ws.is_empty() is True


def test_is_empty_false_for_nonzero_n(subspace_factory):
    ws = subspace_factory(n_vectors=4)
    ws.set_muxed(torch.zeros(2, 4, ws.muxedSize))
    assert ws.is_empty() is False


def test_is_empty_true_for_zero_batch(subspace_factory):
    ws = subspace_factory(n_vectors=4)
    ws.set_muxed(torch.zeros(0, 4, ws.muxedSize))
    assert ws.is_empty() is True, "B=0 is 'nothing to do'"
