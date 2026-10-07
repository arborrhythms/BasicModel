"""Runtime event carriers must not mirror codebook-capacity storage."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import torch
import torch.nn as nn

from Spaces import ConceptualSpace, PartSpace, Space, SubSpace, Tensor


_CAPACITY = 1_000_000
_WIDTH = 24


def _unbuilt_space(cls):
    """Provide only the fields consumed by the event-basis builder."""
    space = object.__new__(cls)
    nn.Module.__init__(space)
    space.nVectors = _CAPACITY
    space.muxedSize = _WIDTH
    space.ergodic = False
    space._codebook = "none"
    return space


def _assert_runtime_event_contract(basis):
    assert isinstance(basis, Tensor)
    assert basis.nVectors == _CAPACITY
    assert basis.nDim == _WIDTH
    assert basis.getW() is None
    assert "W" not in basis.state_dict()

    subspace = SubSpace(
        inputShape=(8, _WIDTH), outputShape=(8, _WIDTH),
        nInputDim=_WIDTH, nOutputDim=_WIDTH, object=basis)
    event = torch.randn(2, 8, _WIDTH)
    subspace.set_event(event)
    assert basis.getW() is event
    torch.testing.assert_close(subspace.materialize(mode="event"), event)

    subspace.Start()
    assert basis.getW() is None
    assert basis.forward(event) is event
    assert basis.getW() is event
    assert basis.reverse(event) is event


def test_partspace_capacity_does_not_allocate_passthrough_event_rows():
    ps = _unbuilt_space(PartSpace)
    _assert_runtime_event_contract(ps._build_object_basis())


def test_conceptualspace_capacity_does_not_allocate_passthrough_event_rows():
    cs = _unbuilt_space(ConceptualSpace)
    # ConceptualSpace intentionally inherits the pure-event builder.
    assert cs.__class__._build_object_basis is Space._build_object_basis
    _assert_runtime_event_contract(cs._build_object_basis())


def test_tensor_eager_default_is_unchanged():
    basis = Tensor(nVectors=3, nDim=5)
    assert tuple(basis.getW().shape) == (3, 5)
    assert torch.count_nonzero(basis.getW()).item() == 0
