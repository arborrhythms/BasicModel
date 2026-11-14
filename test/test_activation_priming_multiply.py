"""Tests for Phase C+D: ``.activation`` post-snap multiplication and
Phase D lifecycle hooks + config + telemetry.

Plan: doc/plans/2026-05-20-primed-reverse-generation.md
§Application — B. After snap.
§Staging — Phase D — Lifecycle + configuration + telemetry.
"""
import sys
from pathlib import Path

import pytest
import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))








# -- Basis.apply_priming ---------------------------------------------------


def _bare_basis():
    """A bare Basis instance for testing apply_priming."""
    from Spaces import Basis
    import torch.nn as nn
    b = object.__new__(Basis)
    nn.Module.__init__(b)
    b.activation = None
    b.nDim = 0
    b.nInput = 0
    b.nVectors = 0
    b.monotonic = True
    b.ergodic = False
    return b


def test_apply_priming_multiplies_activation():
    """activation * priming_mask, element-wise."""
    b = _bare_basis()
    b.activation = torch.tensor([[0.3, 0.5, 0.2]])
    pm = torch.tensor([2.0, 1.0, 1.5])
    b.apply_priming(pm)
    assert torch.allclose(b.activation, torch.tensor([[0.6, 0.5, 0.3]]))


def test_apply_priming_identity_is_noop():
    """All-1.0 priming leaves activation unchanged."""
    b = _bare_basis()
    b.activation = torch.tensor([[0.3, 0.5, 0.2]])
    before = b.activation.clone()
    b.apply_priming(torch.ones(3))
    assert torch.equal(b.activation, before)


def test_apply_priming_none_priming_is_noop():
    """None priming is a no-op (no crash)."""
    b = _bare_basis()
    b.activation = torch.tensor([[0.3, 0.5, 0.2]])
    before = b.activation.clone()
    b.apply_priming(None)
    assert torch.equal(b.activation, before)


def test_apply_priming_none_activation_is_noop():
    """No activation → no-op."""
    b = _bare_basis()
    # activation is None
    b.apply_priming(torch.tensor([2.0, 1.0, 1.5]))
    assert b.activation is None


def test_apply_priming_shape_mismatch_is_noop():
    """Mismatched trailing shape → no-op (safety, not silent broadcast
    over the wrong axis)."""
    b = _bare_basis()
    b.activation = torch.tensor([[0.3, 0.5, 0.2]])
    before = b.activation.clone()
    b.apply_priming(torch.tensor([2.0, 1.0]))   # length 2, activation length 3
    assert torch.equal(b.activation, before)


def test_apply_priming_per_batch_broadcast():
    """priming_mask: [B, V] is multiplied per-row."""
    b = _bare_basis()
    b.activation = torch.tensor([[0.3, 0.5], [0.2, 0.4]])
    pm = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    b.apply_priming(pm)
    assert torch.allclose(b.activation,
                          torch.tensor([[0.6, 0.5], [0.2, 0.8]]))


# -- Config + telemetry ----------------------------------------------------














# -- Lifecycle: soft_reset clears priming ---------------------------------
