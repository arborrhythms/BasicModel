"""Audit per-batch setW/getW writes to Parameter-bearing slots.

Spec: ``doc/specs/2026-05-21-subspace-slot-architecture.md`` migration
section. Plan: ``doc/plans/2026-05-21-active-payload-retirement.md``.

This test is the migration metric: as writers are retargeted to
``set_activation`` / ``set_forward_content`` and readers move to
``materialize(mode=...)``, the counts drop to zero. Stage 4 flips
``setW`` into a hard raise, at which point these counts MUST stay at
zero or the test catches the regression.

Stage 0 / 0.2: fixtures pinned to the minimal codebook-event-write
exerciser's observed baseline. As Stages 1-3 retarget writers and
readers, those baselines drop and these constants get decremented in
the same commit. Stage 4 lands the strict raise and the constants
move to ``== 0``.
"""
import sys
import unittest
from pathlib import Path

import pytest
import torch

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "bin"))

from Spaces import Tensor, Codebook, ProjectionBasis, SubSpace  # noqa: E402


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def _is_per_batch(value):
    """A write is 'per-batch' when it's a 3-D-or-higher plain tensor.

    Excludes ``nn.Parameter`` (legitimate prototype writes during model
    construction) and ``None`` (cache clears).
    """
    return (isinstance(value, torch.Tensor)
            and not isinstance(value, torch.nn.Parameter)
            and value.ndim >= 3)


def _slot_holds_parameter(basis):
    """True iff this Basis owns a registered ``nn.Parameter`` on ``W``.

    Pre-migration the band-aid routes per-batch writes to
    ``_active_payload`` instead of clobbering this Parameter. Each such
    routing is one violation logged by the audit.
    """
    return ("W" in basis._parameters
            and basis._parameters["W"] is not None)


@pytest.fixture
def setw_audit(monkeypatch):
    """Patch each Basis subclass's ``setW`` to record per-batch writes.

    Each violation is appended as ``(class_name, tuple(value.shape))``.
    The original ``setW`` is still called so behavior is unchanged.
    """
    log = []
    for cls in (Tensor, Codebook, ProjectionBasis):
        orig = cls.setW

        def wrapped(self, value, _orig=orig, _cls=cls):
            if _is_per_batch(value) and _slot_holds_parameter(self):
                log.append((_cls.__name__, tuple(value.shape)))
            return _orig(self, value)
        monkeypatch.setattr(cls, "setW", wrapped)
    yield log


@pytest.fixture
def getw_audit(monkeypatch):
    """Patch each Basis subclass's ``getW`` to record per-batch reads
    against a Parameter-bearing slot.

    A pre-migration ``getW`` that returns a 3-D tensor from a slot that
    also holds a Parameter means ``_active_payload`` is shadowing the
    prototype — the exact anti-pattern the spec retires.
    """
    log = []
    for cls in (Tensor, Codebook, ProjectionBasis):
        orig = cls.getW

        def wrapped(self, _orig=orig, _cls=cls):
            r = _orig(self)
            if (isinstance(r, torch.Tensor) and r.ndim >= 3
                    and _slot_holds_parameter(self)):
                log.append((_cls.__name__, tuple(r.shape)))
            return r
        monkeypatch.setattr(cls, "getW", wrapped)
    yield log


# ---------------------------------------------------------------------------
# Representative forwards used to surface violations
# ---------------------------------------------------------------------------

def _codebook_bearing_subspace():
    """A SubSpace whose ``.event`` slot is a Codebook with a registered
    Parameter — the configuration where ``_active_payload`` shadows.

    Mirrors the PerceptualSpace MM_xor / MM_20M layout: muxed event
    holds the codebook prototype.
    """
    cb = Codebook()
    cb.create(nInput=4, nVectors=8, nDim=6)
    cb.addVectors(8)  # registers ``W`` as nn.Parameter
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6], object=cb)
    return sub


def _exercise_codebook_event_write(sub):
    """Drive a per-batch event write through the public setter API.

    Today this routes ``event.setW(per_batch)`` into ``_active_payload``;
    post-migration ``set_event`` on a codebook-bearing ``.event`` raises.
    """
    sub.set_event(torch.randn(2, 4, 6), compute_activation=False)


# ---------------------------------------------------------------------------
# Stage 0 baseline counts (pinned; decremented per migration stage)
# ---------------------------------------------------------------------------

# Post-Stage 4 contract: codebook-bearing slots REFUSE per-batch event
# writes (Codebook.setW raises on 3-D plain-tensor writes). Per-batch
# content reconstructs from prototype + selection (``codebook[_active]``)
# via ``SubSpace.materialize``. The ``_active_payload`` shadow on
# Tensor / Codebook / ProjectionBasis was retired.
#
# Both baselines are pinned at zero — any non-zero value here is a
# regression to the band-aid path.

BASELINE_SETW_PER_EVENT_WRITE = 0
BASELINE_GETW_PER_MATERIALIZE = 0


def test_set_event_on_codebook_snaps_through_codebook():
    """Spec: ``set_event`` / ``set_muxed`` on a muxed subspace snaps
    the per-batch tensor through the codebook (``Codebook.forward``)
    to produce the per-position selection, which lands on ``_active``.
    The selection IS the per-batch storage; ``materialize`` later
    reconstructs as ``codebook[_active]``.

    The Codebook's own ``set_event`` still raises if called directly —
    that's the band-aid path being retired; the spec contract is to
    go through ``SubSpace.set_muxed`` (or ``SubSpace.set_event``) which
    routes via the snap.
    """
    sub = _codebook_bearing_subspace()
    # No raise: set_event snaps through the codebook on .event.
    _exercise_codebook_event_write(sub)
    # Post-snap: _active holds per-position indices selected by the snap.
    assert sub._active is not None
    assert sub._active.ndim == 3 and sub._active.shape[-1] >= 1
    # Direct ``Basis.set_event`` on the codebook STILL raises — that's
    # the spec-aligned strict assertion. The SubSpace setter API
    # (``set_event`` / ``set_muxed``) is the legal write surface.
    with pytest.raises(RuntimeError, match="codebook-bearing"):
        sub.event.set_event(torch.randn(2, 4, 6))


def test_setw_audit_after_selection_write(setw_audit):
    """When the caller goes through the SPEC contract
    (``set_forward_content`` writes the selection; materialize
    reconstructs), zero per-batch setW writes land on the Parameter
    slot — the band-aid path is unreachable.
    """
    cb = Codebook()
    cb.create(nInput=4, nVectors=8, nDim=6)
    cb.addVectors(8)
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6], object=cb)
    # Spec-aligned write: per-position selection indices on ``_active``.
    indices = torch.randint(0, 8, (2, 4))  # [B=2, N=4]
    sub.set_forward_content(indices)
    # Materialize reconstructs from prototype + selection.
    recon = sub.materialize(mode="event")
    assert recon is not None and recon.ndim == 3
    # Zero per-batch setW writes touched the Parameter slot.
    assert len(setw_audit) == BASELINE_SETW_PER_EVENT_WRITE, (
        f"per-batch setW count drifted: {len(setw_audit)} != "
        f"{BASELINE_SETW_PER_EVENT_WRITE}.\n"
        f"sites: {setw_audit!r}")


def test_getw_audit_after_selection_write(getw_audit):
    """Materialize reconstruction reads ``self.W`` directly (Parameter
    access), NOT ``getW()`` — so the audit catches zero per-batch
    ``getW`` hits on the Parameter slot.
    """
    cb = Codebook()
    cb.create(nInput=4, nVectors=8, nDim=6)
    cb.addVectors(8)
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6], object=cb)
    indices = torch.randint(0, 8, (2, 4))
    sub.set_forward_content(indices)
    sub.materialize(mode="event")
    assert len(getw_audit) == BASELINE_GETW_PER_MATERIALIZE, (
        f"per-batch getW count drifted: {len(getw_audit)} != "
        f"{BASELINE_GETW_PER_MATERIALIZE}.\n"
        f"sites: {getw_audit!r}")


# ---------------------------------------------------------------------------
# SubSpace.codebook_slot / muxed / codebook() / lookup() — spec helpers
# ---------------------------------------------------------------------------

def test_codebook_slot_event_for_muxed_codebook():
    """Spec Slots table: Codebook on .event → codebook_slot == 'event';
    self.muxed is True; SubSpace.codebook() returns the .event slot."""
    cb = Codebook()
    cb.create(nInput=4, nVectors=8, nDim=6)
    cb.addVectors(8)
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6], object=cb)
    assert sub.codebook_slot == "event"
    assert sub.muxed is True
    assert sub.codebook() is sub.event
    proto = sub.prototype()
    assert proto is not None and proto.ndim == 2 and proto.shape[0] == 8


def test_codebook_slot_what_for_unmuxed_codebook():
    """Codebook on .what (e.g. SymbolicSpace) → codebook_slot == 'what'."""
    cb = Codebook()
    cb.create(nInput=4, nVectors=8, nDim=2)
    cb.addVectors(8)
    sub = SubSpace(inputShape=[4, 2], outputShape=[4, 2], what=cb)
    assert sub.codebook_slot == "what"
    assert sub.muxed is False
    assert sub.codebook() is sub.what


def test_codebook_slot_none_for_pure_event():
    """Plain Tensor on .event and .what → codebook_slot == None
    (pure-event configs: ConceptualSpace, InputSpace, OutputSpace)."""
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6])
    assert sub.codebook_slot is None
    assert sub.muxed is False
    assert sub.codebook() is None
    assert sub.prototype() is None


def test_subspace_lookup_replaces_getW_indexing():
    """SubSpace.lookup(indices) returns codebook[indices] — replaces the
    pre-spec ``codebook.getW()[indices]`` pattern that the audit fixture
    flags as a per-batch ``getW`` violation on Parameter-bearing slots."""
    cb = Codebook()
    cb.create(nInput=4, nVectors=8, nDim=6)
    cb.addVectors(8)
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6], object=cb)
    indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])  # [B=2, N=4]
    result = sub.lookup(indices)
    assert result.shape == (2, 4, 6)
    # Equivalent direct read for cross-check
    expected = cb.prototype()[indices]
    assert torch.equal(result, expected)


def test_pure_event_lookup_raises():
    """SubSpace.lookup() on a pure-event subspace raises with a clear
    spec-pointing message — callers must check ``muxed`` /
    ``codebook_slot`` before calling."""
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6])
    with pytest.raises(RuntimeError, match="pure-event"):
        sub.lookup(torch.zeros(2, 4, dtype=torch.long))


def test_codebook_forward_writes_selection_for_muxed_subspace():
    """Stage 3.2: Codebook.forward on a muxed destination writes the
    per-position selection to ``_active`` via ``set_forward_content``;
    ``materialize`` reconstructs from prototype + selection (no
    ``_active_payload`` shadow involved).
    """
    cb = Codebook()
    cb.create(nInput=4, nVectors=8, nDim=6)
    cb.addVectors(8)
    sub = SubSpace(inputShape=[4, 6], outputShape=[4, 6], object=cb)
    assert sub.muxed is True

    # Seed the SubSpace with a per-batch event so ``cb.forward(sub)``
    # has something to snap.
    sub.set_event(torch.randn(2, 4, 6), compute_activation=False)
    cb.forward(sub)

    # After forward: ``_active`` holds per-position selection.
    assert sub._active is not None
    assert sub._active.ndim == 3 and sub._active.shape[-1] >= 1
    sel = sub._active[:, :, 0]
    V = cb.prototype().shape[0]
    assert int(sel.max()) < V and int(sel.min()) >= 0

    # ``materialize(mode='event')`` reconstructs via prototype +
    # selection (the new muxed-codebook branch in materialize).
    recon = sub.materialize(mode="event")
    assert recon is not None
    assert recon.ndim == 3
    # The reconstruction equals the direct lookup — the spec contract.
    expected = sub.lookup(sel)
    assert torch.equal(recon, expected)


if __name__ == "__main__":
    unittest.main()
