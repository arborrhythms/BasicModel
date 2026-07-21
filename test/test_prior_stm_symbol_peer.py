"""Canonical prior-tick SymbolSpace peer for aligned concept binding."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from Layers import ShortTermMemory  # noqa: E402
from Spaces import ConceptualSpace, WholeSpace  # noqa: E402


class _IndexedRows(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.rows = nn.Parameter(rows.clone())
        self.calls = []

    def lookup_rows(self, indices):
        self.calls.append(indices.detach().clone())
        return self.rows[indices.long()]


class _Carrier:
    def __init__(self, event):
        self.event = event

    def is_empty(self):
        return self.event is None

    def materialize(self):
        return self.event

    def set_event(self, event):
        self.event = event

    def copy_context(self, _other):
        return None


def _bare_cs(n_locations=8, n_rows=16):
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.nWhat = 4
    cs.nWhere = 2
    cs.nWhen = 2
    cs.concept_dim = 8
    cs.inputShape = [n_locations, 8]
    cs.outputShape = [n_locations, 8]
    rows = torch.zeros(n_rows, 8)
    for row in range(n_rows):
        rows[row, :4] = float(row)
        rows[row, 4:] = 999.0  # bands must come from STM, not the codebook
    cs.similarity_codebook = _IndexedRows(rows)
    return cs


def _full_stm():
    stm = ShortTermMemory(batch=1, capacity=8, concept_dim=8)
    # Push in reverse so final slot i has exact row i+1.
    for row in range(8, 0, -1):
        idea = torch.zeros(1, 8)
        idea[:, 4:] = float(100 + row)
        stm.push_step(
            idea, concept_row=torch.tensor([row]),
            concept_activation=torch.tensor([row / 10.0]))
    return stm


def test_full_eight_slot_prior_slab_is_one_indexed_ss_source():
    cs = _bare_cs()
    stm = _full_stm()

    decoded, valid = cs.decode_prior_stm_peer(
        stm, torch.tensor([True]))

    assert decoded.shape == (1, 8, 8)
    assert bool(valid.all())
    assert len(cs.similarity_codebook.calls) == 1
    assert cs.similarity_codebook.calls[0].numel() == 8
    assert cs.similarity_codebook.calls[0].tolist() == [
        [1, 2, 3, 4, 5, 6, 7, 8]]
    for location, row in enumerate(range(1, 9)):
        assert torch.allclose(
            decoded[0, location, :4],
            torch.full((4,), row * (row / 10.0)))
        assert torch.equal(
            decoded[0, location, 4:],
            torch.full((4,), float(100 + row)))


def test_ss_is_prior_tick_and_never_sees_a_later_push():
    cs = _bare_cs()
    stm = ShortTermMemory(batch=1, capacity=8, concept_dim=8)
    stm.push_step(
        torch.zeros(1, 8), concept_row=torch.tensor([3]),
        concept_activation=torch.tensor([1.0]))

    prior, prior_valid = cs.decode_prior_stm_peer(
        stm, torch.tensor([True]))
    stm.push_step(
        torch.zeros(1, 8), concept_row=torch.tensor([4]),
        concept_activation=torch.tensor([1.0]))
    following, following_valid = cs.decode_prior_stm_peer(
        stm, torch.tensor([True]))

    assert prior_valid[0, 0] and not bool(prior_valid[0, 1:].any())
    assert torch.equal(prior[0, 0, :4], torch.full((4,), 3.0))
    # The already-decoded prior source is immutable; the new row first appears
    # in the following call/tick at location zero.
    assert torch.equal(prior[0, 0, :4], torch.full((4,), 3.0))
    assert following_valid[0, 0] and following_valid[0, 1]
    assert torch.equal(following[0, 0, :4], torch.full((4,), 4.0))
    assert torch.equal(following[0, 1, :4], torch.full((4,), 3.0))


def test_unknown_or_inactive_slots_zero_both_what_and_band():
    cs = _bare_cs()
    stm = ShortTermMemory(batch=2, capacity=8, concept_dim=8)
    stm.push_step_masked(
        torch.ones(2, 8), torch.tensor([[True], [True]]),
        concept_row=torch.tensor([5, 6]),
        concept_activation=torch.tensor([0.5, 0.6]))
    # An unknown identity cannot retain an activation, even if a caller tries.
    stm.push_step_masked(
        torch.full((2, 8), 7.0), torch.tensor([[True], [False]]),
        concept_row=torch.tensor([-1, 9]),
        concept_activation=torch.tensor([42.0, 0.9]))
    assert int(stm._concept_rows[0, 0]) == -1
    assert float(stm._concept_activations[0, 0]) == 0.0

    decoded, valid = cs.decode_prior_stm_peer(
        stm, torch.tensor([True, False]))

    assert not bool(valid[0, 0])
    assert valid[0, 1]
    assert not bool(valid[1].any())
    assert torch.equal(decoded[0, 0], torch.zeros(8))
    assert torch.equal(decoded[1], torch.zeros_like(decoded[1]))


def test_prior_peer_requires_canonical_stm_location_geometry():
    cs = _bare_cs(n_locations=8)
    stm = ShortTermMemory(batch=1, capacity=7, concept_dim=8)
    with pytest.raises(RuntimeError, match="STM capacity 7 != CS locations 8"):
        cs.decode_prior_stm_peer(stm, torch.tensor([True]))


def test_seventh_source_is_location_masked_and_unbind_ignores_ss():
    cs = _bare_cs()
    part = [torch.full((1, 8, 8), float(v)) for v in (1, 2, 3)]
    whole = [torch.full((1, 8, 8), float(v)) for v in (4, 5, 6)]
    symbol = torch.full((1, 8, 8), 999.0)
    symbol[:, 0] = 14.0
    valid = torch.zeros(1, 8, dtype=torch.bool)
    valid[:, 0] = True
    out = _Carrier(torch.zeros(1, 8, 8))

    carrier = cs.bind_fold_streams(
        part, whole, out, part_passes=(0, 1, 2),
        whole_passes=(0, 1, 2), symbol_source=symbol,
        symbol_validity=valid)

    assert carrier.shape == (1, 7, 8, 8)
    assert torch.equal(
        out.materialize()[:, 0], torch.full((1, 8), 5.0))
    # No fixed-seven dilution: absent SS leaves mean(1..6) == 3.5.
    assert torch.equal(
        out.materialize()[:, 1], torch.full((1, 8), 3.5))
    assert out._fold_support["source_count"] == 7
    assert out._fold_support["symbol_sources"] == [{
        "kind": "prior_stm",
        "prior_tick": True,
        "location_aligned": True,
    }]
    assert torch.equal(
        out._aligned_source_validity[:, -1], valid)
    assert WholeSpace._normalize_fold_support(
        out._fold_support)["source_count"] == 7
    assert WholeSpace._normalize_fold_support(
        out._fold_support)["symbol_sources"][0]["prior_tick"] is True

    recovered_part, recovered_whole = cs.unbind(out)
    assert torch.equal(recovered_part, torch.full((1, 8, 8), 2.0))
    assert torch.equal(recovered_whole, torch.full((1, 8, 8), 5.0))
