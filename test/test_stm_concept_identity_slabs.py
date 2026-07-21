"""Exact lifecycle contract for STM concept identity metadata.

``ShortTermMemory`` keeps a durable concept-codebook row and its scalar
activation beside every payload slot.  The three slabs are one logical stack:
every insertion, shift, pop, resize, and clear must transform them in lockstep.
Unknown identity is represented only by ``row == -1, activation == 0``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch


os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parents[1]
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Layers import ShortTermMemory  # noqa: E402


def _stm(*, batch=2, capacity=4, dim=2):
    return ShortTermMemory(
        batch=batch, capacity=capacity, concept_dim=dim)


def _assert_invalid(stm, rows=None):
    selected_rows = stm._concept_rows if rows is None else stm._concept_rows[rows]
    selected_acts = (
        stm._concept_activations
        if rows is None else stm._concept_activations[rows]
    )
    assert bool((selected_rows == -1).all())
    assert torch.equal(selected_acts, torch.zeros_like(selected_acts))


def test_identity_slabs_initialize_and_begin_forward_invalid():
    stm = _stm(batch=2, capacity=4, dim=3)

    assert stm._concept_rows.shape == (2, 4)
    assert stm._concept_rows.dtype == torch.long
    assert stm._concept_activations.shape == (2, 4)
    assert stm._concept_activations.dtype == stm._buffer.dtype
    assert stm._concept_rows.device == stm._buffer.device
    assert stm._concept_activations.device == stm._buffer.device
    _assert_invalid(stm)

    stm.push(
        0, torch.ones(3), concept_row=17, concept_activation=0.75)
    stm.begin_forward(3, device=torch.device("cpu"), dtype=torch.float64)

    assert stm._concept_rows.shape == (3, 4)
    assert stm._concept_activations.shape == (3, 4)
    assert stm._concept_activations.dtype == torch.float64
    _assert_invalid(stm)


def test_push_and_push_step_shift_identity_slabs_with_payload():
    stm = _stm(batch=2, capacity=4, dim=2)
    stm.push(
        0, torch.tensor([1.0, 10.0]),
        concept_row=11, concept_activation=0.1)
    stm.push(
        0, torch.tensor([2.0, 20.0]),
        concept_row=12, concept_activation=0.2)

    assert torch.equal(stm._buffer[0, :2], torch.tensor([[2.0, 20.0],
                                                        [1.0, 10.0]]))
    assert stm._concept_rows[0].tolist() == [12, 11, -1, -1]
    torch.testing.assert_close(
        stm._concept_activations[0], torch.tensor([0.2, 0.1, 0.0, 0.0]))

    stm.push_step(
        torch.tensor([[3.0, 30.0], [4.0, 40.0]]),
        concept_row=torch.tensor([13, 24]),
        concept_activation=torch.tensor([0.3, 0.4]),
    )

    assert stm._concept_rows.tolist() == [
        [13, 12, 11, -1],
        [24, -1, -1, -1],
    ]
    torch.testing.assert_close(
        stm._concept_activations,
        torch.tensor([[0.3, 0.2, 0.1, 0.0],
                      [0.4, 0.0, 0.0, 0.0]]),
    )
    assert torch.equal(stm._buffer[0, :3], torch.tensor([
        [3.0, 30.0], [2.0, 20.0], [1.0, 10.0]]))


def test_push_step_masked_is_exact_noop_for_ungated_rows():
    stm = _stm(batch=2, capacity=4, dim=2)
    stm.push_step(
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        concept_row=torch.tensor([11, 22]),
        concept_activation=torch.tensor([0.1, 0.2]),
    )
    before = {
        "buffer": stm._buffer.clone(),
        "depth": stm._depth.clone(),
        "rows": stm._concept_rows.clone(),
        "activations": stm._concept_activations.clone(),
    }

    stm.push_step_masked(
        torch.tensor([[3.0, 30.0], [4.0, 40.0]]),
        torch.tensor([[True], [False]]),
        concept_row=torch.tensor([13, 24]),
        concept_activation=torch.tensor([0.3, 0.4]),
    )

    assert stm._concept_rows[0].tolist() == [13, 11, -1, -1]
    torch.testing.assert_close(
        stm._concept_activations[0], torch.tensor([0.3, 0.1, 0.0, 0.0]))
    assert torch.equal(stm._buffer[0, :2], torch.tensor([
        [3.0, 30.0], [1.0, 10.0]]))
    assert int(stm._depth[0]) == 2

    assert torch.equal(stm._buffer[1], before["buffer"][1])
    assert torch.equal(stm._depth[1], before["depth"][1])
    assert torch.equal(stm._concept_rows[1], before["rows"][1])
    assert torch.equal(
        stm._concept_activations[1], before["activations"][1])


def test_push_window_reverses_identity_metadata_with_payload():
    stm = _stm(batch=1, capacity=5, dim=2)
    stm.push(
        0, torch.tensor([1.0, 10.0]),
        concept_row=10, concept_activation=0.1)

    # The incoming window is oldest -> newest.  STM is newest-at-slot-zero,
    # so payload, row, and activation axes must all reverse together.
    stm.push_window_batch(
        torch.tensor([[[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]]),
        concept_rows=torch.tensor([[20, 30, 40]]),
        concept_activations=torch.tensor([[0.2, 0.3, 0.4]]),
    )

    assert torch.equal(stm._buffer[0, :4], torch.tensor([
        [4.0, 40.0], [3.0, 30.0], [2.0, 20.0], [1.0, 10.0]]))
    assert stm._concept_rows[0].tolist() == [40, 30, 20, 10, -1]
    torch.testing.assert_close(
        stm._concept_activations[0],
        torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0]),
    )


def test_pop_shifts_and_clears_identity_tail():
    stm = _stm(batch=1, capacity=4, dim=2)
    for marker, row, activation in ((1.0, 11, 0.1),
                                    (2.0, 12, 0.2),
                                    (3.0, 13, 0.3)):
        stm.push(
            0, torch.full((2,), marker),
            concept_row=row, concept_activation=activation)

    torch.testing.assert_close(stm.pop(0), torch.full((2,), 3.0))

    assert stm._concept_rows[0].tolist() == [12, 11, -1, -1]
    torch.testing.assert_close(
        stm._concept_activations[0], torch.tensor([0.2, 0.1, 0.0, 0.0]))
    assert torch.equal(stm._buffer[0, 2], torch.zeros(2))
    assert int(stm._depth[0]) == 2


def test_capacity_growth_preserves_prefix_and_full_push_fails_atomically():
    stm = _stm(batch=1, capacity=2, dim=2)
    stm.push(
        0, torch.tensor([1.0, 10.0]),
        concept_row=11, concept_activation=0.1)
    stm.push(
        0, torch.tensor([2.0, 20.0]),
        concept_row=12, concept_activation=0.2)
    before = {
        "buffer": stm._buffer.clone(),
        "depth": stm._depth.clone(),
        "rows": stm._concept_rows.clone(),
        "activations": stm._concept_activations.clone(),
    }

    with pytest.raises(RuntimeError, match="at capacity"):
        stm.push(
            0, torch.tensor([3.0, 30.0]),
            concept_row=13, concept_activation=0.3)

    assert torch.equal(stm._buffer, before["buffer"])
    assert torch.equal(stm._depth, before["depth"])
    assert torch.equal(stm._concept_rows, before["rows"])
    assert torch.equal(
        stm._concept_activations, before["activations"])

    stm.ensure_capacity(5)
    assert stm.capacity == 5
    assert stm._concept_rows[0].tolist() == [12, 11, -1, -1, -1]
    torch.testing.assert_close(
        stm._concept_activations[0],
        torch.tensor([0.2, 0.1, 0.0, 0.0, 0.0]),
    )
    assert torch.equal(stm._buffer[0, :2], before["buffer"][0])
    assert torch.equal(stm._buffer[0, 2:], torch.zeros(3, 2))


def test_clear_single_and_all_reset_identity_sentinels():
    stm = _stm(batch=2, capacity=3, dim=2)
    stm.push_step(
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        concept_row=torch.tensor([11, 22]),
        concept_activation=torch.tensor([0.1, 0.2]),
    )

    row1_before = (
        stm._buffer[1].clone(),
        stm._depth[1].clone(),
        stm._concept_rows[1].clone(),
        stm._concept_activations[1].clone(),
    )
    stm.clear(b=0)

    _assert_invalid(stm, 0)
    assert torch.equal(stm._buffer[0], torch.zeros_like(stm._buffer[0]))
    assert int(stm._depth[0]) == 0
    assert torch.equal(stm._buffer[1], row1_before[0])
    assert torch.equal(stm._depth[1], row1_before[1])
    assert torch.equal(stm._concept_rows[1], row1_before[2])
    assert torch.equal(stm._concept_activations[1], row1_before[3])

    stm.clear()
    assert torch.equal(stm._buffer, torch.zeros_like(stm._buffer))
    assert torch.equal(stm._depth, torch.zeros_like(stm._depth))
    _assert_invalid(stm)


def test_wholesale_buffer_replacement_invalidates_identity_metadata():
    stm = _stm(batch=2, capacity=3, dim=2)
    stm.push_step(
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        concept_row=torch.tensor([11, 22]),
        concept_activation=torch.tensor([0.1, 0.2]),
    )

    replacement = torch.full((2, 5, 2), 7.0, dtype=torch.float64)
    stm._buffer = replacement

    assert stm._concept_rows.shape == (2, 5)
    assert stm._concept_rows.dtype == torch.long
    assert stm._concept_rows.device == replacement.device
    assert stm._concept_activations.shape == (2, 5)
    assert stm._concept_activations.dtype == replacement.dtype
    assert stm._concept_activations.device == replacement.device
    _assert_invalid(stm)
