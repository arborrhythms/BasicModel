"""Grammar transforms preserve or invalidate exact STM references."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn


os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(ROOT / "bin"))

from test_word_loop_buckets_and_orders import _grammar_model  # noqa: E402


class _Unary(nn.Module):
    def __init__(self, apply):
        super().__init__()
        self.apply = bool(apply)

    def forward(self, x):
        candidate = x + 0.25
        mask = torch.full(
            (x.shape[0], x.shape[1], 1), float(self.apply),
            dtype=x.dtype, device=x.device)
        return candidate, candidate, {"apply_mask": mask}


class _Binary(nn.Module):
    def __init__(self, copy_source=None):
        super().__init__()
        self.copy_source = copy_source

    def forward(self, x):
        B, _, D = x.shape
        if self.copy_source is None:
            parent = (x[:, 0] - x[:, 1]).unsqueeze(1)
            kind = torch.ones(B, 1, dtype=torch.long, device=x.device)
            source = torch.zeros(B, 1, dtype=torch.long, device=x.device)
        else:
            parent = x[:, self.copy_source].unsqueeze(1)
            kind = torch.zeros(B, 1, dtype=torch.long, device=x.device)
            source = torch.full(
                (B, 1), int(self.copy_source), dtype=torch.long,
                device=x.device)
        routing = {"action_kind": kind, "src_left": source}
        return parent, parent, routing


def _with_cached(model, name, value, fn):
    sentinel = object()
    old = getattr(model, name, sentinel)
    object.__setattr__(model, name, value)
    try:
        return fn()
    finally:
        if old is sentinel:
            delattr(model, name)
        else:
            object.__setattr__(model, name, old)


def _seed_two(model):
    stm = model.conceptualSpace.stm
    stm.begin_forward(1, device=torch.device("cpu"))
    D = int(stm.concept_dim)
    gate = torch.ones(1, 1, dtype=torch.bool)
    stm.push_step_masked(
        torch.full((1, D), 1.0), gate,
        concept_row=torch.tensor([11]),
        concept_activation=torch.tensor([0.1]))
    stm.push_step_masked(
        torch.full((1, D), 2.0), gate,
        concept_row=torch.tensor([22]),
        concept_activation=torch.tensor([0.2]))
    return stm


def test_unary_apply_invalidates_and_noop_preserves_reference():
    model = _grammar_model()
    stm = _seed_two(model)

    applied = _with_cached(
        model, "_stm_unary_rewriter_cached", _Unary(True),
        lambda: model._stm_bounded_unary_step())
    assert bool(applied.item())
    assert int(stm._concept_rows[0, 0]) == -1
    assert float(stm._concept_activations[0, 0]) == 0.0
    assert int(stm._concept_rows[0, 1]) == 11

    stm = _seed_two(model)
    before_rows = stm._concept_rows.clone()
    before_activations = stm._concept_activations.clone()
    applied = _with_cached(
        model, "_stm_unary_rewriter_cached", _Unary(False),
        lambda: model._stm_bounded_unary_step())
    assert not bool(applied.item())
    assert torch.equal(stm._concept_rows, before_rows)
    assert torch.equal(stm._concept_activations, before_activations)


def test_binary_transform_invalidates_parent_and_shifts_untouched_tail():
    model = _grammar_model()
    stm = _seed_two(model)
    D = int(stm.concept_dim)
    stm.push_step_masked(
        torch.full((1, D), 3.0), torch.ones(1, 1, dtype=torch.bool),
        concept_row=torch.tensor([33]),
        concept_activation=torch.tensor([0.3]))

    reduced = _with_cached(
        model, "_stm_reducer_cached", _Binary(),
        lambda: model._stm_bounded_reduce_step())

    assert bool(reduced.item())
    assert stm._concept_rows[0, :3].tolist() == [-1, 11, -1]
    torch.testing.assert_close(
        stm._concept_activations[0, :3],
        torch.tensor([0.0, 0.1, 0.0]))


def test_binary_explicit_copy_preserves_selected_operand_reference():
    model = _grammar_model()
    stm = _seed_two(model)
    # Reducer input index 0 is old slot 1 (row 11); index 1 is old slot 0
    # (row 22). A hard COPY may reduce stack depth while preserving exactly
    # the selected operand's reference.
    reduced = _with_cached(
        model, "_stm_reducer_cached", _Binary(copy_source=0),
        lambda: model._stm_bounded_reduce_step())

    assert bool(reduced.item())
    assert int(stm._concept_rows[0, 0]) == 11
    torch.testing.assert_close(
        stm._concept_activations[0, 0], torch.tensor(0.1))
