"""Reverse-path left-shift helper unit tests.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §2R.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_COMPILE", "eager")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "bin"))

import torch


def _shift(gen, mask):
    from Models import BasicModel
    return BasicModel._left_shift_by_mask(gen, mask)


def test_left_shift_packs_real_positions_to_front():
    gen = torch.tensor([[
        [1.0, 1.0],
        [0.0, 0.0],
        [2.0, 2.0],
        [0.0, 0.0],
    ]])
    mask = torch.tensor([[True, False, True, False]])
    out = _shift(gen, mask)
    assert torch.equal(out[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.equal(out[0, 1], torch.tensor([2.0, 2.0]))
    assert torch.equal(out[0, 2], torch.zeros(2))
    assert torch.equal(out[0, 3], torch.zeros(2))


def test_left_shift_all_real_passthrough():
    gen = torch.arange(12.0).view(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = _shift(gen, mask)
    assert torch.equal(out, gen)


def test_left_shift_all_padding_zeros_out():
    gen = torch.randn(2, 5, 3)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    out = _shift(gen, mask)
    assert torch.equal(out, torch.zeros_like(gen))


def test_left_shift_preserves_relative_order():
    gen = torch.tensor([[
        [0.0],
        [3.0],
        [0.0],
        [7.0],
        [9.0],
    ]])
    mask = torch.tensor([[False, True, False, True, True]])
    out = _shift(gen, mask)
    assert torch.equal(out[0, 0], torch.tensor([3.0]))
    assert torch.equal(out[0, 1], torch.tensor([7.0]))
    assert torch.equal(out[0, 2], torch.tensor([9.0]))
    assert torch.equal(out[0, 3], torch.zeros(1))
    assert torch.equal(out[0, 4], torch.zeros(1))
