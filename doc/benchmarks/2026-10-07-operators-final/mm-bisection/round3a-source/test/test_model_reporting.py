"""Focused regression tests for model text-report helpers."""

from pathlib import Path
import sys

import torch

BIN = Path(__file__).resolve().parents[1] / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from Models import _grammar_row_preview, _grammar_rows_for_report


def test_grammar_rows_for_report_accepts_shared_serial_path():
    assert _grammar_rows_for_report([3, 5, 8]) == [[3, 5, 8]]


def test_grammar_rows_for_report_preserves_batched_paths():
    assert _grammar_rows_for_report([[3, 5], [8]]) == [[3, 5], [8]]
    assert _grammar_rows_for_report(torch.tensor([[1, 2], [3, 4]])) == [
        [1, 2], [3, 4],
    ]


def test_grammar_row_preview_bounds_long_padded_path():
    head, omitted, tail = _grammar_row_preview(range(100), limit=10)
    assert head == [0, 1, 2, 3, 4]
    assert omitted == 90
    assert tail == [95, 96, 97, 98, 99]
