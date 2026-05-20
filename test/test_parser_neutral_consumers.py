"""Tests for the parser-neutral consumer migration.

Plan: path-to-complete §6 — "Update syntax dump, SVO extraction,
reverse/generate, and diagnostics to read ``wordSpace.parse_state``
first, falling back to chart fields."

This suite covers the parser-neutral accessor surface:

  * ``WordSpace.parse_state`` is populated by both chart and STM
    backends after ``compose()`` (chart via a projection from its
    derivation trace; STM via the SHIFT/REDUCE loop directly).
  * ``WordSpace.parse_rules_for_tier(tier)`` reads from
    ``parse_state.current_rules`` rather than reaching into
    chart-only fields.
  * The legacy adapter ``parse_state.project_to_chart_fields`` lets
    chart-only consumers keep reading ``_chart_score`` / etc. as a
    transition shim — STM-projected charts get ``None`` for the
    all-cells tensors.
"""
import os
import sys

import torch
import torch.nn as nn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)


def test_parse_state_attribute_exists_after_stm_compose():
    """STM-backed compose leaves ``parse_state`` set on the WordSpace."""
    from test_stm_real_input_loop import _make_word_space_for_stm
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    assert ws.parse_state is not None


def test_parse_rules_for_tier_reads_from_parse_state():
    """``parse_rules_for_tier('S')`` returns the same list as
    ``parse_state.current_rules['S']``."""
    from test_stm_real_input_loop import _make_word_space_for_stm
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    expected = ws.parse_state.current_rules['S']
    actual = ws.parse_rules_for_tier('S')
    assert actual == expected


def test_parse_rules_for_tier_falls_back_to_legacy_current_rules():
    """When ``parse_state`` is None (e.g., the chart fast-path didn't
    populate it), ``parse_rules_for_tier`` falls back to the legacy
    ``current_rules`` attribute that chart-mode populates directly."""
    from Language import WordSpace
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    object.__setattr__(ws, 'parse_state', None)
    ws.current_rules = {'S': [[0, 1]]}
    assert ws.parse_rules_for_tier('S') == [[0, 1]]


def test_parse_rules_for_tier_unknown_tier_returns_empty():
    """An unknown tier returns ``[]`` without raising."""
    from Language import WordSpace
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    object.__setattr__(ws, 'parse_state', None)
    ws.current_rules = {}
    assert ws.parse_rules_for_tier('Z') == []


def test_parse_derivation_trace_reads_from_parse_state_actions():
    """``parse_derivation_trace()`` returns the per-row list of
    (rule_id, span_start, span_end) tuples projected from
    ``parse_state.actions``."""
    from test_stm_real_input_loop import _make_word_space_for_stm
    ws, _ = _make_word_space_for_stm(stm_dim=4)
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    trace = ws.parse_derivation_trace()
    assert isinstance(trace, list)
    assert len(trace) == 1
    row = trace[0]
    # Two REDUCEs fire; each entry is (rule_id, span_start, span_end)
    assert len(row) == 2
    for entry in row:
        assert isinstance(entry, tuple)
        assert len(entry) == 3
