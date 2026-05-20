"""Tests for STM spans + backpointers.

Plan: path-to-complete §5 — "SHIFT creates [i, i+1] leaves. REDUCE
creates a parent span and stores child pointers. This gives you
Viterbi extraction and a chart-compatible projection when needed."

Rather than extend ``TypedStack`` with span buffers and per-slot child
pointers (which would duplicate ``ParseState``'s frame list), the STM
compose loop populates a ``ParseState`` directly and exposes it on
the WordSpace via ``word_space.parse_state``. The frames carry spans,
the actions carry backpointers (``operand_indices`` → parent frame
index). This is the Viterbi substrate downstream consumers will read.
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


def _make_ws():
    """STM-backed WordSpace from the test_stm_real_input_loop fixture."""
    from test_stm_real_input_loop import _make_word_space_for_stm
    return _make_word_space_for_stm(stm_dim=4)


def test_compose_stm_populates_parse_state():
    """After compose, ``word_space.parse_state`` is a ``ParseState``
    with non-empty frames matching the SHIFT count + REDUCE count."""
    from parse_state import ParseState
    ws, _ = _make_ws()
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    ps = ws.parse_state
    assert isinstance(ps, ParseState)
    # 3 SHIFTs + 2 REDUCEs = 5 frames
    assert len(ps.frames) == 5
    assert len(ps.actions) == 2


def test_shift_frames_carry_unit_spans():
    """Each SHIFTed leaf has span ``[i, i+1]`` where i is the token
    position. The 3 leaves under our grammar appear at frame indices
    0, 1, 3 (a REDUCE fires between the second and third SHIFT)."""
    ws, _ = _make_ws()
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    ps = ws.parse_state
    leaf_spans = [(ps.frames[i].span_start, ps.frames[i].span_end)
                  for i in (0, 1, 3)]
    assert leaf_spans == [(0, 1), (1, 2), (2, 3)]


def test_reduce_frames_carry_joined_spans():
    """Each REDUCE parent's span covers all its operands' spans.

    Frame layout (left-corner shift-reduce):
      0=DET[0,1], 1=N[1,2], 2=NP[0,2] (REDUCE after SHIFT 2),
      3=VP[2,3], 4=S[0,3] (REDUCE after SHIFT 3).
    """
    ws, _ = _make_ws()
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    ps = ws.parse_state
    np_frame = ps.frames[2]
    assert (np_frame.span_start, np_frame.span_end) == (0, 2)
    s_frame = ps.frames[4]
    assert (s_frame.span_start, s_frame.span_end) == (0, 3)


def test_actions_record_operand_indices_as_backpointers():
    """Each REDUCE Action's ``operand_indices`` point at the operand
    frame indices, enabling Viterbi extraction.

    Left-corner timing: first REDUCE consumes DET(0)+N(1) → NP(2);
    second REDUCE consumes NP(2)+VP(3) → S(4).
    """
    ws, _ = _make_ws()
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    ps = ws.parse_state
    assert ps.actions[0].operand_indices == (0, 1)
    assert ps.actions[0].parent_index == 2
    assert ps.actions[1].operand_indices == (2, 3)
    assert ps.actions[1].parent_index == 4


def test_parse_state_trace_equals_action_list_for_greedy_path():
    """The greedy STM picks one rule per REDUCE; the trace IS the
    action list (one path, no branching)."""
    ws, _ = _make_ws()
    inp = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],
         [2.0, 0.0, 0.0, 0.0],
         [3.0, 0.0, 0.0, 0.0]],
    ])
    ws.compose(input_vectors=inp)
    ps = ws.parse_state
    assert ps.trace == ps.actions
