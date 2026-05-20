"""Tests for the parser-neutral ParseState.

Plan: path-to-complete §3.

ParseState is the substrate both chart (CKY) and STM populate.
Downstream consumers read from it; the legacy adapter projects to
``_chart_score`` / ``_chart_vec`` / ``_chart_pos`` / ``_derivation_trace``
during the consumer-migration window.
"""
import os
import sys

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)


def test_parse_state_constructs_empty():
    """A fresh ``ParseState`` has empty frames / actions / trace."""
    from parse_state import ParseState
    s = ParseState()
    assert s.frames == []
    assert s.actions == []
    assert s.trace == []
    assert s.current_rules == {}
    assert s.generate_rules == {}
    assert s.leaf_category_probs is None


def test_add_leaf_creates_frame_with_unit_span():
    """``add_leaf`` returns the frame index and the frame carries an
    inclusive-exclusive ``[i, i+1]`` span."""
    from parse_state import ParseState
    s = ParseState()
    idx = s.add_leaf(payload=torch.tensor([1.0, 0.0]),
                     category='DET', order=0, ref_id=3, position=0)
    assert idx == 0
    assert s.frames[0].category == 'DET'
    assert s.frames[0].span_start == 0
    assert s.frames[0].span_end == 1


def test_add_reduce_creates_parent_span_covering_operands():
    """``add_reduce`` joins the operand spans into the parent's span."""
    from parse_state import ParseState
    s = ParseState()
    l = s.add_leaf(payload=torch.tensor([0.0]), category='A',
                   order=0, ref_id=0, position=0)
    r = s.add_leaf(payload=torch.tensor([0.0]), category='B',
                   order=0, ref_id=1, position=1)
    p = s.add_reduce(rule_id=7, operand_indices=(l, r),
                     parent_payload=torch.tensor([0.0]),
                     parent_category='AB', parent_order=0)
    assert p == 2
    assert s.frames[p].span_start == 0
    assert s.frames[p].span_end == 2
    assert s.actions[0].rule_id == 7
    assert s.actions[0].operand_indices == (l, r)
    assert s.actions[0].parent_index == p


def test_add_reduce_carries_score_and_probability():
    """Action records carry the scorer logit and post-mask softmax
    probability so soft training paths can read them later."""
    from parse_state import ParseState
    s = ParseState()
    l = s.add_leaf(payload=torch.tensor([0.0]), category='A',
                   order=0, ref_id=0, position=0)
    s.add_reduce(rule_id=1, operand_indices=(l,),
                 parent_payload=torch.tensor([0.0]),
                 parent_category='A',
                 parent_order=0,
                 score=2.3, probability=0.85)
    assert s.actions[0].score == 2.3
    assert s.actions[0].probability == 0.85


def test_legacy_adapter_writes_derivation_trace():
    """``project_to_chart_fields`` sets ``_derivation_trace`` from
    ``state.trace`` in the chart's expected shape."""
    from parse_state import ParseState, Action, project_to_chart_fields
    s = ParseState()
    l = s.add_leaf(payload=torch.tensor([0.0]), category='A',
                   order=0, ref_id=0, position=0)
    r = s.add_leaf(payload=torch.tensor([0.0]), category='B',
                   order=0, ref_id=1, position=1)
    p = s.add_reduce(rule_id=5, operand_indices=(l, r),
                     parent_payload=torch.tensor([0.0]),
                     parent_category='AB', parent_order=0)
    s.trace = list(s.actions)

    class _Target:
        pass

    target = _Target()
    project_to_chart_fields(s, target)
    assert isinstance(target._derivation_trace, list)
    assert len(target._derivation_trace) == 1
    row = target._derivation_trace[0]
    assert row[0][0] == 5  # rule_id


def test_legacy_adapter_chart_score_left_none_when_projected_from_stm():
    """STM produces no all-cells score tensor; the adapter sets
    ``_chart_score`` to ``None`` so consumers can detect the
    no-soft-chart projection and fall back gracefully."""
    from parse_state import ParseState, project_to_chart_fields
    s = ParseState()

    class _Target:
        pass

    target = _Target()
    project_to_chart_fields(s, target)
    assert target._chart_score is None
    assert target._chart_vec is None


def test_legacy_adapter_chart_pos_carries_leaf_category_probs():
    """When the parser produces leaf-level category probability
    distributions, they project into ``_chart_pos``."""
    from parse_state import ParseState, project_to_chart_fields
    s = ParseState()
    s.leaf_category_probs = torch.tensor([[0.1, 0.9], [0.7, 0.3]])

    class _Target:
        pass

    target = _Target()
    project_to_chart_fields(s, target)
    assert torch.is_tensor(target._chart_pos)
    assert target._chart_pos.shape == (2, 2)
