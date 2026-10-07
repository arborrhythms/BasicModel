"""Tests for the typed STM stack — per-frame metadata (category, order,
ref_id) alongside the vector payload.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§STM Shift/Reduce Runtime — STM stack item metadata.

Post-2026-05-21 (SymbolSubSpace/STM Layer refactor) the typed STM data
lives directly on ``SymbolSubSpace`` (not on a standalone ``TypedStack``
module). The data layout is unchanged:

    ss._buffer    [B, cap, D]
    ss._category  [B, cap]   long, -1 for empty
    ss._order     [B, cap]   long
    ss._ref_id    [B, cap]   long, -1 for unsnapped
    ss._depth     [B]        long

``make_typed_stack(batch, max_depth, dim)`` from ``_stm_test_fixtures``
constructs a bare SymbolSubSpace with those buffers; the tests below
exercise ``push``/``pop``/``top``/``reduce_admissibility`` on it (the
same names as the retired ``TypedStack`` class — those methods now
live on SymbolSubSpace).
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stm_test_fixtures import make_typed_stack


def test_typed_stack_construct():
    """Constructor allocates the five parallel tensors at expected shapes."""
    import torch
    stm = make_typed_stack(batch=2, max_depth=8, dim=4)
    assert stm._buffer.shape == (2, 8, 4)
    assert stm._category.shape == (2, 8)
    assert stm._order.shape == (2, 8)
    assert stm._ref_id.shape == (2, 8)
    assert stm._depth.shape == (2,)
    # Initial state: depth zero, sentinel -1 in category / ref_id
    assert (stm._depth == 0).all()
    assert (stm._category == -1).all()
    assert (stm._ref_id == -1).all()
    # Buffer / order zero-init
    assert (stm._buffer == 0).all()
    assert (stm._order == 0).all()


def test_typed_stack_push_single_row():
    """``push(b, vec, category_id, order, ref_id)`` writes into the top
    slot and increments depth."""
    import torch
    stm = make_typed_stack(batch=1, max_depth=4, dim=3)
    vec = torch.tensor([1.0, 2.0, 3.0])
    stm.push(0, vec, category_id=7, order=2, ref_id=42)
    assert int(stm._depth[0].item()) == 1
    assert torch.allclose(stm._buffer[0, 0], vec)
    assert int(stm._category[0, 0].item()) == 7
    assert int(stm._order[0, 0].item()) == 2
    assert int(stm._ref_id[0, 0].item()) == 42


def test_typed_stack_push_multiple_then_top():
    """Pushing three items, ``top(b)`` returns the most recent; ``top(b, k=2)``
    returns the top two."""
    import torch
    stm = make_typed_stack(batch=1, max_depth=8, dim=2)
    stm.push(0, torch.tensor([1.0, 0.0]), category_id=1, order=0, ref_id=10)
    stm.push(0, torch.tensor([2.0, 0.0]), category_id=2, order=0, ref_id=20)
    stm.push(0, torch.tensor([3.0, 0.0]), category_id=3, order=1, ref_id=30)
    top = stm.top(0)
    assert int(top['category']) == 3
    assert int(top['order']) == 1
    assert int(top['ref_id']) == 30
    assert torch.allclose(top['payload'], torch.tensor([3.0, 0.0]))


def test_typed_stack_pop_returns_metadata_and_decrements_depth():
    """``pop(b)`` returns the top item and decrements depth."""
    import torch
    stm = make_typed_stack(batch=1, max_depth=4, dim=2)
    stm.push(0, torch.tensor([1.0, 1.0]), category_id=5, order=2, ref_id=99)
    popped = stm.pop(0)
    assert int(stm._depth[0].item()) == 0
    assert int(popped['category']) == 5
    assert int(popped['order']) == 2
    assert int(popped['ref_id']) == 99
    assert torch.allclose(popped['payload'], torch.tensor([1.0, 1.0]))


def test_typed_stack_per_row_isolation():
    """Pushes on row 0 don't affect row 1."""
    import torch
    stm = make_typed_stack(batch=2, max_depth=4, dim=2)
    stm.push(0, torch.zeros(2), category_id=1, order=0, ref_id=10)
    assert int(stm._depth[0].item()) == 1
    assert int(stm._depth[1].item()) == 0
    assert int(stm._category[0, 0].item()) == 1
    assert int(stm._category[1, 0].item()) == -1


def test_typed_stack_overflow_raises():
    """Push beyond ``max_depth`` raises AssertionError."""
    import torch
    stm = make_typed_stack(batch=1, max_depth=2, dim=1)
    stm.push(0, torch.zeros(1), category_id=0, order=0, ref_id=0)
    stm.push(0, torch.zeros(1), category_id=0, order=0, ref_id=0)
    import pytest
    with pytest.raises(AssertionError):
        stm.push(0, torch.zeros(1), category_id=0, order=0, ref_id=0)


def test_typed_stack_underflow_raises():
    """Pop on an empty stack raises AssertionError."""
    stm = make_typed_stack(batch=1, max_depth=2, dim=1)
    import pytest
    with pytest.raises(AssertionError):
        stm.pop(0)


def test_typed_stack_top_with_no_items_raises():
    """top() on empty row raises."""
    stm = make_typed_stack(batch=1, max_depth=2, dim=1)
    import pytest
    with pytest.raises(AssertionError):
        stm.top(0)


# -- REDUCE step: admissibility-masked rule scoring ------------------


def _three_rule_list():
    return [
        {'lhs_category': 'S', 'lhs_order': 4,
         'rhs_categories': ['NP', 'VP'], 'rhs_orders': [3, 1],
         'op_name': 'lift', 'order_delta': 1},
        {'lhs_category': 'NP', 'lhs_order': 3,
         'rhs_categories': ['DET', 'NP'], 'rhs_orders': [0, 4],
         'op_name': 'lower', 'order_delta': -1},
        {'lhs_category': 'S', 'lhs_order': 3,
         'rhs_categories': ['S'], 'rhs_orders': [3],
         'op_name': 'not', 'order_delta': 0},
    ]


def test_reduce_admissibility_binary_picks_only_lift():
    """``ss.reduce_admissibility(b, rule_signatures)`` returns a
    boolean mask matching the stack top two items against each rule.
    With NP@3, VP@1 on the stack, only the lift rule is admissible."""
    import torch
    stm = make_typed_stack(batch=1, max_depth=4, dim=2)
    # Push NP@3 then VP@1 (VP is on top, NP underneath)
    # Convention: left operand = second-from-top, right operand = top.
    stm.push(0, torch.tensor([1.0, 0.0]),
             category_id_str='NP', order=3, ref_id=0)
    stm.push(0, torch.tensor([0.0, 1.0]),
             category_id_str='VP', order=1, ref_id=0)
    rules = _three_rule_list()
    mask = stm.reduce_admissibility(0, rules)
    assert mask.tolist() == [True, False, False]


def test_reduce_admissibility_unary_one_item():
    """With one item on the stack (S@3), only the unary not-rule is admissible."""
    import torch
    stm = make_typed_stack(batch=1, max_depth=4, dim=2)
    stm.push(0, torch.zeros(2), category_id_str='S', order=3, ref_id=0)
    rules = _three_rule_list()
    mask = stm.reduce_admissibility(0, rules)
    assert mask.tolist() == [False, False, True]


def test_reduce_admissibility_combined_with_mask_logits():
    """Full integration: stack top → mask → mask_logits → softmax. Only
    the admissible rule's post-softmax probability survives."""
    from embed import mask_logits
    import torch
    stm = make_typed_stack(batch=1, max_depth=4, dim=2)
    stm.push(0, torch.zeros(2), category_id_str='NP', order=3, ref_id=0)
    stm.push(0, torch.zeros(2), category_id_str='VP', order=1, ref_id=0)
    rules = _three_rule_list()
    mask = stm.reduce_admissibility(0, rules)
    logits = torch.tensor([0.0, 0.0, 0.0])
    probs = torch.softmax(mask_logits(logits, mask), dim=-1)
    assert probs[0].item() > 0.99
    assert probs[1].item() < 1e-6
    assert probs[2].item() < 1e-6
