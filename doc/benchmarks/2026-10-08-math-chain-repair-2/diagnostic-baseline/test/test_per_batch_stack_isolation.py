"""B>=2 per-row isolation for CategoryStack and ReconstructionStack.

Task 1 of the microbatch AR refactor (see
basicmodel/doc/specs/2026-04-22-microbatch-ar-refactor-design.md).

These tests run with stacks constructed directly (no full SymbolSpace), since
the stacks themselves are simple data structures whose batched semantics can
be tested in isolation.
"""
import os
import sys

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Language import CategoryStack, ReconstructionStack


# -- CategoryStack --------------------------------------------------------------

def test_category_stack_b2_isolation():
    s = CategoryStack(dim=4, batch=2, max_depth=8)
    v0 = torch.tensor([1., 0., 0., 0.])
    v1 = torch.tensor([0., 1., 0., 0.])
    s.push(0, v0)
    s.push(1, v1)
    s.push(0, v1)
    assert s.depth(0) == 2
    assert s.depth(1) == 1
    top0 = s.pop(0)
    assert torch.equal(top0, v1)
    assert s.depth(0) == 1
    assert s.depth(1) == 1  # untouched


def test_category_stack_flatten_per_row():
    s = CategoryStack(dim=4, batch=2, max_depth=8)
    s.push(0, torch.zeros(4))
    s.push(0, torch.ones(4))
    s.push(1, torch.full((4,), 2.0))
    f0 = s.flatten(0)
    f1 = s.flatten(1)
    assert f0.shape == (8,)
    assert f1.shape == (4,)
    assert torch.equal(f0[:4], torch.zeros(4))
    assert torch.equal(f0[4:], torch.ones(4))
    assert torch.equal(f1, torch.full((4,), 2.0))


def test_category_stack_grad_flows_per_row():
    """Gradient through flatten(b) must reach the pushed vec for that row."""
    s = CategoryStack(dim=4, batch=2, max_depth=8)
    v = torch.randn(4, requires_grad=True)
    s.push(1, v)
    out = s.flatten(1).sum()
    out.backward()
    assert v.grad is not None
    assert torch.any(v.grad != 0)


def test_category_stack_ensure_batch_grows_and_clears():
    s = CategoryStack(dim=4, batch=1, max_depth=8)
    s.push(0, torch.ones(4))
    assert s.depth(0) == 1
    s.ensure_batch(3)
    # ensure_batch reallocates fresh storage; depths are zero per row.
    assert s.depth(0) == 0
    assert s.depth(1) == 0
    assert s.depth(2) == 0


# -- ReconstructionStack ---------------------------------------------------

def test_reconstruction_stack_b2_isolation():
    s = ReconstructionStack(batch=2, max_depth=8)
    s.push(0, rule_id=7, word_id=42)
    s.push(1, rule_id=3, word_id=99)
    assert s.depth(0) == 1 and s.depth(1) == 1
    rule, word = s.peek(0)
    assert rule == 7 and word == 42
    s.pop(1)
    assert s.depth(0) == 1 and s.depth(1) == 0


def test_reconstruction_stack_pop_returns_tuple():
    s = ReconstructionStack(batch=2, max_depth=8)
    s.push(0, rule_id=2, word_id=5)
    s.push(0, rule_id=4, word_id=11)
    assert s.pop(0) == (4, 11)
    assert s.pop(0) == (2, 5)
    assert s.depth(0) == 0


def test_reconstruction_stack_ensure_batch_grows():
    s = ReconstructionStack(batch=1, max_depth=4)
    s.push(0, rule_id=1, word_id=1)
    s.ensure_batch(3)
    assert s.depth(0) == 0
    assert s._entries.shape == (3, 4, 2)


def test_reconstruction_stack_owns_detached_sentence_teacher():
    s = ReconstructionStack(batch=2, max_depth=16)
    s.begin_sentence()
    word_ids = torch.tensor([[11, 12, -1], [21, -1, -1]])
    word_mask = word_ids >= 0
    leaves = torch.randn(2, 3, 5, requires_grad=True)
    marginal = torch.softmax(torch.randn(2, 1, 3), dim=-1)
    can = torch.tensor([True, False])

    s.store_words(word_ids, word_mask)
    word_parts = torch.tensor([
        [[1, 2], [3, -1], [-1, -1]],
        [[4, 5], [-1, -1], [-1, -1]],
    ])
    word_part_mask = word_parts >= 0
    s.store_word_parts(word_parts, word_part_mask)
    s.store_leaves(leaves)
    s.record_reduction(marginal, can)

    stored_ids, stored_mask = s.words()
    stored_leaves = s.leaves()
    stored_step = s.reduction_trace()[0]
    assert torch.equal(stored_ids, word_ids)
    assert torch.equal(stored_mask, word_mask)
    stored_parts, stored_part_mask = s.word_parts()
    assert torch.equal(stored_parts, word_parts)
    assert torch.equal(stored_part_mask, word_part_mask)
    assert stored_leaves.grad_fn is None
    assert stored_step[0].grad_fn is None
    # The trace is an observation, not an alias of mutable producer storage.
    with torch.no_grad():
        leaves.zero_()
        marginal.zero_()
    assert torch.count_nonzero(stored_leaves) > 0
    assert torch.count_nonzero(stored_step[0]) > 0


def test_reconstruction_stack_clear_row_masks_all_teacher_artifacts():
    s = ReconstructionStack(batch=2, max_depth=16)
    s.begin_sentence()
    s.store_words(torch.tensor([[1, 2], [3, 4]]))
    s.store_word_parts(
        torch.tensor([[[1, -1], [2, -1]], [[3, 4], [5, -1]]]),
        torch.tensor([[[True, False], [True, False]],
                      [[True, True], [True, False]]]))
    s.store_leaves(torch.ones(2, 2, 3))
    marginal = torch.ones(2, 1, 1)
    can = torch.tensor([True, True])
    s.record_reduction(marginal, can)

    s.clear_rows(0, 1)

    ids, mask = s.words()
    assert torch.equal(ids[0], torch.tensor([-1, -1]))
    assert not bool(mask[0].any())
    part_ids, part_mask = s.word_parts()
    assert torch.equal(part_ids[0], torch.full((2, 2), -1))
    assert not bool(part_mask[0].any())
    assert torch.count_nonzero(s.leaves()[0]) == 0
    assert not bool(s.reduction_trace()[0][1][0])
    assert bool(s.reduction_trace()[0][1][1])


def test_reconstruction_stack_fixed_choices_are_mapped_and_row_scoped():
    s = ReconstructionStack(batch=2, max_depth=16)
    s.prepare_choices(
        2, 5, device="cpu", unary_rule_ids=(3, 4),
        binary_rule_ids=(7, 8, 9))
    s.record_choice(
        1, torch.tensor([8, 9]), arity=2,
        mask=torch.tensor([True, False]))
    s.record_choice(
        3, torch.tensor([4, 3]), arity=1,
        mask=torch.tensor([True, True]))

    rule_ids, arities, mask = s.choices()
    assert rule_ids.tolist() == [
        [-1, 8, -1, 4, -1],
        [-1, -1, -1, 3, -1],
    ]
    assert arities.tolist() == [
        [0, 2, 0, 1, 0],
        [0, 0, 0, 1, 0],
    ]
    assert mask.tolist() == [
        [False, True, False, True, False],
        [False, False, False, True, False],
    ]
    assert s.rule_map(1).tolist() == [3, 4]
    assert s.rule_map(2).tolist() == [7, 8, 9]

    s.clear_rows(0, 1)
    rule_ids, arities, mask = s.choices()
    assert not bool(mask[0].any())
    assert bool(mask[1, 3])
    assert int(rule_ids[1, 3]) == 3
    assert int(arities[1, 3]) == 1


def test_reconstruction_stack_choice_write_is_fullgraph_visible():
    s = ReconstructionStack(batch=2, max_depth=16)
    s.prepare_choices(
        2, 4, device="cpu", unary_rule_ids=(2,),
        binary_rule_ids=(5, 6))

    def write_choice(rule_ids, mask, local_loss):
        s.record_choice(
            2, rule_ids, arity=2, mask=mask,
            local_structural_loss=local_loss)
        ids, arities, active = s.choices()
        return ids.clone(), arities.clone(), active.clone()

    compiled = torch.compile(write_choice, backend="eager", fullgraph=True)
    local_loss = torch.tensor([0.25, 0.75], requires_grad=True)
    ids, arities, active = compiled(
        torch.tensor([5, 6]), torch.tensor([True, False]), local_loss)
    assert ids[:, 2].tolist() == [5, -1]
    assert arities[:, 2].tolist() == [2, 0]
    assert active[:, 2].tolist() == [True, False]
    forward_loss = s.forward_loss()
    assert forward_loss is not None
    forward_loss.backward()
    assert local_loss.grad.tolist() == [1.0, 0.0]
