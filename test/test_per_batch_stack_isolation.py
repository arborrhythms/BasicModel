"""B>=2 per-row isolation for CategoryStack and ReconstructionStack.

Task 1 of the microbatch AR refactor (see
basicmodel/doc/specs/2026-04-22-microbatch-ar-refactor-design.md).

These tests run with stacks constructed directly (no full SymbolicSpace), since
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
