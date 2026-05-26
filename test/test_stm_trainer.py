"""Tests for the STM scorer training step.

Plan: path-to-complete §7 closeout — train the STM driver's rule
scorer against an oracle (chart Viterbi or hand-crafted) so
``compose(stm)`` matches the oracle on representative grammars.

Post-2026-05-21 (WordSubSpace/STM Layer refactor) the trainer logic
lives on ``ShortTermMemory.train_scorer_step``;
``test/_stm_test_fixtures.make_train_step`` returns a callable matching
the retired ``stm_trainer.train_step(driver, ...)`` signature so these
tests can stay close to their original shape.
"""
import os
import sys

import pytest
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

from _stm_test_fixtures import (
    make_typed_stack, make_driver, make_train_step)

train_step = make_train_step()


def _ambiguous_driver(seed=0):
    """STM driver with two rules sharing RHS (DET, N): rule 0 emits
    ``NPA``, rule 1 emits ``NPB``. The scorer's argmax (unmasked, since
    both are admissible) decides which fires."""
    torch.manual_seed(seed)
    rule_sigs = [
        {'lhs_category': 'NPA', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['DET', 'N'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
        {'lhs_category': 'NPB', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['DET', 'N'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
    ]
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    driver = make_driver(ts, rule_sigs, payload_dim=4)
    return driver


def _stub_snap(payload):
    """Tiny snap function: maps a 4-dim payload to a hand-coded
    category by inspecting which slot is hottest. Slot 0=DET, slot 1=N,
    slot 2=VP."""
    idx = int(torch.argmax(payload).item())
    name = {0: 'DET', 1: 'N', 2: 'VP'}.get(idx, 'UNK')
    return (-1, name, 0)


def _det_n_input():
    """A 2-token input that resolves to (DET, N)."""
    return torch.tensor([
        [[1.0, 0.0, 0.0, 0.0],   # slot 0 hot → DET
         [0.0, 1.0, 0.0, 0.0]],  # slot 1 hot → N
    ])


def test_train_step_reduces_loss():
    """A single training step on the ambiguous grammar drops the loss
    when stepped by an optimizer; over multiple steps the loss
    approaches zero on the supervision target."""
    driver = _ambiguous_driver(seed=42)
    optim = torch.optim.Adam(driver.parameters(), lr=1e-2)
    inp = _det_n_input()
    target = [0]
    pre_loss = train_step(driver, inp, target,
                          snap_fn=_stub_snap, optimizer=None).item()
    for _ in range(50):
        train_step(driver, inp, target,
                   snap_fn=_stub_snap, optimizer=optim)
    post_loss = train_step(driver, inp, target,
                           snap_fn=_stub_snap, optimizer=None).item()
    assert post_loss < pre_loss


def test_post_training_stm_argmax_matches_target():
    """After training, the STM's argmax-driven REDUCE picks the
    supervised rule consistently."""
    driver = _ambiguous_driver(seed=7)
    optim = torch.optim.Adam(driver.parameters(), lr=1e-2)
    inp = _det_n_input()
    target = [0]  # always pick rule 0 (NPA)
    for _ in range(80):
        train_step(driver, inp, target,
                   snap_fn=_stub_snap, optimizer=optim)
    # Now run a single REDUCE and check argmax picks rule 0.
    ts = driver.typed_stack
    while int(ts._depth[0].item()) > 0:
        ts.pop(0)
    driver.shift(0, torch.tensor([1.0, 0.0, 0.0, 0.0]),
                 category='DET', order=0, ref_id=-1)
    driver.shift(0, torch.tensor([0.0, 1.0, 0.0, 0.0]),
                 category='N', order=0, ref_id=-1)
    result = driver.reduce_step(0)
    assert result['rule_index'] == 0


def test_target_inadmissible_raises_when_other_rules_admissible():
    """Supervision must respect the admissibility mask. When some rules
    ARE admissible but the supervisor's target is among the inadmissible
    ones, the trainer raises — the gradient signal would push toward an
    impossible reduce."""
    rule_sigs = [
        {'lhs_category': 'NP', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['DET', 'N'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
        {'lhs_category': 'XX', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['VP', 'VP'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'union', 'order_delta': 0},
    ]
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    driver = make_driver(ts, rule_sigs, payload_dim=4)
    inp = _det_n_input()  # DET, N
    with pytest.raises(ValueError, match='inadmissible'):
        train_step(driver, inp, [1],
                   snap_fn=_stub_snap, optimizer=None)


def test_target_unreachable_no_admissible_returns_zero_loss():
    """When no rule is admissible at the current stack top (the parser
    is genuinely stuck), no loss is generated for that step and the
    trainer returns 0.0 — supervision can't drive an impossible reduce."""
    driver = _ambiguous_driver()
    inp = torch.tensor([
        [[0.0, 0.0, 1.0, 0.0],   # VP
         [0.0, 0.0, 1.0, 0.0]],  # VP
    ])
    loss = train_step(driver, inp, [0],
                      snap_fn=_stub_snap, optimizer=None)
    assert loss.item() == 0.0
