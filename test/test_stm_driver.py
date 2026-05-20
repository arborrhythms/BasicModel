"""Tests for the STM shift/reduce driver and its small rule scorer.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§Phase 2 / step 3 — STM driver + scorer (new small module, NOT a port
of the chart's _rule_embed).

The driver coordinates a typed_stack with a rule scorer and a list of
rule signatures. SHIFT pushes a token frame; REDUCE builds the
admissibility mask, scores admissible rules, picks one (argmax over
masked logits), and reports the chosen rule + the operand frames. The
driver does NOT yet apply rule semantics to compute parent payloads
(that's a follow-up — for now it returns the choice so callers can
verify the masking + scoring are correct).
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _three_rules():
    """Three serialized RuleOrderSignature dicts: lift, lower, not."""
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


def test_rule_scorer_construct_and_forward():
    """``RuleScorer(payload_dim, n_rules)`` is a small nn.Module producing
    per-rule logits from the top-of-stack payload(s)."""
    from stm_driver import RuleScorer
    import torch
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    # binary input: two payloads concatenated
    left = torch.randn(4)
    right = torch.randn(4)
    logits = scorer(left, right)
    assert logits.shape == (3,)


def test_rule_scorer_unary():
    """RuleScorer with only a left operand also produces logits."""
    from stm_driver import RuleScorer
    import torch
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    left = torch.randn(4)
    logits = scorer(left, None)
    assert logits.shape == (3,)


def test_stm_driver_construct():
    """``STMDriver(typed_stack, rule_signatures, scorer)`` wires them
    together."""
    from typed_stack import TypedStack
    from stm_driver import STMDriver, RuleScorer
    stack = TypedStack(batch=1, max_depth=8, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    driver = STMDriver(typed_stack=stack, rule_signatures=_three_rules(),
                       scorer=scorer)
    assert driver.typed_stack is stack
    assert driver.scorer is scorer
    assert driver.rule_signatures == _three_rules()


def test_stm_driver_shift_pushes_frame():
    """``shift(b, payload, category, order, ref_id)`` pushes onto the
    typed_stack."""
    from typed_stack import TypedStack
    from stm_driver import STMDriver, RuleScorer
    import torch
    stack = TypedStack(batch=1, max_depth=8, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    driver = STMDriver(stack, _three_rules(), scorer)
    payload = torch.tensor([1.0, 2.0, 3.0, 4.0])
    driver.shift(0, payload, category='NP', order=3, ref_id=42)
    assert int(stack._depth[0].item()) == 1
    top = stack.top(0)
    assert top['category_str'] == 'NP'
    assert top['order'] == 3
    assert top['ref_id'] == 42


def test_reduce_step_picks_admissible_rule_binary():
    """With NP@3, VP@1 on the stack, REDUCE picks the lift rule (the only
    admissible one)."""
    from typed_stack import TypedStack
    from stm_driver import STMDriver, RuleScorer
    import torch
    stack = TypedStack(batch=1, max_depth=8, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    driver = STMDriver(stack, _three_rules(), scorer)
    driver.shift(0, torch.zeros(4), category='NP', order=3, ref_id=0)
    driver.shift(0, torch.zeros(4), category='VP', order=1, ref_id=0)
    chosen = driver.reduce_step(0)
    assert chosen['rule_index'] == 0
    assert chosen['rule_signature']['op_name'] == 'lift'


def test_reduce_step_picks_admissible_rule_unary():
    """With single S@3 on the stack, REDUCE picks the not rule."""
    from typed_stack import TypedStack
    from stm_driver import STMDriver, RuleScorer
    import torch
    stack = TypedStack(batch=1, max_depth=8, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    driver = STMDriver(stack, _three_rules(), scorer)
    driver.shift(0, torch.zeros(4), category='S', order=3, ref_id=0)
    chosen = driver.reduce_step(0)
    assert chosen['rule_index'] == 2
    assert chosen['rule_signature']['op_name'] == 'not'


def test_reduce_step_raises_when_no_rule_admissible():
    """With incompatible stack top, REDUCE raises a clear error."""
    from typed_stack import TypedStack
    from stm_driver import STMDriver, RuleScorer
    import torch
    stack = TypedStack(batch=1, max_depth=8, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    driver = STMDriver(stack, _three_rules(), scorer)
    driver.shift(0, torch.zeros(4), category='XYZ', order=99, ref_id=0)
    driver.shift(0, torch.zeros(4), category='ABC', order=99, ref_id=0)
    import pytest
    with pytest.raises(RuntimeError, match='no admissible rule'):
        driver.reduce_step(0)


def test_reduce_step_uses_softmax_over_admissible_only():
    """The scorer's logits over inadmissible rules don't influence the
    pick (verified by setting scorer logits so an inadmissible rule has
    the highest raw score; argmax should still pick the admissible one)."""
    from typed_stack import TypedStack
    from stm_driver import STMDriver, RuleScorer
    import torch
    import torch.nn as nn
    stack = TypedStack(batch=1, max_depth=8, dim=4)
    scorer = RuleScorer(payload_dim=4, n_rules=3)
    # Rig the scorer so rule 1 (lower) has the highest raw logit even
    # though only rule 0 (lift) is admissible for the stack state we
    # set up. Easiest: replace the final linear layer's weights /
    # biases with constants.
    with torch.no_grad():
        scorer.head.weight.zero_()
        scorer.head.bias.copy_(torch.tensor([0.0, 10.0, 0.0]))
    driver = STMDriver(stack, _three_rules(), scorer)
    driver.shift(0, torch.zeros(4), category='NP', order=3, ref_id=0)
    driver.shift(0, torch.zeros(4), category='VP', order=1, ref_id=0)
    chosen = driver.reduce_step(0)
    # Despite rule 1 having higher raw logit, only rule 0 is
    # admissible — that's what gets picked.
    assert chosen['rule_index'] == 0
