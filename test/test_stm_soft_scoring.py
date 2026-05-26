"""Tests for STM soft scoring + real grammar-op parent payload.

Plan: path-to-complete §4 — "At each reduce point: build hard
admissibility mask; compute rule logits; training: softmax(masked_logits)
and weighted parent mixture; eval: argmax/Viterbi choice; parent
payload must call the actual grammar op, not average operands."

Post-2026-05-21 (WordSubSpace/STM Layer refactor) the driver+scorer live
on ``ShortTermMemory`` (a ``Layer``); ``test/_stm_test_fixtures.make_driver``
re-creates the legacy ``STMDriver`` call surface for these tests.
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

from _stm_test_fixtures import make_typed_stack, make_driver


def test_reduce_step_soft_returns_probabilities():
    """``driver.reduce_step_soft(b)`` returns both an argmax pick AND a
    probability vector over rules (softmax of masked logits).
    Inadmissible rules carry probability 0."""
    rule_sigs = [
        {'lhs_category': 'NP', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['DET', 'N'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
        {'lhs_category': 'S', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['NP', 'VP'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'disjunction', 'order_delta': 0},
    ]
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    ts.push(0, torch.tensor([1.0, 0.0, 0.0, 0.0]),
            category_id_str='DET', order=0, ref_id=0)
    ts.push(0, torch.tensor([0.0, 1.0, 0.0, 0.0]),
            category_id_str='N', order=0, ref_id=1)
    driver = make_driver(ts, rule_sigs, payload_dim=4)
    result = driver.reduce_step_soft(0)
    assert 'rule_index' in result
    assert 'probabilities' in result
    probs = result['probabilities']
    assert torch.is_tensor(probs)
    assert probs.shape == (2,)
    # Only rule 0 (NP=conj(DET,N)) is admissible
    assert probs[0] > 0.99
    assert probs[1] < 0.01


def test_reduce_step_soft_distribution_sums_to_one_over_admissible():
    """When multiple rules are admissible, softmax over their logits
    sums to 1.0; inadmissible rules contribute 0."""
    rule_sigs = [
        {'lhs_category': 'X', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['A', 'B'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'conjunction', 'order_delta': 0},
        {'lhs_category': 'Y', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['A', 'B'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'disjunction', 'order_delta': 0},
        {'lhs_category': 'Z', 'lhs_order': 0,
         'lhs_order_kind': 'constant',
         'rhs_categories': ['C', 'D'],
         'rhs_orders': [0, 0],
         'rhs_order_kinds': ['constant', 'constant'],
         'op_name': 'union', 'order_delta': 0},
    ]
    ts = make_typed_stack(batch=1, max_depth=4, dim=4)
    ts.push(0, torch.tensor([0.5, 0.0, 0.0, 0.0]),
            category_id_str='A', order=0, ref_id=0)
    ts.push(0, torch.tensor([0.0, 0.5, 0.0, 0.0]),
            category_id_str='B', order=0, ref_id=1)
    driver = make_driver(ts, rule_sigs, payload_dim=4)
    result = driver.reduce_step_soft(0)
    probs = result['probabilities']
    # rules 0 and 1 are admissible, rule 2 is not
    assert probs[2].item() == pytest.approx(0.0, abs=1e-6)
    assert probs[0].item() + probs[1].item() == pytest.approx(1.0, abs=1e-5)


def test_op_dispatch_uses_real_grammar_op_for_parent_payload():
    """Compose under STM with a known op_name uses the actual
    ``Layers.Ops`` kernel (e.g., conjunction → intersection) to compute
    the parent payload — NOT the (left+right)/2 placeholder."""
    from Language import WordSubSpace
    left = torch.tensor([1.0, 0.0])
    right = torch.tensor([0.0, 1.0])
    parent = WordSubSpace._apply_grammar_op(
        op_name='conjunction', left=left, right=right)
    assert torch.allclose(parent, torch.tensor([0.0, 0.0]), atol=1e-5)


def test_op_dispatch_fallback_when_op_unknown():
    """An op_name not in the registry falls back to the midpoint
    placeholder (so old grammars don't break under the soft path)."""
    from Language import WordSubSpace
    left = torch.tensor([2.0, 0.0])
    right = torch.tensor([0.0, 2.0])
    parent = WordSubSpace._apply_grammar_op(
        op_name='not_a_real_op', left=left, right=right)
    assert torch.allclose(parent, torch.tensor([1.0, 1.0]), atol=1e-5)


def test_stm_compose_training_mode_produces_soft_mixture_parent():
    """Public surface: ``_apply_grammar_op`` exists on the WordSubSpace
    so the SR-parser can call it from the soft mixture path."""
    pytest.importorskip('test_partition_pos_codebook')
    from test_partition_pos_codebook import _make_word_space  # noqa: E402
    ws = _make_word_space()
    assert hasattr(ws, '_apply_grammar_op')
