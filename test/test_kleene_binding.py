"""Tests for Kleene variable binding in admissibility checks.

Plan: path-to-complete §2 — restored ``*`` / ``*+1`` / ``*-1`` order
expressions. The admissibility kernel binds the rule-local ``*`` from
the operand's order and propagates the binding consistently across
the rule's other slots. LHS-order propagation (parent gets ``binding
+ lhs.delta``) is the next consumer; this suite covers admissibility.
"""
import os
import sys

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)


def _polymorphic_signature():
    """``S* = lift(NP*, VP1)`` serialized form. LHS variable +0; RHS[0]
    variable +0; RHS[1] constant +1."""
    return {
        'lhs_category': 'S',
        'lhs_order': 0,
        'lhs_order_kind': 'variable',
        'rhs_categories': ['NP', 'VP'],
        'rhs_orders': [0, 1],
        'rhs_order_kinds': ['variable', 'constant'],
        'op_name': 'lift',
        'order_delta': 1,
    }


def _matched_variable_signature():
    """``S* = pi(X*, Y*)`` — both RHS slots are the same variable."""
    return {
        'lhs_category': 'S',
        'lhs_order': 0,
        'lhs_order_kind': 'variable',
        'rhs_categories': ['X', 'Y'],
        'rhs_orders': [0, 0],
        'rhs_order_kinds': ['variable', 'variable'],
        'op_name': 'pi',
        'order_delta': 0,
    }


def test_variable_binds_to_operand_order():
    """A rule with RHS variable accepts any operand order at that slot."""
    from embed import is_rule_admissible
    sig = _polymorphic_signature()
    for k in (0, 1, 2, 5):
        assert is_rule_admissible(
            sig, left_cat='NP', left_order=k,
            right_cat='VP', right_order=1)


def test_constant_slot_still_requires_exact_match():
    """Variable in slot 0 doesn't relax slot 1; ``VP1`` only matches
    ``VP at order 1``."""
    from embed import is_rule_admissible
    sig = _polymorphic_signature()
    assert is_rule_admissible(
        sig, left_cat='NP', left_order=3,
        right_cat='VP', right_order=1)
    assert not is_rule_admissible(
        sig, left_cat='NP', left_order=3,
        right_cat='VP', right_order=2)


def test_shared_variable_must_bind_consistently():
    """When two RHS slots are the same variable, their operand orders
    (modulo the per-slot deltas) must agree."""
    from embed import is_rule_admissible
    sig = _matched_variable_signature()
    # Both X3 and Y3 → binding = 3 in both slots ✓
    assert is_rule_admissible(
        sig, left_cat='X', left_order=3,
        right_cat='Y', right_order=3)
    # X3 and Y4 → binding=3 vs 4 ✗
    assert not is_rule_admissible(
        sig, left_cat='X', left_order=3,
        right_cat='Y', right_order=4)


def test_variable_with_delta_shifts_binding():
    """``rhs[1] = *+1`` means ``operand_order = binding + 1``."""
    from embed import is_rule_admissible
    sig = {
        'lhs_category': 'S',
        'lhs_order': 0,
        'lhs_order_kind': 'variable',
        'rhs_categories': ['X', 'Y'],
        'rhs_orders': [0, 1],
        'rhs_order_kinds': ['variable', 'variable'],
        'op_name': 'pi',
        'order_delta': 0,
    }
    # binding from X3 = 3; Y must be at 3+1=4
    assert is_rule_admissible(
        sig, left_cat='X', left_order=3,
        right_cat='Y', right_order=4)
    assert not is_rule_admissible(
        sig, left_cat='X', left_order=3,
        right_cat='Y', right_order=3)


def test_unary_variable_slot_accepts_any_order():
    """For unary rules, a variable RHS slot accepts any operand
    order."""
    from embed import is_rule_admissible
    sig = {
        'lhs_category': 'NP',
        'lhs_order': 0,
        'lhs_order_kind': 'variable',
        'rhs_categories': ['N'],
        'rhs_orders': [0],
        'rhs_order_kinds': ['variable'],
        'op_name': 'identity',
        'order_delta': 0,
    }
    for k in (0, 1, 2, 5):
        assert is_rule_admissible(sig, left_cat='N', left_order=k)


# test_resolve_lhs_order_* removed: ``SymbolicSubSpace._resolve_lhs_order``
# was retired with the signal-router parser cleanup.


def test_legacy_signature_without_kinds_treats_as_constants():
    """Old artifacts written before the Kleene restoration omit
    ``rhs_order_kinds``; the admissibility check must default those
    slots to ``constant`` so existing behavior is preserved."""
    from embed import is_rule_admissible
    sig = {
        'lhs_category': 'S',
        'lhs_order': 4,
        # no lhs_order_kind / rhs_order_kinds
        'rhs_categories': ['NP', 'VP'],
        'rhs_orders': [3, 1],
        'op_name': 'lift',
        'order_delta': 1,
    }
    assert is_rule_admissible(
        sig, left_cat='NP', left_order=3,
        right_cat='VP', right_order=1)
    assert not is_rule_admissible(
        sig, left_cat='NP', left_order=4,
        right_cat='VP', right_order=1)
