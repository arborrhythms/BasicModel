"""Existence evidence must match the complete conceptual description."""
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner, TRUE, UNKNOWN, BOTH


def _description():
    return torch.eye(6)[:3].clone()


@pytest.mark.parametrize("changed_role", [None, 1, 2])
def test_exist_matches_the_complete_relational_description(changed_role):
    roles = _description()
    store = TernaryTruthStore(6, capacity=8)
    store.append_relation(*roles, trust=.85)
    probe = roles.clone()
    if changed_role is not None:
        probe[changed_role] = torch.eye(6)[4]
    result = TruthGroundedReasoner(store=store).evaluate(
        QuerySpec.from_surface("exist", probe))
    if changed_role is None:
        assert result["posture"] == TRUE
        assert result["support_true"] == pytest.approx(.85)
        assert result["support_false"] == 0
    else:
        assert result["posture"] == UNKNOWN
        assert result["support_true"] == result["support_false"] == 0


def test_exist_preserves_conflicting_fact_support():
    subject = _description()[0]
    store = TernaryTruthStore(6, capacity=8)
    store.append_idea(subject, trust=.8)
    store.append_idea(subject, trust=-.9)
    result = TruthGroundedReasoner(store=store).evaluate(
        QuerySpec.from_surface("exist", subject))
    assert result["posture"] == BOTH
    assert result["support_true"] == pytest.approx(.8)
    assert result["support_false"] == pytest.approx(.9)


def test_exist_requires_fact_evidence_instead_of_activation_truth():
    calls = []

    def activation_truth(value):
        calls.append(value)
        return .97

    model = SimpleNamespace(conceptualSpace=None, isTrue=activation_truth)
    result = TruthGroundedReasoner(model=model).evaluate(
        QuerySpec.from_surface("exist", _description()[0]))
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == result["support_false"] == 0
    assert calls == []
