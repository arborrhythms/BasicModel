"""Existence evidence must match the complete conceptual description."""
from Memory import STM0
from types import SimpleNamespace

import pytest
import torch

from Memory import LTM0
from reasoning import QuerySpec, TruthGroundedReasoner, TRUE, UNKNOWN, BOTH


def _description():
    return torch.eye(6)[:3].clone()




def test_addressed_claims_preserve_conflicting_support():
    from existence_fixtures import read_claim
    subject = _description()[0]
    store = STM0(6, capacity=8, relation_capacity=8192)
    positive = store.append_idea(subject, trust=.8)
    negative = store.append_idea(subject, trust=-.9)
    assert read_claim(store,positive)['support_true'] == pytest.approx(.8)
    assert read_claim(store,negative)['support_false'] == pytest.approx(.9)


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
