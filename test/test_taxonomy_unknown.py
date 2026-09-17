"""A zero acceptance threshold cannot turn absent evidence into truth/conflict."""
import pytest
import torch

from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner


@pytest.mark.parametrize("predicate", ["exist", "isPart"])
def test_no_support_remains_unknown_even_at_zero_threshold(predicate):
    reasoner = TruthGroundedReasoner(store=TernaryTruthStore(8))
    first, second = torch.eye(8)[:2]
    query = QuerySpec.from_surface(predicate, first, second)
    result = reasoner.evaluate(query, tau=0.)
    assert result["support_true"] == result["support_false"] == 0
    assert result["posture"] == "UNKNOWN"


@pytest.mark.parametrize("support, expected", [((1., 0.), "TRUE"), ((0., 1.), "FALSE")])
def test_zero_threshold_cannot_fabricate_the_other_evidence_polarity(support, expected):
    result = TruthGroundedReasoner()._posture(*support, tau=0.)
    assert result["posture"] == expected


def test_empty_legacy_interval_is_unknown_at_zero_threshold():
    from thinking import TruthInterval
    assert TruthInterval().status(tau=0.) == "unknown"
