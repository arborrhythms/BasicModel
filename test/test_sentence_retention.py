"""The clause closing preserves prediction provenance and retrieval leaves."""
from types import SimpleNamespace

import torch
import pytest

from Layers import MeaningExpectation
from test_clause_storage import clause_store, idea_clause, part_clause


def comparison(store, source):
    from meaning_fixtures import symbol_prediction
    from Layers import ExpectationComparison
    prior = symbol_prediction(torch.eye(4), torch.full((4,2), .5))
    actual = torch.zeros(4,2); actual[0,0] = 1.
    return ExpectationComparison(prior, prior.roles, torch.tensor([True,False,False]),
        actual-prior.expected_activation, 'a', (store.occurrence_of(source),),
        ('external','a'), activation=actual)


def test_idea_closing_initializes_activity_from_live_surprise():
    store,_ = clause_store()
    first = store.write_clause(idea_clause())
    surprise = comparison(store,first).surprise
    target = store.write_clause(idea_clause(),surprise=surprise)
    assert len(store) == 2 and target == 1
    assert float(store.activity[target]) == pytest.approx(surprise)
    torch.testing.assert_close(store.slots[target],idea_clause().point)


def test_narrow_idea_retains_its_identity_without_a_predicate_operand():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    from Occurrence import operand_name
    second = store.write_clause(idea_clause(refs=(operand_name(store.row_name(first)),3,-1)))
    assert not hasattr(store, 'refs')
    assert store.meaning_of(second).role_refs == (store.occurrence_of(first), None, None)
