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


def test_idea_closing_keeps_estimate_surprise_and_index_in_the_existing_owner():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    target = store.write_clause(idea_clause(), expectation=comparison(store, first), document_key='next')
    assert len(store) == 3 and target == 1
    pair = store.expectation_pair(target)
    assert pair['source_occurrences'] == (store.occurrence_of(first),)
    assert pair['surprise'] >= 0
    assert 'kind_logit' not in store.expectation_of(1)
    torch.testing.assert_close(store.slots[target], idea_clause().point)


def test_repeated_relation_can_retain_multiple_estimates_without_duplicate_claims():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    target = store.write_clause(part_clause(), expectation=comparison(store, first))
    again = store.write_clause(part_clause(), expectation=comparison(store, first))
    assert again == target
    assert store.relations().tolist() == [target[1]]
    store._validate_expectation_links()
    # Re-reading the same thought address refreshes its one retained estimate.
    assert len(store) == 2
    for index in (1,):
        assert store.expectation_pair(index)['observation_occurrence'] == store.occurrence_of(target)


def test_capacity_requires_room_for_new_estimate_and_observation():
    store, points = clause_store()
    first = store.write_clause(idea_clause())
    for _ in range(store.capacity - 2):
        store.append_idea(points[1], sentence_index=len(store))
    with pytest.raises(OverflowError, match='capacity'):
        store.write_clause(idea_clause(), expectation=comparison(store, first), document_key='last')
    assert len(store) == store.capacity - 1


def test_narrow_idea_retains_its_identity_without_a_predicate_operand():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    second = store.write_clause(idea_clause(refs=(int(store.row_ids[first]), 3, -1)))
    assert not hasattr(store, 'refs')
    assert store.meaning_of(second).role_refs == (store.occurrence_of(first), None, None)


def test_invalid_prediction_does_not_publish_embedded_clauses():
    store, _ = clause_store()
    source = store.write_clause(idea_clause())
    pending = comparison(store, source)
    pending.estimate.presence_logits.fill_(float('nan'))
    with pytest.raises(ValueError):
        store.write_clause(idea_clause(children=(idea_clause(),)), expectation=pending)
    assert len(store) == 1
