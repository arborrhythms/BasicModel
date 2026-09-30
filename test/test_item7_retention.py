"""The clause closing preserves prediction provenance and retrieval leaves."""
from types import SimpleNamespace

import torch
import pytest

from Layers import MeaningExpectation
from test_item7_storage import clause_store, idea_clause, part_clause


def comparison(store, source):
    return SimpleNamespace(estimate=MeaningExpectation(torch.ones(3, 4) * .2,
        torch.zeros(3), kind_logit=torch.tensor(.3)),
        source_occurrences=(store.occurrence_of(source),),
        stream=('external', 'a'), document='a')


def test_idea_closing_keeps_estimate_surprise_and_index_in_the_existing_owner():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    store.configure_leaf_index(unfold=lambda field, limit, **kwargs: ((11, 12, 13), 1, True))
    target = store.write_clause(idea_clause(), expectation=comparison(store, first))
    assert len(store) == 3 and target == 2
    pair = store.expectation_pair(target)
    assert pair['source_occurrences'] == (store.occurrence_of(first),)
    assert pair['surprise'] >= 0
    assert store.expectation_of(1)['kind_logit'] == float(torch.tensor(.3))
    assert tuple(store.leaf_terms(target, role) for role in range(3)) == ((11, 12, 13), (), ())


def test_repeated_relation_can_retain_multiple_estimates_without_duplicate_claims():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    target = store.write_clause(part_clause(), expectation=comparison(store, first))
    again = store.write_clause(part_clause(), expectation=comparison(store, first))
    assert again == target
    assert store.relations(store.REL_PARTOF).tolist() == [target]
    store._validate_expectation_links()
    for index in (1, 3):
        assert store.expectation_pair(index)['observation_occurrence'] == store.occurrence_of(target)


def test_capacity_prefers_observation_to_optional_estimate():
    store, points = clause_store()
    first = store.write_clause(idea_clause())
    for _ in range(store.capacity - 2):
        store.append_idea(points[1])
    last = store.write_clause(idea_clause(), expectation=comparison(store, first))
    assert last == store.capacity - 1
    assert store.row(last)['kind'] == 'observation'
    assert store.row(last)['surprise'] >= 0


def test_idea_chain_is_a_retention_reference_to_its_prior_state():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    second = store.write_clause(idea_clause(refs=(int(store.row_ids[first]), 3, -1)))
    assert store.refs[second, 0] == store.row_ids[first]
    assert store.meaning_of(second).role_refs == (None, None, None)


def test_invalid_prediction_does_not_publish_embedded_clauses():
    store, _ = clause_store()
    source = store.write_clause(idea_clause())
    pending = comparison(store, source)
    pending.estimate.kind_logit.fill_(float('nan'))
    with pytest.raises(ValueError):
        store.write_clause(idea_clause(children=(idea_clause(),)), expectation=pending)
    assert len(store) == 1
