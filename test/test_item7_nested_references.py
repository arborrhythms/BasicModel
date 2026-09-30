"""AF: a lexical predicate over a nested relation has no inventory identity."""
from dataclasses import replace
import torch

from reading_fixtures import finish_reading
from test_item7_acceptance import SentenceFixture


def test_unindexed_lexical_predicate_over_a_relation_has_an_ltm_reference(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', 'said', ('part', 'cat', 'animal'))))
    entry = replace(entry, concept_ids=torch.full_like(entry.concept_ids, -1))
    clause = finish_reading(f.language, entry, registry=f.registry)
    row = f.store.write_clause(clause, trust=.7)
    assert row >= 0 and f.store.rel_type[row] == f.store.REL_OPERATOR
    assert (f.store.refs[f.store.relations()] > 0).all()
    predicate = int(f.store.refs[row, 1])
    assert f.store.index_of_row(predicate) is not None
    torch.testing.assert_close(f.store.point_of_row(predicate), entry.leaves[1])
    assert not f.store.trust[:row].any()


def test_nested_relation_forest_keeps_its_unindexed_predicate(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', 'said', ('part', 'cat', 'animal'))))
    refs = entry.concept_ids.clone()
    refs[1] = -1
    entry = replace(entry, actions=entry.actions[:-2], concept_ids=refs)
    clause = finish_reading(f.language, entry, registry=f.registry, depth=3)
    row = f.store.write_clause(clause, trust=.7)
    assert row >= 0 and f.store.rel_type[row] == f.store.REL_OPERATOR
    assert (f.store.refs[f.store.relations()] > 0).all()
    identity = int(f.store.refs[row, 1])
    assert f.store.index_of_row(identity) is not None
    torch.testing.assert_close(f.store.point_of_row(identity), entry.leaves[1])
    assert not f.store.trust[:row].any()
