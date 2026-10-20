"""AF: a lexical predicate over a nested relation has no inventory identity."""
from dataclasses import replace
import torch

from reading_fixtures import finish_reading
from test_clause_acceptance import SentenceFixture


def test_unindexed_lexical_predicate_over_a_relation_has_an_ltm_reference(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', 'said', ('part', 'cat', 'animal'))))
    entry = replace(entry, concept_ids=torch.full_like(entry.concept_ids, -1))
    clause = finish_reading(f.language, entry, registry=f.registry)
    before = set(f.store.parts.indices().tolist())
    row = f.store.write_clause(clause, trust=.7)
    assert isinstance(row, tuple) and f.store.parts.contains(row[1])
    assert (f.store.parts.refs[f.store.relations()] > 0).all()
    state = int(f.store.row(row)['refs'][1])
    predicate = int(f.store.parts.refs[state, 1])
    assert f.store._concept_owner()._csw_row_of(predicate) is not None
    torch.testing.assert_close(f.store._concept_point(predicate), entry.leaves[1])
    children = sorted(set(f.store.parts.indices().tolist()) - before - {row[1]})
    assert not f.store.parts.truth_plus[children].any()


def test_nested_relation_forest_keeps_its_unindexed_predicate(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', 'said', ('part', 'cat', 'animal'))))
    refs = entry.concept_ids.clone()
    refs[1] = -1
    entry = replace(entry, concept_ids=refs)
    clause = finish_reading(f.language, entry, registry=f.registry)
    before = set(f.store.parts.indices().tolist())
    row = f.store.write_clause(clause, trust=.7)
    assert isinstance(row, tuple) and f.store.parts.contains(row[1])
    assert (f.store.parts.refs[f.store.relations()] > 0).all()
    state = int(f.store.row(row)['refs'][1])
    identity = int(f.store.parts.refs[state, 1])
    assert f.store._concept_owner()._csw_row_of(identity) is not None
    torch.testing.assert_close(f.store._concept_point(identity), entry.leaves[1])
    children = sorted(set(f.store.parts.indices().tolist()) - before - {row[1]})
    assert not f.store.parts.truth_plus[children].any()
