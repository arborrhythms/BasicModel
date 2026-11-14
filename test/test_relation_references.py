"""A lexical predicate over a relation closes as an idea, without asserting it."""
from dataclasses import replace
import pytest
import torch

from reading_fixtures import finish_reading
from test_clause_acceptance import SentenceFixture


def test_unindexed_predicate_over_a_relation_closes_as_an_idea(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', ('part', 'cat', 'animal'), 'said')))
    entry = replace(entry, concept_ids=torch.full_like(entry.concept_ids, -1),
                    word_ids=torch.full_like(entry.word_ids, -1))
    clause = finish_reading(f.language, entry, registry=f.registry)
    before = set(f.store.parts.indices().tolist())
    row = f.store.write_clause(clause, trust=.7)
    assert isinstance(row, int) and 0 <= row < len(f.store)
    from Memory import OCCURRENCE, RELATION, SYMBOL
    assert torch.isin(f.store.parts.ref_kinds[f.store.relations()],
                      torch.tensor([OCCURRENCE, RELATION, SYMBOL])).all()
    torch.testing.assert_close(f.store.slots[row], clause.point)
    assert float(f.store.trust[row]) == pytest.approx(.7)
    children = sorted(set(f.store.parts.indices().tolist()) - before)
    assert not f.store.parts.truth_plus[children].any()


def test_nested_relation_forest_closes_as_an_idea_without_asserting_children(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', ('part', 'cat', 'animal'), 'said')))
    refs = entry.concept_ids.clone()
    refs[3] = -1
    words = entry.word_ids.clone(); words[3] = -1
    entry = replace(entry, concept_ids=refs, word_ids=words)
    clause = finish_reading(f.language, entry, registry=f.registry)
    before = set(f.store.parts.indices().tolist())
    row = f.store.write_clause(clause, trust=.7)
    assert isinstance(row, int) and 0 <= row < len(f.store)
    from Memory import OCCURRENCE, RELATION, SYMBOL
    assert torch.isin(f.store.parts.ref_kinds[f.store.relations()],
                      torch.tensor([OCCURRENCE, RELATION, SYMBOL])).all()
    torch.testing.assert_close(f.store.slots[row], clause.point)
    assert float(f.store.trust[row]) == pytest.approx(.7)
    children = sorted(set(f.store.parts.indices().tolist()) - before)
    assert not f.store.parts.truth_plus[children].any()
