"""AF: a lexical predicate over a nested relation has no inventory identity."""
from dataclasses import replace
import torch

from reading_fixtures import finish_reading
from test_clause_acceptance import SentenceFixture


def test_unindexed_lexical_predicate_over_a_relation_has_an_ltm_reference(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', ('part', 'cat', 'animal'), 'said')))
    entry = replace(entry, concept_ids=torch.full_like(entry.concept_ids, -1),
                    word_ids=torch.full_like(entry.word_ids, -1))
    clause = finish_reading(f.language, entry, registry=f.registry)
    before = set(f.store.parts.indices().tolist())
    row = f.store.write_clause(clause, trust=.7)
    assert isinstance(row, tuple) and f.store.parts.contains(row[1])
    from LTM1 import OCCURRENCE, RELATION, SYMBOL
    assert torch.isin(f.store.parts.ref_kinds[f.store.relations()],
                      torch.tensor([OCCURRENCE, RELATION, SYMBOL])).all()
    state = int(f.store.row(row)['refs'][1])
    predicate = int(f.store.parts.refs[state, 1])
    from ChainRead import point
    tag = int(f.store.parts.ref_kinds[state, 1])
    assert tag == OCCURRENCE
    torch.testing.assert_close(point(f.store, (tag, predicate)), entry.leaves[3])
    children = sorted(set(f.store.parts.indices().tolist()) - before - {row[1]})
    assert not f.store.parts.truth_plus[children].any()


def test_nested_relation_forest_keeps_its_unindexed_predicate(monkeypatch):
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'he', ('verb', ('part', 'cat', 'animal'), 'said')))
    refs = entry.concept_ids.clone()
    refs[3] = -1
    words = entry.word_ids.clone(); words[3] = -1
    entry = replace(entry, concept_ids=refs, word_ids=words)
    clause = finish_reading(f.language, entry, registry=f.registry)
    before = set(f.store.parts.indices().tolist())
    row = f.store.write_clause(clause, trust=.7)
    assert isinstance(row, tuple) and f.store.parts.contains(row[1])
    from LTM1 import OCCURRENCE, RELATION, SYMBOL
    assert torch.isin(f.store.parts.ref_kinds[f.store.relations()],
                      torch.tensor([OCCURRENCE, RELATION, SYMBOL])).all()
    state = int(f.store.row(row)['refs'][1])
    identity = int(f.store.parts.refs[state, 1])
    from ChainRead import point
    tag = int(f.store.parts.ref_kinds[state, 1])
    assert tag == OCCURRENCE
    torch.testing.assert_close(point(f.store, (tag, identity)), entry.leaves[3])
    children = sorted(set(f.store.parts.indices().tolist()) - before - {row[1]})
    assert not f.store.parts.truth_plus[children].any()
