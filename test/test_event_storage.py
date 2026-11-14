"""A kept event keeps its result, field and source authority through closing."""
from dataclasses import replace
import pytest
import torch
from ClauseJournal import finish_clause
from Extension import ConceptSet, Node
from reading_fixtures import record_reading
from test_clause_acceptance import SentenceFixture


@pytest.mark.parametrize('relation', ('part', 'equal'))
def test_event_uses_its_kept_result_after_a_verb_projects_the_relation(monkeypatch, relation):
    f = SentenceFixture(monkeypatch)
    reading = record_reading(f.language, f.program(('verb', (relation, 'member', 'kind'), 'action')))
    events = torch.zeros(len(reading.actions), 6, dtype=torch.long)
    events[2] = events.new_tensor((0, 1, 0, -1, 0, 1))
    reading = replace(reading, identity_actions=events,
        identity_origins=torch.tensor([[1, f.noun('member'), 0]]),
        identity_ids=torch.tensor([f.noun('member')]), identity_values=reading.leaves[:1],
        identity_leaves=torch.zeros(len(reading.leaves), dtype=torch.long))
    clause = finish_clause(f.language, reading, registry=f.registry)
    assert clause.identity['action'] == 2
    result = clause.identity['result']
    from IdentityRows import carrier
    assert not torch.equal(carrier(f.cs, result), carrier(f.cs, clause.composition.point))
    written = {}
    f.store.write_clause(clause, trust=.7, written_rows=written)
    event = written.get('root_event_occurrence') or written['root_occurrence']
    actual = f.store.slots[f.store.location(event)].clone()
    if f.store.when_start is not None:
        actual[f.store.when_start:f.store.when_start+4] = result[f.store.when_start:f.store.when_start+4]
    torch.testing.assert_close(carrier(f.cs, actual), carrier(f.cs, result), rtol=0, atol=0)
    torch.testing.assert_close(clause.identity['result_composition'].point, result, rtol=0, atol=0)


@pytest.mark.parametrize('trust', (0., .4))
def test_embedded_event_takes_its_sentences_authority(monkeypatch, trust):
    f = SentenceFixture(monkeypatch)
    child = f.clause(('part', 'member', 'kind'))
    child = replace(child, evidence=(.8, .25), identity=dict(
        event=(0, 1, 0, -1, 0, True), result=child.composition.children[1].point))
    root = replace(f.clause(('union', 'first', 'second')), companions=(child,))
    written = {}
    f.store.write_clause(root, trust=trust, written_rows=written)
    event = written['event_occurrence', id(child)]
    group = written['occurrence_groups'][event]
    assert [float(f.store.trust[f.store.location(row)]) for row in group] == pytest.approx(
        [.8*trust, -.25*trust] if trust else [0.])


def test_event_close_carries_its_own_operated_field(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'member', 'kind'))
    point = clause.composition.children[1].point
    field = Node(ConceptSet(torch.tensor([f.noun('kind')]), torch.tensor([[.6, .2]])))
    result = replace(clause.composition.children[1], extension=(field,))
    clause = replace(clause, identity=dict(event=(0, 1, 0, -1, 0, True),
        result=point, result_composition=result))
    meanings = []
    append = f.store.append_meaning
    def capture(meaning, **kwargs):
        meanings.append(meaning)
        return append(meaning, **kwargs)
    monkeypatch.setattr(f.store, 'append_meaning', capture)
    f.store.write_clause(clause, trust=.7)
    assert meanings[-1].extension == (field,)
