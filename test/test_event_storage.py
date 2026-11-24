"""A kept event keeps its result, field and source authority through closing."""
from dataclasses import replace
import pytest
import torch
from ClauseJournal import finish_clause
from Extension import ConceptSet, Node
from reading_fixtures import record_reading
from test_clause_acceptance import SentenceFixture


@pytest.mark.parametrize('relation', ('part',))
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
    source = f.cs.mtm.idea_store(event, f.store)
    actual = source.slots[source.location(event)].clone()
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
    source = f.cs.mtm.idea_store(event, f.store)
    assert source is not f.store and f.store.location(event) is None
    assert [float(source.trust[source.location(row)]) for row in group] == pytest.approx(
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
    staged = f.cs.mtm.staging_store(f.store)
    append = staged.append_meaning
    def capture(meaning, **kwargs):
        meanings.append(meaning)
        return append(meaning, **kwargs)
    monkeypatch.setattr(staged, 'append_meaning', capture)
    f.store.write_clause(clause, trust=.7)
    assert meanings[-1].extension == (field,)


def test_staged_action_outcome_survives_restore_and_expires_with_its_participant(monkeypatch):
    from Memory import MediumTermMemory
    f = SentenceFixture(monkeypatch)
    pool = f.cs.mtm
    clause = f.clause(('part', 'member', 'kind'))
    value = clause.composition.children[1].point
    clause = replace(clause, identity=dict(event=(0, 1, 0, -1, 0, True), result=value))
    written = {}
    relation = f.store.write_clause(clause, trust=.7, written_rows=written)
    address = written['root_event_occurrence']
    assert isinstance(relation, tuple) and f.store.location(address) is None
    subject = f.store.append_idea(value)
    subject = f.store.occurrence_of(subject)
    pool.open(subject, kind=None, when=0, value=value)
    pool.event(subject, address, verb=1, role=0, when=1, before=value)
    pool.close(f.store, stream=0, references=(subject,))
    saved = pool.state()
    recovered = MediumTermMemory(pool.capacity, horizon=pool.horizon,
                                 n_vectors=int(pool.transient_ids[0]), field_width=pool.field_width)
    recovered._owner = pool._owner
    f.cs.mtm = recovered
    recovered.restore(saved)
    event, = recovered.entries[subject].events
    torch.testing.assert_close(recovered.event_value(event), value)
    assert recovered.idea_store(address, f.store).row(0)['trust'] == pytest.approx(.7 * clause.evidence[0])
    for _ in range(recovered.horizon + 1):
        recovered.close(f.store, stream=0)
    assert recovered.idea_store(address, f.store).location(address) is None
    assert f.store.location(relation) == relation


def test_active_mtm_idea_moves_to_ltm0_once_at_its_original_address_and_time():
    from Memory import LTM0, MediumTermMemory
    from RowTime import encoder
    pool = MediumTermMemory()
    durable = LTM0(12, capacity=3)
    durable._model_present = lambda: 9
    staged = pool.staging_store(durable)
    point = torch.linspace(0, 1, 12)
    working_row = staged.append_idea(point, evidence=(.8, 0.), surprise=1.)
    name = staged.occurrence_of(working_row)
    expected = staged.slots[working_row].clone()
    assert torch.equal(expected[-4:], encoder().encode(9))
    later = durable.append_idea(1 - point)
    later_name = durable.occurrence_of(later)
    assert name != later_name and durable.location(name) is None
    durable._model_present = lambda: 10
    pool.close(durable, stream=0)
    row = durable.location(name)
    assert row is not None and staged.location(name) is None
    assert pool.idea_store(name, durable) is durable
    assert torch.equal(durable.slots[row], expected)
    assert float(durable.trust[row]) == pytest.approx(.8)
    pool.close(durable, stream=0)
    assert len(durable) == 2
    next_row = durable.append_idea(point)
    assert durable.occurrence_of(next_row)[-1] > later_name[-1]
