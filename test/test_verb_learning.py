"""Kept observations credit a retained verb event once, across sentences."""
from chunk_fixtures import context as chunk_context
from dataclasses import replace
from types import SimpleNamespace

import torch

from Layers import Error
from MeaningKernels import MeaningAction
from Memory import MediumTermMemory
from VerbLearning import observe


def record(f, tree, frames=()):
    """Fixed operations over prepended identities and the production journal."""
    from chunk_fixtures import record as record_chunks
    return record_chunks(f,tree,frames=frames,names=('subject',))[0]



def test_two_kept_sentences_join_the_copular_observation_and_teach_once(monkeypatch):
    from ClauseJournal import finish_clause
    from ReferenceContext import SituationAddress
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    action = MeaningAction(8, capacity=2)
    for word, value in (('subject', .2), ('move', .3), ('property', .4)):
        symbol = f.noun(word)
        with torch.no_grad():
            f.cs.similarity_codebook.W[f.cs._csw_row_of(symbol)].fill_(value)
    f.ops['lift'] = SimpleNamespace(compose=lambda a, b: action(a, b))
    resolver, errors = f.cs.components.reference_close, Error()
    def close(tree, frames=(), position=0):
        clause = finish_clause(f.language, record(f, tree, frames), registry=f.registry)
        plan = resolver.preview(clause, chunk_context(resolver, frames))
        kept = resolver.prepare(clause, plan)
        written = {}
        row = f.store.write_clause(kept, written_rows=written, )
        resolver.commit(plan, row, written, document='teacher', position=position, training=True,
                        verb_action=action)
        return SituationAddress(f.store.occurrence_of(row),
            symbols=written.get(('symbols', id(kept)), ()),
            activations=written.get(('activations', id(kept)), ())).read(f.store)
    frame = close(('lift', ('lower_unary', 'subject'), 'move'))
    token = next(key for key, entry in f.cs.mtm.entries.items() if entry.events)
    assert not resolver.last_verb_observation
    close(('part', ('bind', 'subject'), 'property'), (frame,), position=1)
    assert len(resolver.last_verb_observation) == 1
    assert resolver.last_verb_observation[0]['token'] == token
    torch.testing.assert_close(f.cs.mtm.value_of(token), torch.full((8,), .4))
    assert sum(event.taught for event in f.cs.mtm.entries[token].events) == 1
    assert errors.total(objective='reconstruction') is None
    assert action.readout.grad is None


def test_observed_state_trains_readout_once_without_concept_gradients():
    mtm = MediumTermMemory()
    token, event = ('ltm', 'unit', 1), ('ltm', 'unit', 2)
    before = torch.tensor([.2, .3, 0., .4, .2, .3, 0., .4], requires_grad=True)
    verb = torch.full((8,), .2, requires_grad=True)
    action = MeaningAction(8, capacity=2)
    from memory_fixtures import attach, put
    owner, store = attach(mtm, 8, 'verb-state')
    put(store, token, before)
    put(store, event, before)
    mtm.open(token, kind=11, value=before, when=1)
    mtm.event(token, event, verb=12, role=0, when=2, before=before, basis=-1, verb_meaning=verb)
    previous = dict(mtm.entries)
    observed = torch.tensor([.3, .4, .8, .45, .3, .4, .8, .45], requires_grad=True)
    following = ('ltm', store.address_domain, 3)
    put(store, following, observed)
    mtm.bind(token, following, kind=11, when=3, value=observed)
    errors = Error()
    report = observe(action, mtm, previous, {token})
    assert len(report) == 1 and report[0]['defined'] == 6
    assert errors.total(objective='reconstruction') is None
    assert action.readout.grad is None
    assert before.grad is None and observed.grad is None and verb.grad is None
    expected = action.teacher(before, observed, verb, group=torch.tensor(-1))
    assert report[0]['error'] == float(expected.detach())
    assert observe(action, mtm, dict(mtm.entries), {token}) == []
    copy = MediumTermMemory()
    copy.restore(mtm.state())
    assert copy.entries[token].events[0].taught
    torch.testing.assert_close(copy.entries[token].events[0].before, before.detach())


def test_copular_join_is_kept_and_fires_the_previous_verb_teacher(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from test_identity_threading import recorded_identity_reading
    from ClauseJournal import finish_clause
    from test_medium_term_memory import keep, mention
    from ReferenceContext import SituationAddress
    from ClauseRow import Clause, ClausePredicate
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    action = MeaningAction(8, capacity=2)
    for word, value in (('subject', .2), ('move', .3), ('property', .4)):
        symbol = f.noun(word)
        with torch.no_grad():
            f.cs.similarity_codebook.W[f.cs._csw_row_of(symbol)].fill_(value)
    f.ops['lift'] = SimpleNamespace(compose=lambda a, b: action(a, b))
    entry, _, _ = recorded_identity_reading(f, ('lift', ('lower_unary', 'subject'), 'move'))
    clause = finish_clause(f.language, entry, registry=f.registry)
    resolver = f.cs.components.reference_close
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, )
    resolver.commit(plan, row, written, document='unit', position=0, training=True, verb_action=action)
    token = next(key for key, value in f.cs.mtm.entries.items() if any(event.before is not None for event in value.events))
    prior = f.cs.mtm.value_of(token).clone()
    # The second sentence is a fixed part operation on the retained subject.
    from chunk_fixtures import Walk,choice,value
    from ChunkJournal import ChunkJournal
    walk=Walk((SimpleNamespace(method_name='part',relation_kind='part'),),())
    state=walk.empty(prior[None],1)
    state=walk.shift(state,prior[None],word=12,index=0,order=1)
    stm=walk.journal.publish(state[0],torch.tensor([512]),torch.tensor([0]),prior[None],torch.tensor([True]))
    state=(stm,*state[1:])
    property_value=torch.full_like(prior,.4)
    state=walk.shift(state,property_value[None],word=13,index=1,order=0)
    previous=dict(f.cs.mtm.entries)
    state=walk.apply(state,choice(property_value[None]),torch.stack((prior,property_value))[None],0)
    joined=value(state,512)[0]
    torch.testing.assert_close(joined,torch.maximum(prior,property_value))
    changed = f.store.append_idea(joined)
    f.cs.mtm.bind(token, f.store.occurrence_of(changed), kind=None, when=3, value=joined)
    f.cs.mtm.record_link(token, f.store.occurrence_of(changed), None)
    errors = Error()
    from IdentityRows import point
    reports = observe(action, f.cs.mtm, previous, {token})
    assert len(reports) == 1
    assert errors.total(objective='reconstruction') is None
    assert action.readout.grad is None
    remembered = previous[token].events[-1]
    assert remembered.verb_meaning is not None
    error = action.teacher(remembered.before, f.cs.mtm.value_of(token), remembered.verb_meaning)
    assert reports[0]['error'] == float(error.detach())
