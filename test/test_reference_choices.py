"""Unpriced objects, a zero-logit dummy, and sentence-level bind departures."""
from chunk_fixtures import context as chunk_context
from dataclasses import replace

import pytest
import torch

from ReferenceContext import SituationFrame
from ThoughtReferences import open_slots, with_slots
from test_medium_term_memory import keep, mention, snapshot, unchanged
from test_reference_arrivals import named, world


def noun(atoms, words):
    return replace(mention(atoms[0], mode='bind', reference=words['three']),
        subject_word_id=words['three'],
        composition=replace(mention(atoms[0], reference=words['three']).composition.children[1], order=1))


def test_noun_without_candidates_opens_its_own_object_at_the_kept_close():
    resolver, store, atoms, words, _ = world()
    clause = noun(atoms, words)
    before = snapshot(resolver)
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    kept = resolver.prepare(clause, plan)
    unchanged(resolver, before)
    assert not resolver.components.owner.mtm.entries
    written = {}
    row = store.write_clause(kept, written_rows=written)
    before = snapshot(resolver)
    resolver.commit(plan, row, written, document='noun', position=0, training=False)
    unchanged(resolver, before)
    token = store.occurrence_of(row)
    entry = resolver.components.owner.mtm.entries[token]
    assert plan['items'][0]['reference'] == token
    assert entry.occurrences == (token,) and entry.kind == words['three']
    assert plan['decisions'][0]['selected'] == 0
    assert plan['decisions'][0]['logits'].tolist() == [0.]


@pytest.mark.parametrize('weight', [0., -1.])
def test_definite_with_nonpositive_best_candidate_keeps_its_own_object(weight):
    resolver, store, atoms, words, _ = world()
    first, _ = keep(resolver, mention(atoms[0], reference=words['three']))
    previous = store.occurrence_of(first)
    pool = resolver.components.owner.mtm
    entry = pool.entries[previous]
    with torch.no_grad():
        resolver.policy.fill_(weight)
    row, plan = keep(resolver, noun(atoms, words), (SituationFrame.from_row(store.row(first)),), position=1)
    token = store.occurrence_of(row)
    decision = plan['decisions'][0]
    assert decision['logits'][0] == 0 and decision['logits'][1:].max() <= 0
    assert decision['selected'] == 0 and pool.token_of(token) == token != previous
    assert pool.entries[previous] == entry
    assert pool.entries[token].kind == words['three']


def test_bind_departure_takes_next_option_and_has_the_shared_r_plus_a_credit():
    from SentenceCredit import departure, score_function, comparison
    resolver, store, atoms, words, _ = world()
    first, _ = keep(resolver, mention(atoms[0], reference=words['three']))
    frames = (SituationFrame.from_row(store.row(first)),)
    # The only eligible walk is this bind. The draw changes no grammar scope.
    draw = departure(None, torch.zeros(1, 0, dtype=torch.bool), active=torch.tensor([True]),
                     binds=torch.ones(1, 1, dtype=torch.bool))
    assert draw['walk'].tolist() == [3] and draw['bind_round'].tolist() == [0]
    greedy = resolver.preview(noun(atoms, words), chunk_context(resolver, frames))
    explore = resolver.preview(noun(atoms, words), chunk_context(resolver, frames), departure=int(draw['bind_round'][0]))
    a, b = greedy['decisions'][0], explore['decisions'][0]
    assert a['selected'] == 1 and b['selected'] == b['next'] == 0 and b['departed']
    torch.testing.assert_close(a['logits'], b['logits'])
    # A supplied-referent failure is the advantage; the local menu cost is absent.
    parts = torch.tensor([[[0., 0., 0.], [0., 0., 1.]]])
    audit = comparison(parts, torch.tensor([True]))
    probability = b['probabilities'][b['selected']].reshape(1, 1)
    loss, probe = score_function(probability, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), audit['costs'])
    loss.sum().backward()
    assert resolver.policy.grad is not None and resolver.policy.grad[1] < 0
    assert audit['keep_decision'] == ['greedy'] and probe['mask'].all()
    tied, _ = score_function(probability, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool),
                            torch.zeros(1, 2))
    assert not tied.requires_grad


def test_gap_with_dummy_stays_open_without_an_object():
    resolver, store, atoms, words, _ = world()
    first, _ = keep(resolver, mention(atoms[0], reference=words['three']))
    pool = resolver.components.owner.mtm
    entries = dict(pool.entries)
    with torch.no_grad():
        resolver.policy.zero_()
    row, plan = keep(resolver, mention(atoms[0]*0, mode='elision'),
                     (SituationFrame.from_row(store.row(first)),), position=1)
    assert plan['decisions'][0]['selected'] == 0
    assert plan['items'][0]['reference'] is None
    assert ('referent', 0) in open_slots(store.meaning_of(row))
    assert pool.entries == entries and pool.token_of(store.occurrence_of(row)) is None


def test_supplied_referent_judges_the_trials_bind_without_writing_the_pool():
    from BindingAnswers import cost
    resolver, store, atoms, words, model = world()
    first, _ = keep(resolver, mention(atoms[0], reference=words['three']))
    clause = noun(atoms, words)
    meaning = with_slots(clause.meaning, (('referent', 0),), pair=(0., 0.))
    frame = SituationFrame.from_row(store.row(first))
    before, entries = snapshot(resolver), dict(resolver.components.owner.mtm.entries)
    greedy = resolver.preview(clause, chunk_context(resolver, (frame,)))
    explore = resolver.preview(clause, chunk_context(resolver, (frame,)), departure=0)
    assert cost(model, resolver.answer_meaning(greedy, meaning), 'three') == 0
    assert cost(model, resolver.answer_meaning(explore, meaning), 'three') == 1
    unchanged(resolver, before)
    assert resolver.components.owner.mtm.entries == entries
