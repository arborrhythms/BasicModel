"""Reference departures receive episode credit; supplied answers judge the kept choice."""
from chunk_fixtures import context as chunk_context
from dataclasses import replace

import torch

from ClauseRow import Clause
from Layers import Error
from Meaning import ConceptualMeaning
from ThoughtReferences import bindings
from meaning_fixtures import unsupported as question
from test_reference_close import fixture, close, form_exposure


def context(*, content_atoms=(0,)):
    resolver, store, atoms = fixture()
    resolver.components.weight = 1.
    frames = []
    for index in range(3):
        frame, _, _ = close(resolver, atoms[index], atoms[index+6], 'mint', frames,
                            document='mentions', sentence=index)
        frames.append(frame)
    form_exposure(resolver, atoms[4], atoms[0])
    request = dict(mode='bind', leaves=(0,), cue=atoms[4], predicate=atoms[6], role=0)
    content = atoms[list(content_atoms)].sum(0)
    clause = Clause(ConceptualMeaning.from_description(content), point=content,
                    order=1, reference_request=request)
    return resolver, store, atoms, resolver.preview(clause, chunk_context(resolver, frames))


def test_mention_menu_enters_the_ordinary_departure_with_its_live_probability():
    resolver, _, _, plan = context()
    item = plan['items'][0]
    decision = plan['decisions'][0]
    assert decision['candidates'] == (None, *item['candidates'])
    logits = resolver.logits(item['features'], item['cue'])
    torch.testing.assert_close(decision['logits'][1:], logits)
    selected = decision['selected']
    assert selected > 0 and decision['candidates'][selected] == item['selected']
    credit = -decision['probabilities'][selected].log()
    credit.backward()
    assert resolver.policy.grad is not None and resolver.policy.grad.abs().sum() > 0


def test_answer_judges_selected_roles_without_reselecting_the_mention():
    from ThoughtReferences import open_slots
    resolver, _, atoms, plan = context()
    item = plan['items'][0]
    # Supply a resolved candidate as a compose bind does. The standalone
    # arrival menu above carries MTM tokens that have no durable object yet.
    identity = resolver._fixture_predicates[tuple(atoms[6].tolist())]
    item.update(selected=identity, root_roles=(0,), record=dict(value=atoms[0]))
    meaning = ConceptualMeaning.from_description(atoms[0])
    subject = question(meaning, (('referent', 0),))
    other = question(meaning, (('referent', 2),))
    assert ('referent', 0) not in open_slots(resolver.answer_meaning(plan, subject))
    assert ('referent', 2) in open_slots(resolver.answer_meaning(plan, other))
    assert item['selected'] == identity and resolver.policy.grad is None


def test_later_fill_releases_the_menu_without_an_expectation_loss():
    # §50/§52: both retained alternatives are parts of this mention.
    resolver, store, atoms, plan = context(content_atoms=(0, 1, 2))
    clause = plan['items'][0]['node']
    kept = resolver.prepare(clause, plan)
    written = {}
    row = store.write_clause(kept,   written_rows=written)
    resolver.commit(plan, row, written, document='pending', position=0, training=False)
    original = store.meaning_of(row)
    assert resolver._mentions[store.occurrence_of(row)]
    rows, refs = store.slots.clone(), store.parts.refs.clone()
    # A subsequent known binding refers to this same occurrence and role.
    pending = question(replace(original, role_refs=(None, None, None)),
                       (('referent', 0),))
    errors = Error()
    correct = plan['items'][0]['candidates'][1]
    from ThoughtReferences import fill, open_slots
    resolved = fill(pending, dict(reference=('sym', correct), binding_value=atoms[1]),
                    slots=(('referent', 0),))
    assert ('referent', 0) not in open_slots(resolved)
    resolver.resolved(row, 0, ('sym', correct), errors)
    assert errors.total(objective='expectation') is None
    assert resolver.policy.grad is None
    resolver.resolved(row, 0, ('sym', correct), errors)
    assert store.occurrence_of(row) not in resolver._mentions
    torch.testing.assert_close(store.slots, rows, rtol=0, atol=0)
    torch.testing.assert_close(store.parts.refs, refs, rtol=0, atol=0)


def test_mention_training_record_is_not_a_thought_chooser_feature():
    from ThoughtFeatures import semantic_metadata
    meaning = ConceptualMeaning.from_description(torch.ones(4))
    held = replace(meaning, bindings={'_reference_mentions': ((0, {
        'candidates': (731, 907), 'features': ((.2, .3, .4, .5, .6),)*2,
        'cue': (1., 0., 0., 0.)}),)})
    for base, actual in zip(semantic_metadata((meaning,)*3), semantic_metadata((held,)*3)):
        torch.testing.assert_close(actual, base, rtol=0, atol=0)
