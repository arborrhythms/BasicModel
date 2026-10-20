"""Known referents train existing form/content weights, never a new scorer."""
from dataclasses import replace

import torch

from ClauseRow import Clause
from Layers import Error
from Meaning import ConceptualMeaning
from ThoughtReferences import bindings, question
from test_reference_close import fixture, close, form_exposure


def context():
    resolver, store, atoms = fixture()
    resolver.components.weight = 1.
    frames = []
    for index in range(3):
        frame, _, _ = close(resolver, atoms[index], atoms[index+6], 'mint', frames,
                            document='mentions', sentence=index)
        frames.append(frame)
    form_exposure(resolver, atoms[4], atoms[0])
    request = dict(mode='bind', leaves=(0,), cue=atoms[4], predicate=atoms[6], role=0)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
                    order=1, reference_request=request)
    return resolver, store, atoms, resolver.preview(clause, frames)


def test_mention_objective_marginalizes_all_correct_antecedents_and_updates_only_form_content():
    resolver, _, _, plan = context()
    item = plan['items'][0]
    correct = set(item['candidates'][:2])
    logits = resolver.logits(item['features'], item['cue'])
    expected = -logits.softmax(0)[:2].sum().log()
    loss = resolver.mention_loss(item, correct)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert resolver.policy.grad[0] != 0
    assert resolver.policy.grad[[2,3,5,6,7]].tolist() == [0.] * 5
    assert bool(resolver.policy.grad[4].ne(0))
    assert resolver.mention_loss(item, {-999}) is None
    all_correct = resolver.mention_loss(item, set(item['candidates']))
    assert float(all_correct) == 0.


def test_supplied_answer_labels_only_its_role_without_reselecting_the_menu():
    resolver, _, atoms, plan = context()
    item = plan['items'][0]
    correct = item['candidates'][1]
    resolver.components.owner.word_concepts = lambda _: (correct,)
    original = item['selected']
    meaning = question(ConceptualMeaning.from_description(atoms[0]),
                       (('referent', 0),))
    loss = resolver.answer_loss(plan, meaning, 'supplied answer')
    assert loss is not None and loss.requires_grad
    assert item['selected'] == original
    wrong_role = question(meaning, (('referent', 2),))
    assert resolver.answer_loss(plan, wrong_role, 'supplied answer') is None
    optimizer = torch.optim.SGD(resolver.parameters(), lr=.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert resolver.answer_loss(plan, meaning, 'supplied answer') < loss.detach()


def test_later_fill_trains_the_same_occurrence_menu_without_a_live_graph():
    resolver, store, atoms, plan = context()
    clause = plan['items'][0]['node']
    kept = resolver.prepare(clause, plan)
    written = {}
    row = store.write_clause(kept, document_key='pending', sentence_index=0, written_rows=written)
    resolver.commit(plan, row, written, document='pending', position=0, training=False)
    original = store.meaning_of(row)
    assert resolver._mentions[store.occurrence_of(row)]
    rows, refs = store.slots.clone(), store.refs.clone()
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
    loss = errors.total(objective='expectation')
    assert loss is not None and loss.requires_grad
    loss.backward()
    assert bool(resolver.policy.grad[4].ne(0))
    assert resolver.policy.grad[0] != 0
    # Filling it twice cannot produce duplicate credit.
    resolver.resolved(row, 0, ('sym', correct), errors)
    assert errors._terms['expectation.reference_mentions']['count'] == 1
    torch.testing.assert_close(store.slots, rows, rtol=0, atol=0)
    torch.testing.assert_close(store.refs, refs, rtol=0, atol=0)


def test_mention_training_record_is_not_a_thought_chooser_feature():
    from ThoughtFeatures import semantic_metadata
    meaning = ConceptualMeaning.from_description(torch.ones(4))
    held = replace(meaning, bindings={'_reference_mentions': ((0, {
        'candidates': (731, 907), 'features': ((.2, .3, .4, .5, .6),)*2,
        'cue': (1., 0., 0., 0.)}),)})
    for base, actual in zip(semantic_metadata((meaning,)*3), semantic_metadata((held,)*3)):
        torch.testing.assert_close(actual, base, rtol=0, atol=0)
