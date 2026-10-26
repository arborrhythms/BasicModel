"""The tenth review's ownership, incompatibility and answer-view contracts."""
from chunk_fixtures import context as chunk_context
from dataclasses import replace

import pytest
import torch

from BindingAnswers import cost
from ClauseRow import Clause, Composition, ClausePredicate, predicate_point
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from ThoughtReferences import open_slots, with_slots
from test_medium_term_memory import keep, mention, snapshot, unchanged, seats, unchanged_after_consolidation
from test_reference_arrivals import equality, named, world
from test_reference_choices import noun


@pytest.mark.parametrize('observed_exclusion', (False, True))
def test_later_ball_mention_preserves_the_boxs_key_members_and_count(monkeypatch, observed_exclusion):
    resolver, store, a, words, _ = world()
    owner, frames = resolver.components.owner, []
    for unit, (word, point) in enumerate((('y', a[5]), ('box', a[0]), ('ball', a[1]),
                                         ('red', a[2]), ('blue', a[3]), ('two', a[4])), 200):
        words[word] = owner.interpret.lookup_word([unit], (), form=word)
        if word != 'y':
            obj = owner.new_concept()
            with torch.no_grad():
                owner.similarity_codebook.W[owner._csw_concept_row(0, obj)].copy_(point)
            owner.interpret.define(words[word], obj)
            words[word] = obj

    def close(clause, position):
        plan = resolver.preview(clause, chunk_context(resolver, tuple(frames)))
        retained = dict(owner.mtm.entries)
        prepared = resolver.prepare(clause, plan)
        assert owner.mtm.entries == retained
        written = {}
        row = store.write_clause(prepared, document_key='box-ball', sentence_index=position,
                                 written_rows=written)
        before, seated = snapshot(resolver), seats(resolver)
        resolver.commit(plan, row, written, document='box-ball', position=position, training=False)
        unchanged_after_consolidation(resolver, before, seated)
        frames.append(replace(SituationFrame.from_row(store.row(row)),
            activations=written.get(('activations', id(prepared)), ()),
            symbol_ids=written.get(('symbols', id(prepared)), ())))
        return plan

    def property_clause(word, predicate):
        point, pred = a[0] if word == 'box' else a[1], a[2] if predicate == 'red' else a[3]
        child = mention(point, reference=words[word])
        return Clause(ConceptualMeaning(torch.stack((point, pred, point*0)),
            torch.tensor([True, True, False])), point=point+pred, order=2,
            children=(child,), refs=(('clause', 0), -1, -1),
            composition=Composition(point+pred, (child.composition,
                Composition(pred, reference=words[predicate])), operation='lift', order=2))

    # y is x plus two: x is nested under the right-hand expression.
    x = named(a[0], words['x'])
    two = Clause(ConceptualMeaning.from_description(a[4]), point=a[4],
        composition=Composition(a[4], reference=words['two'], word_id=words['two']))
    addition = Clause(ConceptualMeaning.from_description(a[4]), point=a[4],
        refs=(('clause', 0), -1, ('clause', 1)), children=(x, two))
    pred = predicate_point('equal', a[0])
    first = Clause(ConceptualMeaning(torch.stack((a[5], pred, a[4])), torch.ones(3, dtype=torch.bool)),
        relation='equal', children=(addition,),
        refs=(words['y'], ClausePredicate(pred, 'equal'), ('clause', 0)))
    pending = close(first, 0)['items'][0]['reference']
    box = close(property_clause('box', 'red'), 1)['items'][0]['reference']
    monkeypatch.setattr(resolver, 'prime', lambda value, stream: value[0]*4+value[1])
    close(equality(a[0], words['x'], words['box']), 2)
    assert owner.mtm.token_of(pending) == box
    ball = close(property_clause('ball', 'blue'), 3)['items'][0]['reference']
    before = owner.mtm.entries[box]
    if observed_exclusion:
        # Eight separate observations of each support; never a joint one.
        for support in (a[0], a[1]):
            row = store.write_clause(Clause(ConceptualMeaning.from_description(support),
                point=support, order=1), document_key=('exclusion', int(support.argmax())),
                sentence_index=0)
            store.witness_count[row] = 8
        assert resolver.exclusion_query()(a[0], a[1])
    else:
        assert not resolver.exclusion_query()(a[0], a[1])
        monkeypatch.setattr(resolver, 'prime', lambda value, stream: value[1]*4+value[0])
    final = close(equality(a[1], words['x'], words['ball']), 4)
    new_mention = final['items'][0]
    assert owner.mtm.token_of(box) == box
    assert owner.mtm.entries[box] == before
    assert owner.mtm.token_of(pending) == box
    assert all(item['site_occurrence'] != pending for item in final['arrivals'])
    assert new_mention['reference'] != box
    if observed_exclusion:
        assert box not in new_mention['candidates']
        assert new_mention['reference'] == ball or new_mention['own_object']


def test_arrival_cannot_merge_a_pending_token_with_other_members():
    resolver, store, a, words, _ = world()
    pending, _ = keep(resolver, named(a[0], words['x']))
    token = store.occurrence_of(pending)
    use, _ = keep(resolver, mention(a[0], reference=words['three']), position=1)
    other, _ = keep(resolver, mention(a[1], reference=words['wrong']), position=2)
    pool = resolver.components.owner.mtm
    pool.bind(token, store.occurrence_of(use), kind=words['three'], when=3)
    before, retained = dict(pool.entries), snapshot(resolver)
    candidate = store.occurrence_of(other)
    item = dict(site=token, site_occurrence=token, token=token, selected=candidate,
                candidates=(candidate,))
    resolver.commit(dict(items=[], arrivals=[item]), None, {},
                    document='review', position=3, training=False)
    assert pool.entries == before
    unchanged(resolver, retained)


@pytest.mark.parametrize('has_word', (False, True))
def test_own_content_excludes_an_observed_incompatible_antecedent(monkeypatch, has_word):
    resolver, store, a, words, _ = world()
    row, _ = keep(resolver, mention(a[0], reference=words['three']))
    antecedent = store.occurrence_of(row)
    for index, support in enumerate((a[0], a[1])):
        observation = store.write_clause(Clause(ConceptualMeaning.from_description(support),
            point=support, order=1), document_key='exclusion', sentence_index=index)
        store.witness_count[observation] = 8
    pool = resolver.components.owner.mtm
    before = pool.entries[antecedent]
    monkeypatch.setattr(resolver, 'prime', lambda value, stream: value.new_tensor(100.))
    clause = mention(a[1], mode='bind')
    if has_word:
        clause = replace(clause, subject_word_id=words['wrong'])
    new, plan = keep(resolver, clause, (SituationFrame.from_row(store.row(row)),), position=1)
    item = plan['items'][0]
    assert antecedent not in item['candidates']
    assert item['reference'] == (store.occurrence_of(new) if has_word else None)
    assert pool.entries[antecedent] == before
    assert pool.token_of(store.occurrence_of(new)) == (store.occurrence_of(new) if has_word else None)
    if not has_word:
        assert ('referent', 0) in open_slots(store.meaning_of(new))


@pytest.mark.parametrize('nested', (False, True))
def test_answer_view_and_loss_judge_only_a_direct_root_site(nested):
    resolver, store, atoms, words, model = world()
    antecedent, _ = keep(resolver, mention(atoms[0], reference=words['three']))
    child = noun(atoms, words)
    if nested:
        # what is x plus two? The nested x is local role zero, not the answer.
        child = Clause(ConceptualMeaning.from_description(atoms[0]+atoms[2]),
            point=atoms[0]+atoms[2], refs=(('clause', 0), -1, -1), children=(child,))
    role = 0 if nested else 2
    meaning = with_slots(ConceptualMeaning(torch.stack((atoms[1], atoms[3], child.point)),
        torch.ones(3, dtype=torch.bool), role_refs=tuple(
            None if slot == role else ('sym', words['wrong']) for slot in range(3))),
        (('referent', role),), pair=(0., 0.))
    clause = Clause(meaning, point=atoms[1], refs=(-1, -1, ('clause', 0)), children=(child,))
    before, entries = snapshot(resolver), dict(resolver.components.owner.mtm.entries)
    plan = resolver.preview(clause, chunk_context(resolver, (SituationFrame.from_row(store.row(antecedent)),)))
    address = store.occurrence_of(antecedent)
    assert plan['items'][0]['selected'] == address
    assert plan['items'][0]['root_roles'] == (() if nested else (2,))
    judged = resolver.answer_meaning(plan, meaning)
    assert cost(model, judged, 'three') == (1 if nested else 0)
    assert bool(open_slots(judged)) == nested
    resolver.components.owner.word_concepts = lambda _: (address,)
    loss = resolver.answer_loss(plan, meaning, 'three')
    assert (loss is None) == nested
    unchanged(resolver, before)
    assert resolver.components.owner.mtm.entries == entries


@pytest.mark.parametrize('answer_weight', (0., 1.))
def test_thought_bind_is_credited_only_by_episode_advantage(monkeypatch, tmp_path, answer_weight):
    import WalkTrials
    from test_bind_thought import native, occurrence, site
    from test_item6_2_thinking import run
    from test_reference_close import form_exposure
    f = native(monkeypatch)
    target, _ = occurrence(f, f.atoms[0], f.atoms[4])
    occurrence(f, f.atoms[1], f.atoms[5])
    form_exposure(f.resolver, f.atoms[7], f.atoms[0])
    meaning = replace(site(f, f.atoms[0], predicate=f.atoms[4], cue=f.atoms[7]))
    chooser, original = f.model.shared_grammar, f.model.shared_grammar.thought_logits
    def prefer_binding(active, requests):
        logits = original(active, requests)
        preferred = [request is None or f.registry.signature_for(
            request, verify_reference=False).operation.semantic_id == 'bind' for request in requests]
        return logits + logits.new_tensor(preferred)*(2*logits.detach().abs().sum()+1)
    monkeypatch.setattr(chooser, 'thought_logits', prefer_binding)
    # Select the departure location to exercise this branch. The candidate
    # scores, dummy, next-ranked option and R+A keep remain live.
    capture = WalkTrials.capture_thought_fork
    def at_bind(model, trace, meter, **continuation):
        if trace['choices'][-1].get('category') == 'expectation':
            with monkeypatch.context() as patch:
                patch.setattr(torch, 'rand', lambda *args, **kwargs: torch.zeros(*args, **kwargs))
                capture(model, trace, meter, **continuation)
    monkeypatch.setattr(WalkTrials, 'capture_thought_fork', at_bind)
    reads, original_read = [], f.model.symbolSpace.expectation.situation_references
    def read(row):
        reads.append(row)
        return original_read(row)
    monkeypatch.setattr(f.model.symbolSpace.expectation, 'situation_references', read)
    f.model.train()
    f.model.errors.clear()
    before = {name: value.clone() for name, value in f.resolver.named_parameters()}
    measured = []
    def score(result):
        answer = answer_weight*float(result.meaning.role_refs[0] != f.identities[target])
        measured.append(answer)
        return dict(reconstruction=0., answer=answer)
    from math_chain_observer import Observer
    with Observer(tmp_path) as observer:
        run(f.model, meaning, work_budget=128, score=score)
    assert observer.credit_sample[-1]['category'] == 'expectation'
    audit = f.model._last_thought_comparison
    assert audit['costs'] == pytest.approx(measured)
    assert audit['keep_costs'] == audit['costs']
    assert audit['explore_kept'] == (measured[1] < measured[0])
    assert len(reads) == 1, 'the bind fork must not reread its paid frame menu'
    first, second = (next(choice for choice in trial if choice['step'] == audit['departure'])
                     for trial in audit['bind_choices'])
    assert first['selected'] == first['greedy'] and not first['departed']
    order = sorted(range(len(second['logits'])), key=lambda index: (-second['logits'][index], index))
    assert second['selected'] == next(index for index in order if index != first['selected'])
    assert second['departed'] and second['logits'][0] == 0.
    for name, value in f.resolver.named_parameters():
        assert value.grad is None
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    loss = f.model.errors.total(objective='expectation')
    if answer_weight == 0:
        assert measured == [0., 0.] and loss is None
        assert f.model._last_thought_score_function['surrogate'] is None
    else:
        assert measured == [0., 1.]
        assert loss is not None and loss.requires_grad
        assert f.model._last_thought_score_function['category'] == 'expectation'
        loss.backward()
        assert f.resolver.policy.grad.abs().sum() > 0
        assert f.resolver.policy.grad[4].abs() > 0


def test_merging_into_an_anonymous_token_adopts_the_joined_kind():
    resolver, store, atoms, words, _ = world()
    pending, _ = keep(resolver, named(atoms[0], words['x']))
    known, _ = keep(resolver, mention(atoms[0], reference=words['three']), position=1)
    pool = resolver.components.owner.mtm
    w, t = store.occurrence_of(pending), store.occurrence_of(known)
    assert pool.entries[w].kind is None and pool.entries[t].kind == words['three']
    before = snapshot(resolver)
    pool.merge(t, w, when=3)
    assert pool.token_of(t) == w and pool.entries[w].kind == words['three']
    assert resolver._derivation_symbol(w) == words['three']
    torch.testing.assert_close(resolver.token_value(w), atoms[0])
    unchanged(resolver, before)
