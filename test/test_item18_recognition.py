"""A perceived reference reaches the existing interpretation inverse and output."""
import json
from dataclasses import replace
from pathlib import Path

import torch
import pytest

pytestmark = pytest.mark.assessment_rung(4)

from test_item18_finish import named_scalar_model


def teach_question(model, examples, question):
    """The fixed six rounds of three ordinary supervised questions (§35.3)."""
    from scalar_question_curriculum import configure, train_round
    optimizer = configure(model, examples, question)
    for _ in range(6):
        train_round(model, examples, question, optimizer)



def recognize(model, value, question):
    """The model reads the question text and keeps a conceptual answer."""
    from ThoughtReferences import bindings
    from ThoughtAnswer import accepted
    from What import What
    owner = model._concept_owner()
    with torch.no_grad():
        understood = model.understand(model.inputSpace.prepInput([dict(text=question, scalar=value)]))
        present = model._scalar_present[0]
        answer = model.resolveAnswer(understood, What.inference(0, prompt=question))
        assert answer.answer_meanings[0], understood.sentence_states[0].meaning
        concluded = answer.answer_meanings[0][0]
        assert accepted(concluded)
        assert bindings(concluded)['_evidence_kind'] == 'perception'
        assert concluded.role_refs[0][0] == 'sym'
        assert concluded.role_refs[0] != present['occurrence']
        emitted = model.reverseOutput(understood, answer)
    return emitted


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason='baseline, item 1.8 §43', raises=AssertionError)
def test_90_present_percept_is_said_by_its_class_without_naming_an_object(named_scalar_model):
    model, objects = named_scalar_model
    question = json.loads(Path('data/scalar_lessons.json').read_text())['recognition_question']
    examples = [dict(text=name, scalar=value) for name, value in
        (('alpha', 1.), ('beta', 1.), ('gamma', 2.))]
    from scalar_question_curriculum import acquire
    acquire(model, examples, question)
    teach_question(model, examples, question)
    emitted = recognize(model, 2., question)
    assert emitted.texts == ('gamma',)


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason='baseline, item 1.8 §43', raises=AssertionError)
def test_90_named_percept_does_not_mix_native_spellings(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from ScalarPercepts import code, observe
    from test_output_walk import _stop
    lesson = json.loads(Path('data/scalar_lessons.json').read_text())
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    for item in lesson['lessons'][:3]:
        form = item['text']
        word, *_ = owner.interpret_word(model.perceptualSpace.percept_store.spell_out(form.encode()),
            model.wholeSpace.property_rows_for_bytes(form), key=form)
        for _ in range(lesson['repetitions']):
            row = store.append_idea(code(owner, item['scalar']), trust=1.)
            observe(owner, word, item['scalar'], store.occurrence_of(row), stream=0)
    model.eval(); _stop(model)
    try:
        from scalar_question_curriculum import acquire
        acquire(model, lesson['lessons'][:3], lesson['recognition_question'])
        teach_question(model, lesson['lessons'][:3], lesson['recognition_question'])
        for item in lesson['lessons'][:3]:
            emitted = recognize(model, item['scalar'], lesson['recognition_question'])
            assert emitted.texts == (item['text'],)
            model.End()
    finally:
        from assessment_support import record
        record(model, dict(word=lesson['lessons'][0]['text'],
            words=[item['text'] for item in lesson['lessons'][:3]]), scalar=lesson['lessons'][0]['scalar'])
        model.End(); model.symbolSpace.soft_reset()


def test_scalar_percept_and_its_collection_restore_with_their_typed_references(
        tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    try:
        model.forward(model.inputSpace.prepInput([dict(text='one', scalar=1.)]))
        owner = model._concept_owner()
        state = owner.mtm.state()
        percepts = [key for key in owner.mtm.entries if key[:1] == ('bp',)]
        assert percepts
        expected = {key: owner.mtm.value_of(key).clone() for key in percepts}
        owner.mtm.restore(state)
        assert owner.mtm.state()['names'] == state['names']
        for key in percepts:
            torch.testing.assert_close(owner.mtm.value_of(key), expected[key], atol=0, rtol=0)
            assert owner.mtm.entries[key].transient is None
    finally:
        model.End(); model.symbolSpace.soft_reset()
