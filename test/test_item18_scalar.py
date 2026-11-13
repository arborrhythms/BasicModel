"""Counted scalar boxes use the existing native identity and definition writers."""
import json
from pathlib import Path
import torch
import pytest


def test_scalar_boxes_are_counted_native_definitions(tmp_path,monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from ScalarPercepts import code,observe,reconstruct
    model = _tiny_canonical_model(tmp_path,monkeypatch,batch_size=1)
    owner,store = model._concept_owner(),model.symbolSpace.ltm_store
    lesson = json.loads(Path('data/scalar_lessons.json').read_text())
    words={}
    for item in lesson['lessons']:
        text = item['text']
        parts = model.perceptualSpace.percept_store.spell_out(text.encode())
        word,*_ = owner.interpret_word(parts,model.wholeSpace.property_rows_for_bytes(text),key=text)
        words[text]=word
    for repeat in range(lesson['repetitions']):
        for item in lesson['lessons']:
            if item['scalar'] is None:continue
            point=code(owner,item['scalar'])
            row=store.append_idea(point,trust=1.,document_key=('scalar-test',repeat,item['text']),sentence_index=repeat)
            observe(owner,words[item['text']],item['scalar'],store.occurrence_of(row),stream=0)
    objects=[]
    for item in lesson['lessons'][:3]:
        from PerceptCollections import objects as known
        identity, = known(owner, words[item['text']])
        row = owner._csw_row_of(identity)
        assert int(owner.similarity_codebook.trust.views[row]) == 6
        objects.append(identity)
        assert owner._concept_allocator.order_of(identity) == 0
        assert reconstruct(owner,word=words[item['text']]) == pytest.approx(item['scalar'])
        assert reconstruct(owner,value=item['scalar'])==(item['text'],)
        assert float(store.parts.truth_plus[owner.definitions.row(words[item['text']],identity)])>.5
    assert len(set(objects))==3
    assert not known(owner, words['unfelt'])
    assert reconstruct(owner,value=0.)==()
    assert reconstruct(owner,value=1.5)==()


def test_scalar_joins_the_actual_glance_without_a_place(tmp_path,monkeypatch,eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path,monkeypatch,batch_size=1)
    model._tensor_peer_while_eager = True
    model.train()
    source = model.inputSpace.prepInput([dict(text='one',scalar=1.)])
    try:
        model.forward(source)
        field = model._sentence_open_fields[0]
        assert not field.where[0,-1].any()
        owner = model._concept_owner()
        percept = model._scalar_present[0]['percept']
        assert owner.basic_percepts.occurrences[percept][1]
        assert model._scalar_present[0]['occurrence'] == ('bp', percept)
        assert model._scalar_present[0]['candidate'] in model._concept_owner().mtm.entries
    finally:
        model.End()


def test_scalar_perception_keeps_percept_and_concept_references(tmp_path,monkeypatch,eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from ScalarPercepts import code,observe,decode
    model=_tiny_canonical_model(tmp_path,monkeypatch,batch_size=1)
    model._tensor_peer_while_eager=True
    owner,store=model._concept_owner(),model.symbolSpace.ltm_store
    lesson=json.loads(Path('data/scalar_lessons.json').read_text())
    for item in lesson['lessons']:
        raw=item['text'].encode()
        word,*_=owner.interpret_word(model.perceptualSpace.percept_store.spell_out(raw),
            model.wholeSpace.property_rows_for_bytes(item['text']),key=item['text'])
        if item['scalar'] is None:continue
        for repeat in range(lesson['repetitions']):
            row=store.append_idea(code(owner,item['scalar']),trust=1.,
                document_key=('lesson',word,repeat))
            observe(owner,word,item['scalar'],store.occurrence_of(row),stream=0)
    model.eval()
    try:
        for item in lesson['lessons'][:3]:
            model.forward(model.inputSpace.prepInput([dict(text=lesson['recognition_question'],scalar=item['scalar'])]))
            field=model._sentence_open_fields[0]
            assert decode(owner,field)[0]['words']==(item['text'],)
            # Native question recognition is deferred to compose_extend (§3.17).
            # Retain the membership and distinct-reference contract here.
            reference = model._scalar_present[0]['occurrence']
            assert reference[0] == 'bp'
            from PerceptCollections import objects as known
            word = owner.interpret.word_for_form(item['text'])
            identity, = known(owner, word)
            assert ('sym', identity) != reference
            assert owner._concept_allocator.order_of(identity) == 0
            torch.testing.assert_close(owner.mtm.value_of(reference), code(owner,item['scalar']))
    finally:model.End()


def test_scalar_only_presentation_closes_and_reconstructs_its_modalities(tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from ScalarPercepts import code, observe
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    word, *_ = owner.interpret_word(model.perceptualSpace.percept_store.spell_out(b'one'),
        model.wholeSpace.property_rows_for_bytes('one'), key='one')
    for repeat in range(6):
        row = store.append_idea(code(owner, 1.), trust=1., document_key=('scalar-only', repeat))
        observe(owner, word, 1., store.occurrence_of(row), stream=0)
    model.eval()
    try:
        understanding = model.understand(model.inputSpace.prepInput([dict(text='', scalar=1.)]))
        result = model.reconstructModalities(understanding)
        assert result['scalar'][0]['number'] == pytest.approx(1.)
        assert result['scalar'][0]['words'] == ('one',)
        assert not understanding.reconstruction_carriers['scalar_field'].where.any()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
