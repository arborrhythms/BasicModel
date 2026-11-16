"""A declared collection binds its perceived member by membership."""
from dataclasses import replace
import torch
import pytest
from test_item18_finish import named_scalar_model

@pytest.mark.parametrize('value, belongs', [(1., True), (2., True), (3., True), (4., False)])
def test_bind_percept_collection_reads_membership_not_coordinate_overlap(named_scalar_model, value, belongs):
    from ScalarPercepts import code,observe
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots,bindings
    model,_=named_scalar_model
    owner=model._concept_owner();store=model.symbolSpace.ltm_store
    word,*_=owner.interpret_word(model.perceptualSpace.percept_store.spell_out(b'collection'),
        model.wholeSpace.property_rows_for_bytes('collection'),key='collection')
    for witness in (1.,3.)*3:
        row=store.append_idea(code(owner,witness),trust=1.)
        collection=observe(owner,word,witness,store.occurrence_of(row),stream=0)
    with torch.no_grad():
        understood=model.understand(model.inputSpace.prepInput([dict(text='collection',scalar=value)]))
        present=model._scalar_present[0]
        description=owner.similarity_codebook.lookup_rows(owner._csw_row_of(collection))
        request=with_slots(ConceptualMeaning.from_description(description),(('referent',0),))
        request=replace(request,bindings=dict(bindings(request),_forward_references=((0,word),),
            _reference_sites=((0,(('member_order',0),)),)))
        result=owner.components.reference_close.bind_site(request,0,(),stream=0)
        if belongs:
            assert result['reference'][0] == 'sym'
            assert present['occurrence'][0] == 'bp'
            assert present['occurrence'] != result['reference']
        else:
            assert result.get('reference') is None
        if belongs:
            assert bindings(result['meaning'])['_evidence_kind']=='perception'
            from ThoughtStream import write
            kept = write(model, result['meaning'], row=0)
            assert store.referent_of(kept) == result['reference']
        if value == 2.:
            from BindingAnswers import matches, cost
            assert matches(model, result['meaning'], 'gamma')
            assert float(cost(model, result['meaning'], 'gamma')) == 0.
            assert not matches(model, result['meaning'], 'alpha')
            assert float(cost(model, result['meaning'], 'alpha')) > 0.
            from What import What
            entry=replace(understood.sentence_states[0],meaning=result['meaning'],query=None)
            understood=replace(understood,sentence_states=(entry,))
            answer=model.resolveAnswer(understood,What.inference(0))
            assert model.reverseOutput(understood,answer).texts==('gamma',)


def test_binding_a_lowered_collection_keeps_its_order_when_the_word_also_names_a_kind(named_scalar_model):
    from Meaning import ConceptualMeaning
    from ThoughtReferences import with_slots, bindings
    model, objects = named_scalar_model
    owner = model._concept_owner()
    word = owner.interpret.word_for_form('gamma')
    collection = objects['gamma']
    kind = owner.new_concept(order=1)
    owner.interpret.define(word, kind)
    assert owner.interpret.preferred(word, batch=None) == (kind,)
    with torch.no_grad():
        model.understand(model.inputSpace.prepInput([dict(text='gamma', scalar=2.)]))
        point = owner.similarity_codebook.lookup_rows(owner._csw_row_of(collection))
        request = with_slots(ConceptualMeaning.from_description(point), (('referent', 0),))
        request = replace(request, bindings=dict(bindings(request),
            _forward_references=((0, word),), _reference_sites=((0, (('member_order', 0),)),)))
        result = owner.components.reference_close.bind_site(request, 0, (), stream=0)
        assert result.get('reference') == ('sym', collection)
        assert bindings(result['meaning'])['_evidence_kind'] == 'perception'
