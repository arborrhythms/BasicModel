"""Data-driven authority gates; fixtures supply views and kept occupations."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from ClauseRow import Clause, Composition
from ConceptTrust import positive_length
from IdentityRows import mint
from Meaning import ConceptualMeaning
from definition_fixtures import agreeing_views, learned_operation
from test_review13_subspaces import fixture as paired_fixture


def gate_data(monkeypatch, *, interpret='closed', clause='closed', excluded='interpret sign bind lift'):
    """Close gates in construction data, never by changing live gate state."""
    from util import TheXMLConfig
    data = dict(closedTrust=.1, interpret=dict(start=interpret, fraction=.5),
                clause=dict(start=clause, threshold=.75, **{'except':excluded}))
    original = TheXMLConfig.space
    missing = object()
    def space(name, key, default=missing):
        if name == 'ConceptualSpace' and key == 'gates':
            return copy.deepcopy(data)
        return original(name,key) if default is missing else original(name,key,default)
    monkeypatch.setattr(TheXMLConfig,'space',space)
    return data


def gate_fixture(monkeypatch):
    import Language
    from ClauseRow import attach_clause_index
    from Layers import LTM0
    from PartsOfSpeech import PartsOfSpeech,formal_arguments
    owner,bank,_,alloc = paired_fixture(meaning_pairs=4, capacity=160)
    grammar=Language.Grammar();grammar.load_from_grammar_file('complete.grammar')
    language=SimpleNamespace(parts_of_speech=PartsOfSpeech(formal_arguments(grammar.rules_upward)))
    store=LTM0(owner.nDim,capacity=64)
    attach_clause_index(SimpleNamespace(languageSpace=language),store,owner)
    return SimpleNamespace(cs=owner,store=store,language=language)


def words(fixture, count):
    # Different admitted units keep their own word identities.
    ids = [fixture.cs.interpret.lookup_word([4+i], (), form=f'token{i}') for i in range(count)]
    bank=fixture.cs.similarity_codebook
    for i, word in enumerate(ids):
        row=fixture.cs._csw_row_of(word)
        bank.mereology.placement[row]=torch.eye(bank.mereology.meaning_pairs)[i % bank.mereology.meaning_pairs]
        bank.mereology.placement_witnesses[row]=1
    return ids


def composition(fixture, left, right, operation):
    owner = fixture.cs
    leaves = tuple(Composition(owner.similarity_codebook.lookup_rows(owner._csw_row_of(word)),
        reference=word,word_id=word,order=0) for word in (left,right))
    node = Composition(leaves[0].point,leaves,operation=operation,formal_name=operation)
    return Clause(ConceptualMeaning.from_description(node.point),composition=node,point=node.point)


def named_fixture(monkeypatch):
    data = gate_data(monkeypatch,clause='open')
    f = gate_fixture(monkeypatch);owner=f.cs
    word,second,third = words(f,3)
    agreeing_views(owner,word)
    obj = mint(owner,torch.eye(owner.nDim)[3]);agreeing_views(owner,obj,count=2)
    description = dict(owner.interpret._pending[word],identity_key='gate-form')
    row = owner.interpret.define(word,obj,evidence=(.9,.4),description=description)
    return f,data,(word,second,third),obj,row


def test_interpret_gate_caps_only_binding_and_opens_monotonically(monkeypatch):
    f,data,ids,obj,row = named_fixture(monkeypatch)
    owner,bank,index=f.cs,f.store.parts,f.cs.definitions
    word,second,third=ids
    pair=torch.tensor([.9,.4])
    scale=min(positive_length(owner,word),positive_length(owner,obj),data['closedTrust'])
    torch.testing.assert_close(torch.stack(bank.evidence(row)),pair*scale)
    assert not bank.gates.open['interpret'] and index.objects(word)==()
    assert index.word(form='token0')==index.word(unit=(4,))==index.word(identity='gate-form')==word
    assert owner.interpret.remembered_objects(word)==(obj,)
    count,next_id=len(bank),owner._concept_allocator.next_id
    agreeing_views(owner,second)
    assert owner.trust_measurements()['stable_word_fraction']==pytest.approx(2/3)
    assert index.objects(word)==(obj,) and bank.gates.open['interpret']
    torch.testing.assert_close(torch.stack(bank.evidence(row)),pair*min(positive_length(owner,word),positive_length(owner,obj)))
    assert len(bank)==count and owner._concept_allocator.next_id==next_id
    for identity in words(f,5)[3:]:
        agreeing_views(owner,identity,count=1)
    assert owner.trust_measurements()['stable_word_fraction'] < data['interpret']['fraction']
    assert bank.gates.open['interpret'] and index.objects(word)==(obj,)
    assert (float(bank.truth_plus[row]),float(bank.truth_minus[row]))==pytest.approx((.9,.4))


@pytest.mark.parametrize('except_part',[False,True])
def test_clause_gate_caps_recorded_operations_and_opens_on_lift_classes(monkeypatch,except_part):
    excluded='interpret sign bind lift'+(' part' if except_part else '')
    data=gate_data(monkeypatch,interpret='open',excluded=excluded)
    f=gate_fixture(monkeypatch);owner,bank,speech=f.cs,f.store.parts,f.language.parts_of_speech
    a,b,c,d=words(f,4);agreeing_views(owner,a,b,c,d,count=10)
    learned_operation(speech,'lift',a,b,count=1)
    learned_operation(speech,'lift',a,b,count=1)
    learned_operation(speech,'verb',c,d,count=10)
    addresses=[f.store.write_clause(composition(f,*operands,name),evidence=(.8,.4))
               for name,operands in [('verb',(c,d)),('lift',(a,b)),('part',(c,a))]]
    rows=[f.store.location(address)[1] for address in addresses]
    assert speech.operation('lift')['definiteness']==pytest.approx((-math.expm1(-2/3),)*2)
    assert not bank.gates.open['clause']
    stored=torch.tensor([.8,.4])
    for row,name,operands in zip(rows,('verb','lift','part'),((c,d),(a,b),(c,a))):
        assert bank.operation_catalog[int(bank.grammar_operation[row])]==name
        scale=min(*(positive_length(owner,x) for x in operands),speech.operation(name)['trust'])
        if name=='verb' or name=='part' and not except_part:scale=min(scale,data['closedTrust'])
        torch.testing.assert_close(torch.stack(bank.evidence(row)),stored*scale)
    before=bank.truth_plus.clone(),bank.truth_minus.clone()
    learned_operation(speech,'lift',a,b,count=6)
    measured=speech.operation('lift')
    definiteness=-math.expm1(-8/3)
    assert measured['definiteness']==pytest.approx((definiteness,definiteness))
    assert measured['trust']==pytest.approx(definiteness)
    assert sum(measured['definiteness'])/2 >= data['clause']['threshold']
    for row,name,operands in zip(rows,('verb','lift','part'),((c,d),(a,b),(c,a))):
        scale=min(*(positive_length(owner,x) for x in operands),speech.operation(name)['trust'])
        torch.testing.assert_close(torch.stack(bank.evidence(row)),stored*scale)
    assert bank.gates.open['clause']
    torch.testing.assert_close(bank.truth_plus,before[0]);torch.testing.assert_close(bank.truth_minus,before[1])


def test_open_gates_survive_checkpoint_and_epoch_and_missing_state_is_rejected(monkeypatch):
    from IndependentComponents import _pack_state,_unpack_state
    from LTM1 import LTM1
    f,_,ids,_,row=named_fixture(monkeypatch);bank=f.store.parts
    bank.begin_epoch()
    agreeing_views(f.cs,ids[1]);expected=torch.stack(bank.evidence(row)).clone()
    assert bank.gates.open==dict(interpret=True,clause=True)
    state=copy.deepcopy(bank.state_dict())
    bank.begin_epoch()
    assert bank.gates.open==dict(interpret=True,clause=True)
    torch.testing.assert_close(torch.stack(bank.evidence(row)),expected)
    restored=LTM1(bank.capacity);restored.configure_operations(f.language.parts_of_speech)
    restored._concept_owner=weakref.ref(f.cs)
    pointers=tuple(getattr(restored,name).data_ptr() for name in restored.columns)
    assert not restored.gates.open['interpret']
    restored.load_state_dict(state)
    assert restored.gates.open==bank.gates.open
    torch.testing.assert_close(torch.stack(restored.evidence(row)),expected)
    assert pointers==tuple(getattr(restored,name).data_ptr() for name in restored.columns)
    missing=copy.deepcopy(state);extra=_unpack_state(missing['_extra_state']);extra.pop('gate_open');missing['_extra_state']=_pack_state(extra)
    with pytest.raises(ValueError,match='gate state'):restored.load_state_dict(missing)
    old=copy.deepcopy(state);old['layout_version'].sub_(1)
    with pytest.raises(ValueError,match='layout checkpoint version'):restored.load_state_dict(old)


@pytest.mark.parametrize('lesson',[False,True])
def test_model_gate_start_is_open_by_default_and_closed_for_lesson_configuration(lesson):
    from corpus_lessons import LESSON_CONFIG
    from Models import BaseModel
    from data import TheData
    from util import init_config
    root=Path(__file__).resolve().parents[1]
    config=LESSON_CONFIG if lesson else root/'data/MM_corpus_orders.xml'
    init_config(path=str(config),defaults_path=str(root/'data/model.xml'))
    TheData.processLM({split:dict(text=['.'] if split=='train' else [],label=[]) for split in ('train','validation','test')})
    model=BaseModel.from_config(str(config),data=TheData)[0]
    try:
        bank=model._concept_owner()._concept_allocator.parts
        assert bank.gates.open==dict(interpret=not lesson,clause=not lesson)
        speech = model.languageSpace.parts_of_speech
        assert set(speech.concepts) == set(speech.arguments)
        assert not speech.radii().any()
        assert all(model._concept_owner()._concept_allocator.order_of(cid) == 1
                   for cid in speech.concepts.values())
    finally:
        model.End();model.symbolSpace.soft_reset()


def test_readiness_records_gate_state_and_measurement_without_a_lesson_run(monkeypatch):
    from corpus_lessons import readiness_report,lexical_report
    f,_,ids,_,row=named_fixture(monkeypatch)
    report=readiness_report(f.cs)
    assert report['gates']==dict(interpret=dict(open=False,measurement=1/3),clause=dict(open=True,measurement=0.,arguments=f.language.parts_of_speech.operation('lift')['argument_measurements']))
    assert report['trust_measurements']==f.cs.trust_measurements()
    model=SimpleNamespace(_concept_owner=lambda:f.cs,languageSpace=f.language)
    lexical=lexical_report(model,['token0'])
    assert lexical['gates']==report['gates'] and lexical['trust_measurements']==report['trust_measurements']
    assert not f.store.parts.gates.open['interpret']


def test_empty_lift_classes_have_zero_radius_and_leave_the_clause_gate_closed(monkeypatch):
    gate_data(monkeypatch,interpret='open')
    f=gate_fixture(monkeypatch);bank=f.store.parts
    assert not bank.gates.open['clause']
    a,b=words(f,2);agreeing_views(f.cs,a,b)
    address=f.store.write_clause(composition(f,a,b,'part'),evidence=(1.,0.))
    row=f.store.location(address)[1]
    torch.testing.assert_close(torch.stack(bank.evidence(row)), torch.tensor([bank.gates.closed_trust, 0.]))
    assert f.language.parts_of_speech.operation('lift')['definiteness']==(0.,0.)
    assert f.language.parts_of_speech.operation('lift')['uses']==0
    assert not bank.gates.open['clause']
    assert bank.gates.measurements(f.cs,f.language.parts_of_speech)['clause']==0.


@pytest.mark.parametrize('field,value',[('closedTrust',float('nan')),('fraction',1.1),('threshold',-.1),('start','invalid')])
def test_invalid_gate_data_fails_at_construction(monkeypatch,field,value):
    from LTM1 import LTM1
    data=gate_data(monkeypatch)
    if field=='closedTrust':data[field]=value
    elif field=='fraction':data['interpret'][field]=value
    else:data['clause'][field]=value
    with pytest.raises(ValueError):LTM1(8)


@pytest.mark.parametrize('excluded', [None, '', '   '])
def test_empty_except_list_caps_lift_too(monkeypatch, excluded):
    data=gate_data(monkeypatch,interpret='open',excluded=excluded)
    f=gate_fixture(monkeypatch)
    a,b=words(f,2);agreeing_views(f.cs,a,b)
    learned_operation(f.language.parts_of_speech,'lift',a,b,count=1)
    address=f.store.write_clause(composition(f,a,b,'lift'),evidence=(.8,.4))
    row=f.store.location(address)[1]
    assert not f.store.parts.gates.excluded
    assert not f.store.parts.gates.open['clause']
    torch.testing.assert_close(torch.stack(f.store.parts.evidence(row)),torch.tensor([.8,.4])*data['closedTrust'])
