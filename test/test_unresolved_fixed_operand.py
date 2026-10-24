"""A fixed unresolved operand cannot stand for another region by similarity."""
from dataclasses import replace
import pytest
from ThoughtReferences import bindings, question, fill, open_slots
from test_thought_binding import equality

@pytest.mark.parametrize('same_reference', (False,True))
def test_query_requires_the_address_of_an_unresolved_fixed_operand(monkeypatch,same_reference):
    from Queries import ThoughtLTMCapability
    from test_query_executors import _context,_cs
    name,unresolved,other=('sym',1),('ltm','test',2),('sym',4)
    supplied=equality(name,unresolved)
    supplied=replace(supplied,bindings=dict(bindings(supplied),_unresolved_operands=((2,unresolved),)))
    frame=dict(meaning=supplied,occurrence=('ltm','test',3),kind='observation',evidence=(.8,0.),match=1.)
    pattern=question(equality(None,unresolved if same_reference else other),(('referent',0),))
    data=bindings(pattern); data.pop('_equality')
    pattern=replace(pattern,bindings=data)
    monkeypatch.setattr(ThoughtLTMCapability,'_cued',lambda *args,**kwargs:dict(value=(frame,),records_scanned=1,incomplete=()))
    result=_context(_cs()).ltm.best_match(pattern,max_records=10,work=None)
    assert result['records_scanned']==1
    if same_reference:
        assert result['frames']==(frame,) and result['support_true']==.8
    else:
        assert result['frames']==() and result['witnesses']==()
        assert result['support_true']==result['support_false']==0.

@pytest.mark.parametrize('same_reference',(False,True))
def test_ordinary_fill_keeps_an_unmatched_unresolved_fixed_operand_open(same_reference):
    name,unresolved,other=('sym',1),('ltm','test',2),('sym',4)
    supplied=equality(name,unresolved)
    supplied=replace(supplied,bindings=dict(bindings(supplied),_unresolved_operands=((2,unresolved),)))
    pattern=question(equality(None,unresolved if same_reference else other),(('referent',0),))
    data=bindings(pattern);data.pop('_equality')
    pattern=replace(pattern,bindings=data)
    result=fill(pattern,dict(meaning=supplied,support_true=.8,support_false=0.),witnesses=(('ltm','test',3),))
    if same_reference:
        assert result.role_refs[0]==name and not open_slots(result)
    else:
        assert result is pattern and open_slots(result)==(('referent',0),)
