"""A fixed unresolved operand cannot stand for another region by similarity."""
from dataclasses import replace
import pytest
from ThoughtReferences import bindings, fill, open_slots
from meaning_fixtures import unsupported as question
from test_thought_binding import equality

@pytest.mark.parametrize('same_reference', (False,True))
def test_query_requires_the_address_of_an_unresolved_fixed_operand(monkeypatch,same_reference):
    import torch
    from Extension import ConceptSet, Context, Node
    from Language import UnionLayer
    from order_fixtures import force_order
    from Queries import ThoughtLTMCapability
    from test_query_executors import _context,_cs
    cs = _cs()
    for index in range(4):
        cid = cs.new_concept()
        force_order(cs, cid, 0)
        row = cs._csw_concept_row(0, cid)
        cs.similarity_codebook.W.data[row] = torch.eye(8)[index]
    name,unresolved,other=('sym',1),('ltm','test',2),('sym',4)
    operator = UnionLayer()
    def tree(opened):
        left = ConceptSet(torch.tensor([1]), torch.tensor([[0. if opened else 1., 0.]]), opened)
        right = ConceptSet(torch.tensor([4]), torch.tensor([[1., 0.]]))
        return (Node(operator.compose_extend(left, right),
                     (Node(left, context=Context(open=opened)), Node(right)),
                     operator, Context(name='union')),)
    supplied=equality(name,unresolved)
    supplied=replace(supplied,extension=tree(False),bindings=dict(bindings(supplied),_unresolved_operands=((2,unresolved),)))
    frame=dict(meaning=supplied,occurrence=('ltm','test',3),trust=.8,evidence=(.8,0.),match=1.)
    pattern=question(equality(None,unresolved if same_reference else other),(('referent',0),))
    data=bindings(pattern); data.pop('_equality')
    pattern=replace(pattern,bindings=data,extension=tree(True))
    monkeypatch.setattr(ThoughtLTMCapability,'_cued',lambda *args,**kwargs:dict(value=(frame,),records_scanned=1,incomplete=()))
    result=_context(cs,store=cs._fixture_occurrences).ltm.best_match(pattern,max_records=10,work=None)
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
