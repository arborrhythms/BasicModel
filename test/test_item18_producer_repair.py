"""Item 1.8 §57.6: universal word wholes and signed sparse source weights.

Certificate 7 is the two existing named stopping-gate tests, run unchanged.
"""
import pytest
import torch
from Layers import ConceptualAttentionLayer, SparseLayer


def word_model():
    from data import TheData
    from Models import BaseModel
    TheData.processLM({split:dict(text=['.'] if split=='train' else [],label=[])
                      for split in ('train','validation','test')})
    model=BaseModel.from_config('data/MM_corpus_lessons.xml',data=TheData)[0]
    model._tensor_peer_while_eager=True
    model.eval()
    return model


def test_01_wholes_hold_on_every_byte(eager_reading):
    model=word_model()
    try:
        ws=model.wholeSpace
        capital=ws.well_known_atoms['capital']
        assert capital not in ws.property_rows_for_bytes(b'This')
        assert capital in ws.property_rows_for_bytes(b'A')
        assert ws.property_rows_for_bytes(b'This')==ws.property_rows_for_bytes(b'this')
        assert ws.property_rows_for_bytes(b'')==()
        for form in (b' ',b'\0',b'A ',b'A\0'):
            assert ws.property_rows_for_bytes(form)==()
    finally:model.End();model.symbolSpace.soft_reset()


def test_02_admitted_word_and_basic_percept_are_positive_before_teaching(eager_reading):
    from nomination_fixtures import nominated_forms
    from Assessment import read_sentence
    from postmerge_word_lanes import native_provenance
    model=word_model()
    try:
        nominated_forms(model,['This.'])
        captured=[]
        with native_provenance(captured.append):record=read_sentence(model,'This.')
        word=next(w for w in record['words'] if w['form']=='This')
        # The word's concept reads its basic percept by its column of the
        # order-zero Sigma: tanh(v . u), with u started as v/|v|^2 (rule 8b),
        # v what tells for the percept less what tells against it (rule 54).
        cs=model._concept_owner()
        concept=next(w for w in captured[-1]['words'] if w['form']=='This')['concepts'][0]
        sigma=cs.order_zero
        column=sigma.weight[:,concept['row']].detach()
        pair=cs.basic_percepts.read_codes([concept['features'][0]['bp']])[0].to(column)
        code=pair[0]-pair[1]
        assert not column[sigma.content_width:].any()
        expected=float(torch.tanh(code@column[:sigma.content_width]))
        assert expected==pytest.approx(float(torch.tanh(torch.tensor(1.))),abs=1e-6)
        assert word['native_field']['concepts'][0]['evidence']==pytest.approx([expected,0.],abs=1e-6)
        assert concept['features'][0]['pair']==[1.,0.]
    finally:model.End();model.symbolSpace.soft_reset()


def paired_layer(layout,weights,*,anchor=False):
    if layout=='feature':
        layer=ConceptualAttentionLayer.square(3).features
        layer.nInput=6
        columns=(0,1)
    else:
        layer=ConceptualAttentionLayer.square(3)
        columns=(0,4)
    for c,w in zip(columns,weights):layer.add_edge(2,c,weight=w)
    if anchor:layer.add_edge(2,2 if layout=='feature' else 1,weight=1.)
    return layer


def fold(layer,layout,*,conjunctive=False,values=None):
    values=(.25,.8,1. if conjunctive else 0.) if values is None else values
    if layout=='feature':
        pairs=torch.tensor([[values[0],values[1]],[values[1],values[0]],
                            [values[2],1-values[2]]])[:layer.nnz]
        return layer.fold_presence(torch.empty(0,1),start=2,end=3,conjunctive=conjunctive,
                                   edge_memberships=pairs[:,:1],edge_known=torch.ones(layer.nnz,1,dtype=torch.bool))
    u=torch.tensor([[values[0]],[values[2]],[0.],[1.],[values[1]],[1-values[2]],[0.],[0.]])
    return layer.fold_presence(u,start=2,end=3,conjunctive=conjunctive)


@pytest.mark.parametrize('weights',[(.8,.2),(.2,.8),(.5,.5)])
@pytest.mark.parametrize('conjunctive',[False,True])
def test_05_feature_pair_gradients_and_zero_floor(weights,conjunctive):
    layer=paired_layer('feature',weights,anchor=True)
    out=fold(layer,'feature',conjunctive=conjunctive)
    gradient,=torch.autograd.grad(out.sum(),layer.values)
    if weights[0]!=weights[1]:
        assert gradient[0]!=0
    torch.testing.assert_close(gradient[0],-gradient[1],rtol=0,atol=0)
    if weights[0]==weights[1]:
        # The old unpaired zero-edge rule is still the local derivative.
        control=SparseLayer(6,3);control.add_edge(2,0,weight=0.);control.add_edge(2,1,weight=0.);control.add_edge(2,2,weight=1.)
        g,=torch.autograd.grad(fold(control,'feature',conjunctive=conjunctive).sum(),control.values)
        torch.testing.assert_close(gradient[:2],torch.stack((g[0]-g[1],g[1]-g[0])),rtol=0,atol=0)
        assert bool(gradient[0]==0)==bool(g[0]==g[1])


@pytest.mark.parametrize('weights,expected',[( (1.,1e-7),[1.,0.]),((.2,.7),[0.,1.]),((.5,.5),[0.,0.])])
def test_06_higher_order_pair_values(weights,expected):
    layer=paired_layer('concept',weights)
    u=torch.zeros(8,1);u[0]=1.;u[3]=1.
    pair=torch.stack([layer.fold_presence(u,dual=bool(p))[2,0] for p in (0,1)])
    torch.testing.assert_close(pair,torch.tensor(expected),rtol=0,atol=0)


@pytest.mark.parametrize('weights',[(.8,.2),(.2,.8),(.5,.5)])
def test_06_higher_order_pair_gradients(weights):
    layer=paired_layer('concept',weights,anchor=True)
    gradient,=torch.autograd.grad(fold(layer,'concept').sum(),layer.values)
    assert gradient[0]!=0
    torch.testing.assert_close(gradient[0],-gradient[1],rtol=0,atol=0)
    if weights[0]==weights[1]:
        control=SparseLayer(8,3)
        for column,weight in ((0,0.),(4,0.),(1,1.)):
            control.add_edge(2,column,weight)
        g,=torch.autograd.grad(fold(control,'concept').sum(),control.values)
        torch.testing.assert_close(gradient[:2],torch.stack((g[0]-g[1],g[1]-g[0])),rtol=0,atol=0)


@pytest.mark.parametrize('weights,expected',[
    ((1.,1e-7),[1.,0.]), ((.2,.7),[0.,1.]), ((.5,.5),[0.,0.])])
def test_06_definition_closure_reads_the_same_signed_pair(weights,expected):
    from types import SimpleNamespace
    from FieldProjection import close
    layer=paired_layer('concept',weights)
    owner=SimpleNamespace(_concept_allocator=SimpleNamespace(layer=lambda order:layer))
    evidence=torch.tensor([[1.,0.],[0.,0.]])
    result=close(owner,[0,2],evidence)
    torch.testing.assert_close(result[0],evidence[0],rtol=0,atol=0)
    torch.testing.assert_close(result[1],torch.tensor(expected),rtol=0,atol=0)


@pytest.mark.parametrize('weights',[(.8,.2),(.2,.8),(.5,.5)])
@pytest.mark.parametrize('conjunctive',[False,True])
def test_06_reverse_attributes_only_the_signed_definition(weights,conjunctive):
    layer=paired_layer('concept',weights)
    control=SparseLayer(8,3)
    control.add_edge(2,0,weight=max(weights[0]-weights[1],0.))
    control.add_edge(2,4,weight=max(weights[1]-weights[0],0.))
    query=torch.tensor([[0.],[0.],[.4]])
    observed=torch.ones(8,1)
    for field in (None,observed):
        actual=layer.attribute_presence(query,observed=field,conjunctive=conjunctive)
        expected=control.attribute_presence(query,observed=field,conjunctive=conjunctive)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    if weights[0]==weights[1]:
        assert torch.count_nonzero(actual)==0


def test_06_cancelled_definition_does_not_create_a_dependency_cycle():
    from types import SimpleNamespace
    from FieldProjection import close
    layer=ConceptualAttentionLayer.square(3)
    for target,source in ((0,2),(2,0)):
        layer.add_edge(target,source,weight=.5)
        layer.add_edge(target,source+4,weight=.5)
    owner=SimpleNamespace(_concept_allocator=SimpleNamespace(layer=lambda order:layer))
    evidence=torch.tensor([[.25,0.],[0.,.75]])
    torch.testing.assert_close(close(owner,[0,2],evidence),evidence,rtol=0,atol=0)


@pytest.mark.parametrize('weights,expected',[
    ((.8,.2),[.8,.3]), ((.2,.8),[.3,.8]), ((.5,.5),[0.,0.])])
def test_05_percept_attribution_reads_the_same_signed_feature(weights,expected):
    from test_cs_sparse_weights import _cs
    from basic_fixtures import feature
    cs=_cs()
    feature(cs,0,'ps',7,weights[0])
    bp=next(iter(cs.basic_percepts.index.values()))
    cs.add_concept_feature(0,'bp',bp,weights[1],negated=True)
    query=torch.zeros(sum(cs._order_caps()),1,1,2)
    query[0,0,0]=torch.tensor([.8,.3])
    columns,values,_=cs.cs_percept_attribution(query)
    result=values.new_zeros(2).scatter_add(0,columns%2,values[:,0,0,0])
    torch.testing.assert_close(result,torch.tensor(expected),rtol=0,atol=0)


def test_pairing_survives_binding_and_range_cache_growth():
    layer=paired_layer('concept',(1.,.1))
    view=layer.bind([2,0])
    u=torch.tensor([[0.],[1.],[1.],[0.],[0.],[0.]])
    assert view.fold_presence(u,start=0,end=1).item()==1.
    layer.add_edge(2,1,weight=.2)
    # Both missing sources remap to the bias slot; they must not become partners.
    view=layer.bind([2])
    assert view.fold_presence(torch.zeros(4,1)).item()==0.
