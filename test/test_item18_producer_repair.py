"""Item 1.8 §57.6: universal word wholes and signed sparse source weights.

Certificate 7 is the two existing named stopping-gate tests, run unchanged.
"""
import math
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
        # The word's concept reads its basic percept by its entry on the
        # percept's row: tanh(w s), w started at one, s the row's signed
        # value from the Not, for less against (rule 62).
        cs=model._concept_owner()
        concept=next(w for w in captured[-1]['words'] if w['form']=='This')['concepts'][0]
        sigma=cs.order_zero
        assert sigma.entries(concept['row'])==[(concept['features'][0]['bp'],1.,False)]
        expected=float(torch.tanh(torch.tensor(1.)*1.))
        assert expected==pytest.approx(float(torch.tanh(torch.tensor(1.))),abs=1e-6)
        assert word['native_field']['concepts'][0]['evidence']==pytest.approx([expected,0.],abs=1e-6)
        assert concept['features'][0]['pair']==[1.,0.]
    finally:model.End();model.symbolSpace.soft_reset()


def signed_layer(value,*,anchor=False):
    # Row 2 reads, by one signed entry, a symbol that holds row 0's concept
    # (its size the weight, its sign the option); the anchor reads one
    # that holds row 1's.
    from symbol_fixtures import store_columns
    layer=ConceptualAttentionLayer.square(3)
    layer.add_edge(2,store_columns(layer,0)[0],weight=value)
    if anchor:layer.add_edge(2,store_columns(layer,1)[0],weight=1.)
    return layer


def by_column(layer,rows):
    """The things' two lanes by column, ``[nInput, 1, 2]``, from the lanes
    of each row's symbol (``rows``: row -> (for, against)) and the bias,
    (1, 0)."""
    from symbol_fixtures import store_columns
    u=torch.zeros(layer.nInput,1,2)
    u[layer.bias_column(),0,0]=1.
    for row,pair in rows.items():
        u[store_columns(layer,row)[0],0]=torch.tensor(pair)
    return u


def read(layer,rows,**kw):
    from Layers import NotLayer3,sigma_read
    return sigma_read(layer,NotLayer3.options(by_column(layer,rows)),**kw)


@pytest.mark.parametrize('value,expected',[(1.,[math.tanh(1.),0.]),(-.5,[0.,math.tanh(.5)]),(0.,[0.,0.])])
def test_06_higher_order_pair_values(value,expected):
    # One signed entry: its size the Sigma's weight, its sign the Not's
    # option; the concept is one signed value on two lanes.
    from Layers import signed_lanes
    layer=signed_layer(value)
    pair=signed_lanes(read(layer,{0:(1.,0.)}))[2,0]
    torch.testing.assert_close(pair,torch.tensor(expected))


@pytest.mark.parametrize('value',[.6,-.6,0.])
def test_06_higher_order_pair_gradients(value):
    # The entry's gradient is the symbol's value through the squash, at
    # every value of the entry, zero included: teaching carries an entry
    # from one option to the other through zero.
    layer=signed_layer(value,anchor=True)
    y=read(layer,{0:(.25,.8),1:(0.,1.)},start=2,end=3)
    gradient,=torch.autograd.grad(y.sum(),layer.values)
    s=.25-.8
    torch.testing.assert_close(gradient[0],torch.tensor(s*(1-float(y)**2)))
    assert gradient[0]!=0


@pytest.mark.parametrize('value,expected',[
    (1.,[math.tanh(1.),0.]), (-.5,[0.,math.tanh(.5)]), (0.,[0.,0.])])
def test_06_definition_closure_reads_the_same_signed_pair(value,expected):
    from types import SimpleNamespace
    from FieldProjection import close
    layer=signed_layer(value)
    owner=SimpleNamespace(_concept_allocator=SimpleNamespace(layer=lambda order:layer))
    evidence=torch.tensor([[1.,0.],[0.,0.]])
    result=close(owner,[0,2],evidence)
    torch.testing.assert_close(result[0],evidence[0],rtol=0,atol=0)
    torch.testing.assert_close(result[1],torch.tensor(expected))


@pytest.mark.parametrize('value',[.6,-.6,0.])
def test_06_reverse_attributes_only_the_signed_definition(value):
    # The Sigma's transpose carries the request through the entry's option:
    # to the symbol's for lane by a for entry, its against lane by a Not
    # entry, and nowhere by an entry at zero.
    from Layers import sigma_transpose
    from symbol_fixtures import store_columns
    layer=signed_layer(value)
    column=store_columns(layer,0)[0]
    query=torch.tensor([[[0.,0.]],[[0.,0.]],[[.4,0.]]])
    expected=torch.zeros(layer.nInput,1,2)
    if value>0:expected[column,0,0]=.4
    if value<0:expected[column,0,1]=.4
    for field in (None,torch.ones(layer.nInput,1,2)):
        torch.testing.assert_close(sigma_transpose(layer,query,observed=field),expected,rtol=0,atol=0)


def test_06_cancelled_definition_does_not_create_a_dependency_cycle():
    from types import SimpleNamespace
    from FieldProjection import close
    from symbol_fixtures import store_columns
    layer=ConceptualAttentionLayer.square(3)
    for target,source in ((0,2),(2,0)):
        layer.add_edge(target,store_columns(layer,source)[0],weight=0.)
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
    from Layers import NotLayer3,sigma_read
    layer=signed_layer(.9)
    from symbol_fixtures import store_columns
    view=layer.bind([2,0])
    # The view's input: the one thing read (row 0's symbol) and the bias.
    u=torch.tensor([[[1.,0.]],[[1.,0.]]])
    assert sigma_read(view,NotLayer3.options(u),start=0,end=1).item()==pytest.approx(math.tanh(.9))
    layer.add_edge(2,store_columns(layer,1)[0],weight=.2)
    # Two things whose concepts are not seated are both missing.
    view=layer.bind([2])
    assert sigma_read(view,NotLayer3.options(torch.zeros(view.nInput,1,2))).item()==0.
