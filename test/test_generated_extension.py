"""Focused reproduction: a generated concept spells through its stored members."""
from types import SimpleNamespace
import torch
import pytest
from test_compiled_word_chunk import _tiny_canonical_model
from order_fixtures import force_order


def test_empty_extension_has_no_native_percept_demand():
    from test_cs_sparse_weights import _cs
    from basic_fixtures import feature
    cs = _cs()
    feature(cs, 0, 'ws', 0, 1., parts=(65,))
    empty = torch.empty(0, 2, 3, 2)
    columns, support, spans = cs.cs_percept_attribution(
        empty, observed=empty, inventory_rows=torch.empty(0, dtype=torch.long))
    assert columns.tolist() == [0, 1]
    assert support.shape == (2, 2, 3, 1)
    assert not support.any() and spans is None


@pytest.fixture
def declared_object(tmp_path, monkeypatch):
    from FieldProjection import stimulate
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1, concept_meaning_width=8)
    owner = model._concept_owner()
    word, *_ = owner.interpret_word(
        model.perceptualSpace.percept_store.spell_out(b'petal'),
        model.wholeSpace.property_rows_for_bytes(b'petal'), key='petal')
    obj = owner.new_concept()
    force_order(owner, obj, 1)
    row = owner._csw_concept_row(1, obj)
    owner._concept_allocator.store_of(obj).add_constituent(obj, 'part', ('sym', word))
    chart = owner.similarity_codebook
    with torch.no_grad():
        point = chart.W.new_zeros(chart.W.shape[-1])
        point[chart.mereology.percept_event_width] = 1.
        chart.W[row] = point
        chart.trust.allocated[row] = True
        chart.trust.tracked = True
        chart.trust.lengths[row,0] = 1.
        chart.trust.content[row] = point
        chart.mereology.placement[row].zero_()
        chart.mereology.placement[row,0] = 1.
    field = stimulate(owner, point[None,None])
    assert field.concept_ids[field.evidence[:,0,0,0] == 1.].tolist() == [obj]
    try:
        yield model, word, obj, point
    finally:
        model.End()
        model.symbolSpace.soft_reset()


@pytest.mark.parametrize('read', ['meaning', 'word'])
def test_generated_known_object_uses_its_declared_word(declared_object, read):
    from OutputOccurrences import generated, realize
    model, word, obj, point = declared_object
    owner = model._concept_owner()
    chart = owner.similarity_codebook
    wrow = owner._csw_row_of(word)
    bank = SimpleNamespace(codes=chart.lookup_rows([wrow])[None], weights=torch.ones(1,1),
        rows=torch.tensor([[wrow]]), terminal_valid=torch.ones(1,1,dtype=torch.bool),
        percept_width=chart.mereology.percept_event_width, meaning_start=chart.mereology.percept_event_width)
    stop = len(model.languageSpace._generate_binary_ops) + len(model.languageSpace._generate_unary_ops)
    result = generated(model,point[None,None],torch.tensor([1]),torch.tensor([False]),bank,
        (torch.tensor([[stop]]),),depths=torch.tensor([1]))
    if read == 'word':
        decoded = realize(model,result).materialize()
        assert result.texts(decoded) == ('petal',)
    else:
        i = result.field.concept_ids.tolist().index(word)
        torch.testing.assert_close(result.field.evidence[i,0,0],torch.tensor([1.,0.]))


def test_forced_interpretation_reads_the_explicit_operators_extension(declared_object):
    from Extension import reading, ConceptSet, Context
    from Understanding import ConceptualField
    from Language import InterpretLayer
    model, word, obj, point = declared_object
    owner = model._concept_owner()
    owner.similarity_codebook.trust.lengths[owner._csw_row_of(word),0] = 1.
    owner.interpret.define(word, obj, evidence=(1.,0.))
    assert owner.interpret.denotations(word) == (obj,)
    source = ConceptSet(torch.tensor([word]),torch.tensor([[.6,.2]]))
    field = ConceptualField(source.ids, source.values[:,None,None],
        torch.tensor([[[0,5]]]),torch.tensor([[[1,2]]]),None)
    program = SimpleNamespace(actions=torch.tensor([[0,-1,0]]),
        concept_ids=torch.tensor([obj]),interpret_forced=True,tree=None)
    implicit, = reading(model,program,field,0,torch.tensor([0]))
    explicit = InterpretLayer().compose_extend(source,context=Context(owner,name='interpret'))
    ids = torch.cat((explicit.ids,implicit.field.ids)).unique()
    torch.testing.assert_close(implicit.field.aligned(ids),explicit.aligned(ids))
    assert explicit.ids.tolist() == [word]
    torch.testing.assert_close(explicit.values,source.values)
