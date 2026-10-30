"""Regressions for Claude's repair-candidate review; no selected initialization."""
import copy
import json
from dataclasses import replace
import pytest
import torch
from Expectation import sample_alternative, retention_surprise
from Layers import SymbolExpectation, ExpectationComparison, LTM0
from Meaning import ClosingImage, ConceptualMeaning, retained_symbol_prediction, withheld_symbols
from meaning_fixtures import symbol_prediction, estimate_meaning


def test_inverse_cdf_never_selects_a_zero_probability_edge():
    probabilities=torch.tensor([[0.,.1,.2,0.],[0.,.6,.3,0.]])
    for draw, expected in ((0.,1),(.99999999,2)):
        value=torch.full((2,1),draw).clamp_max(torch.nextafter(torch.tensor(1.),torch.tensor(0.)))
        assert sample_alternative(probabilities,value).tolist()==[expected,expected]


def test_bank_confidence_follows_each_symbol_in_every_role():
    head=SymbolExpectation(4,8,4)
    keys=torch.eye(4);head.set_symbol_bank(keys)
    with torch.no_grad():
        for p in head._inter_predictor.parameters():p.zero_()
        head._inter_predictor.local.bias.copy_(torch.tensor([60.,-60.]))
    prior=head._prediction((keys[None],torch.zeros(1,8),torch.zeros(1,4,2)),head._bank_ids,head._bank_orders)
    for roles in (keys[:3],keys[torch.tensor([2,0,1])]):
        image=ClosingImage.form(roles,prior.roles,prior.confidence,prediction=prior,gain=1.)
        assert not image.conceived.any()
        torch.testing.assert_close(image.restore(),roles,atol=0,rtol=0)
    with pytest.raises(ValueError,match='native symbol'):
        ClosingImage.form(keys[:3],prior.roles,prior.presence_logits.sigmoid(),prediction=prior,gain=1.)


def test_native_activated_word_is_observed_by_handle_with_both_lanes(tmp_path,monkeypatch,eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model,_stage_fullgraph_tensor_peer
    from SymbolField import SymbolField
    model=_tiny_canonical_model(tmp_path,monkeypatch,input_width=16,word_buckets='8',stm_capacity=3,chooser_depth=1)
    model.eval()
    _stage_fullgraph_tensor_peer(model,['cat','dog'])
    head=model.symbolSpace.expectation
    from ModelAttention import stage_symbol_bank
    stage_symbol_bank(model,head)
    from PrependedSymbols import word_view
    ids=word_view(model.inputSpace, "object_rows")[:,0]
    atoms=model.inputSpace._ar_word_object_atoms[:,0]
    lanes=atoms.new_tensor([[.6,.2],[0.,.8]])
    values=model._concept_owner().interpret.forward(atoms,object_atoms=atoms,
        presence=atoms.new_tensor([.75,.5]),evidence=lanes)
    depth=torch.ones(2,dtype=torch.long)
    state=(values[:,None],depth,torch.zeros(2,1,dtype=torch.long),torch.zeros(2,1,dtype=torch.long),ids[:,None],atoms.new_ones(2,1))
    field=SymbolField(head,strength=.5,epsilon=1e-6,saturation='clip')
    empty=(torch.zeros_like(state[0]),torch.zeros_like(depth),*state[2:])
    field.anticipate(empty);field.arrive(state,torch.ones(2,dtype=torch.bool),lanes=lanes[:,None])
    selected=(ids[:,None]==head._bank_addresses[None]).long().argmax(-1)
    target=field.reports[-1]['observed']
    torch.testing.assert_close(target[torch.arange(2),selected],lanes)
    field.anticipate(state)
    read=field.read()
    # Use the real admitted lanes as a controlled perfect pre-arrival prior.
    read=replace(read,expected=read.observed.clone())
    gain=read.gains(state[0],identities=state[4])
    torch.testing.assert_close(gain,torch.full((2,1),.5))
    torch.testing.assert_close(state[0][:,0],values,atol=0,rtol=0)
    assert not torch.allclose(values,head._bank_keys[selected])
    # Without a native handle this altered value is not its nearest bank key.
    torch.testing.assert_close(read.gains(values[:,None]),torch.ones(2,1))
    model.End();model.symbolSpace.soft_reset()


def test_omission_surprise_and_non_marks_agree_at_both_native_writers():
    keys=torch.eye(4)
    prior=symbol_prediction(keys,torch.tensor([[1.,0.],[1.,0.],[0.,0.],[0.,0.]]))
    roles=torch.cat((keys[:1],torch.zeros(2,4)))
    target=torch.tensor([[1.,0.],[0.,0.],[0.,0.],[0.,0.]])
    marks={'_withheld_symbols':((1,tuple(keys[1].tolist())),)}
    ordinary=ConceptualMeaning(roles,torch.tensor([True,False,False]))
    withheld=replace(ordinary,bindings=marks)
    comparison=ExpectationComparison(prior,roles,ordinary.role_mask,target-prior.expected_activation,None,activation=target)
    assert comparison.surprise_for(ordinary)>0
    assert comparison.surprise_for(withheld)==0
    for observation in (ordinary,withheld):
        store=LTM0(4,capacity=8)
        _,row=store.append_expectation_pair(estimate_meaning(prior),observation,presence_logits=prior.presence_logits)
        assert store.expectation_pair(row)['surprise']==pytest.approx(float(comparison.surprise_for(observation)))


def test_retained_prior_keys_are_bounded_by_realized_support_and_replay_is_exact():
    n,width=128,32
    # Distinct complete identities with a sparse actual observation.
    keys=torch.eye(width).repeat(4,1)
    keys[:,0]+=torch.arange(n)/1000
    expected=torch.full((n,2),.25)
    prior=symbol_prediction(keys,expected)
    roles=torch.cat((keys[5:6],torch.zeros(2,width)))
    occupied=torch.tensor([True,False,False])
    target=torch.zeros_like(expected);target[5,0]=1
    bindings=prior.retained_bindings(observed=roles,occupied=occupied,activation=target)
    entries=bindings['_expected_symbols']
    assert sum(key is not None for _,_,key,_ in entries)<=2
    estimate=ConceptualMeaning(prior.roles,torch.ones(3,dtype=torch.bool),bindings=bindings)
    observation=ConceptualMeaning(roles,occupied,bindings={'_observed_symbol_activations':((5,(1.,0.)),)})
    store=LTM0(width,capacity=8)
    _,row=store.append_expectation_pair(estimate,observation,presence_logits=prior.presence_logits)
    restored=LTM0(width,capacity=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    for owner in (store,restored):
        pair=owner.expectation_pair(row,gain=.5)
        torch.testing.assert_close(pair['residual'],target-expected)
        assert pair['surprise']==pytest.approx(float(retention_surprise(target-expected,expected)))
    assert len(json.dumps(bindings)) < len(json.dumps(prior.retained_bindings()))/2


def test_expectation_free_placement_is_independent_of_forecasts_and_percept_moments(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from ModelCandidateAttention import context
    from ModelAttention import stage_symbol_bank
    from test_compiled_word_chunk import _tiny_canonical_model,_stage_fullgraph_tensor_peer
    model=_tiny_canonical_model(tmp_path,monkeypatch,input_width=16,word_buckets='8',stm_capacity=3,chooser_depth=1)
    try:
        _stage_fullgraph_tensor_peer(model,['cat','dog'])
        head=model.symbolSpace.expectation
        stage_symbol_bank(model,head)
        values=torch.ones(2,1,2)
        addresses=torch.full((2,1),model.where_registry.slices['parts'][0],dtype=torch.long)
        valid=torch.ones(2,1,dtype=torch.bool)
        n,width=model._candidate_geometry[1][0]
        parts=torch.ones(2,n,width)
        before=context(model,values,addresses,valid,parts=parts,expectation_free=True)
        predicted=symbol_prediction(head._bank_keys,torch.ones(len(head._bank_keys),2))
        predicted=replace(predicted,symbol_ids=head._bank_ids)
        head._inter_last_meaning=[SimpleNamespace(prediction=predicted) for _ in range(2)]
        model._stm_predicted_idea=torch.full((2,3,model.conceptualSpace.stm.concept_dim),17.)
        model.percept_expectation.moments.mean.fill_(1.)
        model.percept_expectation.moments.count.fill_(100)
        model._percept_prior=None
        after=context(model,values,addresses,valid,parts=parts,expectation_free=True)
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        assert after.shape[-1]==model.candidate_attention.context_width
        assert not torch.equal(after,context(model,values,addresses,valid,parts=parts))
    finally:
        model.End();model.symbolSpace.soft_reset()


@pytest.mark.parametrize('percept_gain',[0.,.75])
def test_native_s_gain_uses_raw_arrival_and_p_preserves_the_full_payload(tmp_path,monkeypatch,eager_reading,percept_gain):
    from test_compiled_word_chunk import _tiny_canonical_model,_stage_fullgraph_tensor_peer
    from SymbolField import SymbolField
    import ModelAttention
    model=_tiny_canonical_model(tmp_path,monkeypatch,input_width=16,word_buckets='8',stm_capacity=3,chooser_depth=1,
        training_overrides={'perceptExpectationGain':percept_gain,'symbolExpectationGain':.5})
    model.eval();model._tensor_peer_while_eager=True
    model._chart_compose_per_word=lambda:None
    stage=ModelAttention.stage_percept_expectation
    def perfect_percept_prior(owner,active):
        stage(owner,active)
        # A P expectation feature cannot attenuate either conceptual pole.
        owner.percept_expectation.last_expected.fill_(1.)
    monkeypatch.setattr(ModelAttention,'stage_percept_expectation',perfect_percept_prior)
    payload=model._attention_sentence_payload
    current={};records=[]
    def record_payload(value,index):
        raw=model._attention_poles.detach()
        w=index.clamp(0,raw.shape[1]-1)
        current['raw']=raw.gather(1,w.reshape(1,1,1).expand(len(raw),1,2)).squeeze(1).clone()
        result=payload(value,index)
        current['content']=result[11].clone()
        return result
    monkeypatch.setattr(model,'_attention_sentence_payload',record_payload)
    read=SymbolField.read
    def record_read(owner,**kwargs):
        result=read(owner,**kwargs)
        raw=current.pop('raw',None)
        if raw is None or result is None:return result
        content=current.pop('content')
        ids=kwargs['incoming_ids'];live=kwargs['incoming_rows']
        known=ids[:,None]==result.identities[None]
        selected=known.long().argmax(-1)
        live=live & known.any(-1) & raw.ne(0).any(-1)
        torch.testing.assert_close(kwargs['incoming_lanes'][live],raw[live])
        torch.testing.assert_close(content[live],raw[live])
        # Hold the same perfect S prior over the raw observation at every P
        # setting; no source or head is trained by this mechanism probe.
        expected=result.expected.clone()
        expected[torch.arange(len(ids)),selected]=raw
        result=replace(result,expected=expected)
        gains=result.gains(result.keys.new_zeros(len(ids),1,result.keys.shape[-1]),identities=ids[:,None])[:,0]
        torch.testing.assert_close(gains[live],torch.full_like(gains[live],.5))
        records.extend(gains[live].tolist())
        return result
    monkeypatch.setattr(SymbolField,'read',record_read)
    try:
        with torch.no_grad():
            raw=_stage_fullgraph_tensor_peer(model,['cat','dog'])
            with model._sentence_run():result=model._forward_with_compiled_sentence_state(raw)
            model._publish_compiled_sentence_state(result);model._end_step()
        assert records
    finally:
        model.End();model.symbolSpace.soft_reset()
