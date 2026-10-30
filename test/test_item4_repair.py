"""Repair-round boundaries: exact identity, one gain, local statistics and credit."""
from types import SimpleNamespace
import pytest
import torch
from Expectation import SymbolRead, symbol_activations, gain_field, retention_surprise, PerceptMoments
from Meaning import ClosingImage
from meaning_fixtures import symbol_prediction


def test_nearby_composed_value_is_neither_a_bank_observation_nor_attenuated():
    keys=torch.eye(32)[None]
    values=keys[:,:3].clone(); values[:,1] *= .99; values[:,2] += 1e-6
    observed=symbol_activations(keys,values,torch.ones(1,3,dtype=torch.bool))
    assert observed[0,0,0] == observed[0,2,0] == 1
    assert observed[0,1,0] == 0
    read=SymbolRead(keys,torch.tensor([1.,0.]).expand(1,32,2),1.,1e-6,'clip')
    torch.testing.assert_close(read.gains(values),torch.tensor([[0.,1.,0.]]))
    roles=values[0,:3]
    prior=symbol_prediction(keys[0],torch.tensor([1.,0.]).expand(32,2))
    view=ClosingImage.form(roles,prior.roles,torch.ones(len(prior.symbol_keys)),prediction=prior,gain=1.)
    torch.testing.assert_close(view.conceived[1],roles[1])
    assert not view.conceived[0].any()


@pytest.mark.parametrize('gain',[0.,.4,1.])
def test_native_compose_reads_dimmed_content_and_preserves_raw_candidates(gain,monkeypatch):
    from Language import OperationSelectionLayer, LanguageSpace
    class Mean(torch.nn.Module):
        def forward(self,a,b,**kwargs): return (a+b)/2
    layer=OperationSelectionLayer(d_model=4,ops=[Mean()],chooser='mlp')
    keys=torch.eye(4)[None]
    layer._symbol_expectation_read=SymbolRead(keys,torch.tensor([1.,0.]).expand(1,4,2),gain,1e-6,'clip', identities=torch.arange(4))
    host=SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
        _structural_context=lambda **kw:None,_compose_binary_rules=(),_compose_unary_rules=())
    values=keys[:,:2].clone()
    state=(values,torch.tensor([2]),torch.tensor([[1,0]]),torch.zeros(1,2,dtype=torch.long),
           torch.tensor([[-1,1]]),torch.tensor([[0.,.7]]))
    from chunk_fixtures import from_stack
    state = from_stack(*state)
    seen=[]; original=layer.chooser.score_binary
    def scores(x,candidates,*args,**kwargs):
        seen.append((x.clone(),candidates.clone()))
        return original(x,candidates,*args,**kwargs)
    monkeypatch.setattr(layer.chooser,'score_binary',scores)
    choice=LanguageSpace.choose_operation(host,state,torch.tensor([True]),slots=2,sample=False)
    raw=values.flip(1)
    torch.testing.assert_close(seen[0][0],raw)
    if gain:
        torch.testing.assert_close(seen[1][0],raw*(1-gain))
    torch.testing.assert_close(state[0],values,rtol=0,atol=0)
    assert state[5].tolist()==[[0.,pytest.approx(.7)]]
    # A new mixed value is not one of the raw bank keys.
    torch.testing.assert_close(seen[-1][1],seen[0][1])


def test_order_zero_pole_handoff_keeps_the_full_percept_signal():
    from Models import BasicModel
    pair=torch.tensor([[[.6,.2]]])
    model=SimpleNamespace(_attention_poles=pair,_attention_words=SimpleNamespace(accepted=torch.ones(1,1,dtype=torch.bool)),
        _word_expectation=SimpleNamespace(gain=torch.tensor([[.5]])),_percept_word_gain=torch.tensor([[.8]]))
    x=torch.ones(1,1,4); one=torch.ones(1,1,dtype=torch.bool)
    payload=(x,None,None,torch.ones(1,1),None,None,x[:,0],one,one,None,None,pair[:,0])
    result=BasicModel._attention_sentence_payload(model,payload,torch.tensor(0))
    torch.testing.assert_close(result[-1],pair[:,0])


def test_norm_and_leaf_coefficients_receive_one_explicit_gain_in_compose_view():
    from Language import OperationSelectionLayer, LanguageSpace
    calls = []
    class Weighted(torch.nn.Module):
        uses_operand_activation = True
        def compose(self, left, right, *, left_activation, right_activation):
            calls.append((left_activation.clone(), right_activation.clone()))
            return left*left_activation[...,None] + right*right_activation[...,None]
    layer = OperationSelectionLayer(d_model=4, ops=[Weighted()], chooser='mlp')
    keys = torch.eye(4)[None]
    layer._symbol_expectation_read = SymbolRead(keys, torch.tensor([1.,0.]).expand(1,4,2), .5, 1e-6, 'clip', identities=torch.arange(4))
    host = SimpleNamespace(language_layer=SimpleNamespace(operation_layer=layer),
        _structural_context=lambda **kw: None, _compose_binary_rules=(), _compose_unary_rules=())
    state = (keys[:,:2].clone(), torch.tensor([2]), torch.tensor([[1,0]]), torch.zeros(1,2,dtype=torch.long),
             torch.tensor([[-1,1]]), torch.tensor([[0.,.7]]))
    from chunk_fixtures import from_stack
    state = from_stack(*state)
    LanguageSpace.choose_operation(host, state, torch.tensor([True]), slots=2, sample=False)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], torch.tensor([[.7]]))
    torch.testing.assert_close(calls[0][1], torch.ones(1,1))
    torch.testing.assert_close(calls[1][0], torch.tensor([[.35]]))
    torch.testing.assert_close(calls[1][1], torch.tensor([[.5]]))
    torch.testing.assert_close(state[0], keys[:,:2], atol=0, rtol=0)
    torch.testing.assert_close(state[5], torch.tensor([[0.,.7]]), atol=0, rtol=0)


def test_retention_surprise_counts_omission_but_not_empty_bank_padding():
    actual=torch.tensor([[1.,0.],[0.,0.]])
    expected=torch.tensor([[1.,0.],[.1,0.]])
    first=retention_surprise(actual-expected,expected)
    padded=torch.cat((expected,torch.zeros(1000,2)))
    target=torch.cat((actual,torch.zeros(1000,2)))
    torch.testing.assert_close(retention_surprise(target-padded,padded),first)
    padded[-1,0]=1.
    assert retention_surprise(target-padded,padded) > first
    assert retention_surprise(torch.tensor([[-1.,0.]]),torch.tensor([[1.,0.]])) == pytest.approx(1/3)


@pytest.mark.parametrize('confidence',[0.,.25,1.])
def test_closing_confidence_multiplies_gain_and_open_role_keeps_prime(confidence):
    prior=symbol_prediction(torch.eye(4),torch.tensor([1.,0.]).expand(4,2))
    raw=torch.eye(4)[:3]
    view=ClosingImage.form(raw,prior.roles,torch.full((len(prior.symbol_keys),),confidence),prediction=prior,gain=.8,
        object_mask=torch.tensor([0.,1.,0.]))
    torch.testing.assert_close(view.conceived[0],raw[0]*(1-.8*confidence))
    torch.testing.assert_close(view.conceived[1],raw[1])
    torch.testing.assert_close(view.expected,prior.expected_activation)


def test_percept_staging_never_teaches_and_native_arrivals_update_shared_owner_once():
    from ModelAttention import stage_percept_expectation
    from Expectation import PerceptExpectation
    owner=PerceptExpectation(2,streams=1,capacity=3)
    values=torch.tensor([[[.1,.3],[.3,.5],[.7,.9]]])
    model=SimpleNamespace(percept_expectation=owner,training=False)
    stage_percept_expectation(model,torch.ones(1,2,dtype=torch.bool))
    assert owner.moments.count == owner.updates == 0
    model.training=True
    stage_percept_expectation(model,torch.ones(1,2,dtype=torch.bool))
    assert owner.moments.count == owner.updates == 0
    bank=lambda: iter([(torch.arange(3),values)])
    for index in range(3):
        owner.arrive(owner.forecast(bank,1),values[:,index],torch.tensor([True]),training=True)
    assert owner.moments.count == owner.updates == 3
    model.training=False
    stage_percept_expectation(model,torch.ones(1,2,dtype=torch.bool))
    assert owner.moments.count == owner.updates == 3
    assert not hasattr(model,'_percept_word_gain')


def test_content_assertion_and_exact_tie_credit_are_fail_loud():
    from SentenceCredit import score_function
    with pytest.raises(RuntimeError,match='nonnegative'):
        gain_field(torch.tensor([[-.1,0.]]),torch.zeros(1,2),strength=0.)
    p=torch.tensor([[.4],[.6]],requires_grad=True)
    # A different row's candidate-attention R cannot credit a tied compose row.
    loss,_=score_function(p,torch.tensor([[51.],[60.]]),torch.tensor([[True],[False]]),
        torch.tensor([[1.,1.],[1.,2.]]))
    assert not loss.requires_grad and not loss.any()
    with pytest.raises(RuntimeError,match='proposal correction'):
        score_function(p,torch.zeros_like(p),torch.ones_like(p,dtype=torch.bool),torch.tensor([[1.,2.],[1.,2.]]))


def test_word_read_starts_from_factual_previous_context_and_never_forecasts_itself():
    from Layers import SymbolExpectation
    from WordExpectation import word_distribution
    head=SymbolExpectation(4,8,4,concept_dim=4)
    head.set_symbol_bank(torch.eye(4))
    keys=torch.eye(4)[None]; words=keys[:,:2];valid=torch.ones(1,4,dtype=torch.bool)
    inputs=[]
    hook=head.word_predictor.register_forward_pre_hook(lambda _,args:inputs.append(tuple(v.detach().clone() for v in args)))
    try:
        head.predict_and_observe_stm_end_state([1],[torch.eye(4)[2:3]],layout='infix',evidence=[(0.,.8)])
        word_distribution(head,words,keys,valid,torch.tensor([[0,1]]))
        torch.testing.assert_close(inputs[-2][1],torch.cat((keys[:,2],keys[:,2]),-1))
        torch.testing.assert_close(inputs[-2][2][0,2],torch.tensor([0.,.8]))
        assert inputs[-2][2][0,0].sum()==0
        assert inputs[-1][2][0,0,0]==1
    finally:hook.remove()
