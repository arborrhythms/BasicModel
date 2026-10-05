"""6.8 §7 ports: bracket coverage/priming and word distribution supervision.

The prior producer's learned next-word objective belongs to expectation;
its scope and coverage belong to the typed bracket table. The complete old
file is retained in the operators-attention source and test-port receipts.
"""
import os
from pathlib import Path
import pytest
import torch
from Attention import BracketKeys, narrow_words
from Language import OperationSelectionLayer
from Layers import BracketExpectation


def _contiguous_spans(B,K,w):
    return torch.tensor([[[k*w,(k+1)*w] for k in range(K)]]*B)


def _module_inputs(B=2,K=5,w=8,D=16,seed=0):
    torch.manual_seed(seed)
    spans=_contiguous_spans(B,K,w)
    percept=torch.randn(B,K*w,D)
    concept_q=torch.randn(B,D)
    symbol_q=torch.randn(B,D)
    return spans,percept,concept_q,symbol_q


def _reading(spans,percept,prior=None):
    keys=BracketKeys._span_keys(percept,spans)
    chooser=OperationSelectionLayer(d_model=keys.shape[-1])
    known=spans[...,1]>spans[...,0]
    return narrow_words(chooser,keys,spans,known,budget=32,prior=prior)


def _distribution(percept,spans,*,training=True):
    keys=BracketKeys._span_keys(percept,spans)
    B,W,D=keys.shape
    owner=BracketExpectation(n_symbols=W,max_depth=W,n_dim=D,concept_dim=D)
    targets=torch.arange(W)[None].expand(B,-1)
    valid=spans[...,1]>spans[...,0]
    def read():
        return owner.expect('word',keys,keys,valid,targets,teacher_forcing=training,active=valid)
    return owner,read


def test_shift_bootstrap_selects_each_word_once():
    spans,percept,*_=_module_inputs()
    value=_reading(spans,percept)
    assert value.accepted.all() and not value.descended.any()
    assert value.table.spent.tolist()==[9,9]
    for row in range(len(spans)):
        assert sorted(map(tuple,value.table.intervals[row,value.table.done[row]].tolist()))==list(map(tuple,spans[row].tolist()))


def test_scope_is_normalized_unit_range():
    spans,percept,*_=_module_inputs();out=_reading(spans,percept)
    scope=out.table.intervals[out.table.done]/percept.shape[1]
    assert scope.shape[-1]==2
    assert float(scope.min())>=0. and float(scope.max())<=1.
    assert (scope[:,1]>=scope[:,0]).all()


def test_coverage_mask_excludes_consumed():
    spans,percept,*_=_module_inputs();out=_reading(spans,percept)
    for actions in out.actions:
        gloss=actions[(actions>=0)&(actions%6==2)]//6
        assert gloss.unique().numel()==len(gloss)==spans.shape[1]


def test_padding_spans_never_selected():
    spans,percept,*_=_module_inputs();spans[:,-1]=0
    out=_reading(spans,percept)
    assert float(out.accepted[:,-1].float().max())<1e-4
    assert out.accepted[:,:-1].all()


def test_codebook_retrieval_prior_is_the_subsymbolic_term():
    B,K,w,D=1,5,6,8
    spans=_contiguous_spans(B,K,w)
    torch.manual_seed(1)
    percept=torch.randn(B,K*w,D)
    keys=BracketKeys._span_keys(percept,spans)
    W=keys[0].clone();boosts=torch.ones(K);boosts[3]=5.
    prior=BracketKeys._codebook_retrieval_prior(keys,W,None,boosts)
    assert prior is not None and prior.shape==(B,K)
    assert int(prior[0].argmax())==3


def test_codebook_path_preserves_gradient_boundary():
    spans,percept,cq,_=_module_inputs()
    keys=BracketKeys._span_keys(percept.requires_grad_(),spans)
    W=torch.randn(32,16,requires_grad=True)
    prior=BracketKeys._codebook_retrieval_prior(keys,W,cq.requires_grad_(),None)
    assert not prior.requires_grad
    out=_reading(spans,percept,prior);out.values.sum().backward()
    assert W.grad is None and cq.grad is None and percept.grad is None


def test_codebook_path_keeps_complete_word_coverage():
    spans,percept,cq,_=_module_inputs()
    keys=BracketKeys._span_keys(percept,spans)
    prior=BracketKeys._codebook_retrieval_prior(keys,torch.randn(32,16),cq,None)
    assert _reading(spans,percept,prior).accepted.all()


def test_no_codebook_falls_back_to_content_choice():
    spans,percept,cq,_=_module_inputs()
    assert BracketKeys._codebook_retrieval_prior(percept,None,cq,None) is None
    assert _reading(spans,percept).accepted.all()


def test_next_word_ce_matches_neg_log_alpha():
    spans,percept,*_=_module_inputs();owner,read=_distribution(percept,spans)
    out=read();targets=torch.arange(spans.shape[1])[None,:,None].expand(len(spans),-1,-1)
    expect=-torch.log(out.probabilities.gather(-1,targets)[...,0].clamp_min(1e-12))
    torch.testing.assert_close(out.loss,expect,atol=1e-4,rtol=0.)


def test_eval_does_not_build_a_training_graph():
    spans,percept,*_=_module_inputs();owner,read=_distribution(percept,spans,training=False)
    owner.eval()
    with torch.no_grad():out=read()
    assert not out.loss.requires_grad and not out.probabilities.requires_grad


def test_no_spans_is_noop():
    # Empty readings have no candidate bank to score; narrowing has no work.
    out=_reading(torch.empty(2,0,2,dtype=torch.long),torch.randn(2,8,4))
    assert not out.table.valid.any() and not out.table.spent.any()


def test_gradient_stops_at_primed_symbols():
    spans,percept,cq,sq=_module_inputs()
    percept.requires_grad_();cq.requires_grad_();sq.requires_grad_()
    owner,read=_distribution(percept,spans);read().loss.mean().backward()
    assert percept.grad is None and cq.grad is None and sq.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in owner.word_predictor.parameters())


def test_loss_trains_distribution_toward_target():
    spans,percept,*_=_module_inputs(seed=3)
    owner,read=_distribution(percept,spans)
    opt=torch.optim.SGD(owner.word_predictor.parameters(),lr=.5)
    first=read()
    for _ in range(25):
        opt.zero_grad();out=read();out.loss.mean().backward();opt.step()
    last=read()
    assert float(last.loss.mean().detach())<float(first.loss.mean().detach())
    assert float(last.probabilities[0,2,2].detach())>=float(first.probabilities[0,2,2].detach())


def _build(name):
    from configuration_fixtures import small_retained
    from recon_bench import _build_model
    with small_retained(name) as path:model,*_=_build_model(path)
    return model


def _batch(model):
    return model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])


@pytest.mark.slow
@pytest.mark.parametrize('name',['MM_reading.xml','MM_20M_xor.xml'])
def test_normal_configs_share_bracket_and_expectation_owners(name):
    model=_build(name)
    assert not hasattr(model,'reading_attention')
    assert set(model._stm_reducer().attention_operations)>={0,1,2}
    assert 'word' in model.symbolSpace.expectation.enabled_levels


@pytest.mark.slow
def test_forward_is_finite_and_deterministic(monkeypatch):
    model=_build('MM_reading.xml');x=_batch(model)
    from configuration_fixtures import freeze_admission
    freeze_admission(model,x,monkeypatch);model.eval()
    with torch.no_grad():a=model(x)[2];b=model(x)[2]
    assert torch.isfinite(a).all() and torch.equal(a,b)


@pytest.mark.slow
def test_train_forward_then_backward_is_finite():
    model=_build('MM_reading.xml');model.train();model(_batch(model))
    out=model._word_expectation
    assert torch.isfinite(out.loss).all();out.loss.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.symbolSpace.expectation.parameters() if p.grad is not None)


@pytest.mark.slow
def test_producer_writes_bracket_scope_and_registers_loss():
    model=_build('MM_reading.xml');model.train();model(_batch(model))
    scope=model.conceptualSpace._passback_scope_where
    assert scope.shape==(4,2) and (scope>=0).all() and (scope<=1).all()
    table=model._attention_words.table
    assert table.done.any() and model._word_expectation.loss.requires_grad


@pytest.mark.slow
def test_registered_loss_backprops_to_producer():
    model=_build('MM_reading.xml');model.train();model(_batch(model))
    model.zero_grad(set_to_none=True);model._word_expectation.loss.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.symbolSpace.expectation.word_predictor.parameters())


@pytest.mark.slow
def test_producer_scope_replaces_previous_input():
    model=_build('MM_reading.xml');model.eval()
    with torch.no_grad():
        model(_batch(model));first=model._attention_words
        model(model.inputSpace.prepInput(['cat']))
    assert model._attention_words is not first
    assert int(model._attention_words.accepted.sum())==1


@pytest.mark.slow
def test_producer_params_reach_the_optimizer():
    model=_build('MM_reading.xml');opt=model.getOptimizer(lr=.01)
    params={p.data_ptr() for p in model.symbolSpace.expectation.word_predictor.parameters()}
    owned={p.data_ptr() for g in opt.param_groups for p in g['params']}
    assert params and params<=owned
