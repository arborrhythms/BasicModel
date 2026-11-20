"""6.8 §7 ports: the surprise projection and word distribution supervision.

The prior producer's learned next-word objective belongs to expectation;
which word is read belongs to the looks (test_item17_looks).
"""
import os
from pathlib import Path
import pytest
import torch
from Attention import BracketKeys
from Language import OperationSelectionLayer
from Layers import SymbolExpectation


def _contiguous_spans(B,K,w):
    return torch.tensor([[[k*w,(k+1)*w] for k in range(K)]]*B)


def _module_inputs(B=2,K=5,w=8,D=16,seed=0):
    torch.manual_seed(seed)
    spans=_contiguous_spans(B,K,w)
    percept=torch.randn(B,K*w,D)
    concept_q=torch.randn(B,D)
    symbol_q=torch.randn(B,D)
    return spans,percept,concept_q,symbol_q


def _distribution(percept,spans,*,training=True):
    keys=BracketKeys._span_keys(percept,spans)
    B,W,D=keys.shape
    owner=SymbolExpectation(n_symbols=W,max_depth=W,n_dim=D,concept_dim=D)
    owner.set_symbol_bank(torch.eye(owner.concept_dim))
    targets=torch.arange(W)[None].expand(B,-1)
    valid=spans[...,1]>spans[...,0]
    def read():
        return owner.expect('word',keys,keys,valid,targets,active=valid)
    return owner,read


def test_projection_reads_the_two_bracket_surprise_components():
    from AttentionContext import project
    B,K,w,D=1,5,6,8
    spans=_contiguous_spans(B,K,w)
    torch.manual_seed(1)
    percept=torch.randn(B,K*w,D)
    keys=BracketKeys._span_keys(percept,spans)
    mapping=torch.nn.Linear(D+2,1,bias=False)
    with torch.no_grad():
        mapping.weight.zero_()
        mapping.weight[0,-2]=1
    surprise=torch.ones(B,K,2)
    surprise[:,3,0]=5
    scores=project(mapping,keys.mean(1),surprise).squeeze(-1)
    assert scores.shape==(B,K) and int(scores[0].argmax())==3


def test_bracket_surprise_preserves_gradient_boundary():
    from AttentionContext import project
    spans,percept,cq,_=_module_inputs()
    W=torch.randn(32,16,requires_grad=True)
    cq.requires_grad_();percept.requires_grad_()
    keys=BracketKeys._span_keys(percept,spans)
    # These inputs stand in for observed and forecast evidence, kept detached.
    pair=(keys @ W.T)[..., :2]
    projection=torch.nn.Linear(cq.shape[-1]+2,cq.shape[-1],bias=False)
    project(projection,cq,pair).sum().backward()
    assert W.grad is None and cq.grad is None and percept.grad is None
    assert projection.weight.grad is not None


def test_next_word_activation_error_matches_both_native_lanes():
    spans,percept,*_=_module_inputs();owner,read=_distribution(percept,spans)
    out=read();B,W=spans.shape[:2]
    target=torch.zeros(B,W,W,2)
    target[:,:, :,0]=torch.eye(W)[None]
    torch.testing.assert_close(out.loss,(target-out.activation).square().mean((-1,-2)))


def test_eval_does_not_build_a_training_graph():
    spans,percept,*_=_module_inputs();owner,read=_distribution(percept,spans,training=False)
    owner.eval()
    with torch.no_grad():out=read()
    assert not out.loss.requires_grad and not out.probabilities.requires_grad


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
def test_normal_configs_share_expectation_owners(name):
    model=_build(name)
    assert not hasattr(model,'reading_attention')
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
def test_reading_takes_looks_and_registers_loss():
    model=_build('MM_reading.xml');model.train();model(_batch(model))
    assert model._attention_words.looks.any() and model._word_expectation.loss.requires_grad


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
