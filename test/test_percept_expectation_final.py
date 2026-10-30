"""The eager local owner and the frozen, signed percept reconstruction chart."""
import copy
import pytest
import torch
from Expectation import PerceptExpectation, recombine_percept, percept_reconstruction


@pytest.mark.parametrize('observed,expected',[(1.,1.),(1.,0.),(0.,1.),(.01,1e8)])
def test_recombination_forward_and_jacobian_are_the_attended_full_signal(observed,expected):
    x=torch.tensor([[observed]],dtype=torch.float64,requires_grad=True)
    p=torch.tensor([[expected]],dtype=torch.float64,requires_grad=True)
    attention=torch.tensor(.3,requires_grad=True)
    result=recombine_percept(x,p,attention)
    assert torch.equal(result.content,attention*x)
    actual=torch.autograd.grad(result.content.sum(),(x,attention,p),allow_unused=True,retain_graph=True)
    direct=torch.autograd.grad((attention*x).sum(),(x,attention))
    assert actual[-1] is None
    for left,right in zip(actual,direct):
        assert torch.equal(left,right)
    assert result.residual.item()==observed-expected


def test_an_arrival_teaches_once_and_the_next_forecast_reads_the_update_without_retaining_its_graph():
    owner=PerceptExpectation(2,streams=2,capacity=4)
    keys=torch.eye(2)[None].expand(2,-1,-1).clone().requires_grad_()
    bank=lambda:iter([(torch.arange(2),keys)])
    observed=torch.tensor([[.2,.8],[0.,0.]],requires_grad=True)
    active=torch.tensor([True,True])  # second stream observes an absence, not padding
    with torch.autograd.detect_anomaly():
        cold=owner.forecast(bank,2)
        first=owner.arrive(cold,observed,active,training=True)
        assert first.expected.eq(0).all() and owner.updates==1
        forecast=owner.forecast(bank,2)
        frozen=forecast[0].detach().clone()
        old=[p.detach().clone() for p in owner.parameters()]
        second=owner.arrive(forecast,observed,active,training=True)
        assert any(not torch.equal(x,y) for x,y in zip(old,owner.parameters()))
        assert owner.updates==2 and torch.equal(second.expected,frozen)
        assert observed.grad is None and keys.grad is None
        # Both later trials use the old arrival, even after a local update.
        for _ in range(2):
            leaf=torch.ones_like(observed,requires_grad=True)
            correction,target,rebuilt,error=percept_reconstruction(leaf,observed,second.expected)
            error.sum().backward()
            assert all(p.grad is None for p in owner.parameters())
        first.content.sum().backward()
        assert observed.grad is not None and all(p.grad is None for p in owner.parameters())
        assert (second.residual[1]<0).all()
        after=owner.forecast(bank,2)[0].detach()
        assert not torch.equal(after,frozen)
    before=int(owner.updates)
    owner.arrive(owner.forecast(bank,2),observed,torch.zeros(2,dtype=torch.bool),training=True)
    assert owner.updates==before


def test_local_optimizer_and_context_resume_exactly():
    first=PerceptExpectation(2,streams=1,capacity=3)
    bank=lambda:iter([(torch.arange(2),torch.eye(2)[None])])
    actual=torch.tensor([[.2,.8]])
    active=torch.tensor([True])
    for _ in range(2):first.arrive(first.forecast(bank,1),actual,active,training=True)
    other=PerceptExpectation(2,streams=1,capacity=3)
    other.load_state_dict(copy.deepcopy(first.state_dict()))
    other.restore_continuation(copy.deepcopy(first.continuation()))
    for model in (first,other):model.arrive(model.forecast(bank,1),actual,active,training=True)
    for left,right in zip(first.state_dict().values(),other.state_dict().values()):
        assert torch.equal(left,right)
    a,b=first.optimizer.state_dict(),other.optimizer.state_dict()
    assert a['param_groups']==b['param_groups']
    for key,values in a['state'].items():
        for name,value in values.items():assert torch.equal(value,b['state'][key][name])


@pytest.mark.parametrize('updates',[0,2])
def test_completed_epoch_checkpoint_preserves_private_optimizer_and_next_update(tmp_path,monkeypatch,updates):
    import util
    from Models import BaseModel
    from data import TheData
    from Forgetting import model_boundary
    monkeypatch.setattr(util,'TheCompileBackend','none')
    TheData.processLM({split:dict(text=['alpha.'] if split=='train' else [],label=[])
                      for split in ('train','validation','test')})
    models=[]
    try:
        first=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0];models.append(first)
        owner=first.percept_expectation
        bank=lambda:iter([(torch.tensor([3]),torch.full((1,1,owner.width),.25))])
        actual=torch.full((1,owner.width),.5);active=torch.tensor([True])
        native=torch.full((1,owner.moments.mean.numel()),.5)
        for _ in range(updates):owner.arrive(owner.forecast(bank,1),actual,active,native=native,training=True)
        model_boundary(first,epoch_boundary=True)
        assert getattr(first,'_checkpoint_corpus_cursor',None) is None
        checkpoint=tmp_path/'completed.ckpt';first.save_weights(str(checkpoint))
        saved=torch.load(checkpoint,weights_only=False)
        private=saved['training_state']['percept_expectation']
        assert private['learning_rate']==owner.learning_rate
        assert (private['optimizer'] is None)==(owner.optimizer is None)
        second=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0];models.append(second)
        assert second.load_weights(str(checkpoint),require_match=True)
        restored=second.percept_expectation
        assert (restored.optimizer is None)==(owner.optimizer is None)
        for learner in (owner,restored):
            learner.arrive(learner.forecast(bank,1),actual,active,native=native,training=True)
        for key,value in owner.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
        left,right=owner.optimizer.state_dict(),restored.optimizer.state_dict()
        assert left['param_groups']==right['param_groups']
        for key,values in left['state'].items():
            for name,value in values.items():
                torch.testing.assert_close(value,right['state'][key][name],rtol=0,atol=0)
        if updates:
            saved['training_state'].pop('percept_expectation')
            broken=tmp_path/'missing-private-state.ckpt';torch.save(saved,broken)
            before=next(second.parameters()).detach().clone()
            with pytest.raises(ValueError,match='private optimizer state'):
                second.load_weights(str(broken),require_match=True)
            assert torch.equal(before,next(second.parameters()))
    finally:
        for model in models:model.End();model.symbolSpace.soft_reset()


def test_residual_loss_identity_signed_omissions_and_detached_observation():
    observed=torch.tensor([[[.2,.8],[0.,0.]]],requires_grad=True)
    expected=torch.tensor([[[.3,.6],[.2,.4]]],requires_grad=True)
    recovered=torch.tensor([[[.1,.7],[0.,.1]]],requires_grad=True)
    correction,target,rebuilt,error=percept_reconstruction(recovered,observed,expected)
    torch.testing.assert_close(error,(rebuilt-observed.detach()).square(),rtol=1e-6,atol=1e-8)
    assert (target[0,1]<0).all() and (correction[0,1]<0).all()
    error.sum().backward()
    assert observed.grad is None and expected.grad is None and recovered.grad is not None


def test_native_sentence_local_boundary_ownership_loss_identity_and_compiled_word_bricks(monkeypatch):
    import util
    import SentenceCompose
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from Corpus import admit_training_forms
    monkeypatch.setattr(util,'TheCompileBackend','none')
    graphs=[]
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    monkeypatch.setattr(SentenceCompose,'compile_word_brick',lambda function:
                        torch.compile(function,backend=backend,fullgraph=True,dynamic=False))
    TheData.processLM({split:dict(text=['alpha beta.'] if split=='train' else [],label=[])
                      for split in ('train','validation','test')})
    TheData.source_addresses['train']=[dict(document=0,sentence=0,split='train',row=0)]
    model=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0]
    try:
        optimizer=model.getOptimizer(lr=.001)
        owner=model.percept_expectation
        parameters=tuple(owner.parameters())
        assert not {id(p) for p in parameters} & {id(p) for group in optimizer.param_groups for p in group['params']}
        arrive=owner.arrive
        checks=[]
        def observed_arrival(*args,**kwargs):
            assert not torch.compiler.is_compiling()
            others=tuple(p for p in model.parameters() if id(p) not in {id(q) for q in parameters})
            versions=tuple(p._version for p in others)
            result=arrive(*args,**kwargs)
            assert versions==tuple(p._version for p in others)
            checks.append((int(owner.updates),result.expected.clone()))
            return result
        monkeypatch.setattr(owner,'arrive',observed_arrival)
        score=model._sentence_path_cost
        costs=[]
        def checked_cost(*args,**kwargs):
            result=score(*args,**kwargs)
            objective=model._sentence_cost_registry.total(objective='reconstruction')
            gradients=torch.autograd.grad(objective.sum(),parameters,allow_unused=True,retain_graph=True)
            assert all(g is None for g in gradients)
            costs.extend(model._last_percept_reconstruction)
            return result
        monkeypatch.setattr(model,'_sentence_path_cost',checked_cost)
        admit_training_forms(model,TheData.train_input)
        present(model,split='train',optimizer=optimizer,batch_size=1)
        assert graphs and checks and costs
        assert len(checks)==int(owner.updates)
        assert [count for count,_ in checks]==list(range(1,len(checks)+1))
        for audit in costs:
            torch.testing.assert_close(audit['residual_loss'],audit['full_loss'],rtol=2e-6,atol=2e-6)
        assert all(p.grad is None for p in parameters)
    finally:
        model.End()
        torch._dynamo.reset()
