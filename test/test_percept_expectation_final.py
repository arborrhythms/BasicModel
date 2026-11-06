"""Arrival loop state, accumulated local training, and the frozen percept chart."""
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
    direct=torch.autograd.grad((attention.detach()*x).sum(),x)[0]
    assert actual[1] is None and actual[2] is None
    assert torch.equal(actual[0],direct)
    assert result.residual.item()==observed-expected


def test_arrival_state_updates_forecast_and_accumulated_loss_teaches_once(monkeypatch):
    import util
    monkeypatch.setattr(util,'TheCompileBackend','none')
    owner=PerceptExpectation(2,streams=2,capacity=4)
    keys=torch.eye(2)[None].expand(2,-1,-1).clone().requires_grad_()
    bank=lambda:iter([(torch.arange(2),keys)])
    contexts=[]
    forward=owner.predictor.forward
    def observe_context(keys,context,*args,**kwargs):
        contexts.append(context.detach().clone())
        assert not context.requires_grad
        return forward(keys,context,*args,**kwargs)
    monkeypatch.setattr(owner.predictor,'forward',observe_context)
    observed=torch.tensor([[.2,.8],[0.,0.]],requires_grad=True)
    active=torch.tensor([True,True])
    parameters=tuple(owner.parameters())
    original=[p.detach().clone() for p in parameters]
    local=[]
    cold=owner.forecast(bank,2)
    first=owner.arrive(cold,observed,active,training=True)
    local.append(owner.last_arrival['local_loss'])
    assert first.expected.eq(0).all() and owner.updates==1
    forecast=owner.forecast(bank,2)
    frozen=forecast[0].detach().clone()
    next_observed=torch.tensor([[.8,.2],[0.,0.]],requires_grad=True)
    second=owner.arrive(forecast,next_observed,active,training=True)
    local.append(owner.last_arrival['local_loss'])
    assert all(torch.equal(x,y) for x,y in zip(original,parameters))
    assert owner.updates==2 and torch.equal(second.expected,frozen)
    assert observed.grad is None and keys.grad is None
    owner.forecast(bank,2)
    torch.testing.assert_close(contexts[-2],torch.cat((observed.detach(),observed.detach()),-1),rtol=0,atol=0)
    torch.testing.assert_close(contexts[-1],torch.cat(((observed.detach()+next_observed.detach())/2,next_observed.detach()),-1),rtol=0,atol=0)
    assert owner.context_count.tolist()==[2,2]
    assert (second.residual[1]<0).all()
    for _ in range(2):
        leaf=torch.ones_like(observed,requires_grad=True)
        correction,target,rebuilt,error=percept_reconstruction(leaf,observed,second.expected)
        error.sum().backward()
        assert all(p.grad is None for p in parameters)
    first.content.sum().backward()
    assert observed.grad is not None and all(p.grad is None for p in parameters)
    observed.grad=None
    loss=torch.stack(local).sum()
    expected=torch.autograd.grad(loss,parameters,retain_graph=True)
    optimizer=torch.optim.Adam(parameters,lr=.01)
    loss.backward()
    for p,g in zip(parameters,expected):torch.testing.assert_close(p.grad,g)
    assert observed.grad is None and next_observed.grad is None and keys.grad is None
    assert any(p.grad.abs().sum() for p in parameters)
    optimizer.step()
    assert any(not torch.equal(x,y) for x,y in zip(original,parameters))
    assert not hasattr(owner,'optimizer')
    before=int(owner.updates)
    owner.arrive(owner.forecast(bank,2),observed,torch.zeros(2,dtype=torch.bool),training=True)
    assert owner.updates==before


def test_arrival_brick_compiles_and_equals_eager_loop_state_loss_and_gradient(monkeypatch,tmp_path):
    import util
    from Expectation import percept_arrival_brick
    from test_one_backward import receipt
    monkeypatch.setattr(util,'TheCompileBackend','none')
    original=PerceptExpectation(2,streams=2,capacity=3).double()
    original.known[:2]=True
    states=copy.deepcopy(original.state_dict())
    observed=torch.tensor([[[.2,.8],[.1,.4]],[[.5,.3],[0.,0.]],[[.9,.1],[.2,.7]]],dtype=torch.float64,requires_grad=True)
    active=torch.tensor([[True,True],[True,False],[True,True]])
    outputs=[];calls=[]
    def backend(graph,inputs):
        calls.append(graph)
        return torch._inductor.compile(graph,inputs)
    for compiled in (False,True):
        owner=PerceptExpectation(2,streams=2,capacity=3).double()
        owner.load_state_dict(copy.deepcopy(states))
        owner._arrival_brick=(torch.compile(percept_arrival_brick,backend=backend,fullgraph=True,dynamic=False)
                              if compiled else percept_arrival_brick)
        bank=lambda:iter([(torch.arange(2),torch.eye(2,dtype=torch.float64)[None].expand(2,-1,-1))])
        losses=[];forecasts=[];contexts=[]
        for index in range(len(observed)):
            contexts.append(torch.cat((owner.context_mean,owner.context_recent),-1).clone())
            forecast=owner.forecast(bank,2);forecasts.append(forecast[0].detach().clone())
            owner.arrive(forecast,observed[index],active[index],training=True)
            losses.append(owner.last_arrival['local_loss'])
            torch.testing.assert_close(owner.context_recent[active[index]],observed[index,active[index]].detach())
        summed=torch.stack(losses).sum()
        summed.backward()
        assert observed.grad is None
        outputs.append(dict(state=copy.deepcopy(owner.state_dict()),loss=summed.detach(),
            gradients=[p.grad.clone() for p in owner.parameters()],forecasts=forecasts,contexts=contexts))
    assert calls
    for key,value in outputs[0]['state'].items():torch.testing.assert_close(value,outputs[1]['state'][key],rtol=0,atol=0)
    for key in ('gradients','forecasts','contexts'):
        for a,b in zip(outputs[0][key],outputs[1][key]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(outputs[0]['loss'],outputs[1]['loss'],rtol=0,atol=0)
    receipt(tmp_path,'compiled-arrival',dict(arrivals=len(observed),graphs=len(calls),backend='inductor',
        state_equal=True,loss_equal=True,gradient_equal=True,target_gradient=False))


def test_arrival_context_resume_exactly_without_a_private_optimizer(monkeypatch):
    import util
    monkeypatch.setattr(util,'TheCompileBackend','none')
    first=PerceptExpectation(2,streams=1,capacity=3)
    bank=lambda:iter([(torch.arange(2),torch.eye(2)[None])])
    actual=torch.tensor([[.2,.8]]);active=torch.tensor([True])
    for _ in range(2):first.arrive(first.forecast(bank,1),actual,active,training=True)
    other=PerceptExpectation(2,streams=1,capacity=3)
    other.load_state_dict(copy.deepcopy(first.state_dict()))
    other.restore_continuation(copy.deepcopy(first.continuation()))
    for model in (first,other):model.arrive(model.forecast(bank,1),actual,active,training=True)
    for left,right in zip(first.state_dict().values(),other.state_dict().values()):assert torch.equal(left,right)
    assert not hasattr(first,'optimizer') and not hasattr(other,'optimizer')
    with pytest.raises(ValueError,match='private optimizers retired'):
        other.restore_continuation({'optimizer':{'state':{}},'learning_rate':.001})


@pytest.mark.parametrize('updates',[0,2])
def test_completed_epoch_checkpoint_preserves_arrival_state_and_shared_optimizer(tmp_path,monkeypatch,updates):
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
        optimizer=first.getOptimizer(lr=.001);first._optimizer=optimizer
        bank=lambda:iter([(torch.tensor([3]),torch.full((1,1,owner.width),.25))])
        actual=torch.full((1,owner.width),.5);active=torch.tensor([True])
        native=torch.full((1,owner.moments.mean.numel()),.5)
        losses=[]
        for _ in range(updates):
            owner.arrive(owner.forecast(bank,1),actual,active,native=native,training=True)
            losses.append(owner.last_arrival['local_loss'])
        if losses:
            torch.stack(losses).sum().backward();optimizer.step();optimizer.zero_grad()
        model_boundary(first,epoch_boundary=True)
        checkpoint=tmp_path/'completed.ckpt';first.save_weights(str(checkpoint))
        from test_one_backward import checkpoint_without_random_state
        saved=checkpoint_without_random_state(checkpoint)
        assert saved['training_state']['percept_expectation']==dict(version=2,adaptation='arrival_state')
        second=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0];models.append(second)
        assert second.load_weights(str(checkpoint),require_match=True)
        restored=second.percept_expectation;resumed=second.getOptimizer(lr=.001)
        for learner,opt in ((owner,optimizer),(restored,resumed)):
            learner.arrive(learner.forecast(bank,1),actual,active,native=native,training=True)
            learner.last_arrival['local_loss'].sum().backward();opt.step();opt.zero_grad()
        for key,value in owner.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
        saved['training_state']['percept_expectation']=dict(optimizer={'state':{}},learning_rate=.001)
        broken=tmp_path/'retired-private-state.ckpt';torch.save(saved,broken)
        before=next(second.parameters()).detach().clone()
        with pytest.raises(ValueError,match='private optimizer retired'):
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
    from CompileGuard import guarded_compile
    monkeypatch.setattr(SentenceCompose,'compile_word_brick',lambda function, *, initial_phase=None:
                        guarded_compile(function,initial_phase=initial_phase,
                                        backend=backend,fullgraph=True,dynamic=False))
    TheData.processLM({split:dict(text=['alpha beta.'] if split=='train' else [],label=[])
                      for split in ('train','validation','test')})
    TheData.source_addresses['train']=[dict(document=0,sentence=0,split='train',row=0)]
    model=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0]
    try:
        optimizer=model.getOptimizer(lr=.001)
        owner=model.percept_expectation
        parameters=tuple(owner.parameters())
        assert {id(p) for p in parameters} <= {id(p) for group in optimizer.param_groups for p in group['params']}
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
