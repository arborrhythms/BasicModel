"""Two answer-owned readers: kept-only presentation and paired comparison."""
from types import SimpleNamespace

import torch
from torch import nn


def test_comparison_copies_tied_parameters_without_rng_and_restores_presentation():
    from AnswerComparison import AnswerComparison
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.reader=nn.Linear(2,1)
            self.alias=self.reader
            self.comparison_reader=AnswerComparison()
        def objective_parameter_groups(self, optimizer):
            return dict(output=tuple(self.parameters()))
        def _sentence_reader_error(self, root):
            return self.alias(root.detach())
    model=Model()
    root=torch.tensor([[1.,2.]],requires_grad=True)
    before=model._sentence_reader_error(root)
    rng=torch.get_rng_state().clone()
    comparison=model.comparison_reader(model,root)
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(before,comparison,rtol=0,atol=0)
    assert len(model.comparison_reader.weights)==2
    assert not ({id(p) for p in model.reader.parameters()} &
                {id(p) for p in model.comparison_reader.parameters()})
    comparison.sum().backward()
    assert root.grad is None and all(p.grad is None for p in model.reader.parameters())
    assert all(p.grad is not None for p in model.comparison_reader.parameters())
    with torch.no_grad():model.comparison_reader.weights['reader/bias'].add_(5.)
    torch.testing.assert_close(model.comparison_reader(model,root),before+5)
    torch.testing.assert_close(model._sentence_reader_error(root),before,rtol=0,atol=0)
    assert model.reader.weight is model.alias.weight
    restored=Model()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.comparison_reader(restored,root),before+5)


def test_presented_rejected_root_has_zero_weight_and_each_reader_steps_once():
    from Layers import Error
    from SentenceCredit import reader_weights,reader_costs
    from ObjectiveOwnership import backward_owned
    active=torch.tensor([True,True,True,False])
    parts=torch.zeros(4,2,3)
    parts[1,0,0]=1.
    draw=dict(compose_round=torch.tensor([0,0,-1,-1]))
    shown=reader_weights(parts,active,None)
    judge=reader_weights(parts,active,draw)
    torch.testing.assert_close(shown,torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,0.]]))
    torch.testing.assert_close(judge,torch.tensor([[.5,.5],[.5,.5],[1.,0.],[0.,0.]]))
    p=nn.Parameter(torch.tensor(1.));q=nn.Parameter(torch.tensor(1.))
    roots=[torch.tensor([1.,2.,3.,4.],requires_grad=True),torch.tensor([9.,8.,7.,6.],requires_grad=True)]
    def costs(parameter,weights):
        registries=[]
        for root in roots:
            registry=Error(row_mask=active)
            registry.error('answer',(parameter*root.detach()).square(),1.,objective='output')
            registries.append(registry)
        return reader_costs(registries,weights)['output']
    presented,comparison=costs(p,shown),costs(q,judge)
    actual=torch.autograd.grad(presented+comparison,(p,q),retain_graph=True)
    torch.testing.assert_close(actual[0],2*(1+64+9)/torch.tensor(3.))
    torch.testing.assert_close(actual[1],2*((1+81)/2+(4+64)/2+9)/torch.tensor(3.))
    optimizer=torch.optim.Adam([p,q],lr=.01)
    backward_owned({'output':presented+comparison},{'output':(p,q)})
    optimizer.step()
    assert optimizer.state[p]['step']==optimizer.state[q]['step']==1
    assert all(root.grad is None for root in roots)


def test_sentence_advantage_reads_comparison_and_optimizer_owns_both(monkeypatch):
    from test_mm_xor import _fresh_model
    model,_,data=_fresh_model('data/XOR_grammar.xml')
    try:
        from AnswerComparison import AnswerComparison
        compare=AnswerComparison.forward
        changed=[]
        def different_reader(reader,model,*args,**kwargs):
            reader.sync(model)
            if not changed:
                key=next(k for k in reader.weights if k.endswith('_readout_bias'))
                with torch.no_grad():reader.weights[key].add_(.3)
                changed.append(True)
            return compare(reader,model,*args,**kwargs)
        monkeypatch.setattr(AnswerComparison,'forward',different_reader)
        original=model._sentence_answer_error
        observations=[]
        def answer(*args,**kwargs):
            cost=original(*args,**kwargs)
            if cost is not None:
                observations.append((cost.detach().clone(),
                    model._sentence_comparison_registry.total(objective='output').detach().clone()))
            return cost
        # _sentence_path_cost deliberately dispatches the class method.
        from Models import BasicModel
        monkeypatch.setattr(BasicModel,'_sentence_answer_error',lambda self,*a,**k:answer(*a,**k))
        optimizer=model.getOptimizer(lr=.01)
        model.runEpoch(optimizer=optimizer,batchSize=4,split='train')
        assert len(observations)==2
        for returned,judge in observations:torch.testing.assert_close(returned,judge,rtol=0,atol=0)
        audit=model._last_sentence_credit
        for i,(_,judge) in enumerate(observations):
            torch.testing.assert_close(audit['components'][:,i,2],judge,rtol=0,atol=0)
        bank=tuple(model.comparison_reader.parameters())
        owners=model.objective_parameter_groups(optimizer)
        assert bank and all(any(p is q for q in owners['output']) for p in bank)
        assert not any(p is q for p in bank for q in owners['reconstruction'])
        assert model._sentence_reader_updates==model._sentence_comparison_reader_updates==1
        assert {float(optimizer.state[p]['step']) for p in bank if p in optimizer.state}=={1.}
        assert all(r is not None for r in audit['readers'])
        assert any(not torch.equal(r['presented']['mse'],r['comparison']['mse']) for r in audit['readers'])
        # A later comparison read can never become the ordinary forward head.
        record=model._last_sentence_understanding
        state=(record.root,record.end_slots,record.end_depth)
        before=model._forward_head(None,sentence_state=state,understanding=record).materialize().detach().clone()
        with torch.no_grad():
            for parameter in bank:parameter.add_(3.)
        after=model._forward_head(None,sentence_state=state,understanding=record).materialize().detach()
        torch.testing.assert_close(after,before,rtol=0,atol=0)
    finally:
        model.End();model.symbolSpace.soft_reset()
