"""Unsigned form interpretation and one answer-owner update per sentence."""
from types import SimpleNamespace
import pytest
import torch


def test_interpret_both_directions_preserve_form_presence_and_both_lanes():
    from Interpret import InterpretLayer
    owner=SimpleNamespace(similarity_codebook=SimpleNamespace(mereology=SimpleNamespace(percept_event_width=2)))
    layer=InterpretLayer(conceptualSpace=owner)
    atoms=torch.tensor([[.8,.4,.6,0.]],requires_grad=True)
    presence=torch.tensor([.5],requires_grad=True)
    evidence=torch.tensor([[.2,.5]])
    event=torch.tensor([[0.,0.,0.,0.,7.,9.]])
    expected=torch.tensor([[.4,.2,.12,.3,7.,9.]])
    torch.testing.assert_close(layer(event,object_atoms=atoms,presence=presence,evidence=evidence),expected)
    torch.testing.assert_close(layer.reverse(event,word_atoms=atoms,presence=presence,evidence=evidence),expected)
    owner.similarity_codebook.mereology.percept_event_width=4
    torch.testing.assert_close(layer(event,object_atoms=atoms,presence=presence,evidence=evidence),
                              layer(event,object_atoms=atoms,presence=presence,evidence=evidence.flip(-1)),rtol=0,atol=0)


@pytest.mark.parametrize('config',['XOR_grammar','MM_grammar'])
@pytest.mark.parametrize('walk',['narrowing','compose'])
def test_zero_meaning_departures_and_one_reader_step(config,walk,monkeypatch):
    from test_mm_xor import _fresh_model
    from Language import OperationSelectionLayer
    import SentenceCredit
    model,_,data=_fresh_model('data/'+config+'.xml')
    original_attend=OperationSelectionLayer.attend
    original_forward=OperationSelectionLayer.forward
    original_departure=SentenceCredit.departure
    def depart(narrowing,compose,**kwargs):
        draw=original_departure(narrowing,compose,**kwargs)
        if walk=='narrowing':
            draw.update(round=torch.zeros_like(draw['round']),narrowing=torch.ones_like(draw['narrowing']),
                        attention_round=torch.zeros_like(draw['round']),compose_round=torch.full_like(draw['round'],-1))
        else:
            chosen=kwargs['compose_round']
            draw.update(round=chosen+narrowing.alternatives.shape[1],narrowing=torch.zeros_like(draw['narrowing']),
                        attention_round=torch.full_like(chosen,-1),compose_round=chosen)
        draw['walk']=torch.full_like(draw['round'],0 if walk=='narrowing' else 1)
        draw['walk_rounds']=(narrowing.alternatives if walk=='narrowing' else compose).sum(-1)
        return draw
    def attend(module,keys,legal,space,**kwargs):
        available=legal.flatten(1)
        action=torch.full((len(keys),),available.shape[1],dtype=torch.long)
        for op in (0,2,1):
            slots=legal[:,:,op]
            action=torch.where(slots.any(-1),slots.long().argmax(-1)*6+op,action)
        mask=kwargs.get('masked_action')
        if mask is not None:
            action=torch.where(mask>=0,torch.full_like(action,5),action)
        kwargs['replay_action']=action
        return original_attend(module,keys,legal,space,**kwargs)
    def disjunction(module,x,**kwargs):
        refs=kwargs.get('reference_data')
        binary=tuple(range(module.r_reduce)) if refs is None else refs['binary_ops']
        unary=tuple(range(module.r_apply)) if refs is None else refs['unary_ops']
        stop=(x.shape[1]-1)*len(binary)+x.shape[1]*len(unary)
        depth=kwargs.get('depth',torch.full((len(x),),x.shape[1]))
        op=torch.full_like(depth,binary.index(1))
        if walk=='compose' and kwargs.get('masked_action') is not None:
            op=torch.where(kwargs['masked_action']>=0,binary.index(0),op)
        from operation_fixtures import selected_action
        with selected_action(module, torch.where(depth>1,op,stop)):
            return original_forward(module,x,**kwargs)
    monkeypatch.setattr(SentenceCredit,'departure',depart)
    monkeypatch.setattr(OperationSelectionLayer,'attend',attend)
    monkeypatch.setattr(OperationSelectionLayer,'forward',disjunction)
    snapshots=[]
    original=model._sentence_path_cost
    def cost(state,*args,**kwargs):
        result=original(state,*args,**kwargs)
        snapshots.append(dict(root=state[1][9].detach().clone(),
            activation=state[1][0].detach().clone(), meanings=result[2]['meanings'], entries=result[2]['entries']))
        return result
    monkeypatch.setattr(model,'_sentence_path_cost',cost)
    try:
        optimizer=model.getOptimizer(lr=.01)
        model.runEpoch(optimizer=optimizer,batchSize=4,split='train')
        assert len(snapshots)==2
        first,second=snapshots
        audit=model._last_sentence_credit
        if walk=='narrowing':
            torch.testing.assert_close(first['activation'],second['activation'],rtol=0,atol=0)
            torch.testing.assert_close(first['root'],second['root'],rtol=0,atol=0)
            assert all(a.polarity and b.polarity for a,b in zip(first['meanings'],second['meanings']))
            for a,b in zip(first['entries'],second['entries']):
                torch.testing.assert_close(a.leaf_evidence, b.leaf_evidence.flip(-1))
            torch.testing.assert_close(audit['components'][:,0],audit['components'][:,1],rtol=0,atol=0)
            assert audit['advantage'].eq(0).all() and not audit['wins'].any()
            assert model._sentence_reader_weights.tolist()==[[1.,0.]]*4
        else:
            torch.testing.assert_close(first['activation'],second['activation'],rtol=0,atol=0)
            assert not torch.equal(first['root'],second['root'])
            assert model._sentence_comparison_reader_weights.tolist()==[[.5,.5]]*4
        kept=torch.stack((~audit['wins'],audit['wins']),-1).to(model._sentence_reader_weights)
        torch.testing.assert_close(model._sentence_reader_weights,kept)
        assert model._sentence_reader_updates==1
        assert model._sentence_comparison_reader_updates==1
        states=[optimizer.state[p]['step'] for p in model.objective_parameter_groups(optimizer)['output']
                if p in optimizer.state and 'step' in optimizer.state[p]]
        assert states and all(float(s)==1 for s in states)
    finally:
        model.End();model.symbolSpace.soft_reset()
