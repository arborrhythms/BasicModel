"""Item 2.7 §§38–40: lexical access, residuals and certified generation."""
from types import SimpleNamespace
import pytest
import torch


class Pair(torch.nn.Module):
    meaning_layout=None
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.5))
    def compose(self,left,right):
        return torch.lerp(left,right,self.weight)


def menu_at(parent):
    from Language import LanguageSpace
    op=Pair()
    bank=torch.tensor([[[.2,.8],[.6,.4]]])
    return LanguageSpace.decoder_eligibility(parent,[bank[:,0],parent],
        [bank[:,1],parent],torch.ones(1,2,dtype=torch.bool),(op,),bank,
        torch.ones(1,2,dtype=torch.bool),details=True),op,bank


def test_approximate_native_pair_is_admissible_with_an_explicit_residual():
    menu,_,bank=menu_at(torch.tensor([[.35,.65]]))
    assert menu.admissible[0,0] and not menu.exact[0,0]
    assert menu.form_residual[0,0]>0 and menu.meaning_residual.eq(0).all()
    assert menu.admissible[0,1] and not menu.exact[0,1]
    assert any(torch.equal(menu.left[:,1],bank[:,i]) for i in range(2))


def test_exact_native_pair_keeps_pair_first_and_exact_stop_keeps_activation():
    menu,_,_=menu_at(torch.tensor([[.4,.6]]))
    assert menu.exact[0,0] and menu.admissible.tolist()==[[True,False]]
    stop,_,_=menu_at(torch.tensor([[.1,.4]]))
    assert stop.exact[0,-1] and stop.admissible.tolist()==[[False,True]]
    torch.testing.assert_close(stop.left[0,-1],torch.tensor([.1,.4]),rtol=0,atol=0)


def test_inverse_domain_guard_survives_approximate_eligibility():
    from Language import LanguageSpace
    op=Pair()
    parent=torch.tensor([[.35,.65]])
    bank=torch.tensor([[[1.2,.8],[.6,.4]]])
    menu=LanguageSpace.decoder_eligibility(parent,[bank[:,0],parent],
        [bank[:,1],parent],torch.ones(1,2,dtype=torch.bool),(op,),bank,
        torch.ones(1,2,dtype=torch.bool),details=True)
    assert not menu.admissible[0,0]


def test_residual_charges_each_step_and_unresolved_frontier_without_cancellation():
    from Generative import walk_residual
    initial=torch.tensor([[[.2,.8],[0.,0.]]])
    errors=torch.tensor([[[.1,0.,0.],[.2,0.,0.],[0.,0.,0.]]])
    pending=walk_residual(errors,initial,initial,torch.tensor([1]))
    drained=walk_residual(errors,initial*0,initial,torch.tensor([0]))
    torch.testing.assert_close(pending,torch.tensor([1.3]))
    torch.testing.assert_close(drained,torch.tensor([.3]))


def test_certified_completion_only_has_priority_and_pending_improvement_gets_credit():
    from WalkTrials import finished_wins,generate_credit
    from Language import GeneratePolicy,LanguageSpace
    from types import MethodType
    pending=torch.tensor([[True,False],[True,False],[True,True]])
    certified=torch.tensor([[False,False],[False,True],[False,False]])
    costs=torch.tensor([[.1,.2],[.1,10.],[.8,.3]])
    wins=finished_wins(costs,pending,certified=certified)
    assert wins.tolist()==[False,True,True]
    policy=GeneratePolicy(2,2)
    language=SimpleNamespace(generate_policy=policy,_generate_policy_width=2)
    for name in ('generate_policy_logits','generate_policy_credit'):
        setattr(language,name,MethodType(getattr(LanguageSpace,name),language))
    tops=torch.ones(3,1,2,requires_grad=True)
    features=torch.ones(3,1,2,5,requires_grad=True)
    features.data[:,0,0,0]=.1
    audit=dict(departure=torch.tensor([0,0,0]),wins=wins,explore=torch.zeros(3,1,dtype=torch.long))
    loss=generate_credit(language,audit,tops,features)
    assert loss[0]==0 and (loss[1:]>0).all()
    loss.sum().backward()
    assert tops.grad is None and features.grad is None
    assert policy.weight.grad.abs().sum()>0
    assert policy.menu_score[-1].weight.grad.abs().sum()>0


def test_selected_residual_has_the_native_forward_gradient():
    from Generative import selected_residual,relative_residual
    parent=torch.tensor([[.35,.65]],requires_grad=True)
    left,right=torch.tensor([[.2,.8]]),torch.tensor([[.6,.4]])
    op=Pair(); language=SimpleNamespace(_generate_binary_ops=(op,),_generate_unary_ops=())
    actual=selected_residual(language,parent,left,right,torch.tensor([0]),torch.tensor([True]))
    expected=relative_residual(parent,op.compose(left,right))[0]
    torch.testing.assert_close(actual.sum(-1),expected,rtol=0,atol=0)
    a=torch.autograd.grad(actual.sum(),(parent,op.weight),retain_graph=True)
    b=torch.autograd.grad(expected.sum(),(parent,op.weight))
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_training_answer_form_admission_is_semantically_empty_and_chunked_retrieval_sees_neutral_heat(monkeypatch):
    import util
    from data import TheData
    from Models import BaseModel
    from Corpus import admit_training_forms
    from Generative import lexical_candidates
    monkeypatch.setattr(util,'TheCompileBackend','none')
    TheData.processLM({s:dict(text=['.'] if s=='train' else [],label=[])
                      for s in ('train','validation','test')})
    model=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0]
    try:
        owner=model._concept_owner()
        before=owner._priming_seen.clone()
        admit_training_forms(model,['alpha.'],['omega'])
        word=owner.interpret.word_for_form('omega')
        assert word is not None and not owner.interpret.denotations(word)
        assert not owner.mtm.entries
        assert torch.equal(before,owner._priming_seen)
        row=owner._csw_row_of(word)
        root=owner.similarity_codebook.lookup_rows(torch.tensor([[row]]),streams=True)[:,0]
        bank,report=lexical_candidates(owner,root,limit=2,chunk_size=1)
        assert row in bank.rows[0].tolist() and report['chunks']==report['vocabulary']
        assert bank.weights[bank.valid].eq(1).all()
        with pytest.raises(ValueError,match='training data only'):
            admit_training_forms(model,[],['heldout'],split='test')
    finally:
        model.End()


def test_single_root_lexical_scan_is_reused_across_walk_steps(monkeypatch, tmp_path):
    import json,os
    from pathlib import Path
    from torch.utils._python_dispatch import TorchDispatchMode
    from data import TheData
    from Models import BaseModel
    from Corpus import admit_training_forms
    import Generative,util
    monkeypatch.setattr(util,'TheCompileBackend','none')
    TheData.processLM({s:dict(text=['.'] if s=='train' else [],label=[])
                      for s in ('train','validation','test')})
    shapes=set(); phases={}; phase='construction'
    class Shapes(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            value=func(*args,**(kwargs or {}))
            def see(item):
                if torch.is_tensor(item):
                    shape=tuple(item.shape)
                    shapes.add(shape);phases.setdefault(phase,set()).add(shape)
                elif isinstance(item,(tuple,list)):
                    for child in item:see(child)
            see(value)
            return value
    with Shapes():
        model=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0]
        try:
            owner=model._concept_owner()
            phase='admission'
            admit_training_forms(model,['alpha beta.'],['omega'])
            word=owner.interpret.word_for_form('omega');row=owner._csw_row_of(word)
            root=owner.similarity_codebook.lookup_rows(torch.tensor([[row]]),streams=True)
            calls=[];native=Generative.lexical_candidates
            def counted(*args,**kwargs):
                calls.append(1);return native(*args,**kwargs)
            monkeypatch.setattr(Generative,'lexical_candidates',counted)
            phase='retrieval'
            bank=Generative.walk_bank(model,root,None)
            first=bank.rows.clone()
            model._desired_answer='not a menu input'
            event=torch.cat((root,root.new_zeros(1,7,root.shape[-1])),1)
            outputs=[]
            phase='walk'
            for budget in (2,12):
                outputs.append(model._output_generate_walk(event,budget,basis=bank.codes,
                    basis_valid=bank.valid,terminal_valid=bank.terminal_valid,return_trace=True))
            assert len(calls)==1 and torch.equal(first,bank.rows)
            for output in outputs:
                assert output[1].tolist()==[1] and output[-5].all()
            torch.testing.assert_close(outputs[0][0][:,:1],outputs[1][0][:,:1],rtol=0,atol=0)
            V=owner.nVectors;K=model.reconstruction_basis_limit
            assert not any(shape.count(V)>=2 for shape in shapes)
            # The fixture's placement width happens to equal K=16. That
            # pre-existing [V,16] geometry is not a candidate score table.
            # Audit retrieval and walking separately from construction.
            conflicts=[(name,shape) for name in ('retrieval','walk') for shape in phases[name]
                if len(shape)>=2 and V in shape and K in shape]
            assert not conflicts, conflicts
            coincident=[name for name,value in model.named_buffers() if tuple(value.shape)==(V,K)]
            assert all(name.endswith('.placement') for name in coincident),coincident
            report=dict(inventory=V,candidate_limit=K,lexical_scans=len(calls),budgets=[2,12],
                shapes=sorted([list(shape) for shape in shapes],key=lambda x:(len(x),x)),
                shapes_by_phase={name:sorted([list(shape) for shape in found],key=lambda x:(len(x),x))
                    for name,found in phases.items()},
                existing_geometry_with_coincident_width=coincident,
                no_inventory_square=True,no_dense_candidate_inventory=True,no_steps_by_vocabulary=True,
                retrieval=model._last_lexical_retrieval)
            target=Path(os.environ.get('ITEM27_RECEIPT',str(tmp_path)));target.mkdir(parents=True,exist_ok=True)
            (target/'final-allocation.json').write_text(json.dumps(report,indent=2)+'\n')
        finally:
            model.End();model.symbolSpace.soft_reset()


def test_retired_percept_gain_handoff_has_no_reader():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'bin'
    for name in ('Models.py','ModelAttention.py','ModelCandidateAttention.py'):
        text=(root/name).read_text()
        assert '_percept_word_gain' not in text
        assert '.region_gain(' not in text


def test_training_output_keeps_lexical_text_and_output_gradients(monkeypatch):
    """The same native lexical idea must be reportable with gradients enabled."""
    from Corpus import admit_training_forms
    from Models import BaseModel
    from data import TheData
    from Meaning import ConceptualMeaning
    from Output import AnswerDerivation
    from Understanding import Understanding
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    TheData.processLM({s:dict(text=['.'] if s=='train' else [],label=[])
                      for s in ('train','validation','test')})
    model=BaseModel.from_config('data/MM_corpus_orders.xml',data=TheData)[0]
    try:
        admit_training_forms(model,['alpha'],[])
        owner=model._concept_owner()
        row=owner._csw_row_of(owner.interpret.word_for_form('alpha'))
        word=owner.similarity_codebook.lookup_rows(torch.tensor([[row]]),streams=True)[:,0]
        meaning=ConceptualMeaning.from_description(word[0].detach())
        held=AnswerDerivation(answer_symbol=None,conceptual_answer=meaning.roles[None],
            sentence_states=(SimpleNamespace(meaning=meaning),))
        understood=Understanding()
        model.eval()
        evaluated=model.reverseOutput(understood,held)
        model.train()
        trained=model.reverseOutput(understood,held)
        assert trained.texts == evaluated.texts == ('alpha',)
        torch.testing.assert_close(trained.concepts,evaluated.concepts,rtol=0,atol=0)
        torch.testing.assert_close(trained.actual,evaluated.actual,rtol=0,atol=0)
        parameters=tuple(model.outputSpace.percept_adapter.parameters())
        expected=torch.autograd.grad(evaluated.actual.sum(),parameters)
        actual=torch.autograd.grad(trained.actual.sum(),parameters)
        assert any(bool(g.abs().any()) for g in actual)
        for first,second in zip(expected,actual,strict=True):
            torch.testing.assert_close(first,second,rtol=0,atol=0)
    finally:
        model.End();model.symbolSpace.soft_reset()
