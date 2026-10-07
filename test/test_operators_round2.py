"""Owner-step credit, detached evidence, and the closing's concept image."""
from types import SimpleNamespace
import pytest
import torch


def test_reconstruction_keeps_total_credits_and_supplied_answer_only():
    from Layers import Error
    from SentenceCredit import components, comparison
    active=torch.tensor([True,True,True])
    def trial(r,e,a=None):
        registry=Error(row_mask=active)
        registry.error('bytes',torch.tensor(r),1.,category='reconstruction')
        if a is not None:
            registry.error('answer',torch.tensor(a),1.,category='output')
        return components(registry,torch.tensor(e),torch.zeros(3),answer=registry.total(objective='output'))
    greedy=trial([1.,1.,1.],[0.,0.,0.],[1.,1.,1.])
    explore=trial([1.,2.,.5],[0.,0.,0.],[0.,0.,3.])
    audit=comparison(torch.stack((greedy,explore),1),active)
    assert audit['wins'].tolist()==[False,False,True]
    assert audit['keep_decision']==['tie:greedy','greedy','explore']
    assert audit['advantage_sign'].tolist()==[-1.,0.,1.]
    assert audit['policy_against_keep'].tolist()==[True,False,True]
    assert audit['answer_against_keep'].tolist()==[True,False,True]
    assert trial([1.,1.,1.],[0.,0.,0.])[:,2].eq(0).all()


def test_expectation_gate_does_not_cancel_against_its_baseline():
    from Layers import Error
    from SentenceCredit import expectation_terms
    pred=torch.ones(3,2,requires_grad=True)
    logits=torch.tensor([0.,0.,0.],requires_grad=True)
    target=torch.full((3,2),2.,requires_grad=True)
    values=[]
    for gain in (0.,.5,1.):
        registry=Error(row_mask=torch.tensor([True]))
        expectation_terms(registry,pred,logits,torch.tensor(0.,requires_grad=True),
            target,torch.ones(3),'idea',row=0,gain=gain)
        values.append(registry.total())
    assert values[0].item()==0
    torch.testing.assert_close(values[1],values[2]/2)
    values[2].sum().backward()
    assert target.grad is None
    assert pred.grad is not None and logits.grad is not None


@pytest.mark.parametrize('walk',['narrowing','compose'])
def test_walk_first_departure_counts_and_gradient(walk,monkeypatch):
    import WalkTrials
    from SentenceCredit import departure,score_function
    attention=SimpleNamespace(alternatives=torch.tensor([[True,False,True]]),
        round_words=torch.tensor([[0,0,1]]))
    seen=[]
    def choose(mask):
        seen.append(mask)
        if len(seen)==1:return torch.tensor([0 if walk=='narrowing' else 1])
        return torch.tensor([2 if walk=='narrowing' else 4])
    monkeypatch.setattr(WalkTrials,'departure_at',choose)
    draw=departure(attention,torch.tensor([[False,True,True,True]]),active=torch.tensor([True]))
    assert seen[0].tolist()==[[True,True]]
    assert seen[1].tolist()==([[True,False,True,False,False,False,False]] if walk=='narrowing'
                             else [[False,False,False,False,True,True,True]])
    assert draw['rounds'].item()==5
    assert draw['walk_count'].item()==2
    assert draw['walk_rounds'].item()==(2 if walk=='narrowing' else 3)
    assert draw['walk'].item()==(0 if walk=='narrowing' else 1)
    assert draw['narrowing'].item()==(walk=='narrowing')
    assert (draw['attention_round']>=0).sum()+(draw['compose_round']>=0).sum()==1
    logits=torch.tensor([[.2,.8,-.3]],dtype=torch.float64,requires_grad=True)
    def value(x):
        return score_function(x.softmax(-1)[:,1:2],
            2*(draw['walk_count']*draw['walk_rounds'])[:,None],
            torch.tensor([[True]]),torch.tensor([[3.,1.]]))[0].sum()
    gradient,=torch.autograd.grad(value(logits),logits)
    probability=logits.detach().softmax(-1)
    expected=-probability*probability[:,1:2]
    expected[:,1]+=probability[:,1]
    expected*=2*draw['walk_count']*draw['walk_rounds']*(-2)
    torch.testing.assert_close(gradient,expected)
    for column in range(3):
        plus,minus=logits.detach().clone(),logits.detach().clone()
        plus[0,column]+=1e-5;minus[0,column]-=1e-5
        torch.testing.assert_close(gradient[0,column],(value(plus)-value(minus))/2e-5)


def field_read(op):
    from Attention import narrow_words
    calls=0
    def choose(keys,legal,space,**kwargs):
        nonlocal calls
        selected=torch.tensor([op]) if calls==0 else legal.flatten(1).long().argmax(-1)
        calls+=1
        return selected,keys.new_ones(1),legal.flatten(1).sum(-1)>1,dict(
            probability=keys.new_ones(1),alternative_count=(legal.flatten(1).sum(-1)-1).clamp_min(0))
    return narrow_words(SimpleNamespace(attention_operations=tuple(range(6)),attend=choose),
        torch.eye(2)[None],torch.tensor([[[0,1],[2,3]]]),torch.ones(1,2,dtype=torch.bool),
        poles=torch.tensor([[[1.,0.],[0.,1.]]]),budget=8)


@pytest.mark.parametrize('op,pair',[(3,[0.,1.]),(4,[1.,0.]),(5,[1.,1.])])
def test_field_pair_survives_bracket_split(op,pair):
    reading=field_read(op)
    assert reading.pole_changes.all()
    torch.testing.assert_close(reading.poles,torch.tensor([[pair,pair]]))
    # Field connectives do not hand off their transformed key values.
    torch.testing.assert_close(reading.values,torch.eye(2)[None])


@pytest.mark.parametrize('op,pair',[(3,[1.,0.]),(4,[1.,0.]),(5,[0.,1.])])
def test_field_scope_resolves_witnesses_acquired_by_later_descent(op,pair):
    from Attention import narrow_words
    calls=0
    def choose(keys,legal,space,**kwargs):
        nonlocal calls
        selected=torch.tensor([op]) if calls==0 else legal.flatten(1).long().argmax(-1)
        calls+=1
        return selected,keys.new_ones(1),legal.flatten(1).sum(-1)>1,dict(
            probability=keys.new_ones(1),alternative_count=(legal.flatten(1).sum(-1)-1).clamp_min(0))
    reading=narrow_words(SimpleNamespace(attention_operations=tuple(range(6)),attend=choose),
        torch.eye(2)[None],torch.tensor([[[0,1],[2,3]]]),torch.zeros(1,2,dtype=torch.bool),
        poles=torch.zeros(1,2,2),budget=8)
    assert reading.descended.all() and reading.accepted.all()
    torch.testing.assert_close(reading.poles,torch.tensor([[pair,pair]]))


@pytest.mark.parametrize('observed,estimate,presence,mask,expected',[
    (2.,2.,1.,0.,0.),(3.,2.,1.,0.,1.),(2.,2.,0.,0.,2.),
    (0.,2.,1.,0.,-2.),(0.,2.,0.,0.,0.),(2.,2.,1.,1.,2.)])
def test_closing_image_six_rows_and_exact_restoration(observed,estimate,presence,mask,expected):
    from Meaning import ClosingImage
    o=torch.tensor([[8.,observed,4.]]).expand(3,-1).clone().requires_grad_()
    e=torch.tensor([[16.,estimate,8.]]).expand(3,-1).clone().requires_grad_()
    image=ClosingImage.form(o,e,torch.full((3,),presence),object_mask=torch.full((3,),mask),
        form_width=1,content_width=2)
    torch.testing.assert_close(image.conceived[:,1],torch.full((3,),expected))
    torch.testing.assert_close(image.conceived[:,[0,2]],o[:,[0,2]],rtol=0,atol=0)
    torch.testing.assert_close(image.restore(),o,rtol=0,atol=0)
    assert not image.conceived.requires_grad and not image.image.requires_grad


def test_zero_complement_is_identically_zero_at_every_gain():
    from Meaning import ClosingImage
    o=torch.arange(18.).reshape(3,6)
    for gain in (0.,.5,1.):
        image=ClosingImage.form(o,o+1,torch.ones(3),form_width=4,content_width=4,gain=gain)
        assert image.concept_width==0 and image.image.eq(0).all()
        torch.testing.assert_close(image.conceived,o,rtol=0,atol=0)


@pytest.mark.parametrize('config',['XOR_grammar','MM_grammar'])
def test_gate_grammars_have_no_image_complement(config):
    from test_mm_xor import _fresh_model
    from Meaning import ClosingImage
    model,_,_=_fresh_model('data/'+config+'.xml')
    try:
        derived=model._concept_owner().similarity_codebook.mereology
        assert derived.percept_width==96 and derived.code_width==104
        assert derived.context_width==0
        observed=torch.ones(3,derived.code_width)
        image=ClosingImage.form(observed,observed,torch.ones(3),
            form_width=model._image_form_width)
        assert image.concept_width==0 and image.image.eq(0).all()
    finally:
        model.End();model.symbolSpace.soft_reset()


def test_not_handoff_changes_pole_and_meaning_but_never_form(monkeypatch):
    from dataclasses import replace
    from Models import BasicModel
    from test_selected_nested_meaning import _nested
    from reading_fixtures import finish_reading
    _,registry,language,entry,_=_nested(monkeypatch)
    code=entry.leaves[:1]
    leaf=replace(entry,rows=entry.rows[:1],word_rows=entry.word_rows[:1],
        activations=torch.ones(1),leaves=code,concept_ids=torch.tensor([1]),
        lexical_forms=(None,),actions=torch.tensor([[0,-1,0]]),targets=torch.tensor([-1]),
        end_state=torch.cat((code,torch.zeros_like(entry.end_state[1:]))))
    event=code[None].requires_grad_()
    payload=(event,torch.zeros(1,1),torch.zeros(1,dtype=torch.long),torch.ones(1,1),
        torch.zeros(1,dtype=torch.long),torch.zeros(1,dtype=torch.long),code,
        torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool),None,None)
    reading=SimpleNamespace(accepted=torch.ones(1,1,dtype=torch.bool),
                            pole_changes=torch.ones(1,1,dtype=torch.bool))
    model=SimpleNamespace(_attention_words=reading,_attention_poles=torch.tensor([[[1.,0.]]]))
    greedy=BasicModel._attention_sentence_payload(model,payload,torch.tensor(0))
    model._attention_poles=model._attention_poles.flip(-1)
    explore=BasicModel._attention_sentence_payload(model,payload,torch.tensor(0))
    torch.testing.assert_close(explore[0],greedy[0],rtol=0,atol=0)
    assert explore[3].item()==-1 and greedy[3].item()==1
    positive=finish_reading(language,replace(leaf,leaf_evidence=torch.tensor([[1.,0.]])),registry=registry)
    negative=finish_reading(language,replace(leaf,leaf_evidence=torch.tensor([[0.,1.]])),registry=registry)
    assert positive.meaning.polarity and not negative.meaning.polarity
    from Interpret import InterpretLayer
    owner = SimpleNamespace(similarity_codebook=SimpleNamespace(mereology=SimpleNamespace(percept_event_width=1)))
    interpret = InterpretLayer(conceptualSpace=owner)
    atoms = torch.tensor([[.8, .6]])
    positive_code = interpret.activate(atoms, greedy[3].reshape(-1))
    negative_code = interpret.activate(atoms, explore[3].reshape(-1))
    torch.testing.assert_close(positive_code[:, :1], negative_code[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(positive_code[:, 1:], -negative_code[:, 1:], rtol=0, atol=0)
    assert (negative_code-positive_code).square().sum() > 0


def test_presented_answer_uses_committed_root_and_keeps_reader_cut():
    from Models import BasicModel
    from SentenceCompose import sentence_pair
    from test_supplied_answer_training import _sentence
    model,state,_=_sentence(True)
    root=state[1][9][:,0]
    greedy=(root,torch.cat((root[:,None],torch.zeros(2,2,2)),1),torch.ones(2,dtype=torch.long))
    explore=(root+7,torch.cat(((root+7)[:,None],torch.zeros(2,2,2)),1),greedy[2])
    active=torch.ones(2,dtype=torch.bool)
    def score(path,alternative):
        return torch.tensor([0.,1.] if alternative else [1.,0.]),path
    kept,_,wins=sentence_pair(None,lambda _,prior:greedy if prior is None else explore,
        score,lambda _:None,active=active)
    answer=BasicModel._forward_head(model,None,sentence_state=kept).materialize()
    expected=torch.stack((explore[0][0].sum(),greedy[0][1].sum()))[:,None]
    torch.testing.assert_close(answer,expected)
    assert wins.tolist()==[True,False]
    assert torch.autograd.grad(answer.sum(),root,allow_unused=True)[0] is None
