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
    assert audit['wins'].tolist()==[True,False,False]
    assert audit['keep_decision']==['explore','tie:greedy','greedy']
    assert audit['advantage_sign'].tolist()==[-1.,0.,1.]
    assert audit['policy_against_keep'].tolist()==[False,False,False]
    assert audit['answer_against_keep'].tolist()==[False,False,False]
    assert trial([1.,1.,1.],[0.,0.,0.])[:,2].eq(0).all()




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
        return torch.tensor([2])
    monkeypatch.setattr(WalkTrials,'departure_at',choose)
    draw=departure(attention,torch.tensor([[False,True,True,True]]),
                   active=torch.tensor([True]), compose_round=torch.tensor([1]))
    assert seen[0].tolist()==[[True,True]]
    assert seen[1].tolist()==[[True,False,True]]
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


@pytest.mark.parametrize('gain',[0.,.5,1.])
@pytest.mark.parametrize('presence,mask',[(1.,0.),(0.,0.),(1.,1.)])
def test_closing_gain_whole_identity_and_exact_restoration(gain,presence,mask):
    from Meaning import ClosingImage
    raw=torch.tensor([[8.,2.,4.]]).expand(3,-1).clone().requires_grad_()
    from meaning_fixtures import symbol_prediction
    prior=symbol_prediction(raw.detach()[:1],torch.tensor([[1.,0.]]))
    image=ClosingImage.form(raw,raw,torch.full((len(prior.symbol_keys),),presence), prediction=prior,
        object_mask=torch.full((3,),mask),gain=gain)
    expected=1-gain*presence*(1-mask)
    torch.testing.assert_close(image.conceived,raw.detach()*expected)
    torch.testing.assert_close(image.restore(),raw,rtol=0,atol=0)
    assert not image.conceived.requires_grad


def test_gain_has_no_coordinatewise_complement_exception():
    from Meaning import ClosingImage
    raw=torch.arange(18.).reshape(3,6)
    for gain in (0.,.5,1.):
        from meaning_fixtures import symbol_prediction
        prior=symbol_prediction(raw,torch.tensor([[1.,0.]]).expand(3,2))
        image=ClosingImage.form(raw,raw,torch.ones(3),prediction=prior,gain=gain)
        assert image.concept_width==6
        torch.testing.assert_close(image.conceived,raw*(1-gain))
        torch.testing.assert_close(image.restore(),raw)


@pytest.mark.parametrize('config',['XOR_grammar','MM_grammar'])
def test_gate_grammars_retain_native_identity_at_zero_content(config):
    from test_mm_xor import _fresh_model
    from Meaning import ClosingImage
    model,_,_=_fresh_model('data/'+config+'.xml')
    try:
        derived=model._concept_owner().similarity_codebook.mereology
        raw=torch.ones(3,derived.code_width)
        from meaning_fixtures import symbol_prediction
        prior=symbol_prediction(raw[:1],torch.tensor([[1.,0.]]))
        image=ClosingImage.form(raw,raw,prior.confidence,prediction=prior,gain=1.)
        assert image.concept_width==derived.code_width
        assert not image.conceived.any()
        torch.testing.assert_close(image.restore(),raw)
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
        torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.bool),None,None,torch.tensor([[1.,0.]]))
    reading=SimpleNamespace(accepted=torch.ones(1,1,dtype=torch.bool),
                            pole_changes=torch.ones(1,1,dtype=torch.bool))
    model=SimpleNamespace(_attention_words=reading,_attention_poles=torch.tensor([[[1.,0.]]]))
    greedy=BasicModel._attention_sentence_payload(model,payload,torch.tensor(0))
    model._attention_poles=model._attention_poles.flip(-1)
    explore=BasicModel._attention_sentence_payload(model,payload,torch.tensor(0))
    torch.testing.assert_close(explore[0],greedy[0],rtol=0,atol=0)
    assert explore[3].item()==greedy[3].item()==1
    torch.testing.assert_close(explore[-1], greedy[-1].flip(-1))
    positive=finish_reading(language,replace(leaf,leaf_evidence=torch.tensor([[1.,0.]])),registry=registry)
    negative=finish_reading(language,replace(leaf,leaf_evidence=torch.tensor([[0.,1.]])),registry=registry)
    assert positive.meaning.polarity and negative.meaning.polarity
    assert positive.evidence == (1.,0.) and negative.evidence == (0.,1.)
    from Interpret import InterpretLayer
    owner = SimpleNamespace(similarity_codebook=SimpleNamespace(mereology=SimpleNamespace(percept_event_width=1)))
    interpret = InterpretLayer(conceptualSpace=owner)
    atoms = torch.tensor([[.8, .6, .2]])
    positive_code = interpret.activate(atoms, greedy[3].reshape(-1), evidence=greedy[-1])
    negative_code = interpret.activate(atoms, explore[3].reshape(-1), evidence=explore[-1])
    torch.testing.assert_close(positive_code[:, :1], negative_code[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(positive_code[:, 1:], negative_code[:, 1:].flip(-1), rtol=0, atol=0)
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
