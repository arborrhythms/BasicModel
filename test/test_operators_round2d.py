"""Walk-first proposals and their inverse-probability correction."""
from types import SimpleNamespace
import torch


def test_walk_probability_does_not_depend_on_its_number_of_rounds(monkeypatch):
    from SentenceCredit import departure
    count=60
    # Midpoints cover each proposal interval exactly, without a seed or a
    # statistical tolerance: two narrowing rounds versus three compose rounds.
    draws=iter(((torch.arange(count)+.5)/count,
                (torch.arange(count).remainder(30)+.5)/30))
    monkeypatch.setattr(torch,'rand',lambda *args,**kwargs:next(draws))
    reading=SimpleNamespace(alternatives=torch.tensor([[True,False,True]]).expand(count,-1))
    draw=departure(reading,torch.tensor([[True,True,True]]).expand(count,-1),
                   active=torch.ones(count,dtype=torch.bool),
                   compose_round=torch.arange(count).remainder(3))
    assert torch.bincount(draw['walk']).tolist()==[30,30]
    assert torch.bincount(draw['attention_round'][draw['narrowing']],minlength=3).tolist()==[15,0,15]
    assert torch.bincount(draw['compose_round'][~draw['narrowing']],minlength=3).tolist()==[10,10,10]
    assert draw['walk_count'].eq(2).all()
    assert draw['walk_rounds'][draw['narrowing']].eq(2).all()
    assert draw['walk_rounds'][~draw['narrowing']].eq(3).all()


def test_eligible_walks_respect_sentence_and_active_rows(monkeypatch):
    from SentenceCredit import departure
    monkeypatch.setattr(torch,'rand',lambda n,**kwargs:torch.full((n,),.75))
    reading=SimpleNamespace(alternatives=torch.tensor([
        [True,True,True], [False,False,False], [False,True,True], [True,True,True]]),
        round_words=torch.tensor([[0,1,-1]]).expand(4,-1))
    draw=departure(reading,torch.tensor([[False,False],[True,True],[False,False],[True,True]]),
        active=torch.tensor([True,True,True,False]),
        sentence_ids=torch.tensor([[0,1]]).expand(4,-1),sentence=0,
        compose_round=torch.tensor([-1,1,-1,1]))
    assert draw['walk'].tolist()==[0,1,-1,-1]
    assert draw['walk_count'].tolist()==[1,1,0,0]
    assert draw['walk_rounds'].tolist()==[1,2,0,0]
    assert draw['attention_round'].tolist()==[0,-1,-1,-1]
    assert draw['compose_round'].tolist()==[-1,1,-1,-1]
    assert draw['rounds'].tolist()==[1,2,0,0]


def test_compose_only_and_no_eligible_rounds(monkeypatch):
    from SentenceCredit import departure
    monkeypatch.setattr(torch,'rand',lambda n,**kwargs:torch.full((n,),.75))
    draw=departure(None,torch.tensor([[True,False,True],[False,False,False]]),
                   active=torch.tensor([True,True]), compose_round=torch.tensor([2,-1]))
    assert draw['walk'].tolist()==[1,-1]
    assert draw['walk_count'].tolist()==[1,0]
    assert draw['walk_rounds'].tolist()==[2,0]
    assert draw['compose_round'].tolist()==[2,-1]
    empty=departure(None,torch.zeros(2,0,dtype=torch.bool),active=torch.tensor([True,False]))
    assert empty['round'].tolist()==[-1,-1]
    assert empty['walk_count'].eq(0).all() and empty['walk_rounds'].eq(0).all()


def test_enumerated_proposals_equal_the_full_score_function_gradient():
    from SentenceCredit import score_function
    logits=torch.tensor([[.3,-.5,.7],[.2,.8,-.4],[-.6,.1,.9]],dtype=torch.float64,requires_grad=True)
    # Two rounds in narrowing; one in compose. The alternatives have K=1,2,2.
    alternatives=((1,), (0,2), (0,1))
    round_counts=(2,2,1)
    costs=logits.new_tensor([[4.,1.,3.],[2.,4.,.5],[.8,2.5,4.]])
    baseline=4.
    expected=logits.new_zeros(())
    estimated=logits.new_zeros(())
    probabilities=logits.softmax(-1)
    for row,(eligible,rounds) in enumerate(zip(alternatives,round_counts,strict=True)):
        for action in eligible:
            probability=probabilities[row,action].reshape(1,1)
            expected=expected+probability.squeeze()*(costs[row,action]-baseline)
            scale=2*rounds*len(eligible)
            trial=score_function(probability,torch.tensor([[scale]]),torch.tensor([[True]]),
                torch.stack((logits.new_tensor(baseline),costs[row,action])).reshape(1,2))[0].sum()
            estimated=estimated+trial/scale
    torch.testing.assert_close(estimated,expected,rtol=0,atol=1e-15)
    actual,=torch.autograd.grad(estimated,logits,retain_graph=True)
    analytic,=torch.autograd.grad(expected,logits)
    torch.testing.assert_close(actual,analytic,rtol=0,atol=1e-15)
