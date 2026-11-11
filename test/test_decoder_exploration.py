"""Generate explores one legal departure and compares before training."""
from types import SimpleNamespace, MethodType

import pytest
import torch


def decoder(monkeypatch=None):
    from Language import LanguageSpace,GeneratePolicy
    from Models import BasicModel
    if monkeypatch is not None:
        # This fixture tests the generic departure/replay kernel. Its toy
        # arithmetic has an explicit all-actions eligibility oracle; the real
        # support mask is exercised in test_review11_contracts instead.
        from generation_fixtures import all_available
        import Generative
        monkeypatch.setattr(LanguageSpace,'decoder_eligibility',staticmethod(all_available))
        monkeypatch.setattr(Generative,'selected_residual',lambda lang,parent,*a,**k:parent.new_zeros(len(parent),3))
    policy = GeneratePolicy(2, 3)
    with torch.no_grad():
        policy.weight.zero_()
        policy.bias.copy_(torch.tensor([0., -1., 2.]))
    language = SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace(exploration_fraction=.5, exploration_temperature=1.), _generate_binary_ops=(SimpleNamespace(meaning_layout=None),),
        _generate_unary_ops=(SimpleNamespace(meaning_layout=None),), generate_policy=policy, _generate_policy_width=2,
        _generate_binary_rule_ids=torch.tensor([0]), _generate_binary_names=('toy_split',),
        _generate_unary_rule_ids=torch.tensor([1]), _generate_unary_names=('toy_negate',))
    from generation_fixtures import undeclared_arguments
    undeclared_arguments(language)
    language.generate_policy_logits = MethodType(LanguageSpace.generate_policy_logits, language)
    language.choose_generate = lambda logits: logits.argmax(-1)
    def binary(parent, *args, return_status=False, **kwargs):
        pair = (parent*.25, parent*.75)
        return (*pair, torch.zeros(parent.shape[0], dtype=torch.bool)) if return_status else pair
    def unary(parent, *args, return_status=False, **kwargs):
        return (-parent, torch.zeros(parent.shape[0], dtype=torch.bool)) if return_status else -parent
    language.reverse_binary_step = binary
    language.generate_unary_step = unary
    model = SimpleNamespace(languageSpace=language)
    model._output_generate_walk = MethodType(BasicModel._output_generate_walk, model)
    return model, policy


@pytest.mark.parametrize('compiled', [False, True])
def test_single_departure_replays_prefix_and_is_greedy_after_it(monkeypatch, compiled):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, policy = decoder(monkeypatch)
    # This prefix/suffix test has one eligible departure. Sampling among
    # several alternatives is covered separately by the review §10 probe.
    model.languageSpace.generate_unary_step = lambda parent, *a, return_status=False, **k: (
        (-parent, torch.ones(parent.shape[0], dtype=torch.bool)) if return_status else -parent)
    event = torch.tensor([[[1., 2.], [3., 4.], [0., 0.]],
                          [[1., 2.], [3., 4.], [0., 0.]]], requires_grad=True)
    def pair(event):
        greedy = model._output_generate_walk(event, 8, return_trace=True,
            basis=event, return_candidates=True)
        explore = model._output_generate_walk(event, 8, return_trace=True,
            basis=event, exploit_actions=greedy[4], departure=torch.tensor([0, 1]))
        return greedy, explore
    fn = torch.compile(pair, backend='inductor', fullgraph=True) if compiled else pair
    greedy, explore = fn(event)
    assert greedy[1].tolist() == [2, 2]
    assert explore[1].tolist() == [3, 3]
    assert greedy[4][:, :2].tolist() == [[2, 2], [2, 2]]
    assert explore[4][:, :4].tolist() == [[0, 2, 2, 2], [2, 0, 2, 2]]
    assert greedy[5][:, :2].all()
    assert not greedy[5][:, 2:].any()
    explore[0].square().sum().backward()
    assert policy.bias.grad is None or not policy.bias.grad.any()  # compiled AD may materialize zero
    assert event.grad.norm() > 0


def test_no_legal_departure_keeps_greedy(monkeypatch):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder(monkeypatch)
    model.languageSpace._generate_unary_ops = ()
    model.languageSpace.generate_policy_logits = lambda top,features=None: top.new_tensor([[3., 2.]]).expand(top.shape[0], -1)
    event = torch.tensor([[[1., 2.]]])
    result = model._output_generate_walk(event, 4, return_trace=True, return_candidates=True)
    assert result[1].tolist() == [1]
    assert result[4].tolist() == [[1, -1, -1, -1]]
    assert not result[5].any()


def test_trial_selects_strictly_lower_full_output_cost_before_backward(monkeypatch):
    from WalkTrials import output_pair
    from Output import AnswerConstruction
    parameter = torch.nn.Parameter(torch.tensor(1.))
    greedy_cost = torch.tensor([4., 2., 1.])
    explore_cost = torch.tensor([1., 2., 4.])
    calls = []
    model = SimpleNamespace(_last_output_walk_trace=None)
    def realize(**kwargs):
        assert parameter.item() == 1. and parameter.grad is None
        explore = kwargs.get('exploit_actions') is not None
        replay = kwargs.get('kept_actions')
        actions = torch.full((3, 4), int(explore)) if replay is None else replay
        calls.append(explore if replay is None else 'replay')
        cost = torch.where(actions[:, 0].bool(), explore_cost, greedy_cost) * parameter
        result = AnswerConstruction(actual=cost, derivation=None)
        return result, (actions, torch.ones(3, 4, dtype=torch.bool),
            torch.zeros(3, dtype=torch.bool), torch.zeros(3, 3),
            torch.ones(3, dtype=torch.bool), torch.zeros(3, 4, 2, 9), torch.zeros(3, 4, 2))
    result, audit = output_pair(realize, lambda result: result.actual, owner=model)
    assert calls == [False, True, 'replay']
    assert audit['wins'].tolist() == [True, False, False]
    torch.testing.assert_close(result.actual, torch.tensor([1., 2., 1.]))
    assert model._last_output_walk_trace[0][:, 0].tolist() == [1, 0, 0]
    result.actual.sum().backward()
    assert parameter.grad.item() == 4.


def test_identical_outer_understandings_get_the_same_inner_output_cost(monkeypatch):
    """Paired output walks reuse their departure and action quantiles."""
    from WalkTrials import output_pair, paired_departure
    from Output import AnswerConstruction
    draws = []
    def uniform(*args, **kwargs):
        draws.append(None)
        return torch.tensor([.1, .9] if len(draws) == 1 else [.9, .1],
                            device=kwargs.get('device')).reshape(*args)
    monkeypatch.setattr(torch, 'rand', uniform)
    def realize(**kwargs):
        explore = kwargs.get('exploit_actions') is not None
        cost = (kwargs['departure'].float()+1+kwargs['departure_uniform'][:, 0]
                if explore else torch.full((2,), 10.))
        actions = torch.full((2, 4), int(explore))
        return AnswerConstruction(actual=cost, derivation=None), (
            actions, torch.ones(2, 4, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, 3), torch.ones(2, dtype=torch.bool),
            torch.zeros(2, 4, 2, 9), torch.zeros(2, 4, 2))
    model = SimpleNamespace(_sentence_generation_draws={})
    costs, departures = [], []
    for _ in range(2):
        model._sentence_generation_cursors = {}
        result, audit = output_pair(realize, lambda result: result.actual, owner=model)
        costs.append(result.actual); departures.append(audit['departure'])
    assert len(draws) == 2
    torch.testing.assert_close(costs[0], costs[1], rtol=0, atol=0)
    assert departures[0].tolist() == departures[1].tolist() == [0, 3]
    model._sentence_generation_cursors = {}
    eligible = torch.tensor([[False, True, False, True], [False, True, False, False]])
    assert paired_departure(model, 'output', eligible).tolist() == [1, 1]
    assert len(draws) == 2


def test_paired_output_inverse_reuses_action_quantile_in_the_actual_walk(monkeypatch):
    """The captured mismatch was at the same departure, in its action draw."""
    import Generative
    import Models
    import util
    from WalkTrials import output_pair
    from Output import AnswerConstruction
    from generation_fixtures import exact_menu
    monkeypatch.setattr(util,'TheCompileBackend','none')
    def loop(cond,body,values):
        while bool(cond(*values)):values=body(*values)
        return values
    monkeypatch.setattr(Models,'_reconstruction_while_loop',loop)
    language=SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace(exploration_fraction=.5, exploration_temperature=1.), _generate_binary_ops=(),
        _generate_unary_ops=(SimpleNamespace(meaning_layout=None),)*3,
        _generate_policy_width=2,
        generate_policy_logits=lambda x,features=None:x.new_zeros(len(x),4),
        choose_generate=lambda logits:logits.argmax(-1))
    def menu(_language,parent,live,**kwargs):
        left=parent[:,None].expand(-1,4,-1).clone()
        left[:,:3]+=parent.new_tensor([1.,2.,3.])[None,:,None]
        legal=torch.cat(((parent[:,0]==1)[:,None].expand(-1,3),
                         (parent[:,0]>1)[:,None]),-1)&live[:,None]
        return exact_menu(left,left,legal)
    monkeypatch.setattr(Generative,'inverse_menu',menu)
    draws=[]
    def uniform(*shape,**kwargs):
        draws.append(shape)
        return torch.full(shape,.1 if len(draws)%2 else .9,**kwargs)
    monkeypatch.setattr(torch,'rand',uniform)
    model=SimpleNamespace(languageSpace=language,_sentence_generation_draws={})
    actions=[]
    def realize(**options):
        with torch.no_grad():
            walk=Models.BasicModel._output_generate_walk(model,
                torch.tensor([[[1.,1.],[0.,0.]]]),144,return_trace=True,
                require_symbols=False,**options)
        alternatives=walk[5] if options.get('return_candidates') else torch.zeros_like(walk[4],dtype=torch.bool)
        actions.append(walk[4])
        construction=AnswerConstruction(walk[0][:,:1],None,walk[0],walk[0],None,(),())
        return construction,(walk[4],alternatives,walk[2],walk[3],walk[-5],walk[-2],walk[-1])
    audits=[]
    for _ in range(2):
        model._sentence_generation_cursors={}
        _,audit=output_pair(realize,lambda result:result.actual.sum((1,2)),owner=model)
        audits.append(audit)
    assert len(draws)==2  # one departure and one alternate action, both shared
    assert actions[1][0,:2].tolist()==[2,3]
    torch.testing.assert_close(actions[1],actions[3],rtol=0,atol=0)
    torch.testing.assert_close(audits[0]['costs'],audits[1]['costs'],rtol=0,atol=0)
