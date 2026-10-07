"""The owner costs complete alternative walks before either can train."""
import torch


def test_output_pair_costs_complete_paths_and_keeps_greedy_ties():
    from WalkTrials import output_pair
    from Output import AnswerConstruction
    parameter=torch.nn.Parameter(torch.tensor(1.))
    calls=[]
    def realize(**options):
        assert parameter.grad is None and parameter.item() == 1.
        explore=options.get('exploit_actions') is not None
        calls.append(('realize',explore))
        costs=torch.tensor([1.,2.,4.] if explore else [4.,2.,1.])*parameter
        trace=(torch.full((3,4),int(explore)),torch.ones(3,4,dtype=torch.bool))
        result=AnswerConstruction(actual=costs[:,None],derivation=None,
            concepts=torch.full((3,2,2),float(explore))*parameter,
            percepts=costs[:,None,None],surface=None,trace=(),texts=())
        return result,trace
    def cost(construction):
        calls.append(('cost',int(construction.concepts[0,0,0].detach())))
        return construction.actual[:,0]
    result,audit=output_pair(realize,cost)
    assert calls == [('realize',False),('cost',0),('realize',True),('cost',1)]
    assert audit['wins'].tolist() == [True,False,False]
    assert audit['stable'].tolist() == [False,False,False]
    assert result.concepts[:,0,0].tolist() == [1.,0.,0.]
    result.actual.sum().backward()
    assert parameter.grad.item() == 4.


def test_output_without_a_legal_alternative_never_switches():
    from WalkTrials import output_pair
    from Output import AnswerConstruction
    def realize(**options):
        x=torch.ones(2,1)
        return AnswerConstruction(x,None,x,x,None,(),()), (torch.zeros(2,3,dtype=torch.long),torch.zeros(2,3,dtype=torch.bool))
    value,audit=output_pair(realize,lambda result:result.actual[:,0])
    assert not audit['wins'].any()
    assert audit['departure'].tolist() == [-1,-1]
    assert audit['stable'].all()


def test_thought_pair_keeps_one_episode_and_costs_before_policy_backward():
    from test_normal_thought_controller import _catalog_world
    model,registry,memory,part,whole=_catalog_world()
    model.train()
    model.selected_thought_policy_weight=1.
    question=registry.form('part',part,whole)
    calls=[]
    def score(result):
        assert all(p.grad is None for p in model.parameters())
        calls.append(tuple(record.operation for record in result.records))
        return torch.tensor(float(len(calls)==1)*10.)
    with model._query_boundary_scope((0,)):
        result=model.run_selected_thought(question,work_budget=16,score=score)
    assert len(calls)==2
    audit=model._last_thought_comparison
    assert audit['explore_kept']
    assert audit['costs'][1]<audit['costs'][0]
    assert tuple(memory.thought_history())==result.records
    assert len([record for record in result.records if record.kind=='begin'])==1
    assert result.work.spent==memory.thought_state().work_spent
    assert all(record[2]==result.records[0].episode for record in model._selected_thought_policy_records)


def test_thought_pair_tie_keeps_greedy_state_and_shared_meter():
    from test_normal_thought_controller import _catalog_world
    from QueryWork import QueryWorkBudget
    model,registry,memory,part,whole=_catalog_world()
    model.train()
    model.selected_thought_policy_weight=1.
    meter=QueryWorkBudget(20)
    meter.require('bracket',4)
    def score(result):
        # Cancel the work term to test an exact tie in the owner's return.
        return torch.tensor(1.-model.WHAT_STEP_COST*result.work.spent,dtype=torch.float64)
    with model._query_boundary_scope((0,)):
        result=model.run_selected_thought(registry.form('part',part,whole),work=meter,score=score)
    assert not model._last_thought_comparison['explore_kept']
    assert result.work is meter
    assert meter.spent==4+memory.thought_state().work_spent
    assert meter.counts['bracket']==4


def test_walk_audit_records_strict_wins_and_sentence_stability():
    from types import SimpleNamespace
    from WalkTrials import observe_comparison
    model=SimpleNamespace(_attention_forms=([['a','b']],None,None,None))
    audit=dict(costs=torch.tensor([[2.,1.]]),wins=torch.tensor([True]),
        departure=torch.tensor([0]),greedy=torch.tensor([[0,1]]),explore=torch.tensor([[2,1]]))
    for kind in ('compose','generate.decoder','generate.output','think'):
        observe_comparison(model,kind,audit,sentence=0)
        observe_comparison(model,kind,audit,sentence=0)
        row=model._walk_audit[kind]
        assert row['walks']==2 and row['explore_wins']==2 and row['strict_violations']==0
        assert row['stability_pairs']==1 and row['stable_pairs']==1
        assert row['explore_fraction']==1.


def test_anticipation_holds_both_walks_until_future_return():
    from test_negative_expectation import _anticipating_model, observe
    model, owner, meaning = _anticipating_model()
    model.train()
    before = tuple(model._what_memory().thought_history())
    model._stage_expectation_queries(training=True)
    pending = owner._inter_last_meaning[0]
    assert pending.walk is not None
    assert tuple(model._what_memory().thought_history()) == before
    assert pending.walk.other.versions == pending.versions
    assert not pending.prediction.roles.requires_grad
    assert not pending.walk.other.prediction.roles.requires_grad
    observe(owner, meaning.roles)
    model._expectation_policy_loss()
    assert model._walk_audit['think.anticipation']['walks'] == 1
    assert model._walk_audit['think.anticipation']['strict_violations'] == 0
    assert tuple(model._what_memory().thought_history()) == before


def test_delayed_forecast_costs_both_held_estimates_and_keeps_strict_ties():
    from dataclasses import replace
    from Layers import MeaningExpectation, _PendingMeaningExpectation
    from WalkTrials import ForecastWalk, select_forecast
    a = _PendingMeaningExpectation(MeaningExpectation(torch.zeros(3, 4), torch.zeros(3)),
        (), 'stream', 'document', versions=(1,), work=2)
    b = replace(a, prediction=MeaningExpectation(torch.ones(3, 4), torch.zeros(3)))
    a = replace(a, walk=ForecastWalk(b, (0, 1), (2, 1), 0, (('greedy',), ('explore',)), .01))
    kept = select_forecast(a, torch.ones(3, 4), torch.ones(3, dtype=torch.bool), None)
    assert kept.comparison['explore_kept']
    assert kept.comparison['records'] == ('explore',)
    assert kept.comparison['costs'][1] < kept.comparison['costs'][0]
    tied = select_forecast(a, torch.full((3, 4), .5), torch.ones(3, dtype=torch.bool), None)
    assert not tied.comparison['explore_kept']
    assert tied.comparison['records'] == ('greedy',)
    assert a.walk is not None and a.comparison is None  # scoring did not publish


def test_forecast_previews_publish_only_the_committed_observation():
    from test_negative_expectation import _anticipating_model
    model, owner, meaning = _anticipating_model()
    model.train()
    model._stage_expectation_queries(training=True)
    pending = owner._inter_last_meaning[0]
    mask = torch.ones(1, dtype=torch.bool)
    for _ in range(2):
        _, _, held = owner.sentence_prediction_cost([3], [meaning.roles], mask,
            layout='infix', role_masks=[meaning.role_mask])
        assert held[0][0].comparison is not None
        assert owner._inter_last_meaning[0] is pending
        assert not owner.consume_expectation_walk_outcomes()
    owner._inter_last_meaning = held[0]
    owner.observe_stm_end_state([3], [meaning.roles], mask=mask, layout='infix',
        role_masks=[meaning.role_mask], train_prediction=False)
    assert len(owner.consume_expectation_walk_outcomes()) == 1
    assert not owner.consume_expectation_walk_outcomes()
