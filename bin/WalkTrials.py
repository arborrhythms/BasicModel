"""Compare complete walks at their owner's scoring boundary."""
from dataclasses import dataclass, replace
import torch


def departure_at(eligible):
    """One uniformly selected legal departure per row; -1 means none."""
    if eligible.shape[-1] == 0:
        return torch.full((eligible.shape[0],), -1, device=eligible.device, dtype=torch.long)
    rank=(torch.rand(eligible.shape[0],device=eligible.device)*eligible.sum(-1)).long()
    selected=eligible & (eligible.long().cumsum(-1)==rank[:,None]+1)
    return torch.where(eligible.any(-1),selected.long().argmax(-1),-1)


def select_output(greedy,explore,wins):
    def selected(a,b):
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape != b.shape:
                raise ValueError('output trials require the same bounded realization shape')
            return torch.where(wins.reshape(-1,*([1]*(a.ndim-1))),b,a)
        if a is None and b is None:return None
        raise ValueError('output trials disagree about their realization carrier')
    values={name:selected(getattr(greedy,name),getattr(explore,name))
            for name in ('actual','concepts','percepts','surface')}
    texts=tuple(b if bool(wins[row]) else a for row,(a,b) in
                enumerate(zip(greedy.texts,explore.texts)))
    return replace(greedy,**values,texts=texts)


def output_pair(realize,score):
    """Realize, then cost, both paths before returning either live graph.

    The scorer is loss-side only. It never enters the walk. Shared generate
    parameters still require the caller's reconstruction ownership boundary.
    """
    greedy,trace=realize(return_candidates=True)
    first=score(greedy)
    departure=departure_at(trace[1])
    explore,other=realize(exploit_actions=trace[0],departure=departure)
    second=score(explore)
    costs=torch.stack((first,second),-1).detach()
    wins=trace[1].any(-1)&(costs[:,1]<costs[:,0])
    audit=dict(costs=costs,wins=wins,departure=departure,
               greedy=trace[0].detach(),explore=other[0].detach(),
               stable=(trace[0]==other[0]).all(-1))
    result=replace(select_output(greedy,explore,wins),trace=greedy.trace+(
        {'operation':'generate:owner_comparison','owner':'output',
         'explore_kept':tuple(wins.tolist())},))
    return result,audit


def _copy_containers(value):
    """Copy mutable scratch containers while retaining each live graph."""
    from collections import deque
    if isinstance(value,deque):return deque(value)
    if isinstance(value,list):return [_copy_containers(v) for v in value]
    if isinstance(value,dict):return {k:_copy_containers(v) for k,v in value.items()}
    return value


def _detached(value):
    """Detach a fork's values without retaining an earlier policy graph."""
    from collections import deque
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return type(value)(_detached(item) for item in value)
    if callable(getattr(value, 'detached', None)):
        return value.detached()
    if callable(getattr(value, 'snapshot', None)):
        return value.snapshot(detach=True)
    return value


def capture_thought_fork(model, trace, meter, **continuation):
    """Replace the episode reservoir with probability 1/k, before execution."""
    if not trace['eligible'][-1]:
        return
    count = sum(trace['eligible'])
    device = trace['choices'][-1]['probability'].device
    if float(torch.rand((), device=device)) >= 1. / count:
        return
    from copy import copy
    work = copy(meter)
    work._counts = meter._counts.copy()
    trace['fork'] = dict(_detached(continuation),
        state=ThoughtTrialState(model, detach=True), work=work,
        departure=len(trace['actions'])-1, excluded=trace['actions'][-1])


class ThoughtTrialState:
    """Snapshot only the declared thought-effect owners, never model weights.

    History records are immutable. Knowing writes fresh field tensors and
    expectation replaces its pending estimate. Their old graphs therefore
    remain safe to restore without detaching the kept policy's credit.
    """
    def __init__(self,model, *, detach=False):
        memory=model._what_memory()
        owners=[(memory,('_what_slots','_what_closure_pressure','_episode_live','_thought_next_id','_address_sources')),
                (model,('expectation_gain', '_answer_attention_obs', '_last_answer_construction',
                        '_last_output_comparison', '_last_output_walk_trace',
                        '_sentence_answer_cost', '_sentence_answer_predictions',
                        '_sentence_answer_raw_cost'))]
        carrier=getattr(getattr(model,'conceptualSpace',None),'subspace',None)
        if carrier is not None:
            owners.append((carrier,tuple(name for name in vars(carrier)
                if name.startswith('_concept_'))+('_thought_occurrence',)))
        expectation=getattr(getattr(model,'symbolSpace',None),'expectation',None)
        if expectation is not None:owners.append((expectation,('_inter_last_meaning',)))
        copy = _detached if detach else _copy_containers
        self.saved=[(owner,{name:copy(getattr(owner,name)) for name in names if hasattr(owner,name)},names)
                    for owner,names in owners if owner is not None]

    def restore(self, *, effects_only=False):
        for owner,values,names in self.saved:
            if effects_only and '_what_slots' in names:
                continue
            # A trial may introduce a knowing carrier for the first time.
            dynamic=tuple(name for name in vars(owner) if name.startswith('_concept_')) if '_thought_occurrence' in names else ()
            for name in set(names)|set(dynamic):
                if name in values:object.__setattr__(owner,name,_copy_containers(values[name]))
                elif name in vars(owner):delattr(owner,name)


@dataclass(frozen=True)
class ForecastWalk:
    """Two prior-only forecasts awaiting one external observation.

    The records belong to the forecast, not the interaction history. Restoring
    a speculative history after the observation would erase newer evidence.
    """
    other: object
    greedy: tuple
    explore: tuple
    departure: int
    records: tuple
    step_cost: float

    def detached(self):
        return replace(self, other=self.other.detached())


def forecast_pair(model, meaning, *, row, run):
    """Hold both forecasts at one parameter version, with no published effects."""
    base = ThoughtTrialState(model)
    old = getattr(model, '_thought_walk', None)
    def trial(exploit=None, departure=-1):
        trace = dict(actions=[], eligible=[], exploit=exploit, departure=departure)
        model._thought_walk = trace
        model._anticipatory_choices = []
        result, pending = run()
        return pending, trace, tuple(result.records)
    try:
        greedy, trace, records = trial()
        base.restore()
        eligible = torch.tensor([trace['eligible']], dtype=torch.bool, device=meaning.roles.device)
        departure = int(departure_at(eligible)[0]) if eligible.numel() else -1
        explore, other, other_records = trial(tuple(trace['actions']), departure)
        if greedy.versions != explore.versions:
            raise RuntimeError('anticipation trials crossed a predictor update')
        walk = ForecastWalk(explore.detached(), tuple(trace['actions']),
            tuple(other['actions']), departure, (records, other_records), model.WHAT_STEP_COST)
        return replace(greedy.detached(), walk=walk)
    finally:
        base.restore()
        model._thought_walk = old


def select_forecast(pending, roles, occupied, kind):
    """Loss-side comparison of held estimates; the target never enters a walk."""
    walk = pending.walk
    if walk is None or pending.comparison is not None:
        return pending
    from torch.nn import functional as F
    def cost(value):
        prediction = value.prediction
        error = (prediction.roles - roles.detach().to(prediction.roles)).square().mean()
        error = error + F.binary_cross_entropy_with_logits(
            prediction.presence_logits, occupied.to(prediction.presence_logits))
        if kind is not None:
            error = error + F.binary_cross_entropy_with_logits(
                prediction.kind_logit, prediction.kind_logit.new_tensor(float(kind == 'relation')))
        return float(error.detach()) + walk.step_cost * value.work
    costs = (cost(pending), cost(walk.other))
    if not all(torch.isfinite(torch.tensor(costs))):
        raise FloatingPointError('non-finite anticipation comparison')
    wins = walk.departure >= 0 and costs[1] < costs[0]
    width = max(len(walk.greedy), len(walk.explore))
    audit = dict(costs=costs, explore_kept=wins, departure=walk.departure,
        greedy=walk.greedy + (-1,) * (width-len(walk.greedy)),
        explore=walk.explore + (-1,) * (width-len(walk.explore)),
        records=walk.records[int(wins)])
    return replace(walk.other if wins else pending, walk=None, comparison=audit)


def thought_pair(model,meaning,*,row,work_budget,registry,work,score):
    """One detached episode fork; keep R, credit R + A and local work."""
    from copy import copy
    from QueryWork import QueryWorkBudget
    base = ThoughtTrialState(model)
    initial = QueryWorkBudget(work_budget) if work is None else work
    old = getattr(model, '_thought_walk', None)

    def run(trace, meter, fork=None):
        model._thought_walk = trace
        result = model._run_selected_thought_once(meaning, row=row, work_budget=work_budget,
                                                  registry=registry, work=meter, fork=fork)
        error = 0. if score is None else score(result)
        # A standalone scorer may provide both owned terms. The ordinary
        # closing has a fixed compose reconstruction and a binding answer.
        if isinstance(error, dict):
            reconstruction, answer = error['reconstruction'], error['answer']
        else:
            reconstruction, answer = 0., error
        def number(value):
            return float(value.detach() if torch.is_tensor(value) else value)
        return result, (number(reconstruction), number(answer)), ThoughtTrialState(model)

    try:
        meter = copy(initial)
        meter._counts = initial._counts.copy()
        trace = dict(actions=[], eligible=[], choices=[], collect_fork=True)
        greedy, first, state = run(trace, meter)
        fork = trace.pop('fork', None)
        departure = -1 if fork is None else fork['departure']
        if fork is None:
            explore, second, other_state = greedy, first, state
            other = dict(actions=list(trace['actions']), eligible=list(trace['eligible']),
                         choices=[], reader_weight=0., comparison_weight=0.)
        else:
            fork['state'].restore()
            other = dict(actions=list(trace['actions'][:departure]),
                eligible=list(trace['eligible'][:departure]),
                choices=_detached(trace['choices'][:departure]),
                departure=departure, excluded=fork['excluded'])
            explore, second, other_state = run(other, fork['work'], fork=fork)
        wins = departure >= 0 and second[0] < first[0]
        trace['reader_weight'], other['reader_weight'] = float(not wins), float(wins)
        trace['comparison_weight'] = .5 if departure >= 0 else 1.
        other['comparison_weight'] = .5 if departure >= 0 else 0.
        result = explore if wins else greedy
        (other_state if wins else state).restore()
        # Local knowing/gain/prediction effects end with the episode. History
        # is the credit trail; its chosen result is committed by the caller.
        base.restore(effects_only=True)
        if work is not None:
            work._spent, work._counts = result.work.spent, result.work._counts.copy()
            result = replace(result, work=work)
        # §14 keeps sentence departure judgement at R + A. The episode's
        # own walk still pays for its metered work, as §5 specifies; that
        # local cost never enters the enclosing compose comparison or keep.
        work_costs = tuple(float(model.WHAT_STEP_COST) * result.work.spent
                           for result in (greedy, explore))
        costs = (sum(first)+work_costs[0], sum(second)+work_costs[1])
        model._last_thought_comparison = dict(costs=costs, keep_costs=(first[0],second[0]),
            components=(first,second), work_costs=work_costs,
            explore_kept=wins, departure=departure,
            greedy=tuple(trace['actions']), explore=tuple(other['actions']),
            stable=trace['actions'] == other['actions'])
        from ThoughtCredit import complete
        complete(model,greedy,explore,trace,other,departure,costs,row)
        width = max(len(trace['actions']),len(other['actions']))
        audit = dict(model._last_thought_comparison, costs=(first[0],second[0]),
            greedy=tuple(trace['actions'])+(-1,)*(width-len(trace['actions'])),
            explore=tuple(other['actions'])+(-1,)*(width-len(other['actions'])))
        observe_comparison(model,'think',audit,row=row)
        return result
    except BaseException:
        base.restore()
        raise
    finally:
        model._thought_walk = old


def observe_comparison(model,kind,audit,*,sentence=None,row=None,active=None):
    """Bounded observation-only counts; geometry is never an acceptance bar."""
    from collections import OrderedDict,deque
    costs=torch.as_tensor(audit['costs']).detach().cpu().reshape(-1,2)
    wins=torch.as_tensor(audit.get('wins',audit.get('explore_kept',False))).detach().cpu().reshape(-1)
    greedy=torch.as_tensor(audit['greedy']).detach().cpu().reshape(len(costs),-1)
    explore=torch.as_tensor(audit['explore']).detach().cpu().reshape(len(costs),-1)
    departure=torch.as_tensor(audit.get('departure',-1)).detach().cpu().reshape(-1)
    valid=torch.ones(len(costs),dtype=torch.bool) if active is None else active.detach().cpu().bool()
    stats=getattr(model,'_walk_audit',None)
    if stats is None:stats={};model._walk_audit=stats
    result=stats.setdefault(kind,dict(walks=0,explorable=0,explore_wins=0,strict_violations=0,
        stability_pairs=0,stable_pairs=0,explore_fraction=0.,derivation_stability=None))
    previous=getattr(model,'_walk_previous',None)
    if previous is None:previous=OrderedDict();model._walk_previous=previous
    details=getattr(model,'_walk_observations',None)
    if details is None:details=deque(maxlen=256);model._walk_observations=details
    staged=getattr(model,'_attention_forms',None)
    for b in range(len(costs)):
        if not bool(valid[b]):continue
        actual=b if row is None else row
        surface=tuple(name for name in staged[0][actual] if name) if staged is not None and actual<len(staged[0]) else ('row',actual)
        key=(kind,surface,sentence)
        kept=tuple((explore if bool(wins[b]) else greedy)[b].tolist())
        result['walks']+=1;result['explorable']+=int(departure[min(b,len(departure)-1)]>=0)
        result['explore_wins']+=int(wins[b]);result['strict_violations']+=int(wins[b] and not costs[b,1]<costs[b,0])
        if key in previous:
            result['stability_pairs']+=1;result['stable_pairs']+=int(previous[key]==kept)
        previous[key]=kept;previous.move_to_end(key)
        while len(previous)>512:previous.popitem(last=False)
        result['explore_fraction']=result['explore_wins']/result['walks']
        result['derivation_stability']=(result['stable_pairs']/result['stability_pairs'] if result['stability_pairs'] else None)
        details.append(dict(kind=kind,surface=surface,sentence=sentence,row=actual,
            costs=costs[b].tolist(),explore_kept=bool(wins[b]),actions=kept))
