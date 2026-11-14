"""Compare complete walks at their owner's scoring boundary."""
from dataclasses import dataclass, replace
import torch


def departure_at(eligible, *, uniform=None):
    """One uniformly selected legal departure per row; -1 means none."""
    if eligible.shape[-1] == 0:
        return torch.full((eligible.shape[0],), -1, device=eligible.device, dtype=torch.long)
    if uniform is None:
        uniform=torch.rand(eligible.shape[0],device=eligible.device)
    elif uniform.shape != eligible.shape[:1] or uniform.device != eligible.device:
        raise ValueError('a departure draw must align with its streams')
    rank=(uniform*eligible.sum(-1)).long()
    selected=eligible & (eligible.long().cumsum(-1)==rank[:,None]+1)
    return torch.where(eligible.any(-1),selected.long().argmax(-1),-1)


def finished_wins(costs, pending, *, certified=None):
    """Certified completion takes precedence; otherwise compare full costs."""
    finished=~pending if certified is None else certified
    return (~finished[:,0]&finished[:,1]) | (
        (finished[:,0]==finished[:,1]) & (costs[:,1]<costs[:,0]))


def generate_credit(language, audit, tops, features):
    """The compose score function, including action and departure proposal q."""
    from Expectation import alternative_probabilities
    from SentenceCredit import score_function
    departure = audit['departure']
    ar = torch.arange(len(departure), device=tops.device)
    step = departure.clamp_min(0)
    observed = features[ar, step].detach()
    from Generative import WALK_FEATURE_WIDTH
    legal = observed[..., WALK_FEATURE_WIDTH - 1].bool()
    logits = language.generate_policy_logits(tops[ar, step].detach(),
        features=observed[..., :WALK_FEATURE_WIDTH - 1])
    logits = logits.masked_fill(~legal, -torch.inf)
    logits = torch.where(legal.any(-1, keepdim=True), logits, 0.)
    excluded = torch.nn.functional.one_hot(audit['greedy'][ar, step].clamp_min(0), logits.shape[-1]).bool()
    proposal = alternative_probabilities(logits.detach().masked_fill(excluded, -torch.inf),
        fraction=language._tree_layer(2).exploration_fraction,
        temperature=language._tree_layer(2).exploration_temperature)
    action = audit['explore'][ar, step].clamp_min(0)[:, None]
    probability = logits.softmax(-1).gather(1, action)
    q_action = proposal.gather(1, action)
    rounds = (audit['proposal_rounds'][:, None] if 'proposal_rounds' in audit else
              audit['eligible'].sum(-1, keepdim=True))
    mask = departure.ge(0)[:, None]
    correction = torch.where(mask, rounds / q_action.clamp_min(torch.finfo(q_action.dtype).tiny), 0.)
    credit, details = score_function(probability, correction.detach(), mask, audit['costs'])
    audit['score_function'] = dict(details, proposal=q_action.detach(), eligible_rounds=rounds.detach())
    return credit


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
    from copy import copy
    work = copy(meter)
    work._counts = meter._counts.copy()
    step = len(trace['actions']) - 1
    trace.setdefault('forks', {})[step] = dict(_copy_containers(continuation),
        state=ThoughtTrialState(model), work=work,
        departure=step, excluded=trace['actions'][-1])


def record_thought_bind(trace, plan):
    """A bind is one more eligible decision in the episode's single fork."""
    decision = plan['decision']
    selected = decision['selected']
    trace['actions'].append(selected)
    trace['eligible'].append(len(decision['candidates']) > 1)
    trace['choices'].append(dict(probability=decision['probabilities'][selected],
        alternatives=1, category='expectation', bind=dict(
            candidates=decision['candidates'], selected=selected,
            greedy=decision['greedy'], departed=decision['departed'],
            features=decision['features'].detach().cpu().tolist(),
            logits=decision['logits'].detach().cpu().tolist(),
            probabilities=decision['probabilities'].detach().cpu().tolist())))


class ThoughtTrialState:
    """Snapshot only the declared thought-effect owners, never model weights.

    History records are immutable. Knowing writes fresh field tensors and
    expectation replaces its pending estimate. Their old graphs therefore
    remain safe to restore without detaching the kept policy's credit.
    """
    def __init__(self,model, *, detach=False):
        memory=model._what_memory()
        self.model = model
        from MemoryActivity import snapshot
        self.activity = snapshot()
        self.clock = model.when_time.detach().clone()
        store = getattr(getattr(model, 'symbolSpace', None), 'ltm_store', None)
        self.store_clock = None if store is None else (store, store.model_clock.detach().clone())
        from Memory import STMChunks
        stm = getattr(getattr(model, 'conceptualSpace', None), 'stm', None)
        owners=[(memory,('_what_slots','_what_closure_pressure','_episode_live','_thought_next_id','_address_sources')),
                (model,('expectation_gain', '_answer_attention_obs', '_last_answer_construction',
                        '_last_output_comparison', '_last_output_walk_trace',
                        '_sentence_answer_cost', '_sentence_answer_predictions',
                        '_sentence_answer_raw_cost'))]
        if stm is not None:
            owners.append((stm, tuple('_live_' + name for name in STMChunks.FIELDS) +
                           ('_thought_contents', '_active_batch', '_max_depth_host')))
        conceptual = (model._concept_owner() if hasattr(model, '_concept_owner') else
                      getattr(model, 'conceptualSpace', None))
        self.field_owner = conceptual
        if conceptual is not None:
            owners.append((conceptual, ('_percept_field', '_cs_field_where', '_cs_field_when',
                                       '_cs_field_concept_ids', '_episode_review_field')))
        carrier=getattr(getattr(model,'conceptualSpace',None),'subspace',None)
        if carrier is not None:
            owners.append((carrier,tuple(name for name in vars(carrier)
                if name.startswith('_concept_'))))
        expectation=getattr(getattr(model,'symbolSpace',None),'expectation',None)
        if expectation is not None:owners.append((expectation,('_inter_last_meaning',)))
        copy = _detached if detach else _copy_containers
        from ThoughtStrata import snapshot
        self.thought_strata = None if stm is None else (stm, snapshot(stm._thought_stores))
        self.saved=[(owner,{name:copy(getattr(owner,name)) for name in names if hasattr(owner,name)},names)
                    for owner,names in owners if owner is not None]
        for owner, values, names in self.saved:
            if '_live_chunk_truth' in values:
                values['_live_chunk_truth'] = values['_live_chunk_truth'].clone()

    def restore(self, *, effects_only=False, row=None):
        if row is not None:
            if effects_only:
                raise ValueError('row restoration requires the complete thought state')
            from copy import copy
            current = type(self)(self.model)
            departing = copy(self)
            # One global clock must not undo another stream's later thought.
            departing.clock = torch.maximum(self.clock, current.clock)
            if self.store_clock is not None:
                departing.store_clock = (self.store_clock[0],
                    torch.maximum(self.store_clock[1], current.store_clock[1]))
            batch = len(self.model._what_memory()._what_slots)
            if not 0 <= int(row) < batch:
                raise IndexError('thought departure stream is outside the active batch')
            selected = torch.zeros(batch, dtype=torch.long, device=self.clock.device)
            selected[int(row)] = 1
            type(self).keep((current, departing), selected)
            return
        if not effects_only:
            from MemoryActivity import restore
            restore(self.activity)
            self.model.when_time.copy_(self.clock)
            self.model.when_encoding.t = self.model.present()
            if self.store_clock is not None:
                store, clock = self.store_clock
                store.model_clock.copy_(clock)
        if self.thought_strata is not None and not effects_only:
            from ThoughtStrata import restore
            stm, snapshot = self.thought_strata
            restore(stm._thought_stores, snapshot)
        for owner,values,names in self.saved:
            if effects_only and ('_what_slots' in names or '_thought_contents' in names
                                 or '_episode_review_field' in names):
                continue
            if '_thought_contents' in names:
                from Memory import STMChunks
                owner.set_state(tuple(values['_live_' + name] for name in STMChunks.FIELDS))
                owner.ensure_batch(values['_active_batch'])
                owner._max_depth_host = values['_max_depth_host']
                owner._thought_contents = values['_thought_contents']
                continue
            # A trial may introduce a knowing carrier for the first time.
            dynamic=tuple(name for name in vars(owner) if name.startswith('_concept_')) if any(name.startswith('_concept_') for name in names) else ()
            for name in set(names)|set(dynamic):
                if name in values:object.__setattr__(owner,name,_copy_containers(values[name]))
                elif name in vars(owner):delattr(owner,name)
        if effects_only:
            review = getattr(self.field_owner, '_episode_review_field', None)
            if review is not None:
                from SentenceField import publish
                publish(self.model, review)

    @classmethod
    def keep(cls, states, winners):
        """Restore the selected stream's effects without executing its episode."""
        if bool((winners == winners[0]).all()):
            states[int(winners[0])].restore()
            return
        from copy import copy
        chosen = copy(states[0])
        batch = len(winners)
        indices = winners.tolist()
        def rows(values, name=None):
            first = values[0]
            if name == '_thought_next_id':
                # Histories are selected per stream; their shared allocator
                # must remain beyond every selected history's frontier.
                return max(values[index] for index in set(indices))
            if name == '_cs_field_concept_ids' and torch.is_tensor(first):
                return first.new_tensor(sorted({int(cid)
                    for row, index in enumerate(indices)
                    for cid in (values[index] if values[index].ndim == 1 else values[index][:, row])}))
            if name in ('_cs_field_where', '_cs_field_when') and torch.is_tensor(first):
                width = max(value.shape[1] for value in values)
                return torch.stack([torch.nn.functional.pad(values[index][row],
                    (0, 0, 0, width - values[index].shape[1]))
                    for row, index in enumerate(indices)])
            if name == '_percept_field' and first is not None:
                keys = tuple(sorted(set().union(*(value.keys for value in values))))
                R, E = max(v.values.shape[2] for v in values), max(v.events.shape[3] for v in values)
                lookup = {key: i for i, key in enumerate(keys)}
                data = first.values.new_zeros(len(keys), batch, R, 2)
                events = first.events.new_zeros(len(keys), batch, R, E, 2)
                where, when = first.where.new_zeros(batch, R, 2), first.when.new_zeros(batch, R, 2)
                spans = first.spans.new_zeros(batch, E, 2)
                bands = None if first.when_band is None else first.when_band.new_zeros(batch, R, 4)
                for row, index in enumerate(indices):
                    value = values[index]
                    r, e = value.values.shape[2], value.events.shape[3]
                    for old, key in enumerate(value.keys):
                        data[lookup[key], row, :r] = value.values[old, row]
                        events[lookup[key], row, :r, :e] = value.events[old, row]
                    where[row, :r], when[row, :r] = value.where[row], value.when[row]
                    spans[row, :e] = value.spans[row]
                    if bands is not None:
                        bands[row, :r] = value.when_band[row]
                return replace(first, keys=keys, values=data, events=events, where=where,
                               when=when, spans=spans, when_band=bands)
            if name == '_sentence_answer_predictions' and torch.is_tensor(first) and first.ndim == 3:
                # Realized text has a path-dependent byte extent. Keep each
                # selected stream, with zero padding just as at public output.
                width = max(value.shape[-1] for value in values)
                return torch.stack([torch.nn.functional.pad(values[index][row],
                    (0, width - values[index].shape[-1]))
                    for row, index in enumerate(indices)])
            if (name is not None and name.startswith('_live_') and torch.is_tensor(first)
                    and first.ndim and first.shape[0] >= batch):
                # STM reserves more streams than this microbatch may use.
                # Select its active prefix, retaining the unused reservation.
                return torch.cat((torch.stack([values[index][row]
                    for row, index in enumerate(indices)]), first[batch:]), 0)
            if torch.is_tensor(first) and first.ndim and first.shape[0] == batch:
                return torch.stack([values[index][row] for row, index in enumerate(indices)])
            if (name in ('_what_slots', '_thought_contents', '_what_closure_pressure',
                         '_inter_last_meaning') and isinstance(first, (list, tuple))
                    and len(first) >= batch):
                return type(first)(values[indices[row]][row] if row < batch else value
                                   for row, value in enumerate(first))
            if isinstance(first, (list, tuple)) and len(first) == batch:
                return type(first)(values[index][row] for row, index in enumerate(indices))
            if isinstance(first, dict):
                keys = set().union(*(value.keys() for value in values))
                if all(type(key) is int and 0 <= key < batch for key in keys):
                    return {key: values[indices[key]][key] for key in keys if key in values[indices[key]]}
            return first
        chosen.clock = torch.stack([states[index].clock for index in set(indices)]).amax(0)
        if chosen.store_clock is not None:
            chosen.store_clock = (chosen.store_clock[0],
                torch.stack([states[index].store_clock[1] for index in set(indices)]).amax(0))
        from MemoryActivity import select
        chosen.activity = select([state.activity for state in states], winners)
        chosen.saved = []
        for position, (owner, values, names) in enumerate(states[0].saved):
            alternatives = [state.saved[position][1] for state in states]
            # Concept axes and reading axes may differ after a review or a
            # generated field. Select the field by identity below, then let
            # publish rebuild its codes and inventory addresses together.
            field_carrier = '_concept_activations' in values
            chosen.saved.append((owner, {name: rows([item[name] for item in alternatives], name)
                for name in values if not (field_carrier and name.startswith('_concept_'))
                and all(name in item for item in alternatives)}, names))
        if chosen.thought_strata is not None:
            chosen.thought_strata = (chosen.thought_strata[0],
                tuple(states[indices[row] if row < batch else 0].thought_strata[1][row]
                      for row in range(len(chosen.thought_strata[1]))))
        chosen.restore()
        from Understanding import ConceptualField
        from SentenceField import select_fields, publish
        for position, (owner, values, names) in enumerate(states[0].saved):
            if '_concept_activations' not in values:
                continue
            fields = [ConceptualField(saved[position][1]['_concept_ids'],
                saved[position][1]['_concept_activations'], saved[position][1]['_concept_where'],
                saved[position][1]['_concept_when'], None) for saved in (state.saved for state in states)]
            publish(chosen.model, select_fields(fields, winners))
        if chosen.field_owner is not None:
            position = next((i for i, (owner, _, _) in enumerate(states[0].saved)
                             if owner is chosen.field_owner), None)
            if position is not None:
                reviewed = any(states[index].saved[position][1].get('_episode_review_field') is not None
                               for index in indices)
                from SentenceField import knowing
                chosen.field_owner._episode_review_field = knowing(chosen.model) if reviewed else None








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
        allowed = (finished_wins(costs, torch.as_tensor(audit['pending']).detach().cpu())[b]
                   if 'pending' in audit else costs[b,1] < costs[b,0])
        result['explore_wins']+=int(wins[b]);result['strict_violations']+=int(wins[b] and not allowed)
        if key in previous:
            result['stability_pairs']+=1;result['stable_pairs']+=int(previous[key]==kept)
        previous[key]=kept;previous.move_to_end(key)
        while len(previous)>512:previous.popitem(last=False)
        result['explore_fraction']=result['explore_wins']/result['walks']
        result['derivation_stability']=(result['stable_pairs']/result['stability_pairs'] if result['stability_pairs'] else None)
        details.append(dict(kind=kind,surface=surface,sentence=sentence,row=actual,
            costs=costs[b].tolist(),explore_kept=bool(wins[b]),actions=kept))
