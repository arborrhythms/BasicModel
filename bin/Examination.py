"""Whole paths consisting only of a glance and one examination walk."""
from dataclasses import replace
from types import SimpleNamespace
import torch
from Layers import Error
from Understanding import ConceptualField


def operated(model, opened, reading, *, proposals=False):
    from FieldLogic import reduce
    from FamiliarTier import observe_intersection
    owner = model._concept_owner()
    evidence = opened.evidence
    # The glance retains one reading per native word bracket.
    for step in range(reading.actions.shape[1]):
        action = reading.actions[:,step]
        scopes = reading.field_scopes[:,step,:evidence.shape[2]]
        scopes = torch.nn.functional.pad(scopes,(0,max(0,evidence.shape[2]-scopes.shape[-1])))
        op = action.remainder(6)
        for name, number in (('intersection',3),('union',4),('not',5)):
            selected = scopes & (op == number)[:,None] & (action >= 0)[:,None]
            if not bool(selected.any()):
                continue
            if name == 'not':
                changed = evidence.flip(-1)
            else:
                pair = reduce(evidence,operation=name,dim=2,mask=selected[None])
                changed = pair[:,:,None].expand_as(evidence)
            if proposals and name == 'intersection':
                for batch in selected.any(-1).nonzero().flatten().tolist():
                    lanes = evidence[:,batch,selected[batch]].amax(1)
                    ids = opened.concept_ids if opened.concept_ids.ndim==1 else opened.concept_ids[:,batch]
                    support = [(owner._csw_row_of(int(ids[c])),pole) for c,pole in (lanes>0).nonzero().tolist()
                               if int(ids[c])>=0 and owner._row_order(owner._csw_row_of(int(ids[c])))==0]
                    observe_intersection(owner,support,batch=batch,examined=True)
            evidence = torch.where(selected[None,...,None],changed,evidence)
    return replace(opened,evidence=evidence)


def run(model, sid, present):
    from FamiliarTier import open_sentence,promote_candidates
    from SentenceField import native_target,publish,select_fields
    from ModelAttention import handoff
    from EpisodicMemory import select,plan_anchor,anchor
    from Occurrence import address_at,native_content_key
    from Meaning import ConceptualMeaning
    from FieldProjection import key
    owner = model._concept_owner()
    open_sentence(model,sid,present)
    opened = model._sentence_open_fields[sid]
    baseline = model._attention_read(sentence=sid)
    readings = [baseline]
    departures = [torch.full_like(present,-1,dtype=torch.long)]
    count = model.explore_paths if model._sentence_training else 0
    for _ in range(count):
        departure = torch.full_like(present,-1,dtype=torch.long)
        for batch in present.nonzero().flatten().tolist():
            eligible = baseline.alternatives[batch].nonzero().flatten()
            if len(eligible):
                departure[batch] = eligible[torch.randint(len(eligible),())]
        alternative = model._attention_read(sentence=sid,exploit=baseline,departure=departure)
        from SentenceCompose import select_rows
        selected = select_rows(baseline,alternative,departure>=0)
        readings.append(type(baseline)(type(baseline.table)(*selected[0]),*selected[1:]))
        departures.append(departure)
    fields,registries,costs,outputs = [],[],[],[]
    target = native_target(model,opened)
    for reading in readings:
        field = operated(model,opened,reading)
        registry = Error(row_mask=present)
        decoded = model.reverse(model.conceptualSpace.subspace.carrier_like(),field=field)
        from Omission import visible_events
        visible,observed = visible_events(model,decoded._reconstruction_percepts,target)
        model._reverse_event_loss(visible,observed,registry=registry,name='reconstruction_reverse',weight=model.loss.reconstruction_scale)
        output = model.outputSpace.read_field(field,owner)
        desired = model._sentence_supplied_answers
        if torch.is_tensor(desired) and model.inputSpace.data.has_supervised_outputs:
            prediction = model.normalizer.denormalize(output,which='output')
            prediction = model._align_output_pred(prediction,desired)
            registry.error('answer',(prediction-desired).square().reshape(len(present),-1).mean(-1),1.,
                           mask=present,category='prediction',objective='output')
        cost = sum(term['weight']*registry._value(term) for term in registry._terms.values())
        if not torch.is_tensor(cost):
            cost = output.new_zeros(len(present))
        fields.append(field);registries.append(registry);costs.append(cost);outputs.append(output)
    costs = torch.stack(costs,1)
    winners = costs.detach().argmin(1)
    field = select_fields(fields,winners)
    from SentenceCompose import select_rows
    reading = baseline
    for i in range(1,len(readings)):
        selected = select_rows(reading,readings[i],winners==i)
        reading = type(reading)(type(reading.table)(*selected[0]),*selected[1:])
    handoff(model,reading)
    from ModelAttention import stage_expectation
    stage_expectation(model)
    publish(model,field)
    decoded = model.reverse(model.conceptualSpace.subspace.carrier_like(),field=field)
    from Understanding import InputReconstruction
    points = key(owner,field).mean(1)
    selected_cost = costs.gather(1,winners[:,None])[:,0]
    sentence_costs = selected_cost.new_zeros(len(present),model.inputSpace._packed_sentence_slot_mask.shape[1])
    sentence_costs[:,sid] = selected_cost
    model._input_reconstructions[sid] = InputReconstruction(
        points[:,None], decoded.materialize(), torch.zeros_like(selected_cost),
        selected_cost, torch.zeros_like(present), sentence_costs,
        decoded._reconstruction_percepts)
    if model.training:
        operated(model,opened,reading,proposals=True)
    store = model.symbolSpace.ltm_store
    ids = field.concept_ids
    orders = ids.new_tensor([owner._row_order(owner._csw_row_of(int(cid))) if int(cid)>=0 else -1 for cid in ids.flatten()]).reshape(ids.shape)
    entry_ids,entry_values = select(field,store.episodic_capacity,present,orders,order_count=store.episodic_orders)
    for batch in present.nonzero().flatten().tolist():
        entry = entry_ids[batch],entry_values[batch]
        document,position = address_at(model,batch,sid)
        meaning = ConceptualMeaning.from_description(points[batch].detach())
        with store.at_address(document,position,native_content_key(meaning)):
            plan = plan_anchor(store,SimpleNamespace(meaning=meaning),entry,batch)
            if plan is not None:
                anchor(store,plan,entry,stream=batch,time=True)
        owner.mtm.close(None,stream=batch,references=owner._field_activated_references.get(batch,()))
    if model.training:
        promote_candidates(owner)
    model._last_examination = dict(fields=fields,readings=readings,departures=departures,costs=costs.detach(),winners=winners)
    model._last_sentence_field = field
    model._sentence_trial_costs.append(costs.detach())
    model._sentence_winners.append(winners)
    if model._sentence_training and torch.is_grad_enabled():
        from ObjectiveOwnership import registry_costs
        from SentenceTraining import combine,surrogate
        objectives={}
        combine(objectives, registry_costs(model.errors))
        for term in model.errors._terms.values():
            term['trained'] = False
        for i,registry in enumerate(registries):
            if i:
                valid = departures[i]>=0
                probability = readings[i].probabilities.gather(1,departures[i].clamp_min(0)[:,None])[:,0]
                advantage = (costs[:,i]-costs[:,0]).detach()
                credit = advantage * probability.clamp_min(torch.finfo(probability.dtype).tiny).log() * valid
                registry.add('output.examination_score_function',credit,category='output',objective='chooser',side='A')
            for term in registry._terms.values():
                if term.get('objective')!='chooser':
                    term['multiplier'] = (1. if len(readings)==1 else (.5 if i==0 else .5/(len(readings)-1))) if term.get('objective')=='output' else 2./len(readings)
            combine(objectives,registry_costs(registry))
        model._single_sentence_objectives = objectives
        model._sentence_cost_registry = registries[int(winners[0])]
        model._sentence_train_step(surrogate(objectives,points.new_ones(())))
        model._percept_arrival_loss = model._percept_arrival_loss.detach()
        model._single_sentence_objectives = None
        model._single_sentence_steps += 1
        from dataclasses import fields as dataclass_fields
        result = model._input_reconstructions[sid]
        model._input_reconstructions[sid] = replace(result, **{
            item.name: getattr(result,item.name).detach()
            for item in dataclass_fields(result) if torch.is_tensor(getattr(result,item.name))})
        for term in model.errors._terms.values():
            for name in ('values','numerators','baselines'):
                if name in term:
                    term[name] = [value.detach() for value in term[name]]
        if any('answer' in registry._terms for registry in registries):
            model._sentence_reader_updates += 1
        points = points.detach()
    from ScalarPercepts import commit
    commit(model, sid, present)
    return points
