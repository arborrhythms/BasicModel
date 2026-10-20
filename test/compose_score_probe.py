"""Observe the current reference-variant menu on saved chooser features; no sampling or extra training.

Re-scoring fixes operands, candidates, priors and eligibility. Its only changing
inputs are the chooser parameters at the actual reconstruction-owner step.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import torch
from chooser_logit_observer import record_logits
def frozen(value):
    if isinstance(value,torch.nn.Parameter): return value
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, dict): return {k:frozen(v) for k,v in value.items()}
    if isinstance(value, tuple): return tuple(frozen(v) for v in value)
    if isinstance(value, list): return [frozen(v) for v in value]
    return value



@contextmanager
def observe_score_function(result):
    from Language import OperationSelectionLayer, MLPTransformChooser, AnchorDotTransformChooser
    from Models import BasicModel
    methods = {(cls, name): getattr(cls, name)
        for cls in (MLPTransformChooser, AnchorDotTransformChooser)
        for name in ('score_binary', 'score_unary')}
    attend = OperationSelectionLayer.attend
    forward = OperationSelectionLayer.forward
    select_logits = OperationSelectionLayer.select_logits
    surrogate = BasicModel._compose_score_function_loss
    train = BasicModel._sentence_train_step
    current, pending, selected = [], [], []
    records = result.setdefault('compose_score_function_steps', [])
    ranges = result.setdefault('chooser_logit_ranges', {})
    batch = {}
    owner_step = 0

    def selection(module, logits, **kwargs):
        if current and current[-1]['module'] is module:
            current[-1]['proposal_logits'] = frozen(kwargs.get('alternative_logits', logits))
        return select_logits(module, logits, **kwargs)

    def scorer(cls, name):
        def observed(chooser, *args, **kwargs):
            value = methods[cls, name](chooser, *args, **kwargs)
            if current and current[-1]['module'].chooser is chooser:
                current[-1][name] = (frozen(args), frozen(kwargs))
                current[-1][name + '_initial'] = frozen(value)
            return value
        return observed

    def operation(module, x, **kwargs):
        masked = kwargs.get('masked_action')
        capture = torch.is_tensor(masked) and bool((masked >= 0).any())
        entry = dict(walk='compose', module=module, depth=frozen(kwargs.get('depth')), width=x.shape[1])
        if capture:
            entry['forced'] = masked.detach().ge(0)
            references = kwargs.get('reference_data') or {}
            binary_names = module.op_names or [str(i) for i in range(module.r_reduce)]
            unary_names = module.unary_names or [str(i) for i in range(module.r_apply)]
            binary = [binary_names[i] for i in references.get('binary_ops', range(module.r_reduce))]
            unary = [unary_names[i] for i in references.get('unary_ops', range(module.r_apply))]
            entry['action_names'] = binary * (x.shape[1] - 1) + unary * x.shape[1] + ['STOP']
            current.append(entry)
        try:
            value = forward(module, x, **kwargs)
        finally:
            if capture: current.pop()
        route = value[2]
        record_logits(ranges, str(result.get('_active_epoch', 0)), route['logits'])
        if capture:
            entry.update(logits=route['logits'], actions=route['action'].detach(),
                         original=route['logits'].detach().clone(),
                         eligible=route['departure_eligible'].detach())
            pending.append(entry)
        return value

    def attention(module, keys, legal, space, **kwargs):
        mask=kwargs.get('masked_action')
        capture=torch.is_tensor(mask) and bool(mask.ge(0).any())
        entry=dict(walk='narrowing',module=module,keys=frozen(keys),legal=frozen(legal),
            space=frozen(space),prior=frozen(kwargs.get('prior')))
        if capture: current.append(entry)
        try:
            value=attend(module,keys,legal,space,**kwargs)
        finally:
            if capture: current.pop()
        if not kwargs.get('return_details'): return value
        logits=value[3]['logits']
        record_logits(ranges, 'narrowing:'+str(result.get('_active_epoch', 0)), logits)
        if capture:
            eligible=torch.isfinite(logits)&(torch.arange(logits.shape[-1],device=logits.device)[None]!=mask[:,None])
            entry.update(forced=mask.ge(0), logits=logits,actions=value[0].detach(),
                         original=logits.detach().clone(),eligible=eligible)
            pending.append(entry)
        return value

    @torch.no_grad()
    def scores(entry):
        if entry['walk']=='narrowing':
            return attend(entry['module'],entry['keys'],entry['legal'],entry['space'],
                prior=entry['prior'],return_details=True)[3]['logits'].detach()
        chooser = entry['module'].chooser
        bargs, bkwargs = entry['score_binary']
        uargs, ukwargs = entry['score_unary']
        stop, binary = methods[type(chooser), 'score_binary'](chooser, *bargs, **bkwargs)
        _, unary = methods[type(chooser), 'score_unary'](chooser, *uargs, **ukwargs)
        B, N = stop.shape[0], entry['width']
        depth = entry['depth']
        if depth is None: depth = torch.full((B,), N, device=stop.device)
        live = torch.arange(N, device=stop.device)[None] < depth[:, None]
        def joined(stop, binary, unary):
            stop = (stop.squeeze(-1) * live).sum(-1) / depth.clamp_min(1)
            return torch.cat((binary.reshape(B, -1), unary.reshape(B, -1), stop[:, None]), -1)
        old_stop, old_binary = entry['score_binary_initial']
        _, old_unary = entry['score_unary_initial']
        return entry['original'] + joined(stop, binary, unary) - joined(old_stop, old_binary, old_unary)

    def compared(model, path, costs):
        value = surrogate(model, path, costs)
        selected.clear()
        mask = model._last_compose_score_function['mask']
        active = model._sentence_cost_registry.row_mask
        active = torch.ones(costs.shape[0], device=costs.device, dtype=torch.bool) if active is None else active.bool()
        batch.clear()
        batch.update(costs=model._last_compose_score_function['costs'], measured_costs=costs.detach(),
                     mask=mask, active=active, scale=model._last_compose_score_function['scale'],
                     epoch=result.get('_active_epoch', 0), sentence=model._open_sentence_slot,
                     departure=frozen(model._sentence_departure))
        for entry in pending:
            entry['gradient'] = (torch.autograd.grad(value, entry['logits'], retain_graph=True,
                allow_unused=True)[0] if value.requires_grad else None)
            selected.append(entry)
        pending.clear()
        return value

    def stepped(model, loss):
        nonlocal owner_step
        owner_step += 1
        if model._sentence_trial != 'explore': return train(model, loss)
        rng = torch.random.get_rng_state()
        for entry in selected: entry['before'] = scores(entry)
        assert torch.equal(rng, torch.random.get_rng_state())
        value = train(model, loss)
        rng = torch.random.get_rng_state()
        for entry in selected: entry['after'] = scores(entry)
        for row in torch.nonzero(batch['active'], as_tuple=False).flatten().tolist():
            costs = batch['costs'][row]
            advantage = float(costs[1] - costs[0])
            departed = bool(batch['mask'][row].any())
            record = dict(step=len(records), epoch=batch['epoch'], sentence=batch['sentence'],
                          owner_step=owner_step,
                          batch_row=row, departed=departed, C_greedy=float(costs[0]),
                          C_explore=float(costs[1]), advantage=advantage,
                          action=None, action_name=None, p_before=None, p_after=None)
            record['measured_costs'] = batch['measured_costs'][row].cpu().tolist()
            entries = [e for e in selected if bool(e['forced'][row])]
            assert len(entries) == int(departed)
            if departed:
                entry, = entries
                action = int(entry['actions'][row])
                assert bool(entry['eligible'][row, action])
                before, after = entry['before'][row], entry['after'][row]
                p = entry['original'][row].softmax(-1)
                expected = -p * p[action]
                expected[action] += p[action]
                scale = float(batch['scale'][row][batch['mask'][row]][0])
                draw=batch['departure']
                walks=int(draw['walk_count'][row]);rounds=int(draw['walk_rounds'][row])
                alternatives=int(entry['eligible'][row].sum())
                # The production proposal mixes a uniform floor with the
                # expectation-free policy. K is its inverse only at φ=1
                # (or when that policy happens to be uniform).
                eligible = entry['eligible'][row]
                free = entry['proposal_logits'][row].masked_fill(~eligible, -torch.inf)
                fraction = entry['module'].exploration_fraction
                temperature = entry['module'].exploration_temperature
                proposal = (fraction * eligible.to(free) / alternatives
                            + (1-fraction) * (free / temperature).softmax(-1))
                expected_scale = walks * rounds * proposal[action].reciprocal()
                torch.testing.assert_close(expected_scale.new_tensor(scale), expected_scale,
                                           rtol=1e-6, atol=1e-6)
                assert int(draw['walk'][row])==(0 if entry['walk']=='narrowing' else 1)
                expected *= scale * advantage / batch['active'].sum()
                actual = (torch.zeros_like(expected) if entry['gradient'] is None
                          else entry['gradient'][row].detach())
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                names = [] if entry['walk']=='narrowing' else entry['action_names']
                if entry['walk']=='narrowing':
                    names=['divide','descend','gloss','and','or','not'] * (entry['original'].shape[-1]//6)+['STOP']
                # Central difference of the very same mean-reduced surrogate.
                z = entry['original'][row].double()
                plus, minus = z.clone(), z.clone()
                epsilon = 1e-4
                plus[action] += epsilon; minus[action] -= epsilon
                fd = float(scale * advantage * (plus.softmax(-1)[action] - minus.softmax(-1)[action])
                           / (2 * epsilon * int(batch['active'].sum())))
                record.update(walk=entry['walk'], action=action, action_name=names[action], sampling_scale=scale,
                    W=walks,R_walk=rounds,K=alternatives,
                    eligible_rounds=rounds,total_eligible_rounds=int(draw['rounds'][row]),
                    proposal_probability=float(proposal[action]), uniform_fraction=fraction,
                    round=int(torch.nonzero(batch['mask'][row], as_tuple=False)[0]),
                    p_forward=float(p[action]), p_before=float(before.softmax(-1)[action]),
                    p_after=float(after.softmax(-1)[action]),
                    logit_gradient=actual.cpu().tolist(), expected_gradient=expected.cpu().tolist(),
                    gradient_max_error=float((actual-expected).abs().max()),
                    finite_difference=dict(logit=action, epsilon=epsilon, numeric=fd,
                        analytic=float(actual[action]), error=abs(fd-float(actual[action]))),
                    eligible_alternatives=alternatives)
            records.append(record)
        counts={}
        for r in records:
            if r['departed'] and r['advantage']!=0:
                key=r['walk']+':'+r['action_name'];counts[key]=counts.get(key,0)+1
        result['nonzero_advantage_by_walk_action']=counts
        result['nonzero_advantage_sentences'] = sum(r['departed'] and r['advantage'] != 0 for r in records)
        assert torch.equal(rng, torch.random.get_rng_state())
        selected.clear()
        return value

    with ExitStack() as stack:
        for cls, name in methods: stack.enter_context(patch.object(cls, name, scorer(cls, name)))
        stack.enter_context(patch.object(OperationSelectionLayer, 'forward', operation))
        stack.enter_context(patch.object(OperationSelectionLayer, 'attend', attention))
        stack.enter_context(patch.object(OperationSelectionLayer, 'select_logits', selection))
        stack.enter_context(patch.object(BasicModel, '_compose_score_function_loss', compared))
        stack.enter_context(patch.object(BasicModel, '_sentence_train_step', stepped))
        yield
