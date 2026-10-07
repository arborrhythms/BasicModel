"""Observe §17 on saved chooser features; no sampling or extra training.

Re-scoring fixes operands, candidates, priors and eligibility. Its only changing
inputs are the chooser parameters at the actual reconstruction-owner step.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import torch
from review15_preference_probe import frozen


@contextmanager
def observe_score_function(result):
    from Language import OperationSelectionLayer, MLPTransformChooser, AnchorDotTransformChooser
    from Models import BasicModel
    methods = {(cls, name): getattr(cls, name)
        for cls in (MLPTransformChooser, AnchorDotTransformChooser)
        for name in ('score_binary', 'score_unary')}
    attend = OperationSelectionLayer.attend
    forward = OperationSelectionLayer.forward
    surrogate = BasicModel._compose_score_function_loss
    train = BasicModel._sentence_train_step
    current, pending, selected = [], [], []
    records = result.setdefault('compose_score_function_steps', [])
    ranges = result.setdefault('chooser_logit_ranges', {})
    batch = {}
    owner_step = 0

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
            current.append(entry)
        try:
            value = forward(module, x, **kwargs)
        finally:
            if capture: current.pop()
        route = value[2]
        if route['logits'].requires_grad:
            finite = route['logits'].detach()[torch.isfinite(route['logits'])]
            if finite.numel():
                epoch = str(result.get('_active_epoch', 0))
                row = ranges.setdefault(epoch, dict(minimum=float('inf'), maximum=-float('inf'), calls=0,
                                                    nan_count=0, positive_infinity_count=0))
                row['minimum'] = min(row['minimum'], float(finite.min()))
                row['maximum'] = max(row['maximum'], float(finite.max()))
                row['calls'] += 1
                row['nan_count'] += int(torch.isnan(route['logits']).sum())
                row['positive_infinity_count'] += int(torch.isposinf(route['logits']).sum())
        if capture:
            entry.update(logits=route['logits'], actions=route['action'].detach(),
                         original=route['logits'].detach().clone(),
                         eligible=route['departure_eligible'].detach())
            pending.append(entry)
        return value

    def attention(module, keys, legal, space, **kwargs):
        value=attend(module,keys,legal,space,**kwargs)
        if not kwargs.get('return_details'): return value
        logits=value[3]['logits']
        if logits.requires_grad:
            finite=logits.detach()[torch.isfinite(logits)]
            if finite.numel():
                epoch=str(result.get('_active_epoch',0))
                row=ranges.setdefault('narrowing:'+epoch,dict(minimum=float('inf'),maximum=-float('inf'),calls=0))
                row['minimum']=min(row['minimum'],float(finite.min()))
                row['maximum']=max(row['maximum'],float(finite.max()))
                row['calls']+=1
        mask=kwargs.get('masked_action')
        if torch.is_tensor(mask) and bool(mask.ge(0).any()):
            eligible=torch.isfinite(logits)&(torch.arange(logits.shape[-1],device=logits.device)[None]!=mask[:,None])
            pending.append(dict(walk='narrowing',module=module,keys=frozen(keys),legal=frozen(legal),
                space=frozen(space),prior=frozen(kwargs.get('prior')),forced=mask.ge(0),
                logits=logits,actions=value[0].detach(),original=logits.detach().clone(),eligible=eligible))
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
                     epoch=result.get('_active_epoch', 0), sentence=model._open_sentence_slot)
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
                expected *= scale * advantage / batch['active'].sum()
                actual = (torch.zeros_like(expected) if entry['gradient'] is None
                          else entry['gradient'][row].detach())
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                names = [] if entry['walk']=='narrowing' else ((entry['module'].op_names or [str(i) for i in range(entry['module'].r_reduce)]) * (entry['width']-1)
                         + (entry['module'].unary_names or [str(i) for i in range(entry['module'].r_apply)]) * entry['width'] + ['STOP'])
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
                    eligible_rounds=int(scale / int(entry['eligible'][row].sum())),
                    round=int(torch.nonzero(batch['mask'][row], as_tuple=False)[0]),
                    p_forward=float(p[action]), p_before=float(before.softmax(-1)[action]),
                    p_after=float(after.softmax(-1)[action]),
                    logit_gradient=actual.cpu().tolist(), expected_gradient=expected.cpu().tolist(),
                    gradient_max_error=float((actual-expected).abs().max()),
                    finite_difference=dict(logit=action, epsilon=epsilon, numeric=fd,
                        analytic=float(actual[action]), error=abs(fd-float(actual[action]))),
                    eligible_alternatives=int(entry['eligible'][row].sum()))
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
        stack.enter_context(patch.object(BasicModel, '_compose_score_function_loss', compared))
        stack.enter_context(patch.object(BasicModel, '_sentence_train_step', stepped))
        yield
