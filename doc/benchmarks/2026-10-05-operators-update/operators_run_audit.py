"""Retain the accepted observers; extend the tenth run for attention and CE."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import os
import torch
from review17_run_audit import observe_run as accepted_observer, json_value


@contextmanager
def observe_round1(result):
    from Language import OperationSelectionLayer, LanguageSpace
    from Models import BasicModel
    import WalkTrials
    attend = OperationSelectionLayer.attend
    attention_loss = WalkTrials.attention_score_function
    teacher = BasicModel._decomposition_walk_teacher_loss
    policy = LanguageSpace.generate_policy_logits
    train = BasicModel._sentence_train_step
    attention, attention_batches, teacher_batches = [], [], []
    teaching = []
    step = 0
    result['attention_score_function_steps'] = []
    result['walk_teacher_steps'] = []

    def attending(module, keys, legal, space, **kwargs):
        value = attend(module, keys, legal, space, **kwargs)
        mask = kwargs.get('masked_action')
        if kwargs.get('return_details') and torch.is_tensor(mask) and bool((mask >= 0).any()):
            attention.append(dict(module=module, keys=keys.detach().clone(), legal=legal.detach().clone(),
                space=space.detach().clone(), prior=None if kwargs.get('prior') is None else kwargs['prior'].detach().clone(),
                forced=mask >= 0, logits=value[3]['logits'], actions=value[0].detach()))
        return value

    def compared(audit):
        value, record = attention_loss(audit)
        active = audit['greedy'].ge(0).any(-1)
        mean = (value * active).sum() / active.sum().clamp_min(1)
        for entry in attention:
            entry['gradient'] = (torch.autograd.grad(mean, entry['logits'], retain_graph=True,
                allow_unused=True)[0] if mean.requires_grad else None)
        attention_batches.append(dict(record=record, entries=list(attention), active=active,
            epoch=result.get('_active_epoch', 0)))
        attention.clear()
        return value, record

    def policy_logits(language, states):
        logits = policy(language, states)
        if teaching:
            teaching[-1].append(dict(language=language, states=states.detach().clone(), logits=logits))
        return logits

    def taught(model, observation, record):
        captured = []
        teaching.append(captured)
        try:
            value = teacher(model, observation, record)
        finally:
            teaching.pop()
        reports = model._last_decomposition_walk_teacher
        rows = sorted(set(r['row'] for r in reports if r['present']))
        assert len(rows) == len(captured)
        active = sum(program is not None for program in observation['entries'])
        for b, entry in zip(rows, captured):
            targets = [r['target'] for r in reports if r['row'] == b and r['present']]
            target = torch.tensor(targets, device=entry['logits'].device)
            gradient, = torch.autograd.grad(value.sum()/max(1, active), entry['logits'], retain_graph=True)
            p = entry['logits'].detach().softmax(-1)
            expected = (p-torch.nn.functional.one_hot(target, p.shape[-1]))/(len(targets)*max(1,active))
            torch.testing.assert_close(gradient, expected, atol=2e-6, rtol=2e-5)
            entry.update(row=b, targets=targets, gradient=gradient.detach(),
                gradient_max_error=float((gradient-expected).abs().max()))
        teacher_batches.append(dict(trial=model._sentence_trial, epoch=result.get('_active_epoch',0),
                                    entries=captured))
        return value

    @torch.no_grad()
    def fixed(entry):
        return attend(entry['module'], entry['keys'], entry['legal'], entry['space'],
                      prior=entry['prior'], return_details=True)[3]['logits'].detach()

    def stepped(model, loss):
        nonlocal step
        step += 1
        batches, attention_batches[:] = list(attention_batches), []
        teachers = [b for b in teacher_batches if b['trial'] == model._sentence_trial]
        teacher_batches[:] = [b for b in teacher_batches if b not in teachers]
        rng = torch.random.get_rng_state()
        for batch in batches:
            for entry in batch['entries']:
                entry['before'] = fixed(entry)
        for batch in teachers:
            for entry in batch['entries']:
                with torch.no_grad():
                    entry['before'] = policy(entry['language'], entry['states']).detach()
        output = train(model, loss)
        for batch in batches:
            record = batch['record']
            for entry in batch['entries']:
                entry['after'] = fixed(entry)
            for b in torch.nonzero(batch['active']).flatten().tolist():
                cost = record['costs'][b]
                advantage = float(record['advantage'][b])
                departed = bool(record['mask'][b].any())
                row = dict(epoch=batch['epoch'], owner_step=step, batch_row=b,
                    departed=departed, C_greedy=float(cost[0]), C_explore=float(cost[1]), advantage=advantage)
                if departed:
                    entry, = [e for e in batch['entries'] if bool(e['forced'][b])]
                    action = int(entry['actions'][b]); logits = entry['logits'][b].detach()
                    p = logits.softmax(-1)
                    scale = float(record['scale'][b][record['mask'][b]][0])
                    expected = -p*p[action]; expected[action] += p[action]
                    expected *= scale*advantage/int(batch['active'].sum())
                    actual = torch.zeros_like(expected) if entry['gradient'] is None else entry['gradient'][b]
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                    plus, minus = logits.double().clone(), logits.double().clone()
                    epsilon=1e-4; plus[action]+=epsilon; minus[action]-=epsilon
                    numeric=float(scale*advantage*(plus.softmax(-1)[action]-minus.softmax(-1)[action])/(2*epsilon*int(batch['active'].sum())))
                    count=int(torch.isfinite(logits).sum())-1
                    row.update(action=action, action_name=('divide','descend','gloss','and','or','not')[action%6],
                        round=int(torch.nonzero(record['mask'][b])[0]), sampling_scale=scale,
                        eligible_alternatives=count, eligible_rounds=int(scale/count),
                        p_forward=float(p[action]), p_before=float(entry['before'][b].softmax(-1)[action]),
                        p_after=float(entry['after'][b].softmax(-1)[action]),
                        logit_gradient=actual.cpu().tolist(), expected_gradient=expected.cpu().tolist(),
                        gradient_max_error=float((actual-expected).abs().max()),
                        finite_difference=dict(numeric=numeric, analytic=float(actual[action]),
                            epsilon=epsilon,error=abs(numeric-float(actual[action]))))
                result['attention_score_function_steps'].append(row)
        for batch in teachers:
            for entry in batch['entries']:
                with torch.no_grad():
                    after = policy(entry['language'], entry['states']).detach()
                result['walk_teacher_steps'].append(dict(epoch=batch['epoch'], owner_step=step,
                    trial=batch['trial'], batch_row=entry['row'], targets=entry['targets'],
                    logits_before=entry['before'].cpu().tolist(), logits_after=after.cpu().tolist(),
                    gradient=entry['gradient'].cpu().tolist(), gradient_max_error=entry['gradient_max_error']))
        assert torch.equal(rng, torch.random.get_rng_state()), 'observation must draw no RNG'
        return output

    with ExitStack() as stack:
        for owner, name, replacement in ((OperationSelectionLayer,'attend',attending),
            (WalkTrials,'attention_score_function',compared),
            (LanguageSpace,'generate_policy_logits',policy_logits),
            (BasicModel,'_decomposition_walk_teacher_loss',taught),
            (BasicModel,'_sentence_train_step',stepped)):
            stack.enter_context(patch.object(owner,name,replacement))
        yield


@contextmanager
def observe_run():
    with accepted_observer() as result:
        if os.environ.get('OWNERSHIP_OBSERVER_OUTPUT'):
            with observe_round1(result):
                yield result
        else:
            yield result
