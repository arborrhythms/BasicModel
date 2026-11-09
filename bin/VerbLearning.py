"""Across-close residuals assign action groups; participation trains the action."""
from dataclasses import replace

import torch


def operands(action, event):
    width = action.readout.shape[-1]
    before = event.before[-width:].to(action.readout).detach()
    verb = event.verb_meaning[-width:].to(action.readout).detach()
    basis = (dict(object_value=event.basis[-width:].to(before).detach()) if torch.is_tensor(event.basis)
             else dict(group=torch.tensor(-1 if event.basis is None else event.basis, device=before.device)))
    return before, verb, basis


def reread(owner, token, timeline, value):
    """Stored forward value, with only the current action's local pullback."""
    if not torch.is_grad_enabled():
        return value
    event = next((e for e in reversed(token.timelines[timeline].events) if e.role == 0), None)
    language = getattr(getattr(owner, '_model', None), 'languageSpace', None)
    action = None if language is None else language.verb_action()
    if event is None or event.before is None or event.verb_meaning is None or action is None:
        return value
    before, verb, basis = operands(action, event)
    predicted = action(before, verb, **basis)
    if timeline and event.modifiers:
        modifier = language.modifier_action()
        if modifier is None:
            raise RuntimeError('a recorded alternative requires its modifier action')
        width = modifier.readout.shape[-1]
        for operand in event.modifiers:
            predicted = modifier(predicted, operand[-width:].to(predicted).detach())
    # Subtract first: v + predicted - predicted.detach() can round v.
    correction = predicted - predicted.detach()
    return torch.cat((value[:-predicted.numel()], value[-predicted.numel():] + correction.to(value)))


def observe(action, mtm, previous, observed, *, training=True):
    reports = []
    if action is None:
        return reports
    width = action.readout.shape[-1]
    for token in observed:
        prior, current = previous.get(token), mtm.entries.get(token)
        if prior is None or current is None:
            continue
        event = next((event for event in reversed(prior.events) if event.role == 0 and event.before is not None), None)
        if event is None or event.taught:
            continue
        verb = event.verb_meaning
        if verb is None:
            continue
        before, verb, context = operands(action, event)
        after = current.value[-width:].to(action.readout).detach()
        with torch.no_grad():
            term = action.teacher(before, after, verb, **context)
        report = dict(token=token, occurrence=event.occurrence, verb=event.verb,
                      error=float(term.detach()), defined=int((before.ne(0) & after.ne(0)).sum()))
        if training and not torch.is_tensor(event.basis):
            with torch.no_grad():
                residuals = [action.teacher(before, after, verb,
                    group=torch.tensor(group, device=before.device)).item() for group in range(action.groups.capacity)]
            report['fit'] = action.groups.observe(event.verb, event.occurrence, residuals, float(term.detach()))
        mtm.entries[token] = current.with_actual(events=tuple(
            replace(item, taught=True) if (item.occurrence, item.role) == (event.occurrence, event.role) else item
            for item in current.events))
        reports.append(report)
    return reports
