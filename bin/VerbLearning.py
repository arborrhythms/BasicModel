"""Across-close observations train the action, never the retained concept rows."""
from dataclasses import replace

import torch


def observe(action, mtm, previous, observed, point, *, errors=None, weight=.1, training=True, deferred=None):
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
        verb = point(event.verb)
        if verb is None:
            continue
        before = event.before[-width:].to(action.readout).detach()
        after = current.value[-width:].to(action.readout).detach()
        verb = verb[-width:].to(action.readout).detach()
        context = (dict(object_value=event.basis[-width:].to(before).detach()) if torch.is_tensor(event.basis)
                   else dict(group=torch.tensor(-1 if event.basis is None else event.basis, device=before.device)))
        with torch.set_grad_enabled(training and deferred is None):
            term = action.teacher(before, after, verb, **context)
        if training and deferred is not None:
            deferred.append((action, before, after, verb, context, weight))
        elif training and errors is not None:
            errors.add('verb_teacher', term, weight=weight, category='reconstruction')
        report = dict(token=token, occurrence=event.occurrence, verb=event.verb,
                      error=float(term.detach()), defined=int((before.ne(0) & after.ne(0)).sum()))
        if not torch.is_tensor(event.basis):
            with torch.no_grad():
                residuals = [action.teacher(before, after, verb,
                    group=torch.tensor(group, device=before.device)).item() for group in range(action.groups.capacity)]
            report['fit'] = action.groups.observe(event.verb, event.occurrence, residuals, float(term.detach()))
        mtm.entries[token] = current.with_actual(events=tuple(
            replace(item, taught=True) if (item.occurrence, item.role) == (event.occurrence, event.role) else item
            for item in current.events))
        reports.append(report)
    return reports


def register(records, errors):
    """Build graphs after the batch's paired steps, at one parameter version.

    Kept-close admission and once-only marking already happened. Delaying
    this numerical evaluation to the existing batch registry adds no update
    and retains no graph across the paired optimizer steps.
    """
    for action, before, after, verb, context, weight in records:
        errors.add('verb_teacher', action.teacher(before, after, verb, **context),
                   weight=weight, category='reconstruction')
    records.clear()
