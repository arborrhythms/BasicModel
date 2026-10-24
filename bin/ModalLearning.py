"""The second difference: a modifier maps equal-time events across timelines."""
from dataclasses import replace

import torch

from MediumTermMemory import Timeline


def teacher(action, actual, alternative, modifiers):
    from Layers import cube_chart
    actual, alternative = actual.detach(), alternative.detach()
    predicted = actual
    for modifier in modifiers:
        predicted = action(predicted, modifier.detach())
    defined = actual.ne(0) & alternative.ne(0)
    error = torch.where(defined, (cube_chart(predicted)-cube_chart(alternative)).square(), 0.)
    return error.sum(-1) / defined.sum(-1).clamp_min(1)


def observe(action, pool, references, *, errors=None, weight=.1, training=True, deferred=None):
    reports = []
    if action is None:
        return reports
    width = action.readout.shape[-1]
    tokens = dict.fromkeys(pool.token_of(reference) for reference in references)
    for key in tokens:
        entry = pool.entries.get(key)
        if entry is None:
            continue
        timelines = list(entry.timelines)
        for index, timeline in enumerate(timelines[1:], 1):
            events = []
            for event in timeline.events:
                actual = next((other for other in reversed(entry.events)
                    if other.role == 0 and other.when == event.when and other.after is not None), None)
                if (event.role != 0 or event.modal_taught or not event.modifiers
                        or event.after is None or actual is None):
                    events.append(event)
                    continue
                before, after = (value[-width:].to(action.readout).detach()
                                 for value in (actual.after, event.after))
                modifiers = tuple(value[-width:].to(action.readout).detach() for value in event.modifiers)
                with torch.set_grad_enabled(training and deferred is None):
                    term = teacher(action, before, after, modifiers)
                if training and deferred is not None:
                    deferred.append((action, before, after, modifiers, weight))
                elif training and errors is not None:
                    errors.add('modal_teacher', term, weight=weight, category='reconstruction')
                reports.append(dict(token=key, timeline=index, occurrence=event.occurrence,
                    actual=actual.occurrence, when=event.when, error=float(term.detach()),
                    defined=int((before.ne(0) & after.ne(0)).sum())))
                events.append(replace(event, modal_taught=True))
            timelines[index] = Timeline(tuple(events), timeline.value)
        pool.entries[key] = replace(entry, timelines=tuple(timelines))
    return reports


def register(records, errors):
    """Construct graphs at the post-pair parameter version, as for the verb."""
    for action, actual, alternative, modifiers, weight in records:
        errors.add('modal_teacher', teacher(action, actual, alternative, modifiers),
                   weight=weight, category='reconstruction')
    records.clear()
