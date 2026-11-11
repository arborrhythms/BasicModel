"""Action errors over a later reading, with the no-change prediction as baseline."""
import torch


def distance(predicted, target, defined_by):
    from Layers import cube_chart
    defined = defined_by.ne(0) & target.ne(0)
    residual = torch.where(defined, (cube_chart(predicted)-cube_chart(target)).square(), 0.)
    return residual.sum(-1)/defined.sum(-1).clamp_min(1)


def verb(action, event, target, errors, *, row=None):
    from VerbLearning import operands
    with torch.enable_grad():
        before, phrase, context = operands(action, event)
        target = target[-before.numel():].to(before).detach()
        error = distance(action(before, phrase, **context), target, before)
        baseline = distance(before, target, before)
        errors.error('memory.verb_state', error, baseline, row=row if errors.row_mask is not None else None,
            category='reconstruction', objective='reconstruction', side='R')
    return error, baseline


def modal(action, actual, alternative, modifiers, errors, *, row=None):
    from ModalLearning import residual
    width = action.readout.shape[-1]
    actual, alternative = (value[-width:].to(action.readout).detach() for value in (actual, alternative))
    modifiers = tuple(value[-width:].to(actual).detach() for value in modifiers)
    with torch.enable_grad():
        error = residual(action, actual, alternative, modifiers)
        baseline = distance(actual, alternative, actual)
        errors.error('memory.modal_state', error, baseline, row=row if errors.row_mask is not None else None,
            category='reconstruction', objective='reconstruction', side='R')
    return error, baseline
