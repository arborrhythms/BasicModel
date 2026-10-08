"""Owner-step cost credit for the grammar's one-departure thought walk."""
from dataclasses import dataclass
import torch


def surrogate(trace, other, departure, costs):
    """The same K R p(a_dep) ΔC as compose; an exact tie has no graph."""
    if departure < 0 or departure >= len(other.get('choices', ())):
        return None
    choice = other['choices'][departure]
    probability = choice['probability']
    delta = probability.new_tensor(float(costs[1])-float(costs[0]))
    if not bool(delta != 0):
        return None
    return choice['alternatives'] * sum(trace['eligible']) * probability * delta


def register(model, value, *, costs, source):
    model._last_thought_score_function = dict(costs=tuple(costs), source=source,
                                             surrogate=value)
    if value is not None and value.requires_grad:
        errors = getattr(model, 'errors', None)
        if errors is not None:
            errors.add('reconstruction.thought_score_function', value,
                       space='SymbolSpace', category='reconstruction')


@torch.no_grad()
def forecast(model, meaning, row):
    """Hold the existing predictor's prior with this conclusion as context."""
    discourse = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
    if (discourse is None or not discourse.expectation_enabled
            or discourse.expectation_scope != 'structured'
            or discourse._inter_predictor is None):
        return None
    from Layers import MeaningExpectation
    predictor = discourse._inter_predictor
    parameter = next(predictor.parameters())
    chain = list(discourse._inter_context[row])
    from ThoughtReferences import open_slots
    if not open_slots(meaning):
        chain.append((int(meaning.role_mask.sum()), meaning.roles.detach(), meaning.role_mask))
    chain = chain[-discourse._inter_chain_window:]
    pad = discourse._inter_chain_window-len(chain)
    roles = [parameter.new_zeros(3, discourse.concept_dim)]*pad + [r.to(parameter) for _, r, _ in chain]
    masks = [torch.zeros(3, device=parameter.device, dtype=torch.bool)]*pad + [m.to(parameter.device) for _, _, m in chain]
    values, presence, kind = predictor(*discourse._prediction_inputs(
        (torch.stack(roles)[None], torch.stack(masks)[None])))
    return MeaningExpectation(values[0], presence[0], kind_logit=kind[0]).detached()


@dataclass
class PendingCredit:
    trace: dict
    other: dict
    departure: int
    costs: tuple
    forecasts: tuple
    document: object
    gains: tuple = (1., 1.)


def complete(model, greedy, explore, trace, other, departure, costs, row, *, forecasts=None, gains=(1.,1.)):
    from Occurrence import source_at
    if forecasts is None:
        forecasts = (forecast(model, greedy.meaning, row), forecast(model, explore.meaning, row))
    if all(value is not None for value in forecasts):
        document = source_at(model, row, int(getattr(model, '_open_sentence_slot', 0) or 0))[0]
        pending = model.__dict__.setdefault('_pending_thought_credit', {})
        pending[row] = PendingCredit(trace, other, departure, tuple(costs), forecasts, document, gains)
    else:
        register(model, surrogate(trace, other, departure, costs), costs=costs,
                 source='answer_and_work')


def observe(model, meanings, *, sentence):
    """Only the loss side sees the next sentence; no query or choice is rerun."""
    from Layers import Error
    from Occurrence import source_at
    from SentenceCredit import expectation_terms
    pending = model.__dict__.get('_pending_thought_credit', {})
    for row, meaning in enumerate(meanings):
        if meaning is None or row not in pending:
            continue
        held = pending.pop(row)
        if source_at(model, row, sentence)[0] != held.document:
            # No cross-document target. Preserve the available answer/work credit.
            costs = held.costs
        else:
            costs = []
            for prior, cost, gain in zip(held.forecasts, held.costs, held.gains):
                errors = Error()
                expectation_terms(errors, prior.roles, prior.presence_logits,
                    prior.kind_logit, meaning.roles, meaning.role_mask, meaning.sentence_kind,
                    row=None, gain=gain)
                costs.append(cost + float(errors.total().detach()) * getattr(model, 'inter_loss_weight', 1.))
        register(model, surrogate(held.trace, held.other, held.departure, costs),
                 costs=costs, source='answer_expectation_and_work')


def finish_documents(model, rows):
    """A final question still receives its available answer and work credit."""
    pending = model.__dict__.get('_pending_thought_credit', {})
    for row in rows:
        held = pending.pop(row, None)
        if held is not None:
            register(model, surrogate(held.trace, held.other, held.departure, held.costs),
                     costs=held.costs, source='document_end_answer_and_work')
