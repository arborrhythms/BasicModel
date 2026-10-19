"""Owner-step cost credit for the grammar's one-departure thought walk."""
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
            or discourse.expectation_scope != 'symbols'
            or discourse._inter_predictor is None):
        return None
    chain = list(discourse._inter_context[row])
    from ThoughtReferences import open_slots
    if not open_slots(meaning):
        chain.append((int(meaning.role_mask.sum()), meaning.roles.detach(), meaning.role_mask))
    prediction = discourse.forecast_chain(chain[-discourse._inter_chain_window:])
    return None if prediction is None else prediction.detached()



def complete(model, greedy, explore, trace, other, departure, costs, row, **_unused):
    """Credit the episode now; prediction stays on its own owner registry."""
    register(model, surrogate(trace, other, departure, costs), costs=costs,
             source='reconstruction_and_answer')


def observe(model, meanings, *, sentence):
    """No deferred policy comparison: ordinary observation trains prediction."""


def finish_documents(model, rows):
    """Credit is complete at the closing, including the last document row."""
