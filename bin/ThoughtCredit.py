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


def register(model, value, *, costs, source, row):
    model._last_thought_score_function = dict(costs=tuple(costs), source=source,
                                             surrogate=value, row=row)
    if value is not None and value.requires_grad:
        errors = getattr(model, 'errors', None)
        if errors is not None:
            errors.add('reconstruction.thought_score_function', value,
                       space='SymbolSpace', category='reconstruction')


def complete(model, trace, other, departure, costs, row):
    """Credit the episode now; prediction stays on its own owner registry."""
    register(model, surrogate(trace, other, departure, costs), costs=costs,
             source='reconstruction_and_answer', row=row)
