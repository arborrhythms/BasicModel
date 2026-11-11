"""Direct output-chain, checkpoint and side/owner certificates for round 6."""
from types import SimpleNamespace as NS
from types import MethodType
import pytest
import torch

from test_output_synthesis import loop_model, _loop_understanding, _loop_answer


def test_output_calls_the_common_decoder_and_reports_retired_checkpoint_keys(loop_model, monkeypatch):
    model = loop_model
    understanding = _loop_understanding(model)
    calls = []
    native = model.reverse
    def reverse(carrier, *, field):
        result = native(carrier, field=field)
        calls.append((field, result))
        return result
    monkeypatch.setattr(model, 'reverse', reverse)
    answer = _loop_answer(model, understanding)
    assert calls and calls[-1][0].percepts is None
    torch.testing.assert_close(answer.surface, calls[-1][1].materialize(), rtol=0, atol=0)
    torch.testing.assert_close(answer.percepts, calls[-1][1]._reconstruction_percepts, rtol=0, atol=0)
    assert [entry['operation'] for entry in answer.trace[-3:]] == [
        'generate:grammar', 'generate:occurrences', 'reverse:order_zero']
    assert not hasattr(model, '_answer_surface_from_percepts')
    assert all(not hasattr(space, 'synthesis_layer') for space in (model.conceptualSpace, model.perceptualSpace))
    state = model.state_dict()
    retired = ('conceptualSpace.synthesis_layer.raw_L', 'perceptualSpace.synthesis_layer.d')
    for name in retired:
        state[name] = torch.ones(1)
    with pytest.warns(UserWarning, match='retired answer synthesis checkpoint keys') as reports:
        model._materialize_answer_path_from_checkpoint(state)
    assert set(model._retired_answer_checkpoint_keys) == set(retired)
    assert all(name in str(reports[0].message) and name not in state for name in retired)
    model.load_state_dict(state, strict=True)


def test_output_departure_credit_reaches_generate_policy_on_the_answer_side():
    from Language import GeneratePolicy, LanguageSpace
    from Generative import WALK_FEATURE_WIDTH
    from WalkTrials import generate_credit
    from Layers import Error
    from ObjectiveOwnership import registry_costs
    from GradientProjection import sides
    policy = GeneratePolicy(2, 3)
    reader = torch.nn.Linear(2, 2)
    language = NS(generate_policy=policy, _generate_policy_width=2,
        _tree_layer=lambda arity: NS(exploration_fraction=.5, exploration_temperature=1.))
    language.generate_policy_logits = MethodType(LanguageSpace.generate_policy_logits, language)
    audit = dict(departure=torch.tensor([0]), greedy=torch.tensor([[0]]),
        explore=torch.tensor([[1]]), eligible=torch.tensor([[True]]), costs=torch.tensor([[.3, .8]]))
    credit = generate_credit(language, audit, reader(torch.ones(1, 1, 2)),
        torch.ones(1, 1, 3, WALK_FEATURE_WIDTH))
    errors = Error()
    errors.add('output.walk_credit', credit, category='output', objective='output', side='A')
    costs = registry_costs(errors)
    assert set(costs) == {'output'}
    split = sides(costs, .5)
    assert split['R'] is None
    split['A'].backward()
    assert policy.weight.grad is not None and policy.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reader.parameters())
