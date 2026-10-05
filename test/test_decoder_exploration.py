"""Generate explores one legal departure and compares before training."""
from types import SimpleNamespace, MethodType

import pytest
import torch


def decoder(monkeypatch=None):
    from Language import LanguageSpace
    from Models import BasicModel
    if monkeypatch is not None:
        # This fixture tests the generic departure/replay kernel. Its toy
        # arithmetic has an explicit all-actions eligibility oracle; the real
        # support mask is exercised in test_review11_contracts instead.
        monkeypatch.setattr(LanguageSpace, 'decoder_eligibility', staticmethod(
            lambda parent, left, right, available, *a, **k: available))
    policy = torch.nn.Linear(2, 3)
    with torch.no_grad():
        policy.weight.zero_()
        policy.bias.copy_(torch.tensor([0., -1., 2.]))
    language = SimpleNamespace(_generate_binary_ops=(object(),),
        _generate_unary_ops=(object(),), generate_policy=policy, _generate_policy_width=2,
        _generate_binary_rule_ids=torch.tensor([0]), _generate_binary_names=('toy_split',),
        _generate_unary_rule_ids=torch.tensor([1]), _generate_unary_names=('toy_negate',))
    language.generate_policy_logits = MethodType(LanguageSpace.generate_policy_logits, language)
    language.choose_generate = lambda logits: logits.argmax(-1)
    language.reverse_binary_step = lambda parent, *a, **k: (
        parent*.25, parent*.75, torch.zeros(parent.shape[0], dtype=torch.bool))
    language.generate_unary_step = lambda parent, *a, **k: (
        -parent, torch.zeros(parent.shape[0], dtype=torch.bool))
    model = SimpleNamespace(languageSpace=language)
    model._output_generate_walk = MethodType(BasicModel._output_generate_walk, model)
    return model, policy


@pytest.mark.parametrize('compiled', [False, True])
def test_single_departure_replays_prefix_and_is_greedy_after_it(monkeypatch, compiled):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, policy = decoder(monkeypatch)
    # This prefix/suffix test has one eligible departure. Sampling among
    # several alternatives is covered separately by the review §10 probe.
    model.languageSpace.generate_unary_step = lambda parent, *a, **k: (
        -parent, torch.ones(parent.shape[0], dtype=torch.bool))
    event = torch.tensor([[[1., 2.], [3., 4.], [0., 0.]],
                          [[1., 2.], [3., 4.], [0., 0.]]], requires_grad=True)
    def pair(event):
        greedy = model._output_generate_walk(event, 8, return_trace=True,
            basis=event, return_candidates=True)
        explore = model._output_generate_walk(event, 8, return_trace=True,
            basis=event, exploit_actions=greedy[4], departure=torch.tensor([0, 1]))
        return greedy, explore
    fn = torch.compile(pair, backend='inductor', fullgraph=True) if compiled else pair
    greedy, explore = fn(event)
    assert greedy[1].tolist() == [2, 2]
    assert explore[1].tolist() == [3, 3]
    assert greedy[4][:, :2].tolist() == [[2, 2], [2, 2]]
    assert explore[4][:, :4].tolist() == [[0, 2, 2, 2], [2, 0, 2, 2]]
    assert greedy[5][:, :2].all()
    assert not greedy[5][:, 2:].any()
    explore[0].square().sum().backward()
    assert policy.bias.grad.norm() > 0
    assert event.grad.norm() > 0


def test_no_legal_departure_keeps_greedy(monkeypatch):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder(monkeypatch)
    model.languageSpace._generate_unary_ops = ()
    model.languageSpace.generate_policy_logits = lambda top: top.new_tensor([[3., 2.]]).expand(top.shape[0], -1)
    event = torch.tensor([[[1., 2.]]])
    result = model._output_generate_walk(event, 4, return_trace=True, return_candidates=True)
    assert result[1].tolist() == [1]
    assert result[4].tolist() == [[1, -1, -1, -1]]
    assert not result[5].any()


def test_trial_selects_strictly_lower_full_reconstruction_before_backward(monkeypatch):
    from Models import BasicModel
    parameter = torch.nn.Parameter(torch.tensor(1.))
    greedy_cost = torch.tensor([4., 2., 1.])
    explore_cost = torch.tensor([1., 2., 4.])
    calls = []
    record = SimpleNamespace(root=torch.ones(3, 2), word_values=torch.ones(3, 2, 2),
        roots=torch.ones(3, 1, 6), depths=torch.ones(3, 1, dtype=torch.long),
        end_slots=torch.ones(3, 3, 2), end_depth=torch.ones(3, dtype=torch.long),
        sentence=torch.tensor(0), primed=None)
    def reconstruct(*args, **kwargs):
        assert parameter.item() == 1.
        assert parameter.grad is None
        explore = kwargs.get('decoder_exploit') is not None
        calls.append(explore)
        cost = (explore_cost if explore else greedy_cost) * parameter
        value = torch.full((3, 2, 2), float(explore)) * parameter
        trace = (value, torch.full((3,), 2), torch.zeros(3, dtype=torch.bool),
                 torch.zeros(3), torch.full((3, 4), int(explore)))
        if kwargs.get('decoder_candidates'):
            trace = (*trace, torch.ones(3, 4, dtype=torch.bool))
        return (value, cost*0, cost, torch.zeros(3, dtype=torch.bool), cost[:, None]), trace
    model = SimpleNamespace(_sentence_training=True, _compiled_reconstruct=lambda: reconstruct)
    result = BasicModel._reconstruct_trial(model, record)
    assert calls == [False, True]
    assert model._last_decoder_comparison['wins'].tolist() == [True, False, False]
    torch.testing.assert_close(result[2], torch.tensor([1., 2., 1.]))
    assert model._last_decoder_trace[4][:, 0].tolist() == [1, 0, 0]
    result[2].sum().backward()
    assert parameter.grad.item() == 4.
