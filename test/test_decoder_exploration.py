"""Generate explores one legal departure and compares before training."""
from types import SimpleNamespace, MethodType

import pytest
import torch


def decoder(monkeypatch=None):
    from Language import LanguageSpace,GeneratePolicy
    from Models import BasicModel
    if monkeypatch is not None:
        # This fixture tests the generic departure/replay kernel. Its toy
        # arithmetic has an explicit all-actions eligibility oracle; the real
        # support mask is exercised in test_review11_contracts instead.
        from generation_fixtures import all_available
        import Generative
        monkeypatch.setattr(LanguageSpace,'decoder_eligibility',staticmethod(all_available))
        monkeypatch.setattr(Generative,'selected_residual',lambda lang,parent,*a,**k:parent.new_zeros(len(parent),3))
    policy = GeneratePolicy(2, 3)
    with torch.no_grad():
        policy.weight.zero_()
        policy.bias.copy_(torch.tensor([0., -1., 2.]))
    language = SimpleNamespace(_tree_layer=lambda arity: SimpleNamespace(exploration_fraction=.5, exploration_temperature=1.), _generate_binary_ops=(SimpleNamespace(meaning_layout=None),),
        _generate_unary_ops=(SimpleNamespace(meaning_layout=None),), generate_policy=policy, _generate_policy_width=2,
        _generate_binary_rule_ids=torch.tensor([0]), _generate_binary_names=('toy_split',),
        _generate_unary_rule_ids=torch.tensor([1]), _generate_unary_names=('toy_negate',))
    from generation_fixtures import undeclared_arguments
    undeclared_arguments(language)
    language.generate_policy_logits = MethodType(LanguageSpace.generate_policy_logits, language)
    language.choose_generate = lambda logits: logits.argmax(-1)
    def binary(parent, *args, return_status=False, **kwargs):
        pair = (parent*.25, parent*.75)
        return (*pair, torch.zeros(parent.shape[0], dtype=torch.bool)) if return_status else pair
    def unary(parent, *args, return_status=False, **kwargs):
        return (-parent, torch.zeros(parent.shape[0], dtype=torch.bool)) if return_status else -parent
    language.reverse_binary_step = binary
    language.generate_unary_step = unary
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
    model.languageSpace.generate_unary_step = lambda parent, *a, return_status=False, **k: (
        (-parent, torch.ones(parent.shape[0], dtype=torch.bool)) if return_status else -parent)
    event = torch.tensor([[[1., 2.], [3., 4.], [0., 0.]],
                          [[1., 2.], [3., 4.], [0., 0.]]], requires_grad=True)
    def pair(event):
        greedy = model._output_generate_walk(event, 8, return_trace=True,
            basis=event, return_candidates=True)
        explore = model._output_generate_walk(event, 8, return_trace=True,
            basis=event, exploit_actions=greedy[4], exploit_prefix=greedy[6], departure=torch.tensor([0, 1]))
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
    assert policy.bias.grad is None or not policy.bias.grad.any()  # compiled AD may materialize zero
    assert event.grad.norm() > 0


def test_no_legal_departure_keeps_greedy(monkeypatch):
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _ = decoder(monkeypatch)
    model.languageSpace._generate_unary_ops = ()
    model.languageSpace.generate_policy_logits = lambda top,features=None: top.new_tensor([[3., 2.]]).expand(top.shape[0], -1)
    event = torch.tensor([[[1., 2.]]])
    result = model._output_generate_walk(event, 4, return_trace=True, return_candidates=True)
    assert result[1].tolist() == [1]
    assert result[4].tolist() == [[1, -1, -1, -1]]
    assert not result[5].any()
