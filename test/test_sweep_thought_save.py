"""Repeated whole-path reads retain their live pre-update cotangent."""
import torch


def test_repeated_reader_uses_one_parameter_and_updates_after_backward():
    from Layers import Error
    from WholePaths import objectives
    from types import SimpleNamespace
    weight = torch.nn.Parameter(torch.arange(36, dtype=torch.float64).reshape(6,6)/36)
    scale = torch.linspace(.5,1.,6,dtype=torch.float64)
    source = torch.linspace(.1,.6,6,dtype=torch.float64)[None].requires_grad_()
    value = sum((source @ weight * scale).square().sum(-1) for _ in range(16))
    expected = torch.autograd.grad(value.sum(), (weight,source), retain_graph=True)
    registry = Error(row_mask=torch.ones(1,dtype=torch.bool))
    registry.add('answer.what', value, category='prediction', objective='output')
    paths = [SimpleNamespace(registry=registry), SimpleNamespace(registry=registry)]
    costs, exposure = objectives(paths, torch.zeros(1,2,3), torch.ones(1,dtype=torch.bool),
        [dict(compose_round=torch.tensor([0]))])
    assert exposure.tolist() == [[.5,.5]]
    actual = torch.autograd.grad(costs['output'], (weight,source))
    for gradient, wanted in zip(actual, expected):
        torch.testing.assert_close(gradient, wanted, rtol=0, atol=0)
