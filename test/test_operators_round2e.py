"""Whole paths share one answer reader and expose the compose alternatives."""
import torch
from torch import nn


def test_answer_exposure_requires_its_departure_draw():
    import pytest
    from SentenceCredit import reader_weights
    with pytest.raises(ValueError, match='departure draw'):
        reader_weights(torch.zeros(1, 2, 3), torch.ones(1, dtype=torch.bool), None)


def test_one_reader_shares_compose_exposure_and_steps_once():
    from Layers import Error
    from SentenceCredit import reader_weights, reader_costs
    from ObjectiveOwnership import backward_owned
    active = torch.tensor([True, True, True, False])
    parts = torch.zeros(4, 2, 3)
    parts[1, 0, 0] = 1.
    weights = reader_weights(parts, active, dict(compose_round=torch.tensor([0, 0, -1, -1])))
    torch.testing.assert_close(weights, torch.tensor([[.5,.5],[.5,.5],[1.,0.],[0.,0.]]))
    parameter = nn.Parameter(torch.tensor(1.))
    roots = [torch.tensor([1.,2.,3.,4.], requires_grad=True), torch.tensor([9.,8.,7.,6.], requires_grad=True)]
    registries = []
    for root in roots:
        registry = Error(row_mask=active)
        registry.error('answer', (parameter * root).square(), 1., objective='output')
        registries.append(registry)
    cost = reader_costs(registries, weights)['output']
    gradients = torch.autograd.grad(cost, (parameter, *roots), retain_graph=True)
    torch.testing.assert_close(gradients[0], 2*((1+81)/2+(4+64)/2+9)/torch.tensor(3.))
    assert gradients[2][2:].tolist() == [0., 0.]
    assert bool(gradients[1][:3].abs().all()) and bool(gradients[2][:2].abs().all())
    optimizer = torch.optim.Adam([parameter], lr=.01)
    backward_owned({'output': cost}, {'output': (parameter,)})
    optimizer.step()
    assert optimizer.state[parameter]['step'] == 1


def test_sentence_advantage_and_optimizer_use_the_same_reader(monkeypatch):
    from test_supplied_answer_training import test_real_supplied_paths_cost_before_updates_and_reach_understanding
    test_real_supplied_paths_cost_before_updates_and_reach_understanding(monkeypatch)
