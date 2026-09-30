"""September 27 review: sampling, model credit, evaluation and winning state."""
import pytest
import torch

from Language import OperationSelectionLayer
from test_compose_operations import Add, Negate, layer


def test_zero_temperature_is_valid_and_keeps_the_chooser_gradient():
    step = OperationSelectionLayer(d_model=1, ops=[Add()], unary_ops=[Negate()],
                                   temperature=0)
    with torch.no_grad():
        step.reduce_anchor.fill_(1)
        step.apply_anchor.zero_()
    x = torch.tensor([[[1.], [2.], [3.]]], requires_grad=True)
    _, path, route = step(x, sample=True)
    path.sum().backward()
    assert route['action'].tolist() == [1]
    assert 0 < route['probability'].item() < 1
    assert step.reduce_anchor.grad.abs().sum() > 0
    assert step.apply_anchor.grad.abs().sum() > 0


def test_temperature_and_forced_mask_do_not_change_model_credit():
    step = layer()
    x = torch.tensor([[[1.], [2.], [3.]]])
    expected = torch.tensor([[3., 5., 0., 0., 0., -torch.inf]]).softmax(-1)
    for temperature in (0., .25, 2.):
        step.temperature = temperature
        _, _, route = step(x, masked_action=torch.tensor([1]))
        assert route['action'].tolist() == [0]
        torch.testing.assert_close(route['probabilities'], expected)
        torch.testing.assert_close(route['probability'], expected[:, 0])


def test_zero_temperature_explore_is_identical_until_the_forced_round(monkeypatch):
    step = layer()
    step.temperature = 0.
    x = torch.tensor([[[1.], [2.], [3.]]]).expand(3, -1, -1)
    exploit = step.derive(x, slots=1, rounds=6)
    # Exploit's optimizer step can change the preferred action. Its prefix
    # must still be replayed when exploring under the updated parameters.
    with torch.no_grad():
        step.reduce_anchor.fill_(-2.)
    monkeypatch.setattr(torch, 'rand', lambda *a, **kw:
                        torch.tensor([.0, .34, .99], device=kw.get('device')))
    explore = step.derive(x, slots=1, rounds=6, exploit=exploit)
    assert explore['forced_round'].tolist() == [0, 1, 2]
    for b, forced in enumerate(explore['forced_round'].tolist()):
        assert torch.equal(explore['actions'][b, :forced], exploit['actions'][b, :forced])
        assert explore['actions'][b, forced] != exploit['actions'][b, forced]


def test_evaluation_ignores_sampling_temperature_and_does_not_explore(monkeypatch):
    step = layer().eval()
    step.temperature = 3.
    def forbidden(*args, **kwargs):
        raise AssertionError('evaluation requested a random draw')
    monkeypatch.setattr(torch, 'rand', forbidden)
    monkeypatch.setattr(torch, 'rand_like', forbidden)
    x = torch.tensor([[[1.], [2.], [3.]]])
    first, unused = step.derive_pair(x, slots=1, rounds=6)
    second = step.derive(x, slots=1, rounds=6)
    assert unused is None
    assert torch.equal(first['actions'], second['actions'])


@pytest.mark.parametrize('temperature', [-1., float('inf'), float('nan')])
def test_sampling_temperature_must_be_finite_and_nonnegative(temperature):
    with pytest.raises(ValueError, match='temperature'):
        OperationSelectionLayer(d_model=1, ops=[Add()], temperature=temperature)


def test_sampling_temperature_changes_only_the_hard_draw(monkeypatch):
    step = layer()
    x = torch.tensor([[[1.], [2.], [3.]]])
    noise = torch.tensor([[3., 0., 0., 0., 0., 0.]])
    monkeypatch.setattr(torch, 'rand_like', lambda like:
                        torch.exp(-torch.exp(-noise.to(like))))
    step.temperature = .1
    _, _, cold = step(x, sample=True)
    step.temperature = 1.
    _, _, warm = step(x, sample=True)
    assert cold['action'].tolist() == [1]
    assert warm['action'].tolist() == [0]
    torch.testing.assert_close(cold['probabilities'], warm['probabilities'])


def test_sampling_temperature_is_read_from_the_shared_xml_element():
    from test_compose_pair_driver import _build
    model = _build('<composeTemperature>0.7</composeTemperature>')
    try:
        assert model.symbolSpace.languageLayer.operation_layer.temperature == .7
        assert model.languageSpace._tree_layer(2) is model.symbolSpace.languageLayer.operation_layer
    finally:
        model.End()
        model.symbolSpace.soft_reset()


def test_zero_temperature_prefix_uses_execution_order_for_packed_ends():
    from types import SimpleNamespace as NS
    from Models import BasicModel
    # Two one-word sentences: first closing is group 1, final closing is group 0.
    actions = torch.full((1, 14), -1, dtype=torch.long)
    actions[0, [0, 10, 3, 6]] = 2
    forced = torch.zeros_like(actions, dtype=torch.bool)
    forced[0, [10, 6]] = True
    model = NS(inputSpace=NS(_word_active_mask=torch.ones(1, 2, dtype=torch.bool),
                            _packed_sentence_ids=torch.tensor([[0, 1]])),
               conceptualSpace=NS(stm=NS(capacity=2)))
    owners, _, _ = BasicModel._compose_round_owners(model, actions)
    prefix = BasicModel._exploration_prefix_slots(model, actions, forced, owners)
    assert prefix.nonzero().tolist() == [[0, 0], [0, 3]]


@pytest.mark.parametrize('explore_loss, expected', [(1., True), (2., False), (3., False)])
def test_only_a_strictly_lower_loss_commits_the_alternative(explore_loss, expected):
    from SentenceCompose import sentence_pair
    counter = []
    def compose(cache, prior):
        return torch.tensor([1. if prior is None else 10.])
    def score(path, alternative):
        return torch.tensor([explore_loss if alternative else 2.]), path
    state, costs, wins = sentence_pair(None, compose, score, lambda loss: counter.append(loss),
        active=torch.tensor([True]))
    assert len(counter) == 2
    assert wins.item() is expected
    assert state.item() == (10 if expected else 1)
