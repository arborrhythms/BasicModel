"""September 27 review: sampling, model credit, evaluation and winning state."""
import pytest
import torch

from Language import OperationSelectionLayer
from test_compose_operations import Add, Negate, layer


def test_zero_temperature_keeps_the_score_function_gradient():
    step = OperationSelectionLayer(d_model=1, ops=[Add()], unary_ops=[Negate()],
                                   temperature=0)
    with torch.no_grad():
        step.reduce_anchor.fill_(1)
        step.apply_anchor.zero_()
    x = torch.tensor([[[1.], [2.], [3.]]], requires_grad=True)
    _, path, route = step(x, sample=True)
    path.sum().backward()
    assert step.reduce_anchor.grad is None
    assert step.apply_anchor.grad is None
    (-route['probability'].sum()).backward()
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
        assert route['departure_eligible'].gather(1, route['action'][:, None]).all()
        torch.testing.assert_close(route['probabilities'], expected)
        torch.testing.assert_close(route['probability'], expected.gather(1, route['action'][:, None]).squeeze(1))


def test_zero_temperature_explore_joins_detached_state_at_its_departure(monkeypatch):
    step = layer()
    step.temperature = 0.
    x = torch.tensor([[[1.], [2.], [3.]]]).expand(3, -1, -1)
    draws = iter(([0., 0., 0.], [.9, .1, .9], [.9, .9, .1]))
    monkeypatch.setattr(torch, 'rand_like', lambda like, **kw:
                        torch.tensor(next(draws), device=like.device))
    exploit = step.derive(x, slots=1, rounds=6)
    assert not exploit['fork'].state[0].requires_grad
    live = []
    forward = step.forward
    def observed(value, **kwargs):
        live.append(kwargs['active'].clone())
        return forward(value, **kwargs)
    monkeypatch.setattr(step, 'forward', observed)
    explore = step.derive(x, slots=1, rounds=6, exploit=exploit)
    assert explore['forced_round'].tolist() == [0, 1, 2]
    assert [row.tolist() for row in live[:3]] == [[True, False, False],
                                                [True, True, False],
                                                [True, True, True]]
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


def test_fork_joins_by_saved_slot_when_packed_end_slots_are_out_of_order():
    from types import SimpleNamespace as NS
    from SentenceFork import SentenceFork
    # Two one-word sentences: first closing is group 1, final closing is group 0.
    fork = SentenceFork(torch.tensor([True, True]))
    fork.slot = torch.tensor([10, 6])
    fork.state = (torch.tensor([[7.], [9.]]),)
    fork.start(dict(compose_round=fork.slot, narrowing=torch.tensor([False, False])))
    state = (torch.zeros(2, 1),)
    seen = []
    for slot in (0, 10, 3, 6):
        state = fork.resume(torch.tensor(slot), state)
        seen.append(state[0].flatten().tolist())
    assert seen == [[0., 0.], [7., 0.], [7., 0.], [7., 9.]]


def test_fork_cannot_resume_on_another_rows_read_of_its_saved_word():
    from SentenceFork import SentenceFork
    fork = SentenceFork(torch.tensor([True, True]))
    fork.slot = torch.tensor([3, 3])
    fork.state = (torch.tensor([[7.], [9.]]),)
    fork.start(dict(compose_round=fork.slot, narrowing=torch.tensor([False, False])))
    fork.current_rows = torch.tensor([True, False])
    state = fork.resume(torch.tensor(3), (torch.zeros(2, 1),))
    assert state[0].flatten().tolist() == [7., 0.]
    assert fork.joined.tolist() == [True, False]
    fork.current_rows = torch.tensor([False, True])
    state = fork.resume(torch.tensor(3), state)
    assert state[0].flatten().tolist() == [7., 9.]


def test_separate_closing_fork_waits_for_the_closing_visit():
    from SentenceFork import SentenceFork
    fork = SentenceFork(torch.tensor([True]))
    fork.word, fork.slot = torch.tensor([1]), torch.tensor([24])
    fork.start(dict(compose_round=fork.slot, narrowing=torch.tensor([False])))
    fork.phase = 'word'
    assert not fork.pending(torch.tensor(1), closing=True, width=8)
    fork.phase = 'closing'
    assert fork.pending(torch.tensor(1), closing=True, width=8)
    assert not fork.pending(torch.tensor(1), closing=False, width=8)


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
