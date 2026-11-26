"""Retained rows and policy credit have one tensor operation graph per batch."""
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires Metal')
    return torch.device(request.param)


@pytest.mark.parametrize('batch_dim', [0, 1])
def test_stream_selection_preserves_values_and_gradients_with_flat_graph(device, batch_dim):
    from SentenceCompose import select_streams
    operations = []
    def backend(graph, inputs):
        operations.append(sum(node.op == 'call_function' for node in graph.graph.nodes))
        return graph.forward
    for batch in (4, 16):
        values = [torch.randn(batch, 5, 3, device=device).movedim(0, batch_dim).requires_grad_()
                  for _ in range(3)]
        winners = torch.arange(batch, device=device) % 3
        function = torch.compile(lambda values, winners: select_streams(values, winners, batch_dim=batch_dim),
                                 backend=backend, fullgraph=True, dynamic=False)
        selected = function(values, winners)
        expected = torch.stack([values[int(winners[row])].movedim(batch_dim, 0)[row]
                                for row in range(batch)]).movedim(0, batch_dim)
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(selected.square().sum(), values, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().sum(), values)
        for actual, wanted in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert len(operations) == 2 and operations[0] == operations[1]


@pytest.mark.parametrize('batch', [4, 16])
@pytest.mark.parametrize('tie', [False, True])
def test_thought_credit_matches_serial_values_and_gradients_in_one_call(device, batch, tie, monkeypatch):
    from Layers import Error
    import SentenceCredit
    from WholePaths import credit
    probabilities = torch.linspace(.1, .8, batch, device=device, requires_grad=True)
    active = torch.arange(batch, device=device) % 4 != 3
    rounds = torch.where(torch.arange(batch, device=device) % 3 == 0, -1, 0)
    draws = dict(thought_round=rounds, walk_count=torch.full_like(rounds, 2), walk_rounds=torch.full_like(rounds, 3))
    parts = torch.ones(batch, 3, device=device)
    alternative = parts.clone()
    if not tie:
        alternative[:, 0] += torch.arange(batch, device=device) % 3 - 1
    choices = {row: dict(choices=[dict(probability=probabilities[row], alternatives=row + 2)])
               for row in range(batch)}
    first = SimpleNamespace(parts=parts, output_trace=None)
    path = SimpleNamespace(parts=alternative, draw=draws, attention_words=None, output_trace=None,
                           thinking=SimpleNamespace(traces=choices), registry=Error(row_mask=active))
    costs = torch.stack((parts[:, 0] + parts[:, 2], alternative[:, 0] + alternative[:, 2]), -1)
    original = SentenceCredit.score_function
    expected = probabilities.new_zeros(batch)
    for row in range(batch):
        if int(rounds[row]) < 0:
            continue
        value, _ = original(probabilities[row:row+1, None], probabilities.new_tensor([[6 * (row + 2)]]),
                            active[row:row+1, None], costs[row:row+1])
        expected = expected.index_add(0, rounds.new_tensor([row]), value)
    calls = []
    def observe(*args):
        calls.append(args[0].shape[0])
        return original(*args)
    monkeypatch.setattr(SentenceCredit, 'score_function', observe)
    credit(None, first, path, active)
    actual = path.registry._value(path.registry._terms['output.thought_score_function'])
    assert calls == [batch]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.requires_grad == expected.requires_grad
    if actual.requires_grad:
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), probabilities, retain_graph=True)[0],
                                   torch.autograd.grad(expected.sum(), probabilities)[0], rtol=0, atol=0)


def test_thought_retention_aligns_percept_keys_extents_and_gradients(device):
    from Memory import ShortTermMemory
    from PerceptField import PerceptField
    from WalkTrials import ThoughtTrialState
    from clock_fixtures import for_stm
    model = for_stm(ShortTermMemory(batch=4, capacity=3, concept_dim=2))
    fields, states = [], []
    for keys, readings, events in [((('ps', 1), ('ps', 3)), 2, 3),
                                   ((('ps', 2), ('ps', 3)), 3, 2),
                                   ((('ps', 1),), 1, 4)]:
        values = torch.randn(len(keys), 4, readings, 2, device=device, requires_grad=True)
        locations = torch.randn(4, readings, 2, device=device)
        field = PerceptField(keys, values, torch.randn(len(keys), 4, events, 2, device=device),
            torch.randn(4, events, 2, device=device), locations, locations + 1,
            when_band=torch.randn(4, readings, 4, device=device),
            reading=torch.zeros(4, events, dtype=torch.long, device=device),
            index=torch.arange(events, device=device)[None].expand(4, -1))
        model.conceptualSpace._percept_field = field
        fields.append(field)
        states.append(ThoughtTrialState(model))
    winners = torch.tensor([2, 0, 1, 2], device=device)
    ThoughtTrialState.keep(states, winners)
    kept = model.conceptualSpace._percept_field
    assert kept.keys == (('ps', 1), ('ps', 2), ('ps', 3))
    expected = kept.values.new_zeros(3, 4, 3, 2)
    for row, selected in enumerate(winners.tolist()):
        field = fields[selected]
        for index, key in enumerate(field.keys):
            expected[kept.keys.index(key), row, :field.values.shape[2]] = field.values[index, row]
        torch.testing.assert_close(kept.where[row, :field.where.shape[1]], field.where[row], rtol=0, atol=0)
        assert (kept.reading[row, field.reading.shape[1]:] == -1).all()
    torch.testing.assert_close(kept.values, expected, rtol=0, atol=0)
    inputs = [field.values for field in fields]
    actual = torch.autograd.grad(kept.values.square().sum(), inputs, retain_graph=True)
    wanted = torch.autograd.grad(expected.square().sum(), inputs)
    for left, right in zip(actual, wanted, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
