"""Every unary application pays the same cost; history retains legality only."""
from types import SimpleNamespace as NS
import torch
from torch import nn

import UnaryHistory
from test_item20_round5_memory import receipt


class Half(nn.Module):
    def forward(self, value):
        return value*.5

    def reverse(self, value):
        return value*2


def test_uniform_operator_cost_does_not_forbid_a_legitimate_unary_chain(monkeypatch):
    from Language import OperationSelectionLayer, SumLayer
    layer = OperationSelectionLayer(d_model=2, ops=[SumLayer()], unary_ops=[Half()], chooser='mlp')
    layer.operator_cost = 1.
    bias = nn.Parameter(torch.tensor(20.))
    monkeypatch.setattr(layer.chooser, 'score_binary', lambda x, candidates, *a, **k:
        (x.new_zeros(*x.shape[:2], 1), x.new_zeros(x.shape[0], x.shape[1]-1, 1)))
    monkeypatch.setattr(layer.chooser, 'score_unary', lambda x, candidates, *a, **k:
        (x.new_zeros(*x.shape[:2], 1), bias.expand(*x.shape[:2], 1)))
    history = torch.full((1, 1), -1, dtype=torch.long)
    value = torch.tensor([[[.8, .4]]])
    penalties, counts, probabilities = [], [], []
    for step in range(6):
        _, value, route = layer(value, slots=1, previous_unary=history, sample=False)
        penalties.append(float((bias-route['logits'][0, 0]).detach()))
        assert torch.isfinite(route['logits'][0, 0])
        counts.append(int(UnaryHistory.count(history)[0, 0]))
        probabilities.append(route['probabilities'][0, 0])
        history = route['previous_unary']
        assert route['kind'].item() == 2
    assert penalties == [1.] * 6
    assert counts == [0, 1, 2, 3, 4, 5]
    torch.autograd.grad(-probabilities[2].log(), (bias,))
    receipt('uniform-operator-cost', dict(penalties=penalties, counts=counts,
        first_application_penalty=1, all_repeats_legal=True,
        graph_preserved=True, weight=1))


def test_unary_count_follows_the_operand_and_resets_on_binary_composition():
    from Language import LanguageSpace
    def choice(kind, *, position=0, operation=0):
        return NS(applied=torch.tensor([True]), kind=torch.tensor([kind]),
            position=torch.tensor([position]), local_op=torch.tensor([operation]), stack_arity=None)
    history = torch.full((1, 4), -1, dtype=torch.long)
    for operation in (1, 2, 1):
        history = LanguageSpace.update_unary_history(history, choice(2, operation=operation))
    assert UnaryHistory.count(history).tolist() == [[3, 0, 0, 0]]
    assert UnaryHistory.last(history).tolist() == [[1, -1, -1, -1]]
    history = LanguageSpace.push_unary_history(history, torch.tensor([True]))
    assert UnaryHistory.count(history).tolist() == [[0, 3, 0, 0]]
    history = LanguageSpace.update_unary_history(history, choice(3))
    assert UnaryHistory.count(history).tolist() == [[3, 0, 0, 0]]
    assert UnaryHistory.last(history)[0, :2].tolist() == [-2, -2]
    history = LanguageSpace.update_unary_history(history, choice(2, operation=2))
    assert UnaryHistory.count(history).tolist() == [[4, 0, 0, 0]]
    history = LanguageSpace.update_unary_history(history, choice(1))
    assert UnaryHistory.count(history).eq(0).all()


def test_sentence_storage_uses_the_existing_execution_bound_without_resize():
    from SentenceStrata import SentenceStrata
    strata = SentenceStrata(1, 2, 8, 4)
    assert strata.operators == 22
    assert strata.words + strata.operators == 24
    assert len(strata._traces) == 3 and not hasattr(strata, 'banks')
    pointers = [store.slots.data_ptr() for store in strata.idea_stores]
    strata.clear()
    assert pointers == [store.slots.data_ptr() for store in strata.idea_stores]
    receipt('sentence-construction-bound', dict(words=2, recent_chunks=8,
        operator_bound_before=1, operator_bound_after=22, trace_action_bound=24,
        syntax_relation_rows=0, resized=False))
