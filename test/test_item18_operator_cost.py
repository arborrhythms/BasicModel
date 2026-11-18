"""Round C: each kept operator pays once, at choice and complete-path credit."""
from types import SimpleNamespace as NS

import pytest
import torch


def test_repeated_unary_has_the_same_cost_as_its_first_use(monkeypatch):
    import UnaryHistory
    from Language import OperationSelectionLayer, SumLayer
    from test_item20_round5_unary import Half
    layer = OperationSelectionLayer(d_model=2, ops=[SumLayer()], unary_ops=[Half()], chooser='mlp')
    layer.operator_cost = .75
    monkeypatch.setattr(layer.chooser, 'score_binary', lambda x, candidates, *a, **k:
        (x.new_full((*x.shape[:2], 1), -2.), x.new_zeros(x.shape[0], x.shape[1]-1, 1)))
    monkeypatch.setattr(layer.chooser, 'score_unary', lambda x, candidates, *a, **k:
        (x.new_zeros(*x.shape[:2], 1), x.new_full((*x.shape[:2], 1), 4.)))
    history = torch.full((1, 1), -1, dtype=torch.long)
    value = torch.tensor([[[.8, .4]]])
    scores = []
    for _ in range(3):
        _, _, route = layer(value, slots=1, previous_unary=history)
        scores.append(float(route['logits'][0, 0]))
        assert route['kind'].item() == 2
        assert route['logits'][0, -1] == -2
        history = UnaryHistory.applied(history, 0)
    assert scores == [3.25, 3.25, 3.25]


def test_binary_bind_dummy_pays_the_declared_operator_cost():
    from Language import OperationSelectionLayer, SumLayer
    from Expectation import OrderPrime
    layer = OperationSelectionLayer(d_model=2, ops=[SumLayer()], chooser='mlp')
    layer.operator_cost = .75
    value = torch.tensor([[[.8, .4], [.2, .1]]])
    refs = dict(binary_ops=(0,), unary_ops=(), binary_bind=(True,),
        binary_bind_dummy=(True,), unary_bind_dummy=(),
        binary_valid=torch.ones(1, 1, 1, dtype=torch.bool),
        unary_valid=torch.zeros(1, 2, 0, dtype=torch.bool),
        left=value[:, :1, None], right=torch.zeros(1, 1, 1, 2), unary=torch.zeros(1, 2, 0, 2))
    for name in ('refs', 'relations', 'choices', 'resolved', 'requested_orders'):
        refs['binary_'+name] = torch.zeros(1, 1, 1, 2, dtype=torch.long)
        refs['unary_'+name] = torch.zeros(1, 2, 0, 2, dtype=torch.long)
    _, _, route = layer(value, slots=2, reference_data=refs,
        expectation_prime=OrderPrime(torch.tensor([[[.5,.5]]]), torch.ones(1,1),
                                    torch.zeros(1,1,dtype=torch.long)),
        candidate_orders=torch.tensor([[0, -1]]))
    assert route['logits'][0, 0] == -.75


def test_generate_cost_leaves_stop_uncharged():
    from Language import LanguageSpace, SumLayer
    from test_item20_round5_unary import Half
    language = NS(_generate_binary_ops=(SumLayer(),), _generate_unary_ops=(Half(),), operator_cost=1.)
    assert LanguageSpace.choose_generate(language, torch.tensor([[.5, .5, 0.]])).tolist() == [2]


def test_complete_path_pays_c_for_one_more_kept_application(monkeypatch):
    import WholePaths
    from Layers import Error
    from WalkTrials import ThoughtTrialState
    from SentenceCredit import comparison
    active = torch.tensor([True])
    model = NS(_what_memory=lambda: None, when_time=torch.tensor(7), when_encoding=NS(t=7),
        present=lambda: 7, _sentence_fields={}, _sentence_answer_predictions=None,
        inputSpace=NS(data=NS(has_supervised_outputs=True)),
        languageSpace=NS(operator_cost=.75, _generate_binary_ops=(None,), _generate_unary_ops=(None,)))
    before = ThoughtTrialState(model)
    paths = []
    monkeypatch.setattr(WholePaths, 'output', lambda *a, **k:
        (torch.tensor([1.]), None, (torch.tensor([[1, 2, -1]]),)))
    for extra in (False, True):
        errors = Error(row_mask=active)
        errors.add('reconstruction', torch.tensor([2.]), objective='reconstruction')
        actions = [[0, -1, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
        if extra:
            actions.append([2, 0, 0])
        actions.append([-1, -1, -1])
        observation = dict(record=NS(root=torch.ones(1, 2)), thoughts=(), thought_fields=(),
            entries=(NS(actions=torch.tensor(actions)),),
            generation=NS(trace=(torch.tensor([[0, 2, -1]]),)))
        base = WholePaths.Path((), (), observation, errors, torch.tensor([[2., 0., 3.]]),
            WholePaths.Thinking(), before, before, output_trace=('held',))
        draw = dict(walk=torch.tensor([6]), reconstruction_round=torch.tensor([-1]),
            thought_round=torch.tensor([-1]), output_round=torch.tensor([0]), compose_round=torch.tensor([-1]))
        paths.append(WholePaths.finish(model, (), 0, active, base=base, draw=draw, trust=1.))
    assert paths[0].parts[0, 0] == 2 + 5*.75
    assert paths[1].parts[0, 0] - paths[0].parts[0, 0] == .75
    parts = torch.stack([path.parts for path in paths], 1)
    result = comparison(parts, active)
    assert result['winner'].tolist() == [0]
    assert result['advantage'].tolist() == [.75]
    costs, _ = WholePaths.objectives(paths, parts, active, [paths[1].draw])
    assert costs['penalty.reconstruction'] == (5+6)*.75
