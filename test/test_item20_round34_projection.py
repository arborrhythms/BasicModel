"""Direct certificates for the two-sided gradient and cost routing."""
from types import SimpleNamespace
import json
import os
from pathlib import Path
import torch
import pytest
from GradientProjection import backward, project, sides
from ObjectiveOwnership import registry_costs


def record(name, value):
    directory = Path(os.environ.get('ITEM20_RECEIPT', '/tmp/item20-round34-direct'))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')


def test_opposing_agreeing_and_single_writer():
    model = torch.nn.Module()
    model.conceptualSpaces = []
    model.shared = torch.nn.Parameter(torch.tensor([1., 2.]))
    model.single = torch.nn.Parameter(torch.tensor([3.]))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    r = model.shared[0] + 4 * model.single.sum()
    a = -2 * model.shared[0] + 3 * model.shared[1]
    original = torch.autograd.grad(r, model.shared, retain_graph=True)[0]
    backward(model, dict(reconstruction=r, output=a), optimizer, 1.)
    torch.testing.assert_close(model.shared.grad, torch.tensor([1., 3.]))
    torch.testing.assert_close(model.single.grad, torch.tensor([4.]))
    assert torch.equal(original, torch.tensor([1., 0.]))
    agreeing, _ = project(original, torch.tensor([2., 3.]), torch.ones(2))
    assert torch.equal(agreeing, torch.tensor([2., 3.]))
    record('projection', dict(before=[-1., 3.], after=model.shared.grad.tolist(),
        reconstruction=original.tolist(), single_writer=model.single.grad.tolist(), agreeing=agreeing.tolist()))


def test_adam_metric_and_sparse_concept_rows():
    from GradientProjection import metric
    model = torch.nn.Module()
    model.conceptualSpaces = []
    model.similarity_codebook = torch.nn.Module()
    model.similarity_codebook.W = torch.nn.Parameter(torch.ones(3, 2))
    p = model.similarity_codebook.W
    optimizer = torch.optim.Adam([p], lr=.1, betas=(.9, .5), eps=1e-8)
    assert torch.equal(metric(optimizer, p, p), torch.ones_like(p))
    optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg_sq=torch.tensor([[.5, 8.]]).expand(3,2).clone())
    r = torch.nn.functional.embedding(torch.tensor([0, 1]), p, sparse=True).sum()
    a = (torch.nn.functional.embedding(torch.tensor([0, 2]), p, sparse=True) * torch.tensor([[-2., 3.], [2., 3.]])).sum()
    backward(model, dict(reconstruction=r, output=a), optimizer, 1.)
    # Metric (1, .25): dot=-1.25, rr=1.25, so A receives +R.
    expected = torch.tensor([[0., 5.], [1., 1.], [2., 3.]])
    torch.testing.assert_close(p.grad.to_dense(), expected)
    record('adam-projection', dict(before=[[-1.,4.],[1.,1.],[2.,3.]], after=p.grad.to_dense().tolist(), metric=[1.,.25]))


def test_registry_routes_all_declared_sides_without_changing_values():
    routes = {
        'reconstruction_reverse.what': ('reconstruction', 'R'),
        'reconstruction_order_zero.what': ('reconstruction', 'R'),
        'expectation.predictor': ('expectation', 'R'),
        'attention.reading_lesson': ('attention', 'R'),
        'attention.candidate_ce': ('attention', 'R'),
        'output.answer': ('output', 'A'),
        'output.walk_residual.form': ('output', 'A'),
        'output.walk_credit': ('output', 'A'),
        'output.compose_score_function': ('chooser', 'A'),
        'expectation.bind_score_function': ('expectation', 'A'),
        'attention.candidate_departure': ('attention', 'A'),
        'output.thought_score_function': ('chooser', 'A'),
        'compose.choice': ('grammar', 'R'), 'generate.choice': ('grammar', 'A'),
        'penalty': ('reg', 'R'),
    }
    value = torch.tensor(2., requires_grad=True)
    for name, (objective, side) in routes.items():
        from Layers import Error
        from ObjectiveOwnership import default_side, term_owner
        registry = Error()
        registry.add(name, value, weight=3., category=objective, objective=objective, side=side)
        costs = registry_costs(registry)
        owner = term_owner(name, objective, objective)
        key = owner if side == default_side(owner) else owner + '.' + side
        assert key in costs or objective == 'reg'
        result = sides(costs, 1.)
        assert result[side].item() == 6.
        assert result['A' if side == 'R' else 'R'] is None
    record('routes', {name:dict(owner=(['reconstruction','expectation','output','attention'] if owner == 'reg'
        else term_owner(name, owner, owner)), side=side) for name,(owner,side) in routes.items()})


def test_generate_credit_uses_proposal_correction_and_ties_have_no_graph():
    from WalkTrials import generate_credit
    from Expectation import alternative_probabilities
    logits = torch.nn.Parameter(torch.tensor([[2., 1., 0.]]))
    language = SimpleNamespace(exploration_fraction=.5, exploration_temperature=2.,
        generate_policy_logits=lambda top, features:logits)
    language._tree_layer = lambda arity:language
    tops = torch.zeros(1, 2, 1)
    features = torch.zeros(1, 2, 3, 9)
    features[..., -1] = 1
    audit = dict(departure=torch.tensor([0]), greedy=torch.tensor([[0, 0]]),
        explore=torch.tensor([[1, 0]]), eligible=torch.tensor([[True, True]]),
        costs=torch.tensor([[3., 1.]]), wins=torch.tensor([True]))
    credit = generate_credit(language, audit, tops, features)
    proposal = alternative_probabilities(logits.detach().masked_fill(torch.tensor([[True, False, False]]), -torch.inf),
        fraction=.5, temperature=2.)
    expected = logits.softmax(-1)[0, 1] * 2 / proposal[0, 1] * -2
    torch.testing.assert_close(credit[0], expected)
    credit.sum().backward()
    assert logits.grad[0, 1] < 0
    details = audit['score_function']
    record('generate-credit', dict(probability=float(details['probabilities'][0,0]),
        proposal=float(proposal[0,1]), correction=float(details['scale'][0,0]),
        delta=-2., credit=float(credit.detach()[0]), alternative_probabilities=proposal.tolist()))
    audit['costs'] = torch.tensor([[1., 1.]])
    assert not generate_credit(language, audit, tops, features).requires_grad


@pytest.mark.large_memory
def test_model_xml_default_projects_one_native_batch_and_steps_once(monkeypatch, eager_reading):
    from test_mm_xor import _fresh_model
    from SentenceTraining import gradient_receipt
    from util import init_config
    from data import TheData
    from Models import BaseModel
    import Language
    init_config('data/model.xml', defaults_path=str(Path('data/model.xml').resolve()))
    Language.TheGrammar._configured = False
    TheData.load('xor')
    data = TheData
    model, _ = BaseModel.from_config('data/model.xml', data=data)
    assert model.separate_chooser is False
    optimizer = model.getOptimizer(lr=.001)
    steps = []
    step = optimizer.step
    def counted(*args, **kwargs):
        steps.append(1)
        return step(*args, **kwargs)
    monkeypatch.setattr(optimizer, 'step', counted)
    try:
        batch = (torch.full((1, *model.inputSpace.inputShape), .2),
                 torch.full((1, *model.outputSpace.outputShape), .3))
        model.runBatch(train=True, optimizer=optimizer, batchSize=1, split='train', batch_override=batch)
        result = gradient_receipt(model)
        assert len(steps) == 1
        assert result['projection']
        assert all(set(('cosine_before','fraction_projected','fraction_answer_removed')) <= set(row)
                   for row in result['projection'].values())
        record('native-default-projection', dict(optimizer_steps=len(steps), separate_chooser=model.separate_chooser,
            projection=result['projection'], realized_steps=result['realized_steps']))
    finally:
        model.End()
