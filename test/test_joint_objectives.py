"""Joint objectives retain prediction, ties, sparse updates and optimizer ownership."""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin"))

import pytest
import torch




def test_shared_grammar_operators_belong_to_the_diagnostic_set():
    from Models import BaseModel
    op = torch.nn.Linear(2, 2, bias=False)
    head = torch.nn.Linear(2, 2, bias=False)
    owner = SimpleNamespace(outputSpace=head, synthesis_parameters=lambda: ())
    optimizer = torch.optim.SGD([*op.parameters(), *head.parameters()], lr=.1)
    groups = BaseModel.objective_parameter_groups(owner, optimizer)
    assert {id(p) for p in groups['reconstruction']} == {id(p) for p in op.parameters()}
    assert {id(p) for p in groups['output']} == {id(p) for p in head.parameters()}


def test_packed_prediction_teardown_records_each_observation_once(monkeypatch):
    from Models import BasicModel
    from Layers import BracketExpectation
    from reading_fixtures import commit_reading
    from test_clause_acceptance import SentenceFixture
    fixture = SentenceFixture(monkeypatch)
    discourse = BracketExpectation(n_symbols=8, max_depth=8, n_dim=8,
        concept_dim=8, expectation_scope='structured')
    calls = []
    observe = discourse.observe_stm_end_state
    def record(*args, **kwargs):
        calls.append('observe')
        return observe(*args, **kwargs)
    monkeypatch.setattr(discourse, 'observe_stm_end_state', record)
    def present(owner=None):
        return commit_reading(fixture.language, fixture.registry, fixture.program('cat'),
            None, discourse=discourse, owner=owner)[0]
    owner = present()
    BasicModel._end_step(owner)
    BasicModel._end_step(owner)
    assert calls == ['observe'], "teardown must not invent a second observation"
    present(owner)
    BasicModel._end_step(owner)
    assert calls == ['observe', 'observe'], "a new presentation must be observable"


@pytest.mark.slow
def test_real_packed_runbatch_trains_prediction_and_representation_with_teacher(tmp_path, monkeypatch, eager_reading):
    tied_reconstruction = True
    from test_meronomy_ladder import _build_ladder_variant

    monkeypatch.setenv("MODEL_COMPILE", "none")
    m = _build_ladder_variant(tmp_path, "joint_prediction", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
        ("<sentenceExpectationLossWeight>0.0</sentenceExpectationLossWeight>", "<sentenceExpectationLossWeight>0.1</sentenceExpectationLossWeight>"),
        ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>"),
    ])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.branch_diagnostics_every = 1
    m.loss.reconstruction_scale = 1.0
    m.reconstruct_in_loop = tied_reconstruction
    m.train()
    m._install_unit_span_fn()
    had_outputs = m.inputSpace.data.has_supervised_outputs
    m.inputSpace.data.has_supervised_outputs = False
    optimizer = m.getOptimizer(lr=1e-5)
    disc = m.symbolSpace.expectation
    assert disc is not None and not m.legacy_prediction_enabled
    predictor = list(disc._inter_predictor.parameters())
    observed_predictor_gradient = []
    backward = m._backward_training_loss
    def backward_probe(total, amp_scaler=None, **kwargs):
        registry = m._sentence_cost_registry if getattr(m, '_sentence_backward', False) else m.errors
        loss = registry.total(objective='expectation')
        if loss is not None and loss.requires_grad:
            shared = m.objective_parameter_groups(optimizer)['reconstruction']
            gradients = torch.autograd.grad(loss.sum(), shared, retain_graph=True, allow_unused=True)
            assert all(g is None for g in gradients), 'expectation must not reach the encoder'
            gradients = torch.autograd.grad(loss.sum(), predictor, retain_graph=True, allow_unused=True)
            observed_predictor_gradient.append(any(g is not None and bool(g.norm() > 0) for g in gradients))
        return backward(total, amp_scaler, **kwargs)
    monkeypatch.setattr(m, '_backward_training_loss', backward_probe)
    try:
        for weight in (0.1, 0.1, 0.):
            m.inter_loss_weight = weight
            # The effective model loss weight must stop Adam credit even if
            # the layer still has an enabled accumulation gate.
            disc.set_inter_loss_weight(0.1)
            before = [p.detach().clone() for p in predictor]
            batch = (m.inputSpace.prepPackedInput([
                ["1 plus 2", "3 plus 4"], ["5 plus 6"]]), torch.zeros(2, 1, 0))
            result, _ = m.runBatch(
                train=True, batchSize=2, split="train", optimizer=optimizer, batch_override=batch)
            assert result is not None
            changed = [not torch.equal(p, old) for p, old in zip(predictor, before)]
            assert any(changed) if weight else not any(changed)
            assert all(not p.requires_grad for chain in disc._inter_context for _, p, _ in chain)
        assert any(observed_predictor_gradient)
    finally:
        m.inputSpace.data.has_supervised_outputs = had_outputs
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_summed_objectives_on_mps():
    p = torch.nn.Parameter(torch.tensor([1., 1.], device="mps"))
    r, o = 2. * p[0], -3. * p[0] + 4. * p[1]
    (r + o).backward()
    torch.testing.assert_close(p.grad.cpu(), torch.tensor([-1., 4.], device="cpu"))


def test_backward_sums_shared_heads_auxiliaries_and_ties():
    shared = torch.nn.Parameter(torch.tensor([1., 2.]))
    output_head = torch.nn.Parameter(torch.tensor(3.))
    reverse_head = torch.nn.Parameter(torch.tensor(2.))
    # The same forward parameter participates again in inverse synthesis.
    h = shared.square()
    reconstruction = 0.75 * (h * reverse_head).sum()
    output = 0.25 * (-10. * h[0] + 4. / shared[1] + output_head.square())
    auxiliary = 0.1 * (shared * output_head).sum()
    total = reconstruction + output + auxiliary
    r = torch.autograd.grad(reconstruction, shared, retain_graph=True)[0]
    o = torch.autograd.grad(output, shared, retain_graph=True)[0]
    a = torch.autograd.grad(auxiliary, shared, retain_graph=True)[0]
    heads = torch.autograd.grad(total, (output_head, reverse_head), retain_graph=True)
    total.backward()
    torch.testing.assert_close(
        shared.grad, r + o + a)
    torch.testing.assert_close(output_head.grad, heads[0])
    torch.testing.assert_close(reverse_head.grad, heads[1])


@pytest.mark.parametrize("scale", [1., 1024.])
def test_common_amp_scale_preserves_summed_gradients(scale):
    from ObjectiveOwnership import backward_owned
    p = torch.nn.Parameter(torch.tensor([1., 1.]))
    head = torch.nn.Parameter(torch.tensor([1., 1.]))
    r, o = 2. * p[0], -3. * head[0] + 4. * head[1]
    backward_owned({'reconstruction': r, 'output': o},
        {'reconstruction': (p,), 'expectation': (), 'output': (head,)}, scale=scale)
    torch.testing.assert_close(p.grad / scale, torch.tensor([2., 0.]))
    torch.testing.assert_close(head.grad / scale, torch.tensor([-3., 4.]))


def test_backward_with_sparse_embedding_and_unshared_head():
    table = torch.nn.Embedding(50, 2, sparse=True)
    head = torch.nn.Parameter(torch.ones(2))
    h = table(torch.tensor([1, 1, 8]))
    r = h.square().sum()
    o = -(h * head).sum()
    rg = torch.autograd.grad(r, table.weight, retain_graph=True)[0]
    og, hg = torch.autograd.grad(o, (table.weight, head), retain_graph=True)
    expected = rg + og
    (r + o).backward()
    assert table.weight.grad.is_sparse
    torch.testing.assert_close(table.weight.grad.to_dense(), expected.to_dense())
    torch.testing.assert_close(head.grad, hg)


def test_detached_reconstruction_leaves_output_head_trainable():
    p = torch.nn.Parameter(torch.tensor(2.))
    head = torch.nn.Parameter(torch.tensor(3.))
    r = p.detach().square()
    o = (p * head).square()
    (r + o).backward()
    torch.testing.assert_close(p.grad, torch.tensor(36.))
    torch.testing.assert_close(head.grad, torch.tensor(24.))


def test_model_parameter_ownership_and_backward_dispatch():
    from Models import BaseModel
    from Layers import Error
    p, w, c, excluded, head = [torch.nn.Parameter(torch.tensor([1., 1.])) for _ in range(5)]
    reader = torch.nn.Module()
    reader.register_parameter('weight', head)
    owner = SimpleNamespace(outputSpace=reader, synthesis_parameters=lambda: (), errors=Error())
    owner.objective_parameter_groups = lambda opt: BaseModel.objective_parameter_groups(owner, opt)
    optimizer = torch.optim.SGD([p, w, c, head], lr=.1)
    selected = owner.objective_parameter_groups(optimizer)['reconstruction']
    assert {id(item) for item in selected} == {id(p), id(w), id(c)}
    r = sum(2. * param[0] for param in (p, w, c))
    o = sum(-3. * param[0] + 4. * param[1] for param in (p, w, c)) + head.sum()
    owner.errors.add('reconstruction.probe', r, objective='reconstruction')
    owner.errors.add('output.probe', o, objective='output')
    BaseModel._backward_training_loss(owner, r + o, optimizer=optimizer)
    for param in (p, w, c):
        torch.testing.assert_close(param.grad, torch.tensor([2., 0.]))
    torch.testing.assert_close(head.grad, torch.ones(2))
    assert excluded.grad is None


def test_truth_modulation_scales_objectives_without_state_feedback():
    from Language import SymbolSubSpace

    p = torch.nn.Parameter(torch.tensor([0.3, 0.4]))
    truth = SimpleNamespace(is_empty=lambda: False)
    owner = SimpleNamespace(truth_layer=truth)
    r, o = p.square().sum(), -3. * p[0] + 4. * p[1]
    parts = {"reconstruction": r, "output": o}
    total = SymbolSubSpace.truth_modulated_loss(
        owner, r + o, symbolic_space=None, universality_score=p.sum(),
        luminosity_weight=0., universality_weight=0.2, balance_weight=0.,
        gradient_objectives=parts)
    multiplier = 1. + 0.2 * (1. - p.sum().detach())
    for key, raw in (("reconstruction", r), ("output", o)):
        expected = torch.autograd.grad(raw * multiplier, p, retain_graph=True)[0]
        actual = torch.autograd.grad(parts[key], p, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)
    rg = torch.autograd.grad(parts["reconstruction"], p, retain_graph=True)[0]
    og = torch.autograd.grad(parts["output"], p, retain_graph=True)[0]
    total.backward()
    torch.testing.assert_close(p.grad, rg + og)


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('config_name, expected', [
    ('MM_xor.xml', ['batch']),
    ('MM_xor_loopback.xml', ['exploit', 'explore', 'batch']),
])
def test_real_runbatch_steps_sentence_trials_then_batch_only_objectives(monkeypatch, config_name, expected):
    import Models
    from data import TheData
    from util import init_config

    monkeypatch.setenv("MODEL_COMPILE", "none")
    project = os.path.dirname(os.path.dirname(__file__))
    config = os.path.join(project, "data", config_name)
    init_config(path=config, defaults_path=os.path.join(project, "data", "model.xml"))
    TheData.load("xor")
    model, _ = Models.BaseModel.from_config(config, data=TheData)
    model.train()
    model.loss.reconstruction_scale = 0.25
    optimizer = model.getOptimizer(lr=1e-5)
    original_backward = model._backward_training_loss
    original_step = optimizer.step
    calls = []
    steps = []

    def inspect_backward(total, *args, **kwargs):
        assert total.requires_grad
        calls.append(model._sentence_trial if getattr(model, '_sentence_backward', False) else 'batch')
        return original_backward(total, *args, **kwargs)

    def inspect_step(*args, **kwargs):
        steps.append(getattr(model, '_sentence_trial', None) or 'batch')
        return original_step(*args, **kwargs)

    monkeypatch.setattr(model, "_backward_training_loss", inspect_backward)
    monkeypatch.setattr(optimizer, "step", inspect_step)
    for _ in range(2):
        loader = model.inputSpace.data.data_loader(split="train", num_streams=2)
        inputs, outputs = next(iter(loader))
        batch = (model.inputSpace.prepInput(inputs), model.outputSpace.prepOutput(outputs))
        result, _ = model.runBatch(
            train=True, batchSize=2, split="train", optimizer=optimizer,
            batch_override=batch)
        assert result is not None
    assert calls == steps == expected * 2


def test_answer_path_operators_are_independently_owned_and_keep_learning(monkeypatch, tmp_path):
    """Dedicated output parameters learn from their own branch. Shared
    operator diagnostics exclude these independent heads."""
    import Models
    from data import TheData
    from util import init_config
    from What import What

    monkeypatch.setenv("MODEL_COMPILE", "none")
    project = os.path.dirname(os.path.dirname(__file__))
    src = open(os.path.join(project, "data", "MM_xor.xml")).read()
    assert src.count("<architecture>") == 1
    config = tmp_path / "MM_xor_synthesis_ownership.xml"
    config.write_text(src.replace(
        "<architecture>", "<architecture>\n    <answerSynthesis>true</answerSynthesis>", 1))
    init_config(path=str(config), defaults_path=os.path.join(project, "data", "model.xml"))
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = Models.BaseModel.from_config(str(config), data=TheData)
    loader = model.inputSpace.data.data_loader(split="train", num_streams=2)
    inputs, outputs = next(iter(loader))
    batch = (model.inputSpace.prepInput(inputs), model.outputSpace.prepOutput(outputs))
    model.eval()
    with torch.no_grad():
        model.what((What.supervised(0), What.supervised(1)), batch[0])   # builds the answer path
    model.train()
    optimizer = model.getOptimizer(lr=1e-2)
    enlisted = list(model.synthesis_parameters())
    groups = model.objective_parameter_groups(optimizer)
    output_ids = {id(p) for p in groups['output']}
    answer_only = [p for p in enlisted if id(p) in output_ids]
    # The generate chooser is enlisted by synthesis but has one writer:
    # reconstruction. Output only chooses between already-costed walks.
    reconstruction_ids = {id(p) for p in groups['reconstruction']}
    assert {id(p) for p in model.languageSpace.generate_policy.parameters()} <= reconstruction_ids
    assert answer_only
    before = [p.detach().clone() for p in answer_only]
    result, _ = model.runBatch(train=True, batchSize=2, split="train",
                               optimizer=optimizer, batch_override=batch)
    assert result is not None
    shared = {p.data_ptr() for p in model.objective_parameter_groups(optimizer)['reconstruction']}
    assert shared                                            # shared forward parameters
    assert not shared & {p.data_ptr() for p in answer_only}  # answer path exempt
    owned = {p.data_ptr() for g in optimizer.param_groups for p in g["params"]}
    assert {p.data_ptr() for p in answer_only} <= owned          # handed to the optimizer
    assert any(not torch.equal(a, b) for a, b in zip(before, answer_only))


def test_real_intermediate_and_final_ends_have_same_canonical_roles(tmp_path, eager_reading):
    from test_meronomy_ladder import _build_ladder_variant
    from test_reverse_traversal import _stage_packed
    meanings = []
    for i, rows in enumerate(([["1 plus 2", "3 plus 4"]], [["1 plus 2 "]])):
        model = _build_ladder_variant(tmp_path, f"closing_layout_{i}", [
            ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
            ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
            ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
        ])
        # This probes the closing layout on a completed derivation. Select one
        # known binary operation and STOP when eligible; random unary loops
        # can exhaust the budget and correctly leave an incomplete forest.
        chooser = model.symbolSpace.languageLayer.operation_layer.chooser
        def binary(x, candidates, *_args, **_kwargs):
            scores = x.new_full(candidates.shape[:-1], -1e6)
            scores[..., 0] = 0.
            return x.new_full((*x.shape[:2], 1), 1e6), scores
        def unary(x, candidates, *_args, **_kwargs):
            return (x.new_full((*x.shape[:2], 1), 1e6),
                    x.new_full(candidates.shape[:-1], -1e6))
        chooser.score_binary, chooser.score_unary = binary, unary
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
        model._install_unit_span_fn()
        try:
            with torch.no_grad():
                model(model.inputSpace.prepPackedInput(rows))
            discourse = model.symbolSpace.expectation
            if i == 0:
                payload = model._tensor_sentence_roots_live[0, 0].reshape(3, -1)
                depth = model._tensor_sentence_roots_depth[0, 0]
            else:
                payload = model._tensor_final_end_slots[0]
                depth = model._tensor_final_end_depth[0]
            value, mask = discourse._canonical_meaning(payload, depth, "stm")
            meanings.append((value.detach().clone(), mask.clone()))
        finally:
            model.End()
            model.symbolSpace.soft_reset()
            torch._dynamo.reset()
    torch.testing.assert_close(meanings[0][0], meanings[1][0])
    torch.testing.assert_close(meanings[0][1], meanings[1][1])
