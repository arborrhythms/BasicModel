"""Joint objectives retain prediction, ties, sparse updates and optimizer ownership."""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin"))

import pytest
import torch


def test_joint_learning_selects_a_predictive_encoding_with_exact_reconstruction():
    # Every rotation permits exact reconstruction using its tied transpose.
    # Prediction must still be able to select the useful rotation.
    angle = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    inputs = torch.eye(2, dtype=torch.float64)
    target = torch.tensor([2. ** -0.5, -(2. ** -0.5)], dtype=torch.float64)
    optimizer = torch.optim.SGD([angle], lr=0.15)
    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        c, s = angle.cos(), angle.sin()
        encode = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
        representation = inputs @ encode.T
        recovered = representation @ encode
        reconstruction = (recovered - inputs).square().mean()
        prediction = (representation[:, 0] - target).square().mean()
        losses.append(float(prediction.detach()))
        (reconstruction + prediction).backward()
        optimizer.step()
        assert float(reconstruction.detach()) < 1.e-25
    assert abs(float(angle.detach())) > 0.3
    assert losses[-1] < losses[0] * 0.02


def test_shared_grammar_operators_belong_to_the_diagnostic_set():
    from Models import BaseModel

    op = torch.nn.Linear(2, 2, bias=False)
    head = torch.nn.Linear(2, 2, bias=False)
    owner = SimpleNamespace(
        spaces=[], symbolSpace=SimpleNamespace(_host_layer_registry={('CS', 'sum'): op}))
    optimizer = torch.optim.SGD([*op.parameters(), *head.parameters()], lr=0.1)
    shared = BaseModel._shared_representation_parameters(owner, optimizer)
    assert {id(p) for p in shared} == {id(p) for p in op.parameters()}


def test_packed_prediction_teardown_records_each_observation_once():
    from Models import BasicModel

    calls = []
    owner = SimpleNamespace(
        inputSpace=SimpleNamespace(_sentence_pack_enabled=True),
        _packed_sentence_roots=torch.ones(1, 2, 3),
        _drain_packed_stm_end_states=lambda: calls.append('observe'),
        symbolSpace=None, wholeSpaces=[], spaces=[])
    BasicModel._end_step(owner)
    BasicModel._end_step(owner)
    assert calls == ['observe'], "teardown must not invent a second observation"
    BasicModel._start_spaces_for_forward(owner)
    BasicModel._end_step(owner)
    assert calls == ['observe', 'observe'], "a new presentation must be observable"


@pytest.mark.slow
@pytest.mark.parametrize("tied_reconstruction", [False, True])
def test_real_packed_runbatch_trains_prediction_and_representation_with_teacher(tmp_path, monkeypatch, tied_reconstruction):
    from test_meronomy_ladder import _build_ladder_variant

    monkeypatch.setenv("MODEL_COMPILE", "none")
    m = _build_ladder_variant(tmp_path, "joint_prediction", [
        ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
        ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
        ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
        ("<interLossWeight>0.0</interLossWeight>", "<interLossWeight>0.1</interLossWeight>"),
        ("<training>", "<training><teacherReconstruction>true</teacherReconstruction>"),
    ])
    m._tensor_peer_while_eager = True
    m._chart_compose_per_word = lambda: None
    m.branch_diagnostics_every = 1
    m.loss.reconstruction_scale = 1.0
    m.reconstruct_in_loop = tied_reconstruction
    if tied_reconstruction:
        # Keep the forward state handoff explicit while using production's
        # cached reconstruction backward. This probe reads each graph several
        # times to check both objectives; uncached HOP backward would rebuild
        # all nested programs for every read. All gradient/update checks below
        # apply to the actual completed reconstruction, including the zero step.
        import util
        monkeypatch.setattr(util, "TheCompileBackend", "eager")
        monkeypatch.setenv("BASICMODEL_RECON_PLACEMENT", "compiled")
        m._compiled_step = m._forward_with_compiled_sentence_state
    m.train()
    m._install_unit_span_fn()
    had_outputs = m.inputSpace.data.has_supervised_outputs
    m.inputSpace.data.has_supervised_outputs = False
    optimizer = m.getOptimizer(lr=1e-5)
    disc = m.symbolSpace.discourse
    assert disc is not None and not m.legacy_prediction_enabled
    predictor = list(disc._inter_predictor.parameters())
    records = {}
    record, backward = m.record_loss, m._backward_training_loss

    def record_probe(name, value, **kwargs):
        records[name] = value
        return record(name, value, **kwargs)

    def backward_probe(total, amp_scaler=None):
        loss = records.get("inter")
        if m.inter_loss_weight == 0:
            gradients = torch.autograd.grad(total, predictor, retain_graph=True, allow_unused=True)
            assert all(g is None for g in gradients), "disabled prediction must not retain Adam credit"
            return backward(total, amp_scaler)
        assert loss is not None and loss.requires_grad, {
            "context": [len(c) for c in disc._inter_context],
            "roots": getattr(m, "_packed_sentence_roots", None),
            "packed": m.inputSpace._sentence_pack_enabled,
            "counts": m.inputSpace._packed_sentence_counts_host,
            "weight": disc._inter_loss_weight,
        }
        shared = m._shared_representation_parameters(optimizer)
        gradients = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        assert any(g is not None and g.norm() > 0 for g in gradients), "inter must reach the real encoder"
        if tied_reconstruction:
            assert any(e["reconstruction_norm"] > 0 for e in m._last_operator_gradients.values())
        assert any(e["expectation_norm"] > 0 for e in m._last_operator_gradients.values())
        assert any(e["reconstruction_expectation_cosine"] is not None
                   for e in m._last_operator_gradients.values())
        active = {name: entry for name, entry in m._last_operator_gradients.items()
                  if entry["reconstruction_expectation_cosine"] is not None}
        import json
        print("MEASURED_EXPECTATION_GRADIENTS", json.dumps(active, sort_keys=True))
        gradients = torch.autograd.grad(loss, predictor, retain_graph=True, allow_unused=True)
        assert any(g is not None and g.norm() > 0 for g in gradients)
        return backward(total, amp_scaler)

    monkeypatch.setattr(m, "record_loss", record_probe)
    monkeypatch.setattr(m, "_backward_training_loss", backward_probe)
    try:
        for weight in (0.1, 0.1, 0.):
            m.inter_loss_weight = weight
            # The effective model loss weight must stop Adam credit even if
            # the layer still has an enabled accumulation gate.
            disc.set_inter_loss_weight(0.1)
            records.clear()
            before = [p.detach().clone() for p in predictor]
            batch = (m.inputSpace.prepPackedInput([
                ["1 plus 2", "3 plus 4"], ["5 plus 6"]]), torch.zeros(2, 1, 0))
            result, _ = m.runBatch(
                train=True, batchSize=2, split="train", optimizer=optimizer, batch_override=batch)
            assert result is not None
            changed = [not torch.equal(p, old) for p, old in zip(predictor, before)]
            assert any(changed) if weight else not any(changed)
            assert all(not p.requires_grad for chain in disc._inter_context for _, p, _ in chain)
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
    p = torch.nn.Parameter(torch.tensor([1., 1.]))
    r, o = 2. * p[0], -3. * p[0] + 4. * p[1]
    from Models import BaseModel
    BaseModel._backward_training_loss(None, r + o,
        SimpleNamespace(scale=lambda loss: scale * loss))
    torch.testing.assert_close(p.grad / scale, torch.tensor([-1., 4.]))


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
    from Spaces import PartSpace, WholeSpace, ConceptualSpace

    p, w, c, excluded, head = [
        torch.nn.Parameter(torch.tensor([1., 1.])) for _ in range(5)]
    spaces = []
    for cls, params in ((PartSpace, [p, excluded]), (WholeSpace, [w]),
                        (ConceptualSpace, [c, p])):
        space = object.__new__(cls)
        torch.nn.Module.__init__(space)
        space.params = params
        # The real getParameters may also validate already-built stores.
        space.getParameters = lambda params=params: params
        spaces.append(space)
    owner = SimpleNamespace(spaces=spaces)
    owner._shared_representation_parameters = lambda opt: (
        BaseModel._shared_representation_parameters(owner, opt))
    optimizer = torch.optim.SGD([p, w, c, head], lr=0.1)
    selected = owner._shared_representation_parameters(optimizer)
    assert {id(item) for item in selected} == {id(p), id(w), id(c)}
    r = sum(2. * param[0] for param in (p, w, c))
    o = sum(-3. * param[0] + 4. * param[1] for param in (p, w, c)) + head.sum()
    BaseModel._backward_training_loss(
        owner, r + o)
    for param in (p, w, c):
        torch.testing.assert_close(param.grad, torch.tensor([-1., 4.]))
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


def test_real_runbatch_uses_one_backward_and_one_optimizer_step(monkeypatch):
    import Models
    from data import TheData
    from util import init_config

    monkeypatch.setenv("MODEL_COMPILE", "none")
    project = os.path.dirname(os.path.dirname(__file__))
    config = os.path.join(project, "data", "MM_xor.xml")
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
        calls.append(True)
        return original_backward(total, *args, **kwargs)

    def inspect_step(*args, **kwargs):
        steps.append(True)
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
    assert len(calls) == len(steps) == 2


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
    answer_only = list(model.synthesis_parameters())
    assert answer_only
    before = [p.detach().clone() for p in answer_only]
    result, _ = model.runBatch(train=True, batchSize=2, split="train",
                               optimizer=optimizer, batch_override=batch)
    assert result is not None
    shared = {p.data_ptr() for p in model._shared_representation_parameters(optimizer)}
    assert shared                                            # shared forward parameters
    assert not shared & {p.data_ptr() for p in answer_only}  # answer path exempt
    owned = {p.data_ptr() for g in optimizer.param_groups for p in g["params"]}
    assert {p.data_ptr() for p in answer_only} <= owned          # handed to the optimizer
    assert any(not torch.equal(a, b) for a, b in zip(before, answer_only))


def test_real_intermediate_and_final_seals_have_same_canonical_roles(tmp_path):
    from test_meronomy_ladder import _build_ladder_variant
    from test_reverse_traversal import _stage_packed
    meanings = []
    for i, rows in enumerate(([["1 plus 2", "3 plus 4"]], [["1 plus 2 "]])):
        model = _build_ladder_variant(tmp_path, f"seal_layout_{i}", [
            ("<serialWordCapacity>8</serialWordCapacity>", "<serialWordCapacity>16</serialWordCapacity>"),
            ("<serialWordBuckets>8</serialWordBuckets>", "<serialWordBuckets>16</serialWordBuckets>"),
            ("<sentenceExpectation>false</sentenceExpectation>", "<sentenceExpectation>true</sentenceExpectation>"),
        ])
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
        model._install_unit_span_fn()
        try:
            _stage_packed(model, rows)
            with torch.no_grad():
                result = model._forward_with_compiled_sentence_state(None)
            model._publish_compiled_sentence_state(result)
            discourse = model.symbolSpace.discourse
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
