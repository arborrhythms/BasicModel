"""Reconstruction-priority credit on shared forward parameters, not heads."""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bin"))

import pytest
import torch

from Optimizer import (backward_reconstruction_priority,
                       reconstruction_priority_gradient)


@pytest.mark.parametrize("output", [[-3., 4.], [3., 4.], [0., 4.], [-3., 0.]])
def test_projection_and_strict_norm_budget(output):
    r = torch.tensor([2., 0.])
    o = torch.tensor(output)
    q = reconstruction_priority_gradient(r, o, max_ratio=0.5)
    assert torch.dot(r, q) >= -1e-6
    assert q.norm() <= 0.5 * r.norm() + 1e-6
    if torch.dot(r, o) >= 0:
        torch.testing.assert_close(q, o * min(1., float(r.norm() * 0.5 / o.norm())))
    else:
        assert q[0] == 0


def test_zero_and_missing_reconstruction_give_no_output_budget():
    o = torch.tensor([-3., 4.])
    assert reconstruction_priority_gradient(None, o) is None
    torch.testing.assert_close(
        reconstruction_priority_gradient(torch.zeros_like(o), o), torch.zeros_like(o))
    assert reconstruction_priority_gradient(o, None) is None


@pytest.mark.parametrize("ratio", [-0.1, 1.0, float("nan"), float("inf")])
def test_invalid_ratio_fails(ratio):
    with pytest.raises(ValueError, match="0 <= ratio < 1"):
        reconstruction_priority_gradient(torch.ones(2), torch.ones(2), max_ratio=ratio)


def test_large_finite_gradients_do_not_overflow_projection():
    r, o = torch.tensor([2.e20, 0.]), torch.tensor([-3.e20, 4.e20])
    q = reconstruction_priority_gradient(r, o)
    assert torch.isfinite(q).all()
    torch.testing.assert_close(q, torch.tensor([0., 1.e20]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_dense_priority_on_mps():
    p = torch.nn.Parameter(torch.tensor([1., 1.], device="mps"))
    r, o = 2. * p[0], -3. * p[0] + 4. * p[1]
    backward_reconstruction_priority(r + o, r, o, [p])
    torch.testing.assert_close(p.grad.cpu(), torch.tensor([2., 1.], device="cpu"))


def test_sparse_projection_matches_dense_without_capacity_allocation():
    # Duplicate rows, disjoint support, and a million-row logical capacity.
    shape = (1_000_000, 2)
    r = torch.sparse_coo_tensor(
        torch.tensor([[1, 1, 8]]), torch.tensor([[1., 0.], [1., 0.], [0., 1.]]), shape)
    o = torch.sparse_coo_tensor(
        torch.tensor([[1, 15]]), torch.tensor([[-3., 4.], [1., 2.]]), shape)
    q = reconstruction_priority_gradient(r, o).coalesce()
    assert q.is_sparse and q._nnz() <= 3
    assert q.shape == shape
    compact_r = torch.tensor([[2., 0.], [0., 1.], [0., 0.]])
    compact_o = torch.tensor([[-3., 4.], [0., 0.], [1., 2.]])
    expected = reconstruction_priority_gradient(compact_r, compact_o)
    torch.testing.assert_close(q.indices(), torch.tensor([[1, 8, 15]]))
    torch.testing.assert_close(q.values(), expected)


def test_backward_protects_shared_but_keeps_heads_auxiliaries_and_ties():
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
    backward_reconstruction_priority(total, reconstruction, output, [shared, shared])
    torch.testing.assert_close(
        shared.grad, r + reconstruction_priority_gradient(r, o) + a)
    torch.testing.assert_close(output_head.grad, heads[0])
    torch.testing.assert_close(reverse_head.grad, heads[1])


def test_enormous_opposing_output_does_not_erase_reconstruction():
    p = torch.nn.Parameter(torch.tensor(1.))
    r, o = p.square(), -1.e20 * p
    backward_reconstruction_priority(r + o, r, o, [p])
    torch.testing.assert_close(p.grad, torch.tensor(2.))


@pytest.mark.parametrize("scale", [1., 1024.])
def test_common_amp_scale_preserves_policy(scale):
    p = torch.nn.Parameter(torch.tensor([1., 1.]))
    r, o = 2. * p[0], -3. * p[0] + 4. * p[1]
    backward_reconstruction_priority(scale * (r + o), scale * r, scale * o, [p])
    torch.testing.assert_close(p.grad / scale, torch.tensor([2., 1.]))


def test_backward_with_sparse_embedding_and_unshared_head():
    table = torch.nn.Embedding(50, 2, sparse=True)
    head = torch.nn.Parameter(torch.ones(2))
    h = table(torch.tensor([1, 1, 8]))
    r = h.square().sum()
    o = -(h * head).sum()
    rg = torch.autograd.grad(r, table.weight, retain_graph=True)[0]
    og, hg = torch.autograd.grad(o, (table.weight, head), retain_graph=True)
    expected = rg + reconstruction_priority_gradient(rg, og)
    backward_reconstruction_priority(r + o, r, o, [table.weight])
    assert table.weight.grad.is_sparse
    torch.testing.assert_close(table.weight.grad.to_dense(), expected.to_dense())
    torch.testing.assert_close(head.grad, hg)


def test_detached_reconstruction_leaves_output_head_trainable():
    p = torch.nn.Parameter(torch.tensor(2.))
    head = torch.nn.Parameter(torch.tensor(3.))
    r = p.detach().square()
    o = (p * head).square()
    backward_reconstruction_priority(r + o, r, o, [p])
    assert p.grad is None or p.grad == 0
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
    owner = SimpleNamespace(spaces=spaces, output_gradient_ratio=0.5)
    owner._reconstruction_priority_parameters = lambda opt: (
        BaseModel._reconstruction_priority_parameters(owner, opt))
    optimizer = torch.optim.SGD([p, w, c, head], lr=0.1)
    selected = owner._reconstruction_priority_parameters(optimizer)
    assert {id(item) for item in selected} == {id(p), id(w), id(c)}
    r = sum(2. * param[0] for param in (p, w, c))
    o = sum(-3. * param[0] + 4. * param[1] for param in (p, w, c)) + head.sum()
    BaseModel._backward_training_loss(
        owner, r + o, {"reconstruction": r, "output": o}, optimizer)
    for param in (p, w, c):
        torch.testing.assert_close(param.grad, torch.tensor([2., 1.]))
    torch.testing.assert_close(head.grad, torch.ones(2))
    assert excluded.grad is None


def test_truth_modulation_keeps_live_primary_derivatives():
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
    multiplier = 1. + 0.2 * (1. - p.sum())
    for key, raw in (("reconstruction", r), ("output", o)):
        expected = torch.autograd.grad(raw * multiplier, p, retain_graph=True)[0]
        actual = torch.autograd.grad(parts[key], p, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)
    rg = torch.autograd.grad(parts["reconstruction"], p, retain_graph=True)[0]
    og = torch.autograd.grad(parts["output"], p, retain_graph=True)[0]
    backward_reconstruction_priority(total, parts["reconstruction"], parts["output"], [p])
    torch.testing.assert_close(p.grad, rg + reconstruction_priority_gradient(rg, og))


def test_real_runbatch_uses_priority_and_one_optimizer_step(monkeypatch):
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
    model.reconstruction_priority = True
    model.output_gradient_ratio = 0.5
    model.loss.reconstruction_scale = 0.25
    optimizer = model.getOptimizer(lr=1e-5)
    original_backward = Models.backward_reconstruction_priority
    original_step = optimizer.step
    calls = []
    steps = []

    def inspect_backward(total, reconstruction, output, protected, **kwargs):
        assert protected and output.requires_grad
        calls.append(True)
        return original_backward(total, reconstruction, output, protected, **kwargs)

    def inspect_step(*args, **kwargs):
        steps.append(True)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(Models, "backward_reconstruction_priority", inspect_backward)
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
