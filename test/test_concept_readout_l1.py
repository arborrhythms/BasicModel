"""Weak incoming-connection lasso in the live row-local Sigma optimizer."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from Layers import IndexedSigmaConceptsFromPercepts, SigmaConceptsFromPercepts
from Models import BaseModel
from Optimizer import (Adam, MultiOptimizer, RowLocalAdam, configure_l1_proximal,
                       backward_reconstruction_priority)


@pytest.mark.parametrize("strength", [-.01, float("nan"), float("inf")])
def test_invalid_readout_l1_fails(strength):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        IndexedSigmaConceptsFromPercepts(3, 32, l1_lambda=strength)


def test_cost_counts_distinct_concepts_not_occurrences_capacity_or_bias():
    layer = IndexedSigmaConceptsFromPercepts(3, 1_000_000, l1_lambda=.01)
    with torch.no_grad():
        layer.coefficients[3] = torch.tensor([.2, -.4, 500., 600.])
        layer.coefficients[7] = torch.tensor([.3, 700., 800., 900.])
    rows = torch.tensor([[3, 3, 7, -1]])
    roles = torch.tensor([[[0, 1, -1], [0, 1, -1], [1, -1, -1], [-1, -1, -1]]])
    selected, mask, strength, cost = layer.l1_batch(rows, roles)
    assert selected.tolist() == [3, 7]
    assert mask.tolist() == [[True, True, False, False], [True, False, False, False]]
    assert strength == .005
    torch.testing.assert_close(cost, torch.tensor(.0045))
    assert not cost.requires_grad and layer.coefficients.grad is None
    duplicate = layer.l1_batch(rows.repeat(5, 1), roles.repeat(5, 1, 1))
    for actual, expected in zip(duplicate, (selected, mask, strength, cost)):
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected
    assert layer.l1_batch(torch.tensor([-1]), torch.full((1, 3), -1)) is None
    layer.l1_lambda = 0.
    assert layer.l1_batch(rows, roles) is None


def _sparse_gradient(param, rows, values):
    return torch.sparse_coo_tensor(
        torch.tensor([rows], device=param.device),
        torch.tensor(values, dtype=param.dtype, device=param.device), param.shape,
        device=param.device)


@pytest.mark.parametrize("moment_dtype", [torch.float32, torch.bfloat16])
def test_adaptive_prox_matches_diagonal_metric_oracle_and_preserves_bias(moment_dtype):
    p = nn.Parameter(torch.zeros(64, 4))
    with torch.no_grad():
        p[3] = torch.tensor([.02, -.2, .005, .002])
    opt = RowLocalAdam([p], lr=.01, betas=(.8, .9), eps=1e-5,
                       moment_dtype=moment_dtype)
    mask = torch.tensor([[True, True, True, False]])
    for step, gradient in enumerate(([.001, -.03, 1.2, .005],
                                     [-.004, .1, -.3, .02]), start=1):
        opt.zero_grad()
        state = opt.state.get(p, {})
        old_m = state["exp_avg"][3].float() if state else torch.zeros(4)
        old_v = state["exp_avg_sq"][3].float() if state else torch.zeros(4)
        g = torch.tensor(gradient)
        m, v = .8 * old_m + .2 * g, .9 * old_v + .1 * g.square()
        denominator = (v / (1 - .9 ** step)).sqrt() + 1e-5
        y = p.detach()[3] - .01 * (m / (1 - .8 ** step)) / denominator
        threshold = .01 * .005 / denominator
        expected = torch.where(mask[0], y.sign() * (y.abs() - threshold).clamp_min(0), y)
        configure_l1_proximal(opt, p, torch.tensor([3]), mask, .005)
        p.grad = _sparse_gradient(p, [0, 3], [[0., 0., 0., 0.], gradient])
        opt.step()
        torch.testing.assert_close(p[3], expected, atol=1e-7, rtol=2e-6)
        assert torch.equal(p[0], torch.zeros(4))  # unknown placeholder is not observed
        torch.testing.assert_close(p[3, 3], expected[3])  # only the smooth Adam step
        if step == 1:
            assert p[3, 0] == 0  # a true zero, not merely a small sigmoid gate
        # KKT conditions for the weighted quadratic + coordinate L1 subproblem.
        residual = denominator / .01 * (p.detach()[3] - y)
        nonzero = mask[0] & p.detach()[3].ne(0)
        torch.testing.assert_close(residual[nonzero], -.005 * p.detach()[3, nonzero].sign(),
                                   atol=2e-6, rtol=1e-4)
        zero = mask[0] & ~nonzero
        assert bool((residual[zero].abs() <= .005 + 2e-6).all())
        assert not opt.inner._l1_proximal


@pytest.mark.parametrize("empty_gradient", [False, True])
def test_zero_smooth_gradient_shrinks_only_observed_admitted_connections(empty_gradient):
    p = nn.Parameter(torch.ones(1_000_000, 4))
    with torch.no_grad():
        p[3] = torch.tensor([.001, .005, .3, .0003])
    opt = RowLocalAdam([p], lr=.1, eps=.1)
    p.grad = (
        torch.sparse_coo_tensor(torch.empty(1, 0, dtype=torch.long),
                                torch.empty(0, 4), p.shape)
        if empty_gradient else _sparse_gradient(p, [0, 3], [[0.] * 4, [0.] * 4]))
    configure_l1_proximal(opt, p, torch.tensor([3]),
                         torch.tensor([[True, True, False, False]]), .01)
    opt.step()
    torch.testing.assert_close(p[3], torch.tensor([0., 0., .3, .0003]))
    assert torch.equal(p[0], torch.ones(4)) and torch.equal(p[900_000], torch.ones(4))
    assert opt.state[p]["exp_avg"].numel() == 4 * 4
    assert opt.state[p]["exp_avg_sq"].numel() == 4 * 4


def test_missing_gradient_and_zero_grad_cannot_replay_stale_l1_policy():
    p = nn.Parameter(torch.ones(8, 3))
    opt = RowLocalAdam([p], lr=.1, eps=.1)
    rows, mask = torch.tensor([2]), torch.tensor([[True, True, False]])
    configure_l1_proximal(opt, p, rows, mask, .01)
    opt.step()  # entirely disconnected parameter is not trained by sparsity alone
    assert torch.equal(p, torch.ones_like(p)) and not opt.inner._l1_proximal
    configure_l1_proximal(opt, p, rows, mask, .01)
    opt.zero_grad()  # also models the boundary after an AMP-skipped step
    p.grad = _sparse_gradient(p, [2], [[0., 0., 0.]])
    opt.step()
    assert torch.equal(p, torch.ones_like(p))
    assert not opt.state


def test_l1_is_opt_in_and_unsupported_optimizer_fails_loudly():
    p = nn.Parameter(torch.ones(8, 3))
    with pytest.raises(TypeError, match="RowLocalAdam"):
        configure_l1_proximal(Adam([p], lr=.01), p, torch.tensor([2]),
                             torch.tensor([[True, True, False]]), .01)
    opt = RowLocalAdam([p], lr=.1)
    with pytest.raises(ValueError, match="sorted, unique"):
        configure_l1_proximal(opt, p, torch.tensor([2, 2]),
                             torch.ones(2, 3, dtype=torch.bool), .01)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        configure_l1_proximal(opt, p, torch.tensor([2]),
                             torch.ones(1, 3, dtype=torch.bool), float("nan"))


def test_shared_model_readout_gets_one_policy_and_no_duplicate_autograd_l1():
    layer = IndexedSigmaConceptsFromPercepts(2, 16, l1_lambda=.01)
    cs = SimpleNamespace(concepts_from_percepts=layer)
    model = SimpleNamespace(
        conceptualSpaces=[cs, cs, cs], inputSpace=SimpleNamespace(
            _ar_word_concept_rows=torch.tensor([[3, 3]]),
            _ar_percept_reference_roles=torch.tensor([[[0, 1], [0, 1]]])))
    head = nn.Parameter(torch.tensor(1.))
    inner = RowLocalAdam(layer.parameters(), lr=.01)
    opt = MultiOptimizer([Adam([head], lr=.01), inner])
    opt.zero_grad()
    cost = BaseModel._configure_concept_readout_l1(model, opt)
    torch.testing.assert_close(cost, torch.tensor(.01))
    assert len(inner.inner._l1_proximal) == 1 and not cost.requires_grad
    coefficient = layer.lookup_coefficients(torch.tensor([3]))[0]
    reconstruction = 2 * coefficient[0]
    output = -3 * coefficient[0] + 4 * coefficient[1] + head
    backward_reconstruction_priority(
        reconstruction + output + cost, reconstruction, output, [layer.coefficients])
    torch.testing.assert_close(layer.coefficients.grad.coalesce().values(),
                               torch.tensor([[2., 1., 0.]]))
    # Reconstruction/output projection is unchanged; L1 is applied only later.
    opt.step()
    assert not inner.inner._l1_proximal


def test_proximal_checkpoint_resume_keeps_moments_but_not_old_observation_policy():
    p = nn.Parameter(torch.ones(16, 3))
    opt = RowLocalAdam([p], lr=.02)
    rows, mask = torch.tensor([3]), torch.tensor([[True, True, False]])
    p.grad = _sparse_gradient(p, [3], [[.2, -.3, .1]])
    configure_l1_proximal(opt, p, rows, mask, .01)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    restored_p = nn.Parameter(p.detach().clone())
    restored = RowLocalAdam([restored_p], lr=.02)
    restored.load_state_dict(saved)
    assert not restored.inner._l1_proximal
    for parameter, optimizer in ((p, opt), (restored_p, restored)):
        optimizer.zero_grad()
        configure_l1_proximal(optimizer, parameter, rows, mask, .01)
        parameter.grad = _sparse_gradient(parameter, [3], [[.1, -.1, .2]])
        optimizer.step()
    torch.testing.assert_close(restored_p, p)
    assert restored.state[restored_p]["step"] == opt.state[p]["step"] == 2


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_row_local_prox_on_mps():
    p = nn.Parameter(torch.full((32, 3), .001, device="mps"))
    opt = RowLocalAdam([p], lr=.1, eps=.1, moment_dtype=torch.bfloat16)
    configure_l1_proximal(opt, p, torch.tensor([3], device="mps"),
                         torch.tensor([[True, True, False]], device="mps"), .01)
    p.grad = _sparse_gradient(p, [0, 3], [[0., 0., 0.], [0., 0., 0.]])
    opt.step()
    torch.testing.assert_close(p[3].cpu(), torch.tensor([0., 0., .001]))
    torch.testing.assert_close(p[0].cpu(), torch.full((3,), .001))


def test_common_amp_scale_does_not_scale_proximal_threshold():
    outcomes = []
    for scale in (1., 1024.):
        layer = IndexedSigmaConceptsFromPercepts(2, 16, l1_lambda=.01)
        opt = RowLocalAdam(layer.parameters(), lr=.01)
        rows, roles = torch.tensor([3]), torch.tensor([[0, 1]])
        batch = layer.l1_batch(rows, roles)
        configure_l1_proximal(opt, layer.coefficients, *batch[:3])
        prediction = layer(torch.tensor([[.2, .5]]), rows)
        (scale * (prediction.square().sum() + batch[3])).backward()
        # GradScaler unscales before calling the optimizer. L1 was never
        # differentiated, so neither lambda nor its threshold is AMP-scaled.
        layer.coefficients.grad = layer.coefficients.grad / scale
        opt.step()
        outcomes.append(layer.coefficients.detach())
    torch.testing.assert_close(*outcomes, rtol=0, atol=0)


def test_weak_l1_fits_two_reference_concepts_with_exact_sparse_support():
    from bench_concepts_from_percepts import make_problem, decode
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        problem = make_problem(2, seed=1, train_size=128, test_size=512)
        layer = IndexedSigmaConceptsFromPercepts(8, 8, l1_lambda=.01)
        opt = RowLocalAdam(layer.parameters(), lr=.01)
        rows = torch.arange(8)
        roles = torch.tensor([[0] * 4 + [1] * 4]).expand(8, -1)
        for _ in range(500):
            opt.zero_grad()
            batch = layer.l1_batch(rows, roles)
            configure_l1_proximal(opt, layer.coefficients, *batch[:3])
            coefficients = layer.lookup_coefficients(rows)
            predictions = SigmaConceptsFromPercepts.from_coefficients(
                problem["train"][:, None, :], coefficients[None, :, :])
            loss = (decode(predictions, problem) - problem["train"]).square().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            predictions = SigmaConceptsFromPercepts.from_coefficients(
                problem["test"][:, None, :], layer.coefficients[None, :, :])
            error = (decode(predictions, problem) - problem["test"]).square().mean()
        assert error < .002
        assert (layer.coefficients[:, :-1].ne(0).sum(-1) == 2).all()
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("detached_reverse", [False, True])
def test_real_runbatch_stages_l1_once_and_reports_it_separately(
        tmp_path, monkeypatch, detached_reverse):
    from test_compiled_word_chunk import _tiny_canonical_model
    model = _tiny_canonical_model(tmp_path, monkeypatch, forward_grammar_weight=.25)
    # Test both live encoder credit and the shipped detached boundary. Weak
    # L1 must not silently reconnect that boundary or train by shrinkage alone.
    model.detached_reverse = detached_reverse
    # Leaf distillation is a separate, lazily materialized auxiliary head;
    # this test needs only the coupled reconstruction/readout objective.
    model.leaf_distill_weight = 0.0
    layer = model.conceptualSpaces[0].concepts_from_percepts
    assert layer.l1_lambda == .01
    optimizer = model.getOptimizer(lr=.001)
    leaf = next(leaf.inner for leaf in getattr(optimizer, "optimizers", [optimizer])
                if any(p is layer.coefficients for group in leaf.param_groups for p in group["params"]))
    calls, steps = [], []
    original_stage, original_step = leaf.set_l1_proximal, optimizer.step

    def stage(*args, **kwargs):
        calls.append(True)
        return original_stage(*args, **kwargs)

    def step(*args, **kwargs):
        expected = layer.l1_batch(model.inputSpace._ar_word_concept_rows,
                                  model.inputSpace._ar_percept_reference_roles)
        # Trie promotion can make a previously seen spelling an honest
        # unknown until its new support is committed at the boundary.
        assert (layer.coefficients in leaf._l1_proximal) == (expected is not None)
        steps.append(expected is not None)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(leaf, "set_l1_proximal", stage)
    monkeypatch.setattr(optimizer, "step", step)
    for train in (True, False, True):
        before = layer.coefficients.detach().clone()
        result, _ = model.runBatch(
            train=train, batchNum=0, batchSize=2, split="train" if train else "test",
            optimizer=optimizer,
            batch_override=(model.inputSpace.prepInput(["alpha beta gamma", "delta epsilon"]),
                            torch.empty(2, 0)))
        assert result is not None and not leaf._l1_proximal
        terms = model.teacher.errors.terms()
        matches = [term for term in terms if term[0] == "concept_readout_l1"]
        if train and steps[-1]:
            assert len(matches) == 1 and not matches[0][1].requires_grad
            assert matches[0][4] == "reg"
            if detached_reverse:
                assert layer.coefficients.grad is None
                assert layer.coefficients not in leaf.state
                torch.testing.assert_close(layer.coefficients, before, atol=0, rtol=0)
            else:
                assert layer.coefficients in leaf.state
                assert leaf.state[layer.coefficients]["step"] > 0
                assert not torch.equal(layer.coefficients, before)
        if not train:
            assert not matches
            torch.testing.assert_close(layer.coefficients, before, atol=0, rtol=0)
    assert len(steps) == 2 and len(calls) == sum(steps) >= 1
