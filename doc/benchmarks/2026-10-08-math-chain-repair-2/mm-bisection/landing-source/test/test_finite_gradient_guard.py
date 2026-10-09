"""Always-on CPU/MPS finite-gradient preflight at the optimizer boundary."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import Optimizer as optimizer_module  # noqa: E402
from Optimizer import (Adam, MultiOptimizer, RowLocalAdam,  # noqa: E402
                       finite_gradient_guard_enabled,
                       preflight_finite_gradients)


@pytest.fixture(autouse=True)
def _auto_guard(monkeypatch):
    monkeypatch.setenv("MODEL_FINITE_GRAD_GUARD", "auto")


def _row_grad(size, rows, values, *, dtype=torch.float32):
    indices = torch.tensor([rows], dtype=torch.long)
    payload = torch.tensor(values, dtype=dtype)
    return torch.sparse_coo_tensor(indices, payload, size=size)


def test_finite_dense_gradient_updates_normally():
    param = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = Adam([param], lr=1e-2)
    before = param.detach().clone()

    param.square().sum().backward()
    optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.equal(param.detach(), before)


def test_single_optimizer_closure_runs_once_then_returns_its_loss():
    param = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = Adam([param], lr=1e-2)
    before = param.detach().clone()
    calls = 0
    losses = []

    def closure():
        nonlocal calls
        calls += 1
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = param.square().sum()
        loss.backward()
        losses.append(loss)
        return loss

    # torch optimizers promise to re-enable grad for closures even if their
    # caller is in a no-grad region.
    with torch.no_grad():
        returned = optimizer.step(closure=closure)

    assert calls == 1
    assert returned is losses[0]
    assert not torch.equal(param.detach(), before)


def test_single_optimizer_closure_is_preflighted_after_it_runs():
    param = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = Adam([param], lr=1e-2)
    param.grad = torch.ones_like(param)
    preflight_finite_gradients(optimizer, cache_for_step=True)
    before = param.detach().clone()
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        param.grad.fill_(float("nan"))
        return param.sum()

    with pytest.raises(FloatingPointError, match="2/2.*nan/inf"):
        optimizer.step(closure=closure)

    assert calls == 1
    torch.testing.assert_close(param, before, rtol=0.0, atol=0.0)
    assert not optimizer.state


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_dense_gradient_raises_without_any_mutation(bad):
    param = nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = Adam([param], lr=1e-2)
    param.grad = torch.tensor([0.25, bad])
    before = param.detach().clone()

    with pytest.raises(FloatingPointError, match="1/2.*nan/inf"):
        optimizer.step()

    torch.testing.assert_close(param, before, rtol=0.0, atol=0.0)
    assert not optimizer.state


def test_multi_optimizer_preflights_later_sparse_child_before_dense_step():
    dense = nn.Parameter(torch.tensor([1.0, 2.0]))
    sparse = nn.Parameter(torch.randn(8, 2))
    dense_leaf = Adam([dense], lr=1e-2)
    sparse_leaf = RowLocalAdam([sparse], lr=1e-2)
    optimizer = MultiOptimizer([dense_leaf, sparse_leaf])
    dense.grad = torch.ones_like(dense)
    sparse.grad = _row_grad(
        tuple(sparse.shape), [3], [[float("nan"), 1.0]])
    dense_before = dense.detach().clone()
    sparse_before = sparse.detach().clone()

    with pytest.raises(FloatingPointError, match="optimizer parameter 1"):
        optimizer.step()

    # In particular, child 0 did not partially commit before child 1 failed.
    torch.testing.assert_close(dense, dense_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(sparse, sparse_before, rtol=0.0, atol=0.0)
    assert not dense_leaf.state
    assert not sparse_leaf.state


def test_multi_optimizer_closure_runs_once_and_returns_its_loss():
    left = nn.Parameter(torch.tensor([1.0]))
    right = nn.Parameter(torch.tensor([-2.0]))
    left_leaf = Adam([left], lr=1e-2)
    right_leaf = Adam([right], lr=1e-2)
    optimizer = MultiOptimizer([left_leaf, right_leaf])
    before = (left.detach().clone(), right.detach().clone())
    calls = 0
    losses = []

    def closure():
        nonlocal calls
        calls += 1
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = left.square().sum() + right.square().sum()
        loss.backward()
        losses.append(loss)
        return loss

    with torch.no_grad():
        returned = optimizer.step(closure=closure)

    assert calls == 1
    assert returned is losses[0]
    assert not torch.equal(left.detach(), before[0])
    assert not torch.equal(right.detach(), before[1])


def test_multi_optimizer_closure_union_preflight_is_atomic():
    first = nn.Parameter(torch.tensor([1.0]))
    second = nn.Parameter(torch.tensor([2.0]))
    first_leaf = Adam([first], lr=1e-2)
    second_leaf = Adam([second], lr=1e-2)
    optimizer = MultiOptimizer([first_leaf, second_leaf])
    before = (first.detach().clone(), second.detach().clone())
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        first.grad = torch.ones_like(first)
        second.grad = torch.full_like(second, float("nan"))
        return first.sum() + second.sum()

    with pytest.raises(FloatingPointError, match="optimizer parameter 1"):
        optimizer.step(closure=closure)

    assert calls == 1
    torch.testing.assert_close(first, before[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(second, before[1], rtol=0.0, atol=0.0)
    assert not first_leaf.state
    assert not second_leaf.state


def test_large_sparse_gradient_scans_values_not_physical_capacity(monkeypatch):
    param = nn.Parameter(torch.zeros(1_000_000, 2))
    optimizer = RowLocalAdam([param], lr=1e-2)
    param.grad = _row_grad(
        tuple(param.shape), [7, 900_000], [[1.0, 2.0], [3.0, 4.0]])
    real_scan = torch._amp_foreach_non_finite_check_and_unscale_
    scanned = []

    def probe(tensors, found_inf, inv_scale):
        scanned.extend((tensor.layout, tensor.numel()) for tensor in tensors)
        return real_scan(tensors, found_inf, inv_scale)

    monkeypatch.setattr(
        torch, "_amp_foreach_non_finite_check_and_unscale_", probe)
    assert preflight_finite_gradients(optimizer)

    assert scanned == [(torch.strided, 4)]
    assert param.grad.layout == torch.sparse_coo
    assert param.grad.is_coalesced()


def test_sparse_duplicate_overflow_is_caught_after_coalesce():
    param = nn.Parameter(torch.zeros(8, 2))
    optimizer = RowLocalAdam([param], lr=1e-2)
    largest = torch.finfo(torch.float32).max
    param.grad = _row_grad(
        tuple(param.shape), [4, 4], [[largest, largest], [largest, largest]])
    before = param.detach().clone()

    with pytest.raises(FloatingPointError, match="2/2.*nan/inf"):
        optimizer.step()

    assert param.grad.is_coalesced()
    assert torch.isinf(param.grad.values()).all()
    torch.testing.assert_close(param, before, rtol=0.0, atol=0.0)
    assert not optimizer.state


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse-coo"])
def test_complex_gradients_are_scanned_through_real_views(monkeypatch, sparse):
    param = nn.Parameter(torch.zeros(4, 2, dtype=torch.complex64))
    optimizer = RowLocalAdam([param], lr=1e-2) if sparse else Adam(
        [param], lr=1e-2)

    def make_grad(bad=False):
        if sparse:
            values = [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]
            if bad:
                values[1][1] = complex(7.0, float("inf"))
            return _row_grad(
                tuple(param.shape), [1, 3], values,
                dtype=torch.complex64)
        grad = torch.ones_like(param)
        if bad:
            grad[3, 1] = complex(float("nan"), 2.0)
        return grad

    real_scan = torch._amp_foreach_non_finite_check_and_unscale_
    scanned = []

    def probe(tensors, found_inf, inv_scale):
        scanned.extend(
            (tensor.dtype, tuple(tensor.shape), tensor.is_complex())
            for tensor in tensors)
        return real_scan(tensors, found_inf, inv_scale)

    monkeypatch.setattr(
        torch, "_amp_foreach_non_finite_check_and_unscale_", probe)
    param.grad = make_grad()
    assert preflight_finite_gradients(optimizer)
    param.grad = make_grad(bad=True)
    logical_entries = 4 if sparse else 8
    with pytest.raises(
            FloatingPointError,
            match=rf"1/{logical_entries}.*nan/inf"):
        preflight_finite_gradients(optimizer)

    expected_shape = (2, 2, 2) if sparse else (4, 2, 2)
    assert scanned == [
        (torch.float32, expected_shape, False),
        (torch.float32, expected_shape, False),
    ]


def test_finite_row_local_sparse_gradient_updates_touched_row():
    param = nn.Parameter(torch.zeros(16, 3))
    optimizer = RowLocalAdam([param], lr=1e-2)
    param.grad = _row_grad(tuple(param.shape), [9], [[1.0, -2.0, 3.0]])

    optimizer.step()

    assert optimizer.state[param]["step"] == 1
    assert param.detach()[9].ne(0).all()
    assert not param.detach()[:9].ne(0).any()
    assert not param.detach()[10:].ne(0).any()


def test_guard_is_independent_of_model_debug(monkeypatch):
    import util

    monkeypatch.setattr(util, "MODEL_DEBUG", False)
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam([param], lr=1e-2)
    param.grad = torch.tensor([float("nan")])

    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        optimizer.step()


def test_runbatch_cache_skips_duplicate_scan_but_invalidates_on_grad_change(
        monkeypatch):
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam([param], lr=1e-2)
    param.grad = torch.tensor([1.0])
    real_scan = torch._amp_foreach_non_finite_check_and_unscale_
    calls = 0

    def probe(tensors, found_inf, inv_scale):
        nonlocal calls
        calls += 1
        return real_scan(tensors, found_inf, inv_scale)

    monkeypatch.setattr(
        torch, "_amp_foreach_non_finite_check_and_unscale_", probe)
    preflight_finite_gradients(
        optimizer, [("weight", param)], cache_for_step=True)
    optimizer.step()
    assert calls == 1

    optimizer.zero_grad()
    param.grad = torch.tensor([1.0])
    preflight_finite_gradients(
        optimizer, [("weight", param)], cache_for_step=True)
    param.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="'weight'"):
        optimizer.step()
    assert calls == 3


def test_policy_auto_disables_complete_cuda_optimizer_without_tensor_scan(
        monkeypatch):
    fake_cuda = SimpleNamespace(device=torch.device("cuda"))
    fake_optimizer = SimpleNamespace(
        param_groups=[{"params": [fake_cuda]}])
    assert not finite_gradient_guard_enabled(fake_optimizer)

    monkeypatch.setenv("MODEL_FINITE_GRAD_GUARD", "on")
    assert finite_gradient_guard_enabled(fake_optimizer)
    monkeypatch.setenv("MODEL_FINITE_GRAD_GUARD", "off")
    assert not finite_gradient_guard_enabled(fake_optimizer)


def test_policy_off_bypasses_fused_scan(monkeypatch):
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam([param], lr=1e-2)
    param.grad = torch.tensor([float("nan")])
    monkeypatch.setenv("MODEL_FINITE_GRAD_GUARD", "off")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled guard must not launch a tensor scan")

    monkeypatch.setattr(
        optimizer_module.torch,
        "_amp_foreach_non_finite_check_and_unscale_", forbidden)
    assert not preflight_finite_gradients(optimizer)


def test_invalid_policy_fails_loud(monkeypatch):
    param = nn.Parameter(torch.tensor([1.0]))
    optimizer = Adam([param], lr=1e-2)
    monkeypatch.setenv("MODEL_FINITE_GRAD_GUARD", "sometimes")
    with pytest.raises(ValueError, match="MODEL_FINITE_GRAD_GUARD"):
        preflight_finite_gradients(optimizer)
