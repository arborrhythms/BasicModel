"""FlattenKWrapper round-trip + backward equivalence (Task 4 of microbatch plan)."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import torch.nn as nn

from Pipeline import FlattenKWrapper


class _ToySubSpace:
    """Minimal stand-in for SubSpace.

    Holds an event tensor + k_axis flag. The body's set_event/materialize
    contract is what FlattenKWrapper actually depends on; nothing else.
    """

    def __init__(self, x, k_axis=False):
        self._event = x
        self.k_axis = k_axis

    def materialize(self, mode=None):
        return self._event

    def set_event(self, x):
        self._event = x


class _ToyBody(nn.Module):
    """Body that linearly transforms the channel dim. Receives a subspace,
    reads .materialize() (shape [B*K, N, D]), writes back via set_event,
    flips k_axis False and returns the same subspace."""

    def __init__(self, D, Dout=None):
        super().__init__()
        Dout = Dout if Dout is not None else D
        self.lin = nn.Linear(D, Dout)

    def forward(self, sub):
        x = sub.materialize()
        sub.set_event(self.lin(x))
        sub.k_axis = False
        return sub


def test_flatten_k_wrapper_grad_matches_manual():
    """Wrapped path's gradients must match a manual flatten/reshape pass."""
    torch.manual_seed(0)
    B, K, N, D = 2, 3, 4, 5
    body = _ToyBody(D)

    x1 = torch.randn(B, K, N, D, requires_grad=True)
    sub1 = _ToySubSpace(x1, k_axis=True)
    out1 = FlattenKWrapper(body)(sub1).materialize()
    out1.sum().backward()
    g_wrapped = x1.grad.clone()

    x2 = x1.detach().clone().requires_grad_(True)
    sub2 = _ToySubSpace(x2.view(B * K, N, D), k_axis=False)
    out2 = body(sub2).materialize().view(B, K, N, D)
    out2.sum().backward()
    g_manual = x2.grad.clone()

    assert out1.shape == (B, K, N, D)
    assert torch.allclose(out1, out2, atol=1e-6)
    assert torch.allclose(g_wrapped, g_manual, atol=1e-6)


def test_flatten_k_wrapper_handles_dim_change():
    """Body that changes channel dim must be reshaped to [B, K, N, Dout]."""
    torch.manual_seed(1)
    B, K, N, D, Dout = 2, 3, 4, 5, 7
    body = _ToyBody(D, Dout=Dout)

    x = torch.randn(B, K, N, D, requires_grad=True)
    sub = _ToySubSpace(x, k_axis=True)
    out = FlattenKWrapper(body)(sub)

    assert out.materialize().shape == (B, K, N, Dout)
    assert out.k_axis is True


def test_flatten_k_wrapper_restores_k_axis():
    """k_axis flag round-trips True -> body sees False -> True on return."""
    torch.manual_seed(2)
    B, K, N, D = 1, 2, 3, 4
    seen_k_axis = []

    class _Probe(nn.Module):
        def forward(self, sub):
            seen_k_axis.append(sub.k_axis)
            return sub

    x = torch.randn(B, K, N, D)
    sub = _ToySubSpace(x, k_axis=True)
    out = FlattenKWrapper(_Probe())(sub)

    assert seen_k_axis == [False]
    assert out.k_axis is True


def test_flatten_k_wrapper_rejects_3d_input():
    """Asserts the [B, K, N, D] contract loudly when violated."""
    body = _ToyBody(4)
    sub = _ToySubSpace(torch.randn(2, 3, 4), k_axis=True)
    try:
        FlattenKWrapper(body)(sub)
    except AssertionError:
        return
    raise AssertionError("FlattenKWrapper should reject 3D input")


def test_cachepoint_reverse_is_identity():
    """CachePoint.reverse must be a pass-through so ReverseAdapter(cache) works
    inside pipeline_rev / pipeline_rt without raising AttributeError."""
    from Pipeline import CachePoint, ReverseAdapter

    cache = CachePoint()
    sub = _ToySubSpace(torch.randn(2, 3, 4), k_axis=False)
    out = ReverseAdapter(cache)(sub)
    assert out is sub, "CachePoint.reverse must return the same subspace"
