"""PiLayer log-domain finiteness regression (root cause of the
reverse-from-STM NaN, doc/plans/2026-05-30-subsymbolic-analyzer-terminal-
emitter.md context).

Both halves of the ``nonlinear=False`` PiLayer fold take ``log`` of a
value that must be strictly positive:

  * FORWARD: ``log(_to_mult(x))`` where ``_to_mult(x) = (1+x)/(1-x)`` is
    positive only on the open interval ``(-1, 1)``. ``_to_mult`` formerly
    clamped its input ONLY when ``nonlinear=True``, so a percept outside
    ``[-1, 1]`` (legitimate -- percept normalization runs AFTER
    ``pi.forward``) drove the ratio ``<= 0`` and injected ``log(<=0) =
    NaN``. (Surfaced as "Finding A": the forward left the C-tier STM /
    ``_stm_single_S`` NON-FINITE.)

  * REVERSE: the ``nonlinear=False`` branch did a BARE ``log(y)`` on the
    signed reverse signal ``y in [-1, 1]``, so ``log(negative) = NaN``.
    (Surfaced as "Finding B": ``PerceptualSpace.reverse`` ->
    ``PiLayer.reverse`` turned a finite seed NaN.)

The fix makes the ``_to_mult`` clamp unconditional and clamps the reverse
log to its positive domain, using the overflow-safe ``tanh(lx/2)`` exit
(algebraically identical to ``_from_mult(exp(lx))``). Crucially the guard
is a DOMAIN clamp, not a NaN scrub: ``clamp`` leaves NaN/Inf untouched, so
a genuine upstream divergence still propagates and fails loud (user
memory: never silently nan_to_num / gate away Inf/NaN).

Existing ``test_invertibility.py`` only exercises the default
``nonlinear=True`` PiLayer, so this ``nonlinear=False`` defect was
uncovered. TEST-ONLY harness; no bin/ edits.
"""

import os
import sys

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch

import Layers
from util import TheDevice, init_device


@pytest.fixture(autouse=True, scope="module")
def _cpu_device():
    """Pin CPU for this module at RUN time, not import time.

    A module-level ``init_device`` executes during pytest COLLECTION and
    flips the process-wide default device under every module imported
    after it -- while modules imported before it already built their
    import-time globals on the original device. That frankenstate is
    exactly the cross-device mix behind the full-suite-only
    test_heat_reverse_wiring failures. As a run-time module fixture the
    pin is scoped: conftest's ``_restore_process_device`` guard unwinds
    it when this module's tests finish.
    """
    init_device("cpu")
    yield


def _pi(nonlinear, naive=True, n=4):
    layer = Layers.PiLayer(n, n, naive=naive, hasBias=True,
                           invertible=True, nonlinear=nonlinear)
    layer.set_sigma(0)
    return layer


# -- Finding A: forward stays finite for out-of-[-1,1] input --------------

@pytest.mark.parametrize("naive", [True, False])
def test_forward_finite_on_out_of_range_input(naive):
    """``PiLayer.forward`` (``nonlinear=False``) must stay finite even when
    fed an input outside [-1, 1] -- the case that used to inject
    ``log((1+x)/(1-x) <= 0) = NaN`` and poison the forward STM (Finding A)."""
    pi = _pi(nonlinear=False, naive=naive)
    # Values straddling and well outside the (-1, 1) domain, incl. the
    # exact singular endpoints +-1 of ``(1+x)/(1-x)``.
    x = torch.tensor([[[-2.5, -1.0, -0.3, 0.0],
                       [0.4, 1.0, 1.7, 3.2]]], device=TheDevice.get())
    with torch.no_grad():
        out = pi.forward(x)
    assert torch.isfinite(out).all(), (
        f"forward must stay finite on out-of-range input; got {out!r}")


# -- Finding B: reverse stays finite for signed / non-positive input ------

@pytest.mark.parametrize("naive", [True, False])
def test_reverse_finite_on_signed_input(naive):
    """``PiLayer.reverse`` (``nonlinear=False``) must stay finite even when
    fed a signed/non-positive ``y`` -- the case that used to inject
    ``log(y <= 0) = NaN`` in the perceptual reverse (Finding B). Output
    must also land in the recovered percept range [-1, 1]."""
    pi = _pi(nonlinear=False, naive=naive)
    y = torch.tensor([[[-0.8, -1e-9, 0.0, 0.3],
                       [-2.0, 0.5, -0.1, 0.9]]], device=TheDevice.get())
    with torch.no_grad():
        out = pi.reverse(y)
    assert torch.isfinite(out).all(), (
        f"reverse must stay finite on signed input; got {out!r}")
    assert bool((out.abs() <= 1.0 + 1e-4).all()), (
        f"reverse output must be in [-1, 1]; got range "
        f"[{out.min().item():.4f}, {out.max().item():.4f}]")


# -- fail-loud preserved: genuine NaN/Inf input still propagates ----------

def test_forward_nan_input_propagates_not_swallowed():
    """The domain clamp must NOT swallow a genuine non-finite input -- a
    NaN/Inf arriving at the layer is a real upstream divergence and must
    propagate (fail-loud), never be silently rescued by ``clamp``."""
    pi = _pi(nonlinear=False)
    x = torch.tensor([[[float("nan"), 0.1, float("inf"), -0.2]]],
                     device=TheDevice.get())
    with torch.no_grad():
        out = pi.forward(x)
    assert not torch.isfinite(out).all(), (
        "a NaN/Inf input must propagate through forward, not be swallowed "
        "by the log-domain clamp.")


def test_reverse_nan_input_propagates_not_swallowed():
    """Reverse counterpart: a non-finite ``y`` must propagate, not be
    silently clamped into a finite value."""
    pi = _pi(nonlinear=False)
    y = torch.tensor([[[float("nan"), 0.1, float("inf"), -0.2]]],
                     device=TheDevice.get())
    with torch.no_grad():
        out = pi.reverse(y)
    assert not torch.isfinite(out).all(), (
        "a NaN/Inf input must propagate through reverse, not be swallowed "
        "by the log-domain clamp.")


# -- no regression on the valid (in-domain) round trip --------------------

def test_roundtrip_accurate_on_in_domain_input():
    """The overflow-safe ``tanh(lx/2)`` exit is algebraically identical to
    ``_from_mult(exp(lx))``, so the genuine inverse on in-domain input is
    unchanged: ``reverse(forward(x)) ~= x`` for ``x in (-1, 1)``."""
    pi = _pi(nonlinear=False)
    x = torch.tensor([[[-0.6, -0.2, 0.1, 0.7]]], device=TheDevice.get())
    with torch.no_grad():
        y = pi.forward(x)          # in (0, inf) -- the positive mult-domain
        x_rec = pi.reverse(y)
    assert torch.isfinite(y).all() and torch.isfinite(x_rec).all()
    err = (x - x_rec).abs().max().item()
    assert err < 1e-3, f"in-domain round trip must be ~exact; max err {err:.2e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
