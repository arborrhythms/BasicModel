"""Model time + serialized absolute clock for the .when encoding.

Implements the tests for doc/plans/2026-06-07-model-time-when-encoding.md
(FINAL single-scaled-quadrature-phasor scheme):

  1. A 0-initialized ``long`` model clock (``when_time``) that increments
     once per processed batch on BOTH train and inference, and that
     serializes through ``state_dict`` / ``save_weights`` / ``load_weights``.
  2. ``WhenRangeEncoding`` stamps ``.when = D * [sin(2*pi*t/period),
     cos(2*pi*t/period)]``: the ANGLE encodes the absolute model time
     (``self.t``, synced from ``when_time``); the MAGNITUDE ``D in [0, 1]`` is
     the TENSE position (0.5 = present default). ``period`` defaults to 65536.
  3. ``next`` / ``previous`` move the tense magnitude by +/-0.1, preserving the
     time-angle.
  4. ``TenseLayer`` PAST/PRESENT/FUTURE compose then reverse round-trip;
     ``AspectLayer`` is a no-op.

CPU + eager (``MODEL_COMPILE=eager``): the clock / encoding semantics are
device-independent and the round-trip exercises the real (small) model.
"""
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest
import torch

_P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_P / "bin"))

from Spaces import (WhenRangeEncoding, _WHEN_PERIOD, _WHEN_TENSE_DEFAULT,
                    _WHEN_TENSE_STEP)


_DATA = _P / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _enc(maxT=None, t=0):
    enc = (WhenRangeEncoding(n_when=2) if maxT is None
           else WhenRangeEncoding(maxT=maxT, n_when=2))
    enc.t = t
    return enc


def _build_xor_model():
    """Smallest real model config (MM_xor.xml), eager, on CPU."""
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(_DATA / "MM_xor.xml"),
                defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    m, _ = Models.BaseModel.from_config(str(_DATA / "MM_xor.xml"), data=TheData)
    return m.to("cpu")


def _one_batch(m):
    """A single (inputTensor, outputTensor) override for runBatch."""
    loader = m.inputSpace.data.data_loader(split="train", num_streams=2)
    inp_items, out_items = next(iter(loader))
    inputTensor = m.inputSpace.prepInput(inp_items)
    outputTensor = m.outputSpace.prepOutput(out_items)
    return inputTensor, outputTensor


# ---------------------------------------------------------------------------
# period default + module knob
# ---------------------------------------------------------------------------
def test_when_period_is_a_module_knob():
    # The period is a single module constant the construction sites read, so a
    # caller can retune it in one place. Default is 65536 (the FINAL choice).
    enc = WhenRangeEncoding(n_when=2)
    assert _WHEN_PERIOD == 65536
    assert math.isclose(enc.div_term, 2 * math.pi / _WHEN_PERIOD, rel_tol=1e-12)
    # Explicit-arg construction still honors a caller-supplied period.
    enc2 = WhenRangeEncoding(maxT=4096, n_when=2)
    assert math.isclose(enc2.div_term, 2 * math.pi / 4096, rel_tol=1e-12)


def test_tense_magnitude_resolution_above_float32_eps():
    # The angle step per time-tick (2*pi/period), scaled by the present D~0.5,
    # stays well above float32 eps so adjacent ticks are distinguishable.
    enc = _enc(t=0)
    a0 = enc.encode(0, D=_WHEN_TENSE_DEFAULT)
    a1 = enc.encode(1, D=_WHEN_TENSE_DEFAULT)
    sep = float((a1 - a0).abs().max())
    assert sep > 100 * 1.2e-7, f"adjacent-tick separation {sep} too small for float32"


# ---------------------------------------------------------------------------
# Encoding: present default + absolute angle via self.t
# ---------------------------------------------------------------------------
def test_forward_at_t0_is_present_default_zero_half():
    # At self.t == 0 the stamped .when is the present default 0.5*[0,1]=[0,0.5].
    enc = _enc(t=0)
    x = torch.zeros(2, 4, 10)
    y = enc.forward(x)
    idx = enc.resolve(y.shape[-1])
    stamped = y[0, 0, idx]
    expect = torch.tensor([0.0, _WHEN_TENSE_DEFAULT], device=stamped.device)
    assert torch.allclose(stamped, expect, atol=1e-6)
    # encode(0) at the present default magnitude equals the same vector.
    assert torch.allclose(enc.encode(enc.t, D=_WHEN_TENSE_DEFAULT), expect, atol=1e-6)


def test_forward_stamps_absolute_time_in_angle():
    # The decoded angle (time) is faithful inside the (-period/2, period/2)
    # window; the magnitude is the present tense default.
    T = _WHEN_PERIOD // 8
    enc = _enc(t=T)
    x = torch.zeros(2, 3, 10)
    y = enc.forward(x)
    idx = enc.resolve(y.shape[-1])
    t_dec, D = enc.decode(y[0, 0, idx])
    assert math.isclose(float(t_dec), float(T), abs_tol=0.05)
    assert math.isclose(float(D), _WHEN_TENSE_DEFAULT, abs_tol=1e-5)


def test_encode_at_T_decodes_angle_to_T():
    T = _WHEN_PERIOD // 8
    enc = _enc(t=0)  # self.t is irrelevant to encode(T) (explicit arg)
    t_dec, D = enc.decode(enc.encode(T))
    assert math.isclose(float(t_dec), float(T), abs_tol=0.05)
    assert math.isclose(float(D), _WHEN_TENSE_DEFAULT, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# next() / previous(): move the tense magnitude, preserve the time-angle
# ---------------------------------------------------------------------------
def test_next_previous_move_magnitude_preserving_time():
    base = _WHEN_PERIOD // 8
    enc = _enc(t=base)
    enc.D = _WHEN_TENSE_DEFAULT
    t_dec, D = enc.decode(enc.next())
    assert math.isclose(float(D), _WHEN_TENSE_DEFAULT + _WHEN_TENSE_STEP, abs_tol=1e-5)
    assert math.isclose(float(t_dec), float(base), abs_tol=0.05)
    enc.D = _WHEN_TENSE_DEFAULT
    t_dec, D = enc.decode(enc.previous())
    assert math.isclose(float(D), _WHEN_TENSE_DEFAULT - _WHEN_TENSE_STEP, abs_tol=1e-5)
    assert math.isclose(float(t_dec), float(base), abs_tol=0.05)


def test_next_previous_clamp_to_unit_interval():
    enc = _enc(t=10)
    enc.D = 0.95; enc.next(); enc.next()
    assert math.isclose(enc.D, 1.0, abs_tol=1e-9)
    enc.D = 0.05; enc.previous(); enc.previous()
    assert math.isclose(enc.D, 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Tense / aspect
# ---------------------------------------------------------------------------
def test_tense_layer_round_trips():
    import Language
    head = torch.randn(2, 3, 4)
    enc = _enc(t=_WHEN_PERIOD // 8)
    when = enc.encode(enc.t, D=_WHEN_TENSE_DEFAULT).expand(2, 3, -1)
    x = torch.cat([head, when], dim=-1)
    for op in ("PAST", "PRESENT", "FUTURE"):
        layer = Language.TenseLayer()
        layer.set_op(op)
        y = layer.forward(x)
        back = layer.reverse(y)
        assert torch.allclose(back, x, atol=1e-5), op


def test_tense_layer_present_is_identity_and_past_future_shift_D():
    import Language
    T = _WHEN_PERIOD // 8
    enc = _enc(t=T)
    head = torch.randn(1, 1, 4)
    when = enc.encode(T, D=_WHEN_TENSE_DEFAULT).expand(1, 1, -1)
    x = torch.cat([head, when], dim=-1)
    pres = Language.TenseLayer(); pres.set_op("PRESENT")
    assert torch.allclose(pres.forward(x), x, atol=1e-6)
    past = Language.TenseLayer(); past.set_op("PAST")
    _t, D = enc.decode(past.forward(x)[..., -2:])
    assert math.isclose(float(D.reshape(-1)[0]), _WHEN_TENSE_DEFAULT - _WHEN_TENSE_STEP,
                        abs_tol=1e-5)
    fut = Language.TenseLayer(); fut.set_op("FUTURE")
    _t, D = enc.decode(fut.forward(x)[..., -2:])
    assert math.isclose(float(D.reshape(-1)[0]), _WHEN_TENSE_DEFAULT + _WHEN_TENSE_STEP,
                        abs_tol=1e-5)


def test_aspect_layer_is_noop():
    # AspectLayer is RETIRED to a no-op by this redesign (duration is gone).
    import Language
    enc = _enc(t=_WHEN_PERIOD // 8)
    head = torch.randn(1, 1, 6)
    x = torch.cat([head, enc.encode(enc.t).expand(1, 1, -1)], dim=-1)
    for kind in ("SIMPLE", "PERFECT", "PROGRESSIVE"):
        a = Language.AspectLayer(); a.set_op(kind)
        assert torch.allclose(a.forward(x), x, atol=1e-7), kind


# ---------------------------------------------------------------------------
# Model clock
# ---------------------------------------------------------------------------
def test_fresh_model_present_is_zero():
    m = _build_xor_model()
    assert m.present() == 0
    assert int(m.when_time) == 0


def test_when_time_is_long_buffer_in_state_dict():
    m = _build_xor_model()
    sd = m.state_dict()
    assert "when_time" in sd, sorted(k for k in sd if "when" in k)
    assert sd["when_time"].dtype == torch.long


def test_clock_increments_once_per_train_batch():
    m = _build_xor_model()
    opt = m.getOptimizer(lr=1e-4)
    N = 3
    for _ in range(N):
        m.runBatch(train=True, batchSize=2, split="train",
                   optimizer=opt, batch_override=_one_batch(m))
    assert m.present() == N


def test_clock_increments_once_per_inference_batch():
    m = _build_xor_model()
    m.eval()
    N = 4
    for _ in range(N):
        m.runBatch(train=False, batchSize=2, split="train",
                   batch_override=_one_batch(m))
    assert m.present() == N


def test_clock_increments_on_runtime_inference_only():
    # split="runtime" with train=False is the inference-only early-return
    # path; the clock must still tick once per batch there.
    m = _build_xor_model()
    m.eval()
    inp, _out = _one_batch(m)
    N = 2
    for _ in range(N):
        m.runBatch(train=False, batchSize=2, split="runtime",
                   batch_override=(inp, None))
    assert m.present() == N


# ---------------------------------------------------------------------------
# Sync: when_time reaches the live encoder used at stamping time
# ---------------------------------------------------------------------------
def test_when_time_syncs_to_live_encoders():
    m = _build_xor_model()
    opt = m.getOptimizer(lr=1e-4)
    N = 3
    for _ in range(N):
        m.runBatch(train=True, batchSize=2, split="train",
                   optimizer=opt, batch_override=_one_batch(m))
    # Every live (enabled) when-encoder on the model's spaces must have
    # had its reference time advanced to present().
    seen_live = False
    for sp in m.spaces:
        sub = getattr(sp, "subspace", None)
        enc = getattr(sub, "whenEncoding", None) if sub is not None else None
        if enc is not None and getattr(enc, "nDim", 0) > 0:
            seen_live = True
            assert int(getattr(enc, "t", 0)) == m.present(), (
                f"{type(sp).__name__}.subspace.whenEncoding.t "
                f"{getattr(enc, 't', None)} != present {m.present()}")
    assert seen_live, "expected at least one live when-encoder on the model"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------
def test_when_time_round_trips_through_save_load():
    m = _build_xor_model()
    opt = m.getOptimizer(lr=1e-4)
    N = 5
    for _ in range(N):
        m.runBatch(train=True, batchSize=2, split="train",
                   optimizer=opt, batch_override=_one_batch(m))
    assert m.present() == N

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "weights.ckpt")
        m.save_weights(ckpt)
        m2 = _build_xor_model()
        assert m2.present() == 0           # fresh
        m2.load_weights(ckpt)
        assert m2.present() == N, (
            f"when_time must survive save/load: got {m2.present()}, want {N}")


def test_old_checkpoint_without_when_time_loads_at_zero():
    # state_dict lacking when_time must load at 0 (strict=False path).
    m = _build_xor_model()
    opt = m.getOptimizer(lr=1e-4)
    for _ in range(2):
        m.runBatch(train=True, batchSize=2, split="train",
                   optimizer=opt, batch_override=_one_batch(m))
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "weights.ckpt")
        m.save_weights(ckpt)
        # Strip when_time to simulate a pre-feature checkpoint.
        bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
        bundle["state_dict"].pop("when_time", None)
        torch.save(bundle, ckpt)
        m2 = _build_xor_model()
        m2.load_weights(ckpt)
        assert m2.present() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
