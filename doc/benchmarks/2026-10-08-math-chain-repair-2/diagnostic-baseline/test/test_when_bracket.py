""".when endpoint-sum BRACKET convention (2026-06-16 redesign).

``.when`` is the same endpoint-sum bracket as ``.where`` over a span of model
TIME ``[start, end]``: the ANGLE decodes the event-time center, the MAGNITUDE the
event duration. An INSTANT (start == end) is the present stamp (magnitude 1).
Tense is the interval-vs-now relation, applied by shifting the event-time center
(``shift_time``); the former magnitude-D tense / aspect-interval duration schemes
are retired (aspect is a no-op).
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import WhenRangeEncoding, _WHEN_TENSE_STEP, _WHEN_PERIOD


def _enc(): return WhenRangeEncoding(_WHEN_PERIOD, 2)


# --- encode -> instant, decode recovers (center, extent) --------------------
def test_instant_decodes_time_and_zero_extent():
    enc = _enc()
    for t in (-1.0, 0.0, 1.0, float(_WHEN_PERIOD // 8)):
        key = enc.encode(t)                          # instant
        assert math.isclose(float(key.norm()), 1.0, abs_tol=1e-6)
        c, ext = enc.decode(key)
        assert math.isclose(float(c), t, abs_tol=0.05)
        assert math.isclose(float(ext), 0.0, abs_tol=1e-3)


def test_bracket_decodes_center_and_duration():
    enc = _enc()
    key = enc.encode(2000, 2400)                      # center 2200, duration 400
    c, ext = enc.decode(key)
    assert math.isclose(float(c), 2200.0, abs_tol=0.1)
    assert math.isclose(float(ext), 400.0, abs_tol=0.5)
    ds, de = enc.decode_span(key)
    assert math.isclose(float(ds), 2000.0, abs_tol=0.5)
    assert math.isclose(float(de), 2400.0, abs_tol=0.5)


def test_encode_tensor_input_round_trips_time():
    enc = _enc(); ts = torch.tensor([-1.0, 0.0, 1.0])
    c, ext = enc.decode(enc.encode(ts))
    assert torch.allclose(c, ts, atol=0.05)
    assert torch.allclose(ext, torch.zeros_like(ext), atol=1e-3)


# --- tense via the event-time center (shift_time), duration preserved -------
def test_tense_shifts_center_by_step_via_shift_time():
    enc = _enc()
    base = _WHEN_PERIOD // 8
    when = enc.encode(base).expand(1, 1, -1).clone()
    # FUTURE: center + step ticks.
    fut = enc.shift_time(when, +_WHEN_TENSE_STEP)
    c, ext = enc.decode(fut)
    assert math.isclose(float(c.reshape(-1)[0]), float(base) + _WHEN_TENSE_STEP, abs_tol=0.05)
    assert math.isclose(float(ext.reshape(-1)[0]), 0.0, abs_tol=1e-3)
    # PAST: center - step ticks.
    past = enc.shift_time(when, -_WHEN_TENSE_STEP)
    c2, _ext = enc.decode(past)
    assert math.isclose(float(c2.reshape(-1)[0]), float(base) - _WHEN_TENSE_STEP, abs_tol=0.05)


def test_aspect_layer_is_identity():
    from Language import AspectLayer
    enc = _enc()
    head = torch.randn(1, 1, 6)
    x = torch.cat([head, enc.encode(0.0).expand(1, 1, -1)], dim=-1)
    for kind in ("SIMPLE", "PERFECT", "PROGRESSIVE"):
        a = AspectLayer(); a.set_op(kind)
        assert torch.allclose(a.forward(x), x, atol=1e-7), kind   # no-op


# --- present-instant stamp -> [0, 1] at t=0 --------------------------------
def test_forward_stamps_present_instant():
    enc = _enc(); y = enc.forward(torch.zeros(2, 3, 10))
    c, ext = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(c), 0.0, abs_tol=1e-4)
    assert math.isclose(float(ext), 0.0, abs_tol=1e-3)


if __name__ == "__main__":
    unittest.main()
