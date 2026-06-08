""".when present-phasor + tense-magnitude convention (2026-06-07 redesign).

The single-stamp ``.when`` is the present scaled phasor ``encode(t, D=0.5) =
0.5 * [sin(2*pi*t/period), cos(2*pi*t/period)]``: the ANGLE encodes the
absolute time ``t``, the MAGNITUDE ``D in [0, 1]`` is the TENSE position
(0.5 = present default). The former unit-bracket / aspect-interval / duration
scheme is retired (the magnitude is tense now, not event duration); aspect is a
no-op. These tests re-express the old bracket tests against the new behavior.
"""

import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import WhenRangeEncoding, _WHEN_TENSE_DEFAULT, _WHEN_TENSE_STEP, _WHEN_PERIOD


def _enc(): return WhenRangeEncoding(_WHEN_PERIOD, 2)


# --- encode -> present scaled phasor, decode recovers (t, D) ----------------
def test_encode_is_present_phasor_and_decodes_t_and_D():
    enc = _enc()
    # Times inside the non-aliasing window so the angle decode is faithful.
    for t in (-1.0, 0.0, 1.0, float(_WHEN_PERIOD // 8)):
        key = enc.encode(t)                          # D defaults to 0.5
        # |key| == D == 0.5 (the present tense magnitude), NOT a duration.
        assert math.isclose(float(key.norm()), _WHEN_TENSE_DEFAULT, abs_tol=1e-6)
        dt, dD = enc.decode(key)
        assert math.isclose(float(dt), t, abs_tol=0.05)
        assert math.isclose(float(dD), _WHEN_TENSE_DEFAULT, abs_tol=1e-6)


def test_encode_tensor_input_round_trips_time():
    enc = _enc(); ts = torch.tensor([-1.0, 0.0, 1.0])
    dt, dD = enc.decode(enc.encode(ts))
    assert torch.allclose(dt, ts, atol=0.05)
    assert torch.allclose(dD, torch.full_like(dD, _WHEN_TENSE_DEFAULT), atol=1e-6)


# --- tense via the .when magnitude (next/previous), angle preserved ---------
def test_tense_magnitude_moves_by_step_via_shift_tense():
    enc = _enc(); enc.t = _WHEN_PERIOD // 8
    when = enc.encode(enc.t, D=_WHEN_TENSE_DEFAULT).expand(1, 1, -1).clone()
    # FUTURE: D + step (toward future), time-angle preserved.
    fut = enc.shift_tense(when, +_WHEN_TENSE_STEP)
    dt, dD = enc.decode(fut)
    assert math.isclose(float(dD.reshape(-1)[0]), _WHEN_TENSE_DEFAULT + _WHEN_TENSE_STEP,
                        abs_tol=1e-5)
    assert math.isclose(float(dt.reshape(-1)[0]), float(enc.t), abs_tol=0.05)
    # PAST: D - step (toward past).
    past = enc.shift_tense(when, -_WHEN_TENSE_STEP)
    _t, pD = enc.decode(past)
    assert math.isclose(float(pD.reshape(-1)[0]), _WHEN_TENSE_DEFAULT - _WHEN_TENSE_STEP,
                        abs_tol=1e-5)


def test_aspect_layer_is_identity():
    from Language import AspectLayer
    enc = _enc()
    head = torch.randn(1, 1, 6)
    x = torch.cat([head, enc.encode(0.0).expand(1, 1, -1)], dim=-1)
    for kind in ("SIMPLE", "PERFECT", "PROGRESSIVE"):
        a = AspectLayer(); a.set_op(kind)
        assert torch.allclose(a.forward(x), x, atol=1e-7), kind   # no-op


# --- present-default stamp -> [0, 0.5] at t=0 -------------------------------
def test_forward_stamps_present_phasor():
    enc = _enc(); y = enc.forward(torch.zeros(2, 3, 10))
    dt, dD = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(dt), 0.0, abs_tol=1e-4)
    assert math.isclose(float(dD), _WHEN_TENSE_DEFAULT, abs_tol=1e-6)


if __name__ == "__main__":
    unittest.main()
