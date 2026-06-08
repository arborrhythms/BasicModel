"""WhenRangeEncoding -- single scaled quadrature phasor (2026-06-07 .when
redesign, FINAL). ``.when = D * [sin(2*pi*t/period), cos(2*pi*t/period)]``:
the ANGLE encodes the absolute model time ``t`` and the MAGNITUDE ``D in
[0, 1]`` is the TENSE position (0=past, 0.5=PRESENT default, 1=future). Event
duration is retired -- the magnitude is tense now. The serialized
``BasicModel.when_time`` long clock owns the exact time; ``.when``'s angle is
the coarse (folded) feature.
"""
import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import (WhenRangeEncoding, _WHEN_PERIOD, _WHEN_TENSE_DEFAULT,
                    _WHEN_TENSE_STEP)


def _enc(maxT=_WHEN_PERIOD):
    return WhenRangeEncoding(maxT=maxT, n_when=2)


# ---------------------------------------------------------------------------
# Shape / disablement
# ---------------------------------------------------------------------------
def test_ndim_is_two_like_where():
    assert _enc().nDim == 2


def test_disabled_when_zero_width():
    assert WhenRangeEncoding(maxT=_WHEN_PERIOD, n_when=0).nDim == 0


# ---------------------------------------------------------------------------
# encode: scaled phasor, present default magnitude 0.5
# ---------------------------------------------------------------------------
def test_present_default_at_t0_is_zero_half():
    # encode(0) with the default D=0.5 -> 0.5 * [sin0, cos0] = [0, 0.5].
    key = _enc().encode(0)
    assert math.isclose(float(key[0]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(key[1]), _WHEN_TENSE_DEFAULT, abs_tol=1e-6)


def test_components_stay_in_unit_interval():
    # Every component is in [-1, 1] (network-friendly) for any time / any D.
    enc = _enc()
    for t in (0, 1, _WHEN_PERIOD // 4, _WHEN_PERIOD // 2, 12345):
        for D in (0.0, 0.4, 0.5, 0.6, 1.0):
            key = enc.encode(t, D=D)
            assert float(key.abs().max()) <= 1.0 + 1e-6, (t, D, key.tolist())


def test_magnitude_is_tense_D_not_position():
    # The magnitude equals D regardless of the time t (it is tense, not a
    # function of when): |encode(t, D)| == D for several distinct t.
    enc = _enc()
    for t in (0, 7, _WHEN_PERIOD // 8, _WHEN_PERIOD // 3):
        for D in (0.0, 0.3, 0.5, 1.0):
            assert math.isclose(float(enc.encode(t, D=D).norm()), D, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# decode: recovers (t, D); the angle tracks absolute time
# ---------------------------------------------------------------------------
def test_decode_recovers_t_and_D():
    enc = _enc()
    # t kept inside the non-aliasing window (-period/2, period/2) so atan2 is
    # faithful; D over the full tense range.
    for t in (0, 1, _WHEN_PERIOD // 8, _WHEN_PERIOD // 4, -_WHEN_PERIOD // 8):
        for D in (0.2, 0.5, 0.8, 1.0):
            dt, dD = enc.decode(enc.encode(t, D=D))
            assert math.isclose(float(dt), float(t), abs_tol=0.05), (t, D)
            assert math.isclose(float(dD), D, abs_tol=1e-5), (t, D)


def test_decode_tensor_batch():
    enc = _enc()
    ts = torch.tensor([0.0, 100.0, float(_WHEN_PERIOD // 8)])
    dt, dD = enc.decode(enc.encode(ts))          # D defaults to 0.5
    assert torch.allclose(dt, ts, atol=0.05)
    assert torch.allclose(dD, torch.full_like(dD, _WHEN_TENSE_DEFAULT), atol=1e-5)


def test_angle_tracks_absolute_time_via_self_t():
    # forward stamps encode(self.t, D=0.5); the decoded angle equals self.t.
    T = _WHEN_PERIOD // 8
    enc = _enc(); enc.t = T
    y = enc.forward(torch.zeros(2, 3, 10))
    idx = enc.resolve(y.shape[-1])
    dt, dD = enc.decode(y[0, 0, idx])
    assert math.isclose(float(dt), float(T), abs_tol=0.05)
    assert math.isclose(float(dD), _WHEN_TENSE_DEFAULT, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# next() / previous(): move the tense magnitude +/- step, preserve the angle
# ---------------------------------------------------------------------------
def test_next_previous_move_magnitude_by_step():
    enc = _enc(); enc.t = 200
    enc.D = _WHEN_TENSE_DEFAULT
    enc.next()
    assert math.isclose(enc.D, _WHEN_TENSE_DEFAULT + _WHEN_TENSE_STEP, abs_tol=1e-9)
    enc.D = _WHEN_TENSE_DEFAULT
    enc.previous()
    assert math.isclose(enc.D, _WHEN_TENSE_DEFAULT - _WHEN_TENSE_STEP, abs_tol=1e-9)


def test_next_previous_clamp_to_unit_interval():
    enc = _enc()
    enc.D = 0.95
    enc.next(); enc.next()                       # 0.95 -> 1.0 (clamped, not 1.15)
    assert math.isclose(enc.D, 1.0, abs_tol=1e-9)
    enc.D = 0.05
    enc.previous(); enc.previous()               # 0.05 -> 0.0 (clamped, not -0.15)
    assert math.isclose(enc.D, 0.0, abs_tol=1e-9)


def test_next_preserves_time_angle():
    # next() moves only the magnitude; the decoded time (angle) is unchanged.
    base = _WHEN_PERIOD // 8
    enc = _enc(); enc.t = base; enc.D = _WHEN_TENSE_DEFAULT
    nt, nD = enc.decode(enc.next())
    assert math.isclose(float(nt), float(base), abs_tol=0.05)
    assert math.isclose(float(nD), _WHEN_TENSE_DEFAULT + _WHEN_TENSE_STEP, abs_tol=1e-5)
    enc.D = _WHEN_TENSE_DEFAULT
    pt, pD = enc.decode(enc.previous())
    assert math.isclose(float(pt), float(base), abs_tol=0.05)
    assert math.isclose(float(pD), _WHEN_TENSE_DEFAULT - _WHEN_TENSE_STEP, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# shift_tense on a .when tensor: rescale preserving angle; D=0 edge guarded
# ---------------------------------------------------------------------------
def test_shift_tense_rescales_and_preserves_angle():
    T = _WHEN_PERIOD // 8
    enc = _enc(); enc.t = T
    when = enc.encode(T, D=_WHEN_TENSE_DEFAULT).expand(2, 3, -1).clone()
    out = enc.shift_tense(when, +_WHEN_TENSE_STEP)
    dt, dD = enc.decode(out)
    assert torch.allclose(dt, torch.full_like(dt, float(T)), atol=0.05)
    assert torch.allclose(dD, torch.full_like(dD, _WHEN_TENSE_DEFAULT + _WHEN_TENSE_STEP),
                          atol=1e-5)


def test_shift_tense_clamps_at_bounds():
    enc = _enc(); enc.t = 10
    when = enc.encode(10, D=0.95).expand(1, 1, -1).clone()
    _t, dD = enc.decode(enc.shift_tense(when, +0.2))     # 0.95 + 0.2 -> 1.0
    assert math.isclose(float(dD.reshape(-1)[0]), 1.0, abs_tol=1e-5)


def test_shift_tense_from_zero_magnitude_reencodes():
    # D == 0 (.when == [0, 0]) has no angle; shifting up re-encodes from self.t
    # (guards the divide-by-zero). The recovered magnitude is the new D.
    enc = _enc(); enc.t = 50
    zero = torch.zeros(1, 1, 2)
    dt0, dD0 = enc.decode(zero)
    assert math.isclose(float(dD0), 0.0, abs_tol=1e-7)
    out = enc.shift_tense(zero, +0.5)
    dt, dD = enc.decode(out)
    assert math.isclose(float(dD.reshape(-1)[0]), 0.5, abs_tol=1e-5)
    assert math.isclose(float(dt.reshape(-1)[0]), 50.0, abs_tol=0.05)


def test_decode_zero_phasor_is_finite():
    # Decoding the all-zero edge .when must not produce NaN/Inf (atan2(0,0)=0).
    enc = _enc()
    dt, dD = enc.decode(torch.zeros(2, 2, 2))
    assert torch.isfinite(dt).all() and torch.isfinite(dD).all()


# ---------------------------------------------------------------------------
# forward stamp + reverse zeroing
# ---------------------------------------------------------------------------
def test_forward_stamps_present_default_and_reverse_zeros():
    enc = _enc(); y = enc.forward(torch.zeros(2, 4, 10))
    dt, dD = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(dt), 0.0, abs_tol=1e-4)
    assert math.isclose(float(dD), _WHEN_TENSE_DEFAULT, abs_tol=1e-5)
    cleaned, decoded = enc.reverse(y)
    idx = enc.resolve(cleaned.shape[-1])
    assert torch.allclose(cleaned[..., idx], torch.zeros_like(cleaned[..., idx]), atol=1e-6)
    # reverse returns the decoded (t, D) tuple from the new decode contract.
    dt2, dD2 = decoded
    assert dt2.shape == y.shape[:2] and dD2.shape == y.shape[:2]


# ---------------------------------------------------------------------------
# period is the module knob
# ---------------------------------------------------------------------------
def test_period_is_module_knob():
    enc = WhenRangeEncoding(n_when=2)
    assert math.isclose(enc.div_term, 2 * math.pi / _WHEN_PERIOD, rel_tol=1e-12)
    enc2 = WhenRangeEncoding(maxT=4096, n_when=2)
    assert math.isclose(enc2.div_term, 2 * math.pi / 4096, rel_tol=1e-12)


def test_default_period_is_65536():
    assert _WHEN_PERIOD == 65536


# ---------------------------------------------------------------------------
# Width guard: SubSpace mux/demux/reverse at nWhen=2 (no XML touched)
# ---------------------------------------------------------------------------
def _import_subspace():
    from Spaces import SubSpace, WhereEncoding
    return SubSpace, WhereEncoding


def test_subspace_muxed_size_includes_two_when_slots():
    SubSpace, WhereEncoding = _import_subspace()
    nWhat, nWhere, nWhen = 4, 2, 2
    dim = nWhat + nWhere + nWhen
    ss = SubSpace(
        whereEncoding=WhereEncoding(64, nWhere, nWhen),
        whenEncoding=WhenRangeEncoding(_WHEN_PERIOD, nWhen),
        inputShape=[3, dim], outputShape=[3, dim],
    )
    assert ss.nWhen == 2
    assert ss.muxedSize == nWhat + nWhere + 2


def test_subspace_demux_round_trips_present_phasor():
    SubSpace, WhereEncoding = _import_subspace()
    nWhat, nWhere, nWhen = 4, 2, 2
    dim = nWhat + nWhere + nWhen
    whenEnc = WhenRangeEncoding(_WHEN_PERIOD, nWhen)
    T = _WHEN_PERIOD // 8
    whenEnc.t = T
    ss = SubSpace(
        whereEncoding=WhereEncoding(64, nWhere, nWhen),
        whenEncoding=whenEnc,
        inputShape=[3, dim], outputShape=[3, dim],
    )
    # Stamp the present phasor at time T (magnitude 0.5) into the .when slots.
    objects = torch.zeros(2, 3, dim)
    key = whenEnc.encode(T, D=_WHEN_TENSE_DEFAULT)      # [2]
    when_idx = whenEnc.resolve(dim)
    objects[..., when_idx] = key.expand(2, 3, -1)
    # Demux / reverse: SubSpace.decode -> (objects, space, time); time is the
    # new (t, D) tuple from decode.
    cleaned, _space, time = ss.decode(objects)
    t_dec, D_dec = time
    assert torch.allclose(t_dec, torch.full_like(t_dec, float(T)), atol=0.05)
    assert torch.allclose(D_dec, torch.full_like(D_dec, _WHEN_TENSE_DEFAULT), atol=1e-4)
    # reverse zeroed the .when slots.
    assert torch.allclose(cleaned[..., when_idx], torch.zeros_like(cleaned[..., when_idx]),
                          atol=1e-6)
    assert ss.muxedSize == nWhat + nWhere + 2


if __name__ == "__main__":
    unittest.main()
