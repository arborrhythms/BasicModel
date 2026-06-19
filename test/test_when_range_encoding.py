"""WhenRangeEncoding -- endpoint-sum BRACKET temporal feature (2026-06-16 .when
redesign). ``.when`` is the same endpoint-sum key as ``.where`` over a span of
model TIME ``[start, end]``:

    .when = 0.5 * [sin(s*dt)+sin(e*dt), cos(s*dt)+cos(e*dt)],  dt = 2*pi/period

so the ANGLE decodes the event-time CENTER and the MAGNITUDE the event DURATION
(extent). An INSTANT (start == end == t) collapses to ``[sin(t*dt), cos(t*dt)]``
(magnitude 1). TENSE is the interval-vs-now relation (not a magnitude); the
former magnitude-D tense scheme is retired. The serialized
``BasicModel.when_time`` long clock owns the exact time; the angle is the coarse
(folded) feature.
"""
import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import WhenRangeEncoding, _WHEN_PERIOD, _WHEN_TENSE_STEP


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
# encode: endpoint-sum bracket; an instant is magnitude 1
# ---------------------------------------------------------------------------
def test_instant_at_t0_is_unit_cos_axis():
    # encode(0) is the instant [sin0, cos0] = [0, 1] (magnitude 1).
    key = _enc().encode(0)
    assert math.isclose(float(key[0]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(key[1]), 1.0, abs_tol=1e-6)
    assert math.isclose(float(key.norm()), 1.0, abs_tol=1e-6)


def test_components_stay_in_unit_interval():
    # Every component is in [-1, 1] (network-friendly) for any instant time and
    # any bracket (endpoint-sum scaled by 0.5 keeps |comp| <= 1).
    enc = _enc()
    for s in (0, 1, _WHEN_PERIOD // 4, _WHEN_PERIOD // 2, 12345):
        assert float(enc.encode(s).abs().max()) <= 1.0 + 1e-6
        for e in (s, s + 10, s + 300):
            assert float(enc.encode(s, e).abs().max()) <= 1.0 + 1e-6, (s, e)


def test_back_compat_D_kwarg_is_ignored():
    # The retired magnitude-tense ``D`` kwarg is accepted (legacy call sites) and
    # has NO effect -- encode(t) == encode(t, D=anything).
    enc = _enc()
    base = enc.encode(7)
    for D in (0.0, 0.3, 0.5, 1.0):
        assert torch.allclose(enc.encode(7, D=D), base, atol=0)


# ---------------------------------------------------------------------------
# decode: recovers (center, extent); the angle tracks absolute time
# ---------------------------------------------------------------------------
def test_decode_instant_recovers_time_and_zero_extent():
    enc = _enc()
    # Times across the period; an instant decodes to (t, 0) with t faithful in
    # the non-aliasing window (-period/2, period/2).
    for t in (0, 1, _WHEN_PERIOD // 8, _WHEN_PERIOD // 4, -_WHEN_PERIOD // 8):
        c, ext = enc.decode(enc.encode(t))
        assert math.isclose(float(c), float(t), abs_tol=0.05), t
        assert math.isclose(float(ext), 0.0, abs_tol=1e-3), t


def test_decode_bracket_recovers_center_and_duration():
    enc = _enc()
    # Durations above the ~15-tick resolution floor decode faithfully.
    for s, e in ((1000, 1100), (5000, 5400), (200, 700)):
        c, ext = enc.decode(enc.encode(s, e))
        assert math.isclose(float(c), (s + e) / 2.0, abs_tol=0.1), (s, e)
        assert math.isclose(float(ext), float(e - s), abs_tol=0.5), (s, e)
        ds, de = enc.decode_span(enc.encode(s, e))
        assert math.isclose(float(ds), float(s), abs_tol=0.5), (s, e)
        assert math.isclose(float(de), float(e), abs_tol=0.5), (s, e)


def test_short_duration_snaps_to_instant():
    # Below the resolution floor (~15 ticks for period 65536) a span reads as a
    # zero-extent instant at its center (no spurious arccos-noise extent).
    enc = _enc()
    c, ext = enc.decode(enc.encode(1000, 1005))
    assert math.isclose(float(ext), 0.0, abs_tol=1e-3)
    assert math.isclose(float(c), 1002.5, abs_tol=0.1)


def test_decode_tensor_batch():
    enc = _enc()
    ts = torch.tensor([0.0, 100.0, float(_WHEN_PERIOD // 8)])
    c, ext = enc.decode(enc.encode(ts))          # instants
    assert torch.allclose(c, ts, atol=0.05)
    assert torch.allclose(ext, torch.zeros_like(ext), atol=1e-3)


def test_angle_tracks_absolute_time_via_self_t():
    # forward stamps encode(self.t) (an instant); the decoded center == self.t.
    T = _WHEN_PERIOD // 8
    enc = _enc(); enc.t = T
    y = enc.forward(torch.zeros(2, 3, 10))
    idx = enc.resolve(y.shape[-1])
    c, ext = enc.decode(y[0, 0, idx])
    assert math.isclose(float(c), float(T), abs_tol=0.05)
    assert math.isclose(float(ext), 0.0, abs_tol=1e-3)


def test_decode_zero_phasor_is_finite():
    # Decoding the all-zero edge .when must not produce NaN/Inf.
    enc = _enc()
    c, ext = enc.decode(torch.zeros(2, 2, 2))
    assert torch.isfinite(c).all() and torch.isfinite(ext).all()


# ---------------------------------------------------------------------------
# next() / previous(): advance / retreat the event time by the tense step
# ---------------------------------------------------------------------------
def test_next_previous_move_time_by_step():
    base = _WHEN_PERIOD // 8
    enc = _enc(); enc.t = base
    c, _ext = enc.decode(enc.next())
    assert math.isclose(float(c), base + _WHEN_TENSE_STEP, abs_tol=0.05)
    c, _ext = enc.decode(enc.previous())
    assert math.isclose(float(c), base - _WHEN_TENSE_STEP, abs_tol=0.05)


# ---------------------------------------------------------------------------
# shift_time on a .when bracket: rotate the center, preserve the duration
# ---------------------------------------------------------------------------
def test_shift_time_moves_center_preserves_extent():
    enc = _enc()
    when = enc.encode(1000, 1200).expand(2, 3, -1).clone()    # center 1100, dur 200
    out = enc.shift_time(when, +50.0)
    c, ext = enc.decode(out)
    assert torch.allclose(c, torch.full_like(c, 1150.0), atol=0.2)
    assert torch.allclose(ext, torch.full_like(ext, 200.0), atol=0.5)


def test_shift_time_is_invertible():
    enc = _enc()
    when = enc.encode(_WHEN_PERIOD // 8).expand(1, 1, -1).clone()
    out = enc.shift_time(enc.shift_time(when, +7.0), -7.0)
    assert torch.allclose(out, when, atol=1e-5)


# ---------------------------------------------------------------------------
# forward stamp + reverse zeroing
# ---------------------------------------------------------------------------
def test_forward_stamps_present_instant_and_reverse_zeros():
    enc = _enc(); y = enc.forward(torch.zeros(2, 4, 10))
    c, ext = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(c), 0.0, abs_tol=1e-4)
    assert math.isclose(float(ext), 0.0, abs_tol=1e-3)
    cleaned, decoded = enc.reverse(y)
    idx = enc.resolve(cleaned.shape[-1])
    assert torch.allclose(cleaned[..., idx], torch.zeros_like(cleaned[..., idx]), atol=1e-6)
    # reverse returns the decoded (center, extent) tuple from the new contract.
    c2, ext2 = decoded
    assert c2.shape == y.shape[:2] and ext2.shape == y.shape[:2]


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
    ws = SubSpace(
        whereEncoding=WhereEncoding(64, nWhere, nWhen),
        whenEncoding=WhenRangeEncoding(_WHEN_PERIOD, nWhen),
        inputShape=[3, dim], outputShape=[3, dim],
    )
    assert ws.nWhen == 2
    assert ws.muxedSize == nWhat + nWhere + 2


def test_subspace_demux_round_trips_present_instant():
    SubSpace, WhereEncoding = _import_subspace()
    nWhat, nWhere, nWhen = 4, 2, 2
    dim = nWhat + nWhere + nWhen
    whenEnc = WhenRangeEncoding(_WHEN_PERIOD, nWhen)
    T = _WHEN_PERIOD // 8
    whenEnc.t = T
    ws = SubSpace(
        whereEncoding=WhereEncoding(64, nWhere, nWhen),
        whenEncoding=whenEnc,
        inputShape=[3, dim], outputShape=[3, dim],
    )
    # Stamp the present instant at time T into the .when slots.
    objects = torch.zeros(2, 3, dim)
    key = whenEnc.encode(T)      # [2]
    when_idx = whenEnc.resolve(dim)
    objects[..., when_idx] = key.expand(2, 3, -1)
    # Demux / reverse: SubSpace.decode -> (objects, space, time); time is the
    # new (center, extent) tuple.
    cleaned, _space, time = ws.decode(objects)
    c_dec, ext_dec = time
    assert torch.allclose(c_dec, torch.full_like(c_dec, float(T)), atol=0.05)
    assert torch.allclose(ext_dec, torch.zeros_like(ext_dec), atol=1e-3)
    # reverse zeroed the .when slots.
    assert torch.allclose(cleaned[..., when_idx], torch.zeros_like(cleaned[..., when_idx]),
                          atol=1e-6)
    assert ws.muxedSize == nWhat + nWhere + 2


if __name__ == "__main__":
    unittest.main()
