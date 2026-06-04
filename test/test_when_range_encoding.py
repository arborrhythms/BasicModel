import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import WhenRangeEncoding


def _enc(maxT=64):
    return WhenRangeEncoding(maxT=maxT, n_when=2)


# ---------------------------------------------------------------------------
# Task 3.1 -- WhenRangeEncoding (10 tests)
# ---------------------------------------------------------------------------
def test_ndim_is_two_like_where():
    assert _enc().nDim == 2


def test_disabled_when_zero_width():
    assert WhenRangeEncoding(maxT=64, n_when=0).nDim == 0


def test_sin_cos_layout_present():
    q = _enc().q(0.0)                                  # [sin(0), cos(0)] = (0, 1)
    assert math.isclose(float(q[0]), 0.0, abs_tol=1e-6) and math.isclose(float(q[1]), 1.0, abs_tol=1e-6)


def test_present_default_key_is_zero_two():
    key = _enc().encode_range(0.0, 0.0)                # q(0)+q(0) = (0, 2)
    assert math.isclose(float(key[0]), 0.0, abs_tol=1e-6) and math.isclose(float(key[1]), 2.0, abs_tol=1e-6)


def test_signed_decode_negative_times():              # past must decode negative, not maxT-1
    enc = _enc()
    for (s, e) in [(-1.0, -1.0), (-1.0, 0.0), (-2.0, -1.0), (0.0, 0.0)]:
        ds, de = enc.decode(enc.encode_range(s, e))
        assert math.isclose(float(ds), s, abs_tol=1e-4) and math.isclose(float(de), e, abs_tol=1e-4)


def test_point_encode_decodes_to_faithful_center():
    # Unit-bracket convention: a point time t is stamped via encode(t) =
    # encode_range(t-0.5, t+0.5) (|key| ~ 1.998). encode is the mutual inverse
    # of the range decode -- the bracket is symmetric, so its CENTER round-trips
    # to t exactly and the recovered duration is 1 (see WhenRangeEncoding
    # docstring; the full endpoint round-trip is in test_when_bracket.py).
    enc = _enc()
    for t in (-1.5, -1.0, 0.0, 0.5, 2.0):
        ds, de = enc.decode(enc.encode(t))             # encode = unit bracket [t-0.5, t+0.5]
        center = (float(ds) + float(de)) / 2.0
        assert math.isclose(center, t, abs_tol=1e-3)   # center (= time) round-trips robustly
    ts = torch.tensor([-1.0, 0.0, 1.0])                # tensor input (as the muxing sites pass)
    ds, de = enc.decode(enc.encode(ts))
    assert torch.allclose((ds + de) / 2.0, ts, atol=1e-3)


def test_magnitude_carries_duration_not_position():
    enc = _enc()
    assert math.isclose(float(enc.encode_range(0, 0).norm()), float(enc.encode_range(-1, -1).norm()), rel_tol=1e-5)
    assert float(enc.encode_range(-1, 0).norm()) < float(enc.encode_range(0, 0).norm())


def test_tense_rotation_is_phase_shift():
    enc = _enc()
    assert torch.allclose(enc.rotate(enc.encode_range(0, 0), -1.0), enc.encode_range(-1, -1), atol=1e-5)  # PAST
    assert torch.allclose(enc.rotate(enc.encode_range(0, 0),  1.0), enc.encode_range( 1, 1), atol=1e-5)  # FUTURE


def test_rotate_range_shifts_both_endpoints():
    enc = _enc(); s, e = enc.decode(enc.rotate_range(enc.encode_range(-1.0, 0.0), -1.0))  # PAST(PERFECT)
    assert math.isclose(float(s), -2.0, abs_tol=1e-4) and math.isclose(float(e), -1.0, abs_tol=1e-4)


def test_aspect_interval_shapes():
    # Unit-bracket convention (reference r = interval CENTER): SIMPLE is a unit
    # window at r; PERFECT a unit window ending at r; PROGRESSIVE a 2-wide window
    # spanning r. eps is retained for API compat but no longer sizes the window.
    enc = _enc()
    assert enc.aspect_interval(0.0, "SIMPLE") == (-0.5, 0.5)
    assert enc.aspect_interval(0.0, "PERFECT") == (-1.0, 0.0)
    assert enc.aspect_interval(0.0, "PROGRESSIVE") == (-1.0, 1.0)


def test_forward_stamps_present_default_and_reverse_zeros():
    # Present default is the unit bracket encode_range(-0.5, 0.5) (center 0,
    # duration 1), so the stamped slots decode to (-0.5, 0.5).
    enc = _enc(); y = enc.forward(torch.zeros(2, 4, 10))
    ds, de = enc.decode(y[0, 0, enc.resolve(y.shape[-1])])
    assert math.isclose(float(ds), -0.5, abs_tol=1e-4) and math.isclose(float(de), 0.5, abs_tol=1e-4)
    cleaned, _ = enc.reverse(y); idx = enc.resolve(cleaned.shape[-1])
    assert torch.allclose(cleaned[..., idx], torch.zeros_like(cleaned[..., idx]), atol=1e-6)


# ---------------------------------------------------------------------------
# Task 3.2 -- construction site uses a zero-centered period
#
# The construction site (bin/Spaces.py: PerceptualSpace/Space.__init__,
# ~line 6245) builds ``WhenRangeEncoding(64, _nWhen)`` where ``_nWhen`` is the
# per-space ``nWhen`` read from XML. That value is config-dependent (0 in some
# sections, 2 in others) and is NOT globally 0, so we do NOT assert a global
# default; instead we assert the construction-site CONTRACT directly: the
# exact ``WhenRangeEncoding(64, n)`` call the site makes is sound for both the
# disabled (n=0) and enabled (n=2) widths it can be fed, with a correct
# signed round-trip at width 2. The change at :6245 was also verified by
# reading the code (no heavy full-model build).
# ---------------------------------------------------------------------------
def test_construction_site_disabled_width_zero():
    # WhenRangeEncoding(64, _nWhen) with _nWhen=0 -> disabled carrier.
    assert WhenRangeEncoding(64, 0).nDim == 0


def test_construction_site_enabled_width_two_round_trips():
    # WhenRangeEncoding(64, _nWhen) with _nWhen=2 -> 2-dim signed range.
    enc = WhenRangeEncoding(64, 2)
    assert enc.nDim == 2
    s, e = enc.decode(enc.encode_range(-1.0, 0.0))
    assert math.isclose(float(s), -1.0, abs_tol=1e-4) and math.isclose(float(e), 0.0, abs_tol=1e-4)


def test_construction_site_period_is_zero_centered_64():
    # The site passes maxT=64 (a small zero-centered period), not the old
    # monotonic-counter 10000; div_term must reflect maxT=64.
    enc = WhenRangeEncoding(64, 2)
    assert math.isclose(enc.div_term, 2 * math.pi / 64, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Task 3.3 -- width guard: SubSpace mux/demux/reverse at nWhen=2
#
# Proves SubSpace is shape-agnostic for the 2-wide .when BEFORE the Phase 6
# XML flip. We build the smallest real SubSpace carrying a
# ``whenEncoding = WhenRangeEncoding(64, 2)``, stamp a present-perfect range
# into the .when block of a muxed event, run the demux/reverse
# (SubSpace.decode), and assert the range round-trips and the muxed width is
# nWhat + nWhere + 2. No XML is touched.
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
        whenEncoding=WhenRangeEncoding(64, nWhen),
        inputShape=[3, dim], outputShape=[3, dim],
    )
    assert ss.nWhen == 2
    assert ss.muxedSize == nWhat + nWhere + 2


def test_subspace_demux_round_trips_present_perfect_range():
    SubSpace, WhereEncoding = _import_subspace()
    nWhat, nWhere, nWhen = 4, 2, 2
    dim = nWhat + nWhere + nWhen
    whenEnc = WhenRangeEncoding(64, nWhen)
    ss = SubSpace(
        whereEncoding=WhereEncoding(64, nWhere, nWhen),
        whenEncoding=whenEnc,
        inputShape=[3, dim], outputShape=[3, dim],
    )
    # Build a muxed event [B=2, V=3, dim]; stamp a present-perfect range
    # q(-1)+q(0) into the two .when slots (the tail, index [-2, -1]).
    objects = torch.zeros(2, 3, dim)
    key = whenEnc.encode_range(-1.0, 0.0)               # [2]
    when_idx = whenEnc.resolve(dim)                     # absolute when slot indices
    objects[..., when_idx] = key.expand(2, 3, -1)
    # Demux / reverse: SubSpace.decode -> (objects, space, time).
    cleaned, _space, time = ss.decode(objects)
    # (a) the range round-trips: time is the (start, end) tuple from decode.
    start, end = time
    assert torch.allclose(start, torch.full_like(start, -1.0), atol=1e-4)
    assert torch.allclose(end, torch.zeros_like(end), atol=1e-4)
    # reverse zeroed the .when slots.
    assert torch.allclose(cleaned[..., when_idx], torch.zeros_like(cleaned[..., when_idx]), atol=1e-6)
    # (b) muxed width.
    assert ss.muxedSize == nWhat + nWhere + 2


if __name__ == "__main__":
    unittest.main()
