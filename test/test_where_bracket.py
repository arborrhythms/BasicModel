"""Muxed WhereEncoding endpoint-sum BRACKET (2026-06-16 redesign).

``.where`` is the invertible endpoint-sum key over a span ``[start, end]`` (the
``EndpointSumWhere`` form adopted into the muxed event tail): the ANGLE decodes
the span center, the MAGNITUDE the span extent. An INSTANT (start == end)
collapses to the legacy single-quadrature point (byte-identical to the
pre-bracket stamp). ``decode`` returns the center (legacy positional contract);
``decode_span`` returns ``(start, end)`` for the mereology contiguity test.
"""
import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import WhereEncoding, QuadratureEncoding


def _enc(maxP=4096):
    return WhereEncoding(maxP, 2, 2)


# --- instant is a single-quadrature point at the ORIGIN-SHIFTED position -----
def test_instant_is_origin_shifted_legacy_point():
    # The instant is still a single-quadrature point (0.5*2 == 1 exact), now at
    # the ORIGIN-SHIFTED position (WhereEncoding.where_origin off the wrap seam,
    # 2026-07-09): encode(p) == the legacy quadrature point of (p + where_origin).
    enc = _enc()
    o = enc.where_origin
    for p in (0.0, 3.0, 37.0, 120.0, 999.0):
        new = enc.encode(torch.tensor(p))                      # end=None -> instant
        legacy = QuadratureEncoding.encode(enc, torch.tensor(p + o))
        assert torch.equal(new, legacy), (p, new, legacy)


def test_forward_stamp_is_origin_shifted_legacy():
    # The auto-counter forward stamp (instants) is the legacy quadrature stamp
    # at the origin-shifted positions -- the bracket capability is opt-in via
    # encode(s, e); the origin shift is applied uniformly by encode.
    enc = _enc(); enc.p = 0
    x = torch.zeros(2, 4, 12)
    y = enc.forward(x.clone())
    # Re-derive the legacy stamp for the same ORIGIN-SHIFTED positions.
    enc2 = _enc(); enc2.p = 0
    idx = enc2.resolve(12)
    pos = torch.arange(0, 2 * 4, dtype=torch.float32) + enc2.where_origin
    legacy = QuadratureEncoding.encode(enc2, pos).reshape(2, 4, 2)
    assert torch.allclose(y[..., idx], legacy, atol=0)


# --- bracket: center via decode (legacy), endpoints via decode_span ---------
def test_bracket_center_and_span():
    enc = _enc()
    for s, e in ((10, 20), (5, 40), (100, 130), (0, 250)):
        key = enc.encode(torch.tensor(float(s)), torch.tensor(float(e)))
        center = enc.decode(key)
        assert math.isclose(float(center), (s + e) / 2.0, abs_tol=0.05), (s, e)
        ds, de = enc.decode_span(key)
        assert math.isclose(float(ds), float(s), abs_tol=0.1), (s, e)
        assert math.isclose(float(de), float(e), abs_tol=0.1), (s, e)


def test_instant_decodes_to_point_with_zero_extent():
    enc = _enc()
    key = enc.encode(torch.tensor(42.0))
    assert math.isclose(float(enc.decode(key)), 42.0, abs_tol=0.05)
    ds, de = enc.decode_span(key)
    assert math.isclose(float(ds), 42.0, abs_tol=1e-2)
    assert math.isclose(float(de), 42.0, abs_tol=1e-2)


def test_batched_decode_span():
    enc = _enc()
    spans = [(10, 20), (50, 90), (200, 230)]
    keys = torch.stack([enc.encode(torch.tensor(float(s)), torch.tensor(float(e)))
                        for s, e in spans], dim=0)
    starts, ends = enc.decode_span(keys)
    for i, (s, e) in enumerate(spans):
        assert math.isclose(float(starts[i]), float(s), abs_tol=0.1), (s, e)
        assert math.isclose(float(ends[i]), float(e), abs_tol=0.1), (s, e)


# --- contiguity / containment: the mereology read off decode_span ----------
def test_containment_and_contiguity_via_span():
    enc = _enc()
    def span(s, e):
        return enc.decode_span(enc.encode(torch.tensor(float(s)), torch.tensor(float(e))))

    # A = [10, 50] contains B = [20, 30].
    a_s, a_e = span(10, 50)
    b_s, b_e = span(20, 30)
    assert float(a_s) <= float(b_s) + 0.1 and float(b_e) <= float(a_e) + 0.1

    # Two sibling parts [10, 30] and [30, 60] are CONTIGUOUS (no gap at 30).
    _p1s, p1e = span(10, 30)
    p2s, _p2e = span(30, 60)
    assert abs(float(p1e) - float(p2s)) < 1.0

    # [10, 30] and [40, 60] are DISCONTIGUOUS (a gap of ~10 between them).
    _q1s, q1e = span(10, 30)
    q2s, _q2e = span(40, 60)
    assert float(q2s) - float(q1e) > 5.0


if __name__ == "__main__":
    unittest.main()
