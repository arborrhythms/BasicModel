"""Muxed WhereEncoding 2-rung LADDER (2026-07-09 multi-rung pass).

``.where`` is a 4-dim two-pair quadrature ladder over ONE quantity, the byte
START position (mirrors the `.when` v2 ladder): LF pair at ``<wherePeriod>``
(range), HF pair at ``wherePeriod/<whereRungRatio>`` (sub-byte resolution);
``decode`` = atan2 per pair + HF branch resolution by LF, minus the
``where_origin`` seam shift. The endpoint-sum bracket is RETIRED from the muxed
band (the END is content-terminated, Alec 2026-07-09); the analyzer's
``EndpointSumWhere`` span key (perceptual_analyzer.py) is a separate codec and
keeps the bracket -- test_ps_where.py covers it.
"""
import math, os, sys, unittest
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from Spaces import WhereEncoding, _WHERE_RUNG_RATIO


def _enc(maxP=8192, ratio=_WHERE_RUNG_RATIO):
    return WhereEncoding(maxP, 4, 4, rung_ratio=ratio)


def test_ladder_shape_and_rungs():
    enc = _enc()
    assert enc.nDim == 4
    assert enc.index == [-8, -7, -6, -5]           # before the 4 when slots
    assert math.isclose(enc.period_hf, 8192 / _WHERE_RUNG_RATIO)
    key = enc.encode(torch.tensor(37.0))
    assert key.shape[-1] == 4
    # Two TRUE pairs: each rung is a unit phasor (constant norm sqrt(2) total).
    assert math.isclose(float(key[..., :2].norm()), 1.0, abs_tol=1e-5)
    assert math.isclose(float(key[..., 2:].norm()), 1.0, abs_tol=1e-5)


def test_ladder_round_trip_exact_across_range():
    # Byte positions round-trip EXACTLY across the full period -- including
    # past the HF period (256), where the LF rung resolves the HF branch.
    # A single quadrature pair cannot do this: range and resolution trade off.
    enc = _enc()
    for p in (0, 1, 5, 6, 11, 255, 256, 257, 1000, 4000, 8000):
        d = float(enc.decode(enc.encode(torch.tensor(float(p)))))
        assert math.isclose(d, float(p), abs_tol=1e-2), (p, d)


def test_origin_shift_clears_the_seam():
    # Offset 0 is stamped OFF the LF angle-0 seam (where_origin, 1/64 period):
    # small negative phase noise must not wrap the decode to ~maxVal.
    enc = _enc()
    key = enc.encode(torch.tensor(0.0))
    noisy = key + torch.tensor([-0.02, 0.0, -0.02, 0.0])
    d = float(enc.decode(noisy))
    assert abs(d) < 3.0, d                          # near 0, NOT near 8192


def test_forward_stamp_round_trip():
    # The auto-counter forward stamp reverses to the stamped positions.
    enc = _enc(); enc.p = 0
    x = torch.zeros(2, 4, 16)
    y = enc.forward(x.clone())
    _cleaned, decoded = enc.reverse(y)
    for b in range(2):
        for v in range(4):
            assert math.isclose(float(decoded[b, v]), float(b * 4 + v),
                                abs_tol=1e-2), (b, v, float(decoded[b, v]))


def test_bracket_is_retired():
    # The muxed band no longer carries the endpoint-sum bracket: encode takes
    # ONE quantity (no end kwarg) and decode_span is gone.
    enc = _enc()
    assert not hasattr(enc, "decode_span")
    try:
        enc.encode(torch.tensor(10.0), torch.tensor(20.0))
        raised = False
    except TypeError:
        raised = True
    assert raised, "encode must reject an end argument (bracket retired)"


if __name__ == "__main__":
    unittest.main()
