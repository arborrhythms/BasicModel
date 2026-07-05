"""Task 2 of the `.where`/`.when` encoding pass (2026-07-04 plan):
``WhenStartDurationEncoding`` -- the 4-dim `.when` v2 codec.

The DECIDED band shape (design doc, Alec FINAL 2026-07-04): two TRUE
quadrature pairs over ONE quantity (onset $s$), a 2-rung ladder

  $[\\sin(s\\omega_{lf}), \\cos(s\\omega_{lf}),
     \\sin(s\\omega_{hf}), \\cos(s\\omega_{hf})]$

with $P_{lf} = $ ``<whenPeriod>`` (default $10^6$) and $P_{hf} = P_{lf} /$
``<whenRungRatio>`` (default 32). Duration is REMOVED from the band (it
was write-only; exact extents ride the record store). Constant norm
$\\sqrt{2}$; decode = atan2 per pair + HF branch resolution by LF;
``shift_time`` = exact phase rotation PER PAIR at its own $\\omega$; the
exact long-int clock (``BasicModel.when_time`` / ``self.t``) is the
addressing side-band and is UNTOUCHED (Option C hybrid).

Pure codec tests -- no model build.
"""

import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch

_P_LF = 10 ** 6
_RATIO = 32
_P_HF = _P_LF / _RATIO  # 31250


def _enc(n_when=4, period=_P_LF, ratio=_RATIO):
    from Spaces import WhenStartDurationEncoding
    return WhenStartDurationEncoding(period, n_when=n_when,
                                     rung_ratio=ratio)


def test_defaults_and_dims():
    """nDim == 4 enabled, 0 disabled; default period 1e6, rung ratio 32."""
    from Spaces import WhenStartDurationEncoding
    enc = WhenStartDurationEncoding(n_when=4)
    assert enc.nDim == 4
    assert enc.maxVal == _P_LF
    assert enc.rung_ratio == _RATIO
    off = WhenStartDurationEncoding(n_when=0)
    assert off.nDim == 0


def test_constant_norm_sqrt2_exact():
    """Both pairs exactly unit: ||encode(s)|| == sqrt(2) pinned exactly
    (float32 sin^2+cos^2 per pair)."""
    enc = _enc()
    for s in (0, 1, 17, 31250, 999999):
        v = enc.encode(s)
        n = float(torch.linalg.vector_norm(v))
        assert abs(n - math.sqrt(2.0)) < 1e-6, (s, n)


def test_lf_pair_roundtrips_onset_over_horizon():
    """LF pair alone recovers the onset over [0, P_lf) at float precision
    (measured float32 phase noise ~0.02 ticks at P=1e6; bound 0.5)."""
    enc = _enc()
    for s in (0, 1, 12345, 499999, 500001, 999998):
        v = enc.encode(s)
        a = torch.atan2(v[..., 0], v[..., 1]) % (2 * math.pi)
        s_lf = float(a) * _P_LF / (2 * math.pi)
        assert abs(s_lf - s) < 0.5, (s, s_lf)


def test_hf_pair_roundtrips_residue_within_rung():
    """HF pair recovers the onset residue mod P_hf within one rung."""
    enc = _enc()
    for s in (0, 3, 31249, 31250, 31251, 123456, 999999):
        v = enc.encode(s)
        a = torch.atan2(v[..., 2], v[..., 3]) % (2 * math.pi)
        r = float(a) * _P_HF / (2 * math.pi)
        want = s % _P_HF
        # residue is periodic: compare on the circle
        d = min(abs(r - want), _P_HF - abs(r - want))
        assert d < 1e-2, (s, r, want)


def test_ladder_decode_exact_on_integer_onsets():
    """The 2-rung ladder decode recovers integer onsets EXACTLY (after
    rounding) across the horizon -- LF localizes the branch, HF supplies
    the fine position."""
    enc = _enc()
    for s in (0, 1, 17, 31249, 31250, 31251, 499999, 731257, 999999):
        start, residue = enc.decode(enc.encode(s))
        assert round(float(start)) == s, (s, float(start))
        assert abs(float(residue)) < 1.0, (s, float(residue))


def test_ladder_branch_boundary():
    """Branch resolution is safe while the LF estimate localizes within
    HALF an HF period, and picks the wrong branch beyond it (the honest
    failure mode -- the design's safe bound)."""
    enc = _enc()
    s_true = 250000  # exactly 8 rungs
    v = enc.encode(s_true)
    w_lf = 2 * math.pi / _P_LF

    def _with_lf_error(delta_ticks):
        a = (s_true + delta_ticks) * w_lf
        out = v.clone()
        out[..., 0] = math.sin(a)
        out[..., 1] = math.cos(a)
        return out

    # just under half a rung: still exact
    start, _ = enc.decode(_with_lf_error(+_P_HF / 2 - 2))
    assert round(float(start)) == s_true
    start, _ = enc.decode(_with_lf_error(-_P_HF / 2 + 2))
    assert round(float(start)) == s_true
    # past half a rung: the neighbouring branch (off by exactly one P_hf)
    start, _ = enc.decode(_with_lf_error(+_P_HF / 2 + 2))
    assert round(float(start)) == s_true + _P_HF
    start, _ = enc.decode(_with_lf_error(-_P_HF / 2 - 2))
    assert round(float(start)) == s_true - _P_HF


def test_shift_time_rotates_both_pairs_coherently():
    """decode(shift_time(encode(s), dt)) == s + dt: each pair rotates at
    its OWN omega, so the ladder stays consistent under shifts."""
    enc = _enc()
    for s, dt in [(1000, 1), (1000, -1), (250000, 12345), (900000, 50000),
                  (31250, -31250)]:
        shifted = enc.shift_time(enc.encode(s), dt)
        start, residue = enc.decode(shifted)
        assert round(float(start)) == s + dt, (s, dt, float(start))
        assert abs(float(residue)) < 1.0
    # invertible: shift there and back is the identity
    v = enc.encode(4242)
    back = enc.shift_time(enc.shift_time(v, 777), -777)
    assert torch.allclose(v, back, atol=1e-5)


def test_batched_tensor_encode_decode():
    """Tensor-shaped onsets round-trip elementwise (the stamp sites pass
    [B, N] grids)."""
    enc = _enc()
    s = torch.tensor([[0.0, 17.0, 31250.0], [999999.0, 123.0, 62500.0]])
    v = enc.encode(s)
    assert tuple(v.shape) == (2, 3, 4)
    start, _ = enc.decode(v)
    assert torch.allclose(torch.round(start), s, atol=0), (start, s)


def test_exact_clock_untouched():
    """The encoding carries the synced ``.t`` reference (a plain int the
    model clock propagates) and owns NO clock state of its own: encode
    does not read or advance ``.t``; ``forward`` stamps the PRESENT
    instant from it."""
    enc = _enc()
    assert enc.t == 0
    enc.t = 41
    v0 = enc.encode(100)
    assert enc.t == 41, "encode must not touch the clock reference"
    x = torch.zeros([2, 3, 12])
    y = enc.forward(x)
    idx = enc.resolve(12)
    start, _ = enc.decode(y[0, 0, idx])
    assert round(float(start)) == 41
    assert enc.t == 41, "forward must not advance the clock (runBatch does)"
    enc.increment(5)
    assert enc.t == 41, "increment stays a no-op (clock lives on BasicModel)"
    assert v0 is not None


def test_tense_relation_keys_on_lf_pair():
    """next()/previous() move the PRESENT instant by whole ticks; the
    LF pair alone orders them against now (tense derivation)."""
    enc = _enc()
    enc.t = 500000
    now = enc.encode(enc.t)
    fut = enc.next(3)
    past = enc.previous(3)
    w_lf = 2 * math.pi / _P_LF

    def _lf_angle(v):
        return float(torch.atan2(v[..., 0], v[..., 1]) % (2 * math.pi))

    assert _lf_angle(fut) > _lf_angle(now) > _lf_angle(past)
    # float32 angle ULP near pi ~ 2.4e-7 rad ~ 0.04 LF ticks at P=1e6.
    assert abs((_lf_angle(fut) - _lf_angle(now)) / w_lf - 3.0) < 0.05


def test_no_endpoint_bracket_api():
    """Endpoint-sum bracketing retires for .when: the v2 codec exposes no
    decode_span, and encode takes ONE quantity (no end/D kwargs)."""
    enc = _enc()
    assert not hasattr(enc, "decode_span")
    with pytest.raises(TypeError):
        enc.encode(3, 7)
    with pytest.raises(TypeError):
        enc.encode(3, D=0.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
