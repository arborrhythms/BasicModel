"""Endpoint-sum ``.where`` span key for the PS analyzer.

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md ("Where
Encoding And Spans"): ``where = phase(start) + phase(end)``; angle decodes
the span center, magnitude decodes the span length. Invertible as long as
the namespace keeps centers in one period and lengths below half the period.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch


def test_endpoint_sum_where_decodes_span():
    """Encode then decode recovers (start, end) on the integer grid."""
    from perceptual_analyzer import EndpointSumWhere
    enc = EndpointSumWhere(namespace=256)
    for (s, e) in [(0, 3), (5, 6), (10, 42), (100, 128), (0, 1), (7, 7)]:
        w = enc.encode(s, e)
        assert tuple(w.shape) == (2,)
        ds, de = enc.decode(w)
        assert (ds, de) == (s, e), (s, e, ds, de)


def test_endpoint_sum_where_rebase_preserves_length():
    """Rebasing a span to a different center keeps the magnitude / decoded
    length (length lives in the radius, center in the angle)."""
    from perceptual_analyzer import EndpointSumWhere
    enc = EndpointSumWhere(namespace=256)
    w1 = enc.encode(10, 14)   # center 12, length 4
    w2 = enc.encode(40, 44)   # center 42, length 4 (rebased)
    r1 = float(torch.linalg.vector_norm(w1))
    r2 = float(torch.linalg.vector_norm(w2))
    assert abs(r1 - r2) < 1e-5, (r1, r2)
    s1, e1 = enc.decode(w1)
    s2, e2 = enc.decode(w2)
    assert (e1 - s1) == (e2 - s2) == 4
    assert s1 != s2


def test_endpoint_sum_where_rejects_ambiguous_period():
    """A span whose center/length approach or exceed the recoverable
    half-period is flagged not-recoverable and does not round-trip."""
    from perceptual_analyzer import EndpointSumWhere
    enc = EndpointSumWhere(namespace=8)   # half-period = 16
    assert enc.is_recoverable(0, 4)
    assert not enc.is_recoverable(0, 100)
    s, e = enc.decode(enc.encode(0, 100))
    assert (s, e) != (0, 100)


def test_endpoint_sum_where_batched():
    """Decode works on a batched [..., 2] tensor of keys."""
    from perceptual_analyzer import EndpointSumWhere
    enc = EndpointSumWhere(namespace=256)
    spans = [(0, 3), (10, 42), (100, 128)]
    keys = torch.stack([enc.encode(s, e) for (s, e) in spans], dim=0)
    starts, ends = enc.decode(keys)
    assert [int(x) for x in starts] == [s for s, _ in spans]
    assert [int(x) for x in ends] == [e for _, e in spans]
