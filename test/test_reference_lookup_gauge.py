"""Stage 5 of doc/plans/MeronomyPlan.md: the reference-half lookup law.

MeronomySpec §3 / §10.3. Stored signs of reference rows are GAUGE (the
referent is positive content; the sign denotes nothing), fixed at mint
by ``gauge_orient`` (+u toward agreement with the positive referent).
The licensed reference-half lookup is therefore the pole quotient --
identity by ``argmax |q·v|``, polarity by ``sign(q·v)`` -- and
certainty survives retrieval end-to-end because ``(a·u)·v = a(u·v)``.
Ground/token codebooks keep full-vector lookup: there the sign IS form
content, and no NEG-quotient may leak in.

The aligned snap path (``Basis._snap_content``, sphere branch) is
knob-gated (<architecture><meronomy>) -- dark until cutover; knob-off
behavior is the legacy cosine snap, byte-identical.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from embed import _pole_aligned_score, _wrapped_mse_score
from Spaces import gauge_orient, Codebook

D, V = 6, 5


def unit_rows(seed=1, v=V, d=D):
    torch.manual_seed(seed)
    rows = torch.randn(v, d)
    return rows / rows.norm(dim=-1, keepdim=True)


def _knob(value):
    from util import TheXMLConfig
    if value is None:
        TheXMLConfig._data.get("architecture", {}).pop("meronomy", None)
    else:
        TheXMLConfig.set("architecture.meronomy", value)


# ---------------------------------------------------------------------------
# Gauge fixing at mint.
# ---------------------------------------------------------------------------

def test_gauge_orient_toward_positive_referent():
    torch.manual_seed(2)
    u = torch.randn(8, D)
    ref = torch.rand(8, D)                  # positive referents (§2)
    g = gauge_orient(u, ref)
    assert ((g * ref).sum(-1) >= 0).all(), "+u agrees with the referent"


def test_gauge_orient_quotient_consistency():
    torch.manual_seed(3)
    u = torch.randn(8, D)
    ref = torch.rand(8, D)
    g1 = gauge_orient(u, ref)
    g2 = gauge_orient(-u, ref)
    assert torch.allclose(g1, g2), (
        "u and -u are one reference: both representatives orient "
        "identically")
    assert torch.allclose(gauge_orient(g1, ref), g1), "idempotent"


def test_gauge_orient_zero_dot_keeps_row():
    u = torch.tensor([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]])
    ref = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])  # orthogonal
    assert torch.equal(gauge_orient(u, ref), u)


# ---------------------------------------------------------------------------
# §10.3: u ↦ −u relabeling is observationally invariant at the
# reference half; certainty and polarity survive retrieval.
# ---------------------------------------------------------------------------

def test_pole_lookup_relabeling_invariance():
    rows = unit_rows()
    torch.manual_seed(4)
    q = torch.randn(16, D)
    flip = torch.ones(V, 1)
    flip[1] = -1.0
    flip[3] = -1.0
    rows_flipped = rows * flip               # relabel two stored rows
    s1 = _pole_aligned_score(q, rows)
    s2 = _pole_aligned_score(q, rows_flipped)
    assert torch.allclose(s1, s2, atol=1e-6), "identity scores invariant"
    idx1, idx2 = s1.argmax(-1), s2.argmax(-1)
    assert torch.equal(idx1, idx2), "selection invariant"
    # The observable VALUE polarity·row is invariant too: sign(q·v)
    # flips together with v, so (sign · v) is fixed.
    v1 = rows[idx1]
    v2 = rows_flipped[idx2]
    p1 = torch.sign((q * v1).sum(-1, keepdim=True))
    p2 = torch.sign((q * v2).sum(-1, keepdim=True))
    assert torch.allclose(p1 * v1, p2 * v2, atol=1e-6), (
        "polarity-aligned retrieval is gauge-invariant")


def test_certainty_survives_retrieval():
    torch.manual_seed(5)
    u = torch.randn(D); u = u / u.norm()
    v = torch.randn(D); v = v / v.norm()
    for a in (-1.0, -0.25, 0.0, 0.5, 1.0):
        lhs = (a * u) @ v
        rhs = a * (u @ v)
        assert abs(lhs - rhs) < 1e-6, "(a·u)·v = a(u·v): one matmul, "\
            "certainty and polarity ride through retrieval"


# ---------------------------------------------------------------------------
# Token/form lookup unaffected: no NEG-quotient leakage.
# ---------------------------------------------------------------------------

def test_token_lookup_distinguishes_negation():
    rows = unit_rows(seed=6)
    w = rows[2]
    s_self = _wrapped_mse_score(w, w)
    s_neg = _wrapped_mse_score(w, -w)
    assert s_self > s_neg + 0.1, (
        "form lookup: word and -word are DIFFERENT entries; the "
        "quotient must not leak into token codebooks")


def test_torus_snap_unaffected_by_knob():
    cb = Codebook()
    cb.use_dot_product = False               # torus geometry
    cb.monotonic = False
    rows = unit_rows(seed=7)
    torch.manual_seed(8)
    content = torch.rand(2, 3, D) * 2 - 1
    _knob(None)
    off = cb._snap_content(content.clone(), weight=rows, nWhat=D)
    _knob("on")
    try:
        on = cb._snap_content(content.clone(), weight=rows, nWhat=D)
    finally:
        _knob(None)
    assert torch.equal(off, on), "token-tier snap ignores the knob"


# ---------------------------------------------------------------------------
# The knob-gated sphere snap: aligned (pole-quotient) lookup at the
# reference tier; legacy cosine with the knob off.
# ---------------------------------------------------------------------------

def _sphere_codebook():
    cb = Codebook()
    cb.use_dot_product = True                # sphere (reference tier)
    cb.monotonic = False
    return cb


def test_sphere_snap_knob_off_is_legacy_cosine():
    cb = _sphere_codebook()
    rows = unit_rows(seed=9)
    torch.manual_seed(10)
    content = torch.randn(2, 4, D)
    _knob(None)
    snapped = cb._snap_content(content.clone(), weight=rows, nWhat=D)
    flat = content.reshape(-1, D)
    cos = torch.nn.functional.cosine_similarity(
        flat.unsqueeze(1), rows.unsqueeze(0), dim=2)
    expected = rows[cos.argmax(dim=1)].reshape(content.shape)
    assert torch.allclose(snapped, expected), "knob off = signed cosine snap"


def test_sphere_snap_knob_on_is_gauge_invariant():
    cb = _sphere_codebook()
    rows = unit_rows(seed=11)
    flip = torch.ones(V, 1); flip[0] = -1.0; flip[4] = -1.0
    torch.manual_seed(12)
    content = torch.randn(2, 4, D)
    _knob("on")
    try:
        s1 = cb._snap_content(content.clone(), weight=rows, nWhat=D)
        s2 = cb._snap_content(content.clone(), weight=rows * flip, nWhat=D)
    finally:
        _knob(None)
    assert torch.allclose(s1, s2, atol=1e-6), (
        "u ↦ -u relabeling of stored rows is observationally invariant "
        "under the aligned snap")


def test_sphere_snap_knob_on_sign_aligns():
    cb = _sphere_codebook()
    rows = unit_rows(seed=13)
    q = -rows[2] * 0.9                        # anti-aligned query
    content = q.reshape(1, 1, D)
    _knob("on")
    try:
        snapped = cb._snap_content(content.clone(), weight=rows, nWhat=D)
    finally:
        _knob(None)
    assert torch.allclose(snapped.reshape(D), -rows[2], atol=1e-6), (
        "identity by |q·v|, polarity by sign(q·v): the snap returns "
        "the polarity-aligned representative")


def test_sphere_snap_knob_on_monotonic_exempt():
    # Monotonic sphere bases store one-sided content -- the gauge
    # argument does not apply there; the quotient stays off.
    cb = _sphere_codebook()
    cb.monotonic = True
    rows = unit_rows(seed=14).abs()           # nonneg rows
    torch.manual_seed(15)
    content = torch.randn(1, 4, D)
    _knob("on")
    try:
        on = cb._snap_content(content.clone(), weight=rows, nWhat=D)
    finally:
        _knob(None)
    off = cb._snap_content(content.clone(), weight=rows, nWhat=D)
    assert torch.equal(on, off), "monotonic bases keep the signed cosine"
