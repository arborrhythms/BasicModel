"""Membership-chart butterfly on the meronymic slots (author direction
2026-06-11; the Stage-9 cutover correction).

The order-preservation law of the meronymic slots constrains the KERNEL
CLASS (contractive, non-negative log-mass weights), not the fold's
TOPOLOGY: order-preserving nodes compose, so the FFT-style cascade with
the contractive law per 2×2 node preserves the partial order end-to-end
while keeping the cross-position reach the per-slot fold lacked (the
XOR_exact regression: distinct word codes sit INCOMPARABLE in the
partial order, so monotonicity never excluded XOR — only the lost reach
did). Pinned here:

  * near-identity at init (the cutover's benign-start contract);
  * exact invertibility (reverse divides by d >= 1 — contractive);
  * order preservation through the full cascade, both kinds;
  * CROSS-SLOT REACH: a perturbation in one slot moves other slots;
  * width-agnosticism over the flattened slab (the legacy cascade's
    call contract);
  * the Spaces cutover sites keep butterfly_enabled=True for
    butterfly-built slots (XOR_exact under <meronomy>on).
"""

import os
import sys
import warnings

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_N, _D = 8, 14            # XOR_exact's slab geometry (flat total 112)


def _adapter(kind, randomize=False, seed=3):
    from Layers import MeronymicFoldAdapter
    ad = MeronymicFoldAdapter(kind, _D, _D, legacy_N=_N * _D,
                              butterfly=True)
    if randomize:
        # A non-trivial but numerically tame law (square reparam):
        # raws in [0.05, 0.6] give taps/diagonal-excesses in
        # ~[0.0025, 0.36] per node (a 7-level cascade of d ~ 2 would
        # underflow float32 log-mass; that regime is reachable only by
        # training pressure, not by a unit test's random draw).
        torch.manual_seed(seed)
        with torch.no_grad():
            for raw in (ad.raw_bfly_L, ad.raw_bfly_d, ad.raw_bfly_U):
                raw.copy_(torch.rand_like(raw) * 0.55 + 0.05)
    return ad


def _slab(seed=11):
    torch.manual_seed(seed)
    return torch.rand(2, _N, _D) * 1.6 - 0.8


@pytest.mark.parametrize("kind", ["sigma", "pi"])
def test_near_identity_at_init(kind):
    ad = _adapter(kind)
    x = _slab()
    with torch.no_grad():
        y = ad.forward(x)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=0.08), (
        f"{kind}: init must stay near the benign pass-through "
        f"(max drift {(y - x).abs().max():.4f})")


@pytest.mark.parametrize("kind", ["sigma", "pi"])
def test_exact_roundtrip_under_arbitrary_law(kind):
    ad = _adapter(kind, randomize=True)
    x = _slab(seed=13) * 0.9
    with torch.no_grad():
        y = ad.forward(x)
        back = ad.reverse(y)
    assert torch.allclose(back, x, atol=5e-4), (
        f"{kind}: cascade round-trip broke "
        f"(max err {(back - x).abs().max():.2e})")


@pytest.mark.parametrize("kind", ["sigma", "pi"])
def test_order_preserved_through_cascade(kind):
    """m <= m' elementwise on the K3 wire => fold(m) <= fold(m'):
    every node carries the contractive non-negative law, and
    order-preserving maps compose."""
    ad = _adapter(kind, randomize=True, seed=5)
    torch.manual_seed(17)
    x_lo = torch.rand(2, _N, _D) * 1.2 - 0.8
    x_hi = (x_lo + torch.rand(2, _N, _D) * 0.4).clamp(max=0.95)
    with torch.no_grad():
        y_lo = ad.forward(x_lo)
        y_hi = ad.forward(x_hi)
    assert torch.all(y_hi >= y_lo - 1e-5), (
        f"{kind}: partial order violated "
        f"(worst {(y_lo - y_hi).max():.2e})")


@pytest.mark.parametrize("kind", ["sigma", "pi"])
def test_per_vector_isolation(kind):
    """RE-PINNED (per-vector order-raise, Alec 2026-07-07): mereological
    order-raising raises EACH vector independently -- it must NOT mix
    across word slots (the old cross_slot_reach contract inverted; the
    cross-word feature leak degraded the per-word round-trip and made the
    serial loop O(N^2)). Perturbing slot 0 moves slot 0 ONLY."""
    ad = _adapter(kind, randomize=True, seed=7)
    x = _slab(seed=19)
    x2 = x.clone()
    x2[:, 0, :] = (x2[:, 0, :] + 0.5).clamp(-0.95, 0.95)
    with torch.no_grad():
        y = ad.forward(x)
        y2 = ad.forward(x2)
    moved_elsewhere = (y2[:, 1:, :] - y[:, 1:, :]).abs().max()
    moved_slot0 = (y2[:, 0, :] - y[:, 0, :]).abs().max()
    assert moved_slot0 > 1e-4, (
        f"{kind}: perturbed slot 0 did not move ({moved_slot0:.2e})")
    assert moved_elsewhere == 0.0, (
        f"{kind}: per-vector fold leaked across slots "
        f"(max move {moved_elsewhere:.2e})")


def test_per_vector_slab_equals_per_row_fold():
    """RE-PINNED (per-vector order-raise): the slab fold equals folding
    each slot independently -- leading dims are pure batch. (Replaces the
    old width-agnostic flat==slab contract: a ``[B, N*D]`` input is now
    ONE N*D-wide vector, not a slab view, and exceeds the per-vector
    cascade width by design.)"""
    ad = _adapter('pi', randomize=True, seed=9)
    x = _slab(seed=23)
    with torch.no_grad():
        y_slab = ad.forward(x)
        y_rows = torch.stack(
            [ad.forward(x[:, i:i + 1, :]).squeeze(1) for i in range(_N)],
            dim=1)
    assert torch.allclose(y_slab, y_rows, atol=1e-6)
    with pytest.raises(RuntimeError):
        ad.forward(x.reshape(2, _N * _D))   # one over-wide vector: fail loud


def test_unary_mode_unchanged():
    """butterfly=False keeps the per-slot fold and its width-mismatch
    identity guard byte-for-byte (non-butterfly configs)."""
    from Layers import MeronymicFoldAdapter
    ad = MeronymicFoldAdapter('pi', _D, _D, legacy_N=_N * _D)
    assert ad.butterfly is False and ad.fold is not None
    x = _slab(seed=29)
    with torch.no_grad():
        y = ad.forward(x)                       # last dim == nInput
        ident = ad.forward(x.reshape(2, -1))    # width mismatch
    assert y.shape == x.shape
    assert torch.equal(ident, x.reshape(2, -1))


# -- Discontiguous specifications + the saturation boundary ---------------
# (author concern, 2026-06-11: the membership folds may saturate to the
# unusable 0000/1111 corners or exclude solutions; contingency = a
# part-whole TREE on the codes. These tests certify what the geometric
# form CAN do — scattered specifications under repeated application —
# and characterize where it degrades.)

def _pair_index(ad, level, lane_a, lane_b):
    """The cascade pair index at ``level`` joining the two lanes
    (computed from the live perms — no hardcoded stride math)."""
    perm = ad.butterfly_perms[level]
    for p in range(perm.numel() // 2):
        pair = {int(perm[2 * p]), int(perm[2 * p + 1])}
        if pair == {lane_a, lane_b}:
            return p
    raise AssertionError(
        f"lanes {lane_a},{lane_b} not paired at level {level}")


def test_repeated_application_preserves_discontiguous_spec():
    """A scattered specification (high lanes in non-adjacent slots)
    survives repeated application at the benign law: ranking intact,
    still discontiguous, nowhere near the 0000/1111 corners."""
    from Layers import MeronymicFoldAdapter
    for kind in ("sigma", "pi"):
        ad = MeronymicFoldAdapter(kind, _D, _D, legacy_N=_N * _D,
                                  butterfly=True)
        x = torch.full((1, _N, _D), -0.8)
        x[:, 1, :] = 0.8                      # slot 1 high
        x[:, 6, :] = 0.8                      # slot 6 high — scattered
        y = x
        with torch.no_grad():
            for _ in range(3):
                y = ad.forward(y)
        hi = y[:, (1, 6), :]
        lo = y[:, (0, 2, 3, 4, 5, 7), :]
        assert float(hi.min()) > float(lo.max()) + 0.5, (
            f"{kind}: scattered selection lost under repetition")
        assert float(y.abs().max()) < 0.999, (
            f"{kind}: corner saturation at the benign law")


def test_repeated_application_produces_discontiguous_spec():
    """Multiple applications PRODUCE a discontiguous specification
    from a contiguous one — RE-PINNED WITHIN one vector (per-vector
    order-raise, Alec 2026-07-07: the cascade never crosses word
    slots; accretion happens along one vector's own feature lanes).
    On the 16-lane cascade (D=14 padded to 16): lanes 0..1 raise
    lanes 8..9 (app 1), which raise lanes 12..13 (app 2). The
    within-application level order (low strides first) makes the
    second hop wait for the second application — the growth is real
    repetition, not one big matrix."""
    from Layers import MeronymicFoldAdapter
    ad = MeronymicFoldAdapter('sigma', _D, _D, legacy_N=_N * _D,
                              butterfly=True)
    with torch.no_grad():
        for raw in (ad.raw_bfly_L, ad.raw_bfly_d, ad.raw_bfly_U):
            raw.zero_()                      # exact identity baseline
        lvl_hi = ad.n_levels - 1             # stride 8: lanes i <-> i+8
        lvl_mid = ad.n_levels - 2            # stride 4: lanes i <-> i+4
        for i in range(2):
            # tap: lane i feeds lane 8+i; raw²=1.0.
            ad.raw_bfly_L[lvl_hi, _pair_index(ad, lvl_hi, i, 8 + i)] = 1.0
            # tap: lane 8+i feeds lane 12+i.
            ad.raw_bfly_L[lvl_mid,
                          _pair_index(ad, lvl_mid, 8 + i, 12 + i)] = 1.0

    vec = torch.full((1, 1, _D), -0.8)
    vec[0, 0, 0:2] = 0.8                     # contiguous spec: lanes 0..1
    with torch.no_grad():
        app1 = ad.forward(vec)
        app2 = ad.forward(app1)

    def hi(t, sl):
        return float(t[0, 0, sl].min())

    def lo(t, sl):
        return float(t[0, 0, sl].max())

    # App 1: the whole accretes lanes 8..9 — already discontiguous
    # (a gap of low lanes 2..7 between its parts).
    assert hi(app1, slice(8, 10)) > 0.5, "first hop did not accrete"
    assert lo(app1, slice(12, 14)) < 0.0, "second hop fired too early"
    assert lo(app1, slice(2, 8)) < 0.0
    # App 2: a further application extends the scattered whole to
    # lanes 12..13 while keeping the earlier parts — three separated
    # bands, produced by repetition, nothing saturated.
    assert hi(app2, slice(0, 2)) > 0.5
    assert hi(app2, slice(8, 10)) > 0.5
    assert hi(app2, slice(12, 14)) > 0.5, "repetition did not extend"
    assert lo(app2, slice(2, 8)) < 0.0
    assert lo(app2, slice(10, 12)) < 0.0
    assert float(app2.abs().max()) < 0.999, "corner saturation"


def test_iterated_aggressive_law_saturates_to_corner():
    """The documented failure mode, pinned as a characterization test
    (the author's recorded concern; docstring 'Recorded risk'): under
    an aggressive law, iterated application drives EVERY lane to the
    same corner — σ to 𝟙 (the 1111 code) — erasing the
    specification. det(LDU) >= 1 per node makes this an eigenvalue
    floor of the law itself, not a trainable defect; if it binds in
    practice, the contingency is a part-whole tree on the codes."""
    from Layers import MeronymicFoldAdapter
    ad = MeronymicFoldAdapter('sigma', _D, _D, legacy_N=_N * _D,
                              butterfly=True)
    with torch.no_grad():
        for raw in (ad.raw_bfly_L, ad.raw_bfly_d, ad.raw_bfly_U):
            raw.fill_(0.9)                   # taps 0.81, d = 1.81
    torch.manual_seed(37)
    x = torch.rand(1, _N, _D) * 1.6 - 0.8    # an informative spec
    y = x
    with torch.no_grad():
        for _ in range(6):
            y = ad.forward(y)
    spread = float(y.max() - y.min())
    assert float(y.mean()) > 0.95 and spread < 0.1, (
        f"expected corner collapse (mean {float(y.mean()):.3f}, "
        f"spread {spread:.3f}) — if this stops saturating, the "
        f"documented risk envelope has changed; update the docstring")


def test_xor_exact_slots_keep_butterfly_under_meronomy():
    """The cutover sites build the cascade adapter and keep
    butterfly_enabled=True for butterfly-built slots."""
    import Models
    import Language
    from Layers import MeronymicFoldAdapter, meronomy_enabled
    from util import init_config
    init_config(path=os.path.join(_PROJECT, "data", "XOR_exact.xml"),
                defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Models.TheData.load("xor")
        m, _ = Models.BaseModel.from_config(
            os.path.join(_PROJECT, "data", "XOR_exact.xml"),
            data=Models.TheData)
    assert meronomy_enabled()
    ps, ws = m.perceptualSpace, m.wholeSpace
    assert isinstance(ps.sigma, MeronymicFoldAdapter)
    assert ps.sigma.butterfly is True and ps.butterfly_enabled is True
    assert isinstance(ws.pi, MeronymicFoldAdapter)
    assert ws.pi.butterfly is True and ws.butterfly_enabled is True
    # RE-PINNED (per-vector order-raise, Alec 2026-07-07): the live cascade
    # raises each slot's vector independently -- perturbing slot 0 moves
    # slot 0 and leaves every other slot byte-identical (the old cross-slot
    # reach contract inverted). Probe law mild (raw 0.28) as before.
    with torch.no_grad():
        for raw in (ps.sigma.raw_bfly_L, ps.sigma.raw_bfly_U):
            raw.fill_(0.28)
        torch.manual_seed(31)
        x = torch.rand(2, _N, _D) * 1.6 - 0.8
        x2 = x.clone()
        x2[:, 0, :] = (x2[:, 0, :] + 0.5).clamp(-0.95, 0.95)
        y, y2 = ps.sigma.forward(x), ps.sigma.forward(x2)
        moved_self = (y2[:, 0, :] - y[:, 0, :]).abs().max()
        moved_other = (y2[:, 1:, :] - y[:, 1:, :]).abs().max()
    assert moved_self > 1e-4, f"perturbed slot did not move ({moved_self:.2e})"
    assert moved_other == 0.0, (
        f"per-vector fold leaked across live slots ({moved_other:.2e})")
