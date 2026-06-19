"""Stage 3 (corrected, spec rev 2026-06-10b): catuṣkoṭi luminosity over codes.

The TruthLayer stays the simple catuṣkoṭi accumulator (signed
idea-vectors weighted by ±trust). The measure — previously stubbed
(returned 0.0 without a decoder handle) and order-dependent (sequential
cumulative fold) — is completed as a pure function over the stored
codes themselves:

    T_k = max_i relu(+truths[i, k])      (true-pole coverage)
    F_k = max_i relu(-truths[i, k])      (false-pole coverage)
    luminosity = mean_k [(T_k - F_k) - min(T_k, F_k)]   in [-1, 1]

— total area weighted by sign, minus the regions where the sign differs
(contradictory evidence; the catuṣkoṭi B corner). No pullback to the
mereological ground: the §4 weight law puts the parthood geometry on
the codes, so the code-tier measure is the (registration-maintained)
approximation, and ``sym`` is accepted but never used.
"""
import itertools
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import TruthLayer

D = 6


def fresh(max_truths=64):
    return TruthLayer(D, max_truths=max_truths)


def record_quiet(tl, vec, degree, basis=None):
    """record() with the legacy anti-parallel cosine warning silenced
    (a contradiction store trips it by design)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.record(vec, degree, basis=basis)


# ---------------------------------------------------------------------------
# Base cases.
# ---------------------------------------------------------------------------

def test_empty_store_is_dark():
    assert fresh().luminosity() == 0.0


def test_positive_store_measures_positive():
    tl = fresh()
    torch.manual_seed(1)
    v = torch.rand(D) * 0.8 + 0.1
    tl.record(v, degree=1.0)
    lum = tl.luminosity()
    assert abs(lum - v.mean().item()) < 1e-6, (
        "single positive truth: luminosity = mean true-pole coverage")
    assert lum > 0


def test_negative_store_measures_negative():
    tl = fresh()
    torch.manual_seed(2)
    v = torch.rand(D) * 0.8 + 0.1
    tl.record(v, degree=-1.0)
    lum = tl.luminosity()
    assert abs(lum + v.mean().item()) < 1e-6, (
        "single negative truth: luminosity = -mean false-pole coverage")
    assert lum < 0


def test_catuskoti_corners_exact():
    # One dimension per corner: T-only -> +T, F-only -> -F,
    # both -> -min(T, F), neither -> 0.
    tl = TruthLayer(4, max_truths=8)
    t1 = torch.tensor([0.8, 0.0, 0.6, 0.0])   # T on dims 0, 2
    t2 = torch.tensor([0.0, -0.5, -0.9, 0.0])  # F on dims 1, 2
    record_quiet(tl, t1, degree=1.0)
    record_quiet(tl, t2, degree=1.0)
    # dim0: T=0.8           -> +0.8
    # dim1: F=0.5           -> -0.5
    # dim2: T=0.6, F=0.9    -> (0.6-0.9) - 0.6 = -0.9
    # dim3: neither         ->  0.0
    expected = (0.8 - 0.5 - 0.9 + 0.0) / 4
    assert abs(tl.luminosity() - expected) < 1e-6


def test_full_contradiction_measures_negative():
    tl = fresh()
    torch.manual_seed(3)
    v = torch.rand(D) * 0.8 + 0.1
    record_quiet(tl, v, degree=1.0)
    record_quiet(tl, v, degree=-1.0)
    # Every dim: T = F = v_k -> (0) - v_k = -v_k.
    assert abs(tl.luminosity() + v.mean().item()) < 1e-6


# ---------------------------------------------------------------------------
# Coverage semantics and trust weighting.
# ---------------------------------------------------------------------------

def test_duplicate_truths_do_not_inflate_coverage():
    tl = fresh()
    torch.manual_seed(4)
    v = torch.rand(D) * 0.8 + 0.1
    tl.record(v, degree=1.0)
    lum_once = tl.luminosity()
    tl.record(v, degree=1.0)
    tl.record(v, degree=1.0)
    assert abs(tl.luminosity() - lum_once) < 1e-6, (
        "coverage (max), not mass (sum): repeating a truth adds no area")


def test_trust_scales_contribution():
    tl_full = fresh()
    tl_half = fresh()
    torch.manual_seed(5)
    v = torch.rand(D) * 0.8 + 0.1
    tl_full.record(v, degree=1.0)
    tl_half.record(v, degree=0.5)
    assert abs(tl_half.luminosity() - 0.5 * tl_full.luminosity()) < 1e-6


def test_larger_region_dominates_coverage():
    tl = fresh()
    v = torch.full((D,), 0.3)
    w = torch.full((D,), 0.7)
    tl.record(v, degree=1.0)
    tl.record(w, degree=1.0)
    assert abs(tl.luminosity() - 0.7) < 1e-6, (
        "union of nested regions = the larger region")


# ---------------------------------------------------------------------------
# Order independence (the old sequential fold was order-dependent and
# its running value was overwritten by the last pair).
# ---------------------------------------------------------------------------

def test_order_independence():
    torch.manual_seed(6)
    truths = [(torch.rand(D) * 0.8 + 0.1, d) for d in (1.0, -0.7, 0.4)]
    values = []
    for perm in itertools.permutations(range(3)):
        tl = fresh()
        for i in perm:
            v, d = truths[i]
            record_quiet(tl, v, d)
        values.append(tl.luminosity())
    assert max(values) - min(values) < 1e-7, (
        f"luminosity must not depend on record order: {values}")


# ---------------------------------------------------------------------------
# Computed over the codes: no pullback, no decoder dependency.
# ---------------------------------------------------------------------------

class _PoisonedSym:
    """A WholeSpace stand-in whose decoder must never be called."""

    def decode_to_concept(self, row):
        raise AssertionError(
            "luminosity must not pull codes back to ground "
            "(decode_to_concept called)")


def test_no_decode_pullback():
    tl = fresh()
    torch.manual_seed(7)
    record_quiet(tl, torch.rand(D), degree=0.9)
    record_quiet(tl, torch.rand(D), degree=-0.6)
    lum_with_sym = tl.luminosity(sym=_PoisonedSym())   # must not raise
    lum_without = tl.luminosity(sym=None)
    assert lum_with_sym == lum_without, "sym is compatibility-only"


def test_previously_stubbed_path_now_computes():
    # The old implementation returned 0.0 whenever sym was None --
    # effectively a stub in every wiring without a decoder handle.
    tl = fresh()
    tl.record(torch.full((D,), 0.5), degree=1.0)
    assert tl.luminosity(sym=None) != 0.0


def test_mereology_delegator_routes_here():
    from Mereology import Mereology

    class _M(Mereology):
        wholeSpace = None

    tl = fresh()
    tl.record(torch.full((D,), 0.4), degree=1.0)
    m = _M()
    assert abs(m._luminosity_truth_fold(tl) - 0.4) < 1e-6
    assert m._luminosity_truth_fold(None) == 0.0


# ---------------------------------------------------------------------------
# Range and degenerate inputs.
# ---------------------------------------------------------------------------

def test_range_clamped():
    tl = fresh()
    torch.manual_seed(8)
    # Unnormalized rows (|v| > 1) must not escape [-1, 1].
    tl.record(torch.full((D,), 3.0), degree=1.0)
    assert tl.luminosity() == 1.0
    tl2 = fresh()
    record_quiet(tl2, torch.full((D,), 3.0), degree=1.0)
    record_quiet(tl2, torch.full((D,), 3.0), degree=-1.0)
    assert tl2.luminosity() == -1.0


def test_zero_degree_truth_is_inert():
    tl = fresh()
    tl.record(torch.rand(D), degree=0.0)
    assert tl.luminosity() == 0.0


def test_paired_bivector_store_reads_as_positive_area():
    # Documented domain boundary: under paired-index bivector storage,
    # A and not(A) land on orthogonal poles -- no sign opposition, so
    # the measure sees pure positive area (B-corner policy for that
    # layout lives in tetralemma_balance_penalty).
    class _MonotonicBasis:
        monotonic = True

    tl = fresh()
    torch.manual_seed(9)
    act = torch.rand(D) * 0.8          # non-negative, D even
    tl.record(act, degree=1.0, basis=_MonotonicBasis())
    tl.record(act, degree=-1.0, basis=_MonotonicBasis())
    assert tl.luminosity() > 0
