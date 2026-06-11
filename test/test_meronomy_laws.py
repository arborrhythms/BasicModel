"""Stage 0 of doc/plans/MeronomyPlan.md: laws and constants.

Pins the evaluation chart (MeronomySpec §3) and the meronomy constants
before any wiring exists:

  * ``Ops.eval_chart(a) = (1 + a) / 2`` -- belief → assumed membership --
    with EXACT corners: ``+1 ↦ 1``, ``0 ↦ ½``, ``−1 ↦ 0``.
  * ``Ops.eval_chart_inv(m) = 2m − 1`` -- the exact inverse (bijection).
  * ``EPS_LOG = 1e-6`` (log-floor near m = 0) and ``D_MAX_STABLE = 4.0``
    (stable-clamp upper bound, config-overridable).
  * GUARD: no ReLU-injection law exists anywhere in bin/ -- the old
    ``max(0, 2m−1)`` helper must NOT exist; the Stage-4 factored
    interface (content selects the row, evidence sets the magnitude)
    replaces it. χ is evaluation, not injection.
"""
import os
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Layers
from Layers import Ops, EPS_LOG, D_MAX_STABLE, meronomy_d_max_stable


# ---------------------------------------------------------------------------
# Chart corners: exact, not approximate (spec §10.6 "evaluation chart
# exact"). (1 + a)/2 at a in {+1, 0, -1} is exact dyadic arithmetic.
# ---------------------------------------------------------------------------

def test_eval_chart_corners_exact():
    a = torch.tensor([1.0, 0.0, -1.0])
    m = Ops.eval_chart(a)
    assert m[0].item() == 1.0    # +1 ↦ 1   (certain-true)
    assert m[1].item() == 0.5    #  0 ↦ ½   (maximal vagueness)
    assert m[2].item() == 0.0    # −1 ↦ 0   (certain-false)


def test_eval_chart_inv_corners_exact():
    m = torch.tensor([1.0, 0.5, 0.0])
    a = Ops.eval_chart_inv(m)
    assert a[0].item() == 1.0
    assert a[1].item() == 0.0
    assert a[2].item() == -1.0


def test_eval_chart_scalar_inputs():
    # Ops convention: non-tensor inputs are coerced via torch.as_tensor.
    assert Ops.eval_chart(1.0).item() == 1.0
    assert Ops.eval_chart(-1.0).item() == 0.0
    assert Ops.eval_chart_inv(0.5).item() == 0.0


# ---------------------------------------------------------------------------
# Bijection: chart and inverse round-trip both ways across the domain.
# ---------------------------------------------------------------------------

def test_eval_chart_bijection():
    torch.manual_seed(1)
    a = torch.rand(64, 7) * 2 - 1            # a in [-1, 1]
    m = torch.rand(64, 7)                    # m in [0, 1]
    assert torch.allclose(Ops.eval_chart_inv(Ops.eval_chart(a)), a,
                          atol=1e-6, rtol=0)
    assert torch.allclose(Ops.eval_chart(Ops.eval_chart_inv(m)), m,
                          atol=1e-6, rtol=0)
    # Corners survive the round-trip exactly.
    corners = torch.tensor([1.0, 0.0, -1.0])
    assert torch.equal(Ops.eval_chart_inv(Ops.eval_chart(corners)), corners)


def test_eval_chart_monotone_and_in_range():
    a = torch.linspace(-1, 1, 101)
    m = Ops.eval_chart(a)
    assert (m >= 0).all() and (m <= 1).all()
    assert (m[1:] > m[:-1]).all()            # strictly increasing


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

def test_meronomy_constants():
    assert EPS_LOG == 1e-6
    assert D_MAX_STABLE == 4.0
    # With no config override loaded, the resolver returns the default.
    assert meronomy_d_max_stable() == D_MAX_STABLE


def test_d_max_stable_config_override():
    from util import TheXMLConfig
    # set() creates intermediate dicts; the autouse singleton-reset
    # fixture clears requirements but not _data, so restore explicitly.
    had_arch = "architecture" in TheXMLConfig._data
    arch = TheXMLConfig._data.get("architecture", {})
    had_mero = had_arch and "meronomy" in arch
    prev = arch.get("meronomy") if had_mero else None
    try:
        TheXMLConfig.set("architecture.meronomy.dMaxStable", "6.5")
        assert meronomy_d_max_stable() == 6.5
        TheXMLConfig.set("architecture.meronomy.dMaxStable", "bogus")
        assert meronomy_d_max_stable() == D_MAX_STABLE
    finally:
        if had_mero:
            TheXMLConfig._data["architecture"]["meronomy"] = prev
        elif had_arch:
            TheXMLConfig._data["architecture"].pop("meronomy", None)
        else:
            TheXMLConfig._data.pop("architecture", None)


# ---------------------------------------------------------------------------
# Guard: the ReLU-injection law must not exist anywhere in bin/.
#
# The retired law mapped beliefs into memberships with rectification --
# max(0, 2m−1) / relu(2m−1) / (2m−1).clamp(min=0) -- which silently
# conflated "false" with "unknown". MeronomySpec §3 replaces it with the
# factored interface (Stage 4); until then the only belief↔membership
# map is the affine chart pair above. This scan keeps any spelling of
# the rectified chart from (re)appearing in production code.
# ---------------------------------------------------------------------------

_RELU_INJECTION_PATTERNS = [
    # relu(2*x - 1) / relu(2x - 1)
    re.compile(r"relu\s*\(\s*2(?:\.0)?\s*\*?\s*[A-Za-z_]\w*\s*-\s*1"),
    # max(0, 2*x - 1) -- python max or torch.max/maximum
    re.compile(r"max(?:imum)?\s*\(\s*0(?:\.0)?\s*,\s*2(?:\.0)?\s*\*?\s*"
               r"[A-Za-z_]\w*\s*-\s*1"),
    # (2*x - 1).clamp(min=0) / .clamp(0, ...)
    re.compile(r"2(?:\.0)?\s*\*?\s*[A-Za-z_]\w*\s*-\s*1(?:\.0)?\s*\)\s*"
               r"\.\s*clamp\s*\(\s*(?:min\s*=\s*)?0"),
]


def test_no_relu_injection_law_in_bin():
    bad = []
    for fname in sorted(os.listdir(_BIN)):
        if not fname.endswith(".py"):
            continue  # skip subdirs (bin/etc is archived) and non-source
        path = os.path.join(_BIN, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            src = f.read()
        for pat in _RELU_INJECTION_PATTERNS:
            for match in pat.finditer(src):
                line = src.count("\n", 0, match.start()) + 1
                bad.append(f"{fname}:{line}: {match.group(0)!r}")
    assert not bad, (
        "ReLU-injection law found (retired by MeronomySpec §3; use "
        "Ops.eval_chart / the factored interface instead):\n  "
        + "\n  ".join(bad))


def test_no_legacy_chart_helper_names():
    # The old helper must not exist under any of its historical names,
    # and the chart pair must be the only belief↔membership map on Ops.
    for name in ("relu_chart", "relu_inject", "membership_inject",
                 "inject_membership", "to_membership_relu"):
        assert not hasattr(Ops, name), f"retired helper Ops.{name} exists"
        assert not hasattr(Layers, name), f"retired helper Layers.{name} exists"
    assert hasattr(Ops, "eval_chart") and hasattr(Ops, "eval_chart_inv")
