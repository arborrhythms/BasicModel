"""``<initScale>`` (fidelity leg, 2026-07-06): the per-row seed magnitude for
a WholeSpace's codebooks. Small values keep the sigma/pi folds in their linear
regime. These tests lock the knob mechanism (both Codebook prefill branches),
its selection-invariance (a uniform scale must not change dot-product argmax),
and the fold-linearity bar (small init -> additive folds; unit init saturates).
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
from Spaces import Codebook  # noqa: E402

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_PROJECT, "data")


def _row_norms(cb):
    W = cb.getW().detach()
    return W.norm(dim=-1)


def test_init_scale_shrinks_rows_dot_product():
    # Dot-product (cosine) codebook: default prefill is unit-L2; initScale
    # rescales every row to that magnitude.
    cb0 = Codebook()
    cb0.use_dot_product = True
    cb0.create(8, 6, 8, customVQ=True, monotonic=False)
    assert torch.allclose(_row_norms(cb0), torch.ones(6), atol=1e-4), \
        "default dot-product rows must be unit-norm"

    cb = Codebook()
    cb.use_dot_product = True
    cb.create(8, 6, 8, customVQ=True, monotonic=False, init_scale=0.02)
    assert torch.allclose(_row_norms(cb), torch.full((6,), 0.02), atol=1e-4), \
        "initScale must rescale dot-product rows to that magnitude"


def test_init_scale_shrinks_rows_generic_branch():
    # Non-customVQ (generic) branch: per-row normalize then scale. Seed both
    # identically (addVectors draws from the global RNG) so the ONLY difference
    # is the 0.02 rescale.
    torch.manual_seed(3)
    cb0 = Codebook()
    cb0.create(8, 5, 8, customVQ=False, monotonic=False)
    n0 = _row_norms(cb0)

    torch.manual_seed(3)
    cb = Codebook()
    cb.create(8, 5, 8, customVQ=False, monotonic=False, init_scale=0.02)
    n = _row_norms(cb)
    # Every row is exactly the default row scaled by 0.02 (same directions).
    assert torch.all(n < n0)
    assert torch.allclose(n, n0 * 0.02, atol=1e-5)
    assert torch.allclose(cb.getW(), cb0.getW() * 0.02, atol=1e-6)


def test_init_scale_zero_is_none_byte_identical():
    # initScale 0 (or None) must reproduce the default prefill EXACTLY.
    torch.manual_seed(0)
    cb_none = Codebook()
    cb_none.use_dot_product = True
    cb_none.create(8, 6, 8, customVQ=True, monotonic=False, init_scale=None)
    torch.manual_seed(0)
    cb_zero = Codebook()
    cb_zero.use_dot_product = True
    cb_zero.create(8, 6, 8, customVQ=True, monotonic=False, init_scale=0)
    assert torch.equal(cb_none.getW(), cb_zero.getW())
    assert cb_none.init_scale is None and cb_zero.init_scale is None


def test_init_scale_selection_invariant():
    # A uniform per-row scale must leave dot-product argmax selection
    # unchanged: argmax_i (x . s*c_i) == argmax_i (x . c_i).
    torch.manual_seed(1)
    cb0 = Codebook()
    cb0.use_dot_product = True
    cb0.create(8, 12, 8, customVQ=True, monotonic=False)
    torch.manual_seed(1)
    cb = Codebook()
    cb.use_dot_product = True
    cb.create(8, 12, 8, customVQ=True, monotonic=False, init_scale=0.02)
    W0, W = cb0.getW().detach(), cb.getW().detach()
    q = torch.randn(20, 8)
    sel0 = (q @ W0.t()).argmax(dim=-1)
    sel = (q @ W.t()).argmax(dim=-1)
    assert torch.equal(sel0, sel), "scale changed nearest-row selection"


def _fold_saturation(scale, depth=8, dim=8):
    """Relative squash error of a depth-``depth`` tanh fold of unit-direction
    rows at magnitude ``scale`` -- the sigma/pi fold's linear-regime proxy
    (tanh(sum atanh x) ~ sum x for small x; here the additive tanh chain)."""
    torch.manual_seed(2)
    rows = torch.nn.functional.normalize(torch.randn(depth, dim), dim=-1) * scale
    additive = rows.sum(dim=0)
    y = torch.zeros(dim)
    for k in range(depth):
        y = torch.tanh(y + rows[k])
    return float((y - additive).norm() / additive.norm().clamp(min=1e-8))


def test_init_scale_keeps_folds_linear():
    # The fidelity bar (design sec 3.2): at a small operating point the fold
    # is effectively additive; at unit scale it saturates.
    small = _fold_saturation(0.02)
    big = _fold_saturation(0.5)
    assert small < 0.02, f"small-init fold should stay near-linear, got {small}"
    assert big > 0.2, f"unit-scale fold should saturate, got {big}"
    assert big > 10 * small, "small init must reduce fold saturation sharply"


def test_whole_space_init_scale_wiring():
    # End-to-end: <WholeSpace><initScale> must reach BOTH WS codebooks so
    # their prefill rows land at the small operating point.
    import Models
    import Language
    from util import init_config

    cfg = os.path.join(_DATA, "MM_init_scale.xml")
    init_config(path=cfg, defaults_path=os.path.join(_DATA, "model.xml"))
    Language.TheGrammar._configured = False
    m, _ = Models.BasicModel.from_config(cfg)
    ws = m.wholeSpace

    what = ws.subspace.what
    assert isinstance(what, Codebook), "fixture must give WS a Codebook .what"
    assert what.init_scale == pytest.approx(0.02)
    assert ws.analysis_store.init_scale == pytest.approx(0.02)
    # The freshly-built .what prototypes sit at the small scale (mean row norm
    # well under the unit / hypercube default).
    mean_norm = float(_row_norms(what).mean())
    assert mean_norm < 0.1, f"WS .what rows not small-init: mean |row|={mean_norm}"
