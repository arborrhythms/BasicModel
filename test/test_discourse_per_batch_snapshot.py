"""B>=2 per-row isolation for InterSentenceLayer (DiscourseSpace).

Task 3 of the microbatch AR refactor (see
basicmodel/doc/specs/2026-04-22-microbatch-ar-refactor-design.md).

Verifies the leading-B dim added to ``_recent`` / ``_recent_count`` /
``_prev_centroids`` / ``_prev_count``: per-row pushes don't bleed
across rows, ``ensure_batch`` resizes correctly, and the contrastive
/ predictive losses operate per-row when given 3D inputs.
"""
import os
import sys

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Layers


N_SYMBOLS = 4
MAX_DEPTH = 6
N_DIM = 8
CTX = 4
CENTROID_HIST = 2
LAM = 1.01
CONCEPT_DIM = 12


def _make(batch=1, with_predictor=False):
    return Layers.InterSentenceLayer(
        n_symbols=N_SYMBOLS,
        max_depth=MAX_DEPTH,
        n_dim=N_DIM,
        context_window=CTX,
        centroid_history=CENTROID_HIST,
        lam=LAM,
        concept_dim=(CONCEPT_DIM if with_predictor else None),
        batch=batch,
    )


# -- shape / ensure_batch --------------------------------------------------

def test_default_batch_is_one():
    d = _make()
    assert d._recent.shape == (1, CTX, N_SYMBOLS + MAX_DEPTH, N_DIM)
    assert d._recent_count.shape == (1,)
    assert d._prev_centroids.shape == (
        1, CENTROID_HIST, N_SYMBOLS + MAX_DEPTH, N_DIM)
    assert d._prev_count.shape == (1,)


def test_construct_with_batch_three():
    d = _make(batch=3)
    assert d._recent.shape == (3, CTX, N_SYMBOLS + MAX_DEPTH, N_DIM)
    assert d._recent_count.shape == (3,)


def test_ensure_batch_grows_and_zeros():
    d = _make(batch=1)
    s = torch.randn(N_SYMBOLS, N_DIM)
    w = torch.randn(MAX_DEPTH, N_DIM)
    d.snapshot(s, w)
    assert int(d._recent_count[0].item()) == 1

    d.ensure_batch(4)
    assert d._recent.shape[0] == 4
    assert d._recent_count.shape == (4,)
    # Resize wipes prior state on every row.
    assert torch.equal(d._recent_count,
                       torch.zeros(4, dtype=torch.long))


def test_ensure_batch_noop_when_same():
    d = _make(batch=2)
    s = torch.randn(2, N_SYMBOLS, N_DIM)
    w = torch.randn(2, MAX_DEPTH, N_DIM)
    d.snapshot(s, w)
    counts_before = d._recent_count.clone()
    d.ensure_batch(2)
    assert torch.equal(d._recent_count, counts_before)


# -- per-row snapshot isolation -------------------------------------------

def test_b3_snapshot_writes_distinct_rows():
    d = _make(batch=3)
    # Tag each row with a distinct scalar so we can verify storage.
    s = torch.stack([
        torch.full((N_SYMBOLS, N_DIM), 1.0),
        torch.full((N_SYMBOLS, N_DIM), 2.0),
        torch.full((N_SYMBOLS, N_DIM), 3.0),
    ])
    w = torch.stack([
        torch.full((MAX_DEPTH, N_DIM), 1.0),
        torch.full((MAX_DEPTH, N_DIM), 2.0),
        torch.full((MAX_DEPTH, N_DIM), 3.0),
    ])
    d.snapshot(s, w)
    assert torch.equal(d._recent_count,
                       torch.tensor([1, 1, 1], dtype=torch.long))
    # Row 0's stored snapshot should be all 1.0s, etc.
    for b, tag in enumerate([1.0, 2.0, 3.0]):
        row = d._recent[b, 0]
        assert torch.allclose(row, torch.full_like(row, tag))


def test_snapshot_mask_skips_inactive_rows():
    d = _make(batch=3)
    s = torch.randn(3, N_SYMBOLS, N_DIM)
    w = torch.randn(3, MAX_DEPTH, N_DIM)
    mask = torch.tensor([True, False, True])
    d.snapshot(s, w, mask=mask)
    # Rows 0 and 2 advanced; row 1 stayed at 0.
    assert int(d._recent_count[0].item()) == 1
    assert int(d._recent_count[1].item()) == 0
    assert int(d._recent_count[2].item()) == 1


def test_b2_ring_eviction_per_row_independent():
    d = _make(batch=2)
    # Row 0: push CTX+1 snapshots so its centroid folds to prev.
    # Row 1: push 0 snapshots (use mask).
    for i in range(CTX + 1):
        s = torch.stack([
            torch.full((N_SYMBOLS, N_DIM), float(i + 1)),
            torch.zeros(N_SYMBOLS, N_DIM),
        ])
        w = torch.stack([
            torch.full((MAX_DEPTH, N_DIM), float(i + 1)),
            torch.zeros(MAX_DEPTH, N_DIM),
        ])
        d.snapshot(s, w, mask=torch.tensor([True, False]))
    # Row 0 saturated at CTX recent + 1 evicted-centroid.
    assert int(d._recent_count[0].item()) == CTX
    assert int(d._prev_count[0].item()) == 1
    # Row 1 untouched.
    assert int(d._recent_count[1].item()) == 0
    assert int(d._prev_count[1].item()) == 0


# -- per-row predict ------------------------------------------------------

def test_predict_per_row_with_b():
    d = _make(batch=2, with_predictor=True)
    # Seed only row 0.
    s = torch.stack([
        torch.randn(N_SYMBOLS, N_DIM),
        torch.zeros(N_SYMBOLS, N_DIM),
    ])
    w = torch.stack([
        torch.randn(MAX_DEPTH, N_DIM),
        torch.zeros(MAX_DEPTH, N_DIM),
    ])
    d.snapshot(s, w, mask=torch.tensor([True, False]))

    # Row 0 has context.
    p0, c0 = d.predict(0)
    assert p0 is not None
    assert tuple(p0.shape) == (d.s_dim,)
    assert c0 is not None
    # Row 1 is empty.
    p1, c1 = d.predict(1)
    assert p1 is None
    assert c1 is None


def test_predict_default_returns_batched():
    d = _make(batch=2, with_predictor=True)
    s = torch.randn(2, N_SYMBOLS, N_DIM)
    w = torch.randn(2, MAX_DEPTH, N_DIM)
    d.snapshot(s, w)
    pred, conf = d.predict()
    # B>1 layer returns stacked per-row tensors.
    assert pred is not None
    assert tuple(pred.shape) == (2, d.s_dim)
    assert tuple(conf.shape) == (2,)


def test_predict_b1_layer_keeps_legacy_shape():
    d = _make(batch=1, with_predictor=True)
    s = torch.randn(N_SYMBOLS, N_DIM)
    w = torch.randn(MAX_DEPTH, N_DIM)
    d.snapshot(s, w)
    pred, conf = d.predict()
    # B=1 layer preserves legacy shape: 1D pred + 0D scalar conf.
    assert pred is not None
    assert tuple(pred.shape) == (d.s_dim,)
    assert conf.ndim == 0


# -- per-row contrastive loss --------------------------------------------

def test_contrastive_loss_3d_batched():
    d = _make(batch=2)
    # Seed history on every row.
    s_hist = torch.randn(2, N_SYMBOLS, N_DIM)
    w_hist = torch.randn(2, MAX_DEPTH, N_DIM)
    d.snapshot(s_hist, w_hist)

    s_cur = torch.randn(2, N_SYMBOLS, N_DIM)
    w_cur = torch.randn(2, MAX_DEPTH, N_DIM)
    loss = d.contrastive_loss(s_cur, w_cur)
    assert loss is not None
    assert loss.ndim == 0


def test_contrastive_loss_skips_empty_rows():
    d = _make(batch=2)
    # Only row 0 has history.
    s_hist = torch.stack([
        torch.randn(N_SYMBOLS, N_DIM),
        torch.zeros(N_SYMBOLS, N_DIM),
    ])
    w_hist = torch.stack([
        torch.randn(MAX_DEPTH, N_DIM),
        torch.zeros(MAX_DEPTH, N_DIM),
    ])
    d.snapshot(s_hist, w_hist, mask=torch.tensor([True, False]))

    s_cur = torch.randn(2, N_SYMBOLS, N_DIM)
    w_cur = torch.randn(2, MAX_DEPTH, N_DIM)
    loss = d.contrastive_loss(s_cur, w_cur)
    assert loss is not None
    # Manual: only row 0 contributes; loss should match the single-row
    # contrastive loss on row 0's input.
    s_only0 = s_cur[0]
    w_only0 = w_cur[0]
    expected = d._contrastive_one_row(0, d._assemble(s_only0, w_only0),
                                      d._recent_centroid(0))
    assert torch.allclose(loss, expected)


def test_contrastive_loss_none_when_all_rows_empty():
    d = _make(batch=2)
    s_cur = torch.randn(2, N_SYMBOLS, N_DIM)
    w_cur = torch.randn(2, MAX_DEPTH, N_DIM)
    assert d.contrastive_loss(s_cur, w_cur) is None


# -- per-row predictive loss ----------------------------------------------

def test_predictive_loss_3d_with_2d_predicted():
    d = _make(batch=2, with_predictor=True)
    s_hist = torch.randn(2, N_SYMBOLS, N_DIM)
    w_hist = torch.randn(2, MAX_DEPTH, N_DIM)
    d.snapshot(s_hist, w_hist)
    pred, _ = d.predict()
    assert pred is not None and pred.ndim == 2  # [B, s_dim]

    s_cur = torch.randn(2, N_SYMBOLS, N_DIM)
    w_cur = torch.randn(2, MAX_DEPTH, N_DIM)
    loss = d.predictive_loss(s_cur, w_cur, pred)
    assert loss is not None
    assert loss.ndim == 0


def test_predictive_loss_3d_with_1d_predicted_broadcasts():
    """1D predicted broadcast across the batch dim of the 3D input."""
    d = _make(batch=2, with_predictor=True)
    s_hist = torch.randn(N_SYMBOLS, N_DIM)
    w_hist = torch.randn(MAX_DEPTH, N_DIM)
    # Use the b=0 single-row predict for a 1D shape.
    d.snapshot(s_hist, w_hist, mask=torch.tensor([True, False]))
    pred, _ = d.predict(0)
    assert pred is not None and pred.ndim == 1

    s_cur = torch.randn(2, N_SYMBOLS, N_DIM)
    w_cur = torch.randn(2, MAX_DEPTH, N_DIM)
    loss = d.predictive_loss(s_cur, w_cur, pred)
    assert loss is not None


# -- per-row prime --------------------------------------------------------

def test_prime_2d_predicted_returns_batched():
    d = _make(batch=2, with_predictor=True)
    s_hist = torch.randn(2, N_SYMBOLS, N_DIM)
    w_hist = torch.randn(2, MAX_DEPTH, N_DIM)
    d.snapshot(s_hist, w_hist)
    pred, conf = d.predict()
    bias = d.prime(pred, conf, 0.5)
    assert bias is not None
    assert tuple(bias.shape) == (2, CONCEPT_DIM)


def test_prime_1d_predicted_returns_single_row():
    d = _make(batch=2, with_predictor=True)
    s = torch.randn(N_SYMBOLS, N_DIM)
    w = torch.randn(MAX_DEPTH, N_DIM)
    d.snapshot(s, w, mask=torch.tensor([True, False]))
    pred, conf = d.predict(0)
    bias = d.prime(pred, conf, 0.5)
    assert bias is not None
    assert tuple(bias.shape) == (CONCEPT_DIM,)


# -- reset clears every row ----------------------------------------------

def test_reset_clears_all_rows():
    d = _make(batch=3)
    s = torch.randn(3, N_SYMBOLS, N_DIM)
    w = torch.randn(3, MAX_DEPTH, N_DIM)
    d.snapshot(s, w)
    assert d._recent_count.sum().item() > 0
    d.reset()
    assert d._recent_count.sum().item() == 0
    assert d._prev_count.sum().item() == 0
    assert torch.allclose(d._recent, torch.zeros_like(d._recent))


# -- backward-compat: B=1 layer accepts 3D input via mean-pool ------------

def test_b1_layer_accepts_3d_input_via_mean_pool():
    """Production currently builds discourse at B=1 but passes 3D
    tensors from MentalModel.forward.  Until Task 9 cascades
    ensure_batch(B*K) at Start(), the B=1 fallback must still
    accept 3D input by mean-pooling.  Removing this fallback is
    Task 9's job, not Task 3's.
    """
    d = _make(batch=1)
    s = torch.randn(4, N_SYMBOLS, N_DIM)
    w = torch.randn(4, MAX_DEPTH, N_DIM)
    d.snapshot(s, w)
    # Snapshot wrote one pooled row to row 0.
    assert int(d._recent_count[0].item()) == 1
