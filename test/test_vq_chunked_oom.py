"""VectorQuantize.forward must scale to large N without materializing
[N, V] distance or one-hot matrices.

Microbatch AR refactor produces flat tensors of length B*K*N inside the
body, where K = T - N + 1 can be in the thousands.  At realistic
configs this drives the [flat_N, V] cdist allocation past TiB scale.
The fix chunks cdist over rows, and replaces one_hot @ matmul with
bincount + index_add_ in the EMA update so neither path allocates an
[N, V] tensor.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                   # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest

from Spaces import VectorQuantize


def _vq(D=6, V=128, train=True, cosine=False):
    vq = VectorQuantize(dim=D, codebook_size=V, use_cosine_sim=cosine)
    vq.train(train)
    return vq


def test_chunked_indices_match_unchunked_l2():
    """Tiny chunk must produce identical indices and call cdist >1 times."""
    torch.manual_seed(0)
    D, V, N = 6, 128, 4096
    vq = _vq(D=D, V=V, train=False)
    x = torch.randn(N, D)

    # Reference: no chunking.
    vq._vq_chunk_rows = None
    quant_full, idx_full, _ = vq(x.clone())

    # Forced tiny chunk; spy on cdist call count.
    calls = {"n": 0}
    real_cdist = torch.cdist

    def _spy(a, b, *args, **kw):
        calls["n"] += 1
        return real_cdist(a, b, *args, **kw)

    torch.cdist = _spy
    try:
        vq._vq_chunk_rows = 256
        quant_chunk, idx_chunk, _ = vq(x.clone())
    finally:
        torch.cdist = real_cdist

    assert calls["n"] == (N + 255) // 256, (
        f"expected {(N+255)//256} cdist calls, got {calls['n']}")
    assert torch.equal(idx_chunk, idx_full)
    assert torch.allclose(quant_chunk, quant_full)


def test_chunked_indices_match_unchunked_cosine():
    torch.manual_seed(1)
    D, V, N = 8, 64, 2048
    vq = _vq(D=D, V=V, train=False, cosine=True)
    x = torch.randn(N, D)

    vq._vq_chunk_rows = None
    quant_full, idx_full, _ = vq(x.clone())
    vq._vq_chunk_rows = 128
    quant_chunk, idx_chunk, _ = vq(x.clone())

    assert torch.equal(idx_chunk, idx_full)
    assert torch.allclose(quant_chunk, quant_full)


def test_ema_update_uses_bincount_not_onehot_matmul():
    """EMA update must not allocate an [N, V] one-hot tensor.

    We monkeypatch `torch.nn.functional.one_hot` to raise; if the EMA
    update path takes the bincount branch the call succeeds.
    """
    torch.manual_seed(2)
    D, V, N = 6, 64, 1024
    vq = _vq(D=D, V=V, train=True)
    x = torch.randn(N, D)

    import torch.nn.functional as F
    original = F.one_hot

    def _boom(*a, **kw):
        raise AssertionError("one_hot must not be called in EMA update")

    F.one_hot = _boom
    try:
        vq(x)
    finally:
        F.one_hot = original


def test_large_flat_does_not_oom_l2():
    """Large flat (simulating B*K*N at body scale) processes without
    allocating an [N, V] distance matrix.

    Choose N*V deliberately past a 1 GB fp32 distance matrix to ensure
    the chunked path is exercised end-to-end.  Fits in CPU RAM if and
    only if chunking is real.
    """
    torch.manual_seed(3)
    # 256k rows * 8192 codebook * 4 bytes = 8 GB unchunked.
    # Chunked path keeps peak well under that.
    D, V, N = 6, 8192, 256_000
    vq = _vq(D=D, V=V, train=False)
    x = torch.randn(N, D)
    quant, indices, _ = vq(x)
    assert indices.shape == (N,)
    assert quant.shape == (N, D)
    assert int(indices.max().item()) < V
    assert int(indices.min().item()) >= 0


def test_large_flat_ema_does_not_oom():
    """EMA path on large N must succeed (bincount + index_add_)."""
    torch.manual_seed(4)
    D, V, N = 6, 8192, 256_000
    vq = _vq(D=D, V=V, train=True)
    x = torch.randn(N, D)
    quant, indices, _ = vq(x)
    assert indices.shape == (N,)
    # Codebook should have been updated (cluster_size accumulator moved).
    assert vq.cluster_size.sum().item() > V  # ones init = V; EMA pushed it higher


def test_dead_code_refresh_replaces_multiple_rows():
    """Dead-code refresh should replace expired rows via explicit indices."""
    torch.manual_seed(5)
    D, V, N = 4, 8, 32
    vq = _vq(D=D, V=V, train=True)
    vq.codebook_retire = True
    vq.threshold_ema_dead_code = 1
    with torch.no_grad():
        codebook = torch.full((V, D), 1000.0)
        codebook[0].zero_()
        vq.codebook.copy_(codebook)
        vq.embed_avg.copy_(codebook)
        vq.cluster_size.zero_()
        vq.cluster_size[0] = 2.0
    x = torch.zeros(N, D)

    _quant, _indices, _loss = vq(x)
    assert vq.cluster_size.shape == (V,)
    assert torch.allclose(vq.cluster_size[1:], torch.ones(V - 1))
    assert torch.allclose(vq.codebook[1:], torch.zeros(V - 1, D))


def test_codebook_retire_false_disables_expiration():
    """Default (codebook_retire=False) must skip dead-code replacement even
    when ``threshold_ema_dead_code`` is positive."""
    torch.manual_seed(6)
    D, V, N = 4, 8, 32
    vq = _vq(D=D, V=V, train=True)
    assert vq.codebook_retire is False
    vq.threshold_ema_dead_code = 1
    with torch.no_grad():
        codebook = torch.full((V, D), 1000.0)
        codebook[0].zero_()
        vq.codebook.copy_(codebook)
        vq.embed_avg.copy_(codebook)
        vq.cluster_size.zero_()
        vq.cluster_size[0] = 2.0
    x = torch.zeros(N, D)

    _quant, _indices, _loss = vq(x)
    # Without retirement, expired rows are left to the EMA to drift; they
    # are not reseeded from the batch and their cluster_size is not pinned
    # to the threshold.
    assert torch.allclose(vq.cluster_size[1:], torch.zeros(V - 1))
