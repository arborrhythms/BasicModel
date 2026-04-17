import torch

from basicmodel.bin.Spaces import topk_by_magnitude_per_batch, Codebook


def test_topk_keeps_largest_magnitude_per_row():
    x = torch.tensor([
        [0.0, 1.0, -3.0, 2.0, 0.5],
        [-0.1, 0.2, 0.0, 0.0, 10.0],
    ])
    out = topk_by_magnitude_per_batch(x, k=2)
    # Row 0: keep -3.0 and 2.0; zero the rest.
    # Row 1: keep 10.0 and 0.2; zero the rest.
    expected = torch.tensor([
        [0.0, 0.0, -3.0, 2.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, 10.0],
    ])
    assert torch.allclose(out, expected)


def test_topk_k_equals_width_is_identity():
    x = torch.randn(3, 7)
    out = topk_by_magnitude_per_batch(x, k=7)
    assert torch.allclose(out, x)


def test_topk_k_zero_is_zeros():
    x = torch.randn(2, 5)
    out = topk_by_magnitude_per_batch(x, k=0)
    assert torch.allclose(out, torch.zeros_like(x))


def test_codebook_forward_topk_prunes_activation():
    """When Codebook.forward receives topK>0, self.activation has at most
    topK nonzero entries per batch row, even though the codebook is wide."""
    torch.manual_seed(0)
    cb = Codebook()
    # Wide codebook: 16 prototypes, small input dim.
    cb.create(nInput=4, nVectors=16, nDim=3, customVQ=False, passThrough=False)
    cb.eval()
    # Non-passthrough, non-VQ path goes through the cosine-similarity loop.
    x = torch.randn(2, 4, 3)  # [batch=2, n_tokens=4, nDim=3]
    _ = cb.forward(x, topK=2)
    # activation shape: [batch, codebookSize] == [2, 16]
    assert cb.activation.shape == (2, 16)
    nonzero_per_row = (cb.activation.abs() > 1e-8).sum(dim=-1)
    assert torch.all(nonzero_per_row <= 2), \
        f"expected ≤2 nonzero per row, got {nonzero_per_row.tolist()}"


def test_codebook_forward_topk_zero_preserves_legacy_activation():
    """topK=0 (default) leaves self.activation unchanged from the legacy path."""
    torch.manual_seed(0)
    cb = Codebook()
    cb.create(nInput=4, nVectors=16, nDim=3, customVQ=False, passThrough=False)
    cb.eval()
    x = torch.randn(2, 4, 3)
    _ = cb.forward(x)  # no topK kwarg → legacy behavior
    assert cb.activation.shape == (2, 16)
    # Legacy activation may have any number of nonzero entries.

