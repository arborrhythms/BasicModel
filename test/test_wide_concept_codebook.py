import torch

from Spaces import Codebook

# ``topk_by_magnitude_per_batch`` was retired — the function body was
# inlined into ``Codebook.forward`` (search for "Inlined
# ``topk_by_magnitude_per_batch``" in bin/Spaces.py). The standalone
# unit tests for the helper are removed accordingly; the inlined
# behavior is exercised by the ``test_codebook_forward_topk_*`` tests
# below, which drive ``Codebook.forward`` directly.


def test_codebook_forward_topk_prunes_activation():
    """When Codebook.forward receives topK>0, self.activation has at most
    topK nonzero entries per batch row, even though the codebook is wide."""

    cb = Codebook()
    # Wide codebook: 16 prototypes, small input dim.
    cb.create(nInput=4, nVectors=16, nDim=3, customVQ=False)
    cb.eval()
    # Non-passthrough, non-VQ path goes through the cosine-similarity loop.
    x = torch.randn(2, 4, 3)  # [batch=2, n_tokens=4, nDim=3]
    _ = cb.forward(x, topK=2)
    # activation shape: [batch, codebookSize] == [2, 16]
    assert cb.activation.shape == (2, 16)
    nonzero_per_row = (cb.activation.abs() > 1e-8).sum(dim=-1)
    assert torch.all(nonzero_per_row <= 2), \
        f"expected <=2 nonzero per row, got {nonzero_per_row.tolist()}"


def test_codebook_forward_topk_zero_preserves_legacy_activation():
    """topK=0 (default) leaves self.activation unchanged from the legacy path."""

    cb = Codebook()
    cb.create(nInput=4, nVectors=16, nDim=3, customVQ=False)
    cb.eval()
    x = torch.randn(2, 4, 3)
    _ = cb.forward(x)  # no topK kwarg -> legacy behavior
    assert cb.activation.shape == (2, 16)
    # Legacy activation may have any number of nonzero entries.


# -------- Codebook.apply_gradient_estimator tests --------


def _estimator_grad(mode, e, q, upstream):
    """Run ``apply_gradient_estimator`` and return grad into ``e``."""
    e = e.detach().clone().requires_grad_(True)
    q = q.detach().clone().requires_grad_(True)
    out = Codebook.apply_gradient_estimator(e, q, mode=mode)
    out.backward(upstream)
    return e.grad, q.grad, out


def test_gradient_estimator_snap_forward_is_q_and_zero_grad_to_e():

    e = torch.randn(2, 4, requires_grad=True)
    q = torch.randn(2, 4, requires_grad=True)
    out = Codebook.apply_gradient_estimator(e, q, mode="snap")
    # Forward = q (value).
    assert torch.allclose(out, q)
    # Snap delivers zero gradient to e: the output is detached, so no
    # backward path reaches either e or q through this op.
    assert not out.requires_grad


def test_gradient_estimator_ste_passes_gradient_through_to_e():

    e = torch.randn(3, 4)
    q = torch.randn(3, 4)
    g = torch.randn_like(q)
    grad_e, _grad_q, out = _estimator_grad("ste", e, q, g)
    # Forward = q (as value).
    assert torch.allclose(out, q)
    # Backward = identity: grad flows straight into e.
    assert grad_e is not None
    assert torch.allclose(grad_e, g)


def test_gradient_estimator_rotation_scales_norm_and_rotates():

    e = torch.randn(5, 6)
    q = torch.randn(5, 6)
    g = torch.randn_like(q)
    grad_e, _grad_q, out = _estimator_grad("rotation", e, q, g)
    # Forward = q.
    assert torch.allclose(out, q)
    assert grad_e is not None
    # Per-row: ||grad_e|| ~= ||g|| * (||q|| / ||e||) (rotation preserves norm,
    # so only the scale factor changes magnitude).
    e_norm = e.norm(dim=-1)
    q_norm = q.norm(dim=-1)
    g_norm = g.norm(dim=-1)
    expected = g_norm * (q_norm / e_norm)
    actual = grad_e.norm(dim=-1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    # Direction differs from incoming when e and q are not collinear: at
    # least some row should have cos(grad_e, g) noticeably below 1.
    cos = (grad_e * g).sum(dim=-1) / (actual * g_norm + 1e-8)
    assert (cos < 0.999).any()


def test_gradient_estimator_invalid_mode_raises():
    e = torch.randn(2, 3)
    q = torch.randn(2, 3)
    import pytest
    with pytest.raises(ValueError):
        Codebook.apply_gradient_estimator(e, q, mode="bogus")


def test_codebook_commit_loss_basic():
    """commit_loss: MSE with stop-gradient on q, gradient flows to e only."""

    cb = Codebook()
    cb.create(nInput=2, nVectors=4, nDim=3, customVQ=False)
    e = torch.randn(2, 3, requires_grad=True)
    q = torch.randn(2, 3, requires_grad=True)
    loss = cb.commit_loss(e, q)
    assert loss.ndim == 0
    loss.backward()
    # Gradient reaches e but not q (q was detached internally).
    assert e.grad is not None
    assert q.grad is None or torch.allclose(q.grad, torch.zeros_like(q))


def test_codebook_commit_loss_zero_on_empty_inputs():
    """commit_loss returns zero when either operand is empty.

    The legacy passThrough short-circuit was retired with Stage 1; the
    only remaining zero-fallback is the empty-tensor guard.
    """
    cb = Codebook()
    cb.create(nInput=2, nVectors=4, nDim=3, customVQ=False)
    e = torch.empty(0, 3)
    q = torch.empty(0, 3)
    assert torch.equal(cb.commit_loss(e, q), torch.tensor(0.0))


# -------- Codebook.freeze_well_learned tests --------


def _step_grad(cb, per_entry_scale):
    """Push a single backward pass through the codebook weight with a
    known per-entry gradient scale so the hook records it."""
    w = cb.getW()
    nVec = w.shape[0]
    # Build a loss whose dW = per_entry_scale tiled across nDim -- each
    # row gets a distinct gradient magnitude.
    scale = per_entry_scale.view(-1, 1).to(w.device)
    (w * scale).sum().backward()
    w.grad = None


def test_codebook_freezing_stable_entry_gets_frozen():

    cb = Codebook()
    cb.create(nInput=2, nVectors=4, nDim=3, customVQ=True)
    cb.attach_freeze_hook(threshold=0.01, window=5)
    # Entry 0 has constant low gradient (stable); entry 1 is noisy.
    stable = torch.tensor([0.001, 0.5, 0.5, 0.5])
    for _ in range(6):
        noisy = torch.tensor([0.001, 0.5 + 0.3 * torch.randn(()).item(),
                              0.5, 0.5])
        _step_grad(cb, noisy)
    cb.freeze_well_learned()
    # Entry 0 was constant -> sigma ~= 0 -> frozen.
    assert cb.frozen_entries[0].item() is True


def test_codebook_freezing_noisy_entry_not_frozen():

    cb = Codebook()
    cb.create(nInput=2, nVectors=3, nDim=3, customVQ=True)
    cb.attach_freeze_hook(threshold=0.01, window=5)
    for _ in range(6):
        noisy = torch.tensor([1.0 + torch.randn(()).item(),
                              1.0 + torch.randn(()).item(),
                              1.0 + torch.randn(()).item()])
        _step_grad(cb, noisy)
    cb.freeze_well_learned()
    # All entries have unit-variance noise, well above threshold 0.01.
    assert not cb.frozen_entries.any().item()


def test_codebook_freezing_zeros_gradient_of_frozen_entries():

    cb = Codebook()
    cb.create(nInput=2, nVectors=3, nDim=3, customVQ=True)
    cb.attach_freeze_hook(threshold=0.01, window=3)
    # Feed stable grads so all entries freeze.
    for _ in range(4):
        _step_grad(cb, torch.tensor([0.001, 0.001, 0.001]))
    cb.freeze_well_learned()
    assert cb.frozen_entries.all().item()
    # Next backward: hook should zero the gradient for frozen rows.
    w = cb.getW()
    (w * w).sum().backward()
    assert torch.allclose(w.grad, torch.zeros_like(w.grad))

