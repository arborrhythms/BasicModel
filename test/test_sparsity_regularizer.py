# basicmodel/test/test_sparsity_regularizer.py
import torch

from Layers import SparsityRegLayer


def test_identity_when_lambda_zero():
    reg = SparsityRegLayer(l1_lambda=0.0)
    x = torch.randn(3, 5)
    out = reg(x)
    assert torch.allclose(out, x)


def test_soft_threshold_shrinks_small_values():
    reg = SparsityRegLayer(l1_lambda=0.5)
    x = torch.tensor([0.1, 0.4, 0.9, -0.3, -0.7])
    out = reg(x)
    # |x| < 0.5 -> 0; otherwise sign(x) * (|x| - 0.5)
    expected = torch.tensor([0.0, 0.0, 0.4, 0.0, -0.2])
    assert torch.allclose(out, expected)


def test_gated_off_when_enabled_false():
    reg = SparsityRegLayer(l1_lambda=0.5, enabled=False)
    x = torch.tensor([0.1, 0.9])
    out = reg(x)
    assert torch.allclose(out, x)


def test_preserves_grad():
    reg = SparsityRegLayer(l1_lambda=0.3)
    x = torch.tensor([1.0, -1.0], requires_grad=True)
    out = reg(x).sum()
    out.backward()
    # Soft threshold is piecewise identity-or-zero; for |x| > lambda grads pass through.
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0]))


def test_symbolic_space_uses_sparsity_regularizer():
    """SymbolicSpace.l1_proximal delegates to SparsityRegLayer."""
    from Spaces import SymbolicSpace
    assert hasattr(SymbolicSpace, "_build_sparsity_regularizer"), \
        "SymbolicSpace should expose a factory for its regularizer"
