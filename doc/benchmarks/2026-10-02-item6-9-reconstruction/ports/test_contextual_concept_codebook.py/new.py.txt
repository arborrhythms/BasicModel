"""§20: reconstruction owns unconstrained, checkpointed concept parameters."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from Models import BaseModel
from Spaces import Codebook


def _devices():
    return ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])


def _codebook(device, rows=8, dim=4):
    cb = Codebook()
    cb.nVectors = rows
    cb.W = nn.Parameter(torch.randn(rows, dim, device=device) * 2)
    cb.sparse_lookup_grad = True
    return cb


@pytest.mark.parametrize('device', _devices())
def test_reconstruction_lookup_trains_selected_codes_without_projection(device):
    cb = _codebook(device)
    before = cb.W.detach().clone()
    scale = nn.Parameter(torch.ones((), device=device))
    selected = torch.tensor([2, 7], device=device)
    loss = (cb.lookup_rows(selected) * scale).square().sum()
    loss.backward()
    assert isinstance(cb.W, nn.Parameter) and cb.W.requires_grad
    assert 'W' in dict(cb.named_parameters())
    assert cb.W.grad is not None and scale.grad is not None
    grad = cb.W.grad.to_dense() if cb.W.grad.is_sparse else cb.W.grad
    assert grad[2].abs().sum() > 0 and grad[7].abs().sum() > 0
    assert grad[5].count_nonzero() == 0
    with torch.no_grad():
        cb.W.add_(grad, alpha=-.07)
    torch.testing.assert_close(cb.W, before - .07 * grad)
    torch.testing.assert_close(cb.W[5], before[5], rtol=0, atol=0)
    assert not torch.allclose(cb.W[selected].norm(dim=-1), torch.ones(2, device=device))


def test_dictionary_is_included_once_in_optimizer_groups():
    cb = _codebook('cpu')
    dense = nn.Parameter(torch.randn(4, 4))
    model = BaseModel()
    model.spaces = [SimpleNamespace(getParameters=lambda: [cb.W, dense])]
    model.conceptualSpaces = [SimpleNamespace(similarity_codebook=cb)]
    optimizer = model.getOptimizer(lr=1e-3)
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert sum(p is dense for p in params) == 1
    assert sum(p is cb.W for p in params) == 1


def test_parameter_preserves_frozen_capacity_and_state_dict_storage():
    cb = _codebook('cpu')
    cb.freeze_capacity('reconstruction-test')
    original = cb.W
    cb.to('cpu')
    cb._assert_frozen_parameter_identity()
    assert cb.W is original and cb.W.requires_grad
    payload = {name: value.detach().clone() for name, value in cb.state_dict().items()}
    restored = _codebook('cpu')
    restored.freeze_capacity('reconstruction-test')
    restored.load_state_dict(payload)
    restored._assert_frozen_parameter_identity()
    torch.testing.assert_close(restored.W, cb.W)
    assert isinstance(restored.W, nn.Parameter) and restored.W.requires_grad
