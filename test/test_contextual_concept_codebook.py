"""Buffer-owned rotational dictionary for the serial ConceptualSpace."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Models import BaseModel  # noqa: E402
from Spaces import Codebook  # noqa: E402


def _devices():
    values = ["cpu"]
    if torch.backends.mps.is_available():
        values.append("mps")
    return values


def _codebook(device, rows=32, dim=8):
    cb = Codebook()
    cb.W = nn.Parameter(F.normalize(
        torch.randn(rows, dim, device=device), dim=-1))
    cb.sparse_lookup_grad = True
    cb.enable_contextual_rotation()
    return cb


@pytest.mark.parametrize("device", _devices())
def test_contextual_codebook_is_non_grad_buffer_and_rotates_without_projection(
        device):
    torch.manual_seed(3)
    cb = _codebook(device)
    before = cb.W.detach().clone()
    scale = nn.Parameter(torch.ones((), device=device))

    # The lookup preserves an eager compiler boundary while W itself remains a
    # true non-autograd buffer. Its consumer can differentiate normally, but
    # the dictionary receives no gradient at all.
    rows = cb.lookup_rows(torch.tensor([2, 7], device=device))
    assert rows.requires_grad
    (rows * scale).sum().backward()
    assert not isinstance(cb.W, nn.Parameter)
    assert not cb.W.requires_grad
    assert "W" in dict(cb.named_buffers())
    assert cb.W.grad is None
    assert scale.grad is not None

    cb.rotate_rows(
        torch.tensor([2, 7], device=device),
        torch.randn(2, int(cb.W.shape[1]), device=device), 0.07)
    if device == "mps":
        torch.mps.synchronize()
    torch.testing.assert_close(
        cb.W.detach()[torch.tensor([2, 7], device=device)].norm(dim=-1),
        torch.ones(2, device=device), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(cb.W.detach()[5], before[5], rtol=0.0, atol=0.0)
    assert not torch.allclose(cb.W.detach()[2], before[2])


def _context_model(cb, device):
    model = BaseModel()
    model.contextual_concept_learning_rate = 0.01
    model.contextual_concept_negatives = 4
    model._training_step_count = 11
    model._contextual_concept_codebooks = (cb,)
    model.conceptualSpaces = [types.SimpleNamespace(similarity_codebook=cb)]
    model.inputSpace = types.SimpleNamespace(
        _ar_word_concept_rows=torch.tensor(
            [[1, 2, 1, -1], [3, 4, -1, -1]], device=device),
        _word_active_mask=torch.tensor(
            [[True, True, True, False], [True, True, False, False]],
            device=device),
    )
    return model


@pytest.mark.parametrize("device", _devices())
def test_contextual_reducer_is_deterministic_row_reduced_and_skips_normalizer(
        device):
    torch.manual_seed(17)
    source = _codebook(device)
    copy = _codebook(device)
    with torch.no_grad():
        copy.W.copy_(source.W)
    model_a = _context_model(source, device)
    model_b = _context_model(copy, device)
    untouched = source.W.detach()[5].clone()

    assert model_a._update_contextual_concept_codebooks() == 4
    assert model_b._update_contextual_concept_codebooks() == 4
    if device == "mps":
        torch.mps.synchronize()
    torch.testing.assert_close(source.W, copy.W, rtol=2e-6, atol=2e-6)
    touched = source.W.detach()[torch.tensor([1, 2, 3, 4], device=device)]
    torch.testing.assert_close(
        touched.norm(dim=-1), torch.ones(4, device=device),
        rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(source.W.detach()[5], untouched, rtol=0.0, atol=0.0)

    # The retained legacy normalizer must be a no-op for the context-owned
    # table: rotations, not a second row write, maintain the unit invariant.
    rotated = source.W.detach().clone()
    model_a._normalize_conceptual_codebooks()
    torch.testing.assert_close(source.W, rotated, rtol=0.0, atol=0.0)


def test_context_owned_dictionary_is_excluded_from_optimizer_groups():
    cb = _codebook("cpu")
    dense = nn.Parameter(torch.randn(4, 4))
    fake_space = types.SimpleNamespace(getParameters=lambda: [cb.W, dense])
    model = BaseModel()
    model.spaces = [fake_space]
    model.conceptualSpaces = [types.SimpleNamespace(similarity_codebook=cb)]

    optimizer = model.getOptimizer(lr=1e-3)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert any(p is dense for p in params)
    assert not any(p is cb.W for p in params)


def test_contextual_buffer_preserves_frozen_capacity_and_state_dict_storage():
    def _frozen_buffer():
        result = Codebook()
        result.nVectors = 8
        result.W = nn.Parameter(F.normalize(torch.randn(8, 4), dim=-1))
        result.freeze_capacity("contextual-test")
        result.enable_contextual_rotation()
        return result

    cb = _frozen_buffer()
    original = cb.W

    assert not isinstance(cb.W, nn.Parameter)
    assert not cb.W.requires_grad
    assert cb.W.data_ptr() == original.data_ptr()
    assert "W" in cb.state_dict()
    cb._assert_frozen_parameter_identity()
    cb.to("cpu")
    cb._assert_frozen_parameter_identity()

    # Buffers stay checkpoint-persistent, while the destination remains a
    # non-grad owner after a normal state_dict restore.
    payload = {name: value.detach().clone()
               for name, value in cb.state_dict().items()}
    restored = _frozen_buffer()
    restored.load_state_dict(payload)
    torch.testing.assert_close(restored.W, cb.W)
    assert not isinstance(restored.W, nn.Parameter)
    assert not restored.W.requires_grad
