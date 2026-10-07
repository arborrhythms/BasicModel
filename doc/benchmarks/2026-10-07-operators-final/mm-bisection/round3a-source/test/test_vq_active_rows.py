"""Static-shape logical occupancy for :class:`Layers.VectorQuantize`."""

from __future__ import annotations

import os
import sys

import pytest
import torch

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import VectorQuantize


def _install_rows(vq, rows):
    rows = torch.as_tensor(rows, dtype=vq.codebook.dtype)
    if vq.use_cosine_sim:
        rows = torch.nn.functional.normalize(rows, dim=-1)
    # The property setter refreshes the EMA and squared-norm mirrors.
    vq.codebook = rows


def test_active_prefix_is_persistent_and_changes_no_storage_identity():
    vq = VectorQuantize(dim=3, codebook_size=7)
    assert vq.active_mask.dtype == torch.bool
    assert vq.active_mask.tolist() == [True] * 7
    assert "active_mask" in vq.state_dict()
    assert "active_mask" not in dict(vq.named_parameters())

    identities = {
        "codebook": id(vq.codebook),
        "cluster_size": vq.cluster_size.data_ptr(),
        "embed_avg": vq.embed_avg.data_ptr(),
        "norms": vq._b_norms_sq.data_ptr(),
        "mask": vq.active_mask.data_ptr(),
    }
    assert vq.set_active_rows(3) is vq
    assert vq.active_mask.tolist() == [True, True, True, False, False, False, False]
    assert id(vq.codebook) == identities["codebook"]
    assert vq.cluster_size.data_ptr() == identities["cluster_size"]
    assert vq.embed_avg.data_ptr() == identities["embed_avg"]
    assert vq._b_norms_sq.data_ptr() == identities["norms"]
    assert vq.active_mask.data_ptr() == identities["mask"]

    with pytest.raises(ValueError, match="1 <= n <= 7"):
        vq.set_active_rows(0)
    with pytest.raises(ValueError, match="1 <= n <= 7"):
        vq.set_active_rows(8)


@pytest.mark.parametrize("cosine", [False, True])
def test_selection_cannot_name_an_inactive_exact_match(cosine):
    vq = VectorQuantize(
        dim=2, codebook_size=5, use_cosine_sim=cosine)
    if cosine:
        rows = [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 1.0],  # exact query direction, but inactive
        ]
        query = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    else:
        rows = [
            [0.0, 0.0],
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, -10.0],
            [1.0, 1.0],  # exact query value, but inactive
        ]
        query = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    _install_rows(vq, rows)
    vq.set_active_rows(2)
    vq._vq_chunk_rows = 1
    vq.eval()

    _quantized, indices, _loss = vq(query)
    assert bool((indices < 2).all())
    assert not bool((indices == 4).any())


@pytest.mark.parametrize("cosine", [False, True])
def test_reserved_vq_is_bit_exact_to_physical_active_size(cosine):
    """A physical reserve must execute exactly the same A-row math."""
    A, physical, D = 4, 11, 3
    torch.manual_seed(117)
    rows = torch.randn(A, D)
    if cosine:
        rows = torch.nn.functional.normalize(rows, dim=-1)
    cluster = torch.rand(A) + 1.0
    embed = torch.randn(A, D)
    x = torch.randn(17, D)

    reserved = VectorQuantize(
        dim=D, codebook_size=physical, decay=0.6,
        use_cosine_sim=cosine)
    exact = VectorQuantize(
        dim=D, codebook_size=A, decay=0.6,
        use_cosine_sim=cosine)
    reserved.set_active_rows(A)
    with torch.no_grad():
        reserved.codebook[:A].copy_(rows)
        exact.codebook.copy_(rows)
        reserved.cluster_size[:A].copy_(cluster)
        exact.cluster_size.copy_(cluster)
        reserved.embed_avg[:A].copy_(embed)
        exact.embed_avg.copy_(embed)
        reserved._b_norms_sq[:A].copy_((rows ** 2).sum(dim=-1))
        exact._b_norms_sq.copy_((rows ** 2).sum(dim=-1))
        reserved.codebook[A:].fill_(31.0)
        reserved.cluster_size[A:].fill_(37.0)
        reserved.embed_avg[A:].fill_(41.0)
        reserved._b_norms_sq[A:].fill_(43.0)
    tail = {
        "codebook": reserved.codebook[A:].detach().clone(),
        "cluster": reserved.cluster_size[A:].clone(),
        "embed": reserved.embed_avg[A:].clone(),
        "norms": reserved._b_norms_sq[A:].clone(),
    }
    reserved.train()
    exact.train()

    q_reserved, i_reserved, loss_reserved = reserved(x.clone())
    q_exact, i_exact, loss_exact = exact(x.clone())

    assert torch.equal(i_reserved, i_exact)
    assert torch.equal(q_reserved, q_exact)
    assert torch.equal(loss_reserved, loss_exact)
    assert torch.equal(reserved.codebook[:A], exact.codebook)
    assert torch.equal(reserved.cluster_size[:A], exact.cluster_size)
    assert torch.equal(reserved.embed_avg[:A], exact.embed_avg)
    assert torch.equal(reserved._b_norms_sq[:A], exact._b_norms_sq)
    assert torch.equal(reserved.codebook[A:].detach(), tail["codebook"])
    assert torch.equal(reserved.cluster_size[A:], tail["cluster"])
    assert torch.equal(reserved.embed_avg[A:], tail["embed"])
    assert torch.equal(reserved._b_norms_sq[A:], tail["norms"])


def test_ema_and_dead_code_revival_leave_inactive_rows_bit_exact():
    vq = VectorQuantize(
        dim=3,
        codebook_size=6,
        decay=0.5,
        threshold_ema_dead_code=1,
        codebook_retire=True,
    )
    vq.set_active_rows(2)
    vq.train()
    with torch.no_grad():
        vq.cluster_size.zero_()
        vq.cluster_size[:2] = 2.0
        vq.codebook[2:].fill_(37.0)
        vq.embed_avg[2:].fill_(41.0)
        vq._b_norms_sq.copy_((vq.codebook ** 2).sum(dim=-1))
    before_codebook = vq.codebook[2:].detach().clone()
    before_cluster = vq.cluster_size[2:].clone()
    before_embed = vq.embed_avg[2:].clone()

    # An external partition is not allowed to re-enable inactive capacity.
    vq.update_mask_fn = lambda V, device: torch.ones(
        V, dtype=torch.bool, device=device)
    _quantized, indices, _loss = vq(torch.zeros(32, 3))

    assert bool((indices < 2).all())
    assert torch.equal(vq.codebook[2:].detach(), before_codebook)
    assert torch.equal(vq.cluster_size[2:], before_cluster)
    assert torch.equal(vq.embed_avg[2:], before_embed)


def test_novelty_growth_uses_only_free_slots_inside_active_prefix():
    vq = VectorQuantize(dim=4, codebook_size=8)
    vq.set_active_rows(3)
    vq.train()
    with torch.no_grad():
        vq.cluster_size.zero_()
        vq.codebook[3:].fill_(23.0)
        vq.embed_avg[3:].fill_(29.0)
    before_codebook = vq.codebook[3:].detach().clone()
    before_cluster = vq.cluster_size[3:].clone()
    before_embed = vq.embed_avg[3:].clone()

    inserted = vq.grow_on_novelty(torch.randn(6, 4), eps=1e-3)

    assert inserted == 3
    assert int((vq.cluster_size[:3] > 0).sum().item()) == 3
    assert torch.equal(vq.codebook[3:].detach(), before_codebook)
    assert torch.equal(vq.cluster_size[3:], before_cluster)
    assert torch.equal(vq.embed_avg[3:], before_embed)


def test_active_mask_roundtrips_and_old_state_preserves_configured_prefix():
    source = VectorQuantize(dim=2, codebook_size=5)
    source.set_active_rows(2)
    state = source.state_dict()

    restored = VectorQuantize(dim=2, codebook_size=5)
    restored.load_state_dict(state, strict=True)
    assert restored.active_mask.tolist() == [True, True, False, False, False]

    legacy = {
        name: value.detach().clone()
        for name, value in state.items()
        if name != "active_mask"
    }
    compatible = VectorQuantize(dim=2, codebook_size=5)
    compatible.set_active_rows(1)
    compatible.load_state_dict(legacy, strict=True)
    assert compatible.active_mask.tolist() == [True, False, False, False, False]

    # A standalone legacy load still has the historical all-active default.
    default_compatible = VectorQuantize(dim=2, codebook_size=5)
    default_compatible.load_state_dict(legacy, strict=True)
    assert default_compatible.active_mask.tolist() == [True] * 5


def test_checkpoint_rejects_non_prefix_active_mask():
    vq = VectorQuantize(dim=2, codebook_size=5)
    state = {
        name: value.detach().clone()
        for name, value in vq.state_dict().items()
    }
    state["active_mask"] = torch.tensor(
        [True, False, True, False, False])
    with pytest.raises(RuntimeError, match="non-empty active prefix"):
        vq.load_state_dict(state, strict=True)


def test_forward_does_not_read_a_tensor_scalar_on_host(monkeypatch):
    """The cached Python A boundary avoids ``active_mask.sum().item()``."""
    vq = VectorQuantize(dim=3, codebook_size=13, decay=0.5)
    vq.set_active_rows(4)
    vq.train()

    def _forbid_item(self, *args, **kwargs):
        raise AssertionError("VectorQuantize.forward performed Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    quantized, indices, loss = vq(torch.randn(9, 3))
    assert quantized.shape == (9, 3)
    assert indices.shape == (9,)
    assert loss.shape == ()


def test_ema_temporaries_are_sized_to_active_not_physical_rows(monkeypatch):
    A, physical, D = 3, 17, 4
    vq = VectorQuantize(dim=D, codebook_size=physical)
    vq.set_active_rows(A)
    vq.train()
    allocations = []
    original_zeros = torch.zeros

    def _tracked_zeros(*shape, **kwargs):
        normalized = tuple(shape[0]) if (
            len(shape) == 1 and isinstance(shape[0], (tuple, list))
        ) else tuple(shape)
        allocations.append(normalized)
        return original_zeros(*shape, **kwargs)

    monkeypatch.setattr(torch, "zeros", _tracked_zeros)
    vq(torch.randn(8, D))

    assert (A, D) in allocations
    assert (A,) in allocations
    assert (physical, D) not in allocations
    assert (physical,) not in allocations
