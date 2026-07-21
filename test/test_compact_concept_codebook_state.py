"""Memory contract for indexed-only ConceptualSpace codebooks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Layers import VectorQuantize  # noqa: E402
from Spaces import Codebook, ConceptualSpace  # noqa: E402
from test_basicmodel import _populate_test_config  # noqa: E402


def _configure(dim=8, rows=64, slots=4):
    _populate_test_config(
        inputDim=dim, perceptDim=dim, conceptDim=dim, symbolDim=dim,
        wordDim=dim, outputDim=dim,
        nInput=slots, nPercepts=slots, nConcepts=rows,
        nSymbols=rows, nWords=rows, nOutput=rows,
        nWhere=0, nWhen=0,
    )


def test_vector_quantize_legacy_default_keeps_full_ema_state():
    vq = VectorQuantize(dim=7, codebook_size=19)

    assert vq.ema_state_enabled is True
    assert vq.ema_update is True
    assert tuple(vq.cluster_size.shape) == (19,)
    assert tuple(vq.embed_avg.shape) == (19, 7)
    state = vq.state_dict()
    assert "cluster_size" in state
    assert "embed_avg" in state


def test_vector_quantize_compact_mode_omits_ema_and_rejects_ema_ops():
    with pytest.raises(ValueError, match="ema_update"):
        VectorQuantize(
            dim=7, codebook_size=19, allocate_ema_state=False)

    vq = VectorQuantize(
        dim=7, codebook_size=19,
        ema_update=False, allocate_ema_state=False)
    assert vq.ema_state_enabled is False
    assert not hasattr(vq, "cluster_size")
    assert not hasattr(vq, "embed_avg")
    assert set(vq.state_dict()) == {
        "_codebook", "_b_norms_sq", "active_mask"}

    vq.train()
    with pytest.raises(RuntimeError, match="requires EMA state"):
        vq.grow_on_novelty(torch.randn(2, 7), eps=0.1)


def test_codebook_compact_contract_preserves_w_active_mask_and_checkpoint():
    _configure()
    source = Codebook().configure_vq_ema(False)
    source.create(4, 64, 8, customVQ=True)
    source.vq.set_active_rows(13)
    source.freeze_capacity("compact-test")

    keys = set(source.state_dict())
    assert "vq.embed_avg" not in keys
    assert "vq.cluster_size" not in keys
    assert "W" in keys
    assert "vq.active_mask" in keys
    assert source.vq.codebook is source.W

    saved = source.state_dict()
    restored = Codebook().configure_vq_ema(False)
    restored.create(4, 64, 8, customVQ=True)
    restored.load_state_dict(saved, strict=True)

    torch.testing.assert_close(restored.W, source.W)
    assert torch.equal(restored.vq.active_mask, source.vq.active_mask)
    assert restored.vq._active_rows_count == 13
    assert restored.vq.codebook is restored.W


def test_indexed_conceptual_stages_share_one_compact_dictionary():
    _configure(dim=16)
    first = ConceptualSpace(
        [4, 16], [64, 16], [4, 16],
        indexed_similarity_codebook=True)
    second = ConceptualSpace(
        [4, 16], [64, 16], [4, 16],
        shared_similarity_codebook=first.similarity_codebook,
        indexed_similarity_codebook=True)

    cb = first.similarity_codebook
    assert second.similarity_codebook is cb
    assert cb.vq.ema_state_enabled is False
    assert cb.vq.ema_update is False
    assert "vq.embed_avg" not in cb.state_dict()
    assert "vq.cluster_size" not in cb.state_dict()


def test_codebook_ema_policy_cannot_change_after_construction():
    _configure()
    cb = Codebook()
    cb.create(4, 16, 8, customVQ=True)
    with pytest.raises(RuntimeError, match="before create"):
        cb.configure_vq_ema(False)
