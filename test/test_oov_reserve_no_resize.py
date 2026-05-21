"""OOV staging preserves codebook Parameter shape and identity.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §4.1, §4.3.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_COMPILE", "eager")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "bin"))

import torch
import torch.nn as nn


def _build_embedding(capacity=64, dim=8, seed_keys=("\x00", "a", "b")):
    from Spaces import Embedding
    from embed import WordVectors
    e = Embedding.__new__(Embedding)
    nn.Module.__init__(e)
    vecs = torch.randn(len(seed_keys), dim)
    wv = WordVectors(vecs, list(seed_keys))
    e.wv = wv
    e.lexicon_capacity = capacity
    e.byte_mode = False
    e._pending_counts = {}
    e._oov_fallback_count = 0
    e._oov_fallback_sample = []
    e._oov_fallback_sample_cap = 16
    e._inflate_to_capacity()
    from embed import PretrainModel
    e.pretrain = PretrainModel(wv, learning_rate=0.01, neg_samples=2)
    return e


def test_stage_oov_does_not_resize_parameter():
    e = _build_embedding(capacity=32, dim=4)
    p_before = e.wv._vectors
    shape_before = tuple(p_before.shape)
    e.stage_oov(["foo", "bar", "baz"])
    p_after = e.wv._vectors
    assert tuple(p_after.shape) == shape_before, (
        f"Parameter shape changed: {shape_before} -> {tuple(p_after.shape)}")
    assert p_after is p_before, (
        "Parameter identity changed (Parameter object was reassigned)")


def test_stage_oov_activates_keys_visible_via_lookup():
    e = _build_embedding(capacity=32, dim=4)
    e.stage_oov(["hello", "world"])
    assert "hello" in e.wv.key_to_index
    assert "world" in e.wv.key_to_index
    assert "hello" in e.pretrain.key_to_index
    assert e.wv.key_to_index["hello"] == 3  # after 3 seed keys
    assert e.wv.key_to_index["world"] == 4


def test_stage_oov_in_place_writes_into_reserve_rows():
    e = _build_embedding(capacity=32, dim=4, seed_keys=("\x00",))
    v_hello = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    e.stage_oov(["hello"], vectors=v_hello)
    idx = e.wv.key_to_index["hello"]
    assert idx == 1
    written = e.wv._vectors.data[idx]
    target = v_hello.squeeze(0)
    assert torch.allclose(written, target, atol=1e-5)
