"""Fixed lexical exhaustion rejects the entire batch before mutation (9b §4e)."""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_COMPILE", "eager")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "bin"))

import pytest
import torch
import torch.nn as nn


def _build_embedding_at_capacity():
    from Spaces import Embedding
    from embed import WordVectors, PretrainModel
    cap = 4
    dim = 3
    seed_keys = ["\x00", "a", "b", "c"]
    e = Embedding.__new__(Embedding)
    nn.Module.__init__(e)
    vecs = torch.randn(cap, dim)
    wv = WordVectors(vecs, seed_keys)
    e.wv = wv
    e.lexicon_capacity = cap
    e.byte_mode = False
    e._pending_counts = {}
    e._oov_fallback_count = 0
    e._oov_fallback_sample = []
    e._oov_fallback_sample_cap = 16
    wv._fixed_capacity = e.lexicon_capacity
    e.pretrain = PretrainModel(wv, learning_rate=0.01, neg_samples=2)
    return e


def test_overflow_names_capacity_and_preserves_shape():
    e = _build_embedding_at_capacity()
    parameter = e.wv._vectors
    with pytest.raises(RuntimeError, match='nVectors=4'):
        e.stage_oov(["x", "y", "z"])
    assert e.wv._vectors is parameter
    assert tuple(parameter.shape) == (4, 3)
    assert all(key not in e.wv.key_to_index for key in ("x", "y", "z"))


def test_overflow_does_not_corrupt_existing_rows():
    e = _build_embedding_at_capacity()
    snap = e.wv._vectors.detach().clone()
    with pytest.raises(RuntimeError, match='nVectors'):
        e.stage_oov(["aaa"])
    assert torch.equal(e.wv._vectors.data, snap)
    assert "aaa" not in e.wv.key_to_index


def test_overflow_is_atomic_with_some_capacity_remaining():
    e = _build_embedding_at_capacity()
    e.wv.index_to_key.pop()
    del e.wv.key_to_index['c']
    snap = e.wv._vectors.detach().clone()
    with pytest.raises(RuntimeError, match='nVectors'):
        e.stage_oov(["x", "y"])
    assert torch.equal(e.wv._vectors.data, snap)
    assert "x" not in e.wv.key_to_index and "y" not in e.wv.key_to_index
