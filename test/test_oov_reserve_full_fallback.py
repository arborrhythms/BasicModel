"""Reserve-full OOV keys route to fallback (counter + sample), codebook
shape unchanged.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §4.4.
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


def _build_embedding_at_capacity():
    from Spaces import Embedding
    from embed import WordVectors, PretrainModel
    cap = 4
    dim = 3
    seed_keys = ["\x00", "a", "b", "c"]
    e = Embedding.__new__(Embedding)
    nn.Module.__init__(e)
    vecs = torch.randn(len(seed_keys), dim)
    wv = WordVectors(vecs, seed_keys)
    e.wv = wv
    e.lexicon_capacity = cap
    e.byte_mode = False
    e._pending_counts = {}
    e._oov_fallback_count = 0
    e._oov_fallback_sample = []
    e._oov_fallback_sample_cap = 16
    e._inflate_to_capacity()
    e.pretrain = PretrainModel(wv, learning_rate=0.01, neg_samples=2)
    return e


def test_overflow_increments_counter_and_captures_sample():
    e = _build_embedding_at_capacity()
    shape_before = tuple(e.wv._vectors.shape)
    overflow = e.stage_oov(["x", "y", "z"])
    assert overflow == ["x", "y", "z"], overflow
    assert e._oov_fallback_count == 3
    assert e._oov_fallback_sample == ["x", "y", "z"]
    assert tuple(e.wv._vectors.shape) == shape_before


def test_overflow_does_not_corrupt_existing_rows():
    e = _build_embedding_at_capacity()
    snap = e.wv._vectors.data.clone()
    e.stage_oov(["aaa"])
    assert torch.equal(e.wv._vectors.data, snap)
    assert "aaa" not in e.wv.key_to_index


def test_overflow_sample_bounded():
    e = _build_embedding_at_capacity()
    e._oov_fallback_sample_cap = 2
    overflow = e.stage_oov([f"k{i}" for i in range(10)])
    assert len(overflow) == 10
    assert len(e._oov_fallback_sample) == 2
    assert e._oov_fallback_count == 10
