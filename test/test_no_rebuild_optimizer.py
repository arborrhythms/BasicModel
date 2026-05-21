"""Optimizer state identity is preserved across OOV staging.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §4.5.
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


def _build_embedding(capacity=16, dim=4):
    from Spaces import Embedding
    from embed import WordVectors, PretrainModel
    e = Embedding.__new__(Embedding)
    nn.Module.__init__(e)
    seed_keys = ["\x00", "a", "b"]
    vecs = torch.randn(len(seed_keys), dim)
    wv = WordVectors(vecs, seed_keys)
    e.wv = wv
    e.lexicon_capacity = capacity
    e.byte_mode = False
    e._pending_counts = {}
    e._oov_fallback_count = 0
    e._oov_fallback_sample = []
    e._oov_fallback_sample_cap = 16
    e._inflate_to_capacity()
    e.pretrain = PretrainModel(wv, learning_rate=0.01, neg_samples=2)
    return e


def _take_step_to_init_adam_state(e):
    p = e.wv._vectors
    loss = (p ** 2).sum()
    loss.backward()
    e.pretrain.optimizer.step()
    e.pretrain.optimizer.zero_grad()


def test_optimizer_identity_preserved_across_stage_oov():
    e = _build_embedding()
    _take_step_to_init_adam_state(e)
    opt_before = e.pretrain.optimizer
    state_obj_before = e.pretrain.optimizer.state[e.wv._vectors]
    e.stage_oov(["foo", "bar"])
    opt_after = e.pretrain.optimizer
    state_obj_after = e.pretrain.optimizer.state[e.wv._vectors]
    assert opt_after is opt_before, "Optimizer was rebuilt (object identity changed)"
    assert state_obj_after is state_obj_before, (
        "Optimizer state for wv._vectors was rebuilt")


def test_newly_activated_row_has_zero_adam_moments():
    e = _build_embedding()
    _take_step_to_init_adam_state(e)
    e.stage_oov(["fresh"])
    idx = e.wv.key_to_index["fresh"]
    state = e.pretrain.optimizer.state[e.wv._vectors]
    assert torch.equal(state["exp_avg"][idx], torch.zeros_like(state["exp_avg"][idx]))
    assert torch.equal(state["exp_avg_sq"][idx], torch.zeros_like(state["exp_avg_sq"][idx]))


def test_stage_oov_skips_rebuild_when_preallocated():
    """Under the preallocated-reserve contract (parameter at full
    ``lexicon_capacity``), ``stage_oov`` does in-place writes — no
    optimizer rebuild. Identity-preserving optimizer is the gate."""
    e = _build_embedding()
    _take_step_to_init_adam_state(e)
    opt_before = e.pretrain.optimizer
    e.stage_oov(["new1", "new2"])
    assert e.pretrain.optimizer is opt_before, (
        "stage_oov on a preallocated codebook must NOT rebuild the "
        "optimizer; identity must be preserved")
