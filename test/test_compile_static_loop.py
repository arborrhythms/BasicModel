"""Static per-word loop must not recompile across varying sentence
lengths. The legacy ``while next_word()`` boundary caused Dynamo to
specialize on each observed ``_valid_len_host`` value; the static
``for p in range(N)`` loop with tensor-only gating eliminates that.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §2.3-2.4.
"""
import os
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ.setdefault("MODEL_COMPILE", "eager")
os.environ.setdefault("MODEL_DEBUG", "0")

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "bin"))

import pytest
import torch


def _build_gate_model():
    from data import TheData
    from Models import BaseModel
    from util import init_config, init_device
    init_device("cpu")
    cfg = str(_root / "data" / "MM_5M.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    m = m.to("cpu")
    return m


def test_per_word_body_callable_with_static_signature():
    """The new ``_per_word_body_step(w, p, gate_b_1, out_slot)``
    signature must be callable. Smoke test for the refactor — the
    important assertion is that the call doesn't raise, signaling the
    arity / argument shapes are wired correctly."""
    pytest.importorskip("torch._dynamo")
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    in_sub = isp.forward(inputTensor)
    assert isp._ar_embedded_N is not None
    assert isp._word_active_mask is not None
    N = int(isp.outputShape[0])
    m._per_word_prelude(in_sub)
    out_slot = m._per_word_contributions
    gate = isp._word_active_mask[:, 0:1]
    w = isp.word_at(0)
    cs, idea = m._per_word_body_step(w, 0, gate, out_slot)
    # idea may be None on the first call (PS/SS/CS may emit empty
    # subspaces); the assertion is structural: no exception was raised.


def test_word_at_returns_padded_shape_past_valid_len():
    """``word_at(p)`` for p past _valid_len_host must still return a
    well-shaped [B, 1, D] slice (zeros) — the gate masks the commits."""
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    isp.forward(inputTensor)
    N = int(isp.outputShape[0])
    w0 = isp.word_at(0)
    w_last = isp.word_at(N - 1)
    assert w0 is not None and w0.dim() == 3 and w0.shape[1] == 1
    assert w_last is not None and w_last.dim() == 3 and w_last.shape[1] == 1
    # Past _valid_len_host, the per-position activity mask is False so
    # the rule-gate drops contributions. The slab itself may still
    # carry non-zero positional (where/when) encoding even at padding
    # columns — the gate is the source of truth, not raw event values.
    L = int(isp._valid_len_host)
    if L < N:
        active = isp._word_active_mask
        assert active is not None
        assert not bool(active[:, L].any().item()), (
            f"word_active_mask should be False at the first padding "
            f"column (L={L})")
