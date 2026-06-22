"""Staged discourse prediction tensors are cast to the active autocast
dtype in ``_begin_step`` so the compiled forward never sees a dtype
mismatch that would split the graph.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §2.7.
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
    cfg = str(_root / "data" / "MM_20M.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    return m.to("cpu")


@pytest.mark.parametrize("mode,dtype", [
    ("bf16", torch.bfloat16),
    ("off",  None),
])
def test_staged_prediction_cast_to_amp_dtype(mode, dtype):
    """``_begin_step`` re-casts the parked ``(pred, conf)`` tuple to the
    active autocast dtype. When MODEL_AMP=off, no cast is applied."""
    import util as _util
    saved_mode = _util.MODEL_AMP
    _util.MODEL_AMP = mode
    try:
        m = _build_gate_model()
        if m.symbolSpace is None or m.symbolSpace.discourse is None:
            pytest.skip("model has no discourse layer")
        disc = m.symbolSpace.discourse
        disc._staged_prediction = (
            torch.randn(1, 4, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32))
        if m._compiled_step is None:
            m._compiled_step = lambda *a, **kw: None  # enable cast branch
        inp = m.inputSpace.prepInput(list(m.inputSpace.getTrainData()[0][:1]))
        m._begin_step(inp)
        staged = disc._staged_prediction
        if staged is None:
            pytest.skip("discourse cleared staged tuple")
        pred, conf = staged
        if dtype is None:
            assert pred is None or pred.dtype == torch.float32
        else:
            assert pred is None or pred.dtype == dtype, (
                f"pred.dtype={pred.dtype if pred is not None else None}, "
                f"expected {dtype}")
    finally:
        _util.MODEL_AMP = saved_mode
