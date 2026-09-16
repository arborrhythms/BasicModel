"""Staged discourse prediction tensors are cast to the active autocast
dtype in ``runBatch`` so the compiled forward never sees a dtype
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
    cfg = str(_root / "data" / "MM_20M_legacy.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    return m.to("cpu")


@pytest.mark.parametrize("mode,dtype", [
    ("bf16", torch.bfloat16),
    ("off",  torch.float32),
])
@pytest.mark.parametrize("train", [True, False])
def test_staged_prediction_cast_to_amp_dtype(mode, dtype, train):
    """Compiled ``runBatch`` re-casts the parked ``(pred, conf)`` tuple to the
    active autocast dtype. When MODEL_AMP=off, no cast is applied."""
    import util as _util
    saved_mode = _util.MODEL_AMP
    _util.MODEL_AMP = mode
    try:
        m = _build_gate_model()
        if m.symbolSpace is None or m.symbolSpace.discourse is None:
            pytest.skip("model has no discourse layer")
        disc = m.symbolSpace.discourse
        m.symbolSpace.ensure_microbatch(1, 1)
        with torch.no_grad():
            disc.observe(torch.randn(1, 1, disc.sentence_dim))
        assert disc._s_count[0] > 0
        class _StagingComplete(Exception):
            pass

        def _stop_after_staging(compiled, *args, **kwargs):
            assert compiled == train
            assert torch.is_grad_enabled() == train
            raise _StagingComplete

        m._compiled_word_loop_fullgraph = False
        m._compiled_word_steps = {}
        m._compiled_step = lambda *a, **kw: _stop_after_staging(True, *a, **kw)
        # Evaluation deliberately uses the eager body under no_grad so it
        # does not retrace the training graph with a different grad-mode guard.
        m.forward = lambda *a, **kw: _stop_after_staging(False, *a, **kw)
        optimizer = torch.optim.SGD(m.parameters(), lr=0.001) if train else None
        inp = m.inputSpace.prepInput(list(m.inputSpace.getTrainData()[0][:1]))
        with pytest.raises(_StagingComplete):
            m.runBatch(train=train, split="runtime", batchSize=1, optimizer=optimizer,
                       batch_override=(inp, None))
        staged = disc._staged_prediction
        assert staged is not None
        pred, conf = staged
        assert pred is not None and pred.dtype == dtype
        assert conf is not None and conf.dtype == dtype
        assert pred.shape == (disc.sentence_dim,)
        assert torch.isfinite(pred).all()
        m._end_step()
    finally:
        _util.MODEL_AMP = saved_mode


def test_full_sentence_loop_skips_legacy_discourse_tuple_staging():
    """The W-loop uses the separate tensor seed, never the ARMA tuple.

    This prevents the cold ``(None, None)`` tuple from becoming a Dynamo
    guard that recompiles the full sentence graph when the discourse ring
    first becomes warm.
    """
    from Models import BaseModel

    class _Discourse:
        def stage_prediction(self):
            raise AssertionError("full sentence loop must not stage legacy ARMA")

    model = BaseModel()
    assert not model._stage_legacy_discourse_prediction(
        _Discourse(), fullgraph_word_loop=True)
