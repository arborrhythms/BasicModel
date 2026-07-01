"""Padding columns of the static per-word loop are no-ops.

STM depth must increment exactly by the active-prefix length, not by N.
The concept buffer at active positions must match the active-prefix-
only run; positions past the active prefix must be zero.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §2.4-2.6.
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


@pytest.mark.xfail(
    reason="MM_20M_legacy.xml: percept_dim+nWhere+nWhen=12 != concept_dim+nWhere+"
           "nWhen=1028 since Stage 1.C retired sigma_percept; the signal "
           "router replacement (Stage 3) is not yet wired.",
    strict=False,
)
def test_stm_depth_tracks_valid_len_not_N():
    """After one forward pass, STM depth (via host mirror) equals the
    real-positions count, not N."""
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    in_sub = isp.forward(inputTensor)
    L = int(isp._valid_len_host)
    N = int(isp.outputShape[0])
    if not (0 < L < N):
        pytest.skip(f"input lacks padding columns (L={L}, N={N}); "
                    "test needs an active prefix shorter than N")
    stm = m.conceptualSpace.stm
    if stm is None:
        pytest.skip("model has no STM")
    depth_before = int(stm._max_depth_host)
    m._forward_body_per_word(in_sub)
    depth_after = int(stm._max_depth_host)
    assert depth_after - depth_before == L, (
        f"STM mirror advanced by {depth_after - depth_before}, "
        f"expected {L} (real-positions count)")


@pytest.mark.xfail(
    reason="MM_20M_legacy.xml percept_dim / concept_dim mismatch.",
    strict=False,
)
def test_concept_buf_zero_past_active_prefix():
    """Per-iteration contributions are zero past the active prefix
    (the gate-masked ``torch.where`` writes zeros for inactive rows).
    After ``_forward_body_per_word``, the stacked event on the
    ConceptualSpace subspace shows that pattern."""
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    in_sub = isp.forward(inputTensor)
    L = int(isp._valid_len_host)
    N = int(isp.outputShape[0])
    if not (0 < L < N):
        pytest.skip(f"input lacks padding columns (L={L}, N={N})")
    m._forward_body_per_word(in_sub)
    cs_event = m.conceptualSpace.subspace.materialize()
    if cs_event is None or cs_event.dim() != 3:
        pytest.skip("CS event not materialized")
    tail = cs_event[:, L:, :]
    assert torch.all(tail == 0), (
        f"CS event[{L}:] should be zero (padded with S→S "
        f"no-ops); max abs value = {tail.abs().max().item()}")
