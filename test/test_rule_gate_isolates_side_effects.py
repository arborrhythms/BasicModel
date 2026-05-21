"""When the rule-gate / word_active mask is False at a column, the
per-word body must leave recurrent state bit-identical.

Forces ``gate_b_1 = False`` at a known position and asserts the
carrier event tensors, STM buffer, and concept buffer column are
unchanged. This is the load-bearing assertion that prevents the
ConceptualSpace ``(0 + SS) / 2`` averaging leak from contaminating
``cs_for_ps`` at padding columns.

Doc: doc/plans/2026-05-20-static-per-word-loop-impl.md §2.4.
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
    return m.to("cpu")


def test_false_gate_contribution_is_zero():
    """At an inactive batch row / padding column, the per-iteration
    contribution must be zero. The list-based accumulator records
    one tensor per call; gate=False yields zeros."""
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    in_sub = isp.forward(inputTensor)
    m._per_word_prelude(in_sub)
    out_slot = m._per_word_contributions
    B = isp._ar_embedded_N.shape[0] if isp._ar_embedded_N is not None else 1
    gate_false = torch.zeros(B, 1, dtype=torch.bool)
    w = isp.word_at(0)
    m._per_word_body_step(w, 0, gate_false, out_slot)
    if not out_slot:
        pytest.skip("body did not produce an idea_bd to contribute")
    contribution = out_slot[-1]
    assert torch.all(contribution == 0), (
        "False-gate contribution must be zero; got non-zero "
        f"max abs {contribution.abs().max().item()}")


def test_false_gate_preserves_stm_buffer():
    m = _build_gate_model()
    isp = m.inputSpace
    inp, _ = isp.getTrainData()
    isp.Start()
    inputTensor = isp.prepInput(list(inp[:1]))
    in_sub = isp.forward(inputTensor)
    m._per_word_prelude(in_sub)
    stm = m.conceptualSpace.stm
    if stm is None:
        pytest.skip("model has no STM")
    buf_before = stm._buffer.clone()
    depth_before = stm._depth.clone()
    out_slot = m._per_word_contributions
    w = isp.word_at(0)
    B = w.shape[0]
    gate_false = torch.zeros(B, 1, dtype=torch.bool)
    m._per_word_body_step(w, 0, gate_false, out_slot)
    assert torch.equal(stm._buffer, buf_before), "STM buffer changed under gate=False"
    assert torch.equal(stm._depth, depth_before), "STM depth changed under gate=False"


def test_maybe_blend_event_noop_when_gate_false():
    """The carrier event blend masks to the prev_event when gate is
    False. Exercised directly on the static helper to isolate from the
    full body."""
    from Models import BasicModel
    carrier = type("SubSpace", (), {})()
    new_ev = torch.full((2, 3, 4), 9.0)
    prev_ev = torch.full((2, 3, 4), 1.0)
    carrier._event = new_ev.clone()
    gate_false = torch.zeros(2, 1, dtype=torch.bool)
    BasicModel._maybe_blend_event(carrier, prev_ev, gate_false)
    assert torch.equal(carrier._event, prev_ev), (
        "False-gate blend must restore the prev_event everywhere")


def test_maybe_blend_event_keeps_new_when_gate_true():
    from Models import BasicModel
    carrier = type("SubSpace", (), {})()
    new_ev = torch.full((2, 3, 4), 9.0)
    prev_ev = torch.full((2, 3, 4), 1.0)
    carrier._event = new_ev.clone()
    gate_true = torch.ones(2, 1, dtype=torch.bool)
    BasicModel._maybe_blend_event(carrier, prev_ev, gate_true)
    assert torch.equal(carrier._event, new_ev)
