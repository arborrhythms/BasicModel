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


def _build_gate_model(tmp_path):
    from data import TheData
    from Models import BaseModel
    from util import init_config, init_device
    init_device("cpu")
    cfg = str(_root / "data" / "MM_ladder.xml")
    init_config(path=cfg, defaults_path=str(_root / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_root / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    # Each fixture presents the first loaded sentence. There cannot be more
    # nonempty lexical units than input bytes; construct that full bound.
    first = TheData.train_input[0]
    width = len(first.encode('utf8')) if isinstance(first, str) else int(first.ne(0).sum())
    width = max(1, width)
    text = Path(cfg).read_text()
    for tag in ('serialWordCapacity', 'serialWordBuckets'):
        old = f'<{tag}>8</{tag}>'
        assert old in text
        text = text.replace(old, f'<{tag}>{width}</{tag}>', 1)
    path = tmp_path/'gate-model.xml'
    path.write_text(text)
    m, _ = BaseModel.from_config(str(path), data=TheData)
    return m.to("cpu")


@pytest.mark.slow
def test_false_gate_contribution_is_zero(tmp_path, monkeypatch):
    """An inactive native word contributes zero while preserving the carry."""
    from test_per_word_capture_gate import _stage_for_per_word
    m = _build_gate_model(tmp_path)
    inp, _ = m.inputSpace.getTrainData()
    step, carry, staged = _stage_for_per_word(m, monkeypatch, samples=list(inp[:1]))
    word = staged['words'][:, :1]
    gate_false = torch.zeros_like(staged['active'][:, :1])
    index = word.new_zeros((), dtype=torch.long)
    payload, _ = staged['stage_cs_sub'](word, gate_false, index,
        torch.ones((), dtype=torch.bool), carry[1])
    contribution = payload[0]
    assert torch.all(contribution == 0), (
        "False-gate contribution must be zero; got non-zero "
        f"max abs {contribution.abs().max().item()}")


@pytest.mark.slow
def test_false_gate_preserves_stm_buffer(tmp_path, monkeypatch):
    from test_per_word_capture_gate import _stage_for_per_word
    m = _build_gate_model(tmp_path)
    inp, _ = m.inputSpace.getTrainData()
    step, carry, staged = _stage_for_per_word(m, monkeypatch, samples=list(inp[:1]))
    stm = m.conceptualSpace.stm
    stm.set_state(carry[0])
    before = tuple(value.clone() for value in stm.state)
    buf_before = stm._buffer.clone()
    depth_before = stm._depth.clone()
    word = staged['words'][:, :1]
    result = step(word, torch.zeros_like(staged['active'][:, :1]),
                  word.new_zeros((), dtype=torch.long), carry)
    stm.set_state(result[0])
    assert torch.equal(stm._buffer, buf_before), "STM buffer changed under gate=False"
    assert torch.equal(stm._depth, depth_before), "STM depth changed under gate=False"
    assert all(torch.equal(a, b) for a, b in zip(stm.state, before))


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
