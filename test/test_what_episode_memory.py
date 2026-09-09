"""Phase 2 of the mathematical thinking plan: the standalone
``WhatInteractionMemory`` (spec 7.1), the episode credit boundary (spec
8.2), per-row episodes, and the byte-identical default (spec 11.1)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
_DATA = _ROOT / "data"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Layers import InterSentenceLayer, WhatInteractionMemory  # noqa: E402
from What import LTMSlot, What, WhatSlotOperation  # noqa: E402


# -- the memory alone ---------------------------------------------------------

def test_slot_mode_detaches_at_append():
    memory = WhatInteractionMemory(batch=1, capacity=8, detach_mode="slot")
    memory.begin_what_episode(0)
    x = torch.ones(3, requires_grad=True) * 2.0
    stored = memory.append_what_slot(LTMSlot(input=x, output=x + 1), b=0)
    assert not stored.input.requires_grad and not stored.output.requires_grad
    assert memory.end_what_episode(0) == 0


def test_episode_mode_keeps_values_live_until_end():
    memory = WhatInteractionMemory(batch=2, capacity=8, detach_mode="episode")
    x = torch.ones(3, requires_grad=True) * 2.0
    # Outside an episode, episode mode still detaches (durable memory).
    outside = memory.append_what_slot(LTMSlot(input=x, output=x), b=1)
    assert not outside.input.requires_grad
    memory.begin_what_episode(0)
    assert memory.in_episode(0) and not memory.in_episode(1)
    opened = memory.append_what_slot(LTMSlot(input=x), b=0)
    closed = memory.append_what_slot(LTMSlot(output=x * 3, closure_pressure=0.5), b=0)
    assert opened.input.requires_grad and closed.output.requires_grad
    # A loss formed from the stored slot reaches the leaf through memory.
    loss = memory.get_what_slots(b=0)[1].output.sum()
    (g,) = torch.autograd.grad(loss, x, retain_graph=True)
    assert torch.all(g == 3.0)
    assert memory.what_at_parity(b=0)
    assert memory.end_what_episode(0) == 2
    after = memory.get_what_slots(b=0)
    assert [s.operation for s in after] == [WhatSlotOperation.OPEN, WhatSlotOperation.CLOSE]
    assert not after[0].input.requires_grad and not after[1].output.requires_grad
    assert torch.equal(after[1].output, (x * 3).detach())
    assert not memory.in_episode(0)


def test_context_exposes_open_question_and_latest_output():
    memory = WhatInteractionMemory(batch=1, capacity=8)
    assert memory.what_context()["open_question"] is None
    memory.append_what_slot(LTMSlot(input="root"))
    memory.append_what_slot(LTMSlot(input="sub", output="sub-answer", closure_pressure=0.2))
    ctx = memory.what_context(question=What.supervised(0))
    assert ctx["open_question"] == "root" and ctx["latest_output"] == "sub-answer"
    assert ctx["open_depth"] == 1 and ctx["closure_pressure"] == 0.2
    memory.append_what_slot(LTMSlot(output="root-answer", closure_pressure=0.9))
    ctx = memory.what_context()
    assert ctx["open_question"] is None and ctx["latest_output"] == "root-answer"
    assert ctx["parity"] and ctx["closure_pressure"] == 0.0


def test_batch_resize_and_resets_are_per_row():
    memory = WhatInteractionMemory(batch=2, capacity=8)
    memory.append_what_slot(LTMSlot(input="q"), b=0)
    memory.append_what_slot(LTMSlot(input="q", output="a"), b=1)
    memory.Reset(batch=0, hard=False)                 # soft: keeps the dialogue
    assert memory.what_open_depth(b=0) == 1
    memory.Reset(batch=0, hard=True)
    assert memory.what_at_parity(b=0) and len(memory.get_what_slots(b=1)) == 1
    memory.ensure_batch(2)                            # same size: untouched
    assert len(memory.get_what_slots(b=1)) == 1
    memory.ensure_batch(3)                            # new shape: fresh rows
    assert memory.batch == 3 and all(memory.what_at_parity(b=b) for b in range(3))


def test_discourse_layer_composes_the_same_memory():
    layer = InterSentenceLayer(n_symbols=2, max_depth=3, n_dim=4, p=1, q=0,
                               concept_dim=None, batch=1, ltm_capacity=4)
    assert isinstance(layer.what_memory, WhatInteractionMemory)
    layer.detach_mode = "episode"
    layer.begin_what_episode(0)
    x = torch.ones(2, requires_grad=True)
    stored = layer.append_what_slot(LTMSlot(input=x, output=x))
    assert stored.input.requires_grad
    assert layer.end_what_episode(0) == 1
    assert not layer.get_what_slots()[0].input.requires_grad
    layer.ensure_batch(3)
    assert layer.what_memory.batch == 3 and layer.get_what_slots(b=2) == []
    layer.append_what_slot(LTMSlot(input="q"), b=2)
    layer.Reset(batch=2, hard=True)
    assert layer.what_at_parity(b=2)


# -- through the model ---------------------------------------------------------

def _build(config_path):
    import Language
    from util import init_config
    from data import TheData
    import Models

    init_config(path=str(config_path), defaults_path=str(_DATA / "model.xml"))
    Language.TheGrammar._configured = False
    TheData.load("xor")
    torch.manual_seed(0)
    m, _ = Models.BaseModel.from_config(str(config_path), data=TheData)
    return m.to("cpu")


def _batch(m, rows=2):
    loader = m.inputSpace.data.data_loader(split="train", num_streams=rows)
    inp_items, out_items = next(iter(loader))
    return (m.inputSpace.prepInput(inp_items),
            m.outputSpace.prepOutput(out_items))


def _config(tmp_path_factory, name, extra_arch):
    src = (_DATA / "MM_xor.xml").read_text()
    assert src.count("<architecture>") == 1
    patched = src.replace("<architecture>", "<architecture>\n" + extra_arch, 1)
    path = tmp_path_factory.mktemp("cfg") / name
    path.write_text(patched)
    return path


@pytest.fixture(scope="module")
def memory_config(tmp_path_factory):
    return _config(tmp_path_factory, "MM_xor_what_memory.xml",
                   "    <answerSynthesis>true</answerSynthesis>\n"
                   "    <whatThinkingMemory>true</whatThinkingMemory>\n"
                   "    <whatThinkingDetach>episode</whatThinkingDetach>")


@pytest.fixture(scope="module")
def plain_config(tmp_path_factory):
    return _config(tmp_path_factory, "MM_xor_plain.xml",
                   "    <answerSynthesis>true</answerSynthesis>")


def test_default_has_no_memory_and_think_is_single_step(plain_config):
    m = _build(plain_config)
    assert m.symbolSpace.discourse is None
    assert getattr(m.symbolSpace, "what_memory", None) is None
    assert m._what_memory() is None
    assert m.what_thinking_detach == "slot"
    x, _ = _batch(m, rows=1)
    with torch.no_grad():
        result = m.think(What.supervised(0), x, max_iterations=3)
    assert result.answer.available and result.iterations == 1
    assert result.forced_closures == 0


def test_standalone_memory_serves_think_without_sentence_prediction(memory_config):
    m = _build(memory_config)
    assert m.symbolSpace.discourse is None
    memory = m._what_memory()
    assert isinstance(memory, WhatInteractionMemory)
    assert memory.detach_mode == "episode"          # applied from the config
    x, _ = _batch(m, rows=2)
    with torch.no_grad():
        m.forward(x)                                 # sizes the memory to the batch
    assert memory.batch == 2
    with torch.no_grad():
        answers = m.what((What.supervised(0), What.supervised(1)), x)
    assert len(answers) == 2 and all(a.available for a in answers)
    # One complete slot per row, recorded in the standalone memory.
    for b in range(2):
        slots = memory.get_what_slots(b=b)
        assert [s.operation for s in slots] == [WhatSlotOperation.COMPLETE]
        assert memory.what_at_parity(b=b)
    # Hard row reset clears the row's dialogue only.
    m.symbolSpace.Reset(batch=0, hard=True)
    assert memory.get_what_slots(b=0) == [] and len(memory.get_what_slots(b=1)) == 1
    with torch.no_grad():
        result = m.think(What.supervised(0), x[:1], max_iterations=2)
    assert result.answer.available and memory.what_at_parity(b=0)


def test_reset_clears_the_standalone_memory_at_epoch_start(memory_config):
    m = _build(memory_config)
    memory = m._what_memory()
    x, _ = _batch(m, rows=1)
    with torch.no_grad():
        m.what(What.supervised(0), x)
    assert len(memory.get_what_slots(b=0)) == 1
    opt = m.getOptimizer(lr=1e-3)
    m.runEpoch(optimizer=opt, batchSize=2, split="train", max_batches=1)
    # runEpoch starts from a cleared memory and records the epoch's own slots
    assert all(s.operation is WhatSlotOperation.COMPLETE
               for b in range(memory.batch) for s in memory.get_what_slots(b=b))
