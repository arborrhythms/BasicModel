"""Contracts for queryable spacetime, absolute dataset ``where``, and
iterative LTM parity."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from data import Data
from Language import MLPTransformChooser
from Layers import InterSentenceLayer, WhatInteractionMemory
from Models import BasicModel
from Spaces import WhereEncoding
from What import (LTMSlot, What, WhatAnswer, WhatRelation,
                  WhatSlotOperation)


def _address(split, row, document, sentence):
    return {
        "split": split,
        "row": row,
        "document": document,
        "sentence": sentence,
        "char_start": 0,
        "char_end": 1,
        "external_id": None,
        "source_time": None,
    }


def _data():
    data = Data()
    data.train_input = ["zero", "one", "two", "three"]
    data.train_output = ["Z", "O", "T", "H"]
    data.has_supervised_outputs = True
    data.source_addresses["train"] = [
        _address("train", 0, 7, 0),
        _address("train", 1, 7, 1),
        _address("train", 2, 7, 2),
        _address("train", 3, 8, 0),
    ]
    return data


def test_question_validation_and_target_coordinate():
    assert What.present(3).target_where == 3
    assert What.past(3, -2).target_where == 1
    assert What.future(3, 4).target_where == 7
    with pytest.raises(ValueError):
        What(where=0, relation=WhatRelation.PAST, offset=1)
    with pytest.raises(ValueError):
        What(where=0, relation=WhatRelation.PRESENT, offset=1)
    with pytest.raises(ValueError):
        What(where=-1)


def test_context_values_carry_absolute_where():
    # The dataset exists all at once: a presentation index is an absolute
    # position (a .where), and the model is allowed to see it.
    assert What.past(8, -1).context_values()[-2:] == (8.0, 7.0)
    assert What.present(0).context_values() != What.present(1).context_values()
    assert len(What.present(0).context_values()) == 9


def test_data_what_present_temporal_supervised_and_inference():
    data = _data()

    present = data.what(What.present(1))
    assert present.available and present.what == "one"
    assert present.source_where == 1 and present.provenance == "data"
    assert data.what(What.past(2)).what == "one"
    assert data.what(What.future(1)).what == "two"
    assert data.what(What.supervised(1)).what == "O"

    unavailable = data.what(What.inference(0, split="train"))
    assert not unavailable.available and unavailable.what is None


def test_temporal_lookup_never_crosses_document_or_split_boundary():
    data = _data()
    assert not data.what(What.past(0)).available
    assert not data.what(What.future(2)).available
    assert not data.what(What.past(3)).available
    assert not data.what(What.future(3)).available
    assert not data.what(What.future(0, split="test")).available


def test_changing_only_temporal_question_changes_desired_answer():
    data = _data()
    questions = (
        What.past(1), What.present(1), What.future(1),
    )
    assert [data.what(question).what for question in questions] == [
        "zero", "one", "two"]
    assert len({question.context_values() for question in questions}) == 3


def test_attached_model_output_keeps_index_and_supplied_target_separate():
    data = _data()
    question = What.inference(1, split="train", prompt="answer this")
    answer = WhatAnswer(question=question, what="generated", provenance="model")

    before_inputs = list(data.train_input)
    before_targets = list(data.train_output)
    presentation = data.attach_output(question, answer)

    assert presentation.where == 1
    assert presentation.input == "answer this"
    assert presentation.output.what == "generated"
    assert data.train_input == before_inputs
    assert data.train_output == before_targets
    assert data.what(What.supervised(1)).what == "O"
    with pytest.raises(ValueError):
        data.attach_output(
            question,
            WhatAnswer(question=question, what="source", provenance="data"))


def _memory(capacity=16, standalone=False):
    # The interaction slots are owned by ``WhatInteractionMemory``; the
    # discourse layer composes one and keeps the same API, so every slot
    # contract below holds for both owners.
    if standalone:
        return WhatInteractionMemory(batch=1, capacity=capacity)
    return InterSentenceLayer(
        n_symbols=2, max_depth=3, n_dim=4, p=1, q=0,
        concept_dim=None, batch=1, ltm_capacity=capacity)


_OWNERS = pytest.mark.parametrize("standalone", [False, True],
                                  ids=["discourse", "standalone"])


@_OWNERS
def test_ltm_slot_stack_parity_lifo_and_immutable_openings(standalone):
    memory = _memory(standalone=standalone)
    root = memory.append_what_slot(LTMSlot(input=torch.tensor([1.0])))
    assert memory.what_open_depth() == 1
    assert not memory.what_at_parity()

    complete = memory.append_what_slot(
        LTMSlot(input=torch.tensor([2.0]), output=torch.tensor([3.0]),
                closure_pressure=0.25))
    assert complete.operation is WhatSlotOperation.COMPLETE
    assert memory.what_open_depth() == 1

    child = memory.append_what_slot(
        LTMSlot(input=torch.tensor([4.0]), closure_pressure=0.5))
    memory.append_what_slot(
        LTMSlot(output=torch.tensor([5.0]), closure_pressure=0.75))
    assert memory.open_what_slots() == [root]
    memory.append_what_slot(
        LTMSlot(output=torch.tensor([6.0]), closure_pressure=1.0))
    assert memory.what_at_parity()
    # Closures are later records; neither opening was filled in place.
    assert root.output is None and child.output is None
    assert [slot.operation for slot in memory.get_what_slots()] == [
        WhatSlotOperation.OPEN, WhatSlotOperation.COMPLETE,
        WhatSlotOperation.OPEN, WhatSlotOperation.CLOSE,
        WhatSlotOperation.CLOSE]


@_OWNERS
def test_ltm_rejects_empty_or_unmatched_close_and_detaches_actual_response(standalone):
    memory = _memory(standalone=standalone)
    with pytest.raises(ValueError):
        LTMSlot()
    with pytest.raises(ValueError):
        memory.append_what_slot(LTMSlot(output=torch.ones(1)))

    actual = torch.tensor([2.0], requires_grad=True)
    memory.append_what_slot(
        LTMSlot(input=torch.tensor([1.0]), output=actual))
    stored = memory.get_what_slots()[0]
    assert stored.output is not actual
    assert stored.output.grad_fn is None and not stored.output.requires_grad
    assert torch.equal(stored.output, actual)


@_OWNERS
def test_ltm_pressure_is_monotonic_until_parity_and_reset_clears_slots(standalone):
    memory = _memory(standalone=standalone)
    memory.append_what_slot(LTMSlot(input="root", closure_pressure=0.1))
    memory.append_what_slot(LTMSlot(input="child", closure_pressure=0.5))
    with pytest.raises(ValueError):
        memory.append_what_slot(
            LTMSlot(input="bad", closure_pressure=0.4))
    memory.append_what_slot(LTMSlot(output="child answer", closure_pressure=0.7))
    memory.append_what_slot(LTMSlot(output="root answer", closure_pressure=0.9))
    assert memory.what_context()["closure_pressure"] == 0.0
    memory.Reset()
    assert memory.get_what_slots() == []


@_OWNERS
def test_ltm_capacity_evicts_only_balanced_prefixes(standalone):
    memory = _memory(capacity=2, standalone=standalone)
    memory.append_what_slot(LTMSlot(input="q0", output="a0"))
    memory.append_what_slot(LTMSlot(input="q1", output="a1"))
    memory.append_what_slot(LTMSlot(input="q2", output="a2"))
    assert [slot.input for slot in memory.get_what_slots()] == ["q1", "q2"]

    blocked = _memory(capacity=2, standalone=standalone)
    blocked.append_what_slot(LTMSlot(input="q0"))
    blocked.append_what_slot(LTMSlot(input="q1", closure_pressure=0.5))
    with pytest.raises(OverflowError):
        blocked.append_what_slot(LTMSlot(input="q2", closure_pressure=1.0))
    assert blocked.what_open_depth() == 2


class _TinyWhatModel(BasicModel):
    def __init__(self, memory=None, actions=()):
        nn.Module.__init__(self)
        self.symbolSpace = SimpleNamespace(
            discourse=memory, languageLayer=SimpleNamespace())
        self._actions = list(actions)
        self.context_sizes = []
        self.calls = 0
        self._concept = None

    def forward(self, value):
        self.calls += 1
        self._concept = value + 1.0
        return value, value + 2.0, value + 3.0, None

    def _reconstruction_seed(self):
        return self._concept

    def choose_what_slot(self, question, *, conceptual_input,
                         conceptual_output, produced, iteration=0,
                         closure_pressure=0.0):
        memory = self._what_memory()
        self.context_sizes.append(
            len(memory.get_what_slots()) if memory is not None else 0)
        action = self._actions.pop(0) if self._actions else "complete"
        kwargs = dict(
            question=question, iteration=iteration,
            closure_pressure=closure_pressure)
        if action == "open":
            return LTMSlot(input=conceptual_input, **kwargs)
        if action == "close":
            return LTMSlot(output=conceptual_output, **kwargs)
        return LTMSlot(
            input=conceptual_input, output=conceptual_output, **kwargs)


def test_model_what_present_wraps_established_forward_byte_identically():
    value = torch.tensor([[1.0, 2.0]])
    model = _TinyWhatModel()
    clean = model.forward(value)
    calls = model.calls
    answer = model.what(What.present(0), execution=clean)
    assert model.calls == calls
    assert answer.what is clean[2]
    assert torch.equal(answer.what, value + 3.0)
    assert answer.question.context_values() in tuple(
        item["temporal"] for item in answer.grammar_trace)


def test_model_what_ltm_records_actual_head_response_not_input_state():
    memory = _memory()
    model = _TinyWhatModel(memory)
    value = torch.tensor([[4.0]])
    answer = model.what(What.supervised(0), input_data=value)
    stored = memory.get_what_slots()[0]
    assert torch.equal(answer.what, value + 3.0)
    assert torch.equal(stored.output, answer.what)
    assert not torch.equal(stored.output, value + 2.0)


def test_thinking_uses_completed_subquestion_context_and_restores_parity():
    memory = _memory()
    model = _TinyWhatModel(memory, actions=("open", "complete", "close"))
    result = model.think(
        What.supervised(0, prompt="what is your name?"),
        torch.tensor([[1.0]]), max_iterations=5)

    assert result.answer.available
    assert result.forced_closures == 0
    assert model.context_sizes == [0, 1, 2]
    assert memory.what_at_parity()
    assert [slot.operation for slot in result.slots] == [
        WhatSlotOperation.OPEN, WhatSlotOperation.COMPLETE,
        WhatSlotOperation.CLOSE]
    assert all(a <= b for a, b in zip(
        result.closure_pressures, result.closure_pressures[1:]))


def test_thinking_limit_forces_every_nested_question_closed():
    memory = _memory(capacity=16)
    model = _TinyWhatModel(memory, actions=("open", "open", "open"))
    result = model.think(
        What.inference(0, prompt="hard question"),
        torch.tensor([[2.0]]), max_iterations=3)

    assert result.iterations == 3
    assert result.forced_closures == 3
    assert memory.what_at_parity()
    assert all(slot.forced for slot in result.slots[-3:])
    assert all(slot.operation is WhatSlotOperation.CLOSE
               for slot in result.slots[-3:])
    assert result.answer.what is not None


def test_mlp_grammar_chooser_consumes_target_free_what_context():
    chooser = MLPTransformChooser(d_model=2, n_copy=1, n_op=1)
    x = torch.tensor([[[0.25, -0.5]]])
    applied = torch.tensor([[[[0.5, 0.25]]]])
    copy_anchor = torch.zeros(1, 2)
    apply_anchor = torch.zeros(1, 2)
    context = torch.zeros(1, MLPTransformChooser.WHAT_CONTEXT_DIM)
    context[0, 0] = 1.0                       # relation: present
    context[0, 17] = 1.0                      # one LTM input
    context[0, 18] = 1.0                      # one LTM output

    baseline = chooser.score_unary(
        x, applied, copy_anchor, apply_anchor)
    conditioned = chooser.score_unary(
        x, applied, copy_anchor, apply_anchor, what_ctx=context)
    assert torch.equal(baseline[0], conditioned[0])
    assert torch.equal(baseline[1], conditioned[1])

    with torch.no_grad():
        chooser.what_projection.weight[1, 0] = 2.0
    shifted = chooser.score_unary(
        x, applied, copy_anchor, apply_anchor, what_ctx=context)
    assert torch.equal(shifted[0], baseline[0])
    assert torch.allclose(shifted[1], baseline[1] + 2.0)


def test_what_grammar_context_uses_model_where_ladder():
    data = _data()
    model = _TinyWhatModel()
    model.inputSpace = SimpleNamespace(data=data)
    question = What.past(2)
    rows, detailed = model._what_grammar_context(
        (question,), device=torch.device("cpu"), dtype=torch.float32)
    row = rows[0]
    assert row.shape[-1] == MLPTransformChooser.WHAT_CONTEXT_DIM
    extent = data.what_extent("train")
    assert extent == 4
    assert torch.allclose(row[7:9], torch.tensor([2 / 4, 1 / 4]))
    ladder = WhereEncoding(extent, 4, 0)
    assert torch.allclose(row[9:13], ladder.encode(2).float())
    assert torch.allclose(row[13:17], ladder.encode(1).float())
    assert detailed[0]["temporal"] == question.context_values()

    other, _ = model._what_grammar_context(
        (What.present(2),), device=torch.device("cpu"), dtype=torch.float32)
    # Same relation and offset as What.present(2) would give at row 1 ...
    shifted, _ = model._what_grammar_context(
        (What.present(1),), device=torch.device("cpu"), dtype=torch.float32)
    # ... but the absolute position separates otherwise identical questions.
    assert not torch.equal(other[0], shifted[0])


def test_what_projection_checkpoint_widens_from_19():
    live = torch.zeros(2, MLPTransformChooser.WHAT_CONTEXT_DIM)
    saved = torch.arange(38, dtype=torch.float32).reshape(2, 19)
    state = {"symbolSpace.chooser.what_projection.weight": saved.clone(),
             "other.weight": torch.ones(3)}
    model_state = {"symbolSpace.chooser.what_projection.weight": live,
                   "other.weight": torch.ones(3)}
    assert BasicModel._widen_what_projection_checkpoint_state(
        state, model_state) == 1
    widened = state["symbolSpace.chooser.what_projection.weight"]
    assert widened.shape == live.shape
    assert torch.equal(widened[:, :19], saved)
    assert torch.equal(widened[:, 19:], torch.zeros(2, live.shape[1] - 19))
    # A mismatch on the row axis is not our migration; leave it for the audit.
    state["symbolSpace.chooser.what_projection.weight"] = torch.zeros(3, 19)
    assert BasicModel._widen_what_projection_checkpoint_state(
        state, model_state) == 0


def test_questions_for_batch_folds_driver_counters_into_the_split():
    # Drivers whose batch counter is not a row (negative warm-up indices,
    # cyclic cursors) still get real presentation positions; -1 pads in
    # source_rows mean "no source row" and fall back to the counter.
    model = _TinyWhatModel()
    model.inputSpace = SimpleNamespace(data=_data())          # extent 4
    rows = [q.where for q in model._questions_for_batch(
        split="train", batch_size=2, sentence_index=-2)]
    assert rows == [2, 3]
    rows = [q.where for q in model._questions_for_batch(
        split="train", batch_size=2, sentence_index=5)]
    assert rows == [1, 2]
    rows = [q.where for q in model._questions_for_batch(
        split="train", batch_size=2, sentence_index=0,
        source_rows=[[-1], [7]])]
    assert rows == [0, 3]
