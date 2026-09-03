"""Teacher reconstruction and objective source-address contracts."""

from dataclasses import replace

import pytest
import torch

from data import Data, SentenceStreamDataset
from Layers import Error
from Teacher import Teacher


def _text_data():
    data = Data()
    data.source_manifest = {
        "dataset": "inline",
        "corpus": "teacher-test",
        "snapshot": "v1",
    }
    data.processLM({
        "train": {
            "text": ["alpha beta", "gamma delta", "epsilon"],
            "label": [],
        },
        "validation": {"text": ["validation"], "label": []},
        "test": {"text": ["test"], "label": []},
    })
    return data


def test_objective_where_when_round_trip_directly_to_what():
    data = _text_data()
    teacher = Teacher(
        data, enabled=True, context_name="unit corpus", errors=Error())

    where = teacher.where("train", 1)
    when = teacher.when(where)

    assert teacher.What(where, when) == "gamma delta"
    assert where.document == 1
    assert where.sentence == 0
    assert (where.char_start, where.char_end) == (0, len("gamma delta"))


def test_what_rejects_forged_location_or_snapshot():
    teacher = Teacher(
        _text_data(), enabled=True, context_name="unit corpus", errors=Error())
    where = teacher.where("train", 0)
    when = teacher.when(where)

    with pytest.raises(KeyError, match="canonical source address"):
        teacher.What(replace(where, char_end=where.char_end + 1), when)
    with pytest.raises(KeyError, match="different corpus snapshot"):
        teacher.What(
            where,
            replace(when, snapshot_identity=when.snapshot_identity + 1),
        )


def test_snapshot_identity_changes_when_manifest_changes():
    data_v1 = _text_data()
    data_v2 = _text_data()
    data_v2.source_manifest = dict(data_v2.source_manifest, snapshot="v2")

    first = Teacher(
        data_v1, context_name="unit corpus", errors=Error())
    again = Teacher(
        data_v1, context_name="unit corpus", errors=Error())
    changed = Teacher(
        data_v2, context_name="unit corpus", errors=Error())

    assert first.context.identity == again.context.identity
    assert first.snapshot_identity == again.snapshot_identity
    assert changed.snapshot_identity != first.snapshot_identity


def test_packed_lesson_keeps_exact_rows_and_clean_teacher_targets():
    teacher = Teacher(
        _text_data(), enabled=True, context_name="unit corpus", errors=Error())
    teacher.stage_batch_sources("train", [[0, 1], [2]])
    lesson = teacher.begin_batch(
        split="train", batch_size=2, training=True)

    assert lesson.clean_what == (
        ("alpha beta", "gamma delta"),
        ("epsilon",),
    )
    assert [
        [where.row for where in row]
        for row in lesson.objective_where
    ] == [[0, 1], [2]]


def test_binding_never_reads_or_changes_subjective_where_when():
    teacher = Teacher(
        _text_data(), enabled=True, context_name="unit corpus", errors=Error())
    teacher.stage_batch_sources("train", [[0, 1], [2]])
    teacher.begin_batch(split="train", batch_size=2, training=True)

    class StudentSubspace:
        valid_mask = torch.tensor([
            [True, True, False],
            [True, False, False],
        ])
        subjective_where = object()
        subjective_when = object()

        def materialize(self):
            raise AssertionError(
                "Teacher must not inspect the subjective event carrier")

    subspace = StudentSubspace()
    old_where = subspace.subjective_where
    old_when = subspace.subjective_when
    lesson = teacher.bind_input(subspace)

    assert subspace.subjective_where is old_where
    assert subspace.subjective_when is old_when
    assert lesson.objective_where_code.shape == (2, 2, 8)
    assert lesson.objective_when_code.shape == (2, 2, 2)
    assert lesson.objective_mask.tolist() == [[True, True], [True, False]]
    assert lesson.objective_where_code.device.type == "cpu"


def test_cursor_reports_source_indices_for_trial_and_packed_ticks():
    values = ["a", "b", "c", "d", "e", "f"]

    trial = SentenceStreamDataset(values, num_streams=2)
    trial.next_tick()
    assert trial.last_source_indices == [0, 3]
    trial.next_tick()
    assert trial.last_source_indices == [1, 4]

    packed = SentenceStreamDataset(values, num_streams=2)
    packed.next_packed_tick(2, lambda _: 1)
    assert packed.last_source_indices == [[0, 1], [3, 4]]
    packed.next_packed_tick(2, lambda _: 1)
    assert packed.last_source_indices == [[2], [5]]


def test_runtime_context_restores_data_and_answers_runtime_query():
    data = _text_data()
    teacher = Teacher(
        data, enabled=True, context_name="unit corpus", errors=Error())
    original_input = data.train_input
    original_context = teacher.context

    with teacher.runtime_batch(["temporary text"], context="conversation"):
        lesson = teacher.begin_batch(
            split="runtime", batch_size=1, training=False)
        where = lesson.objective_where[0][0]
        when = lesson.objective_when[0][0]
        assert teacher.What(where, when) == "temporary text"
        assert teacher.context.name == "conversation"

    assert data.train_input is original_input
    assert teacher.context is original_context
    assert "runtime" not in data.source_addresses
