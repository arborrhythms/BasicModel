"""Reviewed follow-up: native context precedes the serial reading."""
from functools import wraps
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('packed', [False, True])
def test_interleave_reads_native_context_before_any_serial_word(tmp_path, monkeypatch, packed):
    from ModeSchedule import ModeSchedule
    from What import What
    from test_packed_reconstruction_parity import build_model
    # Each sentence has five native units including spaces; the packed pair
    # also includes its joining space. Reserve room for those actual units.
    model = build_model(tmp_path, word_capacity=16 if packed else 8)
    model.mode_schedule = ModeSchedule('interleave:2')
    texts = ['the wug sat', 'the wug flew']
    calls = []
    execute = model._run_batch_once
    @wraps(execute)
    def record(*args, **kwargs):
        rows = model.inputSpace._packed_sentence_rows
        visible = (tuple(text for row in rows for text in row) if rows else
                   tuple(model.inputSpace._last_sentences))
        calls.append((model.serial, visible))
        if model.serial:
            field = model._last_understanding.reconstruction_carriers['field']
            assert field.evidence.shape[1] == len(texts)
        return execute(*args, **kwargs)
    monkeypatch.setattr(model, '_run_batch_once', record)
    raw = (model.inputSpace.prepPackedInput([texts]) if packed else
           model.inputSpace.prepInput(texts))
    batch = 1 if packed else 2
    with torch.no_grad():
        model.runBatch(train=False, batchSize=batch, split='validation',
            batch_override=(raw, torch.empty(batch, 0)),
            questions=tuple(What.present(b, split='validation') for b in range(batch)))
    assert calls == [(False, tuple(texts)), (True, tuple(texts))]
    assert not hasattr(model.mode_schedule, 'resymbolize')
    assert not hasattr(model._concept_owner(), '_label_feedback')


def test_interleave_cursor_stages_exact_groups_and_keeps_tail_and_targets():
    from ModeSchedule import InterleaveCursor
    source = SimpleNamespace(inputs=['a', 'b', 'c', 'd', 'e'],
                             outputs=[10, 20, 30, 40, 50], document_ids=None)
    cursor = InterleaveCursor(source, every=3, batch_size=2)
    ticks = []
    while not cursor.all_done():
        inputs, outputs, hard = cursor.next_tick()
        ticks.append((inputs, outputs, cursor.last_source_indices,
                      cursor.context_sentences, hard))
    assert ticks == [
        (['a', 'b'], [10, 20], [0, 1], ('a', 'b', 'c'), [True, True]),
        (['c'], [30], [2], None, [True]),
        (['d', 'e'], [40, 50], [3, 4], ('d', 'e'), [True, True]),
    ]
    assert cursor.progress() == 1.


def test_fixed_discrimination_metric_keeps_all_probes_without_training():
    from CategoricalDiscrimination import FIXED_PROBES, fixed_probe_discrimination
    assert {name: len(probes['texts']) for name, probes in FIXED_PROBES.items()} == {
        'xor': 4, 'fineweb': 68}
    readings = {name: torch.zeros(len(probes['texts']), 8, requires_grad=True)
                for name, probes in FIXED_PROBES.items()}
    result = fixed_probe_discrimination(readings)
    assert result['categorical_discrimination']['xor']['cp'] == 0.
    assert result['categorical_discrimination']['fineweb']['cp'] == 0.
    assert all(value.grad is None for value in readings.values())


def test_interleave_epoch_reads_a_short_final_group_once(tmp_path, monkeypatch):
    from ModeSchedule import ModeSchedule
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    model.mode_schedule = ModeSchedule('interleave:2')
    texts = ['the wug sat', 'the wug flew', 'the wug ran']
    calls, ticks = [], []
    execute = model._run_batch_once
    advance = model._advance_when_time
    @wraps(execute)
    def record(*args, **kwargs):
        calls.append((model.serial, tuple(model.inputSpace._last_sentences)))
        return execute(*args, **kwargs)
    def clock():
        ticks.append(model.serial)
        return advance()
    monkeypatch.setattr(model, '_run_batch_once', record)
    monkeypatch.setattr(model, '_advance_when_time', clock)
    with model.inputSpace.data.runtime_batch(texts):
        model.runEpoch(None, batchSize=1, split='runtime')
    assert calls == [(False, tuple(texts[:2])), (True, (texts[0],)),
                     (True, (texts[1],)), (False, (texts[2],)), (True, (texts[2],))]
    assert ticks == [True, True, True]
    assert model.mode_schedule.pending == []
    assert model.mode_schedule.completed_parallel == 2


def test_interleave_checkpoint_resumes_unread_serial_prefix(tmp_path, monkeypatch):
    from ModeSchedule import ModeSchedule
    from What import What
    from test_packed_reconstruction_parity import build_model
    source = build_model(tmp_path, word_capacity=8)
    source.mode_schedule = ModeSchedule('interleave:2')
    texts = ('the wug sat', 'the wug flew')
    with torch.no_grad():
        raw = source.inputSpace.prepInput([texts[0]])
        source.runBatch(train=False, batchSize=1, split='validation',
            batch_override=(raw, torch.empty(1, 0)), schedule_context=texts,
            questions=(What.present(0, split='validation'),))
    checkpoint = tmp_path / 'pending.ckpt'
    source.save_weights(checkpoint)
    target_path = tmp_path / 'target'
    target_path.mkdir()
    target = build_model(target_path, word_capacity=8)
    target.mode_schedule = ModeSchedule('interleave:2')
    assert target.load_weights(checkpoint, strict=True, require_match=True)
    assert target.mode_schedule.pending == [texts[1]]
    calls = []
    execute = target._run_batch_once
    @wraps(execute)
    def record(*args, **kwargs):
        calls.append(target.serial)
        return execute(*args, **kwargs)
    monkeypatch.setattr(target, '_run_batch_once', record)
    with torch.no_grad():
        raw = target.inputSpace.prepInput([texts[1]])
        target.runBatch(train=False, batchSize=1, split='validation',
            batch_override=(raw, torch.empty(1, 0)),
            questions=(What.present(0, split='validation'),))
    assert calls == [True]
    assert target.mode_schedule.pending == []
    assert target.mode_schedule.completed_parallel == 1


def test_legacy_pending_schedule_cannot_replay_the_wrong_direction():
    from ModeSchedule import ModeSchedule
    schedule = ModeSchedule('interleave:2')
    with pytest.raises(ValueError, match='serial-first'):
        schedule.load_state_dict(dict(pending=[('already read', (17,))]))
    assert schedule.pending == []


def test_interleave_resume_rejects_a_different_grouping_even_between_groups():
    from ModeSchedule import ModeSchedule
    state = ModeSchedule('interleave:2').state_dict()
    ModeSchedule('interleave:2').validate_resume(state)
    for target in ('interleave:3', 'serial', 'parallel'):
        with pytest.raises(ValueError, match='mid-epoch'):
            ModeSchedule(target).validate_resume(state)
    with pytest.raises(ValueError, match='mid-epoch'):
        ModeSchedule('interleave:2').validate_resume({'pending': []})


def test_interleave_cursor_preserves_document_boundaries():
    from ModeSchedule import InterleaveCursor
    source = SimpleNamespace(inputs=['a', 'b', 'c', 'd'], outputs=None,
                             document_ids=['one', 'one', 'one', 'two'])
    cursor = InterleaveCursor(source, every=2, batch_size=1)
    assert [cursor.next_tick()[2] for _ in source.inputs] == [[False], [False], [True], [True]]


def test_failed_context_restores_serial_mode_input_and_teacher(tmp_path, monkeypatch):
    from ModeSchedule import ModeSchedule
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    model.mode_schedule = ModeSchedule('interleave:2')
    text = 'the wug sat'
    raw = model.inputSpace.prepInput([text])
    model.teacher.stage_batch_sources('validation', [0])
    @wraps(model._run_batch_once)
    def fail(*args, **kwargs):
        assert not model.serial
        raise RuntimeError('context failed')
    monkeypatch.setattr(model, '_run_batch_once', fail)
    with pytest.raises(RuntimeError, match='context failed'):
        model.runBatch(train=False, split='validation',
            batch_override=(raw, torch.empty(1, 0)),
            schedule_context=(text, 'the wug flew'))
    assert model.serial
    assert model.inputSpace._last_sentences == [text]
    assert model.teacher._staged_source_rows == [0]
    assert model.mode_schedule.pending == []


def test_evaluation_epoch_preserves_an_unfinished_training_group(tmp_path):
    from ModeSchedule import ModeSchedule
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    model.mode_schedule = ModeSchedule('interleave:2')
    model.mode_schedule.pending = ['unfinished training sentence']
    with model.inputSpace.data.runtime_batch(['the wug sat']):
        model.runEpoch(None, batchSize=1, split='runtime')
    assert model.mode_schedule.pending == ['unfinished training sentence']


@pytest.mark.parametrize('evaluation,resume', [(True, 0), (True, 2), (False, 0), (False, 2)])
def test_epoch_queue_lifetime_on_success_and_error(evaluation, resume):
    from ModeSchedule import ModeSchedule, scheduled_epoch
    for fail in (False, True):
        schedule = ModeSchedule('interleave:2')
        schedule.pending = ['training tail']
        model = SimpleNamespace(mode_schedule=schedule, _resume_batches_to_skip=resume)
        seen = []
        @scheduled_epoch
        def epoch(model, optimizer=None):
            seen.extend(model.mode_schedule.pending)
            if fail:
                raise RuntimeError('epoch failed')
        if fail:
            with pytest.raises(RuntimeError, match='epoch failed'):
                epoch(model, optimizer=None if evaluation else object())
        else:
            epoch(model, optimizer=None if evaluation else object())
        assert seen == ([] if evaluation or not resume else ['training tail'])
        assert schedule.pending == (['training tail'] if evaluation or resume else [])


def test_interleave_capacity_uses_the_input_encoders_bytes(tmp_path, monkeypatch):
    from ModeSchedule import ModeSchedule
    from test_packed_reconstruction_parity import build_model
    model = build_model(tmp_path, word_capacity=8)
    model.mode_schedule = ModeSchedule('interleave:2')
    monkeypatch.setattr(model.inputSpace.data, 'inputLength', 4)
    texts = ['éééé', 'abcd']
    seen = []
    @wraps(model._run_batch_once)
    def record(*args, **kwargs):
        seen.append((model.serial, kwargs['batch_override'][0].clone()))
        return None, 0
    monkeypatch.setattr(model, '_run_batch_once', record)
    raw = model.inputSpace.prepInput(texts)
    model.runBatch(train=False, batchSize=2, split='runtime',
                   batch_override=(raw, torch.empty(2, 0)))
    assert [serial for serial, _ in seen] == [False, True]
    assert raw[0].flatten().tolist() == [63, 63, 63, 63]
    for _, presented in seen:
        torch.testing.assert_close(presented, raw)
