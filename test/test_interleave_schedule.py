"""The configured schedule changes passes, preserving their shared owners."""
import pytest
import torch


@pytest.mark.parametrize('value', ['', 'interleave:0', 'interleave:-1', 'interleave:1.5', 'mixed'])
def test_schedule_rejects_invalid_values(value):
    from ModeSchedule import ModeSchedule
    with pytest.raises(ValueError, match='modeSchedule'):
        ModeSchedule(value)


def test_schedule_keeps_unread_sentence_prefix_and_remainder():
    from ModeSchedule import ModeSchedule
    schedule = ModeSchedule('interleave:2')
    assert schedule.context_for(['first'], ['first', 'second']) == ('first', 'second')
    schedule.pending = ['second']
    assert schedule.context_for(['second']) is None
    with pytest.raises(ValueError, match='do not match'):
        schedule.context_for(['third'])


@pytest.mark.parametrize("training", [False, True])
def test_native_interleave_supplies_context_then_reads_the_same_sentences(tmp_path, monkeypatch, training, eager_reading):
    from ModeSchedule import ModeSchedule
    from test_packed_reconstruction_parity import build_model
    from What import What
    if training:
        from test_compiled_word_chunk import _tiny_canonical_model
        model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
            concept_rows=128, input_width=16, batch_size=1, chooser_depth=1,
            training_overrides={'reconstructionPlacement': 'eager'},
            architecture_overrides={'answerSynthesis': False})
        model._tensor_peer_while_eager = True
        model._chart_compose_per_word = lambda: None
        model.checkpoint_every_batches = 0
    else:
        model = build_model(tmp_path, word_capacity=8)
    import warnings
    optimizer = model.getOptimizer(lr=1e-3) if training else None
    owner = model._concept_owner()
    parameter = owner.similarity_codebook.getW()
    parallel_gradients = []
    backward = model._backward_training_loss
    def capture_backward(*args, **kwargs):
        result = backward(*args, **kwargs)
        if not model.serial:
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            parallel_gradients.append(any(bool(g.count_nonzero()) for g in gradients))
            assert all(bool(torch.isfinite(g).all()) for g in gradients)
        return result
    if training:
        monkeypatch.setattr(model, '_backward_training_loss', capture_backward)
    texts = ('a b', 'a c')
    if training:
        # Preserve the same two admission/training observations. Alec's
        # September 26 correction now requires NO context backward call.
        for step, text in enumerate(texts):
            raw = model.inputSpace.prepInput([text])
            model.runBatch(train=True, optimizer=optimizer, batchNum=step,
                batchSize=1, split='validation',
                batch_override=(raw, torch.empty(1, 0)),
                questions=(What.present(0, split='validation'),))
    model.mode_schedule = ModeSchedule('interleave:2')
    for step, text in enumerate(texts):
        raw = model.inputSpace.prepInput([text])
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Using a target size.*')
            model.runBatch(train=training, optimizer=optimizer, batchNum=step,
                           batchSize=1, split='validation',
                           batch_override=(raw, torch.empty(1, 0)),
                           questions=(What.present(0, split='validation'),),
                           schedule_context=texts if step == 0 else None)
        assert model.mode_schedule.completed_parallel == 1
    if training:
        assert parallel_gradients == []
    assert model._training_step_count == (4 if training else 0)
    assert model.serial
    assert model._concept_owner() is owner
    assert owner.similarity_codebook.getW().data_ptr() == parameter.data_ptr()
    assert not hasattr(owner, '_label_feedback')
    assert model.mode_schedule.pending == []
    model.End()
