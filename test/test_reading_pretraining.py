"""Reading is certified after native checkpoint restoration, without supplying a walk."""
import pytest
import torch

from pretrain_reading import LESSON, ROOT, load_lesson, qualifies, run


def test_reading_lesson_keeps_reversals_and_repetitions_in_the_held_out_set():
    spec = load_lesson(LESSON)
    unseen = set(spec['test'])
    assert len(unseen) >= 100
    assert all(' '.join(reversed(text.split())) in unseen for text in unseen)
    assert any(len(set(text.split())) < len(text.split()) for text in unseen)
    assert set(spec['required']).issubset(unseen)
    assert not unseen.intersection(spec['train'] + spec['validation'])
    for split in ('train', 'validation'):
        assert set(' '.join(spec[split]).split()) == set(' '.join(unseen).split())


def test_a_required_pair_fails_even_above_the_percentage_bar():
    spec = load_lesson(LESSON)
    row = dict(correct=121, total=122, failures=spec['required'][:1])
    assert not qualifies({'sampled': row, 'evaluation': row}, spec['required'], .99)


def test_reading_lesson_updates_only_during_ordinary_unlabeled_batches(monkeypatch, eager_reading):
    from Models import BaseModel
    from data import Data
    from pretrain_reading import train_reader
    data = Data()
    model, _ = BaseModel.from_config(str(ROOT / 'data/XOR_grammar.xml'), data=data)
    optimizer = model.getOptimizer()
    run_batch, step = model.runBatch, optimizer.step
    batches_seen, steps_seen = [], []
    inside = False

    def observe_batch(*args, **kwargs):
        nonlocal inside
        assert kwargs['train'] and kwargs['batch_override'][1].numel() == 0
        assert not data.has_supervised_outputs
        assert getattr(model, '_attention_supplied_reading', None) is None
        batches_seen.append(kwargs['batchSize'])
        inside = True
        try:
            return run_batch(*args, **kwargs)
        finally:
            inside = False

    def observe_step(*args, **kwargs):
        assert inside, 'the lesson trained outside the model presentation'
        steps_seen.append(True)
        return step(*args, **kwargs)

    monkeypatch.setattr(model, 'runBatch', observe_batch)
    monkeypatch.setattr(optimizer, 'step', observe_step)
    try:
        data.load('xor')
        original_inputs = data.train_input
        spec = dict(train=['hello world', 'world hello'], validation=['hello hello'],
                    batch_size=2, epochs=1, minimum_accuracy=.99)
        train_reader(model, spec, optimizer)
        assert batches_seen == [2] and steps_seen
        assert data.train_input is original_inputs and data.has_supervised_outputs
        assert model._training_step_count == 1
    finally:
        model.End()


def test_checkpoint_certifies_sampled_and_evaluation_reading(tmp_path, eager_reading):
    from Assessment import checkpoint_qualification
    report = run(ROOT / 'data/XOR_grammar.xml', LESSON, tmp_path / 'run')
    assert report['passed'], report['test']
    for row in report['test'].values():
        assert row['total'] == 122 and row['correct'] / row['total'] >= .99
        assert set(row['failures']).isdisjoint(load_lesson(LESSON)['required'])
    restored = checkpoint_qualification(tmp_path / 'run/reader.ckpt', 'what is cat ?')
    assert restored['age'] == pytest.approx(report['age'])
    assert restored['status'] == ('qualified' if report['age'] >= restored['required']
                                   else 'not qualified at that age')

    # Start another corpus from the certified checkpoint using the normal optimizer.
    from data import Data
    from Models import BaseModel
    data = Data()
    model, _ = BaseModel.from_config(str(ROOT / 'data/XOR_grammar.xml'), data=data)
    try:
        assert model.load_weights(str(tmp_path / 'run/reader.ckpt'), require_match=True)
        assert model.inputSpace.data is model.outputSpace.data is data
        before = model._training_step_count
        data.load('xor')
        model.runEpoch(optimizer=model.getOptimizer(), batchSize=4, split='train', max_batches=1)
        assert model._training_step_count == before + 1
    finally:
        model.End()


def test_missing_or_nonfinite_checkpoint_age_cannot_qualify(tmp_path):
    from Assessment import checkpoint_qualification
    path = tmp_path / 'reader.ckpt'
    for state in ({}, {'age': torch.tensor(float('nan'))}):
        torch.save({'state_dict': state}, path)
        with pytest.raises(ValueError, match='age'):
            checkpoint_qualification(path, 'what is cat ?')
