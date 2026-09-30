"""Training provenance, independent of whether a model has learned well."""
from types import SimpleNamespace
import pytest
import torch


def _host():
    from LearningEvaluation import FINEWEB_CORPUS
    addresses = [dict(split='train', document=0, sentence=i) for i in range(4)]
    data = SimpleNamespace(source_manifest=dict(dataset='text', corpus=FINEWEB_CORPUS,
        shards=[dict(path='fixture.parquet', size=123)]), source_addresses={'train': addresses})
    program = SimpleNamespace(leaves=torch.empty(2, 4, device='meta'))
    return SimpleNamespace(inputSpace=SimpleNamespace(data=data),
        _last_understanding=SimpleNamespace(sentence_fields={0: (program, program),
            1: (program, None)}, sentence_states=(program, program)))


def test_progress_counts_completed_ragged_sentences_without_reading_device_values():
    from LearningEvaluation import record_fineweb_training
    model = _host()
    record_fineweb_training(model, split='train', source_rows=[[0, 1], [2]])
    assert model._fineweb_training_progress['sentences'] == 3
    assert model._fineweb_training_progress['updates'] == 1
    record_fineweb_training(model, split='validation', source_rows=[[0, 1], [2]])
    assert model._fineweb_training_progress['sentences'] == 3
    model.inputSpace.data.source_manifest['corpus'] = 'grammar fixture'
    record_fineweb_training(model, split='train', source_rows=[[0, 1], [2]])
    assert model._fineweb_training_progress['sentences'] == 3


def test_unaddressed_or_unfinished_presentations_do_not_claim_fineweb_training():
    from LearningEvaluation import record_fineweb_training
    model = _host()
    record_fineweb_training(model, split='train', source_rows=[[-1, 400], [None]])
    assert getattr(model, '_fineweb_training_progress', {}).get('sentences', 0) == 0
    model._last_understanding.sentence_fields = {0: (None, None)}
    record_fineweb_training(model, split='train', source_rows=[0, 1])
    assert getattr(model, '_fineweb_training_progress', {}).get('sentences', 0) == 0


def test_learning_gate_uses_recorded_sentences_not_epochs_or_batch_estimates():
    from LearningEvaluation import fineweb_readiness, FINEWEB_CORPUS
    state = {'training_step_count': 1_000_000, 'epoch_batches_seen': 1_000_000,
             'data_manifest': {'corpus': FINEWEB_CORPUS}}
    assert not fineweb_readiness(state).eligible
    progress = dict(version=1, corpus=FINEWEB_CORPUS, sentences=999_999,
                    updates=10, manifests={'fixture': {'corpus': FINEWEB_CORPUS}})
    state['fineweb_training_progress'] = progress
    assert not fineweb_readiness(state).eligible
    progress['sentences'] += 1
    assert fineweb_readiness(state).eligible
    assert not fineweb_readiness(state, minimum_sentences=2_000_000).eligible
    progress['corpus'] = 'grammar fixture'
    assert not fineweb_readiness(state).eligible


@pytest.mark.parametrize('value', [-1, True, 1.5, '1000000'])
def test_corrupt_progress_is_an_error_not_a_skipped_quality_result(value):
    from LearningEvaluation import fineweb_readiness, FINEWEB_CORPUS
    with pytest.raises(ValueError, match='sentences'):
        fineweb_readiness({'fineweb_training_progress': dict(version=1,
            corpus=FINEWEB_CORPUS, sentences=value, updates=1, manifests={})})


def test_integrated_checkpoint_preserves_fineweb_progress(tmp_path):
    import Models
    from LearningEvaluation import record_fineweb_training, fineweb_readiness
    source = Models.BaseModel()
    source.spaces = []
    source.conceptualSpaces = []
    source.wholeSpaces = []
    host = _host()
    source.inputSpace = host.inputSpace
    source._last_understanding = host._last_understanding
    record_fineweb_training(source, split='train', source_rows=[[0, 1], [2]])
    path = tmp_path / 'progress.ckpt'
    source.save_weights(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert fineweb_readiness(saved['training_state'], minimum_sentences=3).eligible
    target = Models.BaseModel()
    target.spaces = []
    target.conceptualSpaces = []
    target.wholeSpaces = []
    assert target.load_weights(path, require_match=True)
    assert target._fineweb_training_progress == source._fineweb_training_progress


@pytest.mark.parametrize('skipped,fused,expected', [(False, False, True),
                                                  (True, False, False),
                                                  (False, True, False)])
def test_amp_skips_do_not_inflate_training_exposure(skipped, fused, expected):
    from LearningEvaluation import scaler_step_performed
    class Optimizer:
        _step_supports_amp_scaling = fused
        calls = 0
        def step(self):
            self.calls += 1
    class Scaler:
        def step(self, optimizer):
            if not skipped:
                optimizer.step()
        def update(self):
            pass
    optimizer = Optimizer()
    assert scaler_step_performed(Scaler(), optimizer) is expected
    assert optimizer.calls == int(not skipped)
    assert 'step' not in vars(optimizer)


def test_native_progress_and_trained_artifact_evaluation_path(tmp_path, monkeypatch):
    """Exercise the real small model for plumbing, without a quality assertion."""
    import Models
    from data import Data
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_packed_reconstruction_parity import reset
    from reading_fixtures import force_absolute_reading
    from LearningEvaluation import FINEWEB_CORPUS, checkpoint_readiness
    from eval_fineweb_learning import load_model, read_validation
    from What import What
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')

    # The same tensor cells, without graph-dispatch startup for this counter test.
    def eager_while(condition, body, values):
        while bool(condition(*values)):
            values = body(*values)
        return values
    monkeypatch.setattr(torch, 'while_loop', eager_while)
    restore_weights = Models.BaseModel.load_weights
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        concept_rows=256, part_rows=128,
        architecture_overrides={'ltmConsolidation': True})
    monkeypatch.setattr(Models.BaseModel, 'load_weights', restore_weights)
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model.reconstruction_placement = 'eager'
    # This is checkpoint/prediction plumbing with an explicitly supplied
    # absolute reading, not a measurement of the untrained English parser.
    force_absolute_reading(model)
    data = model.inputSpace.data
    manifest = dict(dataset='text', corpus=FINEWEB_CORPUS,
                    shards=[dict(path='fixture.parquet', size=123)])
    monkeypatch.setattr(data, 'source_manifest', manifest)
    monkeypatch.setattr(data, 'source_addresses', {'train': [
        dict(split='train', document=0, sentence=i) for i in range(3)]})
    monkeypatch.setattr(data, 'grammar_lessons', {})
    optimizer = model.getOptimizer(lr=.0001)
    raw = model.inputSpace.prepPackedInput([['a b', 'c d'], ['a b']])
    model.runBatch(train=True, split='train', batchSize=2, optimizer=optimizer,
        source_rows=[[0, 1], [2]], batch_override=(raw, torch.empty(2, 0)),
        questions=(What.present(0), What.present(1)))
    assert model._fineweb_training_progress['sentences'] == 3
    assert model._fineweb_training_progress['updates'] == 1
    reset(model, packed=True, final=True, batch=2)
    raw = model.inputSpace.prepInput(['a b'])
    with torch.no_grad():
        model.runBatch(train=False, split='train', batchSize=1,
            source_rows=[0], batch_override=(raw, torch.empty(1, 0)),
            questions=(What.present(0),))
    assert model._fineweb_training_progress['sentences'] == 3
    reset(model, packed=False, final=True, batch=1)
    path = tmp_path / 'small-native.ckpt'
    model.save_weights(path)
    model.End()
    assert not checkpoint_readiness(path).eligible  # never call this a mature model
    assert checkpoint_readiness(path, minimum_sentences=3).eligible

    def heldout(self, *_args, **_kwargs):
        self.processLM({
            'train': {'text': ['a b'], 'label': []},
            'validation': {'text': ['a b', 'c d', 'a b', 'c d'], 'label': []},
            'test': {'text': [], 'label': []}})
        self.source_manifest = manifest
        self.source_addresses = {
            'train': [dict(split='train', document=10, sentence=0)],
            'validation': [dict(split='validation', document=i//2, sentence=i%2)
                           for i in range(4)]}
    monkeypatch.setattr(Data, 'load', heldout)
    loaded = load_model(tmp_path/'tiny_chunk_model.xml', path, minimum_sentences=3)
    force_absolute_reading(loaded)
    try:
        report = read_validation(loaded, sentences=4, gain=0.)
        assert report['predicted_targets'] == 2
        assert len(report['thought']) == 2
        assert all(row['answered'] for row in report['thought']), report['thought']
        assert loaded._fineweb_training_progress['sentences'] == 3
    finally:
        loaded.End()
    broken = torch.load(path, map_location='cpu', weights_only=False)
    key = next(k for k in broken['state_dict'] if k.endswith('.posting_codes'))
    broken['state_dict'][key] = broken['state_dict'][key].reshape(1, -1)
    corrupt = tmp_path/'invalid-index.ckpt'
    torch.save(broken, corrupt)
    with pytest.raises(ValueError, match='posting_codes'):
        load_model(tmp_path/'tiny_chunk_model.xml', corrupt, minimum_sentences=3)
