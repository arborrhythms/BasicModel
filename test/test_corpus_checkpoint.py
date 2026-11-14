"""Native epoch/checkpoint boundaries, with the expensive learner stubbed.

This verifies scheduling and persistence, not training parity or acquisition.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import pytest

from test_corpus import manifest


def test_checkpoint_uses_consumed_positions_and_resumes_without_epoch_clear(tmp_path, monkeypatch):
    from data import TheData
    from Models import BaseModel

    root = Path(__file__).resolve().parents[1]
    tree = ET.parse(root / 'data/MM_corpus_orders.xml')
    tree.find('./architecture/training/packSentences').text = 'true'
    tree.find('./architecture/training/numWorkers').text = '3'
    config = tmp_path / 'resume.xml'
    tree.write(config)
    rows = [dict(text=f'word{i}.', split='train', stage=i // 4,
                 document=f'd{i // 2}', sentence=i % 2) for i in range(6)]
    rows += [dict(text='heldout.', split='validation', stage=0,
                  document='validation', sentence=0)]
    TheData.load('curriculum', dat={'corpusManifest': str(manifest(tmp_path, rows))})
    models, observed, batch_numbers = [], [], []

    def create():
        model = BaseModel.from_config(str(config), data=TheData)[0]
        models.append(model)
        # Drive the real epoch cursor and checkpoint code; learner work is
        # outside this test. The counter can change after a checkpoint.
        monkeypatch.setattr(model, '_install_unit_span_fn', lambda: None)
        monkeypatch.setattr(model.inputSpace, 'sentence_unit_count', lambda *a, **k: 1)
        def consume(**kwargs):
            observed.extend(i for stream in kwargs['source_rows'] for i in stream)
            batch_numbers.append(kwargs['batchNum'])
            return None, None
        monkeypatch.setattr(model, 'runBatch', consume)
        return model

    try:
        first = create()
        first.runEpoch(optimizer=object(), batchSize=1, max_batches=1)
        assert observed == [0, 1]
        assert first._epoch_batches_seen == 1
        saved_cursor = first._checkpoint_corpus_cursor
        assert saved_cursor['steps'] == [2]
        first._concept_owner()._priming_seen[0, 3] = 1.75
        first.symbolSpace.expectation._expectation_documents[0] = ('observed-document', 5)
        # Off-stack recent chunks feed the next symbol forecast, even when
        # the previous sentence left no parse stack. These are plain live
        # tensors, so a state_dict round-trip alone cannot preserve them.
        from Memory import STMChunks as chunks
        stm = first.conceptualSpace.CSLang
        chunk = list(stm.state)
        chunk[chunks.BUFFER][0, 0].fill_(.25)
        chunk[chunks.IDENTITY][0, 0] = 37
        chunk[chunks.READ][0, 0] = 12
        stm.set_state(tuple(chunk))
        pool = first._concept_owner().mtm
        candidate = pool.open_candidate(order=1, when=1, value=torch.ones(first._concept_owner().similarity_codebook.W.shape[-1]),
            support={3: .75, 5: .25})
        pool.record_reading(candidate, 0, (.8, .1))
        percept=first.percept_expectation
        width=percept.width
        percept_bank=lambda: iter([(torch.tensor([3]),torch.full((1,1,width),.25))])
        for _ in range(2):
            percept.arrive(percept.forecast(percept_bank,1),torch.full((1,width),.5),
                           torch.tensor([True]),native=torch.full((1,percept.moments.mean.numel()),.5),training=True)
        from Meaning import ConceptualMeaning
        store = first.symbolSpace.ltm_store
        row = store.append_meaning(ConceptualMeaning.from_description(store.slots[0].new_ones(store.nDim)),
            evidence=(.8,.2), )
        occurrence = store.occurrence_of(row)
        owner = first._concept_owner()
        left, right = owner.new_concept(), owner.new_concept()
        handle = store.parts.add(left,right,truth_plus=.75,truth_minus=.25)
        # Materialize the derived authority before the exact checkpoint snapshot.
        assert tuple(map(float, store.parts.evidence(handle))) == pytest.approx((.75,.25))
        relation_state = {k:v.clone() for k,v in store.parts.state_dict().items()}
        checkpoint = tmp_path / 'cursor.ckpt'
        first.save_weights(str(checkpoint))

        resumed = create()
        assert resumed.load_weights(str(checkpoint), require_match=True)
        assert resumed._resume_corpus_cursor == saved_cursor
        restored_store = resumed.symbolSpace.ltm_store
        assert restored_store.location(occurrence) == row
        for key,value in relation_state.items():
            assert torch.equal(value,restored_store.parts.state_dict()[key])
        restored_percept=resumed.percept_expectation
        for key,value in percept.state_dict().items():
            assert torch.equal(value,restored_percept.state_dict()[key])
        # This fixture stubs the learner. Arrivals carry causal context and
        # moments, while parameter updates belong to the sentence optimizer.
        assert not hasattr(percept, 'optimizer') and not hasattr(restored_percept, 'optimizer')
        assert percept.continuation() == restored_percept.continuation() == dict(
            version=2, adaptation='arrival_state')
        parameters = {key:value.detach().clone() for key,value in percept.named_parameters()}
        for owner in (percept,restored_percept):
            owner.arrive(owner.forecast(percept_bank,1),torch.full((1,width),.5),
                         torch.tensor([True]),native=torch.full((1,percept.moments.mean.numel()),.5),training=True)
        for key,value in percept.state_dict().items():
            assert torch.equal(value,restored_percept.state_dict()[key])
        assert all(torch.equal(value, dict(percept.named_parameters())[key])
                   for key, value in parameters.items())
        restored_candidate = resumed._concept_owner().mtm.entries[candidate]
        assert restored_candidate.support == {3: .75, 5: .25}
        assert restored_candidate.readings == pool.entries[candidate].readings
        assert resumed._concept_owner().mtm.candidate_counter == pool.candidate_counter
        assert float(resumed._concept_owner()._priming_seen[0, 3]) == 1.75
        assert resumed.symbolSpace.expectation._expectation_documents[0] == ('observed-document', 5)
        for expected, actual in zip(stm.state, resumed.conceptualSpace.CSLang.state, strict=True):
            assert torch.equal(expected, actual)
        from CorpusCheckpoint import collect, restore
        saved_runtime = collect(resumed)
        key = next(iter(saved_runtime['stm']))
        saved_runtime['stm'][key]['fields']['_live_buffer'] = torch.empty(1, 1, 1)
        before = resumed._concept_owner()._priming_seen.clone()
        with pytest.raises(ValueError, match='working-memory slab'):
            restore(resumed, saved_runtime)
        assert torch.equal(before, resumed._concept_owner()._priming_seen)
        import Forgetting
        boundary = Forgetting.model_boundary
        def no_epoch_clear(model, **options):
            if model is resumed and options.get('epoch_boundary'):
                raise AssertionError('a resumed corpus is the same epoch')
            return boundary(model, **options)
        monkeypatch.setattr(Forgetting, 'model_boundary', no_epoch_clear)
        monkeypatch.setattr(resumed.inputSpace, 'sentence_unit_count', lambda *a, **k: 20)
        resumed.runEpoch(optimizer=object(), batchSize=1)
        assert observed == list(range(6))
        assert batch_numbers == [0, 1, 2, 3, 4]
        assert resumed._checkpoint_corpus_cursor is None
        assert resumed._epoch_batches_seen == 0
        assert restored_store.location(occurrence) == row
        assert restored_store.parts.find(left,right) == handle
        for key,value in relation_state.items():
            assert torch.equal(value,restored_store.parts.state_dict()[key])

        observed.clear()
        resumed.runEpoch(split='validation', batchSize=1)
        assert observed == [0]
    finally:
        for model in models:
            model.End()
            model.symbolSpace.soft_reset()


@pytest.mark.slow
@pytest.mark.parametrize('mixed_documents', [False, True], ids=['single-source', 'mixed-sources'])
def test_real_next_batch_matches_checkpoint_losses_and_kept_programs(tmp_path, monkeypatch, eager_reading, mixed_documents):
    """Compare one uninterrupted update with the same update after native restore."""
    from data import TheData
    from Models import BaseModel, BasicModel
    from CorpusCheckpoint import detached
    import hashlib

    root = Path(__file__).resolve().parents[1]
    tree = ET.parse(root/'data/MM_corpus_orders.xml')
    tree.find('./architecture/training/packSentences').text = 'true'
    config = tmp_path/'real-resume.xml'
    tree.write(config)
    rows = [dict(text=text, split='train', stage=0, document='document', sentence=i)
            for i, text in enumerate(('A cat sleeps.', 'A bird sings.', 'The cat sleeps.', 'The bird sings.') * 3)]
    if mixed_documents:
        # Alternate complete source documents; keep the original total rows,
        # model capacities and next-update equality assertions.
        for i, row in enumerate(rows):
            row.update(document=f'source-{(i//4)%2}:document-{i//4}', sentence=i%4)
    TheData.load('curriculum', dat={'corpusManifest': str(manifest(tmp_path, rows))})
    models, programs, losses = [], [], []
    original = BasicModel._commit_sentence

    def fingerprint(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            return (str(value.dtype), tuple(value.shape), hashlib.sha256(
                value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        if isinstance(value, dict):
            return {key: fingerprint(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(fingerprint(item) for item in value)
        return value

    def keep(model, state, sid, active, observations, predictions, wins):
        selected = []
        for b in range(len(active)):
            if bool(active[b]):
                entry = observations[int(wins[b])]['entries'][b]
                selected.append(None if entry is None else fingerprint({key: getattr(entry, key)
                    for key in ('actions', 'word_ids', 'case_values', 'case_selected')}))
        result = original(model, state, sid, active, observations, predictions, wins)
        programs.extend(selected)
        return result

    monkeypatch.setattr(BasicModel, '_commit_sentence', keep)

    def create():
        import Language
        Language.TheGrammar._configured = False
        model = BaseModel.from_config(str(config), data=TheData)[0]
        models.append(model)
        run = model.runBatch
        def observed(*args, **kwargs):
            result, cursor = run(*args, **kwargs)
            if result is not None:
                losses.append(fingerprint((result.lossIn, result.lossOut)))
            return result, cursor
        monkeypatch.setattr(model, 'runBatch', observed)
        return model

    try:
        first = create()
        optimizer = first.getOptimizer(lr=.001)
        first.runEpoch(optimizer=optimizer, batchSize=1, max_batches=1)
        assert first._checkpoint_corpus_cursor is not None
        checkpoint = tmp_path/'real.ckpt'
        first.save_weights(str(checkpoint))
        saved = detached(first.state_dict())
        # Assert the activity ledgers that replaced surprise/access counts
        # in section 17.14, as well as the remaining checkpoint state.
        ledger_names = [name for name in saved if any(part in name for part in
                       ('activity', 'forget_', 'relevance', 'placement'))]
        assert any(name.endswith('ltm_store.activity') for name in ledger_names)
        assert any('forget_' in name for name in ledger_names)
        # The allocator owns LTM1; its checkpoint is separate from the
        # model's registered tensors, as in the native save/load path.
        saved_part_activity = first.symbolSpace.ltm_store.parts.activity.detach().clone()
        programs.clear(); losses.clear()
        first.runEpoch(optimizer=optimizer, batchSize=1, max_batches=1)
        expected_programs, expected_losses = list(programs), list(losses)
        assert expected_programs and expected_losses
        expected_state = fingerprint(first.state_dict())

        resumed = create()
        assert resumed.load_weights(str(checkpoint), require_match=True)
        actual = resumed.state_dict()
        for name in ledger_names:
            assert fingerprint(saved[name]) == fingerprint(actual[name]), name
        assert torch.equal(saved_part_activity, resumed.symbolSpace.ltm_store.parts.activity)
        # Item 2.0 retired the independent answer comparison reader.
        assert not hasattr(resumed, 'comparison_reader')
        optimizer = resumed.getOptimizer(lr=.001)
        programs.clear(); losses.clear()
        resumed.runEpoch(optimizer=optimizer, batchSize=1, max_batches=1)
        assert losses == expected_losses
        assert programs == expected_programs
        assert fingerprint(resumed.state_dict()) == expected_state
    finally:
        for model in models:
            model.End()
            model.symbolSpace.soft_reset()
