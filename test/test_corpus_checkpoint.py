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
        import STMChunks as chunks
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
        checkpoint = tmp_path / 'cursor.ckpt'
        first.save_weights(str(checkpoint))

        resumed = create()
        assert resumed.load_weights(str(checkpoint), require_match=True)
        assert resumed._resume_corpus_cursor == saved_cursor
        restored_percept=resumed.percept_expectation
        for key,value in percept.state_dict().items():
            assert torch.equal(value,restored_percept.state_dict()[key])
        assert restored_percept.optimizer is not None
        for key,values in percept.optimizer.state_dict()['state'].items():
            for name,value in values.items():
                assert torch.equal(value,restored_percept.optimizer.state_dict()['state'][key][name])
        for owner in (percept,restored_percept):
            owner.arrive(owner.forecast(percept_bank,1),torch.full((1,width),.5),
                         torch.tensor([True]),native=torch.full((1,percept.moments.mean.numel()),.5),training=True)
        for key,value in percept.state_dict().items():
            assert torch.equal(value,restored_percept.state_dict()[key])
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
        def no_epoch_clear():
            raise AssertionError('a resumed corpus is the same presentation')
        monkeypatch.setattr(resumed, 'begin_presentation', no_epoch_clear)
        monkeypatch.setattr(resumed.inputSpace, 'sentence_unit_count', lambda *a, **k: 20)
        resumed.runEpoch(optimizer=object(), batchSize=1)
        assert observed == list(range(6))
        assert batch_numbers == [0, 1, 2, 3, 4]
        assert resumed._checkpoint_corpus_cursor is None
        assert resumed._epoch_batches_seen == 0

        observed.clear()
        resumed.runEpoch(split='validation', batchSize=1)
        assert observed == [0]
    finally:
        for model in models:
            model.End()
            model.symbolSpace.soft_reset()
