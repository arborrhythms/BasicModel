"""Curriculum barriers, complete documents, source integrity and resume."""
import json

import pytest

from Corpus import CurriculumStreamDataset, corpus_rows, file_digest


def fixture_rows():
    # Uneven stages, a stage smaller than the batch, and uneven documents.
    return [dict(text=f's{stage}d{doc}n{sentence}', stage=stage,
                 document=f'{stage}:{doc}', sentence=sentence, split='train')
            for stage, lengths in enumerate(((3, 1, 2), (1,), (2, 1, 1, 3)))
            for doc, length in enumerate(lengths) for sentence in range(length)]


def cursor(rows, streams=4):
    return CurriculumStreamDataset([r['text'] for r in rows], streams,
        stages=[r['stage'] for r in rows], document_ids=[r['document'] for r in rows])


def ticks(ds):
    result = []
    while not ds.packed_done():
        tick = ds.next_packed_tick(2, lambda _: 1)
        result.append((tick, ds.last_source_indices, ds.progress()))
    return result


def manifest(tmp_path, rows):
    shard = tmp_path / 'rows.jsonl'
    shard.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(version=1, corpus='fixture', shards=[dict(
        path=shard.name, sha256=file_digest(shard), rows=len(rows))])))
    return path


def test_every_sentence_survives_and_every_stream_finishes_each_stage():
    rows = fixture_rows()
    observed = ticks(cursor(rows))
    sources = [i for _tick, batch, _progress in observed for stream in batch for i in stream]
    assert sorted(sources) == list(range(len(rows)))
    stage_order = [rows[i]['stage'] for i in sources]
    assert stage_order == sorted(stage_order)
    assert observed[-1][2] == 1
    assert [r[2] for r in observed] == sorted(r[2] for r in observed)
    for (texts, _outputs, ends), sources, _progress in observed:
        for text, indices, end in zip(texts, sources, ends):
            assert text == [rows[i]['text'] for i in indices]
            assert len({rows[i]['document'] for i in indices}) <= 1
            if indices:
                i = indices[-1]
                assert end == (i + 1 == len(rows) or rows[i]['document'] != rows[i + 1]['document'])


@pytest.mark.parametrize('streams', (1, 3, 4))
def test_resume_by_tick_recreates_source_rows_across_stage_barriers(streams):
    rows = fixture_rows()
    expected = ticks(cursor(rows, streams))
    for skip in range(len(expected) + 1):
        resumed = cursor(rows, streams)
        for _ in range(skip):
            resumed.next_packed_tick(2, lambda _: 1)
        assert ticks(resumed) == expected[skip:]
    resumed.reset_cursor()
    assert resumed.progress() == 0
    assert ticks(resumed) == expected


def test_stage_cursor_rejects_unpacked_and_byte_reads():
    ds = cursor(fixture_rows())
    with pytest.raises(ValueError, match='packSentences'):
        ds.next_tick()
    with pytest.raises(ValueError, match='packed sentence'):
        iter(ds)
    with pytest.raises(ValueError, match='complete-sentence'):
        CurriculumStreamDataset(['x'], 1, stages=[0], document_ids=[0], slab_bytes=4)


def test_manifest_verifies_source_hash_and_count(tmp_path):
    rows = fixture_rows()
    path = manifest(tmp_path, rows)
    assert list(corpus_rows(path)) == rows
    content = json.loads(path.read_text())
    content['shards'][0]['rows'] += 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='row count'):
        list(corpus_rows(path))
    (tmp_path / 'rows.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='hash differs'):
        list(corpus_rows(path))


@pytest.mark.parametrize('change,expected', (
    ('split', 'cross splits'), ('stage', 'chronological'), ('sentence', 'addresses')))
def test_manifest_rejects_leakage_reordering_and_duplicate_addresses(tmp_path, change, expected):
    rows = fixture_rows()
    if change == 'split':
        rows[1]['split'] = 'test'
    elif change == 'stage':
        rows[-1]['stage'] = 0
    else:
        rows[1]['sentence'] = rows[0]['sentence']
    with pytest.raises(ValueError, match=expected):
        list(corpus_rows(manifest(tmp_path, rows)))


def test_data_loader_uses_the_stage_cursor_without_supplying_age_as_content(tmp_path):
    from data import Data
    rows = fixture_rows()
    path = manifest(tmp_path, rows)
    data = Data()
    data.load('curriculum', dat={'corpusManifest': str(path)}, max_docs=None)
    assert data.train_input == [r['text'] for r in rows]
    assert not data.has_supervised_outputs and data.text_answers is None
    ds = data.data_loader('train', 4).dataset
    assert isinstance(ds, CurriculumStreamDataset)
    assert sum(len(stream) for _tick, sources, _progress in ticks(ds) for stream in sources) == len(rows)
    assert data.source_manifest['manifest_sha256'] == file_digest(path)
    assert data.source_manifest['stream_policy'] == 'stage_barriers_contiguous_documents_v1'


def test_unit_limit_excludes_the_whole_record_and_preserves_its_source_gap(tmp_path):
    from data import Data
    from Corpus import unit_capacity_bound
    rows = [dict(text=text, split='train', stage=0, document='one', sentence=i)
            for i, text in enumerate(('alphabet.', 'a !!!!!!', 'another.'))]
    data = Data()
    data.load('curriculum', dat={'corpusManifest': str(manifest(tmp_path, rows)),
              'maxSentenceUnits': 4}, max_sentence_words=4)
    assert data.train_input == ['alphabet.', 'another.']
    assert [row['sentence'] for row in data.source_addresses['train']] == [0, 2]
    assert data.source_manifest['excluded_unit_bound'] == 1
    assert unit_capacity_bound('alphabet.') == 3


def test_unit_bound_covers_every_native_refinement_cut():
    from Corpus import unit_capacity_bound
    from Meronomy import pin_word_spans
    # Exhaustive interval partitions of mixed letters/digits/punctuation;
    # the pinning operation is exactly the one used by the live tiler.
    for text in ('a 12!', 'xy??z', 'a-b.c', 'ABC  '):
        raw = text.encode() + b' '
        for bits in range(1 << (len(raw) - 1)):
            cuts = [0] + [i for i in range(1, len(raw)) if bits & (1 << (i - 1))] + [len(raw)]
            assert len(pin_word_spans(raw, list(zip(cuts, cuts[1:])))) <= unit_capacity_bound(text)


def test_saved_positions_resume_after_the_native_unit_count_changes():
    rows = fixture_rows()
    original = cursor(rows, 1)
    original.next_packed_tick(2, lambda _: 1)
    consumed = original.last_source_indices[0]
    saved = original.cursor_state()
    resumed = cursor(rows, 1)
    resumed.restore_cursor(saved)
    later = []
    while not resumed.packed_done():
        resumed.next_packed_tick(2, lambda _: 2)
        later.extend(resumed.last_source_indices[0])
    assert consumed + later == list(range(len(rows)))
    # Replaying the old tick count under the new tiling would revisit a row.
    replay = cursor(rows, 1)
    replay.next_packed_tick(2, lambda _: 2)
    assert replay.cursor_state() != saved


def test_saved_cursor_rejects_another_source_layout_and_invalid_position():
    ds = cursor(fixture_rows(), 4)
    saved = ds.cursor_state()
    for field, value in (('streams', 3), ('source', 'different'), ('layout', 'different'),
                         ('stage', -1), ('steps', [100, 0, 0, 0])):
        with pytest.raises(ValueError, match='curriculum resume'):
            ds.restore_cursor(dict(saved, **{field: value}))
    assert ds.cursor_state() == saved


def test_prefetch_checkpoint_uses_the_consumed_tick_positions():
    from Models import BasicModel
    from functools import partial
    ds = cursor(fixture_rows(), 1)
    producer = BasicModel._TickPrefetcher(ds, queue_size=2,
        next_fn=partial(ds.next_packed_tick, 2, lambda _: 1), done_fn=ds.packed_done)
    try:
        _tick, sources = producer.next()
        saved = producer.last_cursor_state
        resumed = cursor(fixture_rows(), 1)
        resumed.restore_cursor(saved)
        later = [index for _tick, rows, _progress in ticks(resumed) for stream in rows for index in stream]
        assert sources[0] + later == list(range(len(fixture_rows())))
        assert saved['steps'] == [len(sources[0])]
    finally:
        producer.close()
