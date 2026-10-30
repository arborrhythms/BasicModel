"""Source adapters produce immutable, traceable inputs without learner labels."""
import csv
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from Corpus import corpus_rows
from prepare_corpus import build, digest, utterances, vocabulary
from collections import Counter


def test_vocabulary_uses_observed_production_and_retains_spelling_alternatives(tmp_path):
    path = tmp_path / 'responses.parquet'
    rows = [dict(item_definition=label, item_kind='word', age=age, produces=value)
            for label, age, values in (
                ('fruit (food)', 8, [False, True, None]),
                ('fruit (food)', 9, [False, True]),
                ('name/label', 10, [True, True]),
                ('late', 12, [False, None])) for value in values]
    pq.write_table(pa.Table.from_pylist(rows), path)
    found = list(vocabulary([path], {'production_threshold': .5}))
    assert [(row[4], row[0]) for row in found] == [('fruit', 9), ('label', 10), ('name', 10)]
    assert len({row[2] for row in found}) == 3
    assert all(row[3] == 0 for row in found)


def test_utterance_adapter_filters_by_metadata_and_uses_the_recorded_punctuation(tmp_path):
    path = tmp_path / 'utterances.csv'
    columns = ('id', 'speaker_role', 'collection_name', 'target_child_age', 'gloss',
               'type', 'transcript_id', 'utterance_order')
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i, (role, age, text, kind) in enumerate((
                ('adult', 100, 'Some words', 'question'),
                ('child', 100, 'private response', 'declarative'),
                ('adult', 365, 'Later words', 'declarative'),
                ('adult', 500, 'xxx', 'declarative'),
                ('adult', 'NA', 'Unlocated words', 'declarative'))):
            writer.writerow(dict(zip(columns, (i, role, 'collection', age, text, kind, i, 0))))
    config = dict(max_age_days=730, band_days=365, exclude_roles=['child'],
        collections=['collection'], unknown_tokens=['xxx'],
        punctuation={'question': '?', 'declarative': '.'})
    stats = Counter()
    found = list(utterances([path], config, stats))
    assert [(r[0], r[4]) for r in found] == [(100, 'Some words?'), (101, 'Later words.')]
    assert stats == dict(input_utterances=5, excluded_speaker=1, excluded_untranscribed=1,
                         excluded_unknown_age=1)


def test_assembly_pins_source_and_keeps_documents_in_one_split(tmp_path):
    path = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist([dict(text=f'word{i}.') for i in range(12)]), path)
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps(dict(version=1, name='fixture', groups=[dict(
        name='documents', adapter='document_parquet', sources=[dict(path=path.name, sha256=digest(path))])])))
    output = tmp_path / 'corpus'
    report = build(recipe, output, root=tmp_path)
    rows = list(corpus_rows(output / 'manifest.json'))
    assert report['documents'] == report['sentences'] == len(rows) == 12
    assert Counter(r['split'] for r in rows) == dict(train=10, validation=1, test=1)
    assert report['sources'][0]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(set(r) == {'text', 'stage', 'split', 'document', 'sentence', 'external_id'} for r in rows)
    with pytest.raises(FileExistsError, match='immutable'):
        build(recipe, output, root=tmp_path)
