"""Build immutable, age-ordered corpus shards from a pinned source recipe.

Run with --recipe data/corpus-sources.json --output data/corpus.
Source metadata determines presentation order only. This tool creates no
object definitions, grammatical categories, answers or learner seeds.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib.request import urlopen


def digest(path, *, git=False):
    result = hashlib.sha1() if git else hashlib.sha256()
    if git:
        result.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def materialize(source, cache, root):
    if 'path' in source:
        path = root / source['path']
    else:
        path = cache / (hashlib.sha256(source['url'].encode()).hexdigest()[:16]
                        + '-' + source['url'].rsplit('/', 1)[-1])
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            with urlopen(source['url'], timeout=45) as remote, temporary.open('wb') as output:
                while block := remote.read(1024 * 1024):
                    output.write(block)
            temporary.rename(path)
    expected = source.get('sha256') or source.get('git_blob_sha1')
    if not expected:
        raise ValueError('every corpus source needs a pinned content digest')
    if digest(path, git='sha256' not in source) != expected:
        raise ValueError(f'corpus source digest differs: {path.name}')
    return path, dict(source, bytes=path.stat().st_size, sha256=digest(path))


def vocabulary(paths, config):
    """Earliest observed age reaching the declared production threshold."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    counts = {}
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,
                columns=['item_definition', 'item_kind', 'age', 'produces']):
            table = pa.Table.from_batches([batch])
            table = table.filter(pc.equal(table['item_kind'], 'word'))
            table = table.append_column('present', pc.cast(
                pc.fill_null(table['produces'], False), pa.int64()))
            summary = table.group_by(['item_definition', 'age']).aggregate(
                [('present', 'sum'), ('present', 'count')])
            for row in summary.to_pylist():
                if row['age'] is None:
                    continue
                key = row['item_definition'], row['age']
                pos, total = counts.get(key, (0, 0))
                counts[key] = pos + row['present_sum'], total + row['present_count']
    threshold = float(config['production_threshold'])
    if not 0 < threshold <= 1:
        raise ValueError('vocabulary production threshold must lie in (0, 1]')
    first = {}
    for (label, age), (positive, total) in counts.items():
        if positive / total < threshold:
            continue
        # Parentheses explain the questionnaire item; slash-separated forms
        # are its spelling alternatives. Keep every such form, without a POS
        # annotation or a language-specific word substitution.
        label = re.sub(r'\([^)]*\)', '', label)
        for form in label.split('/'):
            form = ' '.join(form.split())
            if form and all(c.isalpha() or c in " '-" for c in form):
                first[form] = min(age, first.get(form, age))
    for form, age in sorted(first.items(), key=lambda pair: (pair[1], pair[0])):
        identity = 'vocabulary:' + hashlib.sha256(form.encode()).hexdigest()
        yield int(age), float(age), identity, 0, form, identity


def utterances(paths, config, stats):
    max_age = float(config['max_age_days'])
    band = int(config['band_days'])
    if not math.isfinite(max_age) or max_age <= 0 or band < 1:
        raise ValueError('positive corpus age and band limits required')
    for path in paths:
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                stats['input_utterances'] += 1
                if row['speaker_role'] in config['exclude_roles']:
                    stats['excluded_speaker'] += 1
                    continue
                if row['collection_name'] not in config['collections']:
                    stats['excluded_collection'] += 1
                    continue
                try:
                    age = float(row['target_child_age'])
                except ValueError:
                    stats['excluded_unknown_age'] += 1
                    continue
                if not math.isfinite(age) or not 0 <= age < max_age:
                    stats['excluded_age'] += 1
                    continue
                text = ' '.join(row['gloss'].split())
                if not text or any(token in text.split() for token in config['unknown_tokens']):
                    stats['excluded_untranscribed'] += 1
                    continue
                punctuation = config['punctuation'].get(row['type'])
                if punctuation is None:
                    raise ValueError(f'unknown utterance type: {row["type"]!r}')
                if text[-1] not in '.?!':
                    text += punctuation
                document = 'transcript:' + row['transcript_id']
                yield (100 + int(age // band), age, document, int(row['utterance_order']),
                       text, 'utterance:' + row['id'])


def documents(paths, config):
    import pyarrow.parquet as pq
    from util import parse
    for shard, path in enumerate(paths):
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['text']):
            for text in batch.column(0).to_pylist():
                identity = f'document:{shard}:{index}'
                for sentence, (value, _start) in enumerate(parse(text, lex='sentences')):
                    if value.strip():
                        yield 1000, 0., identity, sentence, value, identity
                index += 1
                if config.get('max_documents') and index >= int(config['max_documents']):
                    break
            if config.get('max_documents') and index >= int(config['max_documents']):
                break


def build(recipe_path, output, *, root):
    recipe_path, output, root = Path(recipe_path), Path(output), Path(root)
    recipe = json.loads(recipe_path.read_text())
    if recipe.get('version') != 1:
        raise ValueError('corpus recipe version must be 1')
    if (output / 'manifest.json').exists():
        raise FileExistsError('an assembled corpus is immutable; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'sources'
    cache.mkdir(exist_ok=True)
    database = output / 'assembly.sqlite'
    if database.exists():
        raise FileExistsError('retain the incomplete assembly and choose a new output directory')
    started = time.monotonic()
    connection = sqlite3.connect(database)
    connection.execute('CREATE TABLE sentences (stage INTEGER, age REAL, document TEXT, '
                       'sentence INTEGER, text TEXT, external_id TEXT, PRIMARY KEY(document,sentence))')
    sources, stats = [], Counter()
    try:
        for group in recipe['groups']:
            paths = []
            for source in group['sources']:
                path, record = materialize(source, cache, root)
                paths.append(path)
                sources.append(dict(record, group=group['name']))
            if group['adapter'] == 'vocabulary_responses':
                rows = vocabulary(paths, group)
            elif group['adapter'] == 'utterance_csv':
                rows = utterances(paths, group, stats)
            elif group['adapter'] == 'document_parquet':
                rows = documents(paths, group)
            else:
                raise ValueError(f'unknown corpus adapter: {group["adapter"]}')
            connection.executemany('INSERT INTO sentences VALUES (?,?,?,?,?,?)', rows)
            connection.commit()
            print(f'Prepared {group["name"]}', flush=True)
        # One source document belongs to one stage and one observed age.
        inconsistent = connection.execute('SELECT document FROM sentences GROUP BY document '
            'HAVING MIN(stage) != MAX(stage) OR MIN(age) != MAX(age) LIMIT 1').fetchone()
        if inconsistent:
            raise ValueError(f'document has inconsistent age/stage: {inconsistent[0]}')
        connection.execute('CREATE INDEX chronology ON sentences(stage,age,document,sentence)')
        files, stages, documents_seen = [], [], 0
        for (stage,) in connection.execute('SELECT DISTINCT stage FROM sentences ORDER BY stage').fetchall():
            target = output / f'stage-{stage:04d}.jsonl'
            previous, count, counts = None, 0, Counter()
            with target.open('w') as stream:
                for age, document, sentence, text, external_id in connection.execute(
                        'SELECT age,document,sentence,text,external_id FROM sentences '
                        'WHERE stage=? ORDER BY age,document,sentence', (stage,)):
                    if document != previous:
                        split = ('train' if documents_seen % 10 < 8 else
                                 'validation' if documents_seen % 10 == 8 else 'test')
                        documents_seen += 1
                        previous = document
                    row = dict(stage=stage, split=split, document=document, sentence=sentence,
                               text=text, external_id=external_id)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
                    counts[split] += 1
            files.append(dict(path=target.name, sha256=digest(target), rows=count))
            stages.append(dict(stage=stage, sentences=count, splits=dict(counts)))
        result = dict(version=1, corpus=recipe['name'], recipe_sha256=digest(recipe_path),
            implementation_sha256=digest(Path(__file__)),
            sources=sources, shards=files, stages=stages,
            documents=documents_seen, sentences=sum(item['rows'] for item in files),
            split_policy='document_mod10_8_1_1', filtering=dict(stats),
            assembly_seconds=time.monotonic() - started)
        temporary = output / 'manifest.json.partial'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.rename(output / 'manifest.json')
        return result
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = build(args.recipe, args.output, root=args.root)
    print(json.dumps({key: result[key] for key in ('documents', 'sentences', 'assembly_seconds')}))


if __name__ == '__main__':
    main()
