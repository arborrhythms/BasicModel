"""The training corpus manifest and its held-out list (item 0.5 plan, section 8).

data/training-corpus.json names every stage in order with its files and
hashes, and lists every held-out set in one place. Part A is tracked data;
part B is local data recorded by hash. The held-out cost set of item 2.0
(H11) is the union of part A's validation splits, data/eval/heldout-cost.json.
"""
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def normalize(text):
    return ' '.join(text.lower().split())


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


ASSEMBLY_HASH_RULE = 'sha256 of compact sorted JSON excluding assembly_seconds'


def assembly_manifest_sha256(path):
    """Identify the recipe, implementation, sources and rows independently of build time."""
    manifest = json.loads(Path(path).read_text())
    del manifest['assembly_seconds']
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def units(root):
    """Every part A item as (kind, split, key, sentences), the unit of disjointness."""
    root = Path(root)
    out = []
    lessons = json.loads((root / 'data/lessons/parts-of-speech.json').read_text())
    for stage in lessons['stages']:
        for part, split in (('documents', 'train'), ('heldout', 'heldout')):
            for document in stage[part]:
                for record in document['records']:
                    out.append(('lessons', split, record['key'],
                                tuple(record['sentences']) + ((record['answer'],) if 'answer' in record else ())))
    games = json.loads((root / 'data/games/manifest.json').read_text())
    for entry in games['files']:
        for line in (root / 'data/games' / entry['path']).read_text().splitlines():
            row = json.loads(line)
            out.append(('games', row['split'], row['key'], tuple(row['sources']) + (row['target'],)))
    math = json.loads((root / 'data/math/manifest.json').read_text())
    for grade in math['grades']:
        for line in (root / 'data/math' / grade['file']).read_text().splitlines():
            row = json.loads(line)
            out.append(('math', row['split'], row['document']['key'], tuple(row['document']['sentences'])))
    return out


HELD = ('heldout', 'validation', 'test', 'transfer', 'beyond')
VALIDATION = {'lessons': 'heldout', 'games': 'validation', 'math': 'validation'}


def unit_text(sentences):
    return normalize(' '.join(sentences))


def held_out_sentences(items):
    """Sentences that occur in held-out items and in no part A training item."""
    trained = {normalize(s) for kind, split, key, sentences in items if split not in HELD for s in sentences}
    return sorted({normalize(s) for kind, split, key, sentences in items if split in HELD
                   for s in sentences} - trained)


def scan_assembly(sqlite_path, sentences):
    """How often each held-out-only sentence occurs in the developmental assembly.

    Recorded, not required to be zero: part B is read after part A's gate,
    so an occurrence matters only to a measurement of part A repeated after
    part B.
    """
    wanted = set(sentences)
    counts = {}
    rows = 0
    with closing(sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True)) as connection:
        for (text,) in connection.execute('select text from sentences'):
            rows += 1
            key = normalize(text)
            if key in wanted:
                counts[key] = counts.get(key, 0) + 1
    return rows, counts


def freeze(root, local_root):
    root, local_root = Path(root), Path(local_root)
    assembly = local_root / 'data/corpus/assembly-01'
    shard = local_root / 'data/fineweb/shard_00000.parquet'
    for path in (assembly / 'manifest.json', assembly / 'assembly.sqlite', shard):
        if not path.is_file():
            raise FileNotFoundError(f'{path}: prepare part B before freezing its manifest')
    items = units(root)
    def hashed(path):
        return dict(path=str(path), sha256=sha256_file(root / path))
    validation = [dict(kind=kind, key=key, sha256=hashlib.sha256(unit_text(s).encode()).hexdigest(),
                       sentences=list(s))
                  for kind, split, key, s in items if split == VALIDATION[kind]]
    heldout_cost = dict(schema=1, source='the union of part A validation splits (plan sections 8, 9.1, H11)',
                        items=validation,
                        sha256=hashlib.sha256(json.dumps(validation, sort_keys=True).encode()).hexdigest())
    (root / 'data/eval').mkdir(parents=True, exist_ok=True)
    (root / 'data/eval/heldout-cost.json').write_text(json.dumps(heldout_cost, indent=1, sort_keys=True) + '\n')
    sentences = held_out_sentences(items)
    rows, counts = scan_assembly(assembly / 'assembly.sqlite', sentences)
    scan = dict(sentences_scanned=rows, held_out_sentences=len(sentences),
                held_out_digest=hashlib.sha256('\n'.join(sentences).encode()).hexdigest(),
                occurrences=dict(sorted(counts.items())))
    manifest = dict(
        schema=1,
        order=['A1', 'A2', 'A3', 'B1', 'B2', 'B3'],
        stages=dict(
            A1=dict(name='words only', lessons=['words', 'interpret'],
                    files=[hashed('data/lessons/parts-of-speech.json')]),
            A2=dict(name='simple sentences and synthetics',
                    lessons=['clause', 'ask', 'verb', 'modifiers', 'determiners', 'adverb',
                             'prepositions', 'relations'],
                    files=[hashed('data/lessons/parts-of-speech.json'), hashed('data/games/manifest.json')]),
            A3=dict(name='math', files=[hashed('data/math/manifest.json')]),
            B1=dict(name='developmental text', local=True, path='data/corpus/assembly-01/',
                    recipe=hashed('data/corpus-sources.json'),
                    manifest_hash_rule=ASSEMBLY_HASH_RULE,
                    manifest_sha256=assembly_manifest_sha256(assembly / 'manifest.json')),
            B2=dict(name='NanoChat base corpus', local=True, path='data/fineweb/',
                    shards_on_disk=['shard_00000.parquet'],
                    shard_sha256=sha256_file(shard),
                    split_rule='document index modulo 10: 0 to 7 train, 8 validation, 9 test '
                               '(Data.loadShards; bin/eval_nanochat_grammar.split_name)'),
            B3=dict(name='chat-stage mixture', prepared=False)),
        gate='part B starts only on Alec\'s word after the operator table of A3 (plan section 8)',
        heldout=[
            dict(name='lessons.heldout', source='data/lessons/parts-of-speech.json'),
            dict(name='games.validation, games.test, games.transfer', source='data/games/manifest.json'),
            dict(name='math.validation, math.test, math.beyond', source='data/math/manifest.json'),
            dict(name='next-word gate', source='data/eval/nanochat_grammar_gate.json',
                 sha256=sha256_file(root / 'data/eval/nanochat_grammar_gate.json'),
                 rule='its documents are test documents of shard_00000 under B2\'s split rule'),
            dict(name='held-out cost (H11)', source='data/eval/heldout-cost.json',
                 sha256=heldout_cost['sha256']),
            dict(name='item 8 tasks', built=True, source='data/eval/studies/causal-heldout.json',
                 sha256=sha256_file(root/'data/eval/studies/causal-heldout.json')),
            dict(name='6.5 learning tasks', built=True, source='data/eval/studies/learning-heldout.json',
                 sha256=sha256_file(root/'data/eval/studies/learning-heldout.json')),
        ],
        part_b_scan=scan)
    (root / 'data/training-corpus.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Write data/training-corpus.json.')
    parser.add_argument('command', choices=['freeze'])
    parser.add_argument('--root', default='.')
    parser.add_argument('--local-root', default='.')
    args = parser.parse_args()
    result = freeze(args.root, args.local_root)
    scan = result['part_b_scan']
    print('held-out sentences found in the developmental assembly:',
          None if scan is None else sum(scan['occurrences'].values()), 'of', None if scan is None else scan['held_out_sentences'])
