"""Compare every selected case against the source-matched pre-rename receipt."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot
from rename_vocabulary import substitute


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def main():
    mapping = read(HERE / 'rename/mapping.json')
    before = read(ROOT / mapping['baseline'])
    after = read(HERE / 'rename/full/result.json')
    assert Counter(after['selected']) == Counter(after['completed'])
    assert len(after['completed']) == len(set(after['completed']))
    manifest = read(HERE / 'rename/full/source-manifest.json')['validated_source']
    assert manifest == source_snapshot(ROOT) == mapping['after_source']

    def renamed(node):
        path, *rest = node.split('::')
        path = mapping['file_map'].get(path, path)
        # Parametrized documentation cases include immutable receipt paths.
        # Apply the same path protection as the mechanical source rename.
        rest = [substitute(part, mapping['token_map']) for part in rest]
        return '::'.join((path, *rest))

    def outcomes(result, rename):
        rows = {}
        priority = {'failed': 5, 'xpassed': 4, 'xfailed': 3, 'skipped': 2, 'passed': 1}
        for worker in result['workers']:
            for report in worker.get('reports', ()):
                node = rename(report['nodeid'])
                previous = rows.get(node)
                if previous is None or priority.get(report['outcome'], 9) > priority.get(previous, 9):
                    rows[node] = report['outcome']
        assert set(rows) == {rename(n) for n in result['selected']}
        return rows

    left, right = outcomes(before, renamed), outcomes(after, lambda n: n)
    differences = {node: dict(before=left.get(node), after=right.get(node))
                   for node in sorted(set(left) | set(right)) if left.get(node) != right.get(node)}
    result = dict(before_source_sha256=hashlib.sha256(json.dumps(mapping['before_source'], sort_keys=True).encode()).hexdigest(),
        after_source_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        before_counts=dict(Counter(left.values())), after_counts=dict(Counter(right.values())),
        selected_before=len(left), selected_after=len(right), identical=not differences,
        differences=differences, normalized_outcomes=right,
        normalization='Only documented identifier and file renames; no outcome or threshold normalization.',
        seeds='No seed override or pass-seeking rerun.')
    (HERE / 'rename/comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'normalized_outcomes'}))
    return int(bool(differences))


if __name__ == '__main__':
    raise SystemExit(main())
