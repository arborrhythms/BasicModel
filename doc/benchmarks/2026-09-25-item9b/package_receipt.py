"""Package source-matched final checks and preserve development failures."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot

FIXTURE_FILES = {'test/test_relevance_bases.py', 'test/test_sparse_concept_e2e.py',
                 'test/test_xor_spaces.py'}


def source_delta(measured, current):
    delta = {name: dict(measured=measured.get(name), final=current.get(name))
             for name in sorted(set(measured) | set(current))
             if measured.get(name) != current.get(name)}
    # The accepted final correction changes only three test fixtures. None
    # is selected by the earlier slow/erosion jobs or used by the measurements.
    assert set(delta) <= FIXTURE_FILES, delta
    return delta


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def package(label, run, source):
    result = json.loads((run / 'result.json').read_text())
    manifest = json.loads((run / 'source-manifest.json').read_text())
    measured = manifest['validated_source']
    delta = source_delta(measured, source)
    if delta:
        if label == 'full':
            assert set(delta) == {'test/test_sparse_concept_e2e.py'}
            before = (HERE / 'diagnostics/test_sparse_concept_e2e-before-slow-fixture.py').read_bytes()
            assert hashlib.sha256(before).hexdigest() == measured['test/test_sparse_concept_e2e.py']
            assert before.replace(b'int(acts.shape[0]) == sum(cs0._order_caps())',
                                  b'int(acts.shape[0]) == sum(cs0._field_caps())') == (
                ROOT / 'test/test_sparse_concept_e2e.py').read_bytes()
            target = 'test/test_sparse_concept_e2e.py::test_two_phase_forward_cutover_stamps_terminal_activations'
            reports = [r for worker in result['workers'] for r in worker.get('reports', [])
                       if r['nodeid'] == target]
            assert any(r['outcome'] == 'skipped' for r in reports), reports
        else:
            assert label in ('slow', 'erosion'), label
            assert not any(node.split('::')[0] in delta for node in result['selected'])
    assert result['exit_code'] == 0, (label, result['reason'])
    assert Counter(result['selected']) == Counter(result['completed'])
    assert len(result['selected']) == len(set(result['selected']))
    reports = [r for p in run.glob('worker-*.json')
               for r in json.loads(p.read_text()).get('reports', [])]
    outcomes = Counter()
    for node in result['selected']:
        found = [r for r in reports if r['nodeid'] == node]
        calls = [r for r in found if r['phase'] == 'call']
        other = [r for r in found if r['outcome'] in ('skipped', 'xfailed', 'failed')]
        final = (calls or other)[-1]
        assert not any(r['outcome'] in ('failed', 'xpassed') for r in found), found
        outcomes[final['outcome']] += 1
    summary = {k: result[k] for k in ('reason', 'exit_code', 'elapsed_seconds', 'limits',
                                     'peak_aggregate_memory_bytes', 'compile_cache_retries')}
    summary.update(selected=len(result['selected']), completed=len(result['completed']),
                   outcomes=dict(outcomes), source_files=len(measured), fixture_only_delta=delta,
                   source_sha256=hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest())
    write(HERE / f'{label}-summary.json', summary)
    shutil.copyfile(run / 'source-manifest.json', HERE / f'{label}-source-manifest.json')
    (HERE / f'{label}-result.json.gz').write_bytes(gzip.compress((run / 'result.json').read_bytes(), mtime=0))
    logs = b''.join(p.name.encode() + b'\n' + p.read_bytes() for p in sorted(run.glob('worker-*.log')))
    (HERE / f'{label}-workers.log.gz').write_bytes(gzip.compress(logs, mtime=0))
    return summary


def main(before_doc_links):
    source = source_snapshot(ROOT)
    runs = dict(full='item9b-full-green', affected='item9b-affected-isolated',
                slow='item9b-slow-final', erosion='item9b-native-boundary-final')
    if not before_doc_links:
        runs['doc-links'] = 'item9b-doc-links-final'
    checks = {label: package(label, ROOT / 'output' / run, source) for label, run in runs.items()}
    measurements = ROOT / 'output/item9b-measurements-final'
    manifest = json.loads((measurements / 'manifest.json').read_text())
    measurement_delta = source_delta(manifest['source'], source)
    assert manifest['source_unchanged']
    write(HERE / 'measurement-source-delta.json', measurement_delta)
    assert len(manifest['completed']) == 4
    assert all(r['exit_code'] == 0 for r in manifest['completed'])
    assert json.loads((measurements / 'comparison.json').read_text())['parity_demonstrated']
    current = json.loads((measurements / 'baseline.json').read_text())
    previous = json.loads((HERE.parent / '2026-09-25-item9-bank-sync/measurements/baseline.json').read_text())
    before = {p['name']: p['reconstruction_mean'] for p in previous['phases']}
    after = {p['name']: p['reconstruction_mean'] for p in current['phases']}
    assert before.keys() == after.keys()
    write(HERE / 'baseline-comparison.json', dict(
        preceding_receipt='../2026-09-25-item9-bank-sync/measurements/baseline.json',
        prior_means=before, current_means=after,
        deltas={name: after[name]-before[name] for name in before},
        note='Descriptive fixed-seed comparison; no invented numerical tolerance gate.'))
    for name, run in (('measurements', measurements),
                      ('erosion', ROOT / 'output/item9b-erosion-measurements')):
        dest = HERE / name
        dest.mkdir(exist_ok=True)
        for p in run.iterdir():
            if p.suffix == '.log':
                (dest / (p.name + '.gz')).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
            elif p.is_file():
                shutil.copyfile(p, dest / p.name)
    diagnostics = HERE / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for label in ('full-first', 'full-final', 'affected-final', 'affected-green', 'field-checkpoint', 'final-contracts', 'learning-contracts',
                  'erosion-final', 'serial-measurements', 'serial-corrected', 'serial-stable'):
        run = ROOT / 'output' / ('item9b-' + label)
        for name in ('result.json', 'source-manifest.json', 'manifest.json',
                     'baseline.json', 'comparison.json', 'partial-measurement.json'):
            p = run / name
            if p.exists():
                (diagnostics / (label + '-' + name + '.gz')).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
    write(HERE / 'review-source.json', source)
    previous = json.loads((HERE.parent / '2026-09-25-item9-bank-sync/review-source.json').read_text())
    write(HERE / 'source-delta.json', {name: dict(before=previous.get(name), after=source.get(name))
          for name in sorted(set(source) | set(previous)) if source.get(name) != previous.get(name)})
    with tarfile.open(HERE / 'review-source.tar.gz', 'w:gz') as archive:
        for name in sorted(source):
            info = archive.gettarinfo(ROOT / name, arcname=name)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            with (ROOT / name).open('rb') as stream:
                archive.addfile(info, stream)
    (HERE / 'tracked-source.patch').write_bytes(subprocess.check_output(
        ['git', 'diff', '--', 'bin', 'test', 'data'], cwd=ROOT))
    write(HERE / 'validation-summary.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-doc-links', action='store_true')
    main(parser.parse_args().before_doc_links)
