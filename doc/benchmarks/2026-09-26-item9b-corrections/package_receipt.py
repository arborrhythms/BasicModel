"""Archive the frozen review source, final checks and unsuccessful attempts."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def compressed(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))


def test_receipt(run, destination, source):
    result = json.loads((run / 'result.json').read_text())
    manifest = json.loads((run / 'source-manifest.json').read_text())
    assert manifest['validated_source'] == source, run
    assert result['exit_code'] == 0, (run, result['reason'])
    assert Counter(result['selected']) == Counter(result['completed'])
    assert len(set(result['selected'])) == len(result['selected'])
    reports = {}
    for path in run.glob('worker-*.json'):
        for report in json.loads(path.read_text()).get('reports', []):
            reports.setdefault(report['nodeid'], []).append(report)
    outcomes = Counter()
    for node in result['selected']:
        found = reports[node]
        assert not any(r['outcome'] in ('failed', 'xpassed') for r in found), node
        calls = [r for r in found if r['phase'] == 'call']
        other = [r for r in found if r['outcome'] in ('skipped', 'xfailed')]
        outcomes[(calls or other)[-1]['outcome']] += 1
    summary = {k: result[k] for k in
               ('elapsed_seconds', 'limits', 'peak_aggregate_memory_bytes', 'exit_code')}
    summary.update(selected=len(result['selected']), completed=len(result['completed']),
        outcomes=dict(outcomes), peak_worker_memory_bytes=max(
            w['peak_memory_bytes'] for w in result['workers']))
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / 'summary.json', summary)
    for name in ('result.json', 'source-manifest.json'):
        compressed(run / name, destination / (name + '.gz'))
    logs = b''.join(p.name.encode() + b'\n' + p.read_bytes()
                    for p in sorted(run.glob('worker-*.log')))
    (destination / 'workers.log.gz').write_bytes(gzip.compress(logs, mtime=0))
    return summary


def main():
    source = source_snapshot(ROOT)
    assert source == json.loads((HERE / 'source-manifest.json').read_text())
    summary = dict(source_sha256=hashlib.sha256(
        json.dumps(source, sort_keys=True).encode()).hexdigest(), checks={})
    final_runs = set()
    for label in ('full', 'slow', 'doc-links'):
        run = ROOT / 'output' / f'item9b-corrections-{label}-release'
        if label == 'doc-links' and not (run / 'result.json').exists():
            continue
        summary['checks'][label] = test_receipt(run, HERE / label, source)
        final_runs.add(run)
    for label in ('measurements', 'address'):
        run = ROOT / 'output' / f'item9b-corrections-{label}-release'
        final_runs.add(run)
        dest = HERE / label
        dest.mkdir(exist_ok=True)
        if label == 'measurements':
            manifest = json.loads((run / 'manifest.json').read_text())
            assert manifest['source'] == source and manifest['source_unchanged']
            assert all(r['exit_code'] == 0 for r in manifest['completed'])
            assert json.loads((run / 'comparison.json').read_text())['parity_demonstrated']
        else:
            data = json.loads((run / 'result.json').read_text())
            assert data['source'] == source and data['source_unchanged']
            assert json.loads((run / 'process.json').read_text())['exit_code'] == 0
        for path in run.iterdir():
            if path.suffix == '.log':
                compressed(path, dest / (path.name + '.gz'))
            elif path.is_file():
                shutil.copyfile(path, dest / path.name)
    diagnostics = HERE / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    index = []
    for run in sorted((ROOT / 'output').glob('item9b-corrections-*')):
        if run in final_runs or not run.is_dir():
            continue
        entry = dict(run=run.name, files=[])
        for name in ('result.json', 'source-manifest.json', 'manifest.json',
                     'baseline.json', 'comparison.json', 'process.json'):
            path = run / name
            if path.exists():
                compressed(path, diagnostics / f'{run.name}-{name}.gz')
                entry['files'].append(f'{run.name}-{name}.gz')
        result_path = run / 'result.json'
        if result_path.exists():
            data = json.loads(result_path.read_text())
            entry.update(exit_code=data.get('exit_code'), reason=data.get('reason'),
                         selected=len(data.get('selected', [])),
                         completed=len(data.get('completed', [])))
            for worker in data.get('workers', []):
                if worker['exit_code'] == 0:
                    continue
                log = run / Path(worker['log']).name
                for path in (log, log.with_suffix('.json')):
                    if path.exists():
                        compressed(path, diagnostics / f'{run.name}-{path.name}.gz')
                        entry['files'].append(f'{run.name}-{path.name}.gz')
        if entry['files']:
            index.append(entry)
    write(diagnostics / 'index.json', index)
    write(HERE / 'validation-summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
