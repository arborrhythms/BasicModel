"""Package final review checks; retain earlier failures as diagnostics."""
from collections import Counter
import ast
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'output/item10-resolution'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def receipt(name):
    run = RUNS / name
    result = json.loads((run / 'result.json').read_text())
    assert result['exit_code'] == 0, (name, result['reason'])
    assert len(set(result['selected'])) == len(result['selected']) == len(set(result['completed']))
    assert set(result['selected']) == set(result['completed'])
    source = json.loads((run / 'source-manifest.json').read_text())['validated_source']
    assert source == source_snapshot(ROOT), name
    reports = []
    for path in sorted(run.glob('worker-*.json')):
        reports.extend(json.loads(path.read_text()).get('reports', []))
    counts = Counter()
    for node in result['selected']:
        matches = [r for r in reports if r['nodeid'] == node]
        calls = [r for r in matches if r['phase'] == 'call']
        exceptions = [r for r in matches if r['outcome'] in ('failed', 'skipped', 'xfailed')]
        final = (calls or exceptions)[-1]
        assert final['outcome'] not in ('failed', 'xpassed'), final
        counts[final['outcome']] += 1
    summary = {key: result[key] for key in ('reason', 'exit_code', 'elapsed_seconds',
               'peak_aggregate_memory_bytes', 'compile_cache_retries', 'limits')}
    summary.update(selected=len(result['selected']), completed=len(result['completed']),
                   outcomes=dict(counts), source_files=len(source),
                   source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest())
    write_json(OUT / f'{name}-summary.json', summary)
    (OUT / f'{name}-result.json.gz').write_bytes(gzip.compress((run / 'result.json').read_bytes(), mtime=0))
    shutil.copyfile(run / 'source-manifest.json', OUT / f'{name}-source-manifest.json')
    shutil.copyfile(RUNS / f'{name}-driver.log', OUT / f'{name}-driver.log')
    logs = b''.join((f'\n{p.name}\n'.encode() + p.read_bytes()) for p in sorted(run.glob('worker-*.log')))
    (OUT / f'{name}-workers.log.gz').write_bytes(gzip.compress(logs, mtime=0))
    return summary


def main():
    summaries = {name: receipt(name) for name in ('affected-final', 'xor-final', 'explicit-final', 'full')}
    assert summaries['xor-final']['outcomes'] == {'passed': 2}
    diagnostics = OUT / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for folder in ('xor-before-slow', 'xor-compile-probe', 'xor-native-fixed', 'affected',
                   'xor-reviewed', 'fixtures', 'config-fixtures', 'checkpoint-fix',
                   'owned-inverse-before', 'owned-inverse-after', 'explicit-memory-limit',
                   'affected-final-before-owned-inverse', 'xor-final-before-owned-inverse',
                   'full-before-owned-inverse'):
        run = RUNS / folder
        for path in run.glob('*'):
            if path.suffix in ('.json', '.log'):
                (diagnostics / f'{folder}-{path.name}.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    for filename in ('xor-native-initial.log', 'xor-native-initial-2.log',
                     'xor-native-train.log', 'xor-native-train-2.log'):
        path = RUNS / filename
        if path.exists():
            (diagnostics / f'{filename}.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    for filename in ('comparison.json', 'manifest.json'):
        path = RUNS / 'serial-before-owned-inverse' / filename
        (diagnostics / f'serial-before-owned-inverse-{filename}.gz').write_bytes(
            gzip.compress(path.read_bytes(), mtime=0))
    serial = RUNS / 'serial'
    manifest = json.loads((serial / 'manifest.json').read_text())
    assert manifest['source_unchanged'] and manifest['source'] == source_snapshot(ROOT)
    assert all(r['exit_code'] == 0 for r in manifest['completed'])
    destination = OUT / 'serial'
    destination.mkdir(exist_ok=True)
    for path in serial.iterdir():
        if path.suffix == '.log':
            (destination / f'{path.name}.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        elif path.is_file():
            shutil.copyfile(path, destination / path.name)
    # The evaluation mode was never committed. Preserve the exact deletion
    # diff against its measured source, as well as the usual HEAD diff.
    source = source_snapshot(ROOT)
    archive = OUT.parent / '2026-09-24-item10/evaluation-source.tar.gz'
    with tarfile.open(archive) as tar:
        previous = {entry.name: tar.extractfile(entry).read()
                    for entry in tar.getmembers() if entry.isfile()}
    changed, patch = {}, []
    for name in sorted(set(previous) | set(source)):
        old = previous.get(name, b'')
        new = (ROOT / name).read_bytes() if name in source else b''
        if old == new:
            continue
        changed[name] = dict(before=hashlib.sha256(old).hexdigest() if name in previous else None,
                             after=source.get(name))
        patch.extend(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                                         fromfile='evaluation/' + name, tofile='review/' + name))
    (OUT / 'review-resolution.patch').write_text(''.join(patch))
    write_json(OUT / 'review-source-delta.json', changed)
    # Preserve every declared unseeded native learning result, not a selected run.
    xor = []
    for path in sorted((RUNS / 'full').glob('worker-*.json')):
        for report in json.loads(path.read_text()).get('reports', []):
            if report.get('phase') != 'call':
                continue
            if not report['nodeid'].startswith(('test/test_grounded_xor.py::test_',
                                               'test/test_concept_output.py::test_native_cli_')):
                continue
            for line in report.get('stdout', '').splitlines():
                if line.startswith("{'pool':") or line.startswith("{'trial':"):
                    xor.append(dict(nodeid=report['nodeid'], **ast.literal_eval(line)))
    assert len(xor) == 9, xor
    write_json(OUT / 'native-xor.json', xor)
    test_path = 'test/test_explicit_dimensions.py'
    original = subprocess.check_output(['git', 'show', f'HEAD:{test_path}'], cwd=ROOT, text=True)
    def assertion_functions(source):
        group = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
                     and node.name == 'TestXorExactCliReconstruction')
        return {node.name: ast.dump(node, include_attributes=False)
                for node in group.body if isinstance(node, ast.FunctionDef)}
    old_assertions = assertion_functions(original)
    current_assertions = assertion_functions((ROOT / test_path).read_text())
    assert old_assertions == current_assertions
    write_json(OUT / 'unchanged-cli-assertions.json', dict(
        source=test_path, comparison='HEAD function bodies and decorators, excluding line positions',
        unchanged=True, functions={name: hashlib.sha256(value.encode()).hexdigest()
                                  for name, value in current_assertions.items()}))
    write_json(OUT / 'receipt-info.json', dict(
        status='uncommitted, unpushed; awaiting code review',
        basicmodel_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        wikioracle_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip(),
        decision='normalized means dropped; mode deleted',
        summaries=summaries))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
