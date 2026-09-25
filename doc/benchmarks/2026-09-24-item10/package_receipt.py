"""Package completed validation without changing outcomes or selecting trials."""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'output/item10-review'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def compressed(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))


def receipt(run_name, label, *, expected_exit):
    run = RUNS / run_name
    data = json.loads((run / 'result.json').read_text())
    assert data['exit_code'] == expected_exit, (run, data['exit_code'])
    selected = data['selected']
    assert len(set(selected)) == len(selected) == len(set(data['completed']))
    assert set(selected) == set(data['completed'])
    reports = []
    for path in sorted(run.glob('worker-*.json')):
        if '.' not in path.stem:
            reports.extend(json.loads(path.read_text()).get('reports', []))
    counts, failures = Counter(), []
    for node in selected:
        matches = [r for r in reports if r['nodeid'] == node]
        called = [r for r in matches if r['phase'] == 'call']
        exceptional = [r for r in matches if r['outcome'] in ('failed', 'skipped', 'xfailed')]
        row = (called or exceptional)[-1]
        counts[row['outcome']] += 1
        failures.extend(r for r in matches if r['outcome'] in ('failed', 'xpassed'))
    if expected_exit == 0:
        assert not failures
    summary = {k: data[k] for k in ('reason', 'exit_code', 'elapsed_seconds',
                'peak_aggregate_memory_bytes', 'limits', 'compile_cache_retries')}
    source = json.loads((run / 'source-manifest.json').read_text())['validated_source']
    summary.update(selected=len(selected), completed=len(data['completed']), outcomes=dict(counts),
                   source_files=len(source),
                   source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                   failures=[dict(nodeid=r['nodeid'], message=r.get('message', '')) for r in failures])
    write_json(OUT / f'{label}-summary.json', summary)
    compressed(run / 'result.json', OUT / f'{label}-result.json.gz')
    shutil.copyfile(run / 'source-manifest.json', OUT / f'{label}-source-manifest.json')
    shutil.copyfile(RUNS / f'{run_name}-driver.log', OUT / f'{label}-driver.log')
    logs = b''.join((f'\n{p.name}\n'.encode() + p.read_bytes()) for p in sorted(run.glob('worker-*.log')))
    (OUT / f'{label}-workers.log.gz').write_bytes(gzip.compress(logs, mtime=0))
    xor = []
    for row in reports:
        if row['phase'] != 'call':
            continue
        if 'test_native_unseeded_xor' in row['nodeid']:
            for line in row.get('stdout', '').splitlines():
                if line.startswith("{'pool':"):
                    xor.append(dict(nodeid=row['nodeid'], **ast.literal_eval(line)))
        if 'test_written_order0_words_are_present_after_one_smoke_epoch' in row['nodeid']:
            (OUT / 'smoke.log').write_text(row['stdout'])
    if xor:
        assert len(xor) == 6 and Counter(x['pool'] for x in xor) == {4: 3, 8: 3}
        assert all(x['initial_mse'] > 0 and x['mse'] == 0 and x['output'] == [0., 1., 1., 0.] for x in xor)
        write_json(OUT / f'xor-{label}.json', sorted(xor, key=lambda x: x['nodeid']))
    return source, summary


def main():
    current = source_snapshot(ROOT)
    source, full = receipt('full-ready', 'full', expected_exit=0)
    assert source == current
    corrected, _ = receipt('fixture-final', 'fixture-corrections', expected_exit=0)
    assert corrected == current
    old, explicit = receipt('explicit', 'explicit', expected_exit=1)
    assert explicit['outcomes'] == {'passed': 27, 'failed': 1}
    assert explicit['failures'][0]['nodeid'].endswith('TestXorExactCliReconstruction::test_output_mse_is_crisp')
    permitted_delta = {'test/test_pi_sigma_inherit_grammarlayer.py',
                       'test/test_relative_sentence_codebook_insertion.py',
                       'test/test_structural_checkpoint.py'}
    deltas = {}
    def record_delta(name, measured):
        changed = {p: dict(measured=measured.get(p), final=current.get(p))
                   for p in set(measured) | set(current) if measured.get(p) != current.get(p)}
        assert set(changed) <= permitted_delta, (name, changed)
        deltas[name] = changed
    record_delta('explicit', old)
    measure = OUT / 'measurements'
    measure.mkdir(exist_ok=True)
    folds = json.loads((RUNS / 'folds-final.json').read_text())
    assert folds['source_unchanged'] and len(folds['xor_runs']) == 64
    assert folds['probe_sha256'] == hashlib.sha256((OUT / 'measure_folds.py').read_bytes()).hexdigest()
    record_delta('folds', folds['source'])
    shutil.copyfile(RUNS / 'folds-final.json', measure / 'folds.json')
    shutil.copyfile(RUNS / 'folds-final.log', measure / 'folds.log')
    for input_name, output_name, script in (
            ('native-ranges', 'native-selected-only', 'measure_native_selected_only.py'),
            ('native-complete', 'native-complete', 'measure_native.py'),
            ('serial-final', 'serial', None),
            ('cli-paired', 'cli', 'measure_cli.py')):
        folder = RUNS / input_name
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['source_unchanged']
        assert all(r['exit_code'] == 0 for r in manifest.get('runs', manifest.get('completed', [])))
        record_delta(output_name, manifest['source'])
        if script:
            digest = manifest.get('probe_sha256', manifest.get('script_sha256'))
            assert digest == hashlib.sha256((OUT / script).read_bytes()).hexdigest()
        dest = measure / output_name
        dest.mkdir(exist_ok=True)
        for path in folder.iterdir():
            if path.suffix == '.log':
                compressed(path, dest / (path.name + '.gz'))
            elif path.is_file():
                shutil.copyfile(path, dest / path.name)
    write_json(OUT / 'validation-source-delta.json', dict(
        explanation='Only the three documented test fixture corrections changed after measurements. Runtime, configuration, and all measured gate source files match the final full receipt.',
        deltas=deltas))
    diagnostics = OUT / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for path, name in (
            (ROOT / 'output/item11c-residue/item10-before.txt', 'normalized-mode-before.txt'),
            (ROOT / 'output/item11c-residue/diagnosis.txt', 'learning-fixture-before.txt'),
            (RUNS / 'ps-support-before.txt', 'ps-support-before.txt'),
            (RUNS / 'support-after.txt', 'ps-support-after.txt'),
            (RUNS / 'affected-final.txt', 'affected.txt'),
            (RUNS / 'xor-baseline-cli.log', 'unseeded-baseline-cli.log')):
        shutil.copyfile(path, diagnostics / name)
    for label, run_name in (('interrupted-support', 'full'),
                            ('interrupted-doc-links', 'full-final'),
                            ('interrupted-test-fixtures', 'full-complete'),
                            ('checkpoint-discovery', 'full-review')):
        run = RUNS / run_name
        compressed(run / 'result.json', diagnostics / f'{label}-result.json.gz')
        shutil.copyfile(RUNS / f'{run_name}-driver.log', diagnostics / f'{label}-driver.log')
        for path in run.glob('worker-*.log'):
            if 'FAILED ' in path.read_text():
                shutil.copyfile(path, diagnostics / f'{label}-{path.name}')
    native_failed = RUNS / 'native-final/current.log'
    if native_failed.exists():
        compressed(native_failed, diagnostics / 'native-instrumentation-before.log.gz')
    # Earlier source revisions are evidence of the work, never extra runs
    # from which to select a favorable final capability result.
    for path, name in (
            (ROOT / 'output/item11c-residue/folds-first.json', 'folds-first-source.json.gz'),
            (RUNS / 'folds.json', 'folds-before-support-fix.json.gz')):
        if path.exists():
            compressed(path, diagnostics / name)
    write_json(OUT / 'receipt-info.json', dict(
        status='uncommitted, unpushed; awaiting code review',
        basicmodel_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        wikioracle_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip(),
        source_sha256=full['source_sha256'], source_files=full['source_files'],
        runtime_source_matches_all_measurements=True,
        inherited_cli_failure_retained=True, recommendation='drop as a production replacement'))
    print(json.dumps(full, indent=2))


if __name__ == '__main__':
    main()
