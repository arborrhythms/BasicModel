"""Preserve every outcome and package the source submitted for Claude review."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot


def read(path):
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes()))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def delta(measured, current):
    return {p: {'measured': measured.get(p), 'current': current.get(p)}
            for p in sorted(set(measured) | set(current)) if measured.get(p) != current.get(p)}


FIXTURE_PORT = {
    'test/test_detached_reverse_objective.py': {
        'measured': '8a23e95dbb2973a96ab95067d1df7151527d7e816415d44cf934314840690655',
        'current': 'a220b2a4fe129b73a17b5e667734ad04b165b65133602dc4d8036eee21af4e54',
    },
    'test/test_structural_preference.py': {
        'measured': 'be5a8c3de71768b85102c45282f0ecea8bb2dcb7314e77d9a6be652d26d497e5',
        'current': 'afea2543cac11da1f86f02c40faffd5e5ba7be4590664ffaa90b249023336bad',
    }
}


def require_same_runtime(changes):
    # Full sweeps found two synthetic fixtures needing the new trace layout.
    # All production, configuration and measurement inputs stayed identical;
    # both corrected files have their own receipt and the final full run.
    assert all(path in FIXTURE_PORT and values == FIXTURE_PORT[path]
               for path, values in changes.items()), changes


def package_run(run, source):
    result = read(run / 'result.json')
    interrupted_path = run / 'interruption.json'
    interrupted = (read(interrupted_path) if interrupted_path.exists() or
                   interrupted_path.with_suffix('.json.gz').exists() else None)
    if interrupted and result['reason'] == 'running':
        # An interrupted parent can leave an unfinalized snapshot. Preserve
        # its raw bytes and account only for supervisor-collected completions.
        assert interrupted['reason'] == 'supervisor_interrupted'
        result = dict(result, reason=interrupted['reason'],
                      exit_code=interrupted['exit_code'], active_workers=[])
    assert result['reason'] != 'running', f'cannot package a live receipt: {run}'
    assert not result.get('active_workers'), f'cannot package active workers: {run}'
    manifest = read(run / 'source-manifest.json')
    measured = manifest.get('validated_source', manifest.get('source', {}))
    outcomes, failures = {}, []
    for worker in result['workers']:
        for report in worker.get('reports', []):
            if report['phase'] == 'call' or report['outcome'] in ('failed', 'skipped'):
                strict = str(report.get('message', '')).startswith('[XPASS(strict)]')
                outcomes[report['nodeid']] = ('strict_xpassed' if strict else
                    'xfailed' if report.get('wasxfail') else report['outcome'])
            if report['outcome'] == 'failed':
                failures.append({k: report.get(k) for k in ('nodeid', 'phase', 'message')})
    summary = dict(exit_code=result['exit_code'], reason=result['reason'],
        selected=len(result['selected']), completed=len(set(result['completed'])),
        counts=dict(Counter(outcomes.values())), seconds=result.get('elapsed_seconds'),
        limits=result['limits'], source_delta=delta(measured, source), failures=failures,
        peak_worker_bytes=max((w['peak_memory_bytes'] for w in result['workers']), default=0),
        peak_aggregate_memory_bytes=result.get('peak_aggregate_memory_bytes'),
        compile_cache_retries=result.get('compile_cache_retries', []))
    if interrupted:
        summary['interruption'] = interrupted
    # Archive first and verify its membership before compacting our own files.
    # The raw archive retains requests, process accounting, HTML and all logs.
    if (run / 'result.json').exists():
        paths = sorted(p for p in run.iterdir() if p.is_file() and not p.name.endswith('.gz'))
        archive_path = run / 'raw-receipt.tar.gz'
        with tarfile.open(archive_path, 'w:gz') as archive:
            for p in paths:
                archive.add(p, arcname=p.name)
        with tarfile.open(archive_path, 'r:gz') as archive:
            assert set(archive.getnames()) == {p.name for p in paths}
            assert archive.extractfile('result.json').read() == (run / 'result.json').read_bytes()
        for name in ('result.json', 'source-manifest.json', 'collect.log', 'interruption.json'):
            p = run / name
            if p.exists():
                (run / (name + '.gz')).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
        logs = b'\n'.join(p.name.encode() + b'\n' + p.read_bytes()
                          for p in sorted(run.glob('worker-*.log')))
        (run / 'workers.log.gz').write_bytes(gzip.compress(logs, mtime=0))
        for p in paths:
            p.unlink()
    return summary


def main():
    source = source_snapshot(ROOT)
    runs = {p.name: package_run(p, source) for p in sorted(HERE.iterdir())
            if p.is_dir() and ((p / 'result.json').exists() or (p / 'result.json.gz').exists())}
    for name in ('affected-verified', 'trie-verified', 'explicit-final', 'full', 'xor-mm-final'):
        assert name in runs, name
        require_same_runtime(runs[name]['source_delta'])
    assert not runs['full']['source_delta']
    assert runs['layout-fixture-verified']['exit_code'] == 0
    assert not runs['layout-fixture-verified']['source_delta']
    coverage_path = HERE / 'full-coverage.json'
    if coverage_path.exists():
        coverage = read(coverage_path)
        segments = coverage['segments']
        assert segments == ['full', 'full-memory-remainder', 'full-continuation']
        all_completed = []
        full_counts = Counter()
        for name in segments:
            assert not runs[name]['source_delta'], name
            assert runs[name]['limits']['per_worker_memory_bytes'] == 8 * 2**30
            assert runs[name]['limits']['worker_seconds'] == 1800
            all_completed.extend(read(HERE / name / 'result.json')['completed'])
            full_counts.update(runs[name]['counts'])
        selected = read(HERE / 'full/result.json')['selected']
        assert Counter(all_completed) == Counter(selected)
        assert len(set(all_completed)) == len(all_completed)
        assert sum(full_counts.values()) == len(selected)
        assert coverage['segment_exit_codes'] == {
            name: runs[name]['exit_code'] for name in segments}
        assert runs['full']['reason'] == 'memory' and runs['full']['exit_code'] == 137
        assert runs['full-memory-remainder']['exit_code'] == 0
        full_suite = dict(coverage, counts=dict(full_counts),
            reason='complete_coverage_with_memory_stop', exit_code=137,
            seconds=sum(runs[name]['seconds'] for name in segments),
            peak_worker_bytes=max(runs[name]['peak_worker_bytes'] for name in segments),
            failures=[failure for name in segments for failure in runs[name]['failures']])
    else:
        assert runs['full']['selected'] == runs['full']['completed']
        full_suite = runs['full']
    assert runs['affected-verified']['exit_code'] == 0
    assert runs['trie-verified']['exit_code'] == 0
    assert runs['explicit-final']['exit_code'] == 0
    if 'documentation' in runs:
        assert not runs['documentation']['source_delta']
        assert runs['documentation']['exit_code'] == 0
    measurements = HERE / 'measurements'
    measurement_source_delta = delta(read(measurements / 'source-manifest.json'), source)
    require_same_runtime(measurement_source_delta)
    processes = read(measurements / 'processes.json')
    comparison = {'processes': processes, 'source_delta': measurement_source_delta}
    baseline_path = measurements / 'serial-baseline.json'
    if baseline_path.exists():
        baseline = read(baseline_path)
        reviewed = read(HERE.parent / '2026-09-26-item8-review/measurements/serial-baseline.json')
        phases = lambda report: {p['name']: p['reconstruction_mean'] for p in report['phases']}
        comparison['serial'] = dict(reviewed=phases(reviewed), current=phases(baseline))
        comparison['training_sentences_per_second'] = baseline['phases'][1]['sentences_per_second']
    if all((measurements / f'{layout}.json').exists() for layout in ('packed', 'single')):
        packed, single = [read(measurements / f'{layout}.json')['parity'] for layout in ('packed', 'single')]
        fields = ('initial_parameters_sha256', 'initial_dictionary_sha256',
                  'sentences', 'mean_sentence_byte_cost')
        comparison['parity'] = {key: packed[key] == single[key] for key in fields}
        comparison['byte_cost'] = dict(reviewed=.6839025616645813,
            packed=packed['mean_sentence_byte_cost'], single=single['mean_sentence_byte_cost'])
    write(HERE / 'reconstruction-comparison.json', comparison)
    summary = dict(runs=runs, full_suite=full_suite, measurements=comparison, source_files=len(source),
        source_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        test_fixture_port=FIXTURE_PORT,
        learned_utility='unproven', publication='awaiting Claude review; uncommitted')
    write(HERE / 'source-manifest.json', source)
    write(HERE / 'validation-summary.json', summary)
    drivers = [HERE / 'package_receipt.py', HERE / 'run_measurements.py',
        HERE / 'run_review_checks.py', HERE / 'run_full_after_checks.py',
        HERE / 'continue_full.py', HERE / 'full-schedule.json', HERE / 'parity.xml',
        ROOT / 'doc/benchmarks/2026-09-26-item8/measure.py',
        ROOT / 'doc/benchmarks/2026-09-21-item10/probe.py',
        ROOT / 'doc/benchmarks/2026-09-21-item10/parity.py']
    write(HERE / 'receipt-drivers.json', {str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in drivers})
    with tarfile.open(HERE / 'review-source.tar.gz', 'w:gz') as archive:
        for path in sorted(source):
            info = archive.gettarinfo(ROOT / path, arcname=path)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            with (ROOT / path).open('rb') as stream:
                archive.addfile(info, stream)
    patch = subprocess.check_output(['git', 'diff', '--', 'bin', 'test', 'data'], cwd=ROOT)
    for path in ('test/test_compose_operations.py', 'test/test_compose_pair_driver.py',
                 'test/test_compose_records.py'):
        patch += subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', path],
                                cwd=ROOT, capture_output=True).stdout
    (HERE / 'changes.patch').write_bytes(patch)
    print(json.dumps({k: {f: runs[k][f] for f in ('exit_code', 'counts', 'completed')}
                      for k in ('affected-verified', 'explicit-final', 'xor-mm-final', 'full')}))
    print(summary['source_sha256'])
    print(json.dumps({'full_suite': full_suite}))


if __name__ == '__main__':
    main()
