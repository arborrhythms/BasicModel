"""Archive final validation, exact source identities and every diagnostic failure."""
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


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def compress(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))


def package_run(name, directory, current, *, exact=False):
    run = ROOT / 'output' / directory
    result = json.loads((run / 'result.json').read_text())
    manifest = json.loads((run / 'source-manifest.json').read_text())
    measured = manifest.get('validated_source', manifest.get('source', {}))
    delta = {p: dict(measured=measured.get(p), current=current.get(p))
             for p in sorted(set(measured) | set(current)) if measured.get(p) != current.get(p)}
    if exact:
        assert not delta, (name, delta)
    destination = HERE / name
    for filename in ('result.json', 'source-manifest.json', 'collect.log'):
        compress(run / filename, destination / (filename + '.gz'))
    logs = b'\n'.join(p.name.encode() + b'\n' + p.read_bytes()
                      for p in sorted(run.glob('worker-*.log')))
    (destination / 'workers.log.gz').write_bytes(gzip.compress(logs, mtime=0))
    outcomes, failures = {}, []
    for worker in result['workers']:
        for report in worker.get('reports', []):
            if report['phase'] == 'call' or report['outcome'] in ('failed', 'skipped'):
                strict_xpass = str(report.get('message', '')).startswith('[XPASS(strict)]')
                outcomes[report['nodeid']] = ('strict_xpassed' if strict_xpass else
                    'xfailed' if report.get('wasxfail') else report['outcome'])
            if report['outcome'] == 'failed':
                failures.append({key: report.get(key) for key in
                                 ('nodeid', 'phase', 'message')})
    return dict(exit_code=result['exit_code'], reason=result['reason'],
        selected=len(result['selected']), completed=len(result['completed']),
        unique_completed=len(set(result['completed'])), counts=dict(Counter(outcomes.values())),
        seconds=result.get('elapsed_seconds'), limits=result['limits'], source_delta=delta,
        failures=failures,
        peak_worker_bytes=max((w['peak_memory_bytes'] for w in result['workers']), default=0),
        compile_cache_retries=result.get('compile_cache_retries', []))


def main():
    source = source_snapshot(ROOT)
    summary = {}
    for label, directory in (
        ('diagnostics/red-xor', 'item8-red-xor'),
        ('diagnostics/red-preference', 'item8-red-preference'),
        ('diagnostics/preference-fixed', 'item8-fixed-preference'),
        ('diagnostics/affected-first', 'item8-affected-first'),
        ('diagnostics/affected-correction', 'item8-affected-correction'),
        ('diagnostics/xor-width-fixed', 'item8-xor-w6-fixed'),
        ('diagnostics/collection-path-error', 'item8-affected-final'),
        ('diagnostics/affected-context-error', 'item8-affected-verified'),
        ('diagnostics/context-candidate', 'item8-context-probe'),
        ('affected', 'item8-affected-complete'),
        ('xor-gates', 'item8-xor-final'),
        ('full', 'item8-full')):
        summary[label] = package_run(label, directory, source,
                                     exact=label in ('affected', 'xor-gates', 'full'))
    assert summary['affected']['exit_code'] == 0
    # Archive the one complete sweep even when it is red. Its actual exit,
    # strict XPASS and assertion failures remain visible in the summary.
    assert summary['full']['selected'] == summary['full']['unique_completed']
    if (ROOT / 'output/item8-doc-links/result.json').exists():
        summary['documentation'] = package_run('documentation', 'item8-doc-links',
                                             source, exact=True)
        assert summary['documentation']['exit_code'] == 0
    for name, directory in (('measurements', 'item8-measurements'), ('baseline', 'item8-baseline')):
        run = ROOT / 'output' / directory
        measured = json.loads((run / 'source-manifest.json').read_text())
        delta = {p: dict(measured=measured.get(p), current=source.get(p))
                 for p in sorted(set(measured) | set(source)) if measured.get(p) != source.get(p)}
        # The final correction only ports the archived probe's context. No
        # runtime/config/measurement source may drift across these readings.
        assert set(delta) <= {'test/test_arithmetic_isolation.py'}, delta
        summary[name] = dict(source_delta=delta)
        for p in run.iterdir():
            if p.is_file():
                compress(p, HERE / name / (p.name + '.gz'))
    old = json.loads(gzip.decompress((HERE.parent /
        '2026-09-26-item9b-occurrence-fix/measurements/serial-baseline.json.gz').read_bytes()))
    new = json.loads((ROOT / 'output/item8-baseline/serial-baseline.json').read_text())
    phases = lambda report: {p['name']: p['reconstruction_mean'] for p in report['phases']}
    assert phases(old) == phases(new)
    packed, single = [json.loads((ROOT / f'output/item8-measurements/{layout}.json').read_text())['parity']
                      for layout in ('packed', 'single')]
    parity = {k: packed[k] == single[k] for k in
              ('initial_parameters_sha256', 'initial_dictionary_sha256', 'sentences', 'mean_sentence_byte_cost')}
    assert all(parity.values()), parity
    write(HERE / 'reconstruction-comparison.json', dict(baseline=phases(new), exact=True,
        parity=parity, byte_cost=packed['mean_sentence_byte_cost'],
        training_sentences_per_second=new['phases'][1]['sentences_per_second']))
    seeds = []
    processes = json.loads((ROOT / 'output/item8-measurements/processes.json').read_text())
    for seed in (0, 1, 2):
        path = ROOT / f'output/item8-measurements/xor-{seed}.json'
        seeds.append(json.loads(path.read_text()) if path.exists() else dict(
            seed=seed, updates=None, status='incomplete', process=processes[f'xor-{seed}'],
            learned_utility='unproven'))
    write(HERE / 'xor-summary.json', [{k:v for k,v in r.items() if k not in ('losses', 'readings')} for r in seeds])
    write(HERE / 'source-manifest.json', source)
    summary['source_files'] = len(source)
    summary['source_sha256'] = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    summary['learned_utility'] = 'unproven'
    write(HERE / 'validation-summary.json', summary)
    write(HERE / 'measurement-drivers.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [HERE / 'PROTOCOL.md', *HERE.glob('*.py'),
                    ROOT / 'doc/benchmarks/2026-09-21-item10/probe.py',
                    ROOT / 'doc/benchmarks/2026-09-21-item10/parity.py',
                    ROOT / 'doc/benchmarks/2026-09-26-item9b-corrections/measurements/parity.xml']})
    compress(ROOT / 'doc/benchmarks/2026-09-26-item9b-corrections/measurements/parity.xml',
             HERE / 'measurements/parity.xml.gz')
    with tarfile.open(HERE / 'review-source.tar.gz', 'w:gz') as archive:
        for path in sorted(source):
            info = archive.gettarinfo(ROOT / path, arcname=path)
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ''
            with (ROOT / path).open('rb') as stream:
                archive.addfile(info, stream)
    patch = subprocess.check_output(['git', 'diff', '--', 'bin', 'test', 'data'], cwd=ROOT)
    for path in ('bin/GrammarPreference.py', 'bin/GrammarEvidence.py',
                 'test/test_structural_preference.py', 'test/test_arithmetic_isolation.py'):
        patch += subprocess.run(['git', 'diff', '--no-index', '--', '/dev/null', path],
                                cwd=ROOT, capture_output=True).stdout
    (HERE / 'changes.patch').write_bytes(patch)
    print(json.dumps({k:summary[k] for k in ('affected', 'xor-gates', 'full', 'source_sha256')}))


if __name__ == '__main__':
    main()
