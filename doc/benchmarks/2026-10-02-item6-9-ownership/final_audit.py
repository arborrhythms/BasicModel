"""Receipt completeness and source audit; reads results and never runs models."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    source = read(HERE / 'source-final.json')
    full = read(HERE / 'full-sweep/receipt.json', {})
    extra = read(HERE / 'extra-cases/complete.json', {})
    reports = []
    attempts = defaultdict(list)
    stops = defaultdict(list)
    for scope in ('full-sweep', 'extra-cases'):
        for path in sorted((HERE / scope).glob('part-*/result.json')):
            for worker in read(path, {}).get('workers', []):
                if worker.get('phase') == 'collection':
                    continue
                for node in worker.get('completed', []):
                    attempts[node].append(dict(scope=scope, log=worker.get('log')))
                for report in worker.get('reports', []):
                    reports.append(dict(scope=scope, **report))
                node = worker.get('active_case')
                if node and node not in worker.get('completed', []):
                    stops[node].append(dict(scope=scope, reason=worker.get('reason'), log=worker.get('log')))
    for path in sorted((HERE / 'measurements').glob('gate-*/run/result.json')):
        for worker in read(path, {}).get('workers', []):
            reports.extend(dict(scope=path.parents[1].name, **r) for r in worker.get('reports', []))
    native = read(HERE / 'native-driver/process.json', {})
    if native:
        reports.append(dict(scope='native', nodeid='test/test_objective_conflicts_slow.py::test_native_production_objective_measurements',
                            outcome='passed' if native.get('exit_code') == 0 else 'process_failed'))

    by_function = defaultdict(list)
    for report in reports:
        parts = report['nodeid'].split('::')
        by_function[(parts[0], '::'.join(parts[1:]).split('[')[0])].append(report)
    ports = []
    for port in read(HERE / 'test-dispositions.json', []):
        if port['disposition'] not in ('port', 'added'):
            continue
        name = port.get('new_test', port['test'])
        matches = by_function[(port['file'], name)]
        completed = [r for r in matches if r['outcome'] != 'skipped']
        stopped = {node: records for node, records in stops.items()
                   if node.split('::')[0] == port['file'] and node.split('::', 1)[-1].split('[')[0] == name}
        ports.append(dict(file=port['file'], test=name,
                          executed=bool(completed), stopped=stopped,
                          outcomes=dict(Counter(r['outcome'] for r in matches)),
                          cases=sorted({r['nodeid'] for r in matches})))
    attention = [r for r in reports if r['nodeid'].split('::')[0] in (
        'test/test_reading_attention.py', 'test/test_global_attention.py', 'test/test_global_consume.py')
        or 'test_config_builds_runs_and_reconstructs[grammar_reading]' in r['nodeid']]
    mnist = [r for r in reports if 'mnist' in r['nodeid'].lower()]
    full_by_case = defaultdict(list)
    for report in reports:
        if report['scope'] == 'full-sweep':
            full_by_case[report['nodeid']].append(report)
    severity = {'skipped': 0, 'xfailed': 1, 'passed': 2, 'xpassed': 3, 'failed': 4, 'process_failed': 5}
    full_case_counts = dict(Counter(max((r['outcome'] for r in rows), key=severity.__getitem__)
                                    for rows in full_by_case.values()))
    repeated_reports = {node: rows for node, rows in full_by_case.items() if len(rows) > 1}
    case_record = dict(cases=len(full_by_case), case_counts=full_case_counts,
                       raw_report_counts=full.get('counts'), repeated_reports=repeated_reports,
                       rule='one outcome per collected node; failed dominates; subtest reports do not add cases; non-strict XPASS remains separate')
    (HERE / 'full-sweep/case-counts.json').write_text(json.dumps(case_record, indent=2) + '\n')
    # Ordinary sweep skips and explicit weekly execution are distinct scopes.
    duplicate_attempts = {}
    for node, records in attempts.items():
        scope_counts = Counter(r['scope'] for r in records)
        if any(count > 1 for count in scope_counts.values()):
            duplicate_attempts[node] = records
    gates = {}
    for pattern in ('gate-05-trial-*', 'gate-06-trial-*', 'sum-*', 'mm-*'):
        paths = sorted((HERE / 'measurements').glob(pattern))
        gates[pattern] = dict(jobs=len(paths), processes_recorded=sum((p / 'process.json').exists() for p in paths))
    attribution = {}
    for arm in ('R', 'RE', 'RA', 'REA'):
        rows = [read(p) for p in sorted((HERE / 'attribution').glob(arm + '-*/measurement.json'))]
        attribution[arm] = dict(observed=len(rows), class_passes=sum(r['class_bar'] for r in rows),
                                reconstruction_passes=sum(r['reconstruction_bar'] for r in rows))
    mnist_data = {}
    for name, expected in read(HERE / 'mnist-receipt.json', {}).get('data', {}).items():
        path = ROOT / 'data' / name
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        mnist_data[name] = dict(bytes=path.stat().st_size, sha256=digest,
                                matched=digest == expected['sha256'] and path.stat().st_size == expected['bytes'])
    result = dict(
        source_matched=source == bounded_tests.source_snapshot(ROOT),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        checkpoint_changes=subprocess.check_output(['git', 'status', '--porcelain', '--', 'data/*.ckpt', 'test/fixtures/*.pt'], cwd=ROOT, text=True).splitlines(),
        campaign_complete=(HERE / 'campaign-complete.json').exists(),
        gates=gates, attribution=attribution,
        full_sweep={key: full.get(key) for key in ('selected', 'attempted', 'completed', 'complete', 'unattempted', 'duration_seconds', 'counts', 'source_matched', 'limits')},
        full_case_counts=full_case_counts,
        extra_cases=dict(selected=len(extra.get('selected', [])), attempted=len(extra.get('attempted', [])), pending=extra.get('pending'), source_matched=extra.get('source_matched')),
        duplicate_attempts_within_scope=duplicate_attempts,
        port_coverage=ports,
        ports_without_execution_or_stop=[r for r in ports if not r['executed'] and not r['stopped']],
        attention_reports=attention, mnist_reports=mnist, mnist_data=mnist_data,
        completion_is_not_acceptance=True)
    (HERE / 'final-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('source_matched', 'campaign_complete', 'gates', 'attribution', 'extra_cases')}))
    print('ports without execution or stop:', len(result['ports_without_execution_or_stop']))
    print('duplicate attempts within scope:', len(duplicate_attempts))


if __name__ == '__main__':
    main()
