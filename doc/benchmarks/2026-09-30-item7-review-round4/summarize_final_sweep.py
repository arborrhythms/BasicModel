"""Audit final-source coverage and compare every outcome with round 3."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / '2026-09-29-item7-review-round3'
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
from bounded_tests import source_snapshot
from review_source import supporting_inputs


def read(path):
    return json.loads(path.read_text())


def outcomes(result):
    cases, failures = {}, []
    for worker in result['workers']:
        for report in worker.get('reports', []):
            if report['phase'] == 'call' or report['outcome'] != 'passed':
                node = report['nodeid']
                if node not in cases or report['outcome'] == 'failed':
                    cases[node] = report['outcome']
                if report['outcome'] == 'failed':
                    failures.append(dict(report, worker_log=worker['log']))
    return cases, failures


def explicit():
    directory = HERE / 'explicit-final3'
    aggregate = read(directory / 'result.json')
    assert aggregate['reason'] != 'running'
    source = source_snapshot(ROOT)
    manifest = read(directory / 'source-manifest.json')
    assert manifest['validated_source'] == source
    assert manifest['supporting_inputs'] == supporting_inputs(ROOT)
    attempts, failures = [], []
    for group in aggregate['groups']:
        result = read(HERE / group['receipt'])
        cases, failed = outcomes(result)
        assert Counter(result['selected']) == Counter(result['completed'])
        assert set(cases) == set(result['selected'])
        attempts.extend(dict(nodeid=node, outcome=cases[node], receipt=group['receipt'])
                        for node in result['selected'])
        failures.extend(dict(report, receipt=group['receipt']) for report in failed)
    summary = dict(reason=aggregate['reason'], attempts=len(attempts),
                   pytest_outcomes=dict(Counter(a['outcome'] for a in attempts)),
                   complete_attempt_coverage=True, failures=failures,
                   source_matches_current=True, source_files=len(source),
                   source_digest=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                   case_attempts=attempts, diagnostic_only=aggregate['diagnostic_only'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Explicit attempts:', summary['attempts'], summary['pytest_outcomes'])


def sweep():
    out = HERE / 'full-sweep'
    result = read(out / 'run/result.json')
    assert result['reason'] != 'running' and not result.get('active_workers')
    source = source_snapshot(ROOT)
    for directory in (out, HERE / 'explicit-final3'):
        manifest = read(directory / 'source-manifest.json')
        assert manifest['validated_source'] == source
        assert manifest['supporting_inputs'] == supporting_inputs(ROOT)
    cases, failures = outcomes(result)
    prior = read(PRIOR / 'full-sweep/run/result.json')
    before, _ = outcomes(prior)
    previous_failures = read(PRIOR / 'full-sweep/failure-ledger.json')['entries']
    selected, completed = Counter(result['selected']), Counter(result['completed'])
    resource_stops = [{k: w.get(k) for k in ('reason', 'selected', 'completed', 'log', 'peak_memory_bytes')}
                      for w in result['workers'] if w['reason'] not in ('completed', 'exit')]
    changes = [dict(nodeid=node, before=before[node], after=cases[node])
               for node in sorted(before.keys() & cases.keys()) if before[node] != cases[node]]
    summary = dict(reason=result['reason'], exit_code=result['exit_code'],
                   selected=len(result['selected']), completed=len(result['completed']),
                   pytest_outcomes=dict(Counter(cases.values())),
                   complete_unique_coverage=selected == completed and all(n == 1 for n in completed.values()),
                   missing=list((selected - completed).elements()),
                   duplicate_completed={n: c for n, c in completed.items() if c != 1},
                   resource_stops=resource_stops, diagnostic_only=read(out / 'diagnostics.json'),
                   elapsed_seconds=result['elapsed_seconds'], limits=result['limits'],
                   peak_worker_memory_bytes=max(w['peak_memory_bytes'] for w in result['workers']),
                   peak_aggregate_memory_bytes=result['peak_aggregate_memory_bytes'],
                   compile_cache_retries=result['compile_cache_retries'],
                   source_files=len(source), source_matches_current_and_explicit_gates=True,
                   source_digest=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                   changed_outcomes=changes, failures=failures,
                   round3_thirteen_failures={r['nodeid']: cases.get(r['nodeid'], 'unreported') for r in previous_failures},
                   new_selectors={n: cases.get(n, 'unreported') for n in sorted(selected.keys() - set(prior['selected']))},
                   removed_selectors={n: before.get(n, 'unreported') for n in sorted(set(prior['selected']) - selected.keys())})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Full sweep: round-3 failure comparison', '',
             '| Previous failing case | Final-source sweep outcome |', '|---|---|']
    for node, outcome in summary['round3_thirteen_failures'].items():
        lines.append(f"| {node.removeprefix('test/')} | {outcome} |")
    lines += ['', '[Complete summary](summary.json), including new failures, resource stops and every outcome change. No unguarded diagnostic is counted as a guarded pass.', '']
    (out / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps({k: summary[k] for k in ('reason', 'selected', 'completed', 'pytest_outcomes', 'complete_unique_coverage', 'missing', 'round3_thirteen_failures')}, indent=2))


if __name__ == '__main__':
    {'explicit': explicit, 'sweep': sweep}[sys.argv[1]]()
