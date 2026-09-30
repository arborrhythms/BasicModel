"""Audit the completed final sweep without changing its source or results."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded
from review_source import supporting_inputs


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


def main():
    output = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'final')
    result = json.loads((output / 'full/result.json').read_text())
    assert result['reason'] != 'running' and not result.get('active_workers')
    source = json.loads((output / 'full/source-manifest.json').read_text())['validated_source']
    assert source == bounded.source_snapshot(ROOT), 'tested source changed'
    inputs = json.loads((output / 'supporting-inputs.json').read_text())
    assert inputs == supporting_inputs(ROOT), 'fixture inputs changed'
    cases, failures = outcomes(result)
    prior = json.loads((HERE / 'rename/full/result.json').read_text())
    before, _ = outcomes(prior)
    selected, completed = Counter(result['selected']), Counter(result['completed'])
    summary = dict(
        reason=result['reason'], exit_code=result['exit_code'],
        selected=len(result['selected']), completed=len(result['completed']),
        pytest_outcomes=dict(Counter(cases.values())),
        coverage_complete_unique=selected == completed and all(v == 1 for v in completed.values()),
        missing=list((selected-completed).elements()),
        duplicate_completed={n:c for n,c in completed.items() if c != 1},
        elapsed_seconds=result['elapsed_seconds'], limits=result['limits'],
        peak_worker_memory_bytes=max(w['peak_memory_bytes'] for w in result['workers']),
        peak_aggregate_memory_bytes=result['peak_aggregate_memory_bytes'],
        compile_cache_retries=result['compile_cache_retries'],
        source_files=len(source), source_matches_current=True,
        supporting_inputs=inputs,
        source_digest=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        resource_failures=[{k:w.get(k) for k in ('reason','selected','elapsed_seconds','peak_memory_bytes','log')}
                           for w in result['workers'] if w['reason'] not in ('completed','exit')],
        changed_outcomes=[dict(nodeid=n,before=before[n],after=cases[n])
                          for n in sorted(before.keys() & cases.keys()) if before[n] != cases[n]],
        new_cases={n:cases.get(n, 'unreported') for n in sorted(set(result['selected'])-set(prior['selected']))},
        removed_cases={n:before.get(n, 'unreported') for n in sorted(set(prior['selected'])-set(result['selected']))},
        failures=failures,
    )
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in
        ('changed_outcomes','new_cases','removed_cases','failures','limits')},indent=2))


if __name__ == '__main__':
    main()
