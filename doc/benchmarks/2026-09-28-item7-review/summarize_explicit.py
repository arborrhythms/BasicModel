"""Summarize all explicit gate outcomes, retaining failures and resource limits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from review_source import supporting_inputs
from summarize_final_sweep import outcomes, bounded

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    output = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'explicit-final')
    result = json.loads((output / 'result.json').read_text())
    assert result['reason'] != 'running'
    manifest = json.loads((output / 'source-manifest.json').read_text())
    assert manifest['validated_source'] == bounded.source_snapshot(ROOT)
    assert manifest['supporting_inputs'] == supporting_inputs(ROOT)
    cases, failures = outcomes(result)
    reports = []
    for worker in result['workers']:
        reports.extend(report for report in worker.get('reports', [])
                       if report['phase'] == 'call' or report['outcome'] != 'passed')
    summary = dict(
        reason=result['reason'], selected=len(result['selected']),
        completed=len(result['completed']), pytest_outcomes=dict(Counter(cases.values())),
        resource_limited_cases=result['resource_limited_cases'],
        outcomes=reports, failures=failures,
        groups=result['groups'],
        resources=[{key: worker.get(key) for key in
                    ('reason', 'selected', 'elapsed_seconds', 'peak_memory_bytes', 'log')}
                   for worker in result['workers']],
        source_matches_current=True,
        source_digest=hashlib.sha256(json.dumps(manifest['validated_source'], sort_keys=True).encode()).hexdigest(),
        supporting_inputs=manifest['supporting_inputs'],
        driver_sha256=hashlib.sha256((HERE / 'run_explicit.py').read_bytes()).hexdigest(),
        protocol='One run on the final source; unchanged assertions, thresholds and fixture seed policy. The previous source receipt remains visible.',
        depth3='A missing mature checkpoint is a prerequisite skip. The historical [1, 1, 1, 1] campaign remains red.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in
                     ('reason', 'selected', 'completed', 'pytest_outcomes', 'resource_limited_cases')}, indent=2))


if __name__ == '__main__':
    main()
