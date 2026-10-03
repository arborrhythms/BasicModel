"""Count selected cases once without discarding pytest's subtest reports."""
from collections import Counter, defaultdict
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sweep = json.loads((HERE / 'full-sweep/receipt.json').read_text())
reports = defaultdict(list)
retries = []
peak = 0
for name in sweep['segments']:
    segment = json.loads(Path(name).read_text())
    retries.extend(segment.get('compile_cache_retries', []))
    peak = max(peak, segment.get('peak_aggregate_memory_bytes', 0))
    for worker in segment['workers']:
        for report in worker.get('reports', []):
            reports[report['nodeid']].append(report)

priority = ('failed', 'error', 'xpassed', 'passed', 'xfailed', 'skipped')
outcomes = {node: next(kind for kind in priority
                       if any(r['outcome'] == kind for r in rows))
            for node, rows in reports.items()}
result = dict(
    selected=sweep['selected'], completed=sweep['completed'],
    selected_case_counts=dict(Counter(outcomes.values())),
    raw_report_counts=sweep['counts'],
    multiple_reports={node: rows for node, rows in reports.items() if len(rows) > 1},
    multiple_report_explanation=(
        'TestOrthogonalFlags.test_flags_match_expected runs three unittest '
        'subTest configurations. Pytest reports those three passes and the '
        'enclosing case pass under the same node ID. They are not reruns.'),
    duration_seconds=sweep['duration_seconds'],
    peak_aggregate_gib=peak / 2**30,
    compile_cache_retries=retries,
    limits=sweep['limits'], source_matched=sweep['source_matched'],
    no_unattempted_cases=not sweep['unattempted'],
    slow_warning=sweep['slow_warning'])
assert len(outcomes) == sweep['selected'] == sweep['completed']
(HERE / 'full-sweep/case-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'multiple_reports'}))
