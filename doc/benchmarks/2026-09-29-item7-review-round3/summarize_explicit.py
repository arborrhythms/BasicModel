"""Summarize explicit gates without converting diagnostics into gate outcomes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
from bounded_tests import source_snapshot
from review_source import supporting_inputs
from summarize_final_sweep import outcomes


def main():
    directory = HERE / 'explicit-final'
    result = json.loads((directory / 'result.json').read_text())
    assert result['reason'] != 'running'
    manifest = json.loads((directory / 'source-manifest.json').read_text())
    source = source_snapshot(ROOT)
    assert manifest['validated_source'] == source
    assert manifest['supporting_inputs'] == supporting_inputs(ROOT)
    cases, failures = outcomes(result)
    resources = {}
    for worker in result['workers']:
        if worker['reason'] in ('completed', 'exit'):
            continue
        progress = Path(worker['log']).with_suffix('.json')
        data = json.loads(progress.read_text()) if progress.exists() else {}
        node = data.get('active')
        if node and node not in worker.get('completed', []):
            resources[node] = {key: worker.get(key) for key in
                               ('reason', 'peak_memory_bytes', 'elapsed_seconds', 'log')}
    selected, completed = Counter(result['selected']), Counter(result['completed'])
    accounted = completed + Counter(resources.keys())
    summary = dict(
        reason=result['reason'], exit_code=result['exit_code'],
        selected=sum(selected.values()), completed=sum(completed.values()),
        pytest_outcomes=dict(Counter(cases.values())), resource_outcomes=resources,
        complete_unique_coverage=selected == accounted and all(v == 1 for v in accounted.values()),
        missing=list((selected - accounted).elements()),
        duplicate_completed={node: count for node, count in completed.items() if count != 1},
        failures=failures, diagnostic_only=result['diagnostic_only'],
        source_matches_current=True, source_files=len(source),
        source_digest=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        item7_outcomes=dict(Counter(value for node, value in cases.items()
                                   if node.startswith('test/test_item7_'))),
        groups=result['groups'],
        peak_worker_memory_bytes=max((w.get('peak_memory_bytes') or 0
                                      for w in result['workers']), default=0),
    )
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in
                     ('reason', 'selected', 'pytest_outcomes', 'resource_outcomes',
                      'complete_unique_coverage', 'missing', 'item7_outcomes')}, indent=2))


if __name__ == '__main__':
    main()
