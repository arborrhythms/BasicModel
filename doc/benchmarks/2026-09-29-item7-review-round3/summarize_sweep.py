"""Summarize the one frozen-source sweep, retaining every red outcome."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R1 = HERE.parent / '2026-09-28-item7-review'
R2 = HERE.parent / '2026-09-28-item7-review-round2'
sys.path[:0] = [str(ROOT / 'test'), str(R1)]
from bounded_tests import source_snapshot
from review_source import supporting_inputs
from summarize_final_sweep import outcomes


def all_outcomes(result):
    cases, failures = outcomes(result)
    cases.update({node: 'resource_' + value['reason']
                  for node, value in result.get('resource_cases', {}).items()})
    return cases, failures


def main():
    out = HERE / 'full-sweep'
    combined = out / 'combined-result.json'
    path = combined if combined.exists() else out / 'run/result.json'
    result = json.loads(path.read_text())
    assert result['reason'] != 'running' and not result.get('active_workers')
    manifest = json.loads((out / 'source-manifest.json').read_text())
    source = source_snapshot(ROOT)
    assert manifest['validated_source'] == source
    assert manifest['supporting_inputs'] == supporting_inputs(ROOT)
    explicit = json.loads((HERE / 'explicit-after-owner/source-manifest.json').read_text())
    assert explicit['validated_source'] == source
    assert explicit['supporting_inputs'] == manifest['supporting_inputs']
    cases, failures = all_outcomes(result)
    pytest_cases, _ = outcomes(result)
    resources = result.get('resource_cases', {})
    selected, completed = Counter(result['selected']), Counter(result['completed'])
    assert not set(resources).intersection(completed)
    accounted = completed + Counter(resources.keys())
    prior = json.loads((R2 / 'full-sweep/combined-result.json').read_text())
    before, _ = all_outcomes(prior)
    changes = [dict(nodeid=node, before=before[node], after=cases[node])
               for node in sorted(before.keys() & cases.keys()) if before[node] != cases[node]]
    regression_nodes = [entry['nodeid'] for entry in json.loads(
        (R2 / 'full-sweep/failure-ledger.json').read_text())['entries']
        if entry['incoming_outcome'] == 'passed' and entry['outcome'] == 'failed']
    diagnostics_path = out / 'unguarded-diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    summary = dict(
        reason=result['reason'], exit_code=result['exit_code'],
        result=str(path.relative_to(HERE)), selected=sum(selected.values()),
        completed=sum(completed.values()), pytest_outcomes=dict(Counter(pytest_cases.values())),
        resource_outcomes=resources, diagnostic_only=diagnostics,
        complete_unique_coverage=selected == accounted and all(n == 1 for n in accounted.values()),
        complete_pytest_coverage=selected == completed,
        missing=list((selected - accounted).elements()),
        duplicate_completed={node: n for node, n in completed.items() if n != 1},
        elapsed_seconds=result['elapsed_seconds'], limits=result['limits'],
        peak_worker_memory_bytes=max((w.get('peak_memory_bytes') or 0 for w in result['workers']), default=0),
        peak_aggregate_memory_bytes=result.get('peak_aggregate_memory_bytes'),
        compile_cache_retries=result.get('compile_cache_retries', []),
        parts=result.get('parts', ['run/result.json']),
        interrupted_peers=result.get('interrupted_peers', []),
        source_files=len(source), source_matches_current_and_final_gates=True,
        source_digest=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        comparison='Exact selector names against the round-2 full sweep; ports and retirements are separately recorded in final-test-ports.json.',
        changed_outcome_counts=dict(Counter(r['before'] + ' -> ' + r['after'] for r in changes)),
        changed_outcomes=changes,
        round2_sixteen_regressions={node: cases.get(node, 'removed selector') for node in regression_nodes},
        item7_outcomes=dict(Counter(value for node, value in cases.items() if node.startswith('test/test_item7_'))),
        item7_cases={node: value for node, value in cases.items() if node.startswith('test/test_item7_')},
        new_selectors={node: cases.get(node, 'unreported') for node in sorted(set(selected) - set(prior['selected']))},
        removed_selectors={node: before.get(node, 'unreported') for node in sorted(set(prior['selected']) - set(selected))},
        failures=failures,
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    entries = [dict(nodeid=f['nodeid'], outcome='failed',
                    round2_outcome=before.get(f['nodeid'], 'unmatched selector'), report=f)
               for f in failures]
    entries += [dict(nodeid=node, outcome='resource_' + value['reason'],
                     round2_outcome=before.get(node, 'unmatched selector'), report=value,
                     diagnostic_only=diagnostics.get(node)) for node, value in resources.items()]
    (out / 'failure-ledger.json').write_text(json.dumps(dict(
        source_digest=summary['source_digest'], entries=entries), indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in (
        'reason', 'selected', 'pytest_outcomes', 'complete_unique_coverage', 'missing',
        'changed_outcome_counts', 'round2_sixteen_regressions', 'item7_outcomes')}, indent=2))


if __name__ == '__main__':
    main()
