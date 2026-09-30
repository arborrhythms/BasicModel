"""Record complete outcomes, including regressions and resource failures."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / '2026-09-28-item7-review'
sys.path[:0] = [str(ROOT / 'test'), str(PRIOR)]
from bounded_tests import source_snapshot
from review_source import supporting_inputs
from summarize_final_sweep import outcomes


def main():
    output = HERE / 'full-sweep'
    result_path = output / 'combined-result.json'
    result = json.loads(result_path.read_text())
    assert result['reason'] != 'running' and not result.get('active_workers')
    manifest = json.loads((output / 'source-manifest.json').read_text())
    assert manifest['validated_source'] == source_snapshot(ROOT)
    assert manifest['supporting_inputs'] == supporting_inputs(ROOT)
    gates = json.loads((HERE / 'explicit-final/source-manifest.json').read_text())
    assert gates['validated_source'] == manifest['validated_source']
    assert gates['supporting_inputs'] == manifest['supporting_inputs']
    cases, failures = outcomes(result)
    pytest_counts = dict(Counter(cases.values()))
    resources = result.get('resource_cases', {})
    assert not set(resources).intersection(result['completed'])
    cases.update({n: 'resource_' + r['reason'] for n,r in resources.items()})
    prior = json.loads((PRIOR / 'closing-sweep/full/result.json').read_text())
    before, _ = outcomes(prior)
    selected, completed = Counter(result['selected']), Counter(result['completed'])
    accounted = completed + Counter(resources.keys())
    changes = [dict(nodeid=n, before=before[n], after=cases[n])
               for n in sorted(before.keys() & cases.keys()) if before[n] != cases[n]]
    summary = dict(reason=result['reason'], exit_code=result['exit_code'],
        selected=len(result['selected']), completed=len(result['completed']),
        pytest_outcomes=pytest_counts, resource_outcomes=resources,
        complete_unique_coverage=selected == accounted and all(v == 1 for v in accounted.values()),
        complete_pytest_coverage=selected == completed,
        accounted=len(list(accounted.elements())),
        missing=list((selected-accounted).elements()),
        duplicate_completed={n:c for n,c in completed.items() if c != 1},
        elapsed_seconds=result['elapsed_seconds'], limits=result['limits'],
        peak_worker_memory_bytes=max(w['peak_memory_bytes'] or 0 for w in result['workers']),
        peak_aggregate_memory_bytes=result['peak_aggregate_memory_bytes'],
        compile_cache_retries=result['compile_cache_retries'],
        parts=result['parts'], interrupted_peers=result['interrupted_peers'],
        continuation_protocol='continue_sweep.py: retain every completed outcome and resource failure; resume only unfinished coverage under the original cumulative 10800-second allowance',
        source_files=len(manifest['validated_source']), source_matches_current_and_final_gates=True,
        measurement_equivalence='measurement-source-bridge.json',
        source_digest=hashlib.sha256(json.dumps(manifest['validated_source'], sort_keys=True).encode()).hexdigest(),
        resource_failures=[{k:w.get(k) for k in ('reason','selected','elapsed_seconds','peak_memory_bytes','log')}
                           for w in result['workers'] if w['reason'] not in ('completed','exit','peer_aborted')],
        comparison='Exact selector names against review round 2 incoming receipt; no name normalization.',
        changed_outcome_counts=dict(Counter(r['before'] + ' -> ' + r['after'] for r in changes)),
        changed_outcomes=changes,
        prior_outcomes_of_current_failures=dict(Counter(before.get(f['nodeid'], 'unmatched selector') for f in failures)),
        prior_outcomes_of_resource_cases={n:before.get(n, 'unmatched selector') for n in resources},
        item7_outcomes=dict(Counter(v for n,v in cases.items() if n.startswith('test/test_item7_'))),
        item7_cases={n:v for n,v in cases.items() if n.startswith('test/test_item7_')},
        new_selectors={n:cases.get(n, 'unreported') for n in sorted(set(result['selected'])-set(prior['selected']))},
        removed_selectors={n:before.get(n, 'unreported') for n in sorted(set(prior['selected'])-set(result['selected']))},
        failures=failures)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:summary[k] for k in ('reason','pytest_outcomes','complete_unique_coverage',
        'missing','changed_outcome_counts','prior_outcomes_of_current_failures','item7_outcomes')}, indent=2))


if __name__ == '__main__':
    main()
