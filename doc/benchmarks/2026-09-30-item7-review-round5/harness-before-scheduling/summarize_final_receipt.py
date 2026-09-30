"""Read durable round-5 results; preserve guarded outcomes and diagnostics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / '2026-09-30-item7-review-round4'
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
from bounded_tests import source_snapshot
from review_source import supporting_inputs


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def verify(directory):
    source = read(HERE / 'final-source.json')
    inputs = read(HERE / 'final-inputs.json')
    assert source_snapshot(ROOT) == source
    assert supporting_inputs(ROOT) == inputs
    manifest = read(directory / 'source-manifest.json')
    assert manifest['validated_source'] == source, directory
    if 'supporting_inputs' in manifest:
        assert manifest['supporting_inputs'] == inputs


def outcomes(result):
    cases, failures = {}, []
    for worker in result['workers']:
        for report in worker.get('reports', []):
            if report['phase'] == 'call' or report['outcome'] != 'passed':
                node = report['nodeid']
                if node not in cases or report['outcome'] == 'failed':
                    cases[node] = report['outcome']
                if report['outcome'] == 'failed':
                    failures.append(dict(nodeid=node, phase=report['phase'],
                        message=report.get('message'), worker_log=worker['log']))
    return cases, failures


def gates():
    result = dict(groups={}, case_attempts=[], failures=[], diagnostic_only=[])
    for label in ('final-item7', 'final-thought-reasoning', 'final-graph-release', 'final-xor-candidate'):
        directory = HERE / label
        verify(directory)
        aggregate = read(directory / 'result.json')
        assert aggregate['reason'] != 'running'
        attempts, missing, failures = [], [], []
        for group in aggregate['groups']:
            path = directory / group['receipt']
            run = read(path)
            cases, failed = outcomes(run)
            missing.extend((Counter(run['selected']) - Counter(run['completed'])).elements())
            attempts.extend(dict(nodeid=node, outcome=cases.get(node, 'unreported'),
                                 receipt=str(path.relative_to(HERE))) for node in run['selected'])
            failures.extend(failed)
        summary = dict(reason=aggregate['reason'], attempts=len(attempts),
                       pytest_outcomes=dict(Counter(a['outcome'] for a in attempts)),
                       missing=missing, failures=failures,
                       diagnostic_only=aggregate['diagnostic_only'])
        write(directory / 'case-summary.json', summary)
        result['groups'][label] = summary
        result['case_attempts'].extend(attempts)
        result['failures'].extend(failures)
        result['diagnostic_only'].extend(aggregate['diagnostic_only'])
    result['pytest_outcomes'] = dict(Counter(a['outcome'] for a in result['case_attempts']))
    result['attempts'] = len(result['case_attempts'])
    result['source_matches_current'] = True
    write(HERE / 'final-explicit-summary.json', result)
    print(json.dumps({k: v['pytest_outcomes'] for k, v in result['groups'].items()}, indent=2))


def xor():
    before = read(PRIOR / 'final-xor-head/summary.json')
    after = read(HERE / 'xor-candidate/summary.json')
    verify(HERE / 'final-xor-candidate')
    rows = []
    trial = 0
    for head, candidate in zip(before['groups'], after['groups'], strict=True):
        assert head['selector'] == candidate['selector']
        exact = head['selector'].endswith('::test_mm20m_xor_exact_roundtrip')
        trial += int(exact)
        assert head['selected'] == candidate['selected']
        for node in head['selected']:
            rows.append(dict(proof=node, exact_trial=trial if exact else None,
                head=head['outcomes'].get(node, head['reason']),
                candidate=candidate['outcomes'].get(node, candidate['reason']),
                candidate_peak_gib=candidate['peak_memory_bytes'] / 2**30,
                observations=[v for v in candidate['observations'] if v.get('nodeid') in (None, node)],
                diagnostic_only=candidate['diagnostic_only']))
    assert trial == 15
    write(HERE / 'final-xor-comparison.json', dict(head_reused_from=str(PRIOR / 'final-xor-head'), rows=rows))
    lines = ['# XOR: every named proof, after AK', '',
             'HEAD is the unchanged round-4 receipt; candidate is this frozen source.', '',
             '| Proof | HEAD | Candidate | Candidate peak GiB |', '|---|---|---|---|']
    for row in rows:
        name = row['proof'].removeprefix('test/')
        if row['exact_trial']:
            name += f" — trial {row['exact_trial']}/15"
        lines.append(f"| {name} | {row['head']} | {row['candidate']} | {row['candidate_peak_gib']:.2f} |")
    lines += ['', '[All candidate measurements and diagnostics](final-xor-candidate/table.md).',
              '[HEAD measurements](../2026-09-30-item7-review-round4/final-xor-head/table.md).', '']
    (HERE / 'final-xor-comparison.md').write_text('\n'.join(lines))
    print('XOR comparison:', len(rows), 'attempts per tree;', trial, 'exact repetitions.')


def mm():
    verify(HERE / 'final-mm-grammar-candidate')
    reports = {}
    driver_sha = hashlib.sha256((HERE / 'measure_mm_grammar.py').read_bytes()).hexdigest()
    for label, directory in (('HEAD', PRIOR / 'final-mm-grammar-head'),
                             ('before AK', PRIOR / 'final3-mm-grammar-candidate'),
                             ('candidate', HERE / 'final-mm-grammar-candidate')):
        manifest = read(directory / 'source-manifest.json')
        assert manifest['driver_sha256'] == driver_sha
        processes = read(directory / 'processes.json')
        assert set(processes) == set(map(str, range(10)))
        rows = []
        for trial in range(10):
            process = processes[str(trial)]
            measurement = read(directory / f'run-{trial:02}.json')
            rows.append(dict(trial=trial, process=process, measurement=measurement,
                completed_under_guard=process['reason'] == 'exit' and process['exit_code'] == 0
                    and measurement['completed_epochs'] == 900))
        reports[label] = dict(rows=rows, completed_under_guard=sum(r['completed_under_guard'] for r in rows),
            median_ending_mse=statistics.median(r['measurement']['ending_training_mse'] for r in rows))
    write(HERE / 'final-mm-grammar-summary.json', reports)
    lines = ['# MM_grammar: ten full 900-epoch runs', '',
             'Unseeded measurements, unchanged configuration and 8 GiB guard. No threshold or baseline moves.', '',
             '| Trial | HEAD ending MSE | Before AK ending MSE | Candidate ending MSE | Candidate peak GiB |',
             '|---|---|---|---|---|']
    for trial in range(10):
        values = [reports[label]['rows'][trial]['measurement']['ending_training_mse'] for label in reports]
        peak = reports['candidate']['rows'][trial]['process']['peak_memory_bytes'] / 2**30
        lines.append(f'| {trial + 1} | {values[0]:.10f} | {values[1]:.10f} | {values[2]:.10f} | {peak:.2f} |')
    for label, result in reports.items():
        lines += ['', f"{label}: {result['completed_under_guard']}/10 complete under the guard; median ending MSE {result['median_ending_mse']:.10f}."]
    lines += ['', '[All processes and measurements](final-mm-grammar-summary.json).', '']
    (HERE / 'final-mm-grammar-table.md').write_text('\n'.join(lines))
    print({label: {k: v for k, v in report.items() if k != 'rows'} for label, report in reports.items()})


def sweep():
    directory = HERE / 'full-sweep'
    verify(directory)
    result = read(directory / 'run/result.json')
    assert result['reason'] != 'running' and not result.get('active_workers')
    cases, failures = outcomes(result)
    previous, _ = outcomes(read(PRIOR / 'full-sweep/run/result.json'))
    selected, completed = Counter(result['selected']), Counter(result['completed'])
    prior13 = read(PRIOR / 'full-sweep/summary.json')['round3_thirteen_failures']
    prior3 = [node for node, outcome in previous.items() if outcome == 'failed']
    summary = dict(reason=result['reason'], pytest_outcomes=dict(Counter(cases.values())),
        selected=len(result['selected']), completed=len(result['completed']),
        complete_unique_coverage=selected == completed and all(v == 1 for v in completed.values()),
        missing=list((selected - completed).elements()),
        duplicate_completed={node: count for node, count in completed.items() if count != 1},
        unreported=sorted(selected.keys() - cases.keys()),
        failures=failures, limits=result['limits'], elapsed_seconds=result['elapsed_seconds'],
        peak_worker_memory_bytes=max(w['peak_memory_bytes'] for w in result['workers']),
        peak_aggregate_memory_bytes=result['peak_aggregate_memory_bytes'],
        resource_stops=[{k: w.get(k) for k in ('reason', 'selected', 'completed', 'log', 'peak_memory_bytes')}
                        for w in result['workers'] if w['reason'] not in ('completed', 'exit')],
        diagnostic_only=read(directory / 'diagnostics.json'), compile_cache_retries=result['compile_cache_retries'],
        round4_three_failures={node: cases.get(node, 'unreported') for node in prior3},
        round3_thirteen_failures={node: cases.get(node, 'unreported') for node in prior13},
        changed_outcomes=[dict(nodeid=node, before=previous[node], after=cases[node])
                          for node in sorted(previous.keys() & cases.keys()) if previous[node] != cases[node]],
        new_selectors={node: cases.get(node, 'unreported') for node in sorted(cases.keys() - previous.keys())},
        removed_selectors=sorted(previous.keys() - selected.keys()), source_matches_current=True)
    write(directory / 'summary.json', summary)
    lines = ['# Full sweep: previous failures', '', '| Previous failing case | Round 5 outcome |', '|---|---|']
    for node, outcome in (summary['round4_three_failures'] | summary['round3_thirteen_failures']).items():
        lines.append(f"| {node.removeprefix('test/')} | {outcome} |")
    lines += ['', '[Complete summary](summary.json), including every changed outcome and coverage gap.', '']
    (directory / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps({k: summary[k] for k in ('reason', 'pytest_outcomes', 'selected', 'completed',
        'complete_unique_coverage', 'missing', 'round4_three_failures', 'round3_thirteen_failures')}, indent=2))


if __name__ == '__main__':
    {'gates': gates, 'xor': xor, 'mm': mm, 'sweep': sweep}[sys.argv[1]]()
