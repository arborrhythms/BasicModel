"""Summarize retained attempts and verify the final, uncommitted source."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded


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
                    failures.append(dict(nodeid=node, phase=report['phase'],
                        message=report.get('message'), worker_log=worker['log']))
    return cases, failures


def main(last_step, grammar):
    source = read(HERE / 'final-source.json')
    assert source == bounded.source_snapshot(ROOT)
    start = read(HERE / 'starting-record.json')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == start['starting_head']
    preserved = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                 for name in start['preserved']}
    assert preserved == start['preserved']
    original = read(HERE / 'before/candidate/source-manifest.json')['validated_source']
    configs_before = {n: digest for n, digest in original.items() if n.startswith('data/')}
    configs_after = {n: digest for n, digest in source.items() if n.startswith('data/')}
    assert configs_before == configs_after
    stages = {}
    for label in ['before', *(f'step{i}' for i in range(1, last_step + 1))]:
        table = read(HERE / label / 'comparison.json')
        complete = read(HERE / label / 'complete.json')
        stages[label] = dict(elapsed_seconds=complete['elapsed_seconds'],
            peak_aggregate_memory_bytes=complete['peak_aggregate_memory_bytes'], trees={})
        for tree in ('head', 'candidate'):
            rows = table[tree]
            stages[label]['trees'][tree] = dict(named_attempts=len(rows),
                expected_named_attempts=49,
                outcomes=dict(Counter(r['outcome'] for r in rows)),
                exact_roundtrips=dict(Counter(r['outcome'] for r in rows if r['gate'] >= 14)),
                failures=[dict(nodeid=r['nodeid'], gate=r['gate'], outcome=r['outcome'])
                          for r in rows if r['outcome'] != 'passed'])
    final_campaign = HERE / f'step{last_step}' / 'candidate/source-manifest.json'
    assert read(final_campaign)['validated_source'] == source
    assert read(HERE / grammar / 'source-manifest.json')['validated_source'] == source
    gates = read(HERE / grammar / 'summary.json')
    full = read(HERE / 'full-sweep/run/result.json')
    assert full['reason'] != 'running' and not full.get('active_workers')
    assert read(HERE / 'full-sweep/source-manifest.json')['validated_source'] == source
    cases, failures = outcomes(full)
    previous, _ = outcomes(read(HERE.parent / '2026-09-30-item7-review-round5/full-sweep/run/result.json'))
    selected, completed = Counter(full['selected']), Counter(full['completed'])
    sweep = dict(reason=full['reason'], outcomes=dict(Counter(cases.values())),
        selected=len(full['selected']), completed=len(full['completed']),
        complete_unique_coverage=selected == completed and all(v == 1 for v in completed.values()),
        missing=list((selected - completed).elements()),
        duplicate_completed={n: c for n, c in completed.items() if c != 1},
        unreported=sorted(selected.keys() - cases.keys()), failures=failures,
        limits=full['limits'], elapsed_seconds=full['elapsed_seconds'],
        peak_worker_memory_bytes=max(w['peak_memory_bytes'] for w in full['workers']),
        peak_aggregate_memory_bytes=full['peak_aggregate_memory_bytes'],
        resource_stops=[{k: w.get(k) for k in ('reason', 'selected', 'completed', 'log', 'peak_memory_bytes')}
                        for w in full['workers'] if w['reason'] not in ('exit', 'recycle')],
        compile_cache_retries=full['compile_cache_retries'],
        changed_outcomes=[dict(nodeid=n, before=previous[n], after=cases[n])
                          for n in sorted(previous.keys() & cases.keys()) if previous[n] != cases[n]],
        new_selectors={n: cases.get(n, 'unreported') for n in sorted(selected.keys() - previous.keys())},
        removed_selectors=sorted(previous.keys() - selected.keys()), source_matches_current=True)
    bounded.write_json(HERE / 'full-sweep/summary.json', sweep)
    native = {}
    for label in ('native-before', 'native-prepair', 'native-after'):
        process = read(HERE / label / 'driver.process.json')
        measurement = HERE / label / 'measurement.json'
        timing = HERE / label / 'batch-timing.json'
        native[label] = dict(process=process,
            phases=read(measurement)['phases'] if measurement.exists() else [],
            timing=read(timing)['summaries'] if timing.exists() else {})
    assert read(HERE / 'native-after/source-manifest.json')['validated_source'] == source
    result = dict(head=head, preserved_records=preserved,
        configurations_unchanged=True, source_matches_current=True,
        staged_files=subprocess.check_output(['git', 'diff', '--cached', '--name-only'],
                                            cwd=ROOT, text=True).splitlines(),
        stages=stages, native=native,
        grammar=dict(class_successes=gates['class_successes'],
                     all_ten_class_runs_meet_bar=gates['all_ten_class_runs_meet_bar'],
                     decision=gates['decision']),
        full_sweep={k: sweep[k] for k in ('reason', 'outcomes', 'selected', 'completed',
                    'complete_unique_coverage', 'missing', 'unreported', 'resource_stops')})
    bounded.write_json(HERE / 'summary.json', result)
    print(json.dumps({k: result[k] for k in ('head', 'configurations_unchanged', 'grammar', 'full_sweep')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--last-step', type=int, default=5)
    parser.add_argument('--grammar', default='grammar-ten')
    args = parser.parse_args()
    main(args.last_step, args.grammar)
