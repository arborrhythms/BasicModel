"""Build the review summary from the one completed sweep's saved reports."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE.parent


def read(path):
    return json.loads(path.read_text())


def summarize():
    sweep = read(HERE/'full-sweep/receipt.json')
    reports, processes = [], []
    for path in sweep['segments']:
        result = read(Path(path))
        processes += result['workers']
        reports += [r for w in result['workers'] for r in w['reports']]
    worker_peak = max((p['peak_memory_bytes'] for p in processes), default=0)/1024**3
    minutes = sweep['duration_seconds']/60
    baseline = sweep['baseline']
    calls = [r for r in reports if r['phase']=='call']
    call_seconds = sum(r['duration'] for r in calls)
    slowest = sorted(calls, key=lambda r:r['duration'], reverse=True)[:25]
    (HERE/'final-runtime-hotspots.json').write_text(json.dumps(dict(
        completed_call_seconds=call_seconds,
        worker_elapsed_seconds=sum(p['elapsed_seconds'] for p in processes),
        slowest_completed_calls=slowest,
        process_failures=[r for r in sweep['failures'] if r['phase']=='process']),
        indent=2)+'\n')
    measured = []
    for original in read(RECEIPT/'suite-trim/performance-comparison.json'):
        found = [r for r in reports if r['nodeid']==original['nodeid']]
        calls = [r for r in found if r['phase']=='call']
        failed_processes = [r for r in sweep['failures']
            if r['nodeid']==original['nodeid'] and r['phase']=='process']
        measured.append(dict(original,
            final_call_seconds=sum(r['duration'] for r in calls) if calls else None,
            final_outcomes=[r['outcome'] for r in found],
            final_process_failures=failed_processes))
    (HERE/'performance-final.json').write_text(json.dumps(measured,indent=2)+'\n')
    lines = ['', '## Final source-matched sweep', '',
        f"**{sweep['completed']}/{sweep['selected']} cases completed**, "
        f"{sweep['attempted']} attempted; {minutes:.2f} minutes wall time. "
        f"Previous: {baseline['cases']} cases / {baseline['wall_minutes']} minutes. "
        f"Case-count change: {sweep['selected']-baseline['cases']:+d}; "
        f"wall-time change: {minutes-baseline['wall_minutes']:+.2f} minutes.", '',
        f"Outcomes: `{sweep['counts']}`. Source matched: `{sweep['source_matched']}`. "
        f"Largest worker: {worker_peak:.3f} GiB. The unchanged worker ceiling is "
        "8 GiB and deadline is 30 minutes. Two workers shared 16 GiB alongside "
        "the independent weekly ordinary worker's 8 GiB; concurrency differs "
        "from the prior sweep, so this is an observed wall-time comparison.", '',
        'After a process stop, only never-started cases continued on the same source. '
        'Stopped and aborted active cases remain failures; no measured case was retried by this receipt driver.', '',
        f"Weekly coverage warning: {sweep['slow_warning'] or 'none'}.", '',
        '| Failure | Phase | Evidence |', '|---|---|---|']
    for failure in sweep['failures']:
        detail = failure.get('message') or failure.get('reason','')
        detail = detail.replace('|','\\|').replace('\n',' ')[:700]
        lines.append(f"| `{failure['nodeid']}` | {failure['phase']} | {detail} |")
    if not sweep['failures']:
        lines.append('| None | — | — |')
    lines += ['', '## The 25 profiled cases on final source', '',
        'Final values are pytest call times from this sweep, without cProfile. A missing call '
        'is shown as unavailable, with its actual skip/process outcome; the '
        'earlier cProfile figures remain separately preserved.', '',
        '| Current case | Prior sweep seconds | Trim profile seconds | Final call seconds | Final outcome |',
        '|---|---:|---:|---:|---|']
    for case in measured:
        elapsed = '—' if case['final_call_seconds'] is None else f"{case['final_call_seconds']:.3f}"
        outcomes = ', '.join(case['final_outcomes']) or 'no completed report'
        if case['final_process_failures']:
            outcomes += '; '+', '.join(r['reason'] for r in case['final_process_failures'])
        lines.append(f"| `{case['nodeid']}` | {case['before_seconds']:.3f} | "
            f"{case['after']['call_seconds']:.3f} | {elapsed} | {outcomes} |")
    lines += ['', '## Longest completed calls on final source', '',
        'These are observed call times, without an in-process profiler. They exclude setup, '
        'collection and unfinished calls; process stops remain in the failure '
        'table. The full 25-case list and worker elapsed sum are in '
        '`final-runtime-hotspots.json`. Two long cases were also externally '
        'stack-sampled for one second each while running; their elapsed times '
        'include that observation. See `full-sweep/stack-samples/README.md`.', '',
        '| Case | Call seconds | Outcome |', '|---|---:|---|']
    for case in slowest[:10]:
        lines.append(f"| `{case['nodeid']}` | {case['duration']:.3f} | {case['outcome']} |")
    text = '\n'.join(lines)+'\n'
    path = HERE/'README.md'
    prior = path.read_text().split('\n## Final source-matched sweep')[0]
    path.write_text(prior+text)
    summary = dict(selected=sweep['selected'], completed=sweep['completed'],
        attempted=sweep['attempted'], counts=sweep['counts'], minutes=minutes,
        source_matched=sweep['source_matched'], worker_peak_gib=worker_peak,
        baseline=baseline, complete=sweep['complete'])
    (HERE/'sweep-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    summarize()
