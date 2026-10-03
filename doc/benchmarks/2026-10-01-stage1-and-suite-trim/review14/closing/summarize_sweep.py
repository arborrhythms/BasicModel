"""Build the review summary from the one completed sweep's saved reports."""
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def summarize():
    sweep = read(HERE/'full-sweep/receipt.json')
    reports, processes = [], []
    for path in sweep['segments']:
        result = read(Path(path))
        processes += result['workers']
        reports += [r for w in result['workers'] for r in w['reports']]
    by_case = defaultdict(list)
    for report in reports:
        by_case[report['nodeid']].append(report)
    report_cases = set(by_case)
    for failure in sweep['failures']:
        if failure['phase']=='process':
            by_case[failure['nodeid']].append(failure)
    severity = ('failed', 'process_failed', 'xpassed', 'passed', 'xfailed', 'skipped')
    case_counts = Counter(next(outcome for outcome in severity
        if any(r['outcome']==outcome for r in rows)) for rows in by_case.values())
    case_counts['process_failed'] += 0
    accounting = dict(case_counts=dict(case_counts), report_counts=sweep['counts'],
        unique_cases=len(by_case), report_bearing_cases=len(report_cases),
        process_only_cases=sorted(set(by_case) - report_cases),
        extra_report_events=len(reports)-len(report_cases),
        duplicate_cases=sweep.get('duplicate_cases', 0),
        extra_attempts=sweep.get('extra_attempts', 0),
        multiple_reports={node:rows for node,rows in by_case.items() if len(rows)>1})
    assert sum(case_counts.values())==sweep['attempted']
    (HERE/'case-accounting.json').write_text(json.dumps(accounting,indent=2)+'\n')
    known_peaks = [p['peak_memory_bytes'] for p in processes
                   if p.get('peak_memory_bytes') is not None]
    worker_peak = max(known_peaks, default=0)/1024**3
    missing_process_metadata = [p['log'] for p in processes
        if p.get('peak_memory_bytes') is None or p.get('elapsed_seconds') is None]
    minutes = sweep['duration_seconds']/60
    baseline = sweep['baseline']
    calls = [r for r in reports if r['phase']=='call']
    call_seconds = sum(r['duration'] for r in calls)
    slowest = sorted(calls, key=lambda r:r['duration'], reverse=True)[:25]
    (HERE/'final-runtime-hotspots.json').write_text(json.dumps(dict(
        completed_call_seconds=call_seconds,
        worker_elapsed_seconds=sum(p['elapsed_seconds'] for p in processes
                                   if p.get('elapsed_seconds') is not None),
        missing_process_metadata=missing_process_metadata,
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
        f"Case outcomes: `{dict(case_counts)}`. Source matched: `{sweep['source_matched']}`. "
        f"Largest recorded worker peak: {worker_peak:.3f} GiB. The unchanged worker ceiling is "
        "8 GiB and deadline is 30 minutes. Two workers shared 16 GiB. "
        "The weekly run had finished before this sweep. The reference runner "
        "allowed ten workers, 256 selectors per batch and 16 files per batch; "
        "this run used two workers, eight selectors and one file per batch. "
        "Scheduling differs, so this is an observed wall-time comparison.", '',
        (f"**Receipt continuation incident:** {sweep.get('duplicate_cases', 0)} case IDs "
         f"were inadvertently repeated, producing {sweep.get('extra_attempts', 0)} extra attempts. "
         "The original continuation omitted a peer worker aborted after a resource stop. "
         "Eight completed cases were repeated and two interrupted cases were restarted. "
         "Every case retains its first attempt, including interrupted failures; later outcomes "
         "never replace them. All attempts and logs remain in `attempt-history.json` and "
         "`duplicate-attempts.json`. After the receipt-only repair, only never-started cases "
         "continued. Wall time includes the interruption, duplicate work and bookkeeping repair."
         if sweep.get('duplicate_cases') else
         "Only never-started cases continued after any process stop. Stopped and aborted "
         "active cases remain failures; no measured case was retried by this receipt driver."), '',
        f"Process time and memory metadata are unavailable for {len(missing_process_metadata)} "
        "force-stopped duplicate workers. Their case reports remain preserved; the peak and "
        "worker-time sum use recorded values only. The resource-stop records for canonical "
        "first attempts are preserved.", '',
        'The raw runner counts report events. Multiple subtest reports for a collected '
        'node are one case, with the most severe outcome retained. '
        '`case-accounting.json` preserves both counts and all reports. '
        'The corrected accounting uses the saved reports without further reruns.', '',
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
        '| Current case | Prior sweep seconds | Prior outcome | Trim profile seconds | Final call seconds | Final outcome |',
        '|---|---:|---|---:|---:|---|']
    for case in measured:
        elapsed = '—' if case['final_call_seconds'] is None else f"{case['final_call_seconds']:.3f}"
        outcomes = ', '.join(case['final_outcomes']) or 'no completed report'
        if case['final_process_failures']:
            outcomes += '; '+', '.join(r['reason'] for r in case['final_process_failures'])
        lines.append(f"| `{case['nodeid']}` | {case['before_seconds']:.3f} | "
            f"{case['before_outcome']} | "
            f"{case['after']['call_seconds']:.3f} | {elapsed} | {outcomes} |")
    lines += ['', '## Longest completed calls on final source', '',
        'These are observed call times, without an in-process profiler. They exclude setup, '
        'collection and unfinished calls; process stops remain in the failure '
        'table. The full 25-case list and worker elapsed sum are in '
        '`final-runtime-hotspots.json`.', '',
        '| Case | Call seconds | Outcome |', '|---|---:|---|']
    for case in slowest[:10]:
        lines.append(f"| `{case['nodeid']}` | {case['duration']:.3f} | {case['outcome']} |")
    text = '\n'.join(lines)+'\n'
    path = HERE/'README.md'
    prior = path.read_text().split('\n## Final source-matched sweep')[0]
    path.write_text(prior+text)
    summary = dict(selected=sweep['selected'], completed=sweep['completed'],
        attempted=sweep['attempted'], counts=dict(case_counts), report_counts=sweep['counts'], minutes=minutes,
        source_matched=sweep['source_matched'], worker_peak_gib=worker_peak,
        missing_process_metadata=missing_process_metadata,
        duplicate_cases=sweep.get('duplicate_cases', 0), extra_attempts=sweep.get('extra_attempts', 0),
        wall_time_includes_receipt_repair=sweep.get('wall_time_includes_receipt_repair', False),
        process_failures=[r for r in sweep['failures'] if r['phase']=='process'],
        baseline=baseline, complete=sweep['complete'])
    (HERE/'sweep-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    summarize()
