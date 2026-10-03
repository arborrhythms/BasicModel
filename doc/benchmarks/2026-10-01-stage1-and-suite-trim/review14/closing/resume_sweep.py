"""Receipt-only recovery: keep every worker's first attempt, including aborts.

The original continuation lost workers aborted after a peer failed. Its
duplicates remain recorded and never replace first results. Candidate files
and runner limits are unchanged.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def workers_for(part):
    result = read(part / 'result.json', {})
    known = {Path(w['log']).stem: w for w in result.get('workers', [])}
    workers = []
    for request_path in sorted(part.glob('worker-*.request.json')):
        request = read(request_path)
        if request.get('collect'):
            continue
        stem = request_path.name.removesuffix('.request.json')
        progress_path = part / (stem + '.json')
        raw = read(progress_path, {})
        process = read(part / (stem + '.process.json'), known.get(stem))
        if process is None:
            process = dict(log=str(part / (stem + '.log')),
                reason='receipt_repair_interrupt', exit_code=143,
                peak_memory_bytes=None, elapsed_seconds=None,
                missing_process_metadata=True)
        workers.append(dict(process, selected=raw.get('selected', []),
            completed=raw.get('completed', []), reports=raw.get('reports', []),
            active_case=raw.get('active'), device=raw.get('device'),
            recycled=raw.get('recycled', False), raw_progress=str(progress_path),
            omitted_from_original_summary=stem not in known))
    recovered = dict(result, workers=workers)
    bounded.write_json(part / 'recovered.json', recovered)
    return recovered


def collect_receipt():
    folder = HERE / 'full-sweep'
    source = read(HERE / 'measurements/manifest.json')['source']
    parts = sorted(folder.glob('part-*'))
    prior = read(folder / 'receipt.json', {})
    began = prior.get('started', datetime.now(timezone.utc).isoformat())
    selected = read(parts[0] / 'result.json')['selected'] if parts else []
    first, history, reports, completed = {}, defaultdict(list), [], set()
    process_failures, segments, omitted, missing = [], [], [], []
    for part in parts:
        recovered = workers_for(part)
        accounted = []
        for worker in recovered['workers']:
            identity = str(Path(worker['raw_progress']).relative_to(folder))
            done = set(worker['completed'])
            nodes = done | {r['nodeid'] for r in worker['reports']}
            if worker['active_case']:
                nodes.add(worker['active_case'])
            owned = set()
            for node in sorted(nodes):
                history[node].append(dict(worker=identity, completed=node in done,
                    active=node == worker['active_case'], reason=worker['reason']))
                if node not in first:
                    first[node] = identity
                    owned.add(node)
            kept = [dict(r) for r in worker['reports'] if r['nodeid'] in owned]
            for report in kept:
                if report['outcome'] == 'compile_cache_retry':
                    report['outcome'] = report.get('original_outcome', 'failed')
            reports.extend(kept)
            completed.update(done & owned)
            active = worker['active_case']
            if active in owned and active not in done:
                process_failures.append(dict(nodeid=active, phase='process',
                    outcome='process_failed', reason=worker['reason'],
                    exit_code=worker['exit_code'], log=worker['log']))
            if worker['omitted_from_original_summary']:
                omitted.append(identity)
            if worker.get('missing_process_metadata'):
                missing.append(identity)
            accounted.append(dict(worker, reports=kept, completed=sorted(done & owned),
                duplicate_nodes=sorted(nodes - owned)))
        accounted_path = part / 'accounted.json'
        bounded.write_json(accounted_path, dict(recovered, workers=accounted))
        segments.append(str(accounted_path))
    attempted = set(first)
    assert attempted <= set(selected), 'attempted case outside original collection'
    pending = [node for node in selected if node not in attempted]
    duplicates = {node: rows for node, rows in history.items() if len(rows) > 1}
    counts = Counter(r['outcome'] for r in reports)
    counts['process_failed'] += len(process_failures)
    receipt = dict(started=began,
        duration_seconds=(datetime.now(timezone.utc) - datetime.fromisoformat(began)).total_seconds(),
        selected=len(selected), attempted=len(attempted), completed=len(completed),
        complete=bool(selected) and not pending, counts=dict(counts),
        failures=[r for r in reports if r['outcome'] in ('failed', 'xpassed')] + process_failures,
        unattempted=pending, segments=segments,
        source_matched=bounded.source_snapshot(ROOT) == source, source=source,
        baseline=dict(cases=5219, wall_minutes=122),
        limits=dict(worker_gib=8, workers=2, aggregate_gib=16,
            weekly_concurrent=False, worker_seconds=1800, segment_seconds=10800),
        slow_warning=bounded.slow_coverage_warning(ROOT),
        accounting='First attempt retained, including interrupted attempts; duplicates excluded from outcomes.',
        duplicate_cases=len(duplicates),
        extra_attempts=sum(len(rows) - 1 for rows in duplicates.values()),
        restored_worker_records=omitted, missing_process_metadata=missing,
        wall_time_includes_receipt_repair=True)
    bounded.write_json(folder / 'attempt-history.json', dict(history))
    bounded.write_json(folder / 'duplicate-attempts.json', duplicates)
    bounded.write_json(folder / 'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in (
        'selected', 'attempted', 'completed', 'counts', 'duplicate_cases',
        'extra_attempts', 'duration_seconds', 'source_matched')}), flush=True)
    return receipt


def run():
    assert (HERE / 'measurements/complete.json').exists()
    source = read(HERE / 'measurements/manifest.json')['source']
    assert bounded.source_snapshot(ROOT) == source
    folder = HERE / 'full-sweep'
    for name in ('RUN_SLOW', 'RUN_MPS_SLOW', 'BASICMODEL_DEVICE'):
        os.environ.pop(name, None)
    receipt = collect_receipt()
    with bounded.suite_lock(folder / 'candidate.lock'):
        while not receipt['complete']:
            pending = receipt['unattempted']
            attempted = set(read(folder / 'attempt-history.json'))
            assert not attempted.intersection(pending)
            assert bounded.source_snapshot(ROOT) == source
            before = receipt['attempted']
            part = folder / f'part-{len(list(folder.glob("part-*"))):02}'
            result = bounded.run_suite(root=ROOT, selectors=pending, run_dir=part,
                memory_bytes=16 * bounded.GIB, worker_memory_bytes=8 * bounded.GIB,
                workers=2, timeout=1800, suite_timeout=10800,
                batch_size=8, max_files=1, lock_path=folder / 'dispatch.lock')
            receipt = collect_receipt()
            if result['reason'] == 'interrupted' or receipt['attempted'] == before:
                break
    return receipt


if __name__ == '__main__':
    run()
