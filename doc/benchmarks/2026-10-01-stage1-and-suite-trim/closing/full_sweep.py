"""One final source-matched sweep, continuing only cases never started.

Two 8 GiB workers share 16 GiB while the frozen weekly ordinary worker has
8 GiB. A separate lock names this candidate receipt; the weekly lock owns
its immutable copy. Guard failures and aborted active cases are never retried.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'test'))
import bounded_tests as bounded


def run():
    completion = HERE/'measurements/complete.json'
    while not completion.exists():
        time.sleep(5)
    source = json.loads((HERE/'measurements/manifest.json').read_text())['source']
    assert bounded.source_snapshot(ROOT) == source
    folder = HERE/'full-sweep'
    folder.mkdir(exist_ok=False)
    os.environ.pop('RUN_SLOW', None)
    os.environ.pop('RUN_MPS_SLOW', None)
    os.environ.pop('BASICMODEL_DEVICE', None)
    selected, attempted, completed = [], set(), set()
    reports, process_failures, segments = [], [], []
    start = time.monotonic()
    began = datetime.now(timezone.utc).isoformat()
    pending = []
    with bounded.suite_lock(folder/'candidate.lock'):
        while True:
            assert bounded.source_snapshot(ROOT) == source
            part = folder/f'part-{len(segments):02}'
            result = bounded.run_suite(root=ROOT, selectors=pending, run_dir=part,
                memory_bytes=16*bounded.GIB, worker_memory_bytes=8*bounded.GIB,
                workers=2, timeout=1800, suite_timeout=10800,
                batch_size=8, max_files=1, lock_path=folder/'dispatch.lock')
            if not selected:
                selected = result['selected']
            segments.append(str(part/'result.json'))
            before = len(attempted)
            for worker in result['workers']:
                completed.update(worker['completed'])
                attempted.update(worker['completed'])
                reports.extend(worker['reports'])
                raw_path = Path(worker['log']).with_suffix('.json')
                raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
                active = raw.get('active')
                if active and active not in completed:
                    attempted.add(active)
                    process_failures.append(dict(nodeid=active, phase='process',
                        outcome='process_failed', reason=worker['reason'],
                        exit_code=worker['exit_code'], log=worker['log']))
            pending = [node for node in selected if node not in attempted]
            counts = Counter(report['outcome'] for report in reports)
            counts['process_failed'] += len(process_failures)
            receipt = dict(started=began, duration_seconds=time.monotonic()-start,
                selected=len(selected), attempted=len(attempted), completed=len(completed),
                complete=bool(selected) and not pending, counts=dict(counts),
                failures=[r for r in reports if r['outcome'] in ('failed','xpassed')]+process_failures,
                unattempted=pending, segments=segments, source_matched=bounded.source_snapshot(ROOT)==source,
                source=source, baseline=dict(cases=5219, wall_minutes=122),
                limits=dict(worker_gib=8, workers=2, aggregate_gib=16,
                    combined_with_weekly_gib=24, worker_seconds=1800),
                slow_warning=bounded.slow_coverage_warning(ROOT))
            bounded.write_json(folder/'receipt.json', receipt)
            print(json.dumps({k:receipt[k] for k in ('selected','attempted','completed','counts','duration_seconds')}), flush=True)
            if not pending or len(attempted)==before:
                break
    return receipt


if __name__ == '__main__':
    run()
