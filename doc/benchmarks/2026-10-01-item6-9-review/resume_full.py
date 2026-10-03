"""Continue the original collected sweep after a worker guard stopped dispatch.

Completed cases and the case stopped by its own guard are never rerun.
Sibling cases interrupted by the runner can finish in a fresh worker; every
interrupted attempt stays in the ledger. The original 3-hour suite deadline,
8 GiB worker guard, 24 GiB aggregate guard and 1800-second worker limit remain.
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'full-sweep'
PRIOR = HERE.parent / '2026-09-30-item7-review-round5'
sys.path[:0] = [str(ROOT/'test'), str(PRIOR), str(HERE.parent/'2026-09-28-item7-review')]
import bounded_tests as bounded
from resource_schedule import History, scheduled
from review_source import supporting_inputs

SOURCE = json.loads((HERE/'final-source.json').read_text())
INPUTS = json.loads((HERE/'final-inputs.json').read_text())
START = (OUT/'run/collect.request.json').stat().st_mtime
DEADLINE = START + 10800
paths = [OUT/'run/result.json']
selected = json.loads(paths[0].read_text())['selected']
assert len(selected) == len(set(selected))


def collect():
    results = [json.loads(p.read_text()) for p in paths]
    workers, interruptions = [], []
    for path, result in zip(paths, results):
        workers.extend(result['workers'])
        known = {w['log'] for w in result['workers']}
        for process in path.parent.glob('worker-*.process.json'):
            row = json.loads(process.read_text())
            if row['log'] in known:
                continue
            state_path = process.with_name(process.name.replace('.process.json', '.json'))
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            assert not state.get('completed'), 'An interrupted batch completed cases; account them before resuming.'
            interruptions.append(dict(process=row, selected=state.get('selected', [])))
    completed = [n for r in results for n in r['completed']]
    assert len(completed) == len(set(completed)), 'A completed case was repeated.'
    stopped = {}
    for worker in workers:
        if worker['reason'] not in ('exit', 'recycle'):
            for node in worker['selected']:
                if node not in worker['completed']:
                    stopped[node] = dict(reason=worker['reason'], log=worker['log'],
                        elapsed_seconds=worker['elapsed_seconds'],
                        peak_memory_bytes=worker['peak_memory_bytes'])
    remaining = [n for n in selected if n not in set(completed) and n not in stopped]
    return results, workers, interruptions, completed, stopped, remaining


def verify():
    assert SOURCE == bounded.source_snapshot(ROOT)
    assert INPUTS == supporting_inputs(ROOT)


os.environ.pop('BASIC_SEED', None)
os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                  BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT/'bin'))
verify()
assert not (OUT/'resume-plan.json').exists(), 'Do not silently replace a continuation.'
bounded.write_json(OUT/'resume-plan.json', dict(original_cases=len(selected),
    original_result=str(paths[0]), source=SOURCE, supporting_inputs=INPUTS,
    original_start_unix=START, original_deadline_unix=DEADLINE,
    rationale=__doc__, worker_bytes=8*bounded.GIB, aggregate_bytes=24*bounded.GIB,
    worker_seconds=1800, full_sweep_seconds=10800))

while True:
    results, workers, interruptions, completed, stopped, remaining = collect()
    bounded.write_json(OUT/'continuation-progress.json',dict(
        selected=len(selected), completed=len(completed), stopped=stopped,
        remaining=len(remaining), interrupted_attempts=interruptions,
        elapsed_seconds=time.time()-START, segments=[str(p) for p in paths]))
    if not remaining or time.time() >= DEADLINE:
        break
    number = len(paths)
    segment = OUT/f'remaining-{number:02}'
    segment.mkdir(exist_ok=False)
    bounded.write_json(segment/'selectors.json', remaining)
    print(json.dumps(dict(segment=number, cases=len(remaining),
        previously_completed=len(completed), retained_guard_stops=len(stopped),
        interrupted_attempts=len(interruptions))), flush=True)
    verify()
    with scheduled(bounded, history=History([
            PRIOR/'full-sweep/run/result.json',
            HERE.parent/'2026-10-01-item6-9-continuation/full-sweep/run/result.json', *paths]),
            budget=24*bounded.GIB, schedule_path=segment/'schedule.json'):
        result = bounded.run_suite(root=ROOT, selectors=remaining, run_dir=segment/'run',
            memory_bytes=24*bounded.GIB, workers=10, worker_memory_bytes=8*bounded.GIB,
            timeout=1800, suite_timeout=max(1, int(DEADLINE-time.time())),
            batch_size=256, max_files=16)
    verify()
    paths.append(segment/'run/result.json')
    assert set(result['selected']) == set(remaining), 'Continuation changed the original collection.'
    if result['reason'] not in ('passed','test_failure','timeout','memory','aggregate_memory'):
        break

results, workers, interruptions, completed, stopped, remaining = collect()
combined = dict(results[0], workers=workers, completed=completed,
    elapsed_seconds=time.time()-START,
    peak_aggregate_memory_bytes=max(r['peak_aggregate_memory_bytes'] for r in results),
    compile_cache_retries=[e for r in results for e in r['compile_cache_retries']],
    reason='resource_stops' if stopped or remaining else
           'test_failure' if any(r['reason']=='test_failure' for r in results) else 'passed',
    exit_code=124 if stopped or remaining else int(any(r['reason']=='test_failure' for r in results)),
    segments=[str(p) for p in paths], interrupted_attempts=interruptions,
    stopped_cases=stopped, unexecuted_cases=remaining)
bounded.write_json(OUT/'combined-result.json',combined)
bounded.write_json(OUT/'combined-coverage.json',dict(reason=combined['reason'],
    selected=len(selected), completed=len(completed),
    missing=[n for n in selected if n not in set(completed)],
    stopped_cases=stopped, unexecuted=remaining, interrupted_attempts=interruptions,
    duplicate_completed={n:c for n,c in Counter(completed).items() if c!=1},
    resource_stops=[dict(reason=w['reason'],log=w['log']) for w in workers if w['reason'] not in ('exit','recycle')],
    source_matched=True, supporting_inputs_matched=True))
verify()
print(json.dumps(dict(reason=combined['reason'],completed=len(completed),selected=len(selected),
    stopped=len(stopped),unexecuted=len(remaining),seconds=combined['elapsed_seconds'])),flush=True)
raise SystemExit(combined['exit_code'])
