"""Continue uncompleted coverage after a resource stop; never retry an outcome.

Original run and every continuation remain immutable. The cumulative suite
allowance stays 10,800 seconds; workers stay at 1,800 seconds and 8 GiB.
Resource-limited cases stay red; one unguarded diagnostic is recorded apart. Peers interrupted by
the supervisor may resume only cases without a completed outcome.
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'full-sweep'
sys.path[:0] = [str(ROOT / 'test'), str(HERE.parent / '2026-09-28-item7-review')]
import bounded_tests as bounded
from review_source import supporting_inputs


def read_parts():
    paths = [OUT / 'run/result.json'] + sorted(OUT.glob('continuation-*/result.json'))
    parts, workers, resource_cases, interrupted = [], [], {}, []
    for path in paths:
        part = json.loads(path.read_text())
        assert part['reason'] != 'running' and not part.get('active_workers')
        parts.append((path, part))
        accounted = {Path(w['log']).stem: w for w in part['workers']}
        for response in sorted(path.parent.glob('worker-*.json')):
            if '.' in response.stem:
                continue
            data = json.loads(response.read_text())
            name = response.stem
            prior = accounted.get(name)
            worker = dict(prior or {}, selected=data.get('selected', []),
                          completed=data.get('completed', []), reports=data.get('reports', []),
                          log=str(response.with_suffix('.log')), response=str(response))
            if prior is None:
                worker.update(reason='peer_aborted', peak_memory_bytes=None,
                              elapsed_seconds=None, exit_code=data.get('exit_code'))
                interrupted.append(dict(response=str(response), active=data.get('active'),
                                        completed=data.get('completed', [])))
            workers.append(worker)
            if prior and prior['reason'] not in ('exit', 'completed'):
                active = data.get('active')
                if active and active not in data.get('completed', []):
                    resource_cases[active] = dict(reason=prior['reason'],
                        peak_memory_bytes=prior['peak_memory_bytes'], log=prior['log'])
    return parts, workers, resource_cases, interrupted



def diagnose(resource_cases):
    path = OUT / 'unguarded-diagnostics.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    for node, failure in resource_cases.items():
        if failure['reason'] not in ('memory', 'aggregate_memory') or node in records:
            continue
        directory = OUT / f'unguarded-{len(records):02}'
        directory.mkdir()
        started = time.monotonic()
        with (directory/'pytest.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-m', 'pytest', '-q', '--tb=short',
                '-p', 'no:cacheprovider', node], cwd=ROOT, env=os.environ.copy(),
                stdout=log, stderr=log, start_new_session=True)
            tree, peak, reason = bounded.ProcessTree(proc.pid), 0, 'completed'
            while proc.poll() is None:
                peak = max(peak, tree.sample())
                if time.monotonic()-started > 1800:
                    tree.terminate(proc,.5)
                    reason = 'timeout'
                    break
                time.sleep(.1)
        records[node] = dict(diagnostic_only=True, memory_guard=None,
            reason=reason, exit_code=proc.returncode, peak_memory_bytes=peak,
            elapsed_seconds=time.monotonic()-started, log=str(directory/'pytest.log'))
        path.write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(dict(diagnostic=node, **records[node])), flush=True)


def main():
    manifest = json.loads((OUT / 'source-manifest.json').read_text())
    def verify():
        assert manifest['validated_source'] == bounded.source_snapshot(ROOT)
        assert manifest['supporting_inputs'] == supporting_inputs(ROOT)
    verify()
    original = json.loads((OUT / 'run/result.json').read_text())
    schedule = json.loads((OUT / 'schedule.json').read_text())['batches']
    os.environ.pop('BASIC_SEED', None)
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                      BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
    started = time.monotonic()
    allowance = original['limits']['suite_seconds'] - original['elapsed_seconds']
    original_make = bounded.make_batches
    while True:
        parts, workers, resource_cases, interrupted = read_parts()
        completed = [n for w in workers for n in w['completed']]
        assert all(n == 1 for n in Counter(completed).values()), 'completed case repeated'
        accounted = set(completed) | set(resource_cases)
        remaining = [n for n in original['selected'] if n not in accounted]
        budget = allowance - (time.monotonic() - started)
        state = dict(protocol=__doc__, original_selected=original['selected'],
            parts=[str(p.relative_to(OUT)) for p, _ in parts],
            completed=completed, resource_cases=resource_cases,
            interrupted_peers=interrupted, remaining=remaining,
            remaining_seconds=budget, source_unchanged=True)
        (OUT / 'continuation-state.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(dict(pytest_completed=len(completed), resource_cases=resource_cases,
                             remaining=len(remaining), budget_seconds=budget)), flush=True)
        if not remaining or budget <= 0:
            break
        pending = set(remaining)
        def resume_schedule(nodes, batch_size, max_files, devices):
            assert set(nodes) == pending
            batches = [[n for n in batch if n in pending] for batch in schedule]
            batches = [batch for batch in batches if batch]
            assert Counter(n for batch in batches for n in batch) == Counter(nodes)
            return batches
        bounded.make_batches = resume_schedule
        target = OUT / f'continuation-{len(parts):02}'
        verify()
        try:
            result = bounded.run_suite(root=ROOT, selectors=remaining, run_dir=target,
                memory_bytes=24*bounded.GIB, workers=3, worker_memory_bytes=8*bounded.GIB,
                timeout=1800, suite_timeout=budget, batch_size=32, max_files=1)
        finally:
            bounded.make_batches = original_make
        verify()
        if result['reason'] not in ('passed', 'completed', 'test_failure', 'memory', 'timeout',
                                    'aggregate_memory', 'exit'):
            raise RuntimeError(f"Unrecognized continuation stop: {result['reason']}")
    verify()
    parts, workers, resource_cases, interrupted = read_parts()
    gate_elapsed = original['elapsed_seconds'] + time.monotonic() - started
    diagnose(resource_cases)
    verify()
    completed = [n for w in workers for n in w['completed']]
    combined = dict(original, workers=workers, completed=completed,
        elapsed_seconds=gate_elapsed,
        run_dir=str(OUT), reason='resource_failure', exit_code=137,
        peak_aggregate_memory_bytes=max(p.get('peak_aggregate_memory_bytes', 0) for _, p in parts),
        compile_cache_retries=[r for _, p in parts for r in p.get('compile_cache_retries', [])],
        parts=[str(p.relative_to(OUT)) for p, _ in parts], resource_cases=resource_cases,
        interrupted_peers=interrupted, remaining=remaining)
    combined.pop('active_workers', None)
    combined.pop('active_worker', None)
    (OUT / 'combined-result.json').write_text(json.dumps(combined, indent=2) + '\n')
    print(json.dumps(dict(pytest_completed=len(completed), resource_limited=len(resource_cases),
                         remaining=len(remaining))), flush=True)
    return 137


if __name__ == '__main__':
    raise SystemExit(main())
