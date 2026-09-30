"""Remeasure the repaired candidate; retain the unchanged HEAD controls.

All eight predeclared reconstruction seeds or all ten fresh 900-epoch MM
runs are included. The original candidate measurements remain in their
original directories. Guards, drivers, budgets and configuration are unchanged.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_guarded, source_snapshot, ProcessTree

mode = sys.argv[1]
assert mode in ('reconstruction', 'mm-grammar')
out = HERE / ('candidate-' + mode + '-final')
out.mkdir(exist_ok=False)
source = source_snapshot(ROOT)
driver = HERE / ('measure_reconstruction.py' if mode == 'reconstruction' else 'measure_mm_grammar.py')
manifest = source if mode == 'reconstruction' else dict(
    validated_source=source, driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest())
(out / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
env = os.environ.copy()
env.pop('BASIC_SEED', None)
env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ('bin', 'test')))
timeout = 1200 if mode == 'reconstruction' else 1800


def command(index, diagnostic=False):
    stem = f'seed-{index}' if mode == 'reconstruction' else f'run-{index:02}'
    stem += '-diagnostic' if diagnostic else ''
    args = [sys.executable, str(driver)]
    if mode == 'reconstruction':
        args += ['--revision', 'item7-round3-after-owner-repair', '--seed', str(index), '--out']
    return args + [str(out / (stem + '.json'))], out / (stem + '.log')


def run(index):
    args, log = command(index)
    result = run_guarded(args, cwd=ROOT, env=env, log_path=log,
                         memory_bytes=8 * GIB, timeout=timeout)
    assert source_snapshot(ROOT) == source
    return index, result


processes = {}
with ProcessPoolExecutor(max_workers=3, mp_context=get_context('fork')) as pool:
    jobs = [pool.submit(run, i) for i in range(8 if mode == 'reconstruction' else 10)]
    for job in as_completed(jobs):
        index, result = job.result()
        processes[str(index)] = result
        (out / 'processes.json').write_text(json.dumps(processes, indent=2) + '\n')
        print(mode, index, result['reason'], result['exit_code'], flush=True)

for index, result in processes.items():
    if result['reason'] not in ('memory', 'aggregate_memory'):
        continue
    args, log_path = command(int(index), diagnostic=True)
    started = time.monotonic()
    with log_path.open('w') as log:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=log,
                                   start_new_session=True)
        tree, peak, reason = ProcessTree(process.pid), 0, 'completed'
        while process.poll() is None:
            peak = max(peak, tree.sample())
            if time.monotonic() - started > timeout:
                tree.terminate(process, .5)
                reason = 'timeout'
                break
            time.sleep(.1)
    result['diagnostic_only'] = dict(exit_code=process.returncode, peak_memory_bytes=peak,
                                    memory_guard=None, elapsed_seconds=time.monotonic() - started,
                                    reason=reason)
    assert source_snapshot(ROOT) == source
    (out / 'processes.json').write_text(json.dumps(processes, indent=2) + '\n')
(out / 'driver-hashes.json').write_text(json.dumps({
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), driver)
}, indent=2) + '\n')
