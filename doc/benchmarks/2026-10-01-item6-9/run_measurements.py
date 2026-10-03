"""Thirty declared variant measurements and ten final MM measurements, once each."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded

parser = argparse.ArgumentParser()
parser.add_argument('--max-new', type=int, default=40)
args = parser.parse_args()
out = HERE / 'measurements'
out.mkdir(exist_ok=True)
source = bounded.source_snapshot(ROOT)
paths = [HERE / name for name in ('probe_run.py', 'probe_variants.py', 'probe_observation.py', 'deriv69.py', 'measure_mm_grammar.py')]
paths += sorted((HERE / 'probe-patches').rglob('*'))
harness = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in paths if p.is_file()}
manifest = dict(validated_source=source, harness=harness)
manifest_path = out / 'source-manifest.json'
if manifest_path.exists():
    assert json.loads(manifest_path.read_text()) == manifest
else:
    bounded.write_json(manifest_path, manifest)
    bounded.write_json(out / 'plan.json', dict(variants=['a', 'b', 'c'], runs_each=10,
        mm_final_runs=10, epochs=400, seed=None, bar=dict(correct=4, mse_less_than=.05),
        max_workers=3, worker_bytes=8 * bounded.GIB, aggregate_bytes=24 * bounded.GIB,
        timeout_seconds=2150, started_at=time.time()))

jobs = [dict(kind=v, trial=i, name=f'{v}-{i:02}') for i in range(1, 11) for v in ('a', 'b', 'c')]
jobs += [dict(kind='mm', trial=i, name=f'mm-{i:02}') for i in range(1, 11)]
done, pending = [], []
for job in jobs:
    path = out / job['name'] / 'process.json'
    if path.exists():
        process = json.loads(path.read_text())
        assert process['exit_code'] == 0, f'preserve and inspect failed attempt: {path}'
        done.append(dict(**job, process=process))
    else:
        pending.append(job)
active, started_count = [], 0
started = time.monotonic()
prior = out / 'progress.json'
prior = json.loads(prior.read_text()) if prior.exists() else {}
peak = prior.get('peak_memory_bytes', 0)
env = bounded.worker_environment(ROOT)
env.pop('BASIC_SEED', None)
env.update(MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu', BASIC_AUTOLOAD='false',
           RUN_SLOW='1', PYTHONPATH=':'.join(str(p) for p in (ROOT/'bin', ROOT/'test', HERE)))


def verify():
    assert source == bounded.source_snapshot(ROOT), 'candidate changed during measurement'
    assert all(hashlib.sha256((HERE/name).read_bytes()).hexdigest() == value
               for name, value in harness.items()), 'measurement code changed'


def progress():
    bounded.write_json(out / 'progress.json', dict(completed=done,
        active=[dict(**j['job'], pid=j['worker'].proc.pid) for j in active],
        pending=len(pending), session_seconds=time.monotonic()-started,
        peak_memory_bytes=peak, complete=len(done) == len(jobs)))


try:
    while active or (pending and started_count < args.max_new):
        used = 0
        for job in list(active):
            worker = job['worker']
            result = worker.poll()
            used += worker.current_memory_bytes
            if result is None:
                continue
            bounded.write_json(job['path'] / 'process.json', result)
            done.append(dict(**job['job'], process=result))
            active.remove(job)
            verify()
            progress()
            assert result['exit_code'] == 0, f"failed probe: {job['job']['name']}"
            measurement = json.loads((job['path'] / 'measurement.json').read_text())
            if job['job']['kind'] != 'mm':
                assert measurement['training_pairs'] == 400
                print(json.dumps(dict(**job['job'], mse=measurement['mse'],
                    correct=measurement['correct'], settled_bar=measurement['settled_bar'])), flush=True)
            else:
                assert measurement['completed_epochs'] == 900
                print(json.dumps(dict(**job['job'], mse=measurement['ending_training_mse'])), flush=True)
        peak = max(peak, used)
        if used > 24 * bounded.GIB:
            raise RuntimeError('unchanged aggregate memory guard stopped measurements')
        while pending and len(active) < 3 and started_count < args.max_new:
            verify()
            job = pending.pop(0)
            path = out / job['name']
            path.mkdir(exist_ok=False)
            command = ([sys.executable, str(HERE / 'measure_mm_grammar.py'), str(path / 'measurement.json')]
                       if job['kind'] == 'mm' else
                       [sys.executable, str(HERE / 'probe_run.py'), job['kind'], str(path)])
            job_env = dict(env)
            job_env['MODEL_COMPILE'] = 'eager' if job['kind'] == 'mm' else 'none'
            bounded.write_json(path / 'dispatch.json', dict(kind=job['kind'],
                model_compile=job_env['MODEL_COMPILE'], device=job_env['BASICMODEL_DEVICE'],
                seed=job_env.get('BASIC_SEED'), autoload=job_env['BASIC_AUTOLOAD']))
            worker = bounded.GuardedProcess(command, cwd=ROOT, env=job_env,
                log_path=path / 'driver.log', memory_bytes=8 * bounded.GIB, timeout=2150).start()
            active.append(dict(job=job, worker=worker, path=path))
            started_count += 1
            print('START ' + job['name'], flush=True)
        progress()
        time.sleep(.25)
    verify()
    progress()
finally:
    for job in active:
        job['worker'].stop(exit_code=130, reason='campaign_stopped')
        bounded.write_json(job['path'] / 'process.json', job['worker'].result)
