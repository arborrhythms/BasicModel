"""Ten paired, entropy-initialized trainings in each declared condition."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / 'test'), str(HERE)]
import bounded_tests as bounded


def main():
    from verification import validate
    source = bounded.source_snapshot(ROOT)
    validate(source)
    helpers = json.loads((HERE / 'measured-source/measurement-helpers.json').read_text())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    output = HERE / 'math-trainings'
    output.mkdir(exist_ok=False)
    env = bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED', None)
    env.pop('PYTEST_PLUGINS', None)
    env.update(MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu',
               BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false')
    def unchanged():
        assert source == bounded.source_snapshot(ROOT)
        assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                   for name, digest in helpers.items())
    for run in range(1, 11):
        unchanged()
        subprocess.run([sys.executable, str(HERE / 'math_train.py'), 'prepare',
                        str(output / f'paired-{run:02}'), str(protocol['epochs'])],
                       cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    jobs = [dict(run=run, condition=condition, name=f'{condition}-{run:02}')
            for run in range(1, 11) for condition in protocol['conditions']]
    (output / 'plan.json').write_text(json.dumps(dict(jobs=jobs, seed=None, retries=0,
        initialization='One fresh entropy state per run, replayed across its three conditions.',
        corpus='Identical shuffled presentations across conditions; expectation-only removes answer lines and labels.',
        epochs=protocol['epochs'], workers=2, worker_memory_bytes=8 * bounded.GIB,
        worker_timeout=86400, completed_run_required=True), indent=2) + '\n')
    active, done = [], []
    started = time.monotonic()
    try:
        while jobs or active:
            unchanged()
            for job in list(active):
                result = job['process'].poll()
                if result is not None:
                    folder = output / job['name']
                    folder.mkdir(exist_ok=True)
                    bounded.write_json(folder / 'process.json', result)
                    done.append(dict(name=job['name'], condition=job['condition'], run=job['run'], process=result))
                    active.remove(job)
                    print(json.dumps(dict(name=job['name'], exit_code=result['exit_code'], reason=result['reason'])), flush=True)
            while jobs and len(active) < 2:
                job = jobs.pop(0)
                command = [sys.executable, str(HERE / 'math_train.py'), 'train',
                           str(output / job['name']), str(output / f"paired-{job['run']:02}"),
                           job['condition'], str(job['run'])]
                # Logs live beside the run directories: the child atomically
                # creates its own directory, rejecting accidental reruns.
                process = bounded.GuardedProcess(command, cwd=ROOT, env=env,
                    log_path=output / (job['name'] + '.log'),
                    memory_bytes=8 * bounded.GIB, timeout=86400).start()
                active.append(dict(job, process=process))
            if sum(job['process'].current_memory_bytes for job in active) > 16 * bounded.GIB:
                raise RuntimeError('aggregate math memory guard')
            bounded.write_json(output / 'progress.json', dict(done=done, pending=len(jobs),
                active=[dict(name=job['name'], pid=job['process'].proc.pid,
                             memory_bytes=job['process'].current_memory_bytes) for job in active],
                seconds=time.monotonic() - started))
            time.sleep(.5)
    finally:
        for job in active:
            job['process'].stop(exit_code=130, reason='campaign_stopped')
    bounded.write_json(output / 'complete.json', dict(jobs=done, attempts=len(done),
        source_matched=source == bounded.source_snapshot(ROOT), seconds=time.monotonic() - started, retries=0))


if __name__ == '__main__':
    main()
