"""Bounded diagnostic replays of original misses, never new gate attempts."""
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
import bounded_tests as bounded


def main():
    read = lambda path: json.loads(path.read_text())
    summary = read(HERE/'standing-summary-before-bisection.json')
    baseline = read(HERE/'landing-baseline-verification.json')
    assert baseline['all_runtime_and_test_git_blobs_match']
    destination = HERE/'mm-bisection'
    destination.mkdir(exist_ok=False)
    jobs = [(row['name'], variant, source) for row in summary['outcomes']
            if row['kind'] == 'mm' and not row['bar']
            for variant, source in (('closing', ROOT), ('landing', Path(baseline['source_root'])))]
    env = bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu',
               OMP_NUM_THREADS='1', BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false')
    active = []
    try:
        while jobs or active:
            for name, process in list(active):
                result = process.poll()
                if result is not None:
                    bounded.write_json(destination/f'{name}.process.json', result)
                    active.remove((name, process))
                    assert result['exit_code'] == 0, (name, result)
                    print(json.dumps(dict(name=name, exit_code=0, seconds=result['elapsed_seconds'])), flush=True)
            while jobs and len(active) < 2:
                start, variant, source = jobs.pop(0)
                name = f'{start}-{variant}'
                process = bounded.GuardedProcess([sys.executable, str(HERE/'mm_bisection.py'),
                    str(source), variant, start], cwd=ROOT, env=env,
                    log_path=destination/f'{name}.log', memory_bytes=8*bounded.GIB, timeout=1800).start()
                active.append((name, process))
            assert sum(process.current_memory_bytes for _, process in active) <= 16*bounded.GIB
            time.sleep(.25)
    finally:
        for _, process in active:
            process.stop(exit_code=130, reason='diagnostic_stopped')


if __name__ == '__main__':
    main()
