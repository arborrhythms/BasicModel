"""All eight declared seeds, each in a fresh process under the existing cap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[2]
sys.path.insert(0, str(MAIN / 'test'))
from bounded_tests import GIB, run_guarded, source_snapshot

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--revision', required=True)
args = parser.parse_args()
root = args.root.resolve()
output = HERE / (args.label + '-reconstruction')
output.mkdir(exist_ok=False)
source = source_snapshot(root)
(output / 'source-manifest.json').write_text(json.dumps(source, indent=2) + '\n')
env = os.environ.copy()
env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONPATH=os.pathsep.join(str(root / p) for p in ('bin', 'test')))
processes = {}
for seed in range(8):
    command = [str(MAIN / '.venv/bin/python'), str(HERE / 'measure_reconstruction.py'),
        '--revision', args.revision, '--seed', str(seed), '--out', str(output / f'seed-{seed}.json')]
    processes[str(seed)] = run_guarded(command, cwd=root, env=env,
        log_path=output / f'seed-{seed}.log', memory_bytes=8 * GIB, timeout=1200)
    if processes[str(seed)]['reason'] in ('memory', 'aggregate_memory'):
        import subprocess, time
        from bounded_tests import ProcessTree
        repeat = list(command)
        repeat[-1] = str(output / f'seed-{seed}-diagnostic.json')
        started = time.monotonic()
        with (output / f'seed-{seed}-diagnostic.log').open('w') as log:
            proc = subprocess.Popen(repeat, cwd=root, env=env, stdout=log, stderr=log, start_new_session=True)
            tree, peak = ProcessTree(proc.pid), 0
            while proc.poll() is None:
                peak = max(peak, tree.sample())
                if time.monotonic() - started > 1200:
                    tree.terminate(proc, .5)
                    break
                time.sleep(.1)
        processes[str(seed)]['diagnostic_only'] = dict(exit_code=proc.returncode, peak_memory_bytes=peak, memory_guard=None)
    assert source_snapshot(root) == source, 'measurement source changed'
    (output / 'processes.json').write_text(json.dumps(processes, indent=2) + '\n')
    print(args.label, seed, processes[str(seed)]['reason'], processes[str(seed)]['exit_code'], flush=True)
(output / 'driver-hashes.json').write_text(json.dumps({p.name:
    hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE / 'measure_reconstruction.py')}, indent=2) + '\n')
raise SystemExit(int(any(v['exit_code'] for v in processes.values())))
