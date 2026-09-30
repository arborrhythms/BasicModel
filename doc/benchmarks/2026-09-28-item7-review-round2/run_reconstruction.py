"""All three declared seeds, each in a fresh process under the existing cap."""
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
for seed in (0, 1, 2):
    command = [str(MAIN / '.venv/bin/python'), str(HERE / 'measure_reconstruction.py'),
        '--revision', args.revision, '--seed', str(seed), '--out', str(output / f'seed-{seed}.json')]
    processes[str(seed)] = run_guarded(command, cwd=root, env=env,
        log_path=output / f'seed-{seed}.log', memory_bytes=8 * GIB, timeout=1200)
    assert source_snapshot(root) == source, 'measurement source changed'
    (output / 'processes.json').write_text(json.dumps(processes, indent=2) + '\n')
    print(args.label, seed, processes[str(seed)]['reason'], processes[str(seed)]['exit_code'], flush=True)
(output / 'driver-hashes.json').write_text(json.dumps({p.name:
    hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE / 'measure_reconstruction.py')}, indent=2) + '\n')
raise SystemExit(int(any(v['exit_code'] for v in processes.values())))
