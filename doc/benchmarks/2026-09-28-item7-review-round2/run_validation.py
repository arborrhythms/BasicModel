"""Serial extra validation beside the two bounded reconstruction lanes.

Each lane is capped at 8 GiB; these three concurrent lanes total at most
24 GiB. Source, thresholds, and test seeds are unchanged during the run.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / '2026-09-28-item7-review'
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_guarded, run_suite, source_snapshot

head = Path(json.loads((HERE / 'head-copy.json').read_text())['path'])
source = source_snapshot(ROOT)
head_source = source_snapshot(head)
output = HERE / 'head-memory'
output.mkdir(exist_ok=False)
(output / 'source-manifest.json').write_text(json.dumps(head_source, indent=2) + '\n')
os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='1',
                  BASIC_AUTOLOAD='false', PYTHONPATH=str(head / 'bin'))
result = run_suite(root=head,
    selectors=['test/test_word_store.py::test_two_epoch_training_severs_cross_batch_graph'],
    run_dir=output / 'run', memory_bytes=8 * GIB, workers=1,
    worker_memory_bytes=8 * GIB, timeout=1800, suite_timeout=2000,
    batch_size=32, max_files=1)
assert source_snapshot(head) == head_source
assert source_snapshot(ROOT) == source
print('HEAD memory', result['reason'], result['exit_code'], flush=True)
# Failure of a learning or resource gate remains visible and never suppresses
# the other requested evidence.
code = subprocess.call([str(ROOT / '.venv/bin/python'), str(HERE / 'run_explicit.py'), 'explicit'])
assert source_snapshot(ROOT) == source
print('Explicit gates', code, flush=True)
output = HERE / 'parity'
output.mkdir(exist_ok=False)
(output / 'source-manifest.json').write_text(json.dumps(source, indent=2) + '\n')
env = os.environ.copy()
env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONPATH=os.pathsep.join((str(ROOT / 'bin'), str(ROOT / 'test'), str(PRIOR))))
processes = {}
for layout in ('packed', 'single'):
    command = [str(ROOT / '.venv/bin/python'), str(PRIOR / 'probe.py'),
        '--config', str(PRIOR / 'parity.xml'), '--parity', layout,
        '--out', str(output / (layout + '.json'))]
    processes[layout] = run_guarded(command, cwd=ROOT, env=env,
        log_path=output / (layout + '.log'), memory_bytes=8 * GIB, timeout=1200)
    assert source_snapshot(ROOT) == source
    (output / 'processes.json').write_text(json.dumps(processes, indent=2) + '\n')
    print('Parity', layout, processes[layout]['exit_code'], flush=True)
(output / 'driver-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):
    hashlib.sha256(p.read_bytes()).hexdigest() for p in
    (Path(__file__), PRIOR / 'probe.py', PRIOR / 'parity.py', PRIOR / 'parity.xml')}, indent=2) + '\n')
