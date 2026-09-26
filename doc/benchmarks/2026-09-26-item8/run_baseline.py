"""Retry only the baseline after correcting its staticmethod instrumentation."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_guarded, source_snapshot

output = ROOT / 'output/item8-baseline'
output.mkdir(exist_ok=False)
source = source_snapshot(ROOT)
(output / 'source-manifest.json').write_text(json.dumps(source, indent=2) + '\n')
env = os.environ.copy()
env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
result = run_guarded([str(ROOT / '.venv/bin/python'), str(Path(__file__).with_name('measure.py')),
    'baseline', '--out', str(output / 'serial-baseline.json')], cwd=ROOT, env=env,
    log_path=output / 'serial-baseline.log', memory_bytes=8 * GIB, timeout=1200)
assert source_snapshot(ROOT) == source, 'source changed during baseline'
(output / 'process.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
raise SystemExit(result['exit_code'])
