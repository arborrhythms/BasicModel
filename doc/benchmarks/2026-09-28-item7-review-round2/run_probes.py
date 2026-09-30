"""Fresh bounded workers and source receipts for review-round-two probes."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('ITEM7_ROUND2_ROOT') or Path(__file__).resolve().parents[3])
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_suite

os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
                  BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
result = run_suite(root=ROOT, selectors=sys.argv[2:],
    run_dir=HERE / 'probes' / sys.argv[1], memory_bytes=8 * GIB,
    workers=1, worker_memory_bytes=8 * GIB, timeout=1800,
    suite_timeout=5400, batch_size=32, max_files=1)
print(json.dumps(dict(reason=result['reason'], exit_code=result['exit_code'],
    selected=len(result['selected']), completed=len(result['completed']))))
raise SystemExit(result['exit_code'])
