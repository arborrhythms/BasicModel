"""Run the seven reviewed opt-in gates with their prior CPU resource limits.

The first dispatcher inherited RUN_SLOW=0 and therefore skipped all seven.
Preserve that receipt as `explicit`; this invocation is `explicit-verified`.
Run after the one full sweep to avoid exceeding its aggregate resource cap.
"""
import gzip
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_suite, source_snapshot


def main():
    full = json.loads((HERE / 'full/result.json').read_text())
    assert full['reason'] != 'running' and not full.get('active_workers')
    source = source_snapshot(ROOT)
    manifest = json.loads((HERE / 'full/source-manifest.json').read_text())
    assert manifest['validated_source'] == source
    prior = json.loads(gzip.decompress((HERE.parent /
        '2026-09-27-item7-5-pressure/explicit-verified/result.json.gz').read_bytes()))
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
                      BASIC_AUTOLOAD='false', RUN_SLOW='1', PYTHONPATH=str(ROOT / 'bin'))
    result = run_suite(root=ROOT, selectors=prior['selected'], run_dir=HERE / 'explicit-verified',
        memory_bytes=8 * GIB, workers=1, worker_memory_bytes=8 * GIB,
        timeout=1800, suite_timeout=5400, batch_size=32, max_files=1)
    assert source_snapshot(ROOT) == source
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
