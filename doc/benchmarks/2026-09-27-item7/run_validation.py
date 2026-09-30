"""On the frozen candidate: measurements, unchanged gates, one full sweep."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import GIB, run_guarded, run_suite, source_snapshot


def main():
    source = source_snapshot(ROOT)
    affected = HERE / 'affected'
    receipt = json.loads((affected / 'result.json').read_text())
    assert receipt['exit_code'] == 0 and not receipt.get('active_workers')
    assert set(receipt['selected']) == set(receipt['completed'])
    manifest = json.loads((affected / 'source-manifest.json').read_text())
    assert manifest['validated_source'] == source
    os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
                      BASIC_AUTOLOAD='false', RUN_SLOW='0', PYTHONPATH=str(ROOT / 'bin'))
    command = [str(ROOT / '.venv/bin/python')]
    print('Reconstruction measurements', flush=True)
    result = subprocess.run([*command, str(HERE / 'run_measurements.py')], cwd=ROOT)
    assert result.returncode == 0, 'measurement did not finish; no full sweep started'
    print('Small predictor measurement', flush=True)
    identity = run_guarded([*command, str(HERE / 'measure_identity.py'),
        '--out', str(HERE / 'identity.json')], cwd=ROOT, env=os.environ.copy(),
        log_path=HERE / 'identity.log', memory_bytes=8 * GIB, timeout=600)
    (HERE / 'identity-process.json').write_text(json.dumps(identity, indent=2) + '\n')
    assert identity['exit_code'] == 0
    print('Unchanged explicit gates', flush=True)
    prior = json.loads(gzip.decompress((HERE.parent /
        '2026-09-27-item7-5-pressure/explicit-verified/result.json.gz').read_bytes()))
    run_suite(root=ROOT, selectors=prior['selected'], run_dir=HERE / 'explicit',
        memory_bytes=24 * GIB, workers=3, worker_memory_bytes=8 * GIB,
        timeout=1800, suite_timeout=7200, batch_size=1, max_files=1)
    assert source_snapshot(ROOT) == source
    print('Single source-matched full sweep', flush=True)
    result = subprocess.run([*command, str(HERE / 'run_full.py')], cwd=ROOT)
    assert source_snapshot(ROOT) == source
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
