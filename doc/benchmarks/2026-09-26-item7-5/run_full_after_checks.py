"""Run one full receipt after this frozen source's affected checks finish."""
import json
import gzip
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import LOCK, source_snapshot, suite_lock


def read_if_ready(path):
    try:
        return (json.loads(path.read_text()) if path.exists() else
                json.loads(gzip.decompress(path.with_suffix(path.suffix + '.gz').read_bytes())))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    source = source_snapshot(ROOT)
    print('Waiting for affected checks and measurements before the full sweep.', flush=True)
    while True:
        affected = read_if_ready(HERE / 'affected-verified/result.json')
        review = read_if_ready(HERE / 'review-checks-processes.json') or {}
        fixture = read_if_ready(HERE / 'layout-fixture-verified/result.json')
        if affected and affected['reason'] != 'running':
            assert affected['exit_code'] == 0, 'affected checks did not pass'
            if (fixture and fixture.get('exit_code') not in (None, 125)
                    and not fixture.get('active_workers') and 'measurements' in review):
                assert fixture['exit_code'] == 0, 'layout fixture did not pass'
                assert all(review[name] == 0 for name in
                           ('explicit-final', 'trie-verified', 'measurements')), review
                break
        time.sleep(10)
    assert source_snapshot(ROOT) == source, 'source changed while waiting'
    assert not (HERE / 'full').exists(), 'the full receipt already exists'
    env = os.environ.copy()
    env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', RUN_SLOW='0',
               BASIC_AUTOLOAD='false', PYTHONPATH=str(ROOT / 'bin'))
    previous = read_if_ready(HERE / 'full-before-trace-fixture-port/result.json')
    assert previous and Counter(previous['completed']) == Counter(previous['selected'])
    files = list(dict.fromkeys(node.split('::', 1)[0] for node in previous['selected']))
    costs = {path: 0. for path in files}
    for worker in previous['workers']:
        owned = {node.split('::', 1)[0] for node in worker['selected']}
        assert len(owned) <= 1, 'the prior run must isolate test files'
        for path in owned:
            costs[path] += worker['elapsed_seconds']
    files.sort(key=lambda path: -costs[path])
    (HERE / 'full-schedule.json').write_text(json.dumps(dict(
        rationale='Start expensive files first; keep collection order within each file and one file per worker.',
        expected_cases=len(previous['selected']),
        files=[dict(path=path, previous_seconds=costs[path]) for path in files]), indent=2)+'\n')
    command = [sys.executable, 'test/test_report.py', *files, '--workers', '3',
               '--memory-gib', '24', '--max-files', '1', '--batch-size', '32',
               '--run-dir', str(HERE / 'full')]
    # Explicit files bypass the runner's automatic full-suite lock. Retain
    # that same lock here, then verify this is exactly the complete case set.
    with suite_lock(LOCK):
        result = subprocess.run(command, cwd=ROOT, env=env)
    current = read_if_ready(HERE / 'full/result.json')
    assert Counter(current['selected']) == Counter(previous['selected']), 'full case set changed'
    assert source_snapshot(ROOT) == source, 'source changed during the full sweep'
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
