"""Run the complete slow selection once, in fresh bounded workers.

Ordinary cases retain 8 GiB. Only the two native production objective arms
receive 24 GiB, one worker at a time. Both tiers reserve 24 GiB in aggregate.
This command never rebuilds the environment or installs a schedule.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import bounded_tests as bounded

NATIVE_FILE = 'test/test_objective_conflicts_slow.py'
NATIVE_WORKER_GIB = 24
ORDINARY_WORKER_GIB = 8
AGGREGATE_GIB = 24


def run(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    started = time.monotonic()
    date = datetime.now(timezone.utc)
    directory = root / 'tmp/slow-tests' / (date.strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True)
    record = dict(date=date.isoformat(), commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        complete=False, selected=0, completed=0, counts={}, failures=[], runs=[],
        duration_seconds=0, source_matched=False,
        ceilings_gib=dict(ordinary_worker=ORDINARY_WORKER_GIB,
                          native_worker=NATIVE_WORKER_GIB, aggregate=AGGREGATE_GIB),
        record_directory=str(directory))
    frozen = bounded.source_snapshot(root)
    previous = {key:os.environ.get(key) for key in
                ('RUN_SLOW', 'RUN_MPS_SLOW', 'OBJECTIVE_CONFLICTS_OUTPUT', 'BASICMODEL_DEVICE')}
    # The suite's ordinary fixtures declare CPU. MPS-only cases explicitly
    # switch to MPS; CUDA-only cases keep their availability skips.
    os.environ.update(RUN_SLOW='1', RUN_MPS_SLOW='1', BASICMODEL_DEVICE='cpu',
                      OBJECTIVE_CONFLICTS_OUTPUT=str(directory/'native-measurements'))
    exit_code = 125
    try:
        with bounded.suite_lock():
            env = bounded.worker_environment(root)
            env['BASICMODEL_DEVICE'] = 'cpu'  # collection imports never allocate on GPU
            request, response = directory/'collect.request.json', directory/'collect.json'
            bounded.write_json(request, dict(selectors=['test'], collect=True))
            collected = bounded.run_guarded(
                [sys.executable, str(root/'test/pytest_worker.py'), str(request), str(response)],
                cwd=root, env=env, log_path=directory/'collect.log',
                memory_bytes=ORDINARY_WORKER_GIB*bounded.GIB, timeout=1800)
            record['collection'] = collected
            if collected['exit_code']:
                record['failures'].append(dict(phase='collection', reason=collected['reason']))
                exit_code = collected['exit_code']
                return exit_code, directory
            slow = json.loads(response.read_text())['slow_selected']
            assert slow and len(slow) == len(set(slow))
            tiers = [('ordinary', [node for node in slow if not node.startswith(NATIVE_FILE+'::')],
                      ORDINARY_WORKER_GIB),
                     ('native', [node for node in slow if node.startswith(NATIVE_FILE+'::')],
                      NATIVE_WORKER_GIB)]
            record['selected'] = len(slow)
            bounded.write_json(directory/'selected.json', slow)
            counts = Counter()
            exit_code = 0
            for name, nodes, ceiling in tiers:
                if not nodes:
                    continue
                assert bounded.source_snapshot(root) == frozen, 'source changed between slow tiers'
                print(f'Slow {name}: {len(nodes)} cases, {ceiling} GiB worker / 24 GiB aggregate', flush=True)
                result = bounded.run_suite(root=root, selectors=nodes, run_dir=directory/name,
                    memory_bytes=AGGREGATE_GIB*bounded.GIB,
                    worker_memory_bytes=ceiling*bounded.GIB, workers=1,
                    timeout=1800, suite_timeout=86400, batch_size=1, max_files=1)
                record['runs'].append(dict(tier=name, result=str(directory/name/'result.json'),
                    reason=result['reason'], exit_code=result['exit_code'],
                    peak_memory_bytes=result.get('peak_aggregate_memory_bytes',0)))
                record['completed'] += len(result['completed'])
                for worker in result['workers']:
                    for report in worker['reports']:
                        counts[report['outcome']] += 1
                        if report['outcome'] in ('failed','xpassed'):
                            record['failures'].append(report)
                if result['exit_code']:
                    exit_code = result['exit_code']
                    if result['reason'] != 'test_failure':
                        record['failures'].append(dict(tier=name, reason=result['reason']))
                record['counts'] = dict(counts)
            record['complete'] = record['completed'] == record['selected']
            return exit_code, directory
    finally:
        record['duration_seconds'] = time.monotonic() - started
        record['source_matched'] = bounded.source_snapshot(root) == frozen
        record['exit_code'] = exit_code
        bounded.write_json(directory/'record.json',record)
        bounded.write_json(directory.parent/'latest.json',record)
        for key,value in previous.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value


if __name__ == '__main__':
    code,path = run()
    print(path/'record.json',flush=True)
    raise SystemExit(code)
