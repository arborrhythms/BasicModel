"""Run a measurement with the unchanged worker and aggregate memory guards."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded

parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--companion', type=Path)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
out = HERE / args.label
out.mkdir(exist_ok=False)
source = bounded.source_snapshot(ROOT)
bounded.write_json(out / 'source-manifest.json', dict(validated_source=source))
env = bounded.worker_environment(ROOT)
env.pop('BASIC_SEED', None)
env.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager', BASIC_AUTOLOAD='false',
    PYTHONPATH=os.pathsep.join((str(ROOT / 'bin'), str(ROOT / 'test'), str(HERE),
        str(HERE.parent / '2026-09-21-item10'))))
command = [part.replace('{out}', str(out)) for part in args.command]
worker = bounded.GuardedProcess(command, cwd=ROOT, env=env,
    log_path=out / 'driver.log', memory_bytes=8 * bounded.GIB, timeout=1800)
peers, peak = {}, 0
try:
    worker.start()
    while worker.poll() is None:
        current = worker.current_memory_bytes
        if args.companion and args.companion.exists():
            progress = json.loads(args.companion.read_text())
            for job in progress['active']:
                pid = job['pid']
                peers.setdefault(pid, bounded.ProcessTree(pid))
                current += peers[pid].sample()
        peak = max(peak, current)
        if current > 24 * bounded.GIB:
            worker.stop(exit_code=137, reason='aggregate_memory')
            break
        time.sleep(.25)
finally:
    worker.finish()
    bounded.write_json(out / 'aggregate.json', dict(peak_memory_bytes=peak,
        limit_bytes=24 * bounded.GIB, companion=str(args.companion)))
assert source == bounded.source_snapshot(ROOT), 'measured source changed'
print(json.dumps(worker.result))
raise SystemExit(worker.result['exit_code'])
