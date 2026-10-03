"""One measurement arm in a guarded fresh process; no automatic retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT/'test'), str(HERE.parent),
               str(HERE.parent.parent/'2026-09-28-item7-review')]
import bounded_tests as bounded
from measure import environment
from review_source import supporting_inputs

parser = argparse.ArgumentParser()
parser.add_argument('config', choices=('XOR_grammar', 'BasicModel_answers_tied_benchmark'))
parser.add_argument('arm', choices=('step5a', 'cut'))
args = parser.parse_args()
output = HERE / (args.config + '-' + args.arm)
output.mkdir(exist_ok=False)
source, inputs = bounded.source_snapshot(ROOT), supporting_inputs(ROOT)
harness = HERE/'observe.py'
shutil.copyfile(harness, output/'observer-source.py.txt')
bounded.write_json(output/'manifest.json', dict(source=source, supporting_inputs=inputs,
    observer_sha256=hashlib.sha256(harness.read_bytes()).hexdigest(),
    seed=None, worker_bytes=8*bounded.GIB, aggregate_bytes=24*bounded.GIB,
    worker_timeout_seconds=1800, retries=0, config=args.config, arm=args.arm))
worker = bounded.GuardedProcess([sys.executable, str(harness), '--config', args.config,
    '--arm', args.arm, '--output', str(output)], cwd=ROOT, env=environment(ROOT),
    log_path=output/'run.log', memory_bytes=8*bounded.GIB, timeout=1800).start()
started = time.monotonic()
try:
    while (result := worker.poll()) is None:
        bounded.write_json(output/'progress.json', dict(pid=worker.proc.pid,
            seconds=time.monotonic()-started, memory_bytes=worker.current_memory_bytes))
        time.sleep(.25)
finally:
    if not worker.finished:
        worker.stop(exit_code=130, reason='measurement_stopped')
bounded.write_json(output/'process.json', result)
matched = source==bounded.source_snapshot(ROOT) and inputs==supporting_inputs(ROOT)
bounded.write_json(output/'verification.json',dict(source_and_inputs_matched=matched,
    observer_matched=hashlib.sha256(harness.read_bytes()).hexdigest()==
        json.loads((output/'manifest.json').read_text())['observer_sha256']))
print(json.dumps(result))
assert matched
raise SystemExit(result['exit_code'])
