"""Locate the unchanged production allocation using signal-only stack samples."""
import faulthandler
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'test'))
import bounded_tests as bounded

if len(sys.argv) > 1:
    out = Path(sys.argv[1])
    trace = (out/'stacks.txt').open('w')
    faulthandler.register(signal.SIGUSR1, file=trace, all_threads=True)
    sys.argv = [str(ROOT/'test/objective_conflicts_probe.py'), '--config',
                'BasicModel_answers_tied_benchmark', '--arm', 'step5a', '--output', str(out)]
    runpy.run_path(sys.argv[0], run_name='__main__')
else:
    out = HERE/'allocation-diagnostic'
    out.mkdir(exist_ok=False)
    source = bounded.source_snapshot(ROOT)
    env = bounded.worker_environment(ROOT)
    env.pop('BASIC_SEED', None)
    env.update(BASICMODEL_DEVICE='cpu', RUN_SLOW='1')
    worker = bounded.GuardedProcess([sys.executable, str(__file__), str(out)],
        cwd=ROOT, env=env, log_path=out/'run.log', memory_bytes=24*bounded.GIB,
        timeout=1800).start()
    thresholds = [11, 16, 22]
    events = []
    try:
        while (result := worker.poll()) is None:
            while thresholds and worker.current_memory_bytes >= thresholds[0]*bounded.GIB:
                boundary = thresholds.pop(0)
                os.kill(worker.proc.pid, signal.SIGUSR1)
                events.append(dict(threshold_gib=boundary, memory_bytes=worker.current_memory_bytes,
                                   seconds=time.monotonic()-worker.started))
                bounded.write_json(out/'signals.json', events)
            time.sleep(.05)
    finally:
        if not worker.finished:
            worker.stop(exit_code=130, reason='diagnostic_stopped')
    bounded.write_json(out/'process.json', result)
    assert source == bounded.source_snapshot(ROOT)
    print(json.dumps(result))
