"""Save a bounded probe once under a distinct receipt name."""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
import bounded_tests as bounded

name, *command = sys.argv[1:]
env = bounded.worker_environment(ROOT)
env.update(MODEL_COMPILE='none', BASICMODEL_DEVICE='cpu', BASIC_AUTOLOAD='false',
           PYTHONPATH=os.pathsep.join((str(ROOT/'bin'), str(ROOT/'test'))))
env.pop('BASIC_SEED', None)
out = HERE/'probes'/name
out.mkdir(exist_ok=False)
source = bounded.source_snapshot(ROOT)
p = bounded.GuardedProcess(command, cwd=ROOT, env=env, log_path=out/'output.log',
                           memory_bytes=8*bounded.GIB, timeout=1800).start()
try:
    while (result := p.poll()) is None:
        time.sleep(.3)
finally:
    if p.poll() is None:
        p.stop(exit_code=130, reason='interrupted')
bounded.write_json(out/'result.json', dict(command=command, source=source, process=result))
print(json.dumps(result))
raise SystemExit(result['exit_code'])
