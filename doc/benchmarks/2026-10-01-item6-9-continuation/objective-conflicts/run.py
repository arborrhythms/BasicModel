"""Two single runs, sequentially; same 8 GiB worker and 24 GiB aggregate guard."""
import hashlib
import json
import sys
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded

coverage=HERE.parent/'full-sweep/coverage.json'
while not coverage.exists():
    time.sleep(1)
source=bounded.source_snapshot(ROOT)
assert source==json.loads((HERE.parent/'final-source.json').read_text())
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
bounded.write_json(HERE/'plan.json',dict(source=source,harness=manifest,
    configs=['XOR_grammar','BasicModel_answers_tied_benchmark'],runs_per_config=1,
    worker_bytes=8*bounded.GIB,aggregate_bytes=24*bounded.GIB,timeout=1800,
    no_manual_seed=True,environment='same venv; CPU; XOR gate compile=none, native compile=eager',
    comparison='Same-state cost subtraction only unless Alec authorizes a second training arm.'))
results=[]
for config in ['XOR_grammar','BasicModel_answers_tied_benchmark']:
    out=HERE/config
    out.mkdir(exist_ok=False)
    worker=bounded.GuardedProcess([sys.executable,str(HERE/'observe.py'),'--config',config,'--output',str(out)],
        cwd=ROOT,env=bounded.worker_environment(ROOT),log_path=out/'run.log',memory_bytes=8*bounded.GIB,timeout=1800).start()
    try:
        while True:
            result=worker.poll()
            bounded.write_json(HERE/'progress.json',dict(config=config,pid=worker.proc.pid,
                memory_bytes=worker.current_memory_bytes,completed=results))
            if result is not None:
                break
            time.sleep(.25)
    finally:
        if worker.proc.poll() is None:
            worker.stop(exit_code=130,reason='supervisor_stopped')
    bounded.write_json(out/'process.json',result)
    results.append(dict(config=config,process=result))
    assert source==bounded.source_snapshot(ROOT)
    assert manifest=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
bounded.write_json(HERE/'complete.json',dict(results=results,source_matched=True))
print(json.dumps(results))
