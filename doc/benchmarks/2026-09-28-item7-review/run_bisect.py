"""Fixed component ablations; two bounded workers alongside the parity runner."""
from collections import deque
import time
import hashlib
import json
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import GIB, GuardedProcess, termination_signals, source_snapshot


def main():
    source=source_snapshot(ROOT)
    output=HERE/'bisect'
    output.mkdir(exist_ok=False)
    (output/'source-manifest.json').write_text(json.dumps(source,indent=2)+'\n')
    modes=('interpret-head','context-head','identity-no-st','journal-detached')
    driver=HERE/'bisect_components.py'
    (output/'protocol.json').write_text(json.dumps(dict(modes=modes,
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        workers=2,memory_gib_per_worker=8,seconds_per_worker=1800,
        reason='Fixed serial reconstruction rises from 0.1224316135 to 0.1227969993; spec section 12.8.'),indent=2)+'\n')
    env=os.environ.copy()
    env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ('bin','test',str(HERE.relative_to(ROOT)))))
    results, active, pending = {}, {}, deque(modes)
    with termination_signals():
        try:
            while pending or active:
                while pending and len(active) < 2:
                    mode=pending.popleft();folder=output/mode;folder.mkdir()
                    guard=GuardedProcess([str(ROOT/'.venv/bin/python'),str(driver),'--mode',mode,
                        '--out',str(folder/'baseline.json')],cwd=ROOT,env=env,log_path=folder/'run.log',
                        memory_bytes=8*GIB,timeout=1800,terminate_grace=1)
                    active[mode]=guard;guard.start()
                for mode,guard in tuple(active.items()):
                    if guard.poll() is None:
                        continue
                    guard.finish();results[mode]=guard.result;del active[mode]
                    assert source_snapshot(ROOT)==source,'source changed during component check'
                    (output/'processes.json').write_text(json.dumps(results,indent=2)+'\n')
                    print(mode,guard.result['exit_code'],flush=True)
                if active:
                    time.sleep(.25)
        finally:
            for guard in active.values():
                guard.stop(exit_code=130,reason='supervisor_interrupted');guard.finish()

    return int(any(result['exit_code'] for result in results.values()))

if __name__=='__main__':
    raise SystemExit(main())
