"""Run the declared seed-42 parity workload in fresh bounded processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
from bounded_tests import run_guarded, source_snapshot

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
a=parser.parse_args()
out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ('bin','test')),
         BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
env.pop('BASIC_SEED',None)
source=source_snapshot(ROOT)
manifest=dict(source=source,seed=42,completed=[],probes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in [*(HERE/name for name in ('probe.py','compare.py','run_measurements.py','parity.xml')),
           *(HERE.parent/'2026-09-21-item10'/name for name in ('probe.py','parity.py'))]})
for mode in ('baseline','packed','single','comparison'):
    command=([sys.executable,str(HERE/'compare.py'),str(out)] if mode=='comparison' else
             [sys.executable,str(HERE/'probe.py'),'--config',str(HERE/'parity.xml'),'--parity',mode,'--out',str(out/f'{mode}.json')])
    if mode=='baseline':
        command=[sys.executable,str(HERE.parent/'2026-09-21-item10'/'probe.py'),'--out',str(out/'baseline.json')]
    result=run_guarded(command,cwd=ROOT,env=env,log_path=out/f'{mode}.log',memory_bytes=8*2**30,timeout=600)
    manifest['completed'].append(result)
    manifest['source_unchanged']=source_snapshot(ROOT)==source
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if result['exit_code'] or not manifest['source_unchanged']:raise SystemExit(result['exit_code'] or 1)
    print(mode+': completed',flush=True)
