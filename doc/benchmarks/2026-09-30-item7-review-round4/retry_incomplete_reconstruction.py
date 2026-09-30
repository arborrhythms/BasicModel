"""Retry incomplete deadline stops once, serially, with identical inputs and limits.

Completed measurements are never repeated or selected by value. All original
attempts remain in the primary receipt. The eight predeclared seeds stay 0–7.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
from bounded_tests import GIB,run_guarded,source_snapshot,ProcessTree
primary=HERE/'final3-reconstruction-candidate'
print('Waiting for the original eight guarded attempts',flush=True)
while not (primary/'driver-hashes.json').exists():time.sleep(2)
source=json.loads((primary/'source-manifest.json').read_text())
assert source_snapshot(ROOT)==source
initial=json.loads((primary/'processes.json').read_text())
assert set(initial)==set(map(str,range(8)))
out=HERE/'final3-reconstruction-retries';out.mkdir(exist_ok=False)
(out/'source-manifest.json').write_text(json.dumps(source,indent=2)+'\n')
env=os.environ.copy();env.pop('BASIC_SEED',None)
env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='eager',BASIC_AUTOLOAD='false',
 OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
 PYTHONPATH=os.pathsep.join(str(ROOT/p) for p in ('bin','test')))
retries={};effective={}
for seed in range(8):
 key=str(seed);process=initial[key];directory=primary
 if process['reason']=='timeout':
  args=[str(ROOT/'.venv/bin/python'),str(HERE/'measure_reconstruction.py'),
   '--revision','item7-round4-candidate','--seed',key,'--out',str(out/f'seed-{seed}.json')]
  process=run_guarded(args,cwd=ROOT,env=env,log_path=out/f'seed-{seed}.log',memory_bytes=8*GIB,timeout=1200)
  retries[key]=process;directory=out
  (out/'processes.json').write_text(json.dumps(retries,indent=2)+'\n')
  print('retry',seed,process['reason'],process['exit_code'],flush=True)
  if process['reason'] in ('memory','aggregate_memory'):
   args[-1]=str(out/f'seed-{seed}-diagnostic.json');started=time.monotonic()
   with (out/f'seed-{seed}-diagnostic.log').open('w') as log:
    proc=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
    tree,peak,reason=ProcessTree(proc.pid),0,'completed'
    while proc.poll() is None:
     peak=max(peak,tree.sample())
     if time.monotonic()-started>1200:
      tree.terminate(proc,.5);reason='timeout';break
     time.sleep(.1)
   process['diagnostic_only']=dict(exit_code=proc.returncode,peak_memory_bytes=peak,memory_guard=None,elapsed_seconds=time.monotonic()-started,reason=reason)
   (out/'processes.json').write_text(json.dumps(retries,indent=2)+'\n')
 path=directory/f'seed-{seed}.json'
 measurement=json.loads(path.read_text()) if path.exists() else None
 complete=(process['reason']=='exit' and process['exit_code']==0 and measurement is not None
  and [p['name'] for p in measurement['phases']]==['before_training','training','after_training'])
 effective[key]=dict(process=process,measurement=str(path.relative_to(HERE)),complete_under_guard=complete)
 assert source_snapshot(ROOT)==source
 result=dict(validated_source=source,effective=effective,all_guarded_completions=len(effective)==8 and all(v['complete_under_guard'] for v in effective.values()),
  rationale=__doc__,original_receipt=primary.name,retry_receipt=out.name)
 (HERE/'final3-reconstruction-reconciled.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'driver-hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),HERE/'measure_reconstruction.py')},indent=2)+'\n')
if result['all_guarded_completions']:
 (HERE/'final-reconstruction-complete.json').write_text(json.dumps(dict(receipt='final3-reconstruction-reconciled.json',validated_source=source),indent=2)+'\n')
print('all eight complete under unchanged guard:',result['all_guarded_completions'],flush=True)
