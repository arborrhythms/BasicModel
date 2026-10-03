"""Final migrated-source first batches, bounded at the unchanged 8 GiB."""
import json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[3]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as b
out=R/'mode-final-batches';out.mkdir(exist_ok=False)
configs=json.loads((R/'mode-first-batches/manifest.json').read_text())['configs']
frozen=b.source_snapshot(ROOT)
b.write_json(out/'manifest.json',dict(source=frozen,configs=configs,workers=2,worker_gib=8,aggregate_gib=16,timeout_seconds=1800,seed=None,backend='none',reason='Final verification after correcting migrated unit staging and completing the retired-path ports; earlier first-batch exceptions are retained.'))
env=b.worker_environment(ROOT);env.pop('BASIC_SEED',None)
env.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',RUN_SLOW='1')
active=[];done=[];start=time.monotonic()
try:
 while configs or active:
  assert b.source_snapshot(ROOT)==frozen,'source changed during final migrated first batches'
  for job in list(active):
   result=job['process'].poll()
   if result is not None:
    b.write_json(out/(Path(job['config']).stem+'-process.json'),result)
    done.append(dict(config=job['config'],**result));active.remove(job)
    print(json.dumps({k:done[-1][k] for k in ['config','reason','exit_code','elapsed_seconds','peak_memory_bytes']}),flush=True)
  if sum(j['process'].current_memory_bytes for j in active)>16*b.GIB:
   max(active,key=lambda j:j['process'].current_memory_bytes)['process'].stop(exit_code=137,reason='aggregate_memory')
  while configs and len(active)<2:
   config=configs.pop(0);stem=Path(config).stem
   command=[sys.executable,str(R/'mode_first_batches.py'),'--child','--config',config,'--output',str(out/(stem+'.json'))]
   process=b.GuardedProcess(command,cwd=ROOT,env=env,log_path=out/(stem+'.log'),memory_bytes=8*b.GIB,timeout=1800).start()
   active.append(dict(config=config,process=process))
  b.write_json(out/'progress.json',dict(done=done,pending=len(configs),elapsed_seconds=time.monotonic()-start,active=[dict(config=j['config'],pid=j['process'].proc.pid,memory_bytes=j['process'].current_memory_bytes) for j in active]))
  time.sleep(.5)
finally:
 for j in active:j['process'].stop(exit_code=130,reason='stopped')
b.write_json(out/'complete.json',dict(results=done,source_matched=b.source_snapshot(ROOT)==frozen,elapsed_seconds=time.monotonic()-start))
