"""Final candidate receipt: production measurement, named runs, extras, one sweep."""
import ast,json,os,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'test'),str(HERE)]
import bounded_tests as b
import measure

def write(path,value): b.write_json(path,value)

def assert_source():
 assert b.source_snapshot(ROOT)==json.loads((HERE/'source-final.json').read_text()), 'candidate source changed'

def isolated(command,folder,*,gib=8,env=None,timeout=1800):
 folder.mkdir(parents=True,exist_ok=False)
 result=b.run_guarded(command,cwd=ROOT,env=env or measure.environment(),log_path=folder/'driver.log',memory_bytes=gib*b.GIB,timeout=timeout)
 write(folder/'process.json',result);assert_source();return result

def extra_selectors():
 # Moved slow cases run here; moved ordinary cases are supplied by the sweep.
 dispositions=json.loads((HERE/'configuration-dispositions.json').read_text())
 files=set()
 for row in dispositions:
  if row.get('replacement') is not None:
   for line in row['references'].splitlines():
    path=line.split(':',1)[0]
    if path.startswith('test/') and Path(path).name.startswith('test_') and path.endswith('.py') and (ROOT/path).exists():files.add(path)
 files.update(['test/test_global_attention.py','test/test_global_consume.py','test/test_reading_attention.py',
  'test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[grammar_reading]',
  'test/test_reasoning_cde_model.py::TestReasoningCDEModel::test_training_step_uses_the_normal_policy_configuration'])
 for row in json.loads((HERE/'test-time-ports.json').read_text()):
  if 'weekly' in row['change']:
   files.add(row['nodeid'])
 # Remaining explicit recent-weekly contracts and their compiled subjects.
 files.update(['test/test_compiled_expectation_boundary.py',
  'test/test_compiled_word_chunk.py::test_tensor_peer_while_runs_symbolic_reference_transaction_and_releases_owner_state',
  'test/test_reverse_traversal.py','test/test_packed_reconstruction_parity.py',
  'test/test_output_path_supervised.py::test_available_input_targets_do_not_train_answer_modules_after_adam',
  'test/test_concept_readout_l1.py::test_real_runbatch_stages_l1_once_and_reports_it_separately'])
 return sorted(node for node in files if '::' not in node or node.split('::')[0] not in files)

def extras():
 folder=HERE/'extra-cases';folder.mkdir(exist_ok=False)
 selectors=extra_selectors();write(folder/'selectors.json',selectors)
 request=folder/'collect.request.json';response=folder/'collect.json'
 write(request,dict(selectors=selectors,collect=True,keyword=None,marker='slow',recycle_file=None))
 process=b.run_guarded([sys.executable,str(ROOT/'test/pytest_worker.py'),str(request),str(response)],
  cwd=ROOT,env=measure.environment(),log_path=folder/'collect.log',memory_bytes=8*b.GIB,timeout=1800)
 write(folder/'collect.process.json',process)
 if process['exit_code']!=0:raise RuntimeError('extra-case collection failed')
 nodes=json.loads(response.read_text())['selected']
 def already_measured(node):
  return any(node==selector or node.startswith(selector+'::') or node.startswith(selector+'[') for selector in measure.SELECTORS.values())
 selectors=list(dict.fromkeys(node for node in nodes if not already_measured(node)))
 write(folder/'reused-table-cases.json',[node for node in nodes if already_measured(node)])
 write(folder/'selected-once.json',selectors)
 source=b.source_snapshot(ROOT);selected=[];attempted=set();reports=[];process_failures=[];segments=[];start=time.monotonic()
 while True:
  part=folder/f'part-{len(segments):02}'
  result=b.run_suite(root=ROOT,selectors=selectors if not selected else [n for n in selected if n not in attempted],
   marker=None,run_dir=part,memory_bytes=24*b.GIB,worker_memory_bytes=8*b.GIB,
   workers=3,timeout=1800,suite_timeout=21600,batch_size=8,max_files=1,lock_path=folder/'dispatch.lock')
  assert_source()
  if not selected:selected=result['selected']
  before=len(attempted);segments.append(str(part/'result.json'))
  for worker in result['workers']:
   if worker.get('phase')=='collection':continue
   reports.extend(worker['reports']);attempted.update(worker['completed'])
   node=worker.get('active_case')
   if node and node not in worker['completed']:
    attempted.add(node);process_failures.append(dict(nodeid=node,reason=worker['reason'],log=worker['log']))
  pending=[n for n in selected if n not in attempted]
  write(folder/'progress.json',dict(selected=selected,attempted=sorted(attempted),reports=reports,
   process_failures=process_failures,segments=segments,pending=pending,seconds=time.monotonic()-start))
  if not pending or len(attempted)==before:break
 write(folder/'complete.json',dict(selected=selected,attempted=sorted(attempted),reports=reports,
  process_failures=process_failures,segments=segments,pending=pending,seconds=time.monotonic()-start,
  source_matched=source==b.source_snapshot(ROOT)))

def gate_counts():
 from collections import Counter
 counts={}
 for gate in (5,6):
  good=0;observed=0
  for p in sorted((HERE/'measurements').glob(f'gate-{gate:02}-trial-*/observations.jsonl')):
   rows=[json.loads(x) for x in p.read_text().splitlines()]
   row=next((r for r in rows if r['kind']=='grammar'),None)
   if row is None:continue
   observed+=1
   if gate==5:
    pred,target=row['predictions'],row['targets'];mse=sum((x-y)**2 for x,y in zip(pred,target))/4
    good+=sum((x>.5)==(y>.5) for x,y in zip(pred,target))==4 and mse<.05
   else:
    good+=all(Counter(x.split())==Counter((y or '').replace(chr(0),' ').split()) for x,y in zip(row['inputs'],row['gate_reconstructions'])) and not any(row['grammar_reconstruction_unavailable'])
  counts[gate]=dict(passed=good,observed=observed)
 write(HERE/'gate-counts.json',counts);return counts

def attribution():
 counts=gate_counts()
 if any(r['observed']!=10 for r in counts.values()):
  write(HERE/'attribution-decision.json',dict(required=None,reason='a gate did not yield all ten outcomes',counts=counts));return
 required=counts[5]['passed']<8 or counts[6]['passed']<=3
 write(HERE/'attribution-decision.json',dict(required=required,counts=counts))
 if not required:return
 folder=HERE/'attribution';folder.mkdir(exist_ok=False)
 pending=[(arm,i) for arm in ('R','RE','RA','REA') for i in range(1,11)];active=[];done=[];start=time.monotonic()
 try:
  while pending or active:
   assert_source()
   for job in list(active):
    result=job['process'].poll()
    if result is not None:
     write(job['path']/'process.json',result);done.append(dict(arm=job['arm'],run=job['run'],process=result));active.remove(job)
   if sum(j['process'].current_memory_bytes for j in active)>24*b.GIB:
    max(active,key=lambda j:j['process'].current_memory_bytes)['process'].stop(exit_code=137,reason='aggregate_memory')
   while pending and len(active)<3:
    arm,i=pending.pop(0);path=folder/f'{arm}-{i:02}';path.mkdir()
    process=b.GuardedProcess([sys.executable,str(HERE/'attribution_probe.py'),arm,str(path)],cwd=ROOT,
     env=measure.environment(),log_path=path/'run.log',memory_bytes=8*b.GIB,timeout=1800).start()
    active.append(dict(arm=arm,run=i,path=path,process=process))
   write(folder/'progress.json',dict(done=done,pending=len(pending),active=[dict(arm=j['arm'],run=j['run'],pid=j['process'].proc.pid) for j in active],seconds=time.monotonic()-start))
   time.sleep(.5)
 finally:
  for job in active:job['process'].stop(exit_code=130,reason='campaign_stopped')
 write(folder/'complete.json',dict(done=done,seconds=time.monotonic()-start,source_matched=True))

def main():
 assert not (HERE/'source-final.json').exists()
 source=b.source_snapshot(ROOT);write(HERE/'source-final.json',source)
 write(HERE/'documentation-final-start.json',b.documentation_snapshot(ROOT))
 with zipfile.ZipFile(HERE/'source-final.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in source:z.write(ROOT/name,name)
 write(HERE/'campaign-plan.json',dict(source=source,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  worker_gib=8,native_only_gib=24,initial_sweep_workers=10,sweep_memory_gib=b.default_test_memory_bytes()/b.GIB,
  seed=None,baseline=dict(cases=5219,minutes=122),order=['production native stage1','named measurements and gate ownership','conditional attribution','extra slow cases','one full sweep'],
  mnist='real subset, both ergodic arms, in ordinary sweep; test-local matching 784-slot geometry, production XMLs unchanged; saved original shape failures and reconstruction warning',
  no_head_run=True,no_commits=True))
 env=measure.environment();env['OBJECTIVE_CONFLICTS_OUTPUT']=str(HERE/'native-stage1')
 isolated([sys.executable,'-m','pytest','-q','test/test_objective_conflicts_slow.py::test_native_production_objective_measurements'],
  HERE/'native-driver',gib=24,env=env,timeout=1800)
 measure.campaign()
 attribution()
 os.environ.update(measure.environment());os.environ['MODEL_COMPILE']='none'
 extras()
 # Keep ordinary sweep policy, allowing explicit compilation subjects to
 # select their own backend. The usual default remains MODEL_COMPILE=eager.
 os.environ['MODEL_COMPILE']='eager'
 import full_sweep
 full_sweep.run()
 assert_source();write(HERE/'campaign-complete.json',dict(source_matched=True,completed=True))

if __name__=='__main__':main()
