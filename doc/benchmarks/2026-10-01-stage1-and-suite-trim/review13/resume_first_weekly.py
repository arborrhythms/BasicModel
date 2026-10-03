"""Finish only unattempted cases of the first weekly snapshot after its native tier.

Every killed/test-failed attempt remains recorded; no selector is retried.
"""
import json,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as b
import slow_weekly as weekly
RUN=ROOT/'tmp/slow-tests/20261002T012227Z-32db47'

def read(p):return json.loads(p.read_text())
while 'exit_code' not in read(RUN/'record.json'):
 time.sleep(5)
record=read(RUN/'record.json');b.write_json(RUN/'initial-record.json',record)
source=Path(record['source_directory']);frozen=read(RUN/'source.json');assert b.source_snapshot(source)==frozen
selected=read(RUN/'selected.json');attempted=set();reports={};process_failures={}
for run in record['runs']:
 result=read(Path(run['result']))
 for worker in result['workers']:
  attempted.update(worker['selected'])
  for report in worker['reports']:reports[report['nodeid']]=report
  for node in set(worker['selected'])-set(worker['completed']):
   process_failures[node]=dict(nodeid=node,phase='process',outcome='process_failed',reason=worker['reason'])
record['continuation_started']=datetime.now(timezone.utc).isoformat()
import os
os.environ.update(RUN_SLOW='1',RUN_MPS_SLOW='1',OBJECTIVE_CONFLICTS_OUTPUT=str(RUN/'native-measurements'))
os.environ.pop('BASICMODEL_DEVICE',None)
with b.suite_lock():
 for tier,native,ceiling in (('ordinary',False,8),('native',True,24)):
  nodes=[n for n in selected if n not in attempted and n.startswith(weekly.NATIVE_FILE+'::')==native]
  if not nodes:continue
  print(f'Continue {tier}: {len(nodes)} unattempted cases, unchanged {ceiling} GiB ceiling',flush=True)
  segments,seen=weekly.run_tier(source,nodes,RUN/(tier+'-remaining'),ceiling)
  attempted.update(seen)
  for folder,result in segments:
   record['runs'].append(dict(tier=tier,result=str(folder/'result.json'),reason=result['reason'],exit_code=result['exit_code'],
                             peak_memory_bytes=result.get('peak_aggregate_memory_bytes',0)))
   for worker in result['workers']:
    for report in worker['reports']:reports[report['nodeid']]=report
    for node in set(worker['selected'])-set(worker['completed']):
     process_failures[node]=dict(nodeid=node,phase='process',outcome='process_failed',reason=worker['reason'])
   if result['exit_code']:record['exit_code']=result['exit_code']
  assert b.source_snapshot(source)==frozen
 counts=Counter(r['outcome'] for r in reports.values());counts['process_failed']+=len(process_failures)
 inline_completed=sum(r['completed'] for r in record['inline_runs']);inline_attempted=sum(r['expected'] for r in record['inline_runs'])
 for r in record['inline_runs']:counts.update(r['counts'])
 record.update(attempted=len(attempted)+inline_attempted,completed=len(reports)+inline_completed,
   complete=set(selected)<=attempted,counts=dict(counts),
   failures=[r for r in reports.values() if r['outcome'] in ('failed','xpassed')]+list(process_failures.values()),
   duration_seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(record['date'])).total_seconds(),source_matched=True,
   unattempted=sorted(set(selected)-attempted))
 b.write_json(RUN/'record.json',record);b.write_json(RUN.parent/'latest.json',record)
 print(json.dumps({k:record[k] for k in ('complete','selected','attempted','completed','counts','duration_seconds')}),flush=True)
