"""Join final case outcomes with prior evidence; never rerun a case."""
from pathlib import Path
from collections import Counter,defaultdict
import json,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT/'doc/benchmarks/2026-10-01-stage1-and-suite-trim/review14'

def read(p,d=None):return json.loads(p.read_text())if p.exists()else d

def error(message):
 lines=[x.strip()for x in message.splitlines() if re.match(r'^E\s',x)]
 return '\n'.join(lines[-8:])or message[-1500:]

def signature(message):
 text=error(message)
 text=re.sub(r'0x[0-9a-f]+','0x…',text)
 text=re.sub(r'pytest-\d+','pytest-…',text)
 return text

old=read(OLD/'closing/full-sweep/receipt.json',{}).get('failures',[])
weekly=read(OLD/'weekly-triage.json',{})
if isinstance(weekly,dict):weekly=weekly.get('triage',[])
previous=defaultdict(list)
for source,rows in [('section14 sweep',old),('first weekly run',weekly)]:
 if isinstance(rows,dict):rows=[dict(v,nodeid=k)for k,v in rows.items()]
 for r in rows:
  if r.get('nodeid'):previous[r['nodeid']].append(dict(source=source,**r))
current=[]
full=read(HERE/'full-sweep/receipt.json',{})
if (HERE/'campaign-complete.json').exists():
 for r in full.get('failures',[]):current.append(dict(scope='full sweep',**r))
else:
 for part in sorted((HERE/'full-sweep').glob('part-*/result.json')):
  for worker in read(part,{}).get('workers',[]):
   if worker.get('phase')=='collection':continue
   for r in worker.get('reports',[]):
    if r['outcome'] in ('failed','xpassed'):current.append(dict(scope='full sweep',**r))
   node=worker.get('active_case')
   if node and node not in worker.get('completed',[]):
    current.append(dict(scope='full sweep',nodeid=node,outcome='process_failed',reason=worker.get('reason'),log=worker.get('log')))
extra=read(HERE/'extra-cases/complete.json',{})
if not extra:
 extra=dict(reports=[],process_failures=[])
 for part in sorted((HERE/'extra-cases').glob('part-*/result.json')):
  for worker in read(part,{}).get('workers',[]):
   if worker.get('phase')=='collection':continue
   extra['reports'].extend(worker.get('reports',[]))
   node=worker.get('active_case')
   if node and node not in worker.get('completed',[]) and worker.get('exit_code') is not None:
    extra['process_failures'].append(dict(nodeid=node,reason=worker.get('reason'),log=worker.get('log')))
for r in extra.get('reports',[]):
 if r['outcome']in ('failed','xpassed'):current.append(dict(scope='extra slow cases',**r))
for r in extra.get('process_failures',[]):current.append(dict(scope='extra slow cases',outcome='process_failed',**r))
for p in sorted((HERE/'measurements').glob('gate-*/run/result.json')):
 result=read(p,{})
 for w in result.get('workers',[]):
  for r in w.get('reports',[]):
   if r['outcome']in ('failed','xpassed'):current.append(dict(scope=p.parents[1].name,**r))
  if w.get('active_case')and w['active_case']not in w.get('completed',[]):current.append(dict(scope=p.parents[1].name,nodeid=w['active_case'],outcome='process_failed',reason=w['reason']))
investigations=read(HERE/'failure-investigations.json',{})
triage=[]
for r in current:
 node=r['nodeid'];msg=r.get('message','');matches=previous[node]
 exact=[v for v in matches if signature(v.get('message',''))==signature(msg) and msg]
 if r.get('outcome')=='xpassed':
  category='non-strict XPASS; not a failure';reason='The worker reports non-strict unexpected passes separately. Strict XPASS is reported as failed by pytest.'
 elif node=='test/test_mm_xor.py::TestMMXorConvergence::test_convergence':
  category='declared §17 red';reason='Removed cross-word percept lookup; word-level MM_xor belongs to item 6.8. Test and bar unchanged.'
 elif 'test_mm_grammar_learns_xor_signal'in node:
  category='declared intermittent MM_grammar';reason='The unchanged proof can stop at .25; compare observed error, not merely the case name.'
 elif any(x in node for x in ('TestXorGrammarLearnsXor','TestXorGrammarReconstruction')):
  category='grammar gate outcome';reason='Predeclared first runs supply table rows; all ten outcomes retained.'
 elif r.get('outcome')=='process_failed':
  category='guard/process stop';reason=r.get('reason','process did not produce a case result')
 elif exact:
  category='matching previously reported error';reason='Same case and normalized error in '+', '.join(v['source']for v in exact)
 elif matches:
  category='previously failed case; cause requires comparison';reason='Previous evidence exists; this does not establish that the current cause is unchanged.'
 else:
  category='new or previously unrecorded failure';reason='No matching failure by case id in the prior sweep/weekly triage; inspect this round’s message.'
 triage.append(dict(scope=r['scope'],nodeid=node,category=category,reason=reason,error=error(msg),
    previous=[dict(source=v['source'],error=error(v.get('message','')),classification=v.get('classification'),cause=v.get('cause'))for v in matches],
    investigation=investigations.get(node), report=r))
(HERE/'failure-triage.json').write_text(json.dumps(triage,indent=2)+'\n')
print(json.dumps(Counter(r['category']for r in triage)))
# Timings use the actual closing attempts, without a second profile pass.
reports=[];case_workers=defaultdict(list);case_stops=defaultdict(list)
for path in [HERE/'full-sweep',HERE/'extra-cases']:
 for part in sorted(path.glob('part-*/result.json')):
  for worker in read(part,{}).get('workers',[]):
   reports.extend(worker.get('reports',[]))
   for node in {r['nodeid']for r in worker.get('reports',[])}:
    case_workers[node].append(dict(segment=str(part.relative_to(HERE)),log=worker.get('log'),
      elapsed_seconds=worker.get('elapsed_seconds'),peak_memory_bytes=worker.get('peak_memory_bytes'),
      completed_cases=len(worker.get('completed',[]))))
   node=worker.get('active_case')
   if node and node not in worker.get('completed',[]):
    case_stops[node].append(dict(segment=str(part.relative_to(HERE)),log=worker.get('log'),
      reason=worker.get('reason'),worker_elapsed_seconds=worker.get('elapsed_seconds'),
      peak_memory_bytes=worker.get('peak_memory_bytes'),
      completed_cases_before_stop=len(worker.get('completed',[])),
      timing_scope='partial worker wall time; includes startup and any preceding cases, not a completed case duration'))
durations=defaultdict(list)
for r in reports:durations[r['nodeid']].append(dict(phase=r['phase'],seconds=r['duration'],outcome=r['outcome']))
rows=[]
for r in read(HERE/'test-time-ports.json',[]):
 observations=durations.get(r['nodeid'],[])
 executed=[v for v in observations if v['outcome']!='skipped']
 rows.append(dict(r,after_reports=observations,after_seconds=sum(v['seconds']for v in executed)if executed else None,
                  after_workers=case_workers.get(r['nodeid'],[]),
                  after_stops=case_stops.get(r['nodeid'],[]),
                  timing_scope='pytest call intervals plus failed setup/teardown; successful shared setup is not charged to a case; worker wall includes it',
                  verified=bool(executed)and all(v['outcome'] in ('passed','xfailed','xpassed')for v in executed)))
(HERE/'timing-results.json').write_text(json.dumps(rows,indent=2)+'\n')
