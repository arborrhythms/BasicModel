"""Summarize saved first outcomes. No models, training, retries or source edits."""
from collections import Counter
import json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(path):
 p=HERE/path
 return json.loads(p.read_text()) if p.exists() else None

def observed(path):
 p=HERE/path
 return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

def gate_rows():
 rows=[]
 for gate,name in [(5,'class'),(6,'reconstruction')]:
  for i in range(1,11):
   folder=f'measurements/gate-{gate:02}-trial-{i:02}'
   items=observed(folder+'/observations.jsonl');x=next((r for r in items if r.get('kind')=='grammar'),None)
   process=read(folder+'/process.json')
   if x is None:
    rows.append(dict(gate=name,run=i,missing=True,process=process));continue
   pred,target=x['predictions'],x['targets'];mse=sum((a-b)**2 for a,b in zip(pred,target))/len(target)
   correct=sum((a>.5)==(b>.5) for a,b in zip(pred,target))
   texts=x['gate_reconstructions'];unavailable=x['grammar_reconstruction_unavailable']
   recovered=sum(not bad and Counter(source.split())==Counter((text or '').replace(chr(0),' ').split()) for source,text,bad in zip(x['inputs'],texts,unavailable))
   rows.append(dict(gate=name,run=i,answers=pred,mse=mse,correct=correct,class_bar=correct==4 and mse<.05,
    read_backs=texts,recovered=recovered,reconstruction_bar=recovered==4,unavailable=unavailable,
    contrast=pred[0]+pred[3]-pred[1]-pred[2],process=process))
 for i in range(1,11):
  folder=f'measurements/sum-{i:02}';x=read(folder+'/measurement.json');process=read(folder+'/process.json')
  rows.append(dict(gate='sum',run=i,missing=True,process=process) if x is None else dict(x,gate='sum',run=i,process=process))
 return rows

def table():
 import importlib.util
 spec=importlib.util.spec_from_file_location('measure_receipt',HERE/'measure.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 rows=[]
 for gate,selector in m.SELECTORS.items():
  folder=HERE/'measurements'/('gate-%02d-trial-01'%gate if gate in (5,6) else 'gate-%02d'%gate)
  result=json.loads((folder/'run/result.json').read_text()) if (folder/'run/result.json').exists() else {}
  by_node={}
  for worker in result.get('workers',[]):
   for r in worker.get('reports',[]):
    old=by_node.get(r['nodeid'])
    if old is None or r['outcome'] in ('failed','error'):by_node[r['nodeid']]=r
  rows.append(dict(gate=gate,selector=selector,counts=dict(Counter(r['outcome'] for r in by_node.values())),
    cases=list(by_node.values()),process=read(str(folder.relative_to(HERE)/'process.json'))))
 return rows

def main():
 gates=gate_rows();named=table();mm=[]
 for i in range(1,11):
  value=read(f'measurements/mm-{i:02}/measurement.json')
  process=read(f'measurements/mm-{i:02}/process.json')
  finished=bool(value and value.get('completed_epochs')==900 and 'after_900_updates_mse' in value and process and process['exit_code']==0)
  mm.append(dict(run=i,**value,process=process,complete=finished) if value else dict(run=i,missing=True,process=process,complete=False))
 counts={}
 for name,field in [('class','class_bar'),('reconstruction','reconstruction_bar'),('sum','sum_bar')]:
  own=[r for r in gates if r['gate']==name];counts[name]=dict(observed=sum(not r.get('missing',False) for r in own),passed=sum(bool(r.get(field,False)) for r in own))
 mse=[r['ending_training_mse'] for r in mm if r['complete']]
 native_root='native-stage1/BasicModel_answers_tied_benchmark-ownership/'
 native=read('native-driver/process.json')
 def audit(folder):
  own=read(folder+'ownership.json');events=observed(folder+'events.jsonl');rank=[r for r in events if r['kind']=='activated_word_ranking']
  return dict(ownership=own,geometry_start=read(folder+'geometry-start.json'),geometry_end=read(folder+'geometry-end.json'),ranking=rank)
 attribution=[]
 for arm in ('R','RE','RA','REA'):
  for i in range(1,11):
   x=read(f'attribution/{arm}-{i:02}/measurement.json')
   if x:attribution.append(dict(arm=arm,run=i,**x,process=read(f'attribution/{arm}-{i:02}/process.json')))
 report=dict(gates=gates,counts=counts,xor_table=named,mm_grammar=dict(runs=mm,completed_runs=len(mse),median_ending_mse=statistics.median(mse) if mse else None),native_process=native,
  native_outcome=read(native_root+'outcome.json'),native_audit=audit(native_root),xor_audit=audit('xor-ownership/'),
  attribution_decision=read('attribution-decision.json'),attribution=attribution,
  extras=read('extra-cases/complete.json'),sweep=read('full-sweep/receipt.json'),complete=read('campaign-complete.json'))
 (HERE/'results.json').write_text(json.dumps(report,indent=2))
 def fmt(x):return f'{x:.6g}' if isinstance(x,(int,float)) else str(x)
 lines=['# Closing measurements','', 'Generated from first saved outcomes by `report_results.py`. No model is rerun.','',
  '| Campaign | Observed | Passes |','|---|---:|---:|']
 for name,row in counts.items():lines.append(f'| {name} | {row["observed"]}/10 | {row["passed"]}/10 |')
 lines+=['','Answers below follow `hello world`, `hello there`, `loving world`, `loving there`. Contrast is first + fourth − second − third.','']
 for name in ('class','reconstruction','sum'):
  lines+=['## '+name,'','| Run | Four answers | MSE | Read-backs (same order) | Contrast | Seconds | Peak GiB |','|---:|---|---:|---|---:|---:|---:|']
  for row in gates:
   if row['gate']!=name:continue
   if row.get('missing'):lines.append(f'| {row["run"]} | missing outcome | — | see process record | — | — | — |');continue
   process=row.get('process') or {}
   seconds=process.get('elapsed_seconds','—');peak=process.get('peak_memory_bytes')
   lines.append(f'| {row["run"]} | '+', '.join(map(fmt,row['answers']))+' | '+fmt(row['mse'])+' | '+' / '.join(str(x) for x in row['read_backs'])+' | '+fmt(row['contrast'])+' | '+fmt(seconds)+' | '+fmt(peak/2**30 if peak is not None else '—')+' |')
 lines+=['','## Named XOR table','','| Selector | Counts |','|---|---|']
 for row in named:lines.append('| `'+row['selector']+'` | '+str(row['counts'])+' |')
 lines+=['','### Every named case','','| Case | Outcome |','|---|---|']
 for row in named:
  for case in row['cases']:
   lines.append('| `'+case['nodeid']+'` | '+case['outcome']+' |')
 lines+=['','## MM_grammar','','| Run | Completed epochs | Ending training MSE | Final evaluation MSE | Complete | Seconds | Peak GiB |','|---:|---:|---:|---:|---|---:|---:|']
 for row in mm:
  process=row.get('process') or {};peak=process.get('peak_memory_bytes')
  lines.append('| '+str(row['run'])+' | '+str(row.get('completed_epochs',0))+' | '+fmt(row.get('ending_training_mse','missing'))+' | '+fmt(row.get('after_900_updates_mse','missing'))+' | '+str(row['complete'])+' | '+fmt(process.get('elapsed_seconds','—'))+' | '+fmt(peak/2**30 if peak is not None else '—')+' |')
 lines+=['', 'Median ending MSE: '+fmt(report['mm_grammar']['median_ending_mse'])+'.','']
 if attribution:
  lines+=['## Attribution','','Ten fresh, unpaired unseeded runs per arm. R = reconstruction; E = expectation; A = answer. '
   'Without A the reader is not trained and the answer is omitted from trial comparison; its class result is diagnostic. '
   'Fixture, epoch budget and bars are unchanged. The receipt-local writer patches are saved in each run directory.','',
   '| Arm | Completed observations | Class bar | Reconstruction bar | Both existing bars |','|---|---:|---:|---:|---:|']
  for arm in ('R','RE','RA','REA'):
   own=[r for r in attribution if r['arm']==arm]
   lines.append('| '+arm+' | '+str(len(own))+'/10 | '+str(sum(bool(r['class_bar']) for r in own))+'/10 | '+str(sum(bool(r['reconstruction_bar']) for r in own))+'/10 | '+str(sum(bool(r['class_bar']) and bool(r['reconstruction_bar']) for r in own))+'/10 |')
  for arm in ('R','RE','RA','REA'):
   lines+=['','### '+arm,'','| Run | Four answers | MSE | Read-backs | Contrast | Class / reconstruction | Seconds | Peak GiB |','|---:|---|---:|---|---:|---|---:|---:|']
   for row in attribution:
    if row['arm']!=arm:continue
    process=row.get('process') or {};peak=process.get('peak_memory_bytes')
    lines.append('| '+str(row['run'])+' | '+', '.join(map(fmt,row['answers']))+' | '+fmt(row['mse'])+' | '+' / '.join(str(x) for x in row['read_backs'])+' | '+fmt(row['contrast'])+' | '+str(row['class_bar'])+' / '+str(row['reconstruction_bar'])+' | '+fmt(process.get('elapsed_seconds','—'))+' | '+fmt(peak/2**30 if peak is not None else '—')+' |')
 (HERE/'measurements.md').write_text('\n'.join(lines))
 print(json.dumps(dict(counts=counts,median_ending_mse=report['mm_grammar']['median_ending_mse'],complete=report['complete'])))

if __name__=='__main__':main()
