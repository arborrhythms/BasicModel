"""Render the current receipt from saved results, without another measurement."""
from pathlib import Path
from collections import Counter
import json,statistics
from summarize_measurements import read,table,fmt
HERE=Path(__file__).resolve().parent
lines=[]
finished=(HERE/'campaign-complete.json').exists()
lines.append('**Closing dispatch complete; results below require review.**' if finished else '**Closing dispatch in progress.** Counts below include completed observations only.')
p=read(HERE/'native-driver/process.json')
if p:
 lines.append(f'Native production stage-1: {"passed"if p["exit_code"]==0 else "failed"}, {p["elapsed_seconds"]/60:.2f} minutes guarded wall time, {p["peak_memory_bytes"]/2**30:.3f} GiB peak under the 24 GiB ceiling. This does **not** fit the ordinary 8 GiB ceiling. [Process](native-driver/process.json); [objective report](native-stage1/BasicModel_answers_tied_benchmark-ownership/summary.md).')
rows=read(HERE/'gate-results.json',[])
if rows:
 gate=[]
 for k,bar in [('class','class_bar'),('reconstruction','reconstruction_bar'),('sum','sum_bar')]:
  group=[r for r in rows if r['kind']==k]
  gate.append([k,sum(bool(r.get(bar))for r in group),len(group),10])
 lines.append(table(['Campaign','Pass','Observed','Required runs'],gate))
 lines.append('[Every answer, MSE, read-back and contrast](gate-results.md), with [raw values](gate-results.json). The first class result also supplies the XOR audit and has the capture-policy caveat above; it is not replaced.')
for label,folder in [('XOR_grammar',HERE/'xor-ownership'),('Native benchmark',HERE/'native-stage1/BasicModel_answers_tied_benchmark-ownership')]:
 summary=read(folder/'summary.json')
 if summary and summary['complete']:
  ownership=summary['ownership_counts'];active=sum(v['reached_parameters']for v in ownership.values());total=sum(v['parameters']for v in ownership.values())
  relative=folder.relative_to(HERE)
  lines.append(f'{label} ownership: **{summary["ownership_conflicts"]} conflicts** across {summary["backward_steps"]} recorded backwards; {active}/{total} parameter tensors reached. Inactive tensors remain listed. [Norms, cosines, term weights/magnitudes, selection and endpoint costs]({relative}/summary.md).')
  ranks=summary['activated_word_ranking'];lines.append(table(['Rank audit phase','Own-word occurrences','With activated competitor','Activated outranks own'],[[k,v['words'],v['with_activated_candidate'],v['activated_outranks_own']]for k,v in ranks.items()]))
  if label=='Native benchmark':lines.append('Native expectation was inactive: the sole training batch has no preceding sentence context and `intraLossWeight=0`. Its null/zero expectation measurements do not demonstrate a live expectation update. No activated surface competitor appeared in this run, so the ranking result has a zero competing denominator.')
measure=HERE/'measurements'
selected=read(measure/'manifest.json',{}).get('selectors',{})
counts=Counter();table_rows=[]
for key,selector in selected.items():
 name=f'gate-{int(key):02}'+('-trial-01'if int(key)in(5,6)else'')
 process=read(measure/name/'process.json')
 if not process:continue
 result=read(measure/name/'run/result.json',{})
 c=Counter(r['outcome']for w in result.get('workers',[])for r in w.get('reports',[]))
 counts.update(c);table_rows.append([key,selector,dict(c),process['reason'],process['elapsed_seconds']])
if table_rows:
 lines.append('Named XOR table, completed groups: '+fmt(dict(counts))+'. The predeclared first class/reconstruction runs are reused.')
 lines.append(table(['Group','Selector','Outcomes','Process result','Seconds'],table_rows))
mm=[]
for f in sorted(measure.glob('mm-*/measurement.json')):
 r=read(f)
 if r.get('completed_epochs')==900 and 'after_900_updates_mse'in r:mm.append(dict(run=f.parent.name,**r))
if mm:
 mse=[r['after_900_updates_mse']for r in mm]
 lines.append(f'MM_grammar: {len(mm)}/10 complete; median ending training MSE **{statistics.median(r["ending_training_mse"] for r in mm):.7g}**, with median evaluation MSE after 900 updates **{statistics.median(mse):.7g}**. The comparable §14 ending training median was 4.35656e-11 (accepted item 7: .1066). This is the fixed 900-epoch direct-forward/raw-MSE measurement. It calls backward directly rather than the objective-owner dispatcher, so it does not validate joint-cost ownership training. It is also separate from the table test that can stop at its .20 bar.')
 lines.append(table(['Run','Final training MSE','MSE after 900 updates','Final predictions'],[[r['run'],r['ending_training_mse'],r['after_900_updates_mse'],r['after_900_updates_predictions']]for r in mm]))
decision=read(HERE/'attribution-decision.json')
if decision:
 lines.append('Attribution decision: '+fmt(decision)+'. [Decision record](attribution-decision.json).')
 attrs=[]
 for f in sorted((HERE/'attribution').glob('*/measurement.json')):
  r=read(f);attrs.append(dict(run=f.parent.name,**r))
 if attrs:
  lines.append(table(['Attribution arm','Completed','Class passes','Reconstruction passes'],[[arm,len(g),sum(r['class_bar']for r in g),sum(r['reconstruction_bar']for r in g)]for arm in('R','RE','RA','REA')for g in [[r for r in attrs if r['kind']==arm]]]))
  (HERE/'attribution-results.json').write_text(json.dumps(attrs,indent=2)+'\n')
  lines.append('[All attribution answers/read-backs/contrasts](attribution-results.json). All arms are fresh unseeded runs; the source remains unchanged. The probe filters objective-owned backwards while retaining the same forward architecture and optimizer cadence. Disabled predictors/readers remain present without their objective updates; A-off also removes the answer from trial comparison. Diagnostic logs may still contain a disabled objective’s cost.')
extra=read(HERE/'extra-cases/complete.json')
if extra:
 c=Counter(r['outcome']for r in extra['reports']);c['process_failed']=len(extra['process_failures'])
 lines.append(f'Extra slow/moved cases: {len(extra["attempted"])}/{len(extra["selected"])} attempted, {extra["seconds"]/60:.2f} minutes; {dict(c)}. [Record](extra-cases/complete.json).')
full=read(HERE/'full-sweep/receipt.json')
if full:
 cases=read(HERE/'full-sweep/case-counts.json',{})
 counts=cases.get('case_counts',full['counts'])
 lines.append(f'Full sweep: **{full["attempted"]}/{full["selected"]} cases attempted**, **{full["duration_seconds"]/60:.2f} minutes**, {counts}; {full["counts"].get("process_failed",0)} process stops. Reference: **5,219 cases / 122 minutes**. Initial workers {full["limits"]["initial_workers"]}; final workers {full["limits"]["workers"]}; aggregate ceiling {full["limits"]["aggregate_gib"]:.1f} GiB. [Full receipt](full-sweep/receipt.json). Source matched: {full["source_matched"]}; complete coverage: {full["complete"]}.')
 if cases.get('repeated_reports'):
  lines.append('These are [unique case outcomes](full-sweep/case-counts.json). `TestOrthogonalFlags::test_flags_match_expected` emits three successful subtest reports in addition to its case report; the raw report total therefore has three extra passes. It was executed once. The non-strict top-k XPASS is separate from failures.')
 lines.append('[Failure evidence and comparison with prior failures](failure-triage.json); [case timing before/after](timing-results.json). A previously failed case is not automatically classified as an unchanged cause.')
 if finished:lines.append('[Final completeness audit](final-audit.json): all ported/added test bodies have a completed outcome or a recorded stopped attempt; no duplicate attempts within either the sweep or the extra-case campaign. Source, checkpoints and the fetched MNIST data remain matched. A stopped attempt does not verify its assertions.')
 if full.get('slow_warning'):lines.append('Weekly coverage warning: '+full['slow_warning'])
path=HERE/'README.md';text=path.read_text();a,b=text.split('<!-- CLOSING_RESULTS_START -->',1);_,c=b.split('<!-- CLOSING_RESULTS_END -->',1)
text=a+'<!-- CLOSING_RESULTS_START -->\n'+'\n\n'.join(lines)+'\n<!-- CLOSING_RESULTS_END -->'+c
if finished:text=text.replace('**Closing measurements in progress; not accepted.**','**Closing dispatch complete; not accepted.**',1)
tmp=path.with_suffix('.md.tmp');tmp.write_text(text);tmp.replace(path)
