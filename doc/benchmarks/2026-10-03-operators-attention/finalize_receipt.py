"""Record the completed default run and verify this uncommitted checkpoint."""
import collections,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'test'))
import bounded_tests as bounded
p=HERE/'default-03'
r=json.loads((p/'result.json').read_text())
assert len(r['selected'])==len(r['completed'])
assert all((p/f'worker-{i:03}.process.json').exists() for i in range(33))
records={};failures={}
priority={'passed':0,'skipped':1,'xfailed':2,'failed':3}
for f in sorted(p.glob('worker-[0-9][0-9][0-9].json')):
 for report in json.loads(f.read_text())['reports']:
  name=report['nodeid'];old=records.get(name)
  if old is None or priority[report['outcome']]>priority[old['outcome']]:records[name]=report
  if report['outcome']=='failed':failures[name]=report['message']
base=json.loads((HERE/'failure-comparison.json').read_text())['baseline_results']
inherited={n:v for n,v in failures.items() if base.get(n)=='failed'}
unresolved={n:v for n,v in failures.items() if base.get(n)!='failed'}
source=json.loads((HERE/'review-source.json').read_text())
assert source==bounded.source_snapshot(ROOT)
assert source==json.loads((p/'source-manifest.json').read_text())['validated_source']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()=='802abb1acc95e1bddc8cb237b13230a336681c49'
result={'status':'incomplete; not ready to land; no commit or push','selected':len(r['selected']),'completed':len(r['completed']),
 'outcomes':dict(collections.Counter(v['outcome'] for v in records.values())),
 'source_matched':True,'inherited_reproduced_failures':inherited,'remaining_unresolved_failures':unresolved,
 'limits':r['limits'],'elapsed_seconds':r['elapsed_seconds'],'peak_aggregate_memory_bytes':r['peak_aggregate_memory_bytes'],
 'note':'Unique node counts; test_use_flags reports four passing subcases for one selected node. Inherited classification uses the saved published-HEAD subset, not a full baseline sweep.'}
(HERE/'checkpoint-validation.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'remaining-failures.json').write_text(json.dumps(failures,indent=2)+'\n')
lines=['# Remaining failures at the review checkpoint','',
 f"The frozen default suite completes all {result['completed']:,} selected nodes. "
 f"Outcomes: {result['outcomes']}. {len(inherited)} failures reproduced at published HEAD; "
 f"{len(unresolved)} remain outside that reproduced set. None is waived.",'',
 '## Unresolved candidate failures','']
for name in unresolved:lines.append('- `'+name+'`')
lines.extend(['','## Previously reproduced failures',''])
for name in inherited:lines.append('- `'+name+'`')
lines.extend(['','Complete messages are in [remaining-failures.json](remaining-failures.json).',
 'The intersection change is provisional; the other candidate failures concern native output distinction and answer-conditioner gradients. These are failures to repair, not ports justified merely by a changed implementation.',''])
(HERE/'remaining-failures.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in result.items() if k not in ('inherited_reproduced_failures','remaining_unresolved_failures','limits','note')},indent=2))
