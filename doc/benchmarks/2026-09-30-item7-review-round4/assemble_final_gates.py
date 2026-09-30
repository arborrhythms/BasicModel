"""Assemble measured explicit gates only after exact-source verification."""
import json,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[2]
sys.path[:0]=[str(root/'test'),str(here.parent/'2026-09-28-item7-review')]
from bounded_tests import source_snapshot,documentation_snapshot
from review_source import supporting_inputs
source=source_snapshot(root);inputs=supporting_inputs(root)
out=here/'explicit-final3';out.mkdir(exist_ok=False)
(out/'source-manifest.json').write_text(json.dumps(dict(validated_source=source,supporting_inputs=inputs,recorded_documentation=documentation_snapshot(root)),indent=2)+'\n')
result=dict(reason='running',groups=[],selected=[],completed=[],workers=[],diagnostic_only=[])
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ('final3-item7','final3-core-gates','final3-xor-candidate'):
 path=here/name/'result.json'
 while not path.exists() or json.loads(path.read_text()).get('reason')=='running':time.sleep(2)
 assert source_snapshot(root)==source and supporting_inputs(root)==inputs
 manifest=json.loads((here/name/'source-manifest.json').read_text())
 assert manifest['validated_source']==source,name
 measured=json.loads(path.read_text())
 for group in measured['groups']:
  p=here/name/group['receipt'];run=json.loads(p.read_text())
  result['groups'].append(dict(reason=run['reason'],exit_code=run['exit_code'],receipt=str(p.relative_to(here)),reused=True))
  for key in ('selected','completed','workers'):result[key].extend(run[key])
 result['diagnostic_only'].extend(measured['diagnostic_only'])
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
result['exit_code']=int(any(g['exit_code'] for g in result['groups']))
result['reason']='gate_failures' if result['exit_code'] else 'passed'
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['reason'],len(result['selected']),len(result['completed']),flush=True)
