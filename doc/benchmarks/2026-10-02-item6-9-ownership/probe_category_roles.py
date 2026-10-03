import os,sys,json
from pathlib import Path
r=Path(__file__).resolve().parents[3];sys.path[:0]=[str(r/'bin'),str(r/'test')]
import torch,util
util.TheCompileBackend='none'
from test_category_em_smoke import _write_category_config,_build,_run_forwards
path=_write_category_config();m=_build(path);lang=m.languageSpace;owner=m._concept_owner()
record=lang.record_category_observations;calls=[]
def observed(trace,active):
 ids,arities,mask=trace.choices();w=active.shape[1]
 record(trace,active)
 calls.append(dict(words=int(active.sum()),online=int(mask[:,:3*w].sum()),closing=int(mask[:,3*w:].sum()),
    observations=getattr(owner,'_category_role_obs',None),ids=ids[mask].tolist(),names=list(lang.language_layer.operation_layer.op_names or [])))
lang.record_category_observations=observed
try:
 _run_forwards(m)
 result=dict(calls=calls,assigned=getattr(owner,'_category_assign',{}),
  pending={str(k):{'mass':v['mass'],'stable':v['stable']} for k,v in owner._category_learner.pending.items()},
  role_names=owner._category_roles,last_pid=getattr(owner,'_category_last_pid',None))
 Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
 print('CATEGORY',json.dumps(result),flush=True)
finally:os.unlink(path);m.End();m.symbolSpace.soft_reset()
