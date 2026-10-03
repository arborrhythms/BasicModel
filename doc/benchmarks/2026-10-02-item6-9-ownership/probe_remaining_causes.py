"""Receipt-only attribution of the three unchanged behavioural assertions."""
import json, os, sys, tempfile
os.environ['BASICMODEL_DEVICE'] = 'cpu'
os.environ['BASIC_AUTOLOAD'] = 'false'
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test')]
import torch, util
util.TheCompileBackend = 'none'
from test_category_em_smoke import _write_category_config, _build, _run_forwards
results = {}
path = _write_category_config()
m = _build(path)
owner = m._concept_owner()
events = []
import Spaces
stash = Spaces._stash_category_roles
def observed(ws, pids, B, N):
    obs = getattr(ws, '_category_role_obs', None)
    metas = [[ws.concept_of_percept(int(p)) for p in row] for row in pids]
    stash(ws, pids, B, N)
    events.append(dict(owner=id(ws), observations=obs, pids=pids, metas=metas,
                       pending=len(ws._category_learner.pending)))
Spaces._stash_category_roles = observed
try:
    _run_forwards(m, n=15)
    results['category'] = dict(terminal_is_owner=m.conceptualSpace is owner,
        assignments_before=dict(owner._category_assign), events=events)
    owner.Reset(hard=True)
    results['category']['assignments_after_owner_reset'] = dict(owner._category_assign)
finally:
    Spaces._stash_category_roles = stash
    m.End(); m.symbolSpace.soft_reset(); os.unlink(path)

from test_output_synthesis import _build as build, _batch
from What import What
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory)/'synthesis.xml'
    path.write_text((ROOT/'data/MM_xor.xml').read_text().replace('<architecture>',
        '<architecture><answerSynthesis>true</answerSynthesis>', 1))
    m=build(path); x,_=_batch(m)
    rows=[]
    with torch.no_grad():
        for _ in range(4):
            u=m.understand(x)
            rev,_=m.reverseReconstruct(u)
            a=m.reverseOutput(u,m.resolveAnswer(u,What.supervised(0))).actual
            ps=m.perceptualSpace
            rows.append(dict(vocabulary=len(ps.vocabulary),
                percepts=u.perceptual_context.detach().tolist(),
                concepts=u.conceptual_state.detach().tolist(),
                reverse=rev.tolist(), answer=a.tolist()))
    results['repeated_reading']=rows
    m.End();m.symbolSpace.soft_reset()
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results),flush=True)
