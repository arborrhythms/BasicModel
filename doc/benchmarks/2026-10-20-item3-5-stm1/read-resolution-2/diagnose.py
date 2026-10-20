# Diagnostic only; not an acceptance test or a learning result.
import os, sys, json, traceback
from pathlib import Path
sys.path[:0] = ['bin', 'bin/etc', 'test']
os.environ['MODEL_COMPILE'] = 'eager'
import torch, pytest
from reading_fixtures import use_eager_reading
from ClauseRow import ClauseRows
from ThoughtReferences import bindings, open_slots
from test_math_chain_section14 import test_forced_ordinary_pending_premise_in_both_orders
folder=Path('doc/benchmarks/2026-10-20-item3-5-stm1/read-resolution-2')
folder.mkdir(exist_ok=True)
torch.manual_seed(int(Path('doc/benchmarks/2026-10-20-item3-5-stm1/read-resolution/seed.txt').read_text()))
mp=pytest.MonkeyPatch();use_eager_reading(mp)
stream=(folder/'closing.jsonl').open('w')
def native(v):
    if torch.is_tensor(v):return v.detach().tolist()
    if hasattr(v,'identity'):return v.identity
    return str(v)
def composition(c):
    return None if c is None else dict(ref=c.reference,op=c.operation,order=c.order,children=[composition(x) for x in c.children])
def brief(c):
    return dict(composition=composition(c.composition),relation=c.relation, refs=c.refs, marks=bindings(c.meaning), open=open_slots(c.meaning), children=[brief(x) for x in c.children])
orig=ClauseRows.write_clause
active=0
def write(self,clause,**kw):
    global active
    active+=1
    try:out=orig(self,clause,**kw)
    finally:active-=1
    if active==0:
        p=self.parts
        from Spaces import _concept_alloc_of
        a=_concept_alloc_of(self._concept_owner())
        stream.write(json.dumps(dict(clause=brief(clause),result=out,singletons=sorted(a.singletons),words={k:sorted(v) for k,v in a.word_forms.items()},parts=[dict(id=i,refs=p.refs[i],truth=[p.truth_plus[i],p.truth_minus[i]]) for i in p.indices().tolist()]),default=native)+'\n');stream.flush()
    return out
mp.setattr(ClauseRows,'write_clause',write)
(folder/'seed.txt').write_text(str(torch.initial_seed())+'\n')
try:test_forced_ordinary_pending_premise_in_both_orders(folder,False,None)
except Exception:traceback.print_exc()
finally:mp.undo();stream.close()
