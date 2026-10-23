"""Replay all four failing argument-read fixtures with both clause writers."""
import json,random,subprocess
from pathlib import Path
import numpy as np
import pytest,torch
from ClauseRow import ClauseRows,Clause,ClausePredicate
from Understanding import AnswerProgram
from reading_fixtures import finish_reading
from test_clause_acceptance import SentenceFixture
HERE=Path(__file__).resolve().parent
namespace={}
exec(compile(subprocess.check_output(['git','show','bb0045d7a:bin/ClauseRow.py']),
             'bb0045d7a/bin/ClauseRow.py','exec'),namespace)
namespace.update(Clause=Clause,ClausePredicate=ClausePredicate)
current=ClauseRows.write_clause
results=[]
try:
 for registry in (False,True):
  for relation in ('part','implies'):
   name=f'test_test_predicate_identity.py--test_first_three_slot_predicate_occurrence[{registry}-{relation}]'
   state=torch.load(HERE.parent/'development-10/captures'/name/'initial-rng.pt',weights_only=False)
   for label,writer in (('current',current),('pre_2b_writer',namespace['ClauseRows'].write_clause)):
    torch.set_rng_state(state['torch']);random.setstate(state['python']);np.random.set_state(state['numpy'])
    ClauseRows.write_clause=writer
    with pytest.MonkeyPatch.context() as patch:
     f=SentenceFixture(patch)
     left=f.store.write_clause(f.clause(('lift','cat','runs')))
     right=f.store.write_clause(f.clause(('lift','dog','sleeps')))
     predicate=f.registry.clause_reference(relation)
     leaves=torch.stack((f.store.slots[left,0],f.registry._payload(predicate),f.store.slots[right,0]))
     refs=torch.tensor([int(f.store.row_ids[left]),predicate[1],int(f.store.row_ids[right])])
     entry=AnswerProgram(rows=torch.full((3,),-1),word_rows=torch.full((3,),-1),
      word_ids=torch.full((3,),-1),activations=torch.ones(3),leaves=leaves,
      actions=torch.tensor([[0,-1,0],[0,-1,1],[0,-1,2]]),targets=torch.tensor([-1]),end_state=leaves,concept_ids=refs)
     clause=finish_reading(f.language,entry,registry=f.registry if registry else None)
     row=f.store.write_clause(clause)
     arguments=f.store.row(row)['refs'].tolist()
     readings=[]
     for role,argument,source in zip((0,2),arguments,(left,right)):
      retained=f.store.raw_meaning_of(source)
      native=f.store._concept_point(argument)
      assert retained.role_refs[0]==('sym',argument)
      torch.testing.assert_close(retained.roles[0],leaves[role])
      readings.append(dict(argument=argument,source=source,reference=retained.role_refs[0],
       native=None if native is None else native.tolist(),content=retained.roles[0].tolist()))
     results.append(dict(case=name,writer=label,readings=readings))
finally:
 ClauseRows.write_clause=current
(HERE/'predicate-arguments.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
