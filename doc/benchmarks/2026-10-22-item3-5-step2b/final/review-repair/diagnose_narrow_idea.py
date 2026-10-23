"""Compare the sharpened no-occurrence-reference check with the pre-2b writer."""
import json,random,subprocess
from pathlib import Path
import numpy as np
import torch
from ClauseRow import ClauseRows, Clause, ClausePredicate
from test_clause_storage import clause_store,idea_clause
HERE=Path(__file__).resolve().parent
namespace={}
exec(compile(subprocess.check_output(['git','show','bb0045d7a:bin/ClauseRow.py']),
             'bb0045d7a/bin/ClauseRow.py','exec'),namespace)
namespace.update(Clause=Clause,ClausePredicate=ClausePredicate)
current=ClauseRows.write_clause
state=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
torch.save(state,HERE/'narrow-initial-rng.pt')
results=[]
try:
 for name,writer in (('current',current),('pre_2b_writer',namespace['ClauseRows'].write_clause)):
  torch.set_rng_state(state['torch']);random.setstate(state['python']);np.random.set_state(state['numpy'])
  ClauseRows.write_clause=writer
  store,_=clause_store()
  first=store.write_clause(idea_clause())
  second=store.write_clause(idea_clause(refs=(int(store.row_ids[first]),3,-1)))
  refs=store.meaning_of(second).role_refs
  results.append(dict(writer=name,first=first,second=second,role_refs=refs,
      no_occurrence_reference=not any(ref is not None and ref[0]=='ltm' for ref in refs)))
finally:
 ClauseRows.write_clause=current
(HERE/'narrow-idea.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(item['no_occurrence_reference'] for item in results)
print(json.dumps(results,indent=2))
