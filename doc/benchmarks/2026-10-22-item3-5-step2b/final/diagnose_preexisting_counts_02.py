"""Replay the failed fixture draws with current and pre-2b clause writers."""
import json, random, subprocess
from pathlib import Path
import numpy as np
import pytest, torch
from test_clause_acceptance import SentenceFixture
from ClauseRow import ClauseRows, Clause, ClausePredicate

HERE=Path(__file__).resolve().parent
namespace={}
exec(compile(subprocess.check_output(['git','show','bb0045d7a:bin/ClauseRow.py']),
             'bb0045d7a/bin/ClauseRow.py','exec'), namespace)
namespace.update(Clause=Clause, ClausePredicate=ClausePredicate)
old=namespace['ClauseRows'].write_clause
current=ClauseRows.write_clause
cases={
 'test_01_compound_vp_has_one_idea_and_readable_cat_chase_mouse':('lift',('lower','the','cat'),('verb','chased',('lower','the','mouse'))),
 'test_04_absolute_clauses_close_bottom_up_with_their_part_edges':('lift','he',('verb','said',('lift','she',('verb','said',('lift','it',('verb','is',('surface','so','beautiful'))))))),
 'test_03_particular_subject_ends_an_idea[black]':('lift',('lower','the','cat'),'black')}
results=[]
for name, tree in cases.items():
 state=torch.load(HERE/'development-01/captures'/name/'initial-rng.pt',weights_only=False)
 for label,writer in (('working',current),('pre_2b_writer',old)):
  torch.set_rng_state(state['torch']);random.setstate(state['python']);np.random.set_state(state['numpy'])
  with pytest.MonkeyPatch.context() as monkeypatch:
   ClauseRows.write_clause=writer
   f=SentenceFixture(monkeypatch)
   clause=f.clause(tree)
   row=f.store.write_clause(clause,trust=.7)
   results.append(dict(case=name,writer=label,occurrences=len(f.store),parts=len(f.store.parts),
    pairs=f.store.parts.refs[f.store.parts.indices()].tolist(),words=f.words))
ClauseRows.write_clause=current
(HERE/'preexisting-counts.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
