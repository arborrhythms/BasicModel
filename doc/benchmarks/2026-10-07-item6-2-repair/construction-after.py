"""Post-measurement construction diagnostic: no training, seed or gate retry."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin')]
import torch
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from ThoughtReferences import fill, question, open_slots
from ThoughtStream import write

child=ConceptualMeaning.from_description(torch.tensor([1.,0.,0.,0.]))
supplied=replace(child,role_refs=(('constituent',0),None,None),constituents=(child,))
goal=question(replace(child,role_refs=(None,None,None)),(('referent',0),))
resolved=fill(goal,dict(meaning=supplied,support_true=1.,support_false=0.),operation='ask')
store=TernaryTruthStore(4,capacity=8)
model=SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store))
result=dict(training_runs=0,seed=None,gate_retry=False,
    supplied_constituents=len(supplied.constituents),resolved_constituents=len(resolved.constituents),
    resolved_references=resolved.role_refs,open_slots=open_slots(resolved),
    scope='Construction-only reproduction of a dangling local reference after fill; original MM failed meaning was not captured.')
try:
    write(model,resolved,row=0)
    result.update(outcome='repaired',stored_rows=len(store))
except Exception as error:
    result.update(outcome='reproduced',error=repr(error),traceback=traceback.format_exc(),stored_rows=len(store))
(HERE/'construction-after.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='traceback'}))
