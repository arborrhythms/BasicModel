"""Check held content without an executable VP on the frozen source."""
import json,traceback
from pathlib import Path
import torch
from Queries import QueryWorkBudget
from test_negative_expectation import _anticipating_model
HERE=Path(__file__).resolve().parent
state=torch.get_rng_state();torch.save(state,HERE/'held-idea-initial-rng.pt')
model,discourse,meaning=_anticipating_model()
registry=model.grammatical_thoughts
store=model.symbolSpace.ltm_store
index=store.append_idea(torch.ones(8),sentence_index=len(store))
result={}
try:
 with model._query_boundary_scope((0,)):
  context=model._thought_grammar_context(meaning,row=0,work=QueryWorkBudget(64),continuation=None)
  request=registry.form('ask',store.occurrence_of(index),context=context)
  work=QueryWorkBudget(64)
  output=model._selected_thought_expectation(request,row=0,work=work)
  result=dict(mask=output[-4:-1].tolist(),work=work.spent)
except Exception:
 result=dict(error=traceback.format_exc())
(HERE/'held-idea.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
