"""Read-only diagnosis of unchanged prepared-answer and clause tests."""
import json, os
from pathlib import Path
import torch

def emit(kind, **values):
 def cv(v):
  if torch.is_tensor(v):return v.detach().cpu().tolist()
  return repr(v)
 with Path(os.environ['REVIEW15_BOUNDARY_DIAG']).open('a') as f:
  f.write(json.dumps(dict(kind=kind,**values),default=cv)+'\n')
def stats(t):
 if not torch.is_tensor(t):return None
 return dict(shape=list(t.shape),nonzero=int(t.count_nonzero()),norm=float(t.detach().norm()),requires_grad=t.requires_grad)
def pytest_sessionstart(session):
 from Models import BasicModel
 from ClauseRow import ClauseRows
 original=BasicModel.reverseOutput
 def reverse(model,u,d,**kw):
  c=original(model,u,d,**kw)
  g=torch.autograd.grad(c.percepts.sum(),c.concepts,allow_unused=True,retain_graph=True)[0] if c.percepts.requires_grad else None
  emit('answer',answer=stats(d.conceptual_answer),concepts=stats(c.concepts),percepts=stats(c.percepts),actual=stats(c.actual),word_gradient=stats(g),trace=model._last_output_walk_trace)
  return c
 BasicModel.reverseOutput=reverse
 write=ClauseRows.write_clause
 def observed(store,clause,**kw):
  before=len(store);value=write(store,clause,**kw)
  emit('write',before=before,after=len(store),row=value,children=len(clause.children),rel_type=store.rel_type[:len(store)],refs=store.refs[:len(store)],row_ids=store.row_ids[:len(store)])
  return value
 ClauseRows.write_clause=observed
