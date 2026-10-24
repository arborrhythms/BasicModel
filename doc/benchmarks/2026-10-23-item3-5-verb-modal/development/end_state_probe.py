import json,os
from pathlib import Path
import pytest,torch
@pytest.fixture(autouse=True)
def probe(monkeypatch):
 from Models import BasicModel
 original=BasicModel._commit_sentence
 def commit(self,*args,**kwargs):
  result=original(self,*args,**kwargs)
  rows=[]
  for sid,fields in self._sentence_fields.items():
   for field in fields:
    if field is None:continue
    store=self.symbolSpace.ltm_store;i=store.location(field.row_id)
    if not isinstance(i,int):continue
    read,raw=store.meaning_of(i),store.raw_meaning_of(i)
    rows.append(dict(sid=sid,index=i,row_id=field.row_id,kind=store.row(i)['kind'],order=int(store.order[i]),
     field=field.meaning.roles.tolist(),raw=raw.roles.tolist(),working=read.roles.tolist(),
     raw_bindings=repr(raw.bindings),working_bindings=repr(read.bindings),refs=repr(raw.role_refs)))
  Path(os.environ['END_STATE_PROBE']).write_text(json.dumps(rows,indent=2)+'\n')
  return result
 monkeypatch.setattr(BasicModel,'_commit_sentence',commit)
