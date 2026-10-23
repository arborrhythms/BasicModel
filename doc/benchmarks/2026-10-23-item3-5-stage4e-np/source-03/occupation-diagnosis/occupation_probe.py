import gzip,json,os
from pathlib import Path
import pytest
from arrival_capture import plain
@pytest.fixture(autouse=True)
def probe(request,monkeypatch):
 if 'test_kept_readings_alone_supply_occupation_counts' not in request.node.name:return
 from PartsOfSpeech import PartsOfSpeech
 from PartClosing import PartClosing
 from Models import BasicModel
 from identity_keep_probe import clause_graph
 root=Path(os.environ['CONTRACT_CAPTURE']).parent
 phase=['setup'];records=[]
 def record(value):
  records.append(plain(value));(root/'probe.json.gz').write_bytes(gzip.compress(json.dumps(records,indent=2).encode(),mtime=0))
 add,observe,commit,score,keep=PartsOfSpeech.add,PartsOfSpeech.observe_kept,PartClosing._commit,BasicModel._sentence_path_cost,BasicModel._commit_sentence
 def a(owner,word,argument):
  record(dict(call='add',phase=phase[0],word=word,argument=argument));return add(owner,word,argument)
 def o(owner,program,*args):
  prior=phase[0];phase[0]='kept-program'
  try:return observe(owner,program,*args)
  finally:phase[0]=prior
 def c(plan):
  record(dict(call='closing-plan',committed=plan._committed,occupations=plan._occupations,
   plans=[dict(symbol=r['symbol'],seat=r['seat'],word=r['value'].word_id,operation=r['value'].operation,formal=r['value'].formal_name,reference=r['value'].reference,child_words=[x.word_id for x in r['value'].children]) for r in plan._plan]))
  prior=phase[0];phase[0]='closing'
  try:return commit(plan)
  finally:phase[0]=prior
 def s(model,*args,**kwargs):
  result=score(model,*args,**kwargs)
  record(dict(call='trial',trial=model._sentence_trial,clauses=[clause_graph(v) for v in result[2]['clauses']],answer=model._sentence_answer_cost,registry=model._sentence_cost_registry.breakdown()))
  return result
 def k(model,state,sid,active,observations,predictions,wins):
  record(dict(call='keep',credit=model._last_sentence_credit));return keep(model,state,sid,active,observations,predictions,wins)
 monkeypatch.setattr(PartsOfSpeech,'add',a);monkeypatch.setattr(PartsOfSpeech,'observe_kept',o);monkeypatch.setattr(PartClosing,'_commit',c);monkeypatch.setattr(BasicModel,'_sentence_path_cost',s);monkeypatch.setattr(BasicModel,'_commit_sentence',k)
