#!/usr/bin/env python3
"""Known native cat identity through production closing, storage and answer.

The registered reading and perfect prior are controlled mechanism inputs.
This does not assert learned recognition of an untrained surface form.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch,pytest
from types import SimpleNamespace
from dataclasses import replace
from test_clause_acceptance import SentenceFixture
from reading_fixtures import commit_reading
from Layers import SymbolExpectation
from Models import BasicModel
from Understanding import Understanding
from What import What
import ThoughtClosing

def run():
 torch.set_num_threads(1);results=[]
 for gain in (0.,.5,1.):
  with pytest.MonkeyPatch.context() as patch:
   f=SentenceFixture(patch);entry=replace(f.program('cat'),leaf_evidence=torch.tensor([[1.,0.]]))
   head=SymbolExpectation(8,8,8,concept_dim=8)
   head.set_symbol_bank(entry.leaves,identities=entry.concept_ids,orders=torch.ones(1,dtype=torch.long))
   with torch.no_grad():
    for p in head._inter_predictor.parameters():p.zero_()
    head._inter_predictor.local.bias.copy_(torch.tensor([60.,-60.]))
   host=BasicModel();host.document_expectation_gain=gain;host.expectation_gain=0.;host.reconstruct_in_loop=False
   captured=[];close=ThoughtClosing.close
   def image(*args,**kwargs):captured.append(kwargs['images']);return close(*args,**kwargs)
   patch.setattr(ThoughtClosing,'close',image)
   # A causal cold first sentence, then the same known symbol at a warm close.
   for sid in (0,1):
    commit_reading(f.language,f.registry,entry,f.store,discourse=head,sid=sid,document='cat-stream',owner=host)
   host.conceptualSpace.stm=SimpleNamespace(concept_dim=8)
   patch.setattr(host,'_what_grammar_context',lambda *a,**kw:(None,()))
   host.eval()
   field=host._sentence_fields[1][0];view=captured[-1][0]
   u=Understanding(sentence_states=(field,))
   answer=host.resolveAnswer(u,What.present(0))
   idea,ready,_,_=host._materialize_answer_idea(u,answer,What.present(0))
   index=f.store.index_of_row(field.row_id);pair=f.store.expectation_pair(index,gain=gain)
   torch.testing.assert_close(view.conceived,view.observed*(1-gain),atol=1e-7,rtol=1e-6)
   torch.testing.assert_close(idea[0],entry.end_state,atol=0,rtol=0)
   torch.testing.assert_close(pair['observation'].roles,entry.end_state,atol=0,rtol=0)
   assert ready.all() and answer.resolved
   results.append(dict(gain=gain,content_norm=float(view.conceived.norm()),raw_norm=float(view.observed.norm()),
    floor=1-gain,answer_correct=True,store_correct=True,prime=view.expected.tolist(),
    observation_occurrence=pair['observation_occurrence'],estimate_occurrence=pair['estimate_occurrence']))
 return results
if __name__=='__main__':
 result=run();(Path(__file__).parent/'native_closing_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
