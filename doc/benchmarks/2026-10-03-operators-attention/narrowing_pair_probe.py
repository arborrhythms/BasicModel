import torch

def test_normal_input_costs_both_narrowing_walks_before_training(monkeypatch):
 from test_mm_xor import _fresh_model
 import ModelAttention
 model,_,_=_fresh_model(); model.train()
 calls=[]; versions=[]
 original=ModelAttention.narrow_words
 def traced(*args,**kwargs):
  calls.append('explore' if kwargs.get('exploit') is not None else 'greedy')
  versions.append(tuple(p._version for p in model._stm_reducer().parameters()))
  return original(*args,**kwargs)
 monkeypatch.setattr(ModelAttention,'narrow_words',traced)
 value=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
 model._lex_embed_stem(value)
 assert calls==['greedy','explore'], calls
 assert versions[0]==versions[1]
 audit=model._last_attention_comparison
 assert audit['costs'].shape==(4,2)
 assert torch.equal(audit['wins'], (audit['departure']>=0)&(audit['costs'][:,1]<audit['costs'][:,0]))
