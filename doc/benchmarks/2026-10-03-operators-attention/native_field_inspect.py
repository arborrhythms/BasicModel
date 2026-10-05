import torch

def test_dump_native_word_poles():
 from test_mm_xor import _fresh_model
 for filename in ('data/MM_xor.xml', 'data/XOR_grammar.xml'):
  model,_,_=_fresh_model(filename); model.eval()
  inp=model.inputSpace.prepInput(['hello world','hello there','loving world','loving there'])
  with torch.no_grad(): model.forward(inp)
  owner=model._concept_owner()
  if model._reading_word_percepts is not None:
   field=owner.cs_read_memberships(model._reading_word_percepts,model._reading_word_extents)
   print(filename,field.shape,field.tolist(), owner._cs_field_concept_ids.tolist())
   print([(x,owner.definitions.word(form=x),owner.definitions.objects(owner.definitions.word(form=x))) for x in ('hello','world','there','loving')])
