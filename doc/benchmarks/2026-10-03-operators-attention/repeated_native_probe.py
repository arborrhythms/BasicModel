import torch

def test_current_native_reading(monkeypatch):
 import ModelAttention
 from test_sentence_protocol import test_repeated_word_is_not_reminted
 original=ModelAttention.native_word_poles
 def inspect(model,spans,forms,known):
  result=original(model,spans,forms,known)
  owner=model._concept_owner()
  fi=model.perceptualSpace._forward_input
  print('NATIVE',forms,known.tolist(),spans.tolist(),result.tolist())
  print('INPUT', {key:(value.tolist() if torch.is_tensor(value) else str(value)) for key,value in fi.items() if key in ('native_indices','indices','native_part_spans','part_spans','word_texts')})
  print('DEFS', [(cid, owner.definitions.description(cid),owner.definitions.objects(cid)) for cid in owner.definitions.word_ids])
  if hasattr(owner,'_cs_field_concept_ids'): print('FIELD',owner._cs_field_concept_ids.tolist(),owner._cs_order0_raw.tolist())
  return result
 monkeypatch.setattr(ModelAttention,'native_word_poles',inspect)
 test_repeated_word_is_not_reminted()
