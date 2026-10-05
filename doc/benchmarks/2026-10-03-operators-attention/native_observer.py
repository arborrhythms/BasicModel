import pytest, torch
@pytest.fixture(autouse=True)
def observe_native(monkeypatch):
 from Models import BasicModel
 old=BasicModel.reverseOutput
 def inspect(self,understanding,derivation,*a,**k):
  print('NATIVE answer',derivation.conceptual_answer.norm(dim=-1).tolist(), 'depth',derivation.conceptual_depth.tolist() if hasattr(derivation,'conceptual_depth') and derivation.conceptual_depth is not None else None)
  lang=self.languageSpace
  print('NATIVE ops',getattr(lang,'_generate_unary_names',None),getattr(lang,'_generate_binary_names',None))
  for i,s in enumerate(understanding.sentence_states):
   print('NATIVE state',i,str(s)[:1500])
  result=old(self,understanding,derivation,*a,**k)
  print('NATIVE output',result.concepts.norm(dim=-1).tolist())
  return result
 monkeypatch.setattr(BasicModel,'reverseOutput',inspect)
