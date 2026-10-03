import os, sys, json
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/'bin'),str(root/'test')]
os.environ.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false')
import recon_bench, torch
model,*_=recon_bench._build_model(str(root/'data/MM_qa.xml'))
print('CONFIG',model.serial,model.useGrammar,getattr(model,'_sentence_ends',None),model.concept_binding,flush=True)
commit=model._commit_sentence
def observed(*args,**kw):
 print('COMMIT',args[1],args[2],flush=True)
 return commit(*args,**kw)
model._commit_sentence=observed
try:
 model.provision_ltm()
finally:
 print('STATE',getattr(model,'_sentence_ends',None),getattr(model,'_stm_post_depth',None),flush=True)
 print('ERRORS',model.errors.breakdown(),flush=True)
 model.End()
