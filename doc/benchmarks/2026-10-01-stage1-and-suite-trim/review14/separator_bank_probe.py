from pathlib import Path
import os,sys,json
from types import MethodType
ROOT=Path(__file__).resolve().parents[4]
os.environ.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false')
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch,util
from test_mm_xor import _fresh_model
util.TheCompileBackend='none'
def loop(cond,body,args):
    while bool(cond(*args)):args=body(*args)
    return args
torch.while_loop=loop
model,_,_=_fresh_model(str(ROOT/'data/XOR_grammar.xml'))
records=[]
def value(x):
    return x.detach().cpu().tolist() if torch.is_tensor(x) else repr(x)
def state(when):
    isp=model.inputSpace;fi=model.perceptualSpace._forward_input
    records.append(dict(when=when,active=value(isp._word_active_mask),aligned=model._aligned_serial_word_mode(),
        **{n:value(getattr(isp,n,None)) for n in ['_ar_target_word_bytes','_ar_target_word_mask','_ar_grammar_object_rows','_ar_grammar_leaf_mask']},
        percepts={n:value(fi.get(n)) for n in ['tokens','word_texts','part_spans']}))
stage=model._stage_mixing_reconstruction_bank
def observed_stage():
    state('before staging');stage();state('after staging')
model._stage_mixing_reconstruction_bank=observed_stage
commit=model._commit_sentence
def observed_commit(state_,sid,active,*rest):
    state('before readback');model.reconstruct_grammar_sentence(state_,sid,active);state('after readback')
    return commit(state_,sid,active,*rest)
model._commit_sentence=observed_commit
with torch.no_grad(): model(model.inputSpace.prepInput(['hello world','hello there','loving world','loving there']))
state('after forward')
Path(__file__).with_suffix('.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records:
    print(r['when'],'active',r['active'][0], 'targets',r['_ar_target_word_bytes'][0] if isinstance(r['_ar_target_word_bytes'],list) else None,'texts',r['percepts']['word_texts'])
model.End()
