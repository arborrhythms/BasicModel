"""Read-only coverage of the final grammar correction, at the authorized seed zero."""
import hashlib,json,random,sys
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
from util import init_device
from Models import BasicModel
from Interpret import InterpretLayer
from test_mm_xor import _fresh_model
init_device('cpu');random.seed(0);np.random.seed(0);torch.manual_seed(0)
model,_,data=_fresh_model(str(ROOT/'data/MM_xor.xml'))
counts=dict(word_pipeline=0,answer_leaf_slab=0,activate=0,tensor_interpret=0)
from contextlib import ExitStack
with ExitStack() as stack:
    for cls,name,label in ((BasicModel,'_run_tensor_peer_word_pipeline','word_pipeline'),
                           (BasicModel,'_answer_leaf_slab','answer_leaf_slab'),
                           (InterpretLayer,'activate','activate'),(InterpretLayer,'forward','tensor_interpret')):
        method=getattr(cls,name)
        def observed(self,*a,_method=method,_label=label,**kw):
            counts[_label]+=int(_label!='tensor_interpret' or torch.is_tensor(a[0]))
            return _method(self,*a,**kw)
        stack.enter_context(patch.object(cls,name,observed))
    raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
    _,_,value,_=model.forward(model.inputSpace.prepInput(raw))
prior=json.loads((HERE/'paired-mm/round2c-0.json').read_text())
result=dict(seed=0,training_steps=0,word_brackets=model.word_brackets,counts=counts,
    predictions=value.detach().flatten().tolist(),paired_source_manifest_sha256=hashlib.sha256((HERE/'development/second-freeze/source.json').read_bytes()).hexdigest())
result['matches_paired_first_forward']=result['predictions']==prior['trajectory'][0]['predictions']
old=json.loads((HERE/'development/second-freeze/source.json').read_text())
result['changed_runtime_files']={name:dict(paired=sha,final=hashlib.sha256((ROOT/name).read_bytes()).hexdigest())
    for name,sha in old.items() if name.startswith('bin/') and sha!=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
assert not model.word_brackets and not any(counts.values()) and result['matches_paired_first_forward']
result['coverage']='Only tensor interpretation and the word pipeline/readback changed after the paired replay. MM_xor has word_brackets=false throughout its optimizer loop; the guarded word pipeline is unreachable, and the seed-zero forward reaches none of these corrected functions.'
(HERE/'paired-mm/source-coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));model.End();model.symbolSpace.soft_reset()
