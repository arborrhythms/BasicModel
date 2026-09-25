"""Audit the pre-fix captures and separate byte-bank from inverse effects."""
import gzip
import io
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'bin'))
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from Models import BasicModel

def read(name):
    return torch.load(io.BytesIO(gzip.decompress((HERE/'diagnostics'/name).read_bytes())),weights_only=False)
p=read('item9-packed-capture.pt.gz')[1]
s=read('item9-single-capture.pt.gz')[1:]
pr=read('item9-packed-recon.pt.gz')[-1]['arguments']
sr=[x['arguments'] for x in read('item9-single-recon.pt.gz')[1:]]
owner=SimpleNamespace(_BYTE_ASSIGNMENT_TAU=.1)
B,W,D=p['_tensor_pushed_ideas'].shape
atoms=p['input._ar_concept_lookup_atoms']
bank=F.normalize(F.pad(atoms,(0,D-atoms.shape[-1])),dim=-1)
ids=p['input._packed_sentence_ids']
valid=p['input._ar_bank_valid']
records=[]
for b in range(B):
 for sid in range(2):
    mask=p['input._word_active_mask'][b] & (ids[b]==sid)
    single_mask=s[sid]['input._word_active_mask'][b]
    sealed=pr[2][b,sid].reshape(3,D) if sid==0 else pr[4][b]
    expected=sr[sid][4][b]
    scope=torch.cat((ids[b]==sid,ids[b]==sid))
    costs={name:[] for name in ('whole_bank','sentence_bank')}
    for w in mask.nonzero().flatten().tolist():
        for name,selected in [('whole_bank',valid[b:b+1]),('sentence_bank',valid[b:b+1]&scope[None,:,None])]:
            value=BasicModel._byte_word_cost(owner,p['_recon_ideas'][b:b+1,w],torch.tensor(w),bank[b:b+1],
                p['input._ar_bank_bytes'][b:b+1],selected,p['input._ar_target_word_bytes'][b:b+1],
                p['input._ar_target_word_mask'][b:b+1],True)
            costs[name].append(float(value))
    records.append(dict(row=b,sentence=sid,actual_seal_max_difference=float((sealed-expected).abs().max()),
       pre_fix_recovered_max_difference=float((p['_recon_ideas'][b,mask]-s[sid]['_recon_ideas'][b,single_mask]).abs().max()),
       original_packed_byte_cost=float(p['_recon_sentence_costs'][b,sid]),
       rescored_same_recovered_ideas={name:sum(values)/len(values) for name,values in costs.items()},
       single_byte_cost=float(s[sid]['_recon_sentence_costs'][b,0])))
result=dict(method='Same pre-fix packed recovered ideas, changing only byte candidate validity; no re-inversion.',sentences=records)
(HERE/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
