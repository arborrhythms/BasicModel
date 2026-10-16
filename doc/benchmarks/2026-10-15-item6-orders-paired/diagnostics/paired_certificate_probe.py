import json,sys,random
from pathlib import Path
import numpy as np
import torch
import util
util.TheCompileBackend='none'
torch.set_num_threads(1)
import SentenceCredit
from test_mm_xor import _fresh_model
original=SentenceCredit.components
rows=[]
def observe(registry,*args,**kwargs):
    result=original(registry,*args,**kwargs)
    rows.append({name:(value.detach()*weight).tolist() for name,value,weight,_,category in registry.terms() if category=='reconstruction'})
    return result
SentenceCredit.components=observe
reports=[]
for seed in (20261010,0,1):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    rows.clear()
    model,_,_= _fresh_model('data/XOR_grammar.xml')
    try:
        optimizer=model.getOptimizer(lr=.01)
        values=[]
        for epoch in range(3):
            model.runEpoch(optimizer=optimizer,batchSize=4,split='train',max_batches=1)
            values.append([r['components'][...,0].tolist() for r in model._sentence_credit_audits])
        reports.append(dict(seed=seed,values=values,terms=list(rows)))
    finally:model.End();model.symbolSpace.soft_reset()
Path(sys.argv[1]).write_text(json.dumps(dict(backend='none',reports=reports),indent=2)+'\n')
