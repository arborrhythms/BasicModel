"""Development only: an ordinary eight-row optimizer batch, no seed."""
from pathlib import Path
import json, sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
torch.set_num_threads(1)
from test_math_chain import build_model
from math_chain_corpus import MathChainCorpus,flatten
from MathChainTraining import present
config=HERE/'development-compose.xml'
config.write_text((ROOT/'data/MM_math_chain.xml').read_text().replace('<ltmCapacity>1048576</ltmCapacity>','<ltmCapacity>8192</ltmCapacity>').replace('<attentionBudget>64</attentionBudget>','<attentionBudget>0</attentionBudget>'))
model=build_model(config)
data=model.inputSpace.data
docs=MathChainCorpus().counting()[::2][:8]
texts,labels,addresses=flatten(docs,supplied=False)
data.train_input,data.train_output=texts,[torch.zeros(1) for _ in texts]
data.text_answers['train']=labels
data.source_addresses['train']=[dict(a,split='train') for a in addresses]
def observe(model,*_):
    audit=model._last_sentence_credit
    print('departure',audit['departure']['compose_round'].tolist(),flush=True)
    print('costs',audit['costs'].tolist(),flush=True)
try:
    print(present(model,split='train',optimizer=model.getOptimizer(lr=.001),after_batch=observe),flush=True)
finally:
    model.End()
