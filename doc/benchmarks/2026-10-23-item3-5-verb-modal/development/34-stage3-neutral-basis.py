"""One first-batch timing reproduction, not a curriculum or learning gate."""
import json,sys,time,warnings,random
from pathlib import Path
sys.path[:0]=['bin','test']
import torch,numpy as np,pytest
from MeaningKernels import MeaningAction
from identity_experiment import make_model,restore_warmup
from constituent_budget import paired_updates_only
from attention_curriculum_gates import present,STAGES
from Models import BaseModel
restore=BaseModel.load_weights
native=MeaningAction.basis
def basis(self,value,**kw):
 if self.modifier:
  q=self.groups(torch.zeros((),dtype=torch.long,device=value.device))
  return q,torch.ones_like(value[...,:self.readout.shape[0]],dtype=torch.bool)
 return native(self,value,**kw)
MeaningAction.basis=basis
forward0=MeaningAction.forward
def forward(self,value,verb,*,object_value=None,group=None,object_present=None,inverse=False):
 if (not self.modifier and (object_value is None or (object_present is not None and not bool(object_present.any())))
     and (group is None or not bool((group>=0).any()))):
  return forward0(self,value,verb,inverse=inverse)
 return forward0(self,value,verb,object_value=object_value,group=group,object_present=object_present,inverse=inverse)
MeaningAction.forward=forward
out=Path('output/item3-5-verb-modal-20261023/development/34-stage3-neutral-basis');out.mkdir()
torch.set_num_threads(1);random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
with pytest.MonkeyPatch.context() as patch,warnings.catch_warnings():
 warnings.simplefilter('ignore')
 model=make_model(out,patch)
 restore_warmup(model,Path('output/item3-5-verb-modal-20261023/swap-verb/final/source-03/lessons/warmup/warmup.ckpt'),restore)
 optimizer=model.getOptimizer(lr=.003)
 stage=next(s for s in STAGES if s['stage']==3)
 with paired_updates_only(model,optimizer) as updates:
  start=time.perf_counter();initial=present(model,stage);evaluated=time.perf_counter()
  row=present(model,stage,train=True,optimizer=optimizer)
  print(json.dumps(dict(eval_seconds=evaluated-start,train_seconds=time.perf_counter()-evaluated,ce=row['lesson_cross_entropy'],updates=updates,diagnostic_only=True)),flush=True)
 model.End()
