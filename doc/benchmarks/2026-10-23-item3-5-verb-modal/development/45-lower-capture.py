"""Diagnose the retained eighteenth stage-3 batch without changing a choice."""
import json,sys,random,time,warnings,traceback
from pathlib import Path
sys.path[:0]=['bin','test']
import torch,numpy as np,pytest
from identity_experiment import make_model,restore_warmup
from Models import BaseModel
from PartClosing import PartClosing
from attention_curriculum_gates import present,STAGES
from constituent_budget import paired_updates_only
base=Path('output/item3-5-verb-modal-20261023')
source=base/'swap-verb/final/source-04/lessons/stage3-capture'
out=base/'development/45-lower-capture';out.mkdir()
restore=BaseModel.load_weights;original=PartClosing._plan_one

def observed(self,value,as_argument):
    try:return original(self,value,as_argument)
    except ValueError as error:
        if value.formal_name=='lower':
            head=value.children[value.head_role-1]
            fw=self.owner.similarity_codebook.mereology.form_width
            point=value.point.detach(); old=head.point.detach()
            item=dict(error=str(error),order=value.order,as_argument=as_argument,
                shape=list(point.shape),form_width=fw,leaves=value.leaves,
                reference=value.reference,head_reference=head.reference,
                form_max_difference=float((point[:fw]-old[:fw]).abs().max()),
                meaning_max_difference=float((point[fw:]-old[fw:]).abs().max()),
                child_references=[c.reference for c in value.children])
            (out/'lower.json').write_text(json.dumps(item,indent=2)+'\n')
            torch.save(value,out/'lower.pt');print(json.dumps(item),flush=True)
        raise

PartClosing._plan_one=observed
torch.set_num_threads(1)
with pytest.MonkeyPatch.context() as patch,warnings.catch_warnings():
    warnings.simplefilter('ignore');model=make_model(out,patch)
    snapshot=torch.load(source/'last-batch-start.pt',weights_only=False,map_location='cpu')
    drift=restore_warmup(model,source/'last-batch-start.pt',restore)
    (out/'restore.json').write_text(json.dumps(drift,default=str,indent=2)+'\n')
    optimizer=model.getOptimizer(lr=.003);optimizer.load_state_dict(snapshot['optimizer'])
    torch.set_rng_state(snapshot['rng']['torch']);random.setstate(snapshot['rng']['python']);np.random.set_state(snapshot['rng']['numpy'])
    try:
        with paired_updates_only(model,optimizer):
            present(model,next(s for s in STAGES if s['stage']==3),train=True,optimizer=optimizer)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:model.End()
