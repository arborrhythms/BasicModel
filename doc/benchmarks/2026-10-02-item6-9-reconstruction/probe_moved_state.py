"""Locate the retained parallel tests' field/admission observations."""
import json, torch
from configuration_fixtures import parallel_concepts
from recon_bench import _build_model
from Spaces import _concept_alloc_of
with parallel_concepts(category=True) as path:
    model,*_=_build_model(path)
cs=model.conceptualSpaces[0]
optimizer=model.getOptimizer(lr=.01)
def stats():
    def maximum(value):
        return None if not torch.is_tensor(value) else [list(value.shape),float(value.abs().max())]
    store=_concept_alloc_of(cs).layer()
    return dict(sparse=cs._sparse_active(),order=cs._symbolic_order,
        a0=maximum(getattr(cs,'_cs_last_a0',None)),
        rung=getattr(cs,'_cs_level_acts',None),features=store.features.nnz,
        objects=list(cs.definitions.object_ids),
        roles=[getattr(ws,'_category_n_roles',0) for ws in model.wholeSpaces],
        bridge=getattr(cs,'_priming_bridge',None),priming=maximum(cs.priming_weights()))
read=cs.cs_read_memberships
def captured(*a,**kw):
    result=read(*a,**kw)
    print('MEMBERSHIP',result.shape,float(result.max()))
    return result
cs.cs_read_memberships=captured
for epoch in range(3):
    model.runEpoch(optimizer=optimizer,batchSize=4,split='train',max_batches=1)
    print(json.dumps(dict(epoch=epoch,state=stats()),default=str),flush=True)
model.End()
