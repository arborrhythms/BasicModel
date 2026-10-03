"""Count the actual concept quantizer calls in three unseeded training batches."""
from pathlib import Path
import json, torch
from test_mm_xor import _fresh_model
ROOT=Path(__file__).resolve().parents[3]
model, _, data = _fresh_model(str(ROOT/'data/XOR_grammar.xml'))
cb=model._concept_owner().similarity_codebook
quantize=cb.quantize
calls=[]
def counted(*args,**kw):
    before=cb.W.detach().clone()
    result=quantize(*args,**kw)
    calls.append((cb.W-before).norm(dim=-1).tolist())
    return result
cb.quantize=counted
optimizer=model.getOptimizer(lr=.01)
before=cb.W.detach().clone()
cluster=getattr(cb.vq,'cluster_size',None)
cluster_before=None if cluster is None else cluster.detach().clone().tolist()
for _ in range(3):
    raw,target=next(iter(data.data_loader(split='train',num_streams=4)))
    model.runBatch(train=True,optimizer=optimizer,batchSize=4,split='train',
        batch_override=(model.inputSpace.prepInput(raw),model.outputSpace.prepOutput(target)))
cluster=getattr(cb.vq,'cluster_size',None)
print(json.dumps(dict(ema_update=cb.vq.ema_update,quantize_calls=len(calls),
    quantize_row_drift=calls,training_row_drift=(cb.W-before).norm(dim=-1).tolist(),
    cluster_before=cluster_before,cluster_after=None if cluster is None else cluster.tolist())),flush=True)
model.End()
assert not cb.vq.ema_update
