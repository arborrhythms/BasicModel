"""Observe the hidden EMA writer before the §20 repair, with no optimizer step."""
from pathlib import Path
import json, torch
from test_mm_xor import _fresh_model
ROOT=Path(__file__).resolve().parents[3]
model, _, data = _fresh_model(str(ROOT/'data/XOR_grammar.xml'))
owner = model._concept_owner()
cb = owner.similarity_codebook
before = cb.W.detach().clone()
cluster = getattr(cb.vq, 'cluster_size', None)
before_cluster = None if cluster is None else cluster.detach().clone()
model.train()
raw, target = next(iter(data.data_loader(split='train', num_streams=4)))
with torch.no_grad():
    model.forward(model.inputSpace.prepInput(raw))
after_cluster = getattr(cb.vq, 'cluster_size', None)
result=dict(parameter=isinstance(cb.W,torch.nn.Parameter), ema_update=cb.vq.ema_update,
    code_shape=list(cb.W.shape), code_drift=(cb.W-before).norm(dim=-1).tolist(),
    cluster_before=None if before_cluster is None else before_cluster.tolist(),
    cluster_after=None if after_cluster is None else after_cluster.tolist())
print(json.dumps(result),flush=True)
model.End()
assert not result['ema_update'], 'concept codes have a hidden EMA writer'
assert not any(result['code_drift']), 'forward without optimizer changed concept codes'
