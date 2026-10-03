"""No model/training: reconstruct pre-step codes from saved §24 displacements."""
from pathlib import Path
import json
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'2026-10-03-item6-9-corrections/xor-ownership'
start = torch.load(OLD/'geometry-start-stage-1.pt', weights_only=True)
end = torch.load(OLD/'geometry-end-stage-1.pt', weights_only=True)
codes = start['codes'].double().numpy().copy()
rows = list(start['rows'])
world, there = rows.index(1), rows.index(2)
def cosine(a,b):
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d else None
records = []
for event in map(json.loads, (OLD/'events.jsonl').read_text().splitlines()):
    if event['kind'] != 'optimizer_displacement':
        continue
    parameter = next((p for p in event['parameters']
                      if p['parameter'] == 'conceptualSpaces.1.layers.3.W' and p['gradient_present']), None)
    if parameter is None:
        continue
    with np.load(OLD/event['file']) as data:
        key = parameter['key']
        g, delta = data[key+'_gradient'].astype(float), data[key+'_displacement'].astype(float)
        if key+'_rows' in data:
            indices = data[key+'_rows'].tolist()
            full_g, full_delta = np.zeros_like(codes), np.zeros_like(codes)
            for i,row in enumerate(indices):
                if row in rows:
                    full_g[rows.index(row)] = g[i]
                    full_delta[rows.index(row)] = delta[i]
            g, delta = full_g, full_delta
        difference = codes[there]-codes[world]
        # Tangent toward the other code isolates angular alignment from norm.
        a,b = codes[world],codes[there]
        tangent = b/np.linalg.norm(b)-(a@b)/(a@a)/np.linalg.norm(b)*a
        records.append(dict(step=event['step'], before_cosine=cosine(a,b),
            gradient_toward_other=cosine(g[world],difference),
            descent_toward_other=cosine(-g[world],difference),
            descent_angular_toward_other=cosine(-g[world],tangent),
            displacement_toward_other=cosine(delta[world],difference),
            displacement_angular_toward_other=cosine(delta[world],tangent)))
        codes += delta
        records[-1]['after_cosine'] = cosine(codes[world],codes[there])
error=float(np.max(np.abs(codes-end['codes'].double().numpy())))
summary = dict(steps=len(records), recovered_endpoint_max_error=error,
    start_cosine=records[0]['before_cosine'],end_cosine=records[-1]['after_cosine'],
    interpretation='Descent is -gradient. Positive tangent/descent cosine supports angular merging; raw gradient has the opposite sign. These are total reconstruction gradients, not a decomposition by candidate pair.')
for name in records[0]:
    if 'toward' in name:
        values=[r[name] for r in records if r[name] is not None]
        summary[name]=dict(positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),
                           count=len(values),median=float(np.median(values)),mean=float(np.mean(values)))
(HERE/'saved-merging.json').write_text(json.dumps(dict(summary=summary,steps=records),indent=2)+'\n')
print(json.dumps(summary,indent=2))
