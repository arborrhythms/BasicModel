"""Inspect stored focus boundaries without training or changing a model.

Run with the repository Python. Optional first argument: enabled/results.json.
The input contains the exact normalized coordinates and eligibility masks;
the production region reader supplies the same float32 boundary derivative.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault('BASICMODEL_DEVICE', 'cpu')
sys.path.insert(0, str(ROOT / 'bin'))

import torch
from AttentionTraversal import focus_regions
from SpacetimeAttention import AttentionRegions, read_regions


source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / 'output/item6-1-objective-final-gates/enabled/results.json')
report = []
for row in json.loads(source.read_text()):
    for phase in ('initial', 'final'):
        field = row[phase]['trials'][0]
        where, when = [torch.tensor(value, requires_grad=True)
                       for value in field['reads'][0]['regions']]
        slots_where, slots_when = (torch.tensor(field[name]) for name in ('where', 'when'))
        valid = torch.tensor(field['eligible'])
        extent = torch.stack((slots_where, slots_when), 2)
        lower = torch.where(valid[..., None], extent[..., 0], torch.inf).amin(1)
        upper = torch.where(valid[..., None], extent[..., 1], -torch.inf).amax(1)
        softness = (upper - lower).clamp_min(torch.finfo(torch.float32).eps) * .02
        value = torch.ones(*valid.shape, 1)
        weight = read_regions(focus_regions(AttentionRegions(where, when)), value,
            slots_where, slots_when, valid, softness=softness).union(value)
        dw, dt = torch.autograd.grad(weight.sum(), (where, when))
        report.append(dict(stage=row['stage']['name'], phase=phase,
            admitted=weight.detach().sum().item(),
            where_edge_gradient_l1=dw.abs().flatten(1).sum(-1).tolist(),
            when_edge_gradient_l1=dt.abs().flatten(1).sum(-1).tolist()))
print(json.dumps(report, indent=2))
