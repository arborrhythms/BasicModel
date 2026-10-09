"""Descriptive logger metrics over the fixed item 9b probe set.

The caller supplies readings already collected at a logging interval. This
module does no model execution, training, admission, or symbol read-back.
"""
import json
from pathlib import Path

import torch


FIXED_PROBES = json.loads((Path(__file__).resolve().parents[1] /
    'data/categorical_discrimination_probes.json').read_text())['sets']


@torch.no_grad()
def categorical_discrimination(readings, labels):
    """Return between-minus-within L2, counting each unordered pair once.

    Logging owns the host copy. A single pairwise kernel replaces the old
    per-pair device-to-host conversions; neither value is a pass threshold.
    """
    values = (readings if torch.is_tensor(readings) else torch.stack(readings))
    if values.ndim < 2 or values.shape[0] != len(labels):
        raise ValueError('one reading is required for every fixed probe')
    values = values.detach().to(device='cpu', dtype=torch.float32).flatten(1)
    within = torch.tensor([labels[i] == labels[j]
                           for i in range(len(labels))
                           for j in range(i + 1, len(labels))], device='cpu')
    nw = int(within.sum())
    nb = len(within) - nw
    if min(nw, nb) == 0:
        raise ValueError('discrimination requires within- and between-category pairs')
    distances = torch.pdist(values)
    dw, db = torch.stack((distances[within].mean(), distances[~within].mean())).tolist()
    return dict(cp=db-dw, within=dw, between=db, within_pairs=nw, between_pairs=nb)


def fixed_probe_discrimination(readings):
    """One JSON-ready logger field; zero/unknown probe readings stay included."""
    if set(readings) != set(FIXED_PROBES):
        raise ValueError('fixed probe readings must include xor and fineweb')
    return {'categorical_discrimination': {
        name: categorical_discrimination(readings[name], probes['labels'])
        for name, probes in FIXED_PROBES.items()}}
