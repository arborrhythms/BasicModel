"""Capture native decomposition initializations at the two recorded seeds.

Use as a pytest plugin with DECOMPOSITION_SNAPSHOT_DIR set. There is no
parameter migration: each snapshot belongs to the current model layout.
These forced mechanism replays do not constitute a learning experiment.
"""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from Models import BasicModel

SEEDS = {1: 4199722147, 2: 2428440506}
ACTIVE = None
CREATE = BasicModel._create_per_stage


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global ACTIVE
    ACTIVE = None
    if item.nodeid.startswith('test/test_math_chain_decomposition.py::'):
        successors = item.callspec.params['successors']
        seed = SEEDS[successors]
        ACTIVE = dict(nodeid=item.nodeid, successors=successors, seed=seed,
                      budget=512 if successors == 1 else 768)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def constructor(model, *args, **kwargs):
    result = CREATE(model, *args, **kwargs)
    if ACTIVE is not None:
        folder = Path(os.environ['DECOMPOSITION_SNAPSHOT_DIR'])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"successors-{ACTIVE['successors']}.pt"
        with path.open('xb') as stream:
            torch.save({name: value.detach().cpu().clone()
                        for name, value in model.named_parameters()}, stream)
        metadata = dict(ACTIVE, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        parameters={name: list(value.shape) for name, value in model.named_parameters()})
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return result


BasicModel._create_per_stage = constructor
