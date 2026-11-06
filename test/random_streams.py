"""Record random streams without selecting or advancing an initialization."""
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def capture():
    state = np.random.get_state()
    result = dict(torch=torch.get_rng_state().clone(),
                  numpy=(state[0], state[1].copy(), *state[2:]),
                  python=random.getstate())
    device = os.environ.get('BASICMODEL_DEVICE', 'cpu')
    if device.startswith('cuda') or torch.cuda.is_initialized():
        result['cuda'] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    if device == 'mps':
        result['mps'] = torch.mps.get_rng_state().clone()
    return result


def write(path, *, scope, state=None):
    """Persist first, then publish its hash; no seed or state is installed."""
    hash_seed = os.environ.get('PYTHONHASHSEED')
    if hash_seed is None or not hash_seed.isdecimal():
        raise RuntimeError('The launcher must record PYTHONHASHSEED before starting this process')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Refusing to replace recorded initial random streams: ' + str(path))
    state = capture() if state is None else state
    with path.open('xb') as target:
        torch.save(state, target)
    record = dict(torch_version=version('torch'), executorch_version=version('executorch'),
                  pytest_version=version('pytest'), pythonhashseed=hash_seed,
                  path=str(path.resolve()), sha256=sha256(path.read_bytes()).hexdigest(),
                  bytes=path.stat().st_size, scope=scope,
                  streams=['torch.cpu', 'numpy', 'python',
                           *(['torch.cuda'] if 'cuda' in state else []),
                           *(['torch.mps'] if 'mps' in state else [])],
                  seed_selected=False)
    path.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    return record
