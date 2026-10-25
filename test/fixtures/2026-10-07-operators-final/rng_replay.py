"""Record an unseeded entry state; restore only for separately labelled bisections."""
from contextlib import contextmanager
from pathlib import Path
import os
import random
import numpy as np
import torch


def entry(folder):
    replay = os.environ.get('OPERATORS_REPLAY_RNG')
    if replay:
        state = torch.load(replay, weights_only=False, map_location='cpu')
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    torch.save(state, Path(folder)/'unseeded-entry.pt')


@contextmanager
def switches():
    part = os.environ.get('OPERATORS_DISABLE')
    if not part:
        yield
        return
    from unittest.mock import patch
    from util import XMLConfig
    key, value = {'B': ('meaningWidth', 0), 'C': ('symbolCentroid', False),
                  'D': ('membershipPriming', False)}[part]
    original = XMLConfig.__init__
    def initialize(config, *args, **kwargs):
        original(config, *args, **kwargs)
        config.data.setdefault('ConceptualSpace', {})[key] = value
    with patch.object(XMLConfig, '__init__', initialize):
        yield
