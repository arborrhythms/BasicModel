"""Capture unseeded standing entry states; diagnostic replay is separate."""
from contextlib import contextmanager
from pathlib import Path
import random
import numpy as np
import torch

def entry(folder):
    torch.save(dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()), Path(folder)/'unseeded-entry.pt')

@contextmanager
def switches():
    yield
