"""Record diagnostic initializations without changing their RNG states."""
import os, random
from pathlib import Path
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    import numpy as np
    import torch
    target = Path(os.environ['RNG_CAPTURE']) / item.name
    target.mkdir(parents=True, exist_ok=True)
    torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(), numpy=np.random.get_state()), target/'initial-rng.pt')
    (target/'nodeid.txt').write_text(item.nodeid+'\n')
    yield
