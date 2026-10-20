"""Replay one recorded diagnostic initialization; no default seed."""
import os
import random
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    import numpy as np
    import torch
    saved = torch.load(os.environ['REPLAY_RNG'], weights_only=False)
    torch.set_rng_state(saved['torch'])
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    yield
