"""Read-only landing diagnostics; never sets an RNG or changes a test."""
import os
from pathlib import Path
import random
import shutil
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    watched = ('test_math_chain_section14.py', 'test_identity_cube.py',
               'test_both_training_trials_share_bootstrap_snapshot',
               'test_bootstrap_native_grammar')
    if not any(name in item.nodeid for name in watched):
        yield
        return
    import numpy as np
    import torch
    target = Path(os.environ['LANDING_DIAGNOSTICS']) / item.name
    target.mkdir(parents=True, exist_ok=True)
    torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                    numpy=np.random.get_state()), target/'initial-rng.pt')
    (target/'nodeid.txt').write_text(item.nodeid+'\n')
    yield
    folder = item.funcargs.get('tmp_path')
    if folder is not None:
        for source in folder.rglob('*'):
            if source.is_file() and source.suffix in ('.json', '.jsonl'):
                dest = target/source.relative_to(folder)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
