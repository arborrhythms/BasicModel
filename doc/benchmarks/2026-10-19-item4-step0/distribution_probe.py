"""Run the unchanged certificate and separate chooser logits from prime bias."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
import util
from Language import OperationSelectionLayer
from reading_fixtures import use_eager_reading
from test_math_chain_section14 import test_ordinary_initial_binding_distribution_includes_every_retained_candidate

util.TheCompileBackend = 'none'
torch.set_num_threads(1)
folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
original = OperationSelectionLayer.select_logits
records = []


def select(module, logits, **kwargs):
    base = kwargs.get('alternative_logits')
    if base is not None:
        records.append(dict(logits=logits.detach().cpu().tolist(),
            base=base.detach().cpu().tolist(),
            prime=None if getattr(module, '_expectation_prime', None) is None
            else module._expectation_prime.detach().cpu().tolist()))
    return original(module, logits, **kwargs)


try:
    with pytest.MonkeyPatch.context() as monkeypatch, patch.object(OperationSelectionLayer, 'select_logits', select):
        use_eager_reading(monkeypatch)
        test_ordinary_initial_binding_distribution_includes_every_retained_candidate(folder, None)
finally:
    (folder/'score-contributions.json').write_text(json.dumps(records)+'\n')
