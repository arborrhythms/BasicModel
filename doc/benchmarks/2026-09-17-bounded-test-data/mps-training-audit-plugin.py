"""Measurement-only plugin: prove the selected training uses accelerator parameters."""
from collections import Counter
from pathlib import Path
import json
import os
import time

import pytest

_patch = None


def _record(**value):
    path = Path(os.environ['BASICMODEL_GPU_AUDIT'])
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(value)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(records, indent=2) + '\n')
    temporary.replace(path)


def pytest_runtest_setup(item):
    global _patch
    import Models
    import torch
    import util
    expected = os.environ['BASICMODEL_DEVICE']
    assert expected in ('mps', 'cuda') or expected.startswith('cuda:')
    _patch = pytest.MonkeyPatch()
    original_build = Models.BaseModel.from_config
    original_epoch = Models.BasicModel.runEpoch

    def build(*args, **kwargs):
        start = time.perf_counter()
        result = original_build(*args, **kwargs)
        model = result[0]
        counts = dict(Counter(str(p.device) for p in model.parameters()))
        predictor = model.symbolSpace.discourse._inter_predictor
        devices = sorted({str(p.device) for p in predictor.parameters()})
        _record(event='constructed', seconds=time.perf_counter()-start,
                parameter_devices=counts, predictor_devices=devices,
                compile_backend=util.TheCompileBackend, torch_version=torch.__version__)
        requested = torch.device(expected)
        assert devices and all(
            torch.device(value).type == requested.type
            and (requested.index is None or torch.device(value).index == requested.index)
            for value in devices), 'significant predictor training must use the requested GPU'
        return result

    def epoch(model, *args, **kwargs):
        synchronize = torch.mps.synchronize if expected == 'mps' else torch.cuda.synchronize
        synchronize()
        start = time.perf_counter()
        result = original_epoch(model, *args, **kwargs)
        synchronize()
        _record(event='epoch', seconds=time.perf_counter()-start,
                device=expected, max_batches=kwargs.get('max_batches'),
                batch_size=kwargs.get('batchSize'))
        return result

    _patch.setattr(Models.BaseModel, 'from_config', staticmethod(build))
    _patch.setattr(Models.BasicModel, 'runEpoch', epoch)


def pytest_runtest_teardown(item, nextitem):
    global _patch
    if _patch is not None:
        _patch.undo()
        _patch = None
