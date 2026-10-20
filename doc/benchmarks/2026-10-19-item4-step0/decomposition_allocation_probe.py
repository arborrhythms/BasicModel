"""Locate large native question allocations at the recorded initialization."""
from pathlib import Path
import importlib.util
import sys
import traceback
import resource
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
import gate_runtime
from Models import BasicModel
from reading_fixtures import use_eager_reading

gate_runtime.begin('test/test_math_chain_decomposition.py::test_forced_native_decomposition_through_training_driver[1]')
root = Path(__file__).resolve().parents[3]
path = root / 'doc/benchmarks/2026-10-08-math-chain-repair-2/decomposition_certificate.py'
spec = importlib.util.spec_from_file_location('decomposition_allocation_diagnostic', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Allocations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls, self.next_peak = 0, 2 * 1024**3

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        self.calls += 1
        if self.calls % 1000 == 0:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if peak >= self.next_peak:
                print('PEAK', peak, 'CALLS', self.calls,
                      ''.join(traceback.format_stack(limit=15)), flush=True)
                self.next_peak = peak + 512 * 1024**2
        for value in tree_leaves(result):
            if not torch.is_tensor(value) or value.layout != torch.strided:
                continue
            size = value.untyped_storage().nbytes()
            if size > 32 * 1024**2:
                print('ALLOCATION', str(function), list(value.shape), size,
                      ''.join(traceback.format_stack(limit=12)), flush=True)
            if size > 128 * 1024**2:
                raise RuntimeError('diagnostic stopped after first large allocation')
        return result

batch = BasicModel.runBatch
def running(model, *args, **kwargs):
    print('BATCH', kwargs.get('source_rows'), 'budget', model.attention_budget, flush=True)
    if model.attention_budget:
        with Allocations():
            return batch(model, *args, **kwargs)
    return batch(model, *args, **kwargs)

with pytest.MonkeyPatch.context() as patch:
    use_eager_reading(patch)
    patch.setattr(BasicModel, 'runBatch', running)
    module.run(Path(sys.argv[1]), 1)
