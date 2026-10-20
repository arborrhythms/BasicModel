"""Locate the resource stop without changing the recorded initialization."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
import pytest
import gate_runtime
from Models import BasicModel
from reading_fixtures import use_eager_reading

gate_runtime.begin('test/test_math_chain_decomposition.py::test_forced_native_decomposition_through_training_driver[1]')
root=Path(__file__).resolve().parents[3]
path=root/'doc/benchmarks/2026-10-08-math-chain-repair-2/decomposition_certificate.py'
spec=importlib.util.spec_from_file_location('decomposition_diagnostic',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
snapshot=module.footprint.snapshot
digest=module.footprint.value_digest
def audited(model):
    print('SNAPSHOT entered',flush=True)
    return snapshot(model)
def inspected(value,*args,**kwargs):
    if torch.is_tensor(value):
        size=value.numel()*value.element_size()
        if size>64*1024**2:
            print('AUDIT tensor',list(value.shape),str(value.layout),size,flush=True)
        if value.layout==torch.strided and size>512*1024**2:
            raise RuntimeError('diagnostic stopped before large audit tensor materialization')
    return digest(value,*args,**kwargs)
module.footprint.snapshot=audited
module.footprint.value_digest=inspected
batch=BasicModel.runBatch
def running(model,*args,**kwargs):
    print('BATCH',kwargs.get('source_rows'),'budget',model.attention_budget,flush=True)
    result=batch(model,*args,**kwargs)
    print('BATCH complete',flush=True)
    return result
with pytest.MonkeyPatch.context() as patch:
    use_eager_reading(patch)
    patch.setattr(BasicModel,'runBatch',running)
    module.run(Path(sys.argv[1]),1)
