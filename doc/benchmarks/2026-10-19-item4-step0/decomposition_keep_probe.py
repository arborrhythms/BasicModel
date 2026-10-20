"""Observe the remaining one-successor failure at its recorded initialization."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
import pytest
import gate_runtime
from Models import BasicModel
from reading_fixtures import use_eager_reading
from ThoughtReferences import open_slots

gate_runtime.begin('test/test_math_chain_decomposition.py::test_forced_native_decomposition_through_training_driver[1]')
root = Path(__file__).resolve().parents[3]
path = root / 'doc/benchmarks/2026-10-08-math-chain-repair-2/decomposition_certificate.py'
spec = importlib.util.spec_from_file_location('decomposition_keep_diagnostic', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
folder = Path(sys.argv[1])
observed = []

def encode(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    return value

cost = BasicModel._sentence_path_cost
def reading(model, *args, **kwargs):
    result = cost(model, *args, **kwargs)
    observed.append(dict(trial=model._sentence_trial,
        meanings=[None if meaning is None else dict(refs=meaning.role_refs, open=open_slots(meaning))
                  for meaning in result[2]['meanings']]))
    return result

batch = BasicModel.runBatch
def running(model, *args, **kwargs):
    result = batch(model, *args, **kwargs)
    field = model._last_sentence_field
    trials = []
    if field is not None:
        for trial in field['trials']:
            trials.append(dict(positions=trial['positions'], cost=trial['cost'],
                reads=[dict(action=read['action'], admitted=read['admitted']) for read in trial['reads']]))
    item = dict(source_rows=kwargs.get('source_rows'), readings=list(observed), field=trials,
        credit={key: value for key, value in model._last_sentence_credit.items()
                if key not in ('actions', 'narrowing_actions', 'readers')})
    with (folder / 'keep.jsonl').open('a') as stream:
        stream.write(json.dumps(encode(item)) + '\n')
    observed.clear()
    return result

with pytest.MonkeyPatch.context() as patch:
    use_eager_reading(patch)
    patch.setattr(BasicModel, '_sentence_path_cost', reading)
    patch.setattr(BasicModel, 'runBatch', running)
    module.run(folder, 1)
