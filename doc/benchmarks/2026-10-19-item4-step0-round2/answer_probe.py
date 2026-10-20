"""Fixed-initialization diagnostic; no certificate assertion is replaced."""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
import gate_runtime
from Models import BasicModel
from ThoughtReferences import bindings, evidence_pair, open_slots
from reading_fixtures import use_eager_reading


def encode(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    return value


def meaning(value):
    return None if value is None else dict(refs=value.role_refs,
        open=open_slots(value), evidence=evidence_pair(value), bindings=bindings(value))


gate_runtime.begin('test/test_math_chain_decomposition.py::test_forced_native_decomposition_through_training_driver[1]')
folder = Path(sys.argv[1])
diagnostic = folder.with_suffix('.jsonl')
root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('round2_decomposition',
    root/'doc/benchmarks/2026-10-08-math-chain-repair-2/decomposition_certificate.py')
certificate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(certificate)


def record(model, kind, **values):
    with diagnostic.open('a') as stream:
        stream.write(json.dumps(encode(dict(kind=kind,
            trial=getattr(model, '_sentence_trial', None), **values)), default=str)+'\n')


reader = BasicModel._sentence_reader_error
def read(model, state, sid, active, observation, **kwargs):
    result = reader(model, state, sid, active, observation, **kwargs)
    registry = kwargs.get('registry')
    record(model, 'reader', active=active, result=result,
        meanings=[meaning(item) for item in observation['meanings']],
        terms={} if registry is None else {name: registry._value(item)
            for name, item in registry._terms.items()})
    return result


path_cost = BasicModel._sentence_path_cost
def score(model, *args, **kwargs):
    result = path_cost(model, *args, **kwargs)
    record(model, 'sentence', answer=model._sentence_answer_cost,
        reference=result[2].get('reference_credit'),
        clauses=[None if item is None else dict(refs=item.refs,
            meaning=meaning(item.meaning)) for item in result[2]['clauses']],
        terms={name: model._sentence_cost_registry._value(item)
            for name, item in model._sentence_cost_registry._terms.items()})
    return result


once = BasicModel._run_selected_thought_once
def thought(model, *args, **kwargs):
    result = once(model, *args, **kwargs)
    record(model, 'thought', row=kwargs['row'], before=meaning(args[0]),
        after=meaning(result.meaning), work=result.work.spent,
        operations=[r.operation for r in result.records if r.kind == 'thought'])
    return result


with pytest.MonkeyPatch.context() as monkeypatch:
    use_eager_reading(monkeypatch)
    monkeypatch.setattr(BasicModel, '_sentence_reader_error', read)
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', score)
    monkeypatch.setattr(BasicModel, '_run_selected_thought_once', thought)
    certificate.run(folder, 1)
