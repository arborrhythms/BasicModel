"""Exercise the observer's real finalizer with training-only native results.

ModelFactory's native run has no evaluation unless BASIC_RUN_TEST is set.
This uses a factory stand-in, no model construction or training, and executes
the actual observer finalizer to test that it obtains an endpoint itself.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
source = (ROOT/'test/objective_conflicts_probe.py').read_text()
module = ast.parse(source)
block = next(n for n in module.body if isinstance(n, ast.Try)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                     and c.func.attr == '_run_hydrated' for c in ast.walk(n)))
observations = {}
P = SimpleNamespace(training_batches=3, evaluation_batches=0, last_trial={},
                    last_batch={'train': True}, eval_reconstructions=[],
                    start=time.monotonic(), groups=lambda model: None)
class Model:
    inputSpace = SimpleNamespace(data=SimpleNamespace(
        reconstructed_output=[], test_output=[], test_input=[]))
    _bytes_to_text = staticmethod(bytes.decode)
    def __init__(self):
        self.evaluations = []
    def set_sigma(self, value):
        pass
    def runEpoch(self, **kwargs):
        self.evaluations.append(kwargs)
        assert 'optimizer' not in kwargs
        assert kwargs == {'batchSize': 28, 'split': 'test'}
        P.evaluation_batches += 1
        P.last_batch = {'train': False}

model = Model()
class NoGrad:
    def __enter__(self): pass
    def __exit__(self, *args): pass
scope = dict(Models=SimpleNamespace(ModelFactory=SimpleNamespace(
    _run_hydrated=lambda *args: [('native', 0, model)])), P=P,
    args=SimpleNamespace(config='BasicModel_answers_tied_benchmark'),
    config='native.xml',arch={'training': {'batchSize': 28}},
    write=lambda name,value: observations.update({name: value}),
    serial=lambda x:x, SOURCE={}, source_snapshot=lambda root:{}, ROOT=ROOT,
    time=time, torch=SimpleNamespace(no_grad=NoGrad))
exec(compile(ast.Module(body=[block], type_ignores=[]), '<observer finalizer>', 'exec'), scope)
record=dict(outcome=observations.get('outcome.json'),evaluations=model.evaluations)
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
assert record['outcome']['evaluation_batches'] > 0, 'Native endpoint was never evaluated'
assert len(model.evaluations) == 1, 'Exactly one endpoint pass; no extra training'
