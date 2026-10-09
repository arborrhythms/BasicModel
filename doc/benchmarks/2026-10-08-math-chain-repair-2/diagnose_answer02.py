"""Authorized development replay of the rejected source; never a measurement."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / '2026-10-08-math-chain-repair'
BASE = HERE / 'diagnostic-baseline'
OUT = HERE / 'development-answer02'
sys.path[:0] = [str(BASE / 'bin'), str(OLD)]
os.chdir(BASE)

spec = importlib.util.spec_from_file_location('frozen_math_train', OLD / 'math_train.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.ROOT = BASE
sys.path[:0] = [str(BASE / 'bin'), str(OLD)]

import ThoughtAnswer
from Models import BasicModel
from ThoughtReferences import bindings, open_slots

original_score = ThoughtAnswer.scorer
original_batch = BasicModel.runBatch
current = {}


def encode(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def scorer(*args, **kwargs):
    score, trials = original_score(*args, **kwargs)

    def observed(row, result):
        try:
            return score(row, result)
        except BaseException as error:
            value = result.meaning
            report = dict(development=True, error=repr(error), row=row,
                batch=current, role_mask=value.role_mask,
                role_refs=value.role_refs, bindings=bindings(value),
                open_slots=open_slots(value),
                operations=[dict(kind=r.kind, operation=r.operation,
                    meaning=None if r.meaning is None else dict(
                        role_mask=r.meaning.role_mask,
                        role_refs=r.meaning.role_refs,
                        bindings=bindings(r.meaning))) for r in result.records])
            (HERE / 'diagnostic-answer02.json').write_text(json.dumps(encode(report), indent=2) + '\n')
            raise

    return observed, trials


def batch(model, *args, **kwargs):
    global current
    split = kwargs.get('split', 'train')
    rows = kwargs.get('source_rows', ())
    current = dict(rows=list(rows), texts=[getattr(model.inputSpace.data, split + '_input')[i] for i in rows])
    start = time.monotonic()
    try:
        return original_batch(model, *args, **kwargs)
    finally:
        with (HERE / 'diagnostic-answer02-batches.jsonl').open('a') as f:
            f.write(json.dumps(encode(dict(current, seconds=time.monotonic() - start))) + '\n')


ThoughtAnswer.scorer = scorer
BasicModel.runBatch = batch
driver.train(OUT, OLD / 'math-trainings/paired-02', 'answer_and_expectation', 2)
