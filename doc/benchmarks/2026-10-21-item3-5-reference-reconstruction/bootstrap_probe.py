"""Observe the unchanged bootstrap-sharing test; never alter its choices or costs."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from arrival_capture import plain
from arrival_keep_probe import clause_graph


@pytest.fixture(autouse=True)
def capture_bootstrap(request, monkeypatch):
    if request.node.name != 'test_both_training_trials_share_bootstrap_snapshot':
        return
    import test_mm_xor
    from Models import BasicModel
    folder = Path(os.environ['BOOTSTRAP_CAPTURE'])
    folder.mkdir(parents=True, exist_ok=False)
    replay = os.environ.get('BOOTSTRAP_REPLAY')
    if replay:
        state = torch.load(Path(replay)/'initial-rng.pt', weights_only=False)
        torch.set_rng_state(state['torch'])
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
    torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                    numpy=np.random.get_state()), folder/'initial-rng.pt')
    def write(name, value):
        with (folder/name).open('a') as output:
            output.write(json.dumps(plain(value))+'\n')
    create = test_mm_xor._fresh_model
    def fresh(*args, **kwargs):
        result = create(*args, **kwargs)
        model = result[0]
        torch.save({name: value.detach().cpu().clone() for name, value in model.named_parameters()},
                   folder/'initial-parameters.pt')
        write('initial.jsonl', dict(config=args, parameters={name: hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.named_parameters()}))
        return result
    monkeypatch.setattr(test_mm_xor, '_fresh_model', fresh)
    cost = BasicModel._sentence_path_cost
    trials = []
    def observe_cost(model, *args, **kwargs):
        result = cost(model, *args, **kwargs)
        observation = result[2]
        record = observation['record']
        trials.append(plain(dict(trial=model._sentence_trial,
            component_reading=getattr(model, '_component_reading', False),
            word_values=record.word_values, word_valid=record.word_valid,
            word_rows=record.word_rows, meaning_start=record.primed.meaning_start,
            primed=vars(record.primed), reconstruction=result[1],
            decoder_trace=getattr(model, '_last_decoder_trace', None),
            registry=model._sentence_cost_registry.breakdown(),
            clauses=[clause_graph(clause) for clause in observation['clauses']],
            programs=[None if program is None else dict(actions=program.actions,
                operation_values=program.operation_values, concept_ids=program.concept_ids,
                reference_ids=program.reference_ids) for program in observation['entries']],
            reference_credit=observation.get('reference_credit'),
            answer_cost=model._sentence_answer_cost)))
        return result
    monkeypatch.setattr(BasicModel, '_sentence_path_cost', observe_cost)
    batch = BasicModel.runBatch
    def observe_batch(model, *args, **kwargs):
        result = batch(model, *args, **kwargs)
        write('batches.jsonl', dict(trials=trials, credit=model._last_sentence_credit,
            field=model._last_sentence_field,
            mtm={str(key): vars(entry) for key, entry in model._concept_owner().mtm.entries.items()},
            reference_audit=model._concept_owner().components.reference_close.last_audit))
        trials.clear()
        return result
    monkeypatch.setattr(BasicModel, 'runBatch', observe_batch)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if item.name == 'test_both_training_trials_share_bootstrap_snapshot' and report.when == 'call':
        (Path(os.environ['BOOTSTRAP_CAPTURE'])/'outcome.json').write_text(json.dumps(
            dict(outcome=report.outcome, failure=str(report.longrepr)), indent=2)+'\n')
