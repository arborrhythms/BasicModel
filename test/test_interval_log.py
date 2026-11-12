"""Interval logging owns no learner state and executes probes outside the learner."""
import json
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import torch
import pytest
import IntervalLog


def test_disabled_interval_does_no_collection(monkeypatch):
    from util import TheXMLConfig
    monkeypatch.setattr(TheXMLConfig, 'training', lambda key, default=None: 0)
    monkeypatch.setattr(IntervalLog, 'captured_metrics', lambda *_: (_ for _ in ()).throw(AssertionError('unexpected read')))
    assert IntervalLog.observe_interval(SimpleNamespace(), training=True, split='train') is None


def test_interval_is_read_only_and_records_probe_cost(tmp_path, monkeypatch):
    from util import TheXMLConfig
    output = tmp_path/'intervals.jsonl'
    monkeypatch.setattr(TheXMLConfig, 'training', lambda key, default=None:
        {'intervalLogEveryBatches': 2, 'intervalLogFile': str(output)}.get(key, default))
    model = SimpleNamespace(_train_batches_seen=2, _config_path=tmp_path/'model.xml', weights=torch.randn(3))
    model.save_weights = lambda path: torch.save(model.weights, path)
    monkeypatch.setattr(IntervalLog, 'captured_metrics', lambda model: {'reconstruction_loss': .25})
    def process(command, **options):
        assert options['env']['BASICMODEL_DEVICE'] == 'cpu'
        assert Path(command[command.index('--checkpoint')+1]).is_file()
        Path(command[command.index('--output')+1]).write_text(json.dumps({'status':'complete','probe_counts':{'xor':4,'fineweb':68}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(IntervalLog.subprocess, 'run', process)
    before = (model.weights.clone(), torch.get_rng_state().clone(), random.getstate(), np.random.get_state())
    state = dict(vars(model))
    report = IntervalLog.observe_interval(model, training=True, split='train')
    assert vars(model).keys() == state.keys() and all(vars(model)[key] is value for key, value in state.items())
    assert torch.equal(before[0], model.weights) and torch.equal(before[1], torch.get_rng_state())
    assert before[2] == random.getstate()
    assert np.array_equal(before[3][1], np.random.get_state()[1])
    assert report['batch'] == 2 and report['collection_seconds'] >= 0
    assert report['probes']['probe_counts'] == {'xor':4,'fineweb':68}
    assert json.loads(output.read_text()) == report


def test_failed_probes_are_not_reported_as_measurements(tmp_path, monkeypatch):
    from util import TheXMLConfig
    monkeypatch.setattr(TheXMLConfig, 'training', lambda key, default=None:
        {'intervalLogEveryBatches':1,'intervalLogFile':str(tmp_path/'log')}.get(key,default))
    model=SimpleNamespace(_train_batches_seen=0,_config_path=tmp_path/'model.xml',save_weights=lambda path: Path(path).touch())
    monkeypatch.setattr(IntervalLog,'captured_metrics',lambda _: {})
    monkeypatch.setattr(IntervalLog.subprocess,'run',lambda *a,**k: SimpleNamespace(returncode=1))
    report=IntervalLog.observe_interval(model,training=True,split='train')
    assert report['probes'] == dict(status='failed',exit_code=1,detail='')


def test_each_forgetting_pass_is_observed_without_running_probes(tmp_path, monkeypatch):
    from util import TheXMLConfig
    output = tmp_path/'intervals.jsonl'
    monkeypatch.setattr(TheXMLConfig, 'training', lambda key, default=None:
        {'intervalLogEveryBatches': 10, 'intervalLogFile': str(output)}.get(key, default))
    monkeypatch.setattr(IntervalLog.subprocess, 'run', lambda *a, **k:
        (_ for _ in ()).throw(AssertionError('forgetting must not run probes')))
    store = SimpleNamespace(forget_passes=3)
    monkeypatch.setattr(IntervalLog, 'provisioned_luminosity', lambda model: .25)
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store,
        truth_layer=SimpleNamespace(luminosity=lambda: .25)))
    report = {'before': 12, 'after': 8, 'deleted_by_origin': {0: 0, 1: 3, 2: 1}}
    first = IntervalLog.observe_forgetting(model, report)
    store.forget_passes = 4
    second = IntervalLog.observe_forgetting(model, report)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row['pass_number'] for row in rows] == [3, 4]
    assert all(row['provisioned_truth_luminosity'] == .25 for row in rows)
    assert first['report'] is report and second['report'] is report


def test_luminosity_uses_provisioned_facts_and_independent_poles():
    codes = torch.tensor([[1., 0.], [0., 1.], [100., 100.]])
    store = SimpleNamespace(ORIGIN_PROVISIONED=1,
        rows_of_origin=lambda origin: torch.tensor([0, 1, 2]) if origin == 1 else None,
        kind_of=lambda row: 'fact' if row < 2 else 'word',
        truth_plus=torch.tensor([1., .5, 1.]), truth_minus=torch.tensor([.25, 0., 1.]))
    truth = SimpleNamespace(nDim=2, truths=torch.empty(0, 2),
        _flatten_ltm_row=lambda store, row: codes[row])
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store, truth_layer=truth))
    # Dimension 0: 1 - .25 - .25; dimension 1: .5 - 0 - 0.
    assert IntervalLog.provisioned_luminosity(model) == .5


def test_probe_copy_rejects_changed_checkpoint_tensors():
    model = torch.nn.Linear(2, 1)
    saved = {'state_dict': {name: value.clone() for name, value in model.state_dict().items()}}
    IntervalLog.validate_probe_copy(model, saved)
    with torch.no_grad():
        model.weight[0, 0] += .01
    with pytest.raises(ValueError, match='restore changed tensors: weight'):
        IntervalLog.validate_probe_copy(model, saved)
