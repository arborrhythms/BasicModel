"""Pair the changed bind catalogue across gains at the original measurement seeds."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import util
from Models import BasicModel

receipt = Path(__file__).resolve().parent
root = receipt.parents[2]
arm = os.environ['BIND_GAIN_ARM']
state = {}
active = None
parse = util.XMLConfig._parse_xml
create = BasicModel._create_per_stage


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    global active
    active = item.nodeid
    nonce = 'd15d7bc18ad8170f2bf77e628184fd4a'
    seed = int.from_bytes(hashlib.sha256((nonce+active).encode()).digest()[:4], 'big')
    state[active] = dict(seed=seed, constructors=[])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configuration(path):
    data = parse(path)
    training = data.setdefault('architecture', {}).setdefault('training', {})
    values = dict(sentenceExpectation=True, expectationPrime=True,
        explorationUniformFraction=.5, **{name: 0. if arm == 'zero' else .5 for name in (
            'perceptExpectationGain', 'symbolExpectationGain',
            'rowExpectationGain', 'documentExpectationGain')})
    if active is not None:
        values['seed'] = state[active]['seed']
        state[active]['configuration'] = values
    training.update(values)
    return data


def constructor(model, *args, **kwargs):
    result = create(model, *args, **kwargs)
    params = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
    index = len(state[active]['constructors'])
    token = hashlib.sha256(active.encode()).hexdigest()[:20]
    directory = root/'output/item3-5-bind-initial'/token
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{index}.pt'
    if arm == 'zero':
        if path.exists():
            baseline = torch.load(path, weights_only=True)
            assert params.keys() == baseline.keys() and all(
                torch.equal(params[name], baseline[name]) for name in params)
        else:
            torch.save(params, path)
        comparison = dict(saved_for_enabled_arm=True)
    else:
        baseline = torch.load(path, weights_only=True)
        unequal = sorted(set(params) ^ set(baseline))
        unequal += [name for name in params.keys() & baseline.keys()
                    if not torch.equal(params[name], baseline[name])]
        comparison = dict(equal_to_zero=not unequal, unequal=unequal)
        assert not unequal, comparison
    previous = root/'output/item4-step0-initial'/token/f'test-{index}.pt'
    if previous.exists():
        baseline = torch.load(previous, weights_only=True)
        changed = sorted(set(params) ^ set(baseline))
        changed += [name for name in params.keys() & baseline.keys()
                    if not torch.equal(params[name], baseline[name])]
        comparison['changed_from_item4'] = sorted(changed)
    state[active]['constructors'].append(dict(parameters=len(params), **comparison))
    flush()
    return result


def flush():
    directory = receipt/'bind-decomposition-metrics'/arm
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f'process-{os.getpid()}.json').write_text(json.dumps(state, indent=2)+'\n')


def pytest_sessionfinish(session, exitstatus):
    flush()


util.XMLConfig._parse_xml = staticmethod(configuration)
BasicModel._create_per_stage = constructor
