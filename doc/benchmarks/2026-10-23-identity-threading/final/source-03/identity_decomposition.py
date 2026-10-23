# Measurement protocol inherited from the review-three bind gate.
# Restores its original snapshots/seeds with the recorded retired-predictor
# parameter-name mapping; writes metrics only in this source receipt.
"""Restore surviving parameters by grammar identity at the original measurement seeds."""
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

receipt = Path(os.environ['IDENTITY_GATE_DIR'])
root = receipt.parents[4]
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
    directory = root/'output/item3-5-identity-initial'/token
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{index}.pt'
    # Both arms start from the original, immutable pre-training snapshot.
    # Retiring the extra predictor shifts the ModuleList indices. This is a
    # measurement ledger, not a production checkpoint adapter.
    import re
    baseline = torch.load(path, weights_only=True)
    mapped, retired, names = {}, [], {}
    for name, value in baseline.items():
        match = re.match(r'conceptualSpaces\.(\d+)\.layers\.(\d+)\.(.*)', name)
        target = name
        if match:
            stage, layer, suffix = match.groups()
            if int(layer) == 0:
                retired.append(name)
                continue
            target = f'conceptualSpaces.{stage}.layers.{int(layer)-1}.{suffix}'
        assert target not in mapped
        mapped[target], names[target] = value, name
    assert params.keys() == mapped.keys(), dict(missing=sorted(params.keys()-mapped.keys()),
                                               retired=sorted(mapped.keys()-params.keys()))
    parameters = dict(model.named_parameters())
    # The compose bind face adds one action and two grammatical roles. Keep
    # every original action/category row by its name; only the new rows retain
    # their initialization from this original-seed construction.
    from Language import Grammar, TheGrammar
    old = Grammar()
    old.load_from_grammar_file(str(receipt/'original-complete.grammar'))
    categories = list(TheGrammar.categories)
    rule_keys = list(TheGrammar.rule_table.values())
    assert len(set(rule_keys)) == len(rule_keys)
    rows = {
        'symbolSpace.subspace.category_embedding.weight': [categories.index(name) for name in old.categories],
        'symbolSpace.subspace.rule_predictor.2.weight': [rule_keys.index(name) for name in old.rule_table.values()],
        'symbolSpace.subspace.rule_predictor.2.bias': [rule_keys.index(name) for name in old.rule_table.values()],
    }
    layer = model.symbolSpace.subspace.layers[0].operation_layer
    insertion = 1 + layer.r_reduce + layer.r_apply - 1
    prefix = 'symbolSpace.subspace.layers.0.operation_layer.'
    rows[prefix+'apply_anchor'] = list(range(layer.r_apply-1))
    for suffix in ('chooser.tool_embedding', 'chooser.what_projection.weight'):
        rows[prefix+suffix] = list(range(insertion)) + list(range(insertion+1, insertion+7))
    migrated = {}
    with torch.no_grad():
        for name, value in mapped.items():
            destination = parameters[name]
            if destination.shape == value.shape:
                destination.copy_(value.to(destination))
            else:
                indices = rows[name]
                assert len(indices) == len(value) and destination.shape[1:] == value.shape[1:], name
                destination.index_copy_(0, torch.tensor(indices, device=destination.device), value.to(destination))
                migrated[name] = dict(old_shape=list(value.shape), new_shape=list(destination.shape), rows=indices)
    assert all(torch.equal((value if name not in migrated else value[migrated[name]['rows']]).detach().cpu(),
                           mapped[name]) for name,value in parameters.items())
    comparison = dict(original_snapshot=str(path.relative_to(root)),
        snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_parameters=len(baseline), retired_predictor_parameters=retired,
        name_mapping=names, catalogue_rows=migrated, equal_to_original_surviving_parameters=True)
    torch.save(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                    parameters={name: value.detach().cpu() for name,value in parameters.items()}),
               receipt/f'decomposition-{arm}-{token}-{index}-initial.pt')
    params = {name:value.detach().cpu().clone() for name,value in parameters.items()}
    previous = root/'output/item3-5-bind-initial'/token/f'{index}.pt'
    if previous.exists():
        baseline = torch.load(previous, weights_only=True)
        changed = sorted(set(params) ^ set(baseline))
        changed += [name for name in params.keys() & baseline.keys()
                    if not torch.equal(params[name], baseline[name])]
        comparison['changed_from_step1'] = sorted(changed)
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
