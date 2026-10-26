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

receipt = Path(os.environ['IDENTITY_GATE_DIR']).resolve()
output_root = Path(os.environ['LAYOUT_GATE_DIR']).resolve().parent
output_root.mkdir(parents=True,exist_ok=True)
root = next(parent for parent in receipt.parents if (parent/'bin/Models.py').is_file())
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
    # The bind and absent-marker compound faces add actions and roles. Keep
    # every original action/category row by its name; only the new rows retain
    # their initialization from this original-seed construction.
    from Language import Grammar, TheGrammar, GRAMMAR_LAYER_CLASSES
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
    def original_names(arity):
        # Independent grammar declarations authenticate the numerical row map.
        declared = sorted((rule.space_role, index, rule.method_name)
            for index, rule in enumerate(old.rules_upward)
            if rule.arity == arity and rule.method_name and not rule.mandatory
            and not getattr(GRAMMAR_LAYER_CLASSES.get(rule.method_name), 'field_only', False))
        return [name for _, _, name in declared]
    binary_names, unary_names = original_names(2), original_names(1)
    assert [name for name in layer.op_names if name != 'bind'] == binary_names
    # A grammar may declare several roles of the same numerical operation.
    # Preserve each occurrence in declaration order; list.index aliases them.
    binary_rows = [i for i,name in enumerate(layer.op_names) if name != 'bind']
    assert len(set(binary_rows)) == len(binary_names)
    assert len(set(layer.unary_names)) == len(layer.unary_names)
    assert set(layer.unary_names) - set(unary_names) == {'compound_unary'}
    assert [name for name in layer.unary_names if name not in ('bind', 'compound_unary')] == unary_names
    unary_rows = [layer.unary_names.index(name) for name in unary_names]
    assert [layer.unary_names[index] for index in unary_rows] == unary_names
    prefix = 'symbolSpace.subspace.layers.0.operation_layer.'
    rows[prefix+'apply_anchor'] = unary_rows
    rows[prefix+'reduce_anchor'] = binary_rows
    # STOP, binary actions, named unary actions, then the six field actions.
    first_unary = 1 + layer.r_reduce
    chooser_rows = ([0] + [1+index for index in binary_rows] + [first_unary+index for index in unary_rows]
                    + list(range(first_unary+layer.r_apply, first_unary+layer.r_apply+6)))
    for suffix in ('chooser.tool_embedding', 'chooser.what_projection.weight'):
        rows[prefix+suffix] = chooser_rows
    # The absent-marker face also adds one generate choice. Authenticate
    # the old names against the archived grammar, then preserve every row.
    generation = model.languageSpace
    def generated_names(grammar):
        return [rule.method_name for arity in (2, 1)
                for _, _, rule in sorted((rule.space_role, i, rule)
                    for i, rule in enumerate(grammar.rules_downward)
                    if len(str(rule.lhs).split(',')) == arity and rule.method_name
                    and not rule.mandatory and not getattr(
                        GRAMMAR_LAYER_CLASSES.get(rule.method_name), 'preserves_pair', False))]
    old_generated = generated_names(old)
    new_generated = list(generation._generate_binary_names + generation._generate_unary_names)
    assert [name for name in new_generated if name != 'compound_unary'] == old_generated
    generation_rows = [i for i, name in enumerate(new_generated) if name != 'compound_unary'] + [len(new_generated)]
    for suffix in ('weight', 'bias'):
        rows['symbolSpace.languageSpace.generate_policy.'+suffix] = generation_rows
    migrated = {}
    with torch.no_grad():
        for name, value in mapped.items():
            destination = parameters[name]
            if destination.shape == value.shape:
                destination.copy_(value.to(destination))
            elif name.endswith('_inter_predictor.context.0.weight'):
                assert destination.shape == (value.shape[0], value.shape[1]+4)
                destination[:, :value.shape[1]].copy_(value.to(destination))
                migrated[name] = dict(old_shape=list(value.shape), new_shape=list(destination.shape),
                                      columns=list(range(value.shape[1])))
            else:
                indices = rows[name]
                assert len(indices) == len(value) and destination.shape[1:] == value.shape[1:], name
                destination.index_copy_(0, torch.tensor(indices, device=destination.device), value.to(destination))
                migrated[name] = dict(old_shape=list(value.shape), new_shape=list(destination.shape), rows=indices)
    unequal = [name for name,value in parameters.items() if not torch.equal(
        (value if name not in migrated else value[:,migrated[name]['columns']]
         if 'columns' in migrated[name] else value[migrated[name]['rows']]).detach().cpu(),mapped[name])]
    assert not unequal, unequal
    comparison = dict(original_snapshot=str(path.relative_to(root)),
        snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_parameters=len(baseline), retired_predictor_parameters=retired,
        name_mapping=names, catalogue_rows=migrated,
        action_names=dict(binary=binary_names, original_unary=unary_names,
                          current_unary=list(layer.unary_names), unary_rows=unary_rows,
                          current_binary=list(layer.op_names),binary_rows=binary_rows),
        equal_to_original_surviving_parameters=True)
    torch.save(dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                    parameters={name: value.detach().cpu() for name,value in parameters.items()}),
               output_root/f'decomposition-{arm}-{token}-{index}-initial.pt')
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
    directory = output_root/'bind-decomposition-metrics'/arm
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f'process-{os.getpid()}.json').write_text(json.dumps(state, indent=2)+'\n')


def pytest_sessionfinish(session, exitstatus):
    flush()


util.XMLConfig._parse_xml = staticmethod(configuration)
BasicModel._create_per_stage = constructor
