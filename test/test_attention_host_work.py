"""Fixed-input contracts for exact host-work reductions after ownership.

The reference functions are loaded from the original, unchanged Git source;
these tests do not replace existing fixtures or select an RNG seed.
"""
import ast
from collections.abc import Mapping
import math
import subprocess
import json
from types import SimpleNamespace

import pytest
import torch


def original_function(file, name, namespace):
    source = subprocess.check_output(
        ['git', 'show', f'3926005f12:bin/{file}.py'], text=True)
    tree = ast.parse(source)
    definition = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module([definition], type_ignores=[]),
                 f'3926005f12/{file}.py', 'exec'), namespace)
    return namespace[name]


@pytest.fixture(scope='module')
def original_freeze():
    return original_function('Meaning', '_freeze_metadata',
                             dict(Mapping=Mapping, math=math))


def assert_same_metadata(actual, expected):
    assert type(actual) is type(expected)
    assert actual == expected
    if isinstance(actual, tuple):
        for a, b in zip(actual, expected):
            assert_same_metadata(a, b)


@pytest.mark.parametrize('value', [
    None, True, False, 0, -7, 'reference', 1.5, (), [],
    (('sym', 17), None, ('ltm', ('document', 3), 8)),
    {'scope': [('reference', 3), None], 'bindings': {'z': False, 'a': 1}},
    tuple([('scope', (('sym', 4), ('sym', 4))), ('polarity', False)]),
])
def test_metadata_freeze_preserves_values_and_types(original_freeze, value):
    from Meaning import _freeze_metadata
    assert_same_metadata(_freeze_metadata(value), original_freeze(value))


@pytest.mark.parametrize('value', [
    float('nan'), float('inf'), -float('inf'), {3: 'bad key'},
    ('ok', [torch.tensor(1)]), {'scope': (object(),)},
])
def test_metadata_freeze_retains_rejections(original_freeze, value):
    from Meaning import _freeze_metadata
    with pytest.raises(TypeError) as before:
        original_freeze(value)
    with pytest.raises(TypeError) as after:
        _freeze_metadata(value)
    assert str(after.value) == str(before.value)


def test_scalar_subclass_keeps_mapping_precedence(original_freeze):
    from Meaning import _freeze_metadata

    class ScalarMapping(str, Mapping):
        def __iter__(self):
            return iter(('reference',))

        def __getitem__(self, key):
            return ['sym', 9]

        def __len__(self):
            return 1

    value = ScalarMapping('not a scalar')
    assert_same_metadata(_freeze_metadata(value), original_freeze(value))


def test_mutable_metadata_is_still_owned(original_freeze):
    from Meaning import _freeze_metadata
    value = {'bindings': [('sym', 3), ['ltm', 7]]}
    expected = original_freeze(value)
    actual = _freeze_metadata(value)
    value['bindings'][1][1] = 99
    value['bindings'].append(None)
    assert_same_metadata(actual, expected)


@pytest.fixture(scope='module')
def original_features():
    from ThoughtFeatures import METADATA_BYTES
    return original_function('ThoughtFeatures', 'semantic_metadata',
                             dict(json=json, torch=torch, METADATA_BYTES=METADATA_BYTES))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('binding,scope', [
    ((), ()),
    ((('x', ('sym', 9)), ('y', ('sym', 9))), (('location', 'café 🐈'),)),
    ((('_open_references', ('sym', 9)), ('kept', ('ltm', 23))), (True, 1, 1.5, None)),
    ((('long', 'x' * 80),), tuple(('reference', i) for i in range(20))),
    ((('nested', ((((((((((((((((((('sym', 9),),),),),),),),),),),),),),),),),),))), ()),
])
def test_metadata_features_match_original_bytes_and_aliases(original_features, dtype, binding, scope):
    from ThoughtFeatures import semantic_metadata
    roles = torch.arange(24, dtype=dtype).reshape(3, 8).requires_grad_()
    meanings = tuple(SimpleNamespace(roles=roles, role_refs=refs, polarity=polarity,
                                    bindings=binding, scope=scope)
                     for refs, polarity in [((('sym', 9), None, ('ltm', 23)), True),
                                            ((('sym', 9), ('sym', 11), None), False)])
    before = original_features(meanings)
    after = semantic_metadata(meanings)
    assert len(before) == len(after)
    for a, b in zip(after, before):
        assert torch.equal(a, b)
        assert a.dtype == b.dtype and a.device == b.device
        assert a.requires_grad == b.requires_grad
