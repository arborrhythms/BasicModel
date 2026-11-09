"""Declared operation identity survives a catalogue edit and its aliases."""
import torch
import pytest

from GrammarCheckpoint import align, align_optimizer, catalog
from Language import OperationSelectionLayer


def host(names):
    model = torch.nn.Module()
    model.operation_layer = OperationSelectionLayer(d_model=4,
        ops=[torch.nn.Linear(4, 4) for _ in names], op_names=list(names),
        unary_names=[], chooser='mlp')
    model.alias = model.operation_layer
    return model


def test_operation_parameters_and_optimizer_follow_identity_across_removed_position():
    previous = host(('first', 'removed', 'last'))
    current = host(('first', 'last'))
    with torch.no_grad():
        for index, operator in enumerate(previous.operation_layer.ops):
            operator.weight.fill_(index+10.)
            operator.bias.fill_(index+20.)
    saved = previous.state_dict()
    migrated, drift, transforms = align(current, saved, catalog(previous))
    for alias in ('operation_layer', 'alias'):
        for suffix in ('weight', 'bias'):
            assert torch.equal(migrated[f'{alias}.ops.1.{suffix}'],
                               saved[f'{alias}.ops.2.{suffix}'])
            assert f'{alias}.ops.2.{suffix}' not in migrated
            assert transforms[f'{alias}.ops.1.{suffix}'] is None
            assert transforms[f'{alias}.ops.2.{suffix}'][0] == f'{alias}.ops.1.{suffix}'
    source = 'operation_layer.ops.2.weight'
    manifest = dict(leaves=[dict(param_groups=[[dict(name=source,shape=[4,4])]])])
    optimizer = dict(state={0:dict(step=torch.tensor(1.),exp_avg=torch.full((4,4),.25))},
                     param_groups=[dict(params=[0])])
    restored, metadata = align_optimizer(optimizer, manifest, transforms, migrated)
    assert metadata['leaves'][0]['param_groups'][0][0]['name'] == 'operation_layer.ops.1.weight'
    assert torch.equal(restored['state'][0]['exp_avg'], optimizer['state'][0]['exp_avg'])
    current.load_state_dict(migrated, strict=True)
    unchanged, drift, transforms = align(current, current.state_dict(), catalog(current))
    assert not drift and not transforms
    assert unchanged.keys() == current.state_dict().keys()


def test_count_checkpoint_and_changed_argument_catalogue_require_explicit_replay():
    from PartsOfSpeech import PartsOfSpeech
    from IndependentComponents import _pack_state, _unpack_state
    previous, current = host(('first',)), host(('first',))
    for model, arguments in ((previous, ('first_argin1', 'removed_argin1')),
                             (current, ('first_argin1', 'new_argin1'))):
        model.language = torch.nn.Module()
        model.language.parts_of_speech = PartsOfSpeech(arguments)
    with pytest.raises(ValueError, match='catalogue'):
        align(current, previous.state_dict(), catalog(previous))
    key = 'language.parts_of_speech._extra_state'
    saved = current.state_dict()
    saved[key] = _pack_state(dict(version=3, arguments=current.language.parts_of_speech.arguments,
        counts={17: {('first_argin1', -1): 2}}, uses={'first':2}))
    with pytest.raises(ValueError, match='replay its producer'):
        align(current, saved, catalog(current))
