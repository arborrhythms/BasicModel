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


def test_added_declaration_preserves_concept_identity_and_checkpoint_rows(tmp_path, monkeypatch):
    import copy
    from pathlib import Path
    from Models import BaseModel
    from test_compiled_word_chunk import _tiny_canonical_model
    from test_reference_close import known_concept
    load = BaseModel.load_weights
    source = Path(__file__).resolve().parents[1]/'data/complete.grammar'
    previous_grammar = tmp_path/'previous.grammar'
    previous_grammar.write_text('\n'.join(line for line in source.read_text().splitlines()
                                         if 'some.' not in line)+'\n')
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1,
        word_buckets='8', input_width=32, concept_rows=32, grammar=previous_grammar)
    restored = None
    try:
        owner = model._concept_owner()
        concept = known_concept(owner, torch.eye(owner.nDim)[0], order=0)
        row = owner._csw_row_of(concept)
        point = owner.similarity_codebook.W[row].detach().clone()
        pool = owner.mtm
        store = model.symbolSpace.ltm_store
        written = store.append_idea(point, trust=.3)
        occurrence = store.occurrence_of(written)
        pool.open(occurrence, kind=None, value=point, when=0)
        arguments = copy.deepcopy(owner._concept_allocator.formal_arguments)
        before = {name: getattr(store, name).clone() for name in store.columns}
        checkpoint = tmp_path/'previous.pt'
        model.save_weights(str(checkpoint))
        model.End(); model.symbolSpace.soft_reset()
        restored = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1,
            word_buckets='8', input_width=32, concept_rows=32)
        assert load(restored, str(checkpoint), require_match=True)
        current = restored._concept_owner()
        assert current._csw_row_of(concept) == row
        torch.testing.assert_close(current.similarity_codebook.W[row], point, rtol=0, atol=0)
        assert all(current._concept_allocator.formal_arguments[key] == value
                   for key, value in arguments.items())
        for name, value in before.items():
            torch.testing.assert_close(getattr(restored.symbolSpace.ltm_store, name), value, rtol=0, atol=0)
        assert current.mtm.token_of(occurrence) is not None
        assert set(current.mtm.transient_ids.tolist()).isdisjoint(
            range(current.nVectors))
        # The migrated identities must also survive a native second load.
        migrated = tmp_path/'current.pt'
        restored.save_weights(str(migrated))
        identities = dict(current._concept_allocator.formal_arguments)
        assert load(restored, str(migrated), require_match=True)
        assert current._concept_allocator.formal_arguments == identities
    finally:
        for instance in (model, restored):
            if instance is not None:
                instance.End(); instance.symbolSpace.soft_reset()
