"""Round-2 ownership, address migration and rejecting-close certificates."""
import copy
from types import SimpleNamespace

import pytest
import torch

from checkpoint_migrations import checkpoint_schema, migrate_wholespace_checkpoint


@pytest.mark.parametrize('configuration', ['model.xml', 'BasicModel.xml', 'MM_xor_fixture.xml'])
def test_one_whole_inventory_on_every_configuration(configuration):
    from test_conceptual_recurrence import _build
    from Spaces import WholeSpace
    model = _build(configuration)
    try:
        assert not hasattr(model, 'wholeSpaces')
        assert [value for value in model.modules() if isinstance(value, WholeSpace)] == [model.wholeSpace]
        assert all(stage['ws'] is model.wholeSpace for stage in model.body_stages)
        assert all(cs.wholeSpace_ref is model.wholeSpace for cs in model.conceptualSpaces)
        assert len(model.conceptualSpaces) == max(1, model.bindingDepth)
        start, end = model.where_registry.slices['wholes']
        assert end - start == model.wholeSpace.subspace.what.nVectors
        assert model.wholeSpace.subspace.what.where_offset == start
    finally:
        model.End()


def migration_fixture(chosen=2):
    member = '_owned_bases.what.primitive_properties.members'
    observed = '_owned_bases.what.primitive_properties.observed'
    live = {'wholeSpace.' + member: torch.zeros(3, 256),
            'wholeSpace.' + observed: torch.zeros(3, 256, dtype=torch.bool),
            'candidate_attention.hidden.0.weight': torch.ones(2, 5)}
    state = {f'wholeSpaces.{i}.{suffix}': value.clone()
             for i in range(3) for suffix, value in
             ((member, live['wholeSpace.' + member]), (observed, live['wholeSpace.' + observed]))}
    state[f'wholeSpaces.{chosen}.{member}'][0, 7] = .625
    state[f'wholeSpaces.{chosen}.{observed}'][0, 7] = True
    state['candidate_attention.hidden.0.weight'] = torch.full((2, 47), 9.)
    saved = dict(checkpoint_schema=checkpoint_schema(), state_dict=state,
        structural_extras={'whole_properties': {
            0: {'attributes': {'_standalone_run_bytes': {7}, '_property_rows_used': {2}},
                'vocab_extras': {'standalone_run_bytes': [7]}},
            2: {'attributes': {'_standalone_run_bytes': {11}},
                'vocab_extras': {'standalone_run_bytes': [11]}}}})
    return saved, live, member, observed


@pytest.mark.parametrize('chosen', [0, 2])
def test_migration_selects_the_changed_copy_and_merges_host_extras(chosen):
    saved, live, member, _ = migration_fixture(chosen)
    original = copy.deepcopy(saved)
    result = migrate_wholespace_checkpoint(saved, live)
    assert result.diagnostics.whole_copy == chosen
    assert result.diagnostics.whole_copies == 3
    assert result.diagnostics.rebuilt_parameters == ('candidate_attention.hidden.0.weight',)
    state = result.checkpoint['state_dict']
    assert not any(key.startswith('wholeSpaces.') for key in state)
    torch.testing.assert_close(state['wholeSpace.' + member], saved['state_dict'][f'wholeSpaces.{chosen}.{member}'])
    torch.testing.assert_close(state['candidate_attention.hidden.0.weight'], live['candidate_attention.hidden.0.weight'])
    entry = result.checkpoint['structural_extras']['whole_properties'][0]
    assert entry['attributes']['_standalone_run_bytes'] == ({7, 11} if chosen == 2 else {7})
    assert entry['attributes']['_property_rows_used'] == {2}
    for key, tensor in saved['state_dict'].items():
        torch.testing.assert_close(tensor, original['state_dict'][key], rtol=0, atol=0)


def test_migration_rejects_two_changed_property_inventories():
    saved, live, member, _ = migration_fixture()
    saved['state_dict']['wholeSpaces.0.' + member][1, 3] = .25
    with pytest.raises(ValueError, match='more than one changed WholeSpace'):
        migrate_wholespace_checkpoint(saved, live)


def test_migration_merges_split_rows_without_overwriting_learned_members():
    saved, live, member, observed = migration_fixture()
    saved['state_dict']['wholeSpaces.0.' + member][2, 8] = .75
    saved['state_dict']['wholeSpaces.0.' + observed][2] = True
    result = migrate_wholespace_checkpoint(saved, live)
    assert result.diagnostics.whole_copy == 2
    assert result.diagnostics.merged_property_rows == (2,)
    assert result.checkpoint['state_dict']['wholeSpace.' + member][2, 8] == .75
    assert result.checkpoint['state_dict']['wholeSpace.' + member][0, 7] == .625
    saved['state_dict']['wholeSpaces.2.' + member][2, 8] = .125
    with pytest.raises(ValueError, match='split property row 2 conflicts'):
        migrate_wholespace_checkpoint(saved, live)


def test_migration_moves_persisted_bands_and_keeps_identity_keys():
    from Spaces import WhereEncoding
    saved, live, _, _ = migration_fixture()
    slices = dict(input=(0, 8), parts=(8, 12), wholes=(12, 15), symbols=(15, 21), conceptual=(21, 29))
    old = WhereEncoding(nWhere=4, nWhen=4).set_capacity(35)
    new = WhereEncoding(nWhere=4, nWhen=4).set_capacity(29)
    ids = torch.tensor([0, 9, 12, 15, 20, 21, 26, 27, 34])
    expected = torch.tensor([0, 9, 12, 12, 14, 15, 20, 21, 28])
    band = old.encode(ids)
    slots = torch.cat((torch.full((len(ids), 4), .125), band, torch.full((len(ids), 4), .75)), -1)
    saved['state_dict']['store.percept_layout'] = torch.tensor(12)
    saved['state_dict']['store.slots'] = slots
    saved['state_dict']['store.addressing.recurrence_codes'] = slots.clone()
    saved['state_dict']['relations.percept'] = slots[:, 4:]
    saved['structural_extras']['conceptual_spaces'] = {0: {
        'epoch_mtm': {'version': 10, 'candidate_counter': 1,
                      'entries': ({'key': ('ltm', 'unchanged', 27), 'where': band[6],
                                  'timelines': ((((('occurrence', 27), 3, 1, (2, 3), slots[6], slots[6], False, slots[6], (slots[6],), False, slots[6]),), slots[6]),)},)},
        'perception': {'percept_where': band},
        'knowing': {'where': torch.stack((ids, ids+1), -1)}}}
    from collections import deque
    saved['structural_extras']['epoch_rows'] = {'store': {'slots': slots.clone()}}
    saved['training_state'] = {'corpus_runtime': {
        'stm': {'owner': {'fields': {'_live_where': band.clone(), '_live_buffer': slots.clone()}}},
        'expectation': {'fields': {'_inter_context': deque([slots.clone()], maxlen=2)}},
        'mtm': [copy.deepcopy(saved['structural_extras']['conceptual_spaces'][0]['epoch_mtm'])]}}
    result = migrate_wholespace_checkpoint(saved, live, where_slices=slices, percept_width=12)
    moved = result.checkpoint['state_dict']['store.slots']
    assert torch.equal(new.decode_index(moved[:, 4:8]), expected)
    torch.testing.assert_close(moved[:, :4], slots[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(moved[:, 8:], slots[:, 8:], rtol=0, atol=0)
    extra = result.checkpoint['structural_extras']['conceptual_spaces'][0]
    assert extra['epoch_mtm']['entries'][0]['key'] == ('ltm', 'unchanged', 27)
    value = extra['epoch_mtm']['entries'][0]['timelines'][0][1]
    assert int(new.decode_index(value[4:8])) == int(expected[6])
    assert torch.equal(new.decode_index(extra['perception']['percept_where']), expected)
    assert torch.equal(extra['knowing']['where'][:, 0], expected)
    event = extra['epoch_mtm']['entries'][0]['timelines'][0][0][0]
    assert event[0] == ('occurrence', 27) and event[1:4] == (3, 1, (2, 3))
    for field in (event[4], event[5], event[7], event[8][0], event[10]):
        assert int(new.decode_index(field[4:8])) == int(expected[6])
    for value in (result.checkpoint['state_dict']['store.addressing.recurrence_codes'],
                  result.checkpoint['structural_extras']['epoch_rows']['store']['slots']):
        assert torch.equal(new.decode_index(value[:, 4:8]), expected)
    runtime = result.checkpoint['training_state']['corpus_runtime']
    fields = runtime['stm']['owner']['fields']
    assert torch.equal(new.decode_index(fields['_live_where']), expected)
    assert torch.equal(new.decode_index(fields['_live_buffer'][:, 4:8]), expected)
    context = runtime['expectation']['fields']['_inter_context']
    assert context.maxlen == 2 and torch.equal(new.decode_index(context[0][:, 4:8]), expected)
    assert int(new.decode_index(runtime['mtm'][0]['entries'][0]['timelines'][0][1][4:8])) == int(expected[6])
    assert result.diagnostics.migrated_addresses > 0
    torch.testing.assert_close(saved['state_dict']['store.slots'], slots, rtol=0, atol=0)


def test_migration_keeps_corpus_continuation_owners_and_selected_optimizer_slots():
    from checkpoint_migrations import build_optimizer_param_manifest, remap_optimizer_state_by_name
    saved, live, member, _ = migration_fixture()
    old_names = [(key, torch.nn.Parameter(value.clone())) for key, value in saved['state_dict'].items()
                 if value.is_floating_point()]
    old = torch.optim.SGD([value for _, value in old_names], lr=.1, momentum=.9)
    for i, (_, parameter) in enumerate(old_names):
        old.state[parameter]['momentum_buffer'] = torch.full_like(parameter, i+1.)
    saved['optimizer_param_names'] = build_optimizer_param_manifest(old, old_names)
    saved['training_state'] = {'corpus_runtime': {'surfaces': {
        f'wholeSpaces.{index}.subspace': {'marker': index} for index in range(3)}}}
    result = migrate_wholespace_checkpoint(saved, live)
    assert result.checkpoint['training_state']['corpus_runtime']['surfaces'] == {
        'wholeSpace.subspace': {'marker': 2}}
    names = [(key, torch.nn.Parameter(value.clone())) for key, value in live.items()
             if value.is_floating_point()]
    optimizer = torch.optim.SGD([value for _, value in names], lr=.1, momentum=.9)
    restored = remap_optimizer_state_by_name(old.state_dict(),
        result.checkpoint['optimizer_param_names'], optimizer.state_dict(),
        build_optimizer_param_manifest(optimizer, names))
    optimizer.load_state_dict(restored.state)
    parameters = dict(names)
    old_parameters = dict(old_names)
    torch.testing.assert_close(optimizer.state[parameters['wholeSpace.' + member]]['momentum_buffer'],
        old.state[old_parameters['wholeSpaces.2.' + member]]['momentum_buffer'])
    assert not optimizer.state[parameters['candidate_attention.hidden.0.weight']]


def test_rejecting_commit_captures_value_pool_and_source_then_reraises(tmp_path, monkeypatch):
    from test_reference_close import fixture, close
    resolver, _, atoms = fixture()
    owner = resolver.components.owner
    object.__setattr__(owner, '_model', SimpleNamespace(_receipt_directory=tmp_path))
    frame, _, _ = close(resolver, atoms[0], atoms[4], 'mint', (), document='capture', sentence=0)
    original = owner.mtm.bind
    errors = []
    def reject(*args, **kwargs):
        kwargs['value'] = torch.full_like(atoms[0], float('nan'))
        try:
            return original(*args, **kwargs)
        except ValueError as error:
            errors.append(error)
            raise
    monkeypatch.setattr(owner.mtm, 'bind', reject)
    with pytest.raises(ValueError, match='one finite working value') as raised:
        close(resolver, atoms[0], atoms[5], 'bind', (frame,), document='capture', sentence=1, selected=0)
    assert raised.value is errors[0]
    files = list(tmp_path.glob('mtm-value-rejection-*.pt'))
    assert len(files) == 1
    packet = torch.load(files[0], weights_only=False)
    assert packet['document'] == 'capture' and packet['position'] == 1
    assert torch.isnan(packet['working_value']).all()
    assert packet['token'] is not None and packet['members']
    assert packet['pool']['version'] == 10 and packet['pool']['entries']
