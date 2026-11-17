"""Round A certificates, including the carried continuation repairs."""
from pathlib import Path
import json

import pytest
import torch
from test_item18_trust import truth_model


def test_first_definition_edge_joins_its_objective_optimizer():
    from Layers import SparseLayer
    from Optimizer import Adam, SGD, MultiOptimizer
    anchor = torch.nn.Parameter(torch.tensor(.2))
    adaptive = Adam([anchor], lr=.001)
    adaptive.objective_owner = 'adaptive'
    reconstruction = SGD([dict(params=[], action_rate_power=0)], lr=.002, momentum=.9)
    reconstruction.objective_owner = 'reconstruction'
    optimizer = MultiOptimizer([adaptive, reconstruction])
    optimizer.route_parameters = lambda parameters: {'reconstruction': parameters}
    matrix = SparseLayer(2, 1)
    matrix._optimizer = optimizer
    matrix.add_edge(0, 0, weight=.5)
    assert all(matrix.values is not p for group in adaptive.param_groups for p in group['params'])
    group = next(group for group in reconstruction.param_groups
                 if any(matrix.values is p for p in group['params']))
    assert group['action_rate_power'] == 0 and group['lr'] == .002
    matrix.values.sum().backward()
    optimizer.step()
    assert 'momentum_buffer' in optimizer.state[matrix.values]
    torch.testing.assert_close(matrix.values, torch.tensor([.498]))




def test_argument_metadata_round_trip_has_identical_bytes():
    from PartsOfSpeech import PartsOfSpeech
    arguments = ('part_argin1', 'part_argin2', 'part_argout1')
    first = PartsOfSpeech(arguments)
    first.concepts = dict(zip(first.arguments, (1, 2, 3)))
    first.uses[first.operations[0]] = 4
    saved = first.get_extra_state()
    restored = PartsOfSpeech(tuple(name.encode().decode() for name in arguments))
    restored.concepts = dict(zip(restored.arguments, (1, 2, 3)))
    restored.set_extra_state(saved)
    assert torch.equal(saved, restored.get_extra_state())


def test_later_sentence_reads_a_promoted_name_from_its_current_reference():
    from types import SimpleNamespace
    from PrependedSymbols import SymbolEntries, refresh_words, word_view, lookup_view
    from test_cs_sparse_weights import _cs
    owner = _cs(nS=128, order=2, dim=16)
    word = owner.interpret.lookup_word([1], [2], form='label')
    transient, row, order = owner.interpret.reading(word)
    entries = SymbolEntries(1, 2, 2, 16)
    for field, values in (
            ('concept_ids', [word, word]), ('concept_rows', [row, row]),
            ('object_ids', [transient, transient]), ('object_rows', [row, row]),
            ('object_orders', [order, order]), ('lookup_rows', [row]*4),
            ('lookup_sentence_ids', [0, 1, 0, 1])):
        entries.stage_words(field, torch.tensor([values]))
    atoms = owner.similarity_codebook.lookup_rows(torch.tensor([[row, row]]))
    entries.stage_words('lookup_atoms', torch.cat((atoms, atoms), 1))
    source = SimpleNamespace(symbol_entries=entries, _ar_word_object_atoms=atoms,
                             _packed_sentence_ids=torch.tensor([[0, 1]]))
    model = SimpleNamespace(inputSpace=source, _concept_owner=lambda: owner)
    obj = owner.new_concept(order=0)
    owner.mtm.promote(('name', word), obj)
    reference = owner._csw_row_of(obj)
    refresh_words(model, 1)
    assert word_view(source, 'object_ids').tolist() == [[transient, obj]]
    assert word_view(source, 'object_rows').tolist() == [[row, reference]]
    assert word_view(source, 'object_orders')[0, 1] == owner._row_order(reference)
    assert lookup_view(source, 'lookup_rows').tolist() == [[row, row, row, reference]]
    expected = owner.similarity_codebook.lookup_rows(torch.tensor(reference))
    torch.testing.assert_close(source._ar_word_object_atoms[0, 1], expected)
    torch.testing.assert_close(lookup_view(source, 'lookup_atoms')[0, 3], expected)


def test_modifier_preserves_checkpoint_gain_and_pullbacks(monkeypatch):
    """Certificate 69: fixed, nonzero gain from the pushed pre-rename code."""
    import Language
    from util import TheXMLConfig
    assert hasattr(Language, 'ModifierLayer')
    fixture = json.loads((Path(__file__).parent/'fixtures/item18-modifier-gain.json').read_text())
    monkeypatch.setitem(TheXMLConfig.data, 'architecture', {'data': {'dataType': 'embedding'}})
    monkeypatch.setitem(TheXMLConfig.data, 'PartSpace', {'nDim': 12})
    monkeypatch.setitem(TheXMLConfig.data, 'ConceptualSpace', {'nDim': 20})
    operator = Language.ModifierLayer(20, 20).double()
    with torch.no_grad():
        for name, parameter in operator.meaning_action.named_parameters():
            parameter.copy_(torch.tensor(fixture['parameters'][name], dtype=parameter.dtype))
    left = torch.tensor(fixture['left'], dtype=torch.float64, requires_grad=True)
    right = torch.tensor(fixture['right'], dtype=torch.float64, requires_grad=True)
    result = operator.compose(left, right)
    result.sum().backward()
    for value, name in ((result, 'parent'), (left.grad, 'left_gradient'), (right.grad, 'right_gradient')):
        torch.testing.assert_close(value, torch.tensor(fixture[name], dtype=value.dtype), rtol=0, atol=0)
    assert not torch.equal(result[:, 12:], left[:, 12:])


def test_conversation_passes_zero_trust_to_its_closings(truth_model, monkeypatch):
    import WholePaths
    model = truth_model
    finish = WholePaths.finish
    seen = []
    def closing(*args, **kwargs):
        assert '_conversation_read' not in model.__dict__
        seen.append(kwargs['trust'])
        return finish(*args, **kwargs)
    monkeypatch.setattr(WholePaths, 'finish', closing)
    store = model.symbolSpace.ltm_store
    start = len(store)
    model.infer('hello')
    assert seen and all(value == 0. for value in seen)
    assert len(store) > start and not store.trust[start:len(store)].any()


def test_75_concept_orders_are_fixed_contiguous_blocks(truth_model):
    owner = truth_model._concept_owner()
    alloc = owner._concept_allocator
    caps = owner._order_caps()
    assert not hasattr(alloc, 'reference_orders')
    assert all(value is None for value in alloc.symbols.values())
    assert sum(caps) == owner.nVectors
    start = 0
    for order, capacity in enumerate(caps):
        assert owner.order_slice(order) == (start, start + capacity)
        start += capacity
    for row in alloc.layer()._tensor_row_keys:
        concept = owner.concept_id_at_row(row)
        order = alloc.order_of(concept)
        low, high = owner.order_slice(order)
        assert low <= row < high, (concept, order, row, low, high)


def test_75_full_order_block_raises_before_minting():
    from test_concept_capacity_policy import _cs
    from Spaces import ConceptRowCapacityError
    owner = _cs(16)
    alloc = owner._concept_allocator
    capacity = owner._order_caps()[1]
    owner.prebuild_formal_arguments(tuple(f'argument_{n}' for n in range(capacity)))
    before = (alloc.next_id, dict(alloc.layer()._tensor_rows), dict(alloc.formal_arguments))
    with pytest.raises(ConceptRowCapacityError, match='order 1'):
        owner.prebuild_formal_arguments(('another_argument',))
    assert (alloc.next_id, dict(alloc.layer()._tensor_rows), dict(alloc.formal_arguments)) == before


def test_word_bank_refresh_keeps_the_previous_sentences_gradient():
    from PrependedSymbols import SymbolEntries
    bank = SymbolEntries(1, 2, 1, 3)
    first = torch.arange(6., requires_grad=True).reshape(1, 2, 3)
    bank.stage_words('lookup_atoms', first)
    reading = bank.read('lookup_atoms', words=True).gather(1, torch.tensor([[[1, 1, 1]]]))
    bank.stage_words('lookup_atoms', first * 2)
    gradient, = torch.autograd.grad(reading.sum(), first)
    torch.testing.assert_close(gradient, torch.tensor([[[0., 0., 0.], [1., 1., 1.]]]))


def test_73_one_word_keeps_its_objects_at_each_order():
    from test_cs_sparse_weights import _cs
    owner = _cs(nS=128, order=3, dim=16)
    word = owner.interpret.lookup_word([1], [2], form='label')
    member = owner.new_concept()
    owner.add_part(member, 7)
    owner._csw_concept_row(0, member)
    parent = owner.new_concept()
    owner.add_part(parent, ('sym', member))
    owner._csw_concept_row(1, parent)
    owner.interpret.define(word, member)
    owner.interpret.define(word, parent)
    owner.definitions.rebuild()
    assert set(owner.definitions.objects(word)) == {member, parent}
    assert owner.definitions.words(member) == owner.definitions.words(parent) == (word,)
    assert owner.interpret.forward(word) == parent
    assert owner.interpret.forward(word, order=0) == member


def test_73_staged_interpretation_reads_each_words_highest_object():
    from ReferenceContext import ReferenceTypes, resolve_highest
    types = ReferenceTypes(torch.tensor([[10, 20]]), torch.tensor([0, 1, 2]),
        torch.tensor([[[1, 2, 3], [4, -1, 5]]]),
        torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4))
    for identities in (types.sources, torch.tensor([[1, 4]])):
        values, chosen = resolve_highest(torch.zeros(1, 2, 4), identities, types)
        assert chosen.tolist() == [[3, 5]]
        torch.testing.assert_close(values, types.values[:, :, 2])


def test_75_checkpoint_reallocation_preserves_every_row_and_identity():
    from GrammarConceptCheckpoint import row_plan
    from test_cs_sparse_weights import _cs
    owner = _cs(nS=16, order=2)
    layer = dict(nOutput=16, tensor_rows={('shared', 10): 14, ('shared', 11): 15,
                                       ('shared', 12): 2})
    allocator = dict(reference_orders={10: 0, 11: 1, 12: 2})
    indices, seats, additions = row_plan(owner, allocator, layer, [])
    assert not additions and sorted(indices) == list(range(16))
    for key, old in layer['tensor_rows'].items():
        assert indices[seats[key]] == old
        assert owner._row_order(seats[key]) == allocator['reference_orders'][key[1]]
    values = torch.arange(32).reshape(16, 2)
    moved = values[indices]
    for key, old in layer['tensor_rows'].items():
        assert torch.equal(moved[seats[key]], values[old])


def test_retained_time_lookup_decodes_once_and_native_writes_update_it(monkeypatch):
    import RowTime
    from Memory import LTM0
    store = LTM0(12, capacity=16, percept_width=8)
    for value in range(8):
        store.append_idea(torch.full((12,), value / 8.))
    store.clear_caches()
    calls = []
    decode = RowTime._ModelTimeEncoding.decode_index
    def measured(clock, values):
        calls.append(values.shape)
        return decode(clock, values)
    monkeypatch.setattr(RowTime._ModelTimeEncoding, 'decode_index', measured)
    for _ in range(3):
        for row in range(8):
            assert store.index_of_row(store.row_name(row)) == row
    assert len(calls) == 1
    saved = {key: value.clone() for key, value in store.state_dict().items()}
    row = store.append_idea(torch.ones(12))
    name = store.row_name(row)
    assert store.index_of_row(name) == row and len(calls) == 1
    store.restore_runtime_state(saved)
    assert store.index_of_row(name) is None and len(calls) == 2
    store.compact_rows((1, 3, 5))
    assert [store.row_name(row) for row in range(3)] == [1, 3, 5]
    assert store.index_of_row(3) == 1 and store.index_of_row(2) is None
    assert not set(store.caches) & set(store.state_dict())


def test_retained_exclusion_counts_follow_append_compaction_and_restore():
    from Memory import LTM0
    store = LTM0(12, capacity=24, percept_width=8)
    points = torch.zeros(2, 12)
    points[0, 8], points[1, 9] = 1., 1.
    for point in points:
        for _ in range(8):
            store.append_idea(point)
    assert store.coordinate_exclusions(8,order=0) == ((8, 9),)
    before = {key: value.clone() for key, value in store.state_dict().items()}
    counts = store._coordinate_counts
    joint = store._coordinate_joint
    store.append_idea(points.sum(0))
    assert store.coordinate_exclusions(8,order=0) == ()
    assert store._coordinate_counts is counts and store._coordinate_joint is joint
    store.restore_runtime_state(before)
    assert store.coordinate_exclusions(8,order=0) == ((8, 9),)
    store.compact_rows(range(1, 16))
    assert store.coordinate_exclusions(8,order=0) == ()


def test_derived_concept_projection_and_rebuild_are_identical():
    """A checkpoint rebuild must read the same placement as a kept update."""
    from test_review13_subspaces import fixture
    owner, bank, _, _ = fixture(meaning_pairs=16)
    bank = owner.similarity_codebook
    identity = owner.new_concept(order=0)
    row = owner._csw_row_of(identity)
    with torch.no_grad():
        # The placement and radius of a changed word in the failed native
        # continuation. They expose repeated normalization's float32 drift.
        bank.mereology.placement[row] = torch.tensor([
            0., .5278531312942505, 0., 0., .5755428075790405,
            .5755428075790405, 0., .0014073170023038983, 0., 0.,
            .2149057537317276, 0., .11263097077608109, 0., 0., 0.])
        bank.trust.lengths[row, 0] = .4842294454574585
    before = bank.lookup_rows(row).clone()
    bank.trust.project([row])
    assert torch.equal(bank.W[row], before)
    bank.mereology.placement_dirty.fill_(True)
    bank.mereology.begin_forward()
    assert torch.equal(bank.W[row], before)
