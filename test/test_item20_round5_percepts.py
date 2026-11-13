"""Basic percept boundaries, live reads, recurrence and fixed construction."""
from types import SimpleNamespace as NS
import torch
import pytest
from torch import nn

from BasicPercepts import BasicPercepts
from test_item20_round5_memory import receipt


def percepts(capacity=4):
    from Spaces import Codebook
    from test_form_subspaces import property_book
    native = Codebook(); native.W = nn.Parameter(torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
    whole = property_book(rows=2)
    with torch.no_grad():
        whole.primitive_properties.members[0, 97] = .4
        whole.primitive_properties.members[1, 97] = .7
        whole.primitive_properties.observed[:, 97] = True
    whole.begin_forward()
    basic = BasicPercepts(NS(_basis=native), 3, 256, capacity)
    basic.bind(whole)
    return basic, native, whole


def test_basic_table_is_preallocated_and_full_raises_without_resize():
    basic, native, whole = percepts(2)
    pointer, shape = basic.codes.data_ptr(), basic.codes.shape
    a = basic.reserve((0,), (0,), word=10)
    b = basic.reserve((0, 1), (1,), word=11)
    assert a != b and int(basic.count) == 0
    torch.testing.assert_close(basic.code(a), torch.cat((native.W[0], whole.lookup_rows(0)[3:])))
    torch.testing.assert_close(basic.code(b)[:3], native.W[:2].amax(0))
    with pytest.raises(OverflowError, match='capacity'): basic.reserve((2,), ())
    assert (basic.codes.data_ptr(), basic.codes.shape) == (pointer, shape)
    receipt('basic-construction', dict(capacity=2, reserved=2, promoted=0, shape=list(shape),
        full_raises=True, resized=False, centroid=True))


def test_basic_read_ties_both_towers_and_keeps_the_property_gradient():
    basic, native, whole = percepts()
    row = basic.reserve((0,), (0,))
    primitive = whole.primitive_properties
    spans = torch.tensor([[[0, 1]]])
    source = (torch.tensor([[0]]), spans, primitive, torch.tensor([[97]]), spans)
    field = basic.read(source, spans, (row,))
    torch.testing.assert_close(field.values[0, 0, 0], torch.tensor([.4, .6]))
    gradient, = torch.autograd.grad(field.values[0, 0, 0, 0], (primitive.members,))
    assert gradient[0, 97] == 1 and gradient.count_nonzero() == 1
    absent = (torch.tensor([[1]]), spans, primitive, torch.tensor([[97]]), spans)
    assert basic.read(absent, spans, (row,)).values.count_nonzero() == 0
    from FormContainment import audit
    result = audit(basic)
    assert result['premises'] == 'basic_percepts'
    assert not result['tower0']['violating_pairs'] and not result['tower1']['violating_pairs']
    receipt('basic-read-gradient', dict(positive=.4, negative=.6, absent=0,
        member_gradient=float(gradient[0, 97]), audit=result))


def test_basic_read_preserves_ordered_part_witnesses_after_native_fusion():
    from Layers import RadixLayer
    basic, _, whole = percepts()
    native = RadixLayer(3, initial_cap=8)
    a, b, ab = [native.insert(value) for value in (b'a', b'b', b'ab')]
    native.part_groups[ab] = (a, b)
    object.__setattr__(basic, 'native', native)
    forward, reverse, repeat = [basic.reserve(parts, ()) for parts in ((a, b), (b, a), (a, a))]
    spans = torch.tensor([[[0, 2]]])
    field = basic.read((torch.tensor([[ab]]), spans, whole.primitive_properties,
        torch.tensor([[97, 98]]), spans), spans, (forward, reverse, repeat))
    assert field.values[:, 0, 0, 0].tolist() == [1., 0., 0.]
    assert field.spans[0, :2].tolist() == [[0, 1], [1, 2]]
    assert field.events[0, 0, 0, :2, 0].tolist() == [1., 1.]
    receipt('basic-native-descent', dict(fused_rows=1, part_events=2,
        preserved_order=True, preserved_multiplicity=True, byte_retiling=False))


def test_basic_compose_presence_keeps_the_existing_weighted_mean_pullback():
    from FormCentroids import BasicSpanWholes, presence_read
    basic, native, whole = percepts()
    a, b = basic.reserve((0,), (0,)), basic.reserve((1,), (1,))
    rows = torch.tensor([[a, b]])
    counts = torch.nn.functional.one_hot(torch.tensor([[97, 97]]), 256).float()
    weights, codes = basic.word_weights(counts, rows)
    weights = torch.diag_embed(weights)
    prefix = torch.cat((weights.new_zeros(1, 1, 2), weights.cumsum(1)), 1)
    bank = BasicSpanWholes(prefix, codes)
    whole_read = bank.read_presence(torch.tensor([[[0, 2]]]))
    expected = presence_read(weights.sum(1, keepdim=True), codes)
    torch.testing.assert_close(whole_read.carrier, expected.carrier)
    gradients = torch.autograd.grad(whole_read.carrier.sum(), (whole.primitive_properties.members, native.W))
    assert all(torch.isfinite(value).all() and bool(value.abs().any()) for value in gradients)
    receipt('basic-compose-gradient', dict(member_norm=float(gradients[0].norm()),
        native_norm=float(gradients[1].norm()), finite=True, weighted_presence_unchanged=True))


def test_recurring_basic_collection_promotes_once_without_an_order_zero_concept():
    from test_cs_sparse_weights import _cs
    from Memory import LTM0
    from ClauseRow import attach_clause_index
    from MemoryLink import Occurrence
    basic, native, whole = percepts()
    owner = _cs(nS=64, dim=259)
    object.__setattr__(owner, '_basic_percepts', basic)
    store = LTM0(259, capacity=8, percept_width=0)
    attach_clause_index(NS(), store, owner)
    owner.concept_mint_count = 3
    row = basic.reserve((0,), (0,), word=10)
    concept_ids = set(owner._concept_allocator.symbol_types)
    collection = None
    for clock in range(3):
        point = basic.code(row)
        index = store.append_idea(point, address=301+clock, time=False)
        key = owner.link(Occurrence(store.occurrence_of(index), concept=('bp', row), kind='percept', order=-1), {},
            value=point, where=None, when=clock, stream=0)
        assert collection is None or key == collection
        collection = key
        owner.mtm.close(store, stream=0)
        owner.mtm.promote_candidates(owner)
        assert int(basic.count) == int(clock == 2)
    owner.mtm.promote_candidates(owner)
    assert int(basic.count) == 1 and basic.promoted[row]
    assert set(owner._concept_allocator.symbol_types) == concept_ids
    receipt('basic-recurrence', dict(closes=3, promoted_rows=1, new_order_zero_concepts=0,
        transient_released=owner.mtm.entries[collection].transient is None))


def test_supplied_training_form_declares_the_basic_percept_at_admission(monkeypatch):
    from test_mm_xor import _fresh_model
    import util
    from Corpus import admit_training_forms
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        words = admit_training_forms(model, ('hello world',))
        table = model._concept_owner().basic_percepts
        assert words and all(bool(table.promoted[table.words[word]]) for _, word in words)
        assert table.capacity == 256
        receipt('basic-declared', dict(forms=len(words), declared=len(words), capacity=table.capacity))
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_basic_centroid_small_mass_pullback_preserves_the_presence_repair():
    """§45.8's local reciprocal-mass failure mechanism on the new percept bank."""
    from FormCentroids import BasicSpanWholes, weighted_mean
    basic, native, whole = percepts()
    basic = basic.double(); native = native.double(); whole = whole.double()
    basic.native._basis = native
    basic.bind(whole)
    mass = 1.23036884e-7
    with torch.no_grad():
        whole.primitive_properties.members[:, 97] = torch.tensor([.25, .75], dtype=torch.double) * mass
    whole.begin_forward()
    a, b = basic.reserve((0,), (0,)), basic.reserve((1,), (1,))
    counts = torch.nn.functional.one_hot(torch.tensor([[97, 97]]), 256).double()
    weights, codes = basic.word_weights(counts, torch.tensor([[a, b]]))
    weights = torch.diag_embed(weights)
    weights.retain_grad()
    bank = BasicSpanWholes(torch.cat((weights.new_zeros(1, 1, 2), weights.cumsum(1)), 1), codes)
    read = bank.read_presence(torch.tensor([[[0, 2]]]))
    cotangent = torch.linspace(.1, .9, codes.shape[-1], dtype=torch.double)[None, None]
    actual = torch.autograd.grad((read.carrier*cotangent).sum(),
        (weights, whole.primitive_properties.members), retain_graph=True)
    mean, total = weighted_mean(weights.sum(1, keepdim=True), codes)
    hypothetical = torch.autograd.grad((mean*cotangent).sum(),
        (weights, whole.primitive_properties.members), retain_graph=True)
    expected = torch.autograd.grad(((weights.sum(1, keepdim=True)@codes)*cotangent).sum(),
        (weights, whole.primitive_properties.members))
    for gradient, wanted in zip(actual, expected):
        torch.testing.assert_close(gradient, wanted)
        assert torch.isfinite(gradient).all()
    assert hypothetical[0].norm() > actual[0].norm()*1e4
    receipt('basic-small-mass-attribution', dict(mass=float(total.detach()),
        source='basic-percept compose centroid', registered_objective=False,
        probe='local carrier cotangent; native objective attribution is recorded separately',
        normalized_mean_is_hypothetical=True, archived_trajectories_replayed=False,
        cotangent_norm=float(cotangent.norm()),
        presence_weight_norm=float(actual[0].norm()), presence_member_norm=float(actual[1].norm()),
        normalized_weight_norm=float(hypothetical[0].norm()), normalized_member_norm=float(hypothetical[1].norm()),
        reciprocal_mass_absent_from_consumer=True))


def test_one_admitted_form_uses_one_basic_percept_and_only_two_pole_columns(monkeypatch):
    from test_mm_xor import _fresh_model
    from Corpus import admit_training_forms
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        owner = model._concept_owner()
        first = admit_training_forms(model, ('hello',))
        basic = owner.basic_percepts
        before = len(basic.index)
        second = admit_training_forms(model, ('hello',))
        assert first == second and len(basic.index) == before
        _, word = first[0]
        bp = basic.words[word]
        parts, wholes = basic.occurrences[bp]
        expected = torch.cat(basic.subspace_bounds(bp))
        torch.testing.assert_close(basic.code(bp), expected)
        definitions = owner._concept_allocator.layer().features
        word_row = owner._csw_row_of(word)
        assert {column for (row, column) in definitions._index if row == word_row} == {2*bp}
        assert definitions.nInput == 2*basic.capacity
        receipt('basic-admission-layout', dict(forms=1, repeated_admission_new_rows=0,
            basic_rows=before, word_column=2*bp, stride=2, capacity=basic.capacity,
            parts=len(parts), wholes=len(wholes), centroid=True))
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_native_typed_slots_and_composed_whole_keep_the_basic_member_pullback(tmp_path, monkeypatch):
    from test_compiled_word_chunk import _tiny_canonical_model
    from reading_fixtures import force_absolute_reading
    from Corpus import admit_training_forms
    from FormCentroids import stage, BasicSpanWholes
    import util
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8',
        concept_rows=256, part_rows=128)
    model._tensor_peer_while_eager = True
    model.reconstruction_placement = 'eager'
    model._chart_compose_per_word = lambda: None
    force_absolute_reading(model)
    try:
        admit_training_forms(model, ('a b',))
        model.forward(model.inputSpace.prepPackedInput([['a b']]))
        member = model.wholeSpace.subspace.what.primitive_properties.members
        slots = model.inputSpace._ar_whole_reference_vectors
        slot_gradient, = torch.autograd.grad(slots.sum(), member, retain_graph=True)
        bank = stage(model.wholeSpace, basic=model._concept_owner().basic_percepts,
                     word_rows=model.inputSpace._ar_basic_percept_ids)
        assert isinstance(bank, BasicSpanWholes)
        whole = bank.read_presence(torch.tensor([[[0, 2]]]))
        direction = torch.linspace(.1, .9, whole.carrier.shape[-1])[None, None]
        gradient, = torch.autograd.grad((whole.carrier*direction).sum(), member)
        for value in (slot_gradient, gradient):
            assert torch.isfinite(value).all() and bool(value.abs().any())
        receipt('basic-native-pullbacks', dict(typed_slot_norm=float(slot_gradient.norm()),
            compose_member_norm=float(gradient.norm()), mass=whole.total.detach().tolist(),
            native_forward=True, supplied_forms=2, bank=type(bank).__name__,
            finite=True, extra_training_steps=0))
    finally:
        model.End()


def test_admitted_form_with_weak_definition_remains_known_without_a_denotation():
    from test_cs_sparse_weights import _cs
    from order_fixtures import force_order
    owner = _cs()
    word = owner.interpret.lookup_word((97,), (), form='weak-definition')
    concept = owner.new_concept(); force_order(owner, concept, 0)
    owner._csw_concept_row(0, concept)
    owner.interpret.define(word, concept, evidence=(0., 0.))
    assert word in owner.definitions.word_ids and not owner.definitions.objects(word)
    assert owner.interpret.lookup_word((97,), (), form='weak-definition') == word
    before = dict(owner._concept_allocator.placement)
    assert owner.interpret.forward(word) is None
    assert owner._concept_allocator.placement == before
    with pytest.raises(ValueError, match='admitted word'):
        owner.interpret.forward(999999)
    receipt('weak-definition-admission', dict(admitted_forms=1, denotations=0,
        invented_objects=0, unknown_identity_rejected=True))
