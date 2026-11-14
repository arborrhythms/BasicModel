"""Perceptual bounds and the common order-zero collection reader (§13.1)."""
import inspect
from pathlib import Path
import pytest
import torch
from Models import BaseModel
from test_item18_finish import named_scalar_model, scalar_answer

_load_weights = BaseModel.load_weights


def test_shared_value_does_not_mix_two_words_definition_views(named_scalar_model):
    from ScalarPercepts import observe, perceive
    from PerceptCollections import objects
    model, _ = named_scalar_model
    owner = model._concept_owner()
    native = model.perceptualSpace.percept_store
    words = [owner.interpret_word(native.spell_out(form.encode()),
        model.wholeSpace.property_rows_for_bytes(form), key=form)[0]
        for form in ('epsilon', 'zeta')]
    reference = ('bp', perceive(owner, 4.))
    for view in range(6):
        for word in words:
            observe(owner, word, 4., reference, stream=0)
        if view + 1 < owner.concept_mint_count:
            assert all(not objects(owner, word) for word in words)
    identities = [objects(owner, word)[0] for word in words]
    assert len(set(identities)) == 2
    for identity in identities:
        assert int(owner.similarity_codebook.trust.views[owner._csw_row_of(identity)]) == 6


def test_basic_percept_reference_reads_without_a_concept_annotation(named_scalar_model):
    from ScalarPercepts import perceive, code
    from MemberRead import read
    model, _ = named_scalar_model
    owner = model._concept_owner()
    reference = ('bp', perceive(owner, 4.))
    torch.testing.assert_close(read(owner, reference), code(owner, 4.))


def test_scalar_object_collects_value_bounded_basic_percepts(named_scalar_model):
    model, objects = named_scalar_model
    owner = model._concept_owner()
    cid = objects['alpha']
    assert owner._concept_allocator.order_of(cid) == 0
    row = owner._csw_row_of(cid)
    features = owner._concept_allocator.layer(0).features
    percepts = [col // 2 for target, col in features._index if target == row]
    assert percepts
    basic = owner.basic_percepts
    from NumericPercepts import feeling
    value = feeling(torch.tensor(1.))
    for percept in percepts:
        parts, wholes = basic.occurrences[percept]
        assert parts and wholes
        assert basic.part_codes(parts).amax().item() == pytest.approx(value.item())
        assert basic.whole_codes(wholes).amax().item() == pytest.approx(value.item())


def test_intervals_generalize_their_percepts_and_have_distinct_codes(named_scalar_model):
    from ScalarPercepts import code, observe, reconstruct
    model, objects = named_scalar_model
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    for form, value in (('alpha', 3.), ('beta', 2.)):
        word = owner.interpret.word_for_form(form)
        row = store.append_idea(code(owner, value))
        observe(owner, word, value, store.occurrence_of(row), stream=0)
    a, b = [owner.similarity_codebook.lookup_rows([owner._csw_row_of(objects[name])])[0]
            for name in ('alpha', 'beta')]
    assert not torch.equal(a, b)
    assert set(reconstruct(owner, value=1.5)) == {'alpha', 'beta'}
    assert reconstruct(owner, value=2.5) == ('alpha',)
    assert reconstruct(owner, value=4.) == ()


def test_scalar_has_no_private_bounds_or_membership_reader():
    import ScalarPercepts, FieldProjection
    assert not hasattr(ScalarPercepts, 'state')
    assert not hasattr(ScalarPercepts, 'membership')
    retired = 'field_' + 'memberships'
    for path in Path('bin').glob('*.py'):
        assert retired not in path.read_text(), path
    assert 'ScalarPercepts' not in inspect.getsource(FieldProjection.stimulate)


def test_point_and_native_read_share_the_collection_boundaries(named_scalar_model):
    from ScalarPercepts import code, observe, reading
    from FieldProjection import stimulate
    model, objects = named_scalar_model
    owner, store = model._concept_owner(), model.symbolSpace.ltm_store
    word = owner.interpret.word_for_form('alpha')
    row = store.append_idea(code(owner, 3.))
    observe(owner, word, 3., store.occurrence_of(row), stream=0)
    from NumericPercepts import feeling
    low, high = feeling(torch.tensor([1., 3.]))
    for value in (.5, 1., 2., 3., 4.):
        point = code(owner, value)[None, None]
        native, projected = reading(owner, point), stimulate(owner, point)
        actual = [field.evidence[field.concept_ids == objects['alpha'], 0, 0]
                  for field in (native, projected)]
        torch.testing.assert_close(*actual)
        felt = feeling(torch.tensor(value))
        expected = min(1., float(felt/low), float(high/felt))
        torch.testing.assert_close(actual[0], torch.tensor([[expected, 0.]]))


def test_value_percepts_use_the_basic_checkpoint_and_shared_retirement(named_scalar_model):
    import copy
    from ScalarPercepts import reconstruct
    model, objects = named_scalar_model
    owner = model._concept_owner()
    basic = owner.basic_percepts
    before = tuple(reconstruct(owner, value=value) for value in (1., 2., 4.))
    saved = copy.deepcopy(basic.state_dict())
    basic.value_codes.zero_()
    basic.value_count.zero_()
    basic.index.clear()
    basic.load_state_dict(saved)
    assert tuple(reconstruct(owner, value=value) for value in (1., 2., 4.)) == before
    owner._concept_allocator.retired.add(objects['alpha'])
    assert reconstruct(owner, value=1.) == ('beta',)


def test_text_uses_the_same_percept_collection_reader(named_scalar_model, monkeypatch):
    from PerceptCollections import observe
    model, _ = named_scalar_model
    owner = model._concept_owner()
    native = model.perceptualSpace.percept_store
    word, *_ = owner.interpret_word(native.spell_out(b'delta'),
        model.wholeSpace.property_rows_for_bytes('delta'), key='delta')
    source = owner.interpret.word_for_form('alpha')
    percept = owner.basic_percepts.words[source]
    for view in range(6):
        from MemberRead import read
        store = model.symbolSpace.ltm_store
        row = store.append_idea(read(owner, ('bp', percept), ('bp', percept)))
        identity = observe(owner, word, percept, store.occurrence_of(row), stream=0)
    calls = []
    shared = owner.basic_percepts.read_collection
    monkeypatch.setattr(owner.basic_percepts, 'read_collection',
                        lambda *args: (calls.append(True), shared(*args))[1])
    with torch.no_grad():
        model.understand(model.inputSpace.prepInput(['alpha']))
    assert calls
    assert owner._concept_allocator.order_of(identity) == 0
    assert ('bp', percept) in owner._percept_field.keys


def test_percept_reference_promotes_without_an_event_row(named_scalar_model):
    from ScalarPercepts import observe, perceive
    model, _ = named_scalar_model
    owner = model._concept_owner()
    native = model.perceptualSpace.percept_store
    word, *_ = owner.interpret_word(native.spell_out(b'delta'),
        model.wholeSpace.property_rows_for_bytes('delta'), key='delta')
    percept = perceive(owner, 2.)
    reference = ('bp', percept)
    for _ in range(owner.concept_mint_count):
        identity = observe(owner, word, 2., reference, stream=0)
    assert identity is not None
    assert owner._concept_allocator.order_of(identity) == 0
    row = owner._csw_row_of(identity)
    assert (row, 2 * percept) in owner._concept_allocator.layer(0).features._index
    assert owner._closed_clause_store().location(reference) is None


def test_value_collection_restores_through_the_integrated_checkpoint(
        named_scalar_model, tmp_path, monkeypatch):
    from ScalarPercepts import reconstruct
    from test_compiled_word_chunk import _tiny_canonical_model
    model, objects = named_scalar_model
    path = tmp_path / 'percept-collections.ckpt'
    model.save_weights(path)
    folder = tmp_path / 'restore'
    folder.mkdir()
    restored = _tiny_canonical_model(folder, monkeypatch, batch_size=1)
    try:
        # The construction fixture disables automatic checkpoint loading.
        assert _load_weights(restored, path)
        owner = restored._concept_owner()
        assert reconstruct(owner, value=1.) == ('alpha', 'beta')
        assert reconstruct(owner, value=2.) == ('gamma',)
        assert reconstruct(owner, value=4.) == ()
        for form, cid in objects.items():
            assert owner.interpret.lexical_form(cid) == form
            assert owner._concept_allocator.order_of(cid) == 0
            assert int(owner.similarity_codebook.trust.views[owner._csw_row_of(cid)]) == 6
    finally:
        restored.End()
        restored.symbolSpace.soft_reset()
