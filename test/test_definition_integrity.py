"""Additional definition invariants probed before repairs."""
import copy
import pytest
import torch
from test_definition_rows import model, admit
from definition_fixtures import witness_object


def test_copied_definition_index_is_owned_by_the_copied_store():
    m=model(); word,obj=admit(m,'hello'); original=m._concept_owner()._concept_allocator
    restored=copy.deepcopy(original)
    restored.parts.retain([])
    assert original.definitions.deref(word)==obj
    assert restored.definitions.deref(word) is None
    assert restored.definitions._store() is restored.parts


def test_two_pending_words_cannot_overbook_the_last_definition_row():
    m=model(); cs=m._concept_owner(); store=m.symbolSpace.ltm_store
    store.parts.capacity = cs._concept_allocator.next_id + 3
    first=cs.interpret.lookup_word([7],[],form='first')
    alloc=cs._concept_allocator
    before=alloc.next_id,dict(alloc.placement),dict(alloc.layer()._tensor_rows)
    with pytest.raises(RuntimeError,match='capacity'):
        cs.interpret.lookup_word([8],[],form='second')
    assert before==(alloc.next_id,dict(alloc.placement),dict(alloc.layer()._tensor_rows))
    obj=witness_object(cs, first)
    assert cs.interpret.forward(first)==obj
    assert cs.definitions.deref(first)==obj


def test_word_concepts_exist_before_grounded_case_discovery(tmp_path):
    import inspect
    import test_grounded_xor as original
    source=inspect.getsource(original.learn_grounded_xor).split('    # The second presentation admits')[0]
    source+='    cs._commit_autobind_from_stash()\n'
    source+="    assert all(cs.word_concepts(word) for word in ('00', '01', '10', '11'))\n"
    namespace=dict(vars(original));exec(compile(source,__file__+':word-first','exec'),namespace)
    namespace['learn_grounded_xor'](tmp_path,4)


def test_word_definition_survives_identity_resolution_and_pruning():
    from test_cs_sparse_weights import _cs
    cs=_cs()
    word,obj=cs.interpret_word([7],[1],key='word')
    before=cs.concept_parts(word),cs.concept_wholes(word)
    cs.resolve_identities()
    cs.refine_over_collected()
    assert (cs.concept_parts(word),cs.concept_wholes(word))==before


from pathlib import Path
from types import SimpleNamespace

def test_decoder_follows_the_definition_rows_after_reassociation_and_forgetting():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    word, obj = admit(m, 'hello')
    other, _ = admit(m, 'there')
    row = cs._csw_row_of(obj)
    assert cs.word_surface_for_row(row) == b'hello'
    definition = cs.definitions.row(word, obj)
    store.parts.retain([r for r in store.parts.indices().tolist() if r != definition])
    assert cs.word_surface_for_row(row) is None
    cs.interpret.define(other, obj)
    assert cs.word_surface_for_row(row) == b'there'


def test_pending_field_word_reservation_survives_checkpoint(tmp_path):
    from test_grounded_xor import grounded_model
    m, _ = grounded_model(tmp_path)
    cs = m._concept_owner()
    word = cs.interpret.lookup_word([48, 48], [1], form='00', word_reading=True)
    assert word in cs.interpret._field_pending
    saved = m._collect_structural_extras()
    fresh, _ = grounded_model(tmp_path)
    fresh.symbolSpace.ltm_store.load_state_dict(m.symbolSpace.ltm_store.state_dict())
    fresh._restore_structural_extras(saved)
    target = fresh._concept_owner()
    assert target.interpret._pending == cs.interpret._pending
    assert target.interpret._field_pending == cs.interpret._field_pending
    assert word in target.word_concepts('00')


def test_taxonomy_keeps_the_forms_derived_from_definition_rows():
    from test_cs_sparse_weights import _cs
    from ConceptIndex import index_part_row
    cs = _cs(nS=64, order=3)
    word, whole = cs.interpret_word([7], [1], key='animal')
    _, part = cs.interpret_word([8], [1], key='cat')
    parent = index_part_row(cs, part, whole)
    assert parent in cs.word_concepts('animal')
    assert cs._concept_source_order(parent) == cs._concept_source_order(part) + 1


def test_retired_meta_checkpoint_is_permanently_rejected():
    from test_cs_sparse_weights import _cs
    from test_structural_checkpoint import _model_with
    from Layers import TernaryTruthStore
    fixture = torch.load(Path('test/fixtures/item7_legacy_definition.pt'), weights_only=False)
    cs = _cs(nS=64, order=3)
    target = _model_with(cs, SimpleNamespace())
    target.symbolSpace = SimpleNamespace(ltm_store=TernaryTruthStore(cs.nDim, capacity=32))
    object.__setattr__(cs, '_model', target)
    with pytest.raises(ValueError, match='retired META'):
        target._restore_structural_extras(fixture['structural'])


from types import SimpleNamespace

def test_generation_uses_the_definition_vocabulary_without_a_row_spelling_cache():
    m = model()
    cs = m._concept_owner()
    _, obj = admit(m, 'hello')
    cs.__dict__.pop('_row_surfaces', None)
    cs.__dict__.pop('_surface_object_rows', None)
    point = cs.similarity_codebook.getW()[cs._csw_row_of(obj)].detach()
    assert m._generated_word_text(point[None, None], torch.tensor([1])) == ('hello',)


def test_objectless_form_can_generate_and_supply_a_target_without_inventing_a_definition():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    word = cs.interpret.lookup_word([7], [], form='m')
    row = cs._csw_row_of(word)
    before = len(store), cs._concept_allocator.next_id
    assert cs.interpret.no_object(word)
    assert cs.word_surface_for_row(row) == b'm'
    point = cs.similarity_codebook.getW()[row].detach()
    assert m._generated_word_text(point[None, None], torch.tensor([1])) == ('m',)
    torch.testing.assert_close(m._grammar_target_word('m'), point, rtol=0, atol=0)
    assert cs.definitions.objects(word) == ()
    assert before == (len(store), cs._concept_allocator.next_id)


def test_reference_update_law_uses_the_definition_index():
    from Spaces import Space, reference_update_mask
    from test_codebook_update_law import _knob
    m = model()
    cs = m._concept_owner()
    _, obj = admit(m, 'hello')
    vq = SimpleNamespace()
    space = Space.__new__(Space)
    object.__setattr__(space, 'subspace', SimpleNamespace(codebook=lambda: SimpleNamespace(vq=vq)))
    space.serial_mode = False
    _knob('on')
    try:
        assert space.install_reference_update_law(lambda: cs.definitions, side='object')
        assert torch.equal(vq.update_mask_fn(32, None), reference_update_mask(False, [obj], 32))
    finally:
        _knob(None)


def test_object_resolution_does_not_treat_its_word_as_a_second_object():
    m = model()
    cs = m._concept_owner()
    word, obj = admit(m, 'hello')
    assert word != obj
    assert cs.resolve_word_concept('hello', order=0) == obj
@pytest.mark.usefixtures('eager_reading')
def test_reading_and_closing_share_the_grammar_identity_owner():
    from test_ltm_consolidation import _make_model_provisioned, _SERIAL_CONFIG
    model = _make_model_provisioned(_SERIAL_CONFIG)
    assert model._concept_owner() is model.grammatical_thoughts.space


@pytest.mark.usefixtures('eager_reading')
def test_new_definitions_do_not_alias_later_sentence_identities():
    from test_ltm_consolidation import _make_model_provisioned, _SERIAL_CONFIG
    model = _make_model_provisioned(_SERIAL_CONFIG)
    store = model.symbolSpace.ltm_store
    symbols = set(store.parts.refs[store.relations('definition')].reshape(-1).tolist())
    rows = store.rows_of_origin(store.ORIGIN_PROVISIONED)
    assert symbols.isdisjoint(store.row_ids[rows].tolist())
