"""Definition-row acceptance probes, declared before Stage B implementation."""
from pathlib import Path
import pytest
import torch
from definition_fixtures import witness_object


def model():
    from test_mm_xor import _fresh_model, _PROJECT
    return _fresh_model(str(Path(_PROJECT) / 'data/MM_grammar.xml'))[0]


def admit(m, text):
    cs, ps, ws = m._concept_owner(), m.perceptualSpace, m.wholeSpaces[0]
    native = getattr(ps, 'percept_store', None)
    parts = native.spell_out(text.encode()) if native is not None else list(text.encode())
    pair = cs.interpret_word(parts, ws.property_rows_for_bytes(text), key=text)
    if pair is None:
        return None
    word, obj = pair[:2]
    if cs.interpret.no_object(word):
        obj = witness_object(cs, word)
    return word, obj


def test_22_supplied_object_keeps_its_form_row_and_writes_one_definition():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    before = alloc.next_id, len(alloc.layer()._tensor_rows), len(store.parts)
    learned_at = m.present()
    word, obj = admit(m, 'hello')
    assert word != obj
    assert (alloc.next_id, len(alloc.layer()._tensor_rows), len(store.parts)) == (
        before[0] + 3, before[1] + 2, before[2] + 1)
    assert cs._csw_row_of(word) is not None
    assert cs._csw_row_of(obj) is not None
    row = cs.definitions.row(word, obj)
    assert 'definition' in store.part_row(row)['readings']
    assert store.parts.refs[row].tolist() == [obj, word]
    from RowTime import encoder
    assert int(encoder().decode_index(store.parts.when[row])) == learned_at
    assert ('sym', word) not in cs.concept_parts(obj)
    assert all(not hasattr(cs, name) for name in ('bind_meta', 'meta_members'))
    assert all(not hasattr(alloc, name) for name in ('interpretations', 'word_obj_meta'))


def test_24_index_is_bidirectional_and_rebuilds_from_saved_rows(monkeypatch):
    from Layers import TernaryTruthStore
    m = model()
    word, obj = admit(m, 'hello')
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    index = cs.definitions
    restored_model = model()
    restored_model.load_state_dict(m.state_dict())
    restored_model._restore_structural_extras(m._collect_structural_extras())
    restored = restored_model._concept_owner().definitions
    assert restored.word(form='hello') == word
    assert restored.objects(word) == (obj,)
    assert restored.words(obj) == (word,)
    def forbidden(*args, **kwargs):
        pytest.fail('a definition lookup scanned or rebuilt the store')
    monkeypatch.setattr(store, 'relations', forbidden)
    monkeypatch.setattr(index, 'rebuild', forbidden)
    assert index.word(form='hello') == word
    assert index.objects(word) == (obj,)
    assert index.words(obj) == (word,)


def test_25_two_objects_require_a_selection_and_synonyms_have_a_reverse_index():
    m = model()
    cs = m._concept_owner()
    word, obj = admit(m, 'hello')
    synonym, second = admit(m, 'there')
    cs.interpret.define(word, second)
    cs.interpret.define(synonym, obj)
    assert cs.interpret.forward(word) is None
    assert cs.interpret.forward(word, selected=second) == second
    assert set(cs.definitions.words(obj)) == {word, synonym}
    assert set(cs.definitions.objects(word)) == {obj, second}


def test_26_full_store_refuses_the_whole_word_transaction():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    # The next native symbol is outside this deliberately exhausted Part pool.
    store.parts.capacity = alloc.next_id
    before = alloc.next_id, dict(alloc.placement), dict(alloc.layer()._tensor_rows), len(store)
    try:
        result = admit(m, 'hello')
    except RuntimeError as error:
        assert 'capacity' in str(error) or 'full' in str(error)
    else:
        assert result is None
    assert before == (alloc.next_id, dict(alloc.placement), dict(alloc.layer()._tensor_rows), len(store))


def test_27_definition_rewitness_keeps_its_when_and_store_position():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    first = admit(m, 'hello')
    r0 = cs.definitions.row(*first)
    when = store.parts.when[r0].clone()
    m._advance_when_time()
    admit(m, 'there')
    count = len(store.parts)
    witness = int(store.parts.access_count[r0])
    m._advance_when_time()
    assert admit(m, 'hello') == first
    assert len(store.parts) == count
    assert int(store.parts.access_count[r0]) == witness + 1
    assert cs.definitions.row(*first) == r0
    torch.testing.assert_close(store.parts.when[r0], when, atol=0, rtol=0)
    chain = m.symbolSpace.ensure_sentence_expectation().get_stm_chain(n=count)
    assert len(chain) == len(store.stream_rows(0))


def test_28_forgotten_definition_does_not_survive_in_any_lookup():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    word, obj = admit(m, 'hello')
    row = cs.definitions.row(word, obj)
    store.parts.retain([i for i in store.parts.indices().tolist() if i != row])
    assert cs.definitions.word(form='hello') is None
    assert cs.definitions.objects(word) == ()
    assert cs.definitions.words(obj) == ()
    next_word, next_obj = admit(m, 'hello')
    assert next_word != word and next_obj != obj


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('configuration', ['MM_xor.xml', 'MM_grammar.xml', 'XOR_grammar.xml'])
def test_31_read_forms_wait_for_object_testimony_without_changing_the_configuration(configuration):
    from test_mm_xor import _fresh_model, _PROJECT
    path = Path(_PROJECT) / 'data' / configuration
    before = path.read_bytes()
    m = _fresh_model(str(path))[0]
    with torch.no_grad():
        m(m.inputSpace.prepInput(['hi no', 'hi go', 'we no', 'we go']))
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    for form in ('hi', 'no', 'we', 'go'):
        word = cs.definitions.word(form=form)
        if word is None:
            word, = (word for word, value in cs.interpret._pending.items()
                     if form in value['forms'])
            assert cs.interpret.no_object(word)
            assert cs.interpret.forward(word) == word
            assert cs._csw_row_of(word) is not None
        else:
            obj, = cs.definitions.objects(word)
            assert 'definition' in store.part_row(cs.definitions.row(word, obj))['readings']
            assert cs._csw_row_of(obj) is not None
    assert path.read_bytes() == before


def test_34_learning_codes_cannot_change_definition_identity():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    word, obj = admit(m, 'hello')
    row = cs.definitions.row(word, obj)
    stored = store.parts.refs[row].clone()
    with torch.no_grad():
        cs.similarity_codebook.getW().mul_(.5)
    assert cs.definitions.objects(word) == (obj,)
    assert cs.definitions.words(obj) == (word,)
    assert cs.interpret.forward(word) == obj
    torch.testing.assert_close(store.parts.refs[row], stored, atol=0, rtol=0)


def test_32_retired_head_meta_checkpoint_is_rejected():
    from types import SimpleNamespace
    from Layers import TernaryTruthStore
    from test_cs_sparse_weights import _cs
    from test_structural_checkpoint import _model_with
    fixture = torch.load(Path(__file__).with_name('fixtures') / 'item7_legacy_definition.pt', weights_only=False)
    cs = _cs(nS=64, order=3)
    target = _model_with(cs, SimpleNamespace())
    target.symbolSpace = SimpleNamespace(ltm_store=TernaryTruthStore(cs.nDim, capacity=32))
    object.__setattr__(cs, '_model', target)
    with pytest.raises(ValueError, match='retired META'):
        target._restore_structural_extras(fixture['structural'])
