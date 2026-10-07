"""Definition-row acceptance probes, declared before Stage B implementation."""
from pathlib import Path
import pytest
import torch


def model():
    from test_mm_xor import _fresh_model, _PROJECT
    return _fresh_model(str(Path(_PROJECT) / 'data/MM_grammar.xml'))[0]


def admit(m, text):
    cs, ps, ws = m._concept_owner(), m.perceptualSpace, m.wholeSpaces[0]
    native = getattr(ps, 'percept_store', None)
    parts = native.spell_out(text.encode()) if native is not None else list(text.encode())
    pair = cs.interpret_word(parts, ws.property_rows_for_bytes(text), key=text)
    return None if pair is None else pair[:2]


def test_22_new_word_replaces_one_inventory_row_and_writes_one_definition():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    before = alloc.next_id, len(alloc.layer()._tensor_rows), len(store)
    word, obj = admit(m, 'hello')
    assert word != obj
    assert (alloc.next_id, len(alloc.layer()._tensor_rows), len(store)) == (
        before[0] + 2, before[1] + 1, before[2] + 1)
    assert cs._csw_row_of(word) is None
    assert cs._csw_row_of(obj) is not None
    row = cs.definitions.row(word, obj)
    assert int(store.rel_type[row]) == store.REL_DEF
    assert store.refs[row, [0, 2]].tolist() == [word, obj]
    assert store.when[row].count_nonzero() == 0  # DEF is not a located sentence
    assert ('sym', word) not in cs.concept_parts(obj)
    assert all(not hasattr(cs, name) for name in ('bind_meta', 'meta_members'))
    assert all(not hasattr(alloc, name) for name in ('interpretations', 'word_obj_meta'))


def test_24_index_is_bidirectional_and_rebuilds_from_saved_rows(monkeypatch):
    from Layers import TernaryTruthStore
    m = model()
    word, obj = admit(m, 'hello')
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    index = cs.definitions
    saved = store.semantic_extras()
    restored = TernaryTruthStore(store.nDim, capacity=store.capacity, content_width=store.content_width)
    restored.load_state_dict(store.state_dict())
    restored.load_semantic_extras(saved)
    assert restored.definitions.word(form='hello') == word
    assert restored.definitions.objects(word) == (obj,)
    assert restored.definitions.words(obj) == (word,)
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
    with pytest.raises(ValueError, match='ambiguous|selection'):
        cs.interpret.forward(word)
    assert cs.interpret.forward(word, selected=second) == second
    assert set(cs.definitions.words(obj)) == {word, synonym}
    assert set(cs.definitions.objects(word)) == {obj, second}


def test_26_full_store_refuses_the_whole_word_transaction():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    while len(store) < store.capacity:
        store.append_idea(torch.zeros(store.nDim), sentence_index=len(store))
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(cs)
    before = alloc.next_id, dict(alloc.placement), dict(alloc.layer()._tensor_rows), len(store)
    try:
        result = admit(m, 'hello')
    except RuntimeError as error:
        assert 'capacity' in str(error) or 'full' in str(error)
    else:
        assert result is None
    assert before == (alloc.next_id, dict(alloc.placement), dict(alloc.layer()._tensor_rows), len(store))


def test_27_definition_has_a_fixed_when_and_refreshes_recency_without_append():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    first = admit(m, 'hello')
    r0 = cs.definitions.row(*first)
    when = store.when[r0].clone()
    m._advance_when_time()
    admit(m, 'there')
    count = len(store)
    assert int(store.recent(1)[0]) != r0
    m._advance_when_time()
    assert admit(m, 'hello') == first
    assert len(store) == count
    assert int(store.recent(1)[0]) == r0
    torch.testing.assert_close(store.when[r0], when, atol=0, rtol=0)
    chain = m.symbolSpace.ensure_sentence_expectation().get_stm_chain(n=count)
    assert len(chain) == count


def test_28_forgotten_definition_does_not_survive_in_any_lookup():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    word, obj = admit(m, 'hello')
    row = cs.definitions.row(word, obj)
    assert int(store.origin[row]) == store.ORIGIN_CONVERSATION
    store.clear_origin(store.ORIGIN_CONVERSATION)
    assert cs.definitions.word(form='hello') is None
    assert cs.definitions.objects(word) == ()
    assert cs.definitions.words(obj) == ()
    next_word, next_obj = admit(m, 'hello')
    assert next_word != word and next_obj != obj


@pytest.mark.usefixtures('eager_reading')
@pytest.mark.parametrize('configuration', ['MM_xor.xml', 'MM_grammar.xml', 'XOR_grammar.xml'])
def test_31_read_words_have_definition_rows_without_changing_the_configuration(configuration):
    from test_mm_xor import _fresh_model, _PROJECT
    path = Path(_PROJECT) / 'data' / configuration
    before = path.read_bytes()
    m = _fresh_model(str(path))[0]
    with torch.no_grad():
        m(m.inputSpace.prepInput(['hi no', 'hi go', 'we no', 'we go']))
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    for form in ('hi', 'no', 'we', 'go'):
        word = cs.definitions.word(form=form)
        assert word is not None
        obj, = cs.definitions.objects(word)
        assert int(store.rel_type[cs.definitions.row(word, obj)]) == store.REL_DEF
        assert cs._csw_row_of(obj) is not None
    assert path.read_bytes() == before


def test_34_learning_codes_cannot_change_definition_identity():
    m = model()
    cs, store = m._concept_owner(), m.symbolSpace.ltm_store
    word, obj = admit(m, 'hello')
    row = cs.definitions.row(word, obj)
    stored = store.slots[row].clone()
    with torch.no_grad():
        cs.similarity_codebook.getW().add_(3.)
    assert cs.definitions.objects(word) == (obj,)
    assert cs.definitions.words(obj) == (word,)
    assert cs.interpret.forward(word) == obj
    torch.testing.assert_close(store.slots[row], stored, atol=0, rtol=0)


def test_32_head_meta_checkpoint_migrates_to_definitions():
    from types import SimpleNamespace
    from Layers import TernaryTruthStore
    from test_cs_sparse_weights import _cs
    from test_structural_checkpoint import _model_with
    fixture = torch.load(Path(__file__).with_name('fixtures') / 'item7_legacy_definition.pt', weights_only=False)
    cs = _cs(nS=64, order=3)
    target = _model_with(cs, SimpleNamespace())
    target.symbolSpace = SimpleNamespace(ltm_store=TernaryTruthStore(cs.nDim, capacity=32))
    object.__setattr__(cs, '_model', target)
    target._restore_structural_extras(fixture['structural'])
    word, obj, meta = (fixture[key] for key in ('word','obj','meta'))
    assert cs.definitions.objects(word) == (obj,)
    assert cs.definitions.words(obj) == (word,)
    assert cs.definitions.word(form='legacy') == word
    assert meta in cs._concept_allocator.retired
    assert not hasattr(cs, 'meta_members')
    store = target.symbolSpace.ltm_store
    assert int(store.rel_type[cs.definitions.row(word, obj)]) == store.REL_DEF
