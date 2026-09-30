"""Word/object binding and taxonomy belong to the native concept inventory."""
import pytest
from Spaces import WholeSpace, _concept_alloc_of
from test_item9b_interpret import _operator


def test_word_spelling_is_ordered_and_binding_is_idempotent():
    cs, interpret = _operator()
    word = interpret.lookup_word([10, 11, 12], [], form='abc')
    obj = interpret.forward(word)
    assert interpret.lookup_word([10, 11, 12], [], form='abc') == word
    assert interpret.forward(word) == obj
    assert set(cs.concept_parts(word)) == {10, 11, 12}
    assert cs.definitions.deref(word) == obj
    assert interpret.lookup_word([12, 11, 10], [], form='cba') != word




def test_two_words_do_not_create_a_sentence_concept():
    cs, interpret = _operator()
    first = interpret.lookup_word([10, 11], [], form='ab')
    second = interpret.lookup_word([20, 21], [], form='cd')
    objects = interpret.forward(first), interpret.forward(second)
    assert objects[0] != objects[1]
    assert not hasattr(cs, '_joint_concepts')
    assert not hasattr(cs, 'conceptualize_chain')
    assert not hasattr(_concept_alloc_of(cs), 'chain_idx')


def test_taxonomy_and_word_resolution_are_native_not_wholespace_forwarders(monkeypatch):
    cs, interpret = _operator()
    word = interpret.lookup_word([10, 11], [], form='ab')
    obj = interpret.forward(word)
    monkeypatch.setattr(cs, 'taxonomy_children', lambda *a: pytest.fail('taxonomy walk during translation'))
    assert cs.definitions.deref(word) == obj
    assert not hasattr(WholeSpace, 'taxonomy_children')
    assert not hasattr(WholeSpace, 'insert_meta')
