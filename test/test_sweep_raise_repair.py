"""Equality testimony survives an ineligible class-member order."""
import pytest
import torch

from test_orders_learned import equality_fixture, event, equal_close


@pytest.mark.parametrize('mismatch_first', (False, True))
def test_unlike_order_equality_keeps_evidence_without_a_partial_raise(mismatch_first):
    from LTM1 import OCCURRENCE, SYMBOL
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word([1], [2], form='label')
    first = event(cs, store, 1)
    other = event(cs, store, 2, order=1)
    second = event(cs, store, 3)
    pointer = alloc.singleton_values.data_ptr()
    sequence = (other, first) if mismatch_first else (first, other)
    for address in sequence:
        equal_close(cs, store, address, word, evidence=(1., 1.))
    edge = alloc.parts.find((OCCURRENCE, other), (SYMBOL, word))
    assert edge is not None
    assert float(alloc.parts.truth_plus[edge]) == float(alloc.parts.truth_minus[edge]) == 1.
    assert not alloc.raised
    assert alloc.singleton_members[word][0] == (OCCURRENCE, first)
    assert cs.interpret.forward(word) == word
    equal_close(cs, store, second, word)
    parent = cs.interpret.forward(word)
    assert parent != word and alloc.order_of(parent) == 1
    assert set(alloc.parts.parts((SYMBOL, parent))) == {(OCCURRENCE, first), (OCCURRENCE, second)}
    assert alloc.singleton_values.data_ptr() == pointer
    assert cs._concept_source_order(other) == 1
