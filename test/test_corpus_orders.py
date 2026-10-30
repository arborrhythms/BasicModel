"""Item 2.7 mechanisms; native learning is measured separately."""
import pytest

from test_orders_learned import equality_fixture, event, equal_close


def named_individual(cs, store, alloc, name, offset):
    word = cs.interpret.lookup_word([offset], [offset + 1], form=name)
    first, second = event(cs, store, offset), event(cs, store, offset + 1)
    equal_close(cs, store, first, word)
    equal_close(cs, store, second, word)
    individual = cs.interpret.forward(word)
    assert individual != word and alloc.order_of(individual) == 1
    row = store.index_of_row(first)
    from ChainRead import point
    from LTM1 import OCCURRENCE
    cs.mtm.open(store.occurrence_of(row), kind=individual, native=individual,
                when=offset, value=point(store, (OCCURRENCE, first)))
    return individual


def symbol_equality(cs, store, left, right):
    from ClauseRow import Clause, Composition, predicate_identity
    from Meaning import ConceptualMeaning
    from ChainRead import point
    from LTM1 import SYMBOL
    operands = tuple(Composition(point(store, (SYMBOL, value)), reference=value,
        word_id=value, order=cs._concept_source_order(value)) for value in (left, right))
    composed = Composition(operands[0].point, operands, operation='equal', order=operands[0].order)
    return store.write_clause(Clause(ConceptualMeaning.from_description(composed.point),
        relation='equal', refs=(left, predicate_identity('equal'), right),
        composition=composed, order=composed.order), evidence=(1., 0.))


def test_two_witnessed_individuals_name_a_kind_without_forcing_any_order():
    from LTM1 import SYMBOL
    cs, store, alloc = equality_fixture()
    left = named_individual(cs, store, alloc, 'alpha', 1)
    other = named_individual(cs, store, alloc, 'beta', 3)
    word = cs.interpret.lookup_word([5], [6], form='category')
    pointer = alloc.singleton_values.data_ptr()
    symbol_equality(cs, store, left, word)
    assert alloc.order_of(word) == 0
    assert cs.interpret.forward(word) == word
    symbol_equality(cs, store, other, word)
    kind = cs.interpret.forward(word)
    assert kind != word and alloc.order_of(kind) == 2
    assert alloc.order_of(word) == 0
    assert set(alloc.parts.parts((SYMBOL, kind))) == {(SYMBOL, left), (SYMBOL, other)}
    assert alloc.singleton_values.data_ptr() == pointer


@pytest.mark.parametrize('member_first', (False, True))
def test_an_unbound_label_keeps_first_witnesses_of_different_orders_independent(member_first):
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word([5], [6], form='category')
    individual = named_individual(cs, store, alloc, 'alpha', 1)
    occurrence = event(cs, store, 5)
    before = len(alloc.raised)
    calls = (lambda: symbol_equality(cs, store, individual, word),
             lambda: equal_close(cs, store, occurrence, word))
    for close in calls if member_first else reversed(calls):
        close()
    assert len(alloc.raised) == before
    assert cs.interpret.forward(word) == word and alloc.order_of(word) == 0
    snapshots = [entry for entry in alloc.singleton_members.values()
                 if entry[0][1] in (individual, occurrence)]
    assert len(snapshots) == 2
    assert snapshots[0][2].data_ptr() != snapshots[1][2].data_ptr()
