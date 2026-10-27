"""Item 3.0 step 5: witnessed orders, equality classes and occupations."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from ReferenceContext import denoted_order, OrderReference
from order_fixtures import force_order


def equality_fixture():
    from test_cs_sparse_weights import _cs
    from Spaces import _concept_alloc_of
    cs = _cs(nS=128, order=3, dim=16)
    return cs, cs._fixture_occurrences, _concept_alloc_of(cs)


def event(cs, store, place, *, order=0, value=.5, token=False):
    from Meaning import ConceptualMeaning
    point = torch.zeros(16)
    point[0] = value
    point[store.when_start-4] = place
    row = store.append_meaning(ConceptualMeaning.from_description(point), order=order,
                              document_key=('event', place), when=False)
    address = int(store.address_keys[row])
    if token:
        cs.mtm.open(store.occurrence_of(row), kind=17, when=place, value=point,
                    pending=False, where=point[store.when_start-4:store.when_start])
    return address


def equal_close(cs, store, left, right, *, word=None, evidence=(1., 0.)):
    from ClauseRow import Clause, Composition, predicate_identity
    from Meaning import ConceptualMeaning
    from LTM1 import OCCURRENCE, SYMBOL
    from ChainRead import point
    operands = (Composition(point(store, (OCCURRENCE, left)), reference=left,
                            order=cs._concept_source_order(left)),
                Composition(point(store, (SYMBOL, right)), reference=right,
                            word_id=right if word is None else word, order=cs._concept_source_order(right)))
    composed = Composition(operands[0].point, operands, operation='equal', order=operands[0].order)
    return store.write_clause(Clause(ConceptualMeaning.from_description(composed.point),
        relation='equal', refs=(left, predicate_identity('equal'), right),
        composition=composed, order=composed.order), evidence=evidence)


def test_two_event_witnesses_raise_the_named_object_without_work(tmp_path, monkeypatch):
    from QueryWork import QueryWorkBudget
    from LTM1 import OCCURRENCE, SYMBOL
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word([1], [2], form='label')
    first, second = event(cs, store, 1), event(cs, store, 2)
    monkeypatch.setattr(QueryWorkBudget, 'consume', lambda *a, **k: pytest.fail('equality raise charged work'))
    pointer = alloc.singleton_values.data_ptr()
    equal_close(cs, store, first, word)
    assert cs.interpret.forward(word) == word and alloc.order_of(word) == 0
    equal_close(cs, store, second, word)
    parent = cs.interpret.forward(word)
    assert parent != word and alloc.reference_orders[parent] == alloc.order_of(parent) == 1
    assert set(alloc.parts.parts((SYMBOL, parent))) == {(OCCURRENCE, first), (OCCURRENCE, second)}
    assert alloc.singleton_values.data_ptr() == pointer
    third = event(cs, store, 3)
    equal_close(cs, store, third, parent, word=word)
    assert cs.interpret.forward(word) == parent and alloc.order_of(parent) == 1
    assert set(alloc.parts.parts((SYMBOL, parent))) == {(OCCURRENCE, first), (OCCURRENCE, second), (OCCURRENCE, third)}
    (tmp_path/'certificate.json').write_text(json.dumps(dict(parent=parent, order=alloc.order_of(parent),
        members=alloc.parts.parts((SYMBOL, parent)), work=0, fixed_snapshot_slab=True), indent=2)+'\n')


@pytest.mark.parametrize('same_token,same_value,raises', [(False, False, True), (False, True, False), (True, False, False)])
def test_only_distinct_published_identities_raise(same_token, same_value, raises):
    from LTM1 import OCCURRENCE, SYMBOL
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word([1], [2], form='category')
    force_order(cs, word, 1)
    first = event(cs, store, 1, order=1, value=.25, token=True)
    second = event(cs, store, 2, order=1, value=.25 if same_value else .75, token=not same_token)
    # Distinct occurrence addresses need distinct source keys even when their
    # published values agree. The second event has a separate native entry.
    if same_value:
        first_key = store.occurrence_of(store.index_of_row(first))
        second_key = store.occurrence_of(store.index_of_row(second))
        cs.mtm.entries[second_key] = cs.mtm.entries[second_key].with_actual(value=cs.mtm.entries[first_key].value)
    if same_token:
        cs.mtm.bind(store.occurrence_of(store.index_of_row(first)), store.occurrence_of(store.index_of_row(second)),
                    kind=17, when=2)
    equal_close(cs, store, first, word)
    equal_close(cs, store, second, word)
    result = cs.interpret.forward(word)
    assert (result != word) == raises
    if raises:
        assert alloc.reference_orders[result] == 2
        assert set(alloc.parts.parts((SYMBOL, result))) == {(OCCURRENCE, first), (OCCURRENCE, second)}


def test_one_identity_equality_and_weak_testimony_do_not_raise(monkeypatch):
    import util
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word([1], [2], form='label')
    first, second = event(cs, store, 1), event(cs, store, 2)
    equal_close(cs, store, first, word)
    assert not alloc.raised
    monkeypatch.setitem(util.TheXMLConfig.data.setdefault('ConceptualSpace', {}), 'raiseEvidence', .5)
    equal_close(cs, store, first, word)
    assert not alloc.raised
    equal_close(cs, store, second, word, evidence=(.5, 0.))
    assert not alloc.raised


@pytest.mark.parametrize('name,orders,expected', [
    ('lift', (1, 0), 0), ('lift', (2, 0), 1), ('lift', (0, 0), 0),
    ('lift', (-1, 0), -1), ('verb', (2, 0), 2),
    ('lower', (0, 2), 1), ('lower', (0, 1), 1), ('lower', (0, 0), 0),
    ('lower', (0, -1), -1), ('equal', (1, 3), 1),
    ('part', (0, 2), 2), ('whole', (2, 0), 2), ('bind', (0, 2), 2),
    ('sum', (1, 2), 2), ('sum', (1, -1), -1), ('not', (2,), 2),
])
def test_denoted_lookup(name, orders, expected):
    rule = NS(method_name=name)
    assert denoted_order(rule, orders, (), None) == expected
    tensors = tuple(torch.tensor([value, value]) for value in orders)
    assert denoted_order(rule, tensors, (), None).tolist() == [expected, expected]


def test_modal_predication_keeps_the_subject_order_without_a_floor_one():
    rule = NS(method_name='lift')
    subjects = torch.tensor([0, 1, 2, -1])
    refs = (OrderReference(None), OrderReference(None, torch.ones(4, dtype=torch.bool)))
    assert torch.equal(denoted_order(rule, (subjects, torch.zeros_like(subjects)), refs, None), subjects)


def test_retired_order_stamps_and_formation_are_absent():
    import ReferenceContext
    from Language import Grammar, GRAMMAR_LAYER_CLASSES, SymbolSubSpace
    assert all(not hasattr(ReferenceContext, name) for name in ('shifted_order', 'operation_order', 'result_order'))
    assert 'order_delta' not in Grammar.RuleDef._fields
    assert 'order_delta' not in Grammar.RuleOrderSignature._fields
    assert all(not hasattr(layer, 'order_delta') for layer in GRAMMAR_LAYER_CLASSES.values())
    assert not hasattr(SymbolSubSpace, 'conceptualize')


def test_unwitnessed_order_and_inconsistent_record():
    from Layers import ConceptAllocator
    alloc = ConceptAllocator()
    atom, parent = alloc.new_concept(), alloc.new_concept()
    assert alloc.order_of(atom) == -1
    alloc.add(atom, 'part', 17)
    assert alloc.order_of(atom) == 0
    alloc.add(parent, 'part', ('sym', atom))
    assert alloc.order_of(parent) == 1
    alloc.reference_orders[parent] = 2
    with pytest.raises(ValueError, match='disagrees'):
        alloc.settle(parent)


def test_row_request_cannot_write_an_order():
    from test_cs_sparse_weights import _cs
    from Spaces import _concept_alloc_of
    cs = _cs(nS=32, order=3)
    cid = cs.new_concept()
    alloc = _concept_alloc_of(cs)
    with pytest.raises(ValueError, match='foreign order'):
        cs._csw_concept_row(0, cid)
    assert cid not in alloc.reference_orders
    cs.add_part(cid, 1)
    with pytest.raises(ValueError, match='foreign order'):
        cs._csw_concept_row(1, cid)
    assert cs._csw_concept_row(0, cid) is not None


def test_occupations_follow_current_order_and_checkpoint_version(tmp_path):
    from test_cs_sparse_weights import _cs
    from PartsOfSpeech import PartsOfSpeech
    from IndependentComponents import _pack_state, _unpack_state
    cs = _cs(nS=32, order=3)
    word = cs.new_concept()
    parts = PartsOfSpeech(('select_argin1', 'modify_argin1'), cs)
    assert parts.order_of(word) == -1
    parts.add(word, 'select_argin1', -1)
    force_order(cs, word, 1)
    # The same kept position is observed at two witnessed orders; an
    # additional kept modifier makes the two distributions distinguishable.
    def kept(name):
        rule = NS(method_name=name, surface_name=name, rhs_symbols=('x',), lhs='y', head_role=0)
        program = NS(word_ids=torch.tensor([word]), actions=torch.tensor([[0,-1,0],[2,0,0]]))
        parts.observe_kept(program, (), (rule,))
    kept('select')
    first = parts.probabilities(torch.tensor([word]), torch.zeros(1))
    force_order(cs, word, 2)
    kept('select')
    kept('modify')
    kept('modify')
    second = parts.probabilities(torch.tensor([word]), torch.zeros(1))
    assert first.tolist() == [[1., 0.]]
    torch.testing.assert_close(second, torch.tensor([[1/3, 2/3]]))
    assert parts.counts[word]['select_argin1', 1] == parts.counts[word]['select_argin1', 2] == 1
    assert parts.labels[word] == 'modify_argin1'
    restored = PartsOfSpeech(parts.arguments, cs)
    restored.set_extra_state(parts.get_extra_state())
    assert restored.counts == parts.counts and restored.distribution(word) == parts.distribution(word)
    old = dict(_unpack_state(parts.get_extra_state()), version=1)
    with pytest.raises(ValueError, match='schema'):
        restored.set_extra_state(_pack_state(old))
    (tmp_path/'certificate.json').write_text(json.dumps(dict(first=first.tolist(), second=second.tolist(),
        counts=[(argument, order, count) for (argument, order), count in parts.counts[word].items()]), indent=2)+'\n')


def test_relation_handles_read_their_tagged_reference_orders():
    from LTM1 import SYMBOL, OCCURRENCE, RELATION
    cs, store, alloc = equality_fixture()
    member, whole, equal_side = (cs.new_concept() for _ in range(3))
    force_order(cs, member, 1)
    force_order(cs, whole, 2)
    force_order(cs, equal_side, 3)
    taxonomy = alloc.parts.add((SYMBOL, member), (SYMBOL, whole))
    alloc.singletons.add(equal_side)
    equality = alloc.parts.add((SYMBOL, member), (SYMBOL, equal_side))
    subject = event(cs, store, 1, order=2)
    predication = alloc.parts.add((OCCURRENCE, subject), (SYMBOL, whole))
    nested = alloc.parts.add((OCCURRENCE, subject), (RELATION, predication))
    assert [cs._concept_source_order(row) for row in (taxonomy, equality, predication, nested)] == [2, 1, 1, 1]
    # The resulting class is not a unit equality side merely because the
    # allocator exempts its first witnessed member from minimum support.
    alloc.singletons.add(whole)
    alloc.raised.add(whole)
    assert cs._concept_source_order(taxonomy) == 2


def test_first_equality_witness_slab_is_bounded_and_restorable():
    from Layers import ConceptAllocator
    default = ConceptAllocator(member_width=8)
    assert default.singleton_values.shape == (default.parts.capacity, 8)
    alloc = ConceptAllocator(member_capacity=1, member_width=8)
    point = torch.arange(8.)
    alloc.remember_singleton(2, (1, 10), None, point)
    pointer = alloc.singleton_values.data_ptr()
    point.zero_()
    assert alloc.singleton_members[2][2].sum() == 28
    with pytest.raises(OverflowError, match='constructed relation capacity'):
        alloc.remember_singleton(3, (1, 11), None, point)
    assert alloc.singleton_values.data_ptr() == pointer
    restored = ConceptAllocator(member_capacity=1, member_width=8)
    for right, (member, token, value) in alloc.singleton_members.items():
        restored.remember_singleton(right, member, token, value)
    torch.testing.assert_close(restored.singleton_values, alloc.singleton_values, rtol=0, atol=0)


def test_forgotten_first_equality_renews_its_existing_snapshot_seat():
    from LTM1 import OCCURRENCE, SYMBOL
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word([1], [2], form='label')
    first, second, third = (event(cs, store, place) for place in (1, 2, 3))
    equal_close(cs, store, first, word)
    pointer = alloc.singleton_members[word][2].data_ptr()
    alloc.parts.remove((OCCURRENCE, first), (SYMBOL, word))
    equal_close(cs, store, second, word)
    assert not alloc.raised
    assert alloc.singleton_members[word][0] == (OCCURRENCE, second)
    assert alloc.singleton_members[word][2].data_ptr() == pointer
    equal_close(cs, store, third, word)
    parent = cs.interpret.forward(word)
    assert set(alloc.parts.parts((SYMBOL, parent))) == {(OCCURRENCE, second), (OCCURRENCE, third)}


def test_retirement_handles_class_base_cache_without_losing_another_class():
    cs, store, alloc = equality_fixture()
    word = cs.interpret.lookup_word([1], [2], form='label')
    first, second = event(cs, store, 1), event(cs, store, 2)
    equal_close(cs, store, first, word)
    equal_close(cs, store, second, word)
    parent = cs.interpret.forward(word)
    base = alloc.relate_idx['part-base', parent]
    cs.retire_concept(cs.new_concept())
    assert alloc.relate_idx['part-base', parent] == base
    cs.retire_concept(parent)
    assert ('part-base', parent) not in alloc.relate_idx
    assert ('part-kind', base, 1) not in alloc.relate_idx


@pytest.mark.parametrize('order', (0, 1, -1))
def test_determiner_opens_a_chunk_identity_only_at_member_order_one(order):
    import STMChunks as chunks
    from chunk_fixtures import Walk, choice
    rule = NS(method_name='lower', head_role=2, determiner_mode='mint')
    walk = Walk((rule,), ())
    point = torch.tensor([[.2, .3, .4]])
    state = walk.empty(point, 4)
    state = walk.shift(state, point, word=101, index=0, order=0)
    state = walk.shift(state, point, word=102, index=1, order=order)
    state = walk.apply(state, choice(point, local=0, order=order, head=2),
                       torch.stack((point, point), 1), 0)
    top = chunks.at(state[0])
    identity = chunks.gather(state[0][chunks.IDENTITY], top[:,None])[:,0]
    assert bool((identity >= 0).all()) == (order == 1)
    # The last journal column is the publication-kind code.
    assert (int(state[2][0,0,-1]) == 3) == (order == 1)


@pytest.mark.parametrize('subject_order,closed_order', [(0,0),(1,0),(2,1)])
def test_kept_clause_and_durable_row_read_the_predication_order(monkeypatch, subject_order, closed_order):
    from dataclasses import replace
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import finish_reading
    f = SentenceFixture(monkeypatch)
    entry = f.program(('lift', 'subject', 'predicate'))
    force_order(f.cs, int(entry.concept_ids[0]), subject_order)
    entry = replace(entry, leaf_orders=torch.tensor([subject_order,0]))
    clause = finish_reading(f.language, entry, registry=f.registry)
    assert clause.order == closed_order
    row = f.store.write_clause(clause)
    assert int(f.store.order[row]) == closed_order
