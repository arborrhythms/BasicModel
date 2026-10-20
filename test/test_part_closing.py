"""The kept close consumes a derivation, never a reconstructed STM identity."""
import pytest
import torch
from test_clause_acceptance import SentenceFixture


def test_relative_close_is_one_part_edge_and_rewitnesses_its_two_lanes(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'a', 'b'))
    before = len(f.cs._concept_allocator.parts)
    location = f.store.write_clause(clause, evidence=(.7, .2))
    assert location[0] == 'sym'
    assert len(f.store) == 0
    assert len(f.store.parts) == before + 1
    row = f.store.row(location)
    assert row['refs'].tolist() == [f.noun('a'), f.noun('b')]
    assert row['evidence'] == pytest.approx((.7, .2))
    assert f.store.write_clause(clause, evidence=(.4, .8)) == location
    assert f.store.row(location)['evidence'] == pytest.approx((.7, .8))


def test_word_composition_is_not_a_definition_without_an_object_argument(monkeypatch):
    f = SentenceFixture(monkeypatch)
    a, b = f.noun('a'), f.noun('b')
    word_a, word_b = f.words['a'][0], f.words['b'][0]
    f.store.parts.add(word_a, word_b, truth_plus=1.)
    assert f.cs.definitions.objects(word_b) == (b,)
    assert f.cs.definitions.words(a) == (word_a,)
    assert word_a not in f.cs.definitions.objects(word_b)


def test_absolute_close_keeps_only_the_concept_and_derivation_pairs(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', 'a', ('verb', 'v', 'b')))
    before = len(f.store.parts)
    row = f.store.write_clause(clause, evidence=(.9, 0.))
    assert isinstance(row, int)
    torch.testing.assert_close(f.store.slots[row, 0], clause.point)
    assert not f.store.slots[row, 1:].any()
    root = f.store.raw_meaning_of(row).role_refs[0]
    assert root[0] == 'sym' and f.store.parts.contains(root[1])
    assert f.store.refs[row].eq(-1).all()
    assert len(f.store.parts) == before + 2
    assert not any(key.startswith('_individual') for key, _ in f.store.meaning_of(row).bindings)


def test_complete_derivation_capacity_failure_has_no_prefix(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', 'a', ('verb', 'v', 'b')))
    alloc = f.cs._concept_allocator
    alloc.parts.capacity = alloc.next_id
    before = alloc.next_id, len(alloc.parts), len(f.store)
    with pytest.raises(OverflowError):
        f.store.write_clause(clause)
    assert (alloc.next_id, len(alloc.parts), len(f.store)) == before


def test_nested_terminal_mint_keeps_its_word_and_admitted_occupation(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    participant = f.noun('participant')
    word = f.cs.interpret.lookup_word([81], [], form='new-form')
    left = Composition(f.registry._payload(('sym', participant)), reference=participant)
    right = Composition(torch.eye(8)[3], word_id=word)
    value = torch.maximum(left.point, right.point)
    root = Composition(value, (left, right), operation='verb', formal_name='verb', head_role=1)
    clause = Clause(ConceptualMeaning.from_description(value), point=value, composition=root)
    row = f.store.write_clause(clause, evidence=(.8, .1))
    root_id = f.store.raw_meaning_of(row).role_refs[0][1]
    a, minted = f.store.parts.refs[root_id].tolist()
    assert a == participant and minted != word
    assert f.cs.definitions.words(minted) == (word,)
    assert f.store.parts.find(minted, word) is not None
    counts = f.language.parts_of_speech.counts[minted]
    assert counts['verb_argin2'] == 1
    # One existing owner serializes the new symbol's type as an occupation.
    from PartsOfSpeech import PartsOfSpeech
    restored = PartsOfSpeech(f.language.parts_of_speech.arguments)
    restored.set_extra_state(f.language.parts_of_speech.get_extra_state())
    assert restored.counts[minted] == counts


def test_bound_terminal_admits_its_new_word_without_reminting_the_object(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    identity = f.noun('existing-object')
    word = f.cs.interpret.lookup_word([82], [], form='new-form')
    point = f.registry._payload(('sym', identity))
    term = Composition(point, reference=identity, word_id=word)
    clause = Clause(ConceptualMeaning.from_description(point), point=point, composition=term)
    f.store.write_clause(clause, evidence=(.8, .1))
    assert f.cs.definitions.objects(word) == (identity,)
    assert f.store.parts.find(identity, word) is not None
    assert f.store._concept_point(identity) is not None


def test_self_reference_of_a_word_with_an_object_use_is_not_a_definition(monkeypatch):
    f = SentenceFixture(monkeypatch)
    obj = f.noun('a')
    word = f.words['a'][0]
    alloc = f.cs._concept_allocator
    # A name can itself be referred to as an object. Its self-reference
    # remains legal; it is not a definition of itself.
    alloc.symbol_types[word].add('object')
    edge = f.store.parts.add(word, word, truth_plus=1.)
    assert 'definition' not in f.store.parts.readings(edge, alloc.symbol_types.get)
    assert f.cs.definitions.objects(word) == (obj,)
    assert f.store.parts.find(word, word) == edge


def test_unseated_native_argument_does_not_acquire_the_working_value_at_close(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    participant = f.noun('participant')
    alloc = f.cs._concept_allocator
    word = alloc.new_concept()
    alloc.symbol_types[word] = {'word'}
    alloc.reference_orders[word] = 0
    assert f.cs._csw_row_of(word) is None
    left = Composition(f.registry._payload(('sym', participant)), reference=participant)
    right = Composition(torch.eye(8)[3], reference=word)
    point = torch.maximum(left.point, right.point)
    term = Composition(point, (left, right), operation='verb', formal_name='verb', head_role=1)
    clause = Clause(ConceptualMeaning.from_description(point), point=point, composition=term)
    f.store.write_clause(clause)
    assert f.store.parts.find(participant, word) is not None
    assert f.cs._csw_row_of(word) is None
    assert alloc.symbol_types[word] == {'word'}
