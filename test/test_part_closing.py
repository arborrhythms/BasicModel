"""The kept close consumes a derivation, never a reconstructed STM identity."""
from order_fixtures import force_order
from chunk_fixtures import context as chunk_context
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


def test_absolute_close_keeps_one_idea_without_derivation_pairs(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', 'a', ('verb', 'v', 'b')))
    before = len(f.store.parts)
    row = f.store.write_clause(clause, evidence=(.9, 0.))
    assert isinstance(row, int)
    torch.testing.assert_close(f.store.slots[row], clause.point)
    root = f.store.raw_meaning_of(row).role_refs[0]
    assert root == ('sym', f.noun('a'))
    assert not hasattr(f.store, 'refs')
    assert len(f.store.parts) == before
    assert not any(key.startswith('_individual') for key, _ in f.store.meaning_of(row).bindings)


def test_complete_idea_capacity_failure_has_no_prefix(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', 'a', ('verb', 'v', 'b')))
    alloc = f.cs._concept_allocator
    f.store.count.fill_(f.store.capacity)
    before = alloc.next_id, len(alloc.parts), len(f.store)
    with pytest.raises(OverflowError):
        f.store.write_clause(clause)
    assert (alloc.next_id, len(alloc.parts), len(f.store)) == before


def test_sentence_projection_keeps_its_own_entry_after_an_embedded_mention(monkeypatch):
    from dataclasses import replace
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    # The same unresolved head appears first as a working mention and then
    # under a projecting wrapper. Only the sentence owns external testimony.
    point = torch.eye(8)[0]
    head = Composition(point)
    mention = Clause(ConceptualMeaning.from_description(point), point=point,
                     composition=head, reference_request={'mode':'bind'})
    root = Composition(point, (head,), operation='project', head_role=1)
    clause = replace(mention, composition=root, companions=(mention,), reference_request=None)
    written = {}
    row = f.store.write_clause(clause, document_key='entry', sentence_index=0,
                               evidence=(.7,.2), written_rows=written)
    embedded = written[id(mention)]
    assert isinstance(row,int) and isinstance(embedded,int) and row != embedded
    assert f.store.KINDS[int(f.store.record_kind[row])] == 'observation'
    assert f.store.KINDS[int(f.store.record_kind[embedded])] == 'unverified'
    assert not f.store.entry_when(embedded).any()
    assert f.store.row(row)['evidence'] == pytest.approx((.7,.2))


def test_nested_unknown_terminal_keeps_its_word_without_minting(monkeypatch):
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
    from LTM1 import SYMBOL
    assert row[0] == 'sym'
    assert f.store.parts.references(row[1]) == ((SYMBOL, participant), (SYMBOL, word))
    assert f.store.parts.rows_referencing(SYMBOL, word) == (row[1],)
    assert not f.cs.definitions.objects(word)
    assert not f.language.parts_of_speech.counts


def test_bound_terminal_keeps_its_object_without_defining_a_new_word(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    identity = f.noun('existing-object')
    word = f.cs.interpret.lookup_word([82], [], form='new-form')
    point = f.registry._payload(('sym', identity))
    term = Composition(point, reference=identity, word_id=word)
    clause = Clause(ConceptualMeaning.from_description(point), point=point, composition=term)
    f.store.write_clause(clause, evidence=(.8, .1))
    assert f.cs.definitions.objects(word) == ()
    assert f.store.parts.find(identity, word) is None
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
    assert 'definition' not in f.store.parts.readings(edge, lambda ref: alloc.symbol_types.get(ref[1], ()))
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
    force_order(f.cs, word, 0)
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


def test_new_referent_closes_as_an_unminted_occurrence(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    marker = f.noun('marker')
    kind = f.noun('description')
    left = Composition(f.registry._payload(('sym', marker)), reference=marker)
    description = Composition(f.registry._payload(('sym', kind)), reference=kind)
    root = Composition(description.point, (left, description), operation='lower',
                       formal_name='lower', head_role=2, order=1)
    clause = Clause(ConceptualMeaning.from_description(root.point), point=root.point,
                    order=1, composition=root, minted=True,
                    reference_request={'mode': 'mint'})
    before = f.cs._concept_allocator.next_id, len(f.store.parts)
    bank = f.cs.similarity_codebook.W.detach().clone()
    row = f.store.write_clause(clause)
    assert isinstance(row, int) and int(f.store.order[row]) == 1
    torch.testing.assert_close(f.store.slots[row], root.point)
    assert f.store.raw_meaning_of(row).role_refs[0] == ('sym', kind)
    assert f.store.location(f.store.occurrence_of(row)) is not None
    # The description is the new referent, not a relation argument. Its
    # introduction consumes neither a seat nor an unseated Part handle.
    assert (f.cs._concept_allocator.next_id, len(f.store.parts)) == before
    torch.testing.assert_close(f.cs.similarity_codebook.W, bank, rtol=0, atol=0)
    counts = f.language.parts_of_speech.counts
    assert not counts  # only observe_kept(reading) updates occupations


def test_lower_word_alias_does_not_double_count_kept_occupations(monkeypatch):
    from types import SimpleNamespace
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from PartsOfSpeech import operation_name
    f = SentenceFixture(monkeypatch)
    words = [f.cs.interpret.lookup_word([91+i], [], form=f'leaf-{i}') for i in range(3)]
    leaves = [Composition(f.registry._payload(('sym', word)), reference=word,
                          word_id=word) for word in words]
    lower = Composition(leaves[1].point, tuple(leaves[:2]), operation='lower',
                        formal_name='lower', head_role=2, order=1)
    root = Composition(torch.maximum(lower.point, leaves[2].point), (lower, leaves[2]),
                       operation='intersection', formal_name='intersection')
    operations = {operation_name(rule): i for i, rule in enumerate(f.binary)}
    program = SimpleNamespace(word_ids=torch.tensor(words), actions=torch.tensor([
        (0, -1, 0), (0, -1, 1), (1, operations['lower'], 0),
        (0, -1, 2), (1, operations['intersection'], 0)]))
    parts = f.language.parts_of_speech
    parts.observe_kept(program, f.binary, f.unary)
    expected = {word: dict(parts.counts[word]) for word in words}
    clause = Clause(ConceptualMeaning.from_description(root.point), point=root.point,
                    composition=root)
    row = f.store.write_clause(clause)
    assert {word: dict(parts.counts[word]) for word in words} == expected
    assert row[0] == 'sym'
    assert row[1] not in parts.counts
    assert parts.counts[words[1]]['lower_argout1', parts.order_of(words[1])] == 1
    assert parts.counts[words[1]]['intersection_argin1', parts.order_of(words[1])] == 1


def test_same_description_tokens_keep_distinct_occurrence_identities(monkeypatch):
    from dataclasses import replace
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from ReferenceContext import SituationFrame
    f = SentenceFixture(monkeypatch)
    marker, kind = f.noun('marker'), f.noun('description')
    left = Composition(f.registry._payload(('sym', marker)), reference=marker)
    head = Composition(f.registry._payload(('sym', kind)), reference=kind)
    term = Composition(head.point, (left, head), operation='lower', formal_name='lower',
                       head_role=2, order=1)
    clause = Clause(ConceptualMeaning.from_description(head.point), point=head.point,
                    order=1, minted=True, composition=term)
    alloc = f.cs._concept_allocator
    before = alloc.next_id, len(f.store.parts)
    bank = f.cs.similarity_codebook.W.detach().clone()
    frames, addresses = [], []
    for sentence in range(2):
        written = {}
        row = f.store.write_clause(clause, document_key='tokens', sentence_index=sentence,
                                   written_rows=written)
        addresses.append(f.store.occurrence_of(row))
        assert int(f.store.order[row]) == 1
        assert f.store.raw_meaning_of(row).role_refs[0] == ('sym', kind)
        assert written[('symbols', id(clause))] == ()
        assert written[('activations', id(clause))] == ((addresses[-1], None, 0),)
        frames.append(replace(SituationFrame.from_row(f.store.row(row)),
            symbol_ids=written[('symbols', id(clause))],
            activations=written[('activations', id(clause))]))
    resolver = f.cs.components.reference_close
    from chunk_fixtures import recall
    identities = resolver.individuals(resolver.accessible(recall(resolver, frames)))
    assert len(set(addresses)) == 2 and set(identities) == set(addresses)
    assert (alloc.next_id, len(f.store.parts)) == before
    torch.testing.assert_close(f.cs.similarity_codebook.W, bank, rtol=0, atol=0)


def test_pronoun_at_occurrence_closes_verb_over_antecedent_kind_without_seating(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    from ReferenceContext import SituationFrame
    f = SentenceFixture(monkeypatch)
    kind, verb = f.noun('description'), f.noun('predicate')
    point, predicate = (f.registry._payload(('sym', identity)) for identity in (kind, verb))
    antecedent = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                        composition=Composition(point, reference=kind, order=1))
    first = f.store.write_clause(antecedent, document_key='pronoun', sentence_index=0)
    address = f.store.occurrence_of(first)
    request = dict(mode='pronoun', leaves=(0,), cue=torch.zeros_like(point),
                   predicate=predicate, role=0)
    term = Composition(point, order=1)
    np = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                composition=term, reference_request=request)
    value = torch.maximum(point, predicate)
    clause = Clause(ConceptualMeaning.from_description(value), point=value, children=(np,),
        composition=Composition(value, (term, Composition(predicate, reference=verb)),
                                operation='lift', formal_name='lift', head_role=2))
    resolver = f.cs.components.reference_close
    from chunk_fixtures import recall
    plan = resolver.preview(clause, recall(resolver, (SituationFrame.from_row(f.store.row(first)),)))
    assert plan['items'][0]['selected'] == address
    kept = resolver.prepare(clause, plan)
    bank = f.cs.similarity_codebook.W.detach().clone()
    seats = dict(f.cs._concept_allocator.layer()._tensor_rows)
    written = {}
    row = f.store.write_clause(kept, document_key='pronoun', sentence_index=1,
                               written_rows=written)
    resolver.commit(plan, row, written, document='pronoun', position=1, training=False)
    assert isinstance(row, int)
    torch.testing.assert_close(f.store.slots[row], value)
    assert dict(f.store.raw_meaning_of(row).bindings)['_identity'] == address
    assert resolver.last_audit[0]['reference'] == address
    assert dict(f.cs._concept_allocator.layer()._tensor_rows) == seats
    torch.testing.assert_close(f.cs.similarity_codebook.W, bank, rtol=0, atol=0)


def test_inline_lower_retains_its_kept_journal_value(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    marker, kind = f.noun('marker'), f.noun('description')
    left = Composition(f.registry._payload(('sym', marker)), reference=marker)
    head = Composition(f.registry._payload(('sym', kind)), reference=kind)
    changed = torch.maximum(left.point, head.point)
    term = Composition(changed, (left, head), operation='lower', formal_name='lower',
                       head_role=2, order=1)
    clause = Clause(ConceptualMeaning.from_description(changed), point=changed, composition=term)
    alloc = f.cs._concept_allocator
    before = alloc.next_id, len(f.store.parts), len(f.store)
    row = f.store.write_clause(clause)
    torch.testing.assert_close(f.store.slots[row], changed)
    assert (alloc.next_id, len(f.store.parts)) == before[:2]


def test_new_relation_argument_needs_an_occurrence_and_no_concept_seat(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    native = f.noun('participant')
    left = Composition(f.registry._payload(('sym', native)), reference=native)
    argument = Composition(torch.eye(8)[3], order=1)
    root = Composition(torch.maximum(left.point, argument.point), (left, argument),
                       operation='verb', formal_name='verb', head_role=1)
    clause = Clause(ConceptualMeaning.from_description(root.point), point=root.point,
                    composition=root)
    # The occurrence value needs no order-1 concept seat.
    caps = f.cs._order_caps()
    monkeypatch.setattr(f.cs, '_order_caps', lambda: [caps[0], 0, f.cs.nVectors-caps[0]])
    seats = set(f.cs._concept_allocator.layer()._tensor_row_keys)
    row = f.store.write_clause(clause)
    from LTM1 import OCCURRENCE
    tag, address = f.store.parts.references(row[1])[1]
    assert tag == OCCURRENCE
    torch.testing.assert_close(f.store.point_of_row(address), argument.point)
    assert set(f.cs._concept_allocator.layer()._tensor_row_keys) == seats


def test_relation_argument_does_not_use_the_concept_tail(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    native = f.noun('participant')
    left = Composition(f.registry._payload(('sym', native)), reference=native)
    argument = Composition(torch.eye(8)[3], order=1)
    root = Composition(torch.maximum(left.point, argument.point), (left, argument),
                       operation='verb', formal_name='verb', head_role=1)
    clause = Clause(ConceptualMeaning.from_description(root.point), point=root.point,
                    composition=root)
    capacity = f.cs.nVectors
    shape = f.cs.similarity_codebook.W.shape
    caps = (f.cs._order_caps()[0], 0)
    monkeypatch.setattr(f.cs, '_order_caps', lambda: caps)
    row = f.store.write_clause(clause)
    from LTM1 import OCCURRENCE
    tag, address = f.store.parts.references(row[1])[1]
    assert tag == OCCURRENCE and f.store.index_of_row(address) is not None
    assert f.cs._csw_row_of(address) is None
    assert f.cs.nVectors == capacity
    assert f.cs.similarity_codebook.W.shape == shape


def test_surface_phrase_keeps_the_unknown_head_word_without_minting(monkeypatch):
    from ClauseRow import Clause, Composition
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    native = f.noun('participant')
    marker = Composition(f.registry._payload(('sym', native)), reference=native)
    word = f.cs.interpret.lookup_word([84], [], form='new-head')
    description = Composition(torch.eye(8)[3], word_id=word)
    phrase = Composition(description.point, (marker, description), operation='surface',
                         formal_name='surface', head_role=2)
    root = Composition(torch.maximum(phrase.point, marker.point), (phrase, marker),
                       operation='verb', formal_name='verb', head_role=1)
    clause = Clause(ConceptualMeaning.from_description(root.point), point=root.point,
                    composition=root)
    row = f.store.write_clause(clause)
    from LTM1 import SYMBOL
    assert f.store.parts.references(row[1])[0] == (SYMBOL, word)
    assert not f.cs.definitions.objects(word)
    assert f.store.parts.find(word, native) == row[1]
