"""Source-addressed recurrence, pool lifetime and retained identity promotion."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ClauseRow import Clause, Composition
from Meaning import ConceptualMeaning
from MemoryConsolidation import consolidate
from MediumTermMemory import MediumTermMemory
from ReferenceContext import SituationFrame
from test_reference_close import fixture
from test_medium_term_memory import mention, snapshot, unchanged


def kept(resolver, clause, frames=(), *, document='consolidation', position=0, stream=0):
    plan = resolver.preview(clause, frames, stream=stream)
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = resolver.store.write_clause(prepared, document_key=document,
        sentence_index=position, stream=stream, written_rows=written)
    resolver.commit(plan, row, written, document=document, position=position, training=False)
    return row, plan


def native(owner, point):
    symbol = owner.new_concept()
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_concept_row(0, symbol)].copy_(point)
    return symbol


@pytest.mark.parametrize('distinct', (False, True))
def test_reading_recurrence_counts_distinct_documents_not_reobservations(distinct):
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    left, right = (native(owner, atoms[n]) for n in (0, 1))
    value = atoms[0] + atoms[1]
    tree = Composition(value, (Composition(atoms[0], reference=left),
        Composition(atoms[1], reference=right)), operation='sum')
    clause = Clause(ConceptualMeaning.from_description(value), point=value, composition=tree)
    for n in range(3):
        row = store.write_clause(clause, document_key=('source', n if distinct else 0), sentence_index=n)
        reading = store.raw_meaning_of(row).role_refs[0][1]
        before = store.slots.clone(), store.parts.refs.clone()
        consolidate(owner, store)
        if n < 2 or not distinct:
            assert reading not in owner._recurring_readings
        torch.testing.assert_close(store.slots, before[0], rtol=0, atol=0)
        torch.testing.assert_close(store.parts.refs, before[1], rtol=0, atol=0)
    if distinct:
        symbol = owner._recurring_readings[reading]
        assert symbol != reading
        torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(symbol)], value)
        before = snapshot(resolver)
        consolidate(owner, store)
        unchanged(resolver, before)
        saved = owner.vocab_extras()
        owner._recurring_readings.clear()
        owner.load_vocab_extras(saved)
        assert owner._recurring_readings[reading] == symbol


def test_token_promotes_at_three_sentences_with_the_occurrence_fold_and_keeps_its_state():
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    kind = native(owner, atoms[0])
    first, _ = kept(resolver, mention(atoms[0], reference=kind))
    token = store.occurrence_of(first)
    frame = SituationFrame.from_row(store.row(first))
    second, _ = kept(resolver, mention(atoms[0]+atoms[1], mode='bind'), (frame,), position=1)
    assert pool.entries[token].symbol is None
    # Reclosing the same source sentence increments accesses, not recurrence.
    kept(resolver, mention(atoms[0]+atoms[1], mode='bind'), (frame,), position=1)
    assert pool.entries[token].symbol is None
    third, _ = kept(resolver, mention(atoms[0]+atoms[2], mode='bind'), (frame,), position=2)
    entry = pool.entries[token]
    assert entry.occurrences == (token, store.occurrence_of(second), store.occurrence_of(third))
    assert entry.kind == kind and entry.symbol is not None and entry.count == 4
    expected = atoms[0] + atoms[1] + atoms[2]
    torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(entry.symbol)], expected)
    assert resolver._derivation_symbol(token) == entry.symbol
    pool.update(token, atoms[3], when=50)
    consolidate(owner, store)
    torch.testing.assert_close(pool.value_of(token), atoms[3])
    torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(entry.symbol)], expected)
    restored = MediumTermMemory()
    restored.restore(pool.state())
    assert restored.entries[token].symbol == entry.symbol and restored.names == pool.names
    assert restored.closes == pool.closes and restored.accesses == pool.accesses


def test_many_mentions_in_one_sentence_do_not_promote():
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    rows = [store.write_clause(mention(atoms[n]), document_key='one', sentence_index=0)
            for n in range(3)]
    token = store.occurrence_of(rows[0])
    pool.open(token, kind=None, value=atoms[0], when=1)
    for row in rows[1:]:
        pool.bind(token, store.occurrence_of(row), kind=None, value=store.slots[row, 0], when=2)
    before = snapshot(resolver)
    consolidate(owner, store)
    assert pool.entries[token].symbol is None
    unchanged(resolver, before)


def test_expiry_counts_kept_closes_per_stream_and_document_end_keeps_rows():
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    row, _ = kept(resolver, mention(atoms[0]), document='first', stream=0)
    token = store.occurrence_of(row)
    before = snapshot(resolver)
    for _ in range(20):
        pool.close(store, stream=1)
    assert token in pool.entries
    for _ in range(16):
        pool.close(store, stream=0)
    assert token in pool.entries
    pool.close(store, stream=0)
    assert token not in pool.entries and pool.token_of(token) is None
    unchanged(resolver, before)
    row, _ = kept(resolver, mention(atoms[1]), document='second', position=1)
    token = store.occurrence_of(row)
    for _ in range(16):
        pool.close(store, stream=0)
    pool.close(store, stream=0, references=(token,))
    assert token in pool.entries
    before = snapshot(resolver)
    # The production teardown consumes the cursor's document-end marks.
    from Models import BasicModel
    model = SimpleNamespace(_thought_document_ends=(True,), _sentence_sources=((('second', 1),),),
        _concept_owner=lambda: owner)
    BasicModel._end_step(model)
    assert token not in pool.entries and not pool.names
    unchanged(resolver, before)


@pytest.mark.parametrize('operation', ('open', 'bind'))
def test_reclosing_a_mention_does_not_erase_another_tokens_event(operation):
    resolver, store, atoms = fixture()
    pool = resolver.components.owner.mtm
    first, _ = kept(resolver, mention(atoms[0]), position=0)
    second, _ = kept(resolver, mention(atoms[1]), position=1)
    a, b = store.occurrence_of(first), store.occurrence_of(second)
    pool.event(a, b, verb=1, role=2, when=3)
    events = pool.entries[a].events
    before = snapshot(resolver)
    if operation == 'open':
        pool.open(b, kind=None, when=4, value=atoms[1])
    else:
        pool.bind(a, b, kind=None, when=4, value=atoms[1])
    assert pool.entries[a].events == events
    unchanged(resolver, before)
    pool.forget(b)
    assert not pool.entries[a].events


def test_standalone_order_one_composite_gets_a_site_through_its_head():
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    word = native(owner, atoms[0])
    tree = Composition(atoms[0], (Composition(atoms[1]),
        Composition(atoms[0], reference=word, word_id=word)), operation='surface', head_role=2, order=1)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0], order=1, composition=tree)
    before = snapshot(resolver)
    plan = resolver.preview(clause, ())
    assert len(plan['items']) == 1 and plan['items'][0]['own_object']
    torch.testing.assert_close(plan['items'][0]['cue'], atoms[0])
    unchanged(resolver, before)
    row, plan = kept(resolver, clause)
    token = owner.mtm.token_of(store.occurrence_of(row))
    assert token == store.occurrence_of(row)
    torch.testing.assert_close(owner.mtm.value_of(token), atoms[0])


def test_definition_cache_rebuilds_after_forgetting_and_restore():
    from test_definition_rows import model, admit
    from Definitions import DefinitionIndex
    m = model()
    owner = m._concept_owner()
    word, obj = admit(m, 'sample')
    parts, index = owner._concept_allocator.parts, owner.definitions
    saved = {key: value.clone() for key, value in parts.state_dict().items()}
    edge = index.row(word, obj)
    parts.retain([row for row in parts.indices().tolist() if row != edge])
    assert index.objects(word) == () and index.words(obj) == ()
    parts.load_state_dict(saved)
    assert index.objects(word) == (obj,) and index.words(obj) == (word,)
    rebuilt = DefinitionIndex(parts, owner._concept_allocator)
    assert rebuilt.row(word, obj) == edge and rebuilt.description(word) == index.description(word)


def test_event_without_factored_refs_uses_the_captured_verb_reading(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from test_identity_threading import recorded_identity_reading
    from ClauseJournal import finish_clause
    f = SentenceFixture(monkeypatch)
    entry, _, _ = recorded_identity_reading(f, ('lift', ('lower_unary', 'subject'), 'action'))
    clause = finish_clause(f.language, entry, registry=f.registry)
    verb = native(f.cs, torch.ones(8)*.25)
    word = f.cs.interpret.lookup_word([901], (), form='event-form')
    f.cs.interpret.define(word, verb)
    # A captured record can carry the lexical handle when the factored view
    # is absent; the word has no conceptual seat in this fixture.
    from Spaces import _concept_alloc_of
    assert f.cs._csw_row_of(word) is None
    identity = dict(clause.identity)
    identity['event'] = tuple(word if index == 3 else value for index, value in enumerate(identity['event']))
    clause = replace(clause, factored_refs=None, identity=identity)
    resolver = f.cs.components.reference_close
    _, plan = kept(resolver, clause)
    events = [event for token in f.cs.mtm.entries.values() for event in token.events]
    assert events and events[-1].verb == verb
    assert resolver._participants()
