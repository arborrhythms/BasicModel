"""Cover-segment recurrence, pool lifetime and retained identity promotion."""
from order_fixtures import force_order
from chunk_fixtures import context as chunk_context
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
    from DocumentCover import begin_sentence
    model = resolver.__dict__.setdefault('_cover_fixture_cursor', SimpleNamespace(
        _concept_owner=lambda: resolver.components.owner))
    model._sentence_sources = tuple(((document, position),) if b == stream else () for b in range(stream+1))
    present = torch.zeros(stream+1, dtype=torch.bool)
    present[stream] = True
    begin_sentence(model, 0, present)
    plan = resolver.preview(clause, chunk_context(resolver, frames), stream=stream)
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = resolver.store.write_clause(prepared, document_key=document,
        sentence_index=position, stream=stream, written_rows=written)
    resolver.commit(plan, row, written, document=document, position=position, training=False)
    return row, plan


def native(owner, point):
    symbol = owner.new_concept()
    force_order(owner, symbol, 0)
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
    from DocumentCover import open_cover
    for n in range(3):
        if n == 0 or distinct:
            open_cover(owner, store, stream=0, document=('source', n if distinct else 0))
        row = store.write_clause(clause, document_key=('source', n if distinct else 0), sentence_index=n,
                                 stream=0, evidence=(1., 0.))
        before = store.slots.clone(), store.parts.refs.clone()
        consolidate(owner, store)
        reading = int(store.recurrence_readings[row])
        assert int(store.recurrence_documents[reading]) == (n + 1 if distinct else 1)
        if n < 2 or not distinct:
            assert int(store.recurrence_symbols[reading]) == -1
        torch.testing.assert_close(store.slots, before[0], rtol=0, atol=0)
        torch.testing.assert_close(store.parts.refs, before[1], rtol=0, atol=0)
    if distinct:
        symbol = int(store.recurrence_symbols[reading])
        assert symbol > 0
        torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(symbol)], value)
        before = snapshot(resolver)
        consolidate(owner, store)
        unchanged(resolver, before)
        from Layers import LTM0
        restored = LTM0(12, capacity=512, percept_width=0)
        restored.load_state_dict(store.state_dict())
        restored.load_semantic_extras(store.semantic_extras())
        assert int(restored.recurrence_symbols[reading]) == symbol
        torch.testing.assert_close(restored.recurrence_codes, store.recurrence_codes)


def test_token_promotes_at_three_sentences_with_the_occurrence_fold_and_keeps_its_state():
    resolver, store, atoms = fixture()
    owner, pool = resolver.components.owner, resolver.components.owner.mtm
    kind = native(owner, atoms[0])
    verb = native(owner, atoms[3])
    def actual(node, frames=(), *, position=0):
        root = Clause(ConceptualMeaning.from_description(node.point), point=node.point,
            order=0, evidence=(1., 0.), children=(node,), refs=(('clause', 0), -1, -1),
            factored_refs=(('clause', 0), verb, -1))
        plan = resolver.preview(root, chunk_context(resolver, frames), stream=0)
        prepared = resolver.prepare(root, plan)
        written = {}
        row = store.write_clause(prepared, document_key='consolidation', sentence_index=position,
                                 stream=0, written_rows=written)
        resolver.commit(plan, row, written, document='consolidation', position=position, training=False)
        owner.mtm.promote_candidates(owner)
        return store.location(written[('occurrence', id(plan['items'][0]['written_node']))]), plan
    first, _ = actual(mention(atoms[0], reference=kind))
    token = store.occurrence_of(first)
    frame = SituationFrame.from_row(store.row(first))
    second, _ = actual(mention(atoms[0]+atoms[1], mode='bind'), (frame,), position=1)
    assert pool.entries[token].symbol is None
    # Reclosing the same source sentence increments accesses, not recurrence.
    actual(mention(atoms[0]+atoms[1], mode='bind'), (frame,), position=1)
    assert pool.entries[token].symbol is None
    third, _ = actual(mention(atoms[0]+atoms[2], mode='bind'), (frame,), position=2)
    entry = pool.entries[token]
    assert entry.occurrences == (token, store.occurrence_of(second), store.occurrence_of(third))
    assert entry.kind == kind and entry.symbol is not None and entry.count == 4
    expected = atoms[0] + (atoms[1] + atoms[2]) / 3
    torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(entry.symbol)], expected)
    assert resolver._derivation_symbol(token) == entry.symbol
    pool.update(token, atoms[3], when=50)
    consolidate(owner, store)
    torch.testing.assert_close(pool.value_of(token), atoms[3])
    torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(entry.symbol)], expected)
    restored = MediumTermMemory(pool.capacity,pool.horizon,n_vectors=int(pool.transient_ids[0]), field_width=pool.field_width)
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
        pool.bind(token, store.occurrence_of(row), kind=None, value=store.slots[row], when=2)
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
    assert token in pool.entries and pool.names
    for _ in range(pool.horizon + 1):
        pool.close(store, stream=0)
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
    plan = resolver.preview(clause, chunk_context(resolver, ()))
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
    from Spaces import _concept_alloc_of
    alloc = _concept_alloc_of(f.cs)
    word = alloc.new_concept()
    alloc.symbol_types[word] = {'word'}
    alloc.symbol_types[verb] = {'concept', 'object'}
    alloc.word_descriptions[word] = dict(forms=('event-form',), parts=((901,),), wholes=())
    alloc.parts.add(verb, word)
    f.cs.definitions.rebuild()
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


def test_pending_alias_and_later_self_arrival_leave_the_same_token_pending():
    from test_reference_arrivals import world, named
    resolver, store, atoms, words, _ = world()
    def site(leaf):
        node = named(atoms[0], words['x'])
        return replace(node, point=atoms[0], meaning=ConceptualMeaning.from_description(atoms[0]),
            composition=replace(node.composition, point=atoms[0], leaves=(leaf,)),
            reference_request=dict(node.reference_request, content=atoms[0], cue=atoms[0], leaves=(leaf,)))
    first, second = site(0), site(1)
    clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
        children=(first, second), refs=(('clause', 0), -1, ('clause', 1)))
    row, plan = kept(resolver, clause)
    token = plan['items'][0]['reference']
    assert plan['items'][1]['reference'] == token
    assert len(plan['arrivals']) == 1 and plan['arrivals'][0]['reference'] == token
    assert resolver.components.owner.mtm.entries[token].pending
    frame = replace(SituationFrame.from_row(store.row(row)), activations=tuple(
        (use, None, 0) for use in resolver.components.owner.mtm.entries[token].occurrences))
    before = snapshot(resolver)
    clocks = dict(resolver.components.owner.mtm.closes)
    later = resolver.preview(site(2), chunk_context(resolver, (frame,)))
    assert later['items'][0]['selected'] == token
    assert len(later['arrivals']) == 1 and later['arrivals'][0]['selected'] == token
    assert resolver.components.owner.mtm.closes == clocks
    unchanged(resolver, before)


@pytest.mark.parametrize('corruption', ('clock', 'access', 'document', 'symbol'))
def test_invalid_expiry_or_promotion_checkpoint_keeps_the_existing_pool(corruption):
    resolver, store, atoms = fixture()
    row, _ = kept(resolver, mention(atoms[0]))
    pool = resolver.components.owner.mtm
    key = store.occurrence_of(row)
    saved, bad = pool.state(), pool.state()
    if corruption == 'clock':
        bad['closes'] = ((0, -1),)
    elif corruption == 'access':
        bad['accesses'] = ((key, ((0, 99),)),)
    elif corruption == 'document':
        bad['documents'] = ((key, b'bad'),)
    else:
        bad['entries'][0]['symbol'] = 0
    with pytest.raises(ValueError):
        pool.restore(bad)
    assert pool.names == {key: pool.names[key]} and pool.closes == dict(saved['closes'])
    torch.testing.assert_close(pool.value_of(key), saved['entries'][0]['timelines'][0][1])


def test_kept_closes_with_one_model_tick_do_not_refresh_an_unmentioned_token():
    resolver, store, atoms = fixture()
    store._model_present = lambda: 7
    pool = resolver.components.owner.mtm
    first, _ = kept(resolver, mention(atoms[0]), position=0)
    token = store.occurrence_of(first)
    for position in range(1, 17):
        kept(resolver, mention(atoms[1]), position=position)
    assert token in pool.entries
    kept(resolver, mention(atoms[1]), position=17)
    assert token not in pool.entries
    assert store.location(token) is not None


def test_capacity_eviction_during_close_drops_only_the_evicted_tokens_history(monkeypatch):
    from ClauseJournal import finish_clause
    from test_clause_acceptance import SentenceFixture
    from test_identity_threading import recorded_identity_reading
    f = SentenceFixture(monkeypatch)
    f.cs.mtm = MediumTermMemory(capacity=2,n_vectors=f.cs.nVectors)
    tree = ('sum',
        ('lift', ('lower_unary', 'subject'), ('verb', 'move', ('lower', 'd', 'place'))),
        ('lift', ('lower_unary', 'alias'), ('verb', 'affect', ('lower', 'd', 'target'))))
    entry, _, _ = recorded_identity_reading(f, tree)
    resolver = f.cs.components.reference_close
    clause = finish_clause(f.language, entry, registry=f.registry)
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='capacity-close', sentence_index=0)
    before = snapshot(resolver)
    resolver.commit(plan, row, written, document='capacity-close', position=0, training=False)
    names = [written[('occurrence', id(item['written_node']))] for item in plan['items']]
    assert len(names) == 4 and len(f.cs.mtm.entries) == 2
    assert all(f.cs.mtm.token_of(name) is None for name in names[:2])
    assert all(f.cs.mtm.token_of(name) is not None for name in names[2:])
    assert all(store_row is not None for store_row in map(f.store.location, names))
    assert all(f.cs.mtm.entries[f.cs.mtm.token_of(name)].events for name in names[2:])
    unchanged(resolver, before)


def test_a_kept_bind_refreshes_its_token_even_when_model_time_is_unchanged():
    resolver, store, atoms = fixture()
    store._model_present = lambda: 7
    pool = resolver.components.owner.mtm
    first, _ = kept(resolver, mention(atoms[0]), position=0)
    token = store.occurrence_of(first)
    frame = SituationFrame.from_row(store.row(first))
    for position in range(1, 17):
        kept(resolver, mention(atoms[1]), position=position)
    bound, _ = kept(resolver, mention(atoms[0], mode='bind'), (frame,), position=17)
    assert pool.token_of(store.occurrence_of(bound)) == token
    for position in range(18, 34):
        kept(resolver, mention(atoms[1]), position=position)
    assert token in pool.entries
    kept(resolver, mention(atoms[1]), position=34)
    assert token not in pool.entries and store.location(token) is not None


def test_surface_wrapped_relation_does_not_add_a_standalone_object():
    resolver, store, atoms = fixture()
    relation = Composition(atoms[0], (Composition(atoms[1]), Composition(atoms[2])), operation='equal')
    wrapped = Composition(atoms[0], (Composition(atoms[3]), relation), operation='surface', head_role=2, order=1)
    clause = Clause(ConceptualMeaning(atoms[:3], torch.ones(3,dtype=torch.bool)),
        point=None, order=1, relation='part', composition=wrapped)
    before = snapshot(resolver)
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    assert plan['items'] == []
    unchanged(resolver, before)


@pytest.mark.parametrize('retirement', ('capacity', 'expiry'))
def test_retiring_a_token_keeps_another_tokens_event_until_row_reclamation(retirement):
    resolver, store, atoms = fixture()
    pool = resolver.components.owner.mtm = MediumTermMemory(capacity=2)
    arow, _ = kept(resolver, mention(atoms[0]), position=0)
    brow, _ = kept(resolver, mention(atoms[1]), position=1)
    crow = store.write_clause(mention(atoms[2]), document_key='consolidation', sentence_index=2)
    a, b, c = map(store.occurrence_of, (arow, brow, crow))
    pool.event(a, b, verb=1, role=2, when=50)
    events = pool.entries[a].events
    before = snapshot(resolver)
    if retirement == 'capacity':
        pool.close(store, stream=0, references=(a,))  # Capacity now uses the shared access clock.
        pool.open(c, kind=None, when=51, value=atoms[2])
    else:
        for _ in range(17):
            pool.close(store, stream=0, references=(a,))
    assert pool.token_of(b) is None
    assert pool.entries[a].events == events
    unchanged(resolver, before)
    pool.forget(b)
    assert not pool.entries[a].events


def test_angular_recurrence_admits_drift_and_keeps_evidence_weighted_content():
    from MemoryConsolidation import recurrence_groups
    from DocumentCover import open_cover
    resolver, store, atoms = fixture()
    owner = resolver.components.owner
    first = native(owner, atoms[0])
    pointers = tuple(getattr(store, name).data_ptr() for name in
        ('recurrence_codes', 'recurrence_documents', 'recurrence_readings'))
    rows = []
    for n, weight in enumerate((.25, .5, .75)):
        open_cover(owner, store, stream=0, document=('angle', n))
        value = atoms[0] + .05*n*atoms[1]
        meaning = replace(ConceptualMeaning.from_description(value), role_refs=(('sym', first), None, None))
        rows.append(store.append_meaning(meaning, evidence=(weight, 0.), document_key=('angle', n),
                                        sentence_index=n, stream=0))
        recurrence_groups(store)
    seat = int(store.recurrence_readings[rows[0]])
    assert store.recurrence_readings[rows].tolist() == [seat]*3
    assert int(store.recurrence_documents[seat]) == 3
    assert store._recurrence_identities[seat] == ('sym', first)
    weights = store.c_plus[rows] + store.c_minus[rows]
    expected = (store.slots[rows] * weights[:, None]).sum(0) / weights.sum()
    torch.testing.assert_close(store.recurrence_codes[seat], expected)
    before = store.recurrence_codes.clone(), store.recurrence_weights.clone()
    recurrence_groups(store)
    torch.testing.assert_close(store.recurrence_codes, before[0], rtol=0, atol=0)
    torch.testing.assert_close(store.recurrence_weights, before[1], rtol=0, atol=0)
    separate = store.append_idea(atoms[2], trust=1.)
    recurrence_groups(store)
    assert int(store.recurrence_readings[separate]) != seat
    assert pointers == tuple(getattr(store, name).data_ptr() for name in
        ('recurrence_codes', 'recurrence_documents', 'recurrence_readings'))


def test_recurrence_time_is_ignored_and_compaction_keeps_assignments():
    from MemoryConsolidation import recurrence_groups
    from Layers import LTM0
    store = LTM0(12, capacity=8, percept_width=8)
    value = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., .5, 0., 0., 0.])
    a = store.append_idea(value, trust=1., document_key='time-a')
    b = store.append_idea(value, trust=1., document_key='time-b')
    store.slots[a, 4:8] = 3.
    store.slots[b, 4:8] = -2.
    recurrence_groups(store)
    seat = int(store.recurrence_readings[a])
    assert int(store.recurrence_readings[b]) == seat
    assert store.recurrence_codes[seat, 4:8].eq(0).all()
    store.compact_rows([b])
    recurrence_groups(store)
    assert int(store.recurrence_readings[0]) == seat
    assert store.recurrence_readings[1:].eq(-1).all()
