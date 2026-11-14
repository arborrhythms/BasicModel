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
from Memory import MediumTermMemory
from ReferenceContext import SituationFrame
from test_reference_close import fixture
from test_medium_term_memory import mention, snapshot, unchanged


def kept(resolver, clause, frames=(), *, document='consolidation', position=0, stream=0):
    plan = resolver.preview(clause, chunk_context(resolver, frames), stream=stream)
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = resolver.store.write_clause(prepared,
          written_rows=written)
    resolver.commit(plan, row, written, document=document, position=position, training=False)
    return row, plan


def native(owner, point):
    symbol = owner.new_concept()
    force_order(owner, symbol, 0)
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_concept_row(0, symbol)].copy_(point)
    return symbol






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
        row = store.write_clause(prepared,
                                  written_rows=written)
        resolver.commit(plan, row, written, document='consolidation', position=position, training=False)
        owner.mtm.promote_candidates(owner)
        return store.location(written[('occurrence', id(plan['items'][0]['written_node']))]), plan
    first, _ = actual(mention(atoms[0], reference=kind))
    token = store.occurrence_of(first)
    frame = SituationFrame.from_row(store.row(first))
    second, _ = actual(mention(atoms[0]+atoms[1], mode='bind'), (frame,), position=1)
    assert pool.entries[token].symbol is None
    # A repeated sentence is a distinct retained observation (§17.21).
    third, _ = actual(mention(atoms[0]+atoms[1], mode='bind'), (frame,), position=1)
    assert pool.entries[token].symbol is not None
    # §50: the third mention contains the held description instead of only overlapping it.
    fourth, _ = actual(mention(atoms[0]+atoms[1]+atoms[2], mode='bind'), (frame,), position=2)
    entry = pool.entries[token]
    assert entry.occurrences == (token, store.occurrence_of(second), store.occurrence_of(third), store.occurrence_of(fourth))
    assert entry.kind == kind and entry.symbol is not None and entry.count == 4
    expected = atoms[0] + 2*atoms[1]/3
    torch.testing.assert_close(owner.similarity_codebook.W[owner._csw_row_of(entry.symbol)], expected)
    assert resolver._derivation_symbol(token) == entry.symbol
    changed = store.append_idea(atoms[3])
    pool.event(token, store.occurrence_of(changed), verb=verb, role=0, when=50,
        before=pool.value_of(token))
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
    rows = [store.write_clause(mention(atoms[n]), )
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
    # §52 item 109: only the definition writer names this lexical pair.
    f.cs.interpret.define(word, verb,
        description=dict(forms=('event-form',), parts=((901,),), wholes=()))
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
    # §50: a lexical site's held meaning is its dictionary's current content.
    owner = resolver.components.owner
    with torch.no_grad():
        owner.similarity_codebook.W[owner._csw_row_of(words['x'])].copy_(atoms[0])
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
    later = resolver.preview(site(2), chunk_context(resolver, (frame,)), stream=0)
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
    value = pool.value_of(key).clone()
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
    torch.testing.assert_close(pool.value_of(key), value)


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
    import weakref
    f.cs.mtm._owner = weakref.ref(f.cs)
    tree = ('sum',
        ('lift', ('lower_unary', 'subject'), ('verb', 'move', ('lower', 'd', 'place'))),
        ('lift', ('lower_unary', 'alias'), ('verb', 'affect', ('lower', 'd', 'target'))))
    entry, _, _ = recorded_identity_reading(f, tree)
    resolver = f.cs.components.reference_close
    clause = finish_clause(f.language, entry, registry=f.registry)
    plan = resolver.preview(clause, chunk_context(resolver, ()))
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, )
    before = snapshot(resolver)
    resolver.commit(plan, row, written, document='capacity-close', position=0, training=False)
    names = [written[('occurrence', id(item['written_node']))] for item in plan['items']]
    assert len(names) == 4 and len(f.cs.mtm.entries) == 2
    # Event collections share this capacity: the final participant and the
    # final event occupy the two seats, and every evicted history is gone.
    assert all(f.cs.mtm.token_of(name) is None for name in names[:-1])
    assert f.cs.mtm.token_of(names[-1]) is not None
    assert sum(entry.refinement.get('member_kind') == 'event'
               for entry in f.cs.mtm.entries.values()) == 1
    assert all(store_row is not None for store_row in map(f.store.location, names))
    assert f.cs.mtm.entries[f.cs.mtm.token_of(names[-1])].events
    unchanged(resolver, before)


def test_a_kept_bind_refreshes_its_token_even_when_model_time_is_unchanged():
    resolver, store, atoms = fixture()
    store._model_present = lambda: 7
    pool = resolver.components.owner.mtm
    first, _ = kept(resolver, mention(atoms[0]), position=0)
    token = store.occurrence_of(first)
    frame = SituationFrame.from_row(store.row(first))
    for position in range(1, 16):
        kept(resolver, mention(atoms[1]), position=position)
    # The link must arrive within the horizon; mere retention is insufficient.
    bound, _ = kept(resolver, mention(atoms[0], mode='bind'), (frame,), position=16)
    assert pool.token_of(store.occurrence_of(bound)) == token
    for position in range(17, 33):
        kept(resolver, mention(atoms[1]), position=position)
    assert token in pool.entries
    kept(resolver, mention(atoms[1]), position=33)
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
    import weakref
    pool._owner = weakref.ref(resolver.components.owner)
    arow, _ = kept(resolver, mention(atoms[0]), position=0)
    brow, _ = kept(resolver, mention(atoms[1]), position=1)
    crow = store.write_clause(mention(atoms[2]), )
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
