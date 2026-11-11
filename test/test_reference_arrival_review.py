"""Regressions from the review of the frozen anonymous-binding source."""
from chunk_fixtures import context as chunk_context
import torch

from ReferenceContext import SituationFrame
from test_bind_thought import execute, native, occurrence, site
from test_medium_term_memory import keep, mention, snapshot, unchanged
from test_reference_arrivals import equality, force_named, named, described, world


def test_occurrence_bind_witness_is_a_locatable_address(monkeypatch):
    f = native(monkeypatch)
    target, _ = occurrence(f, f.atoms[0], f.atoms[4])
    address = f.identities[target]
    result = execute(f, site(f, f.atoms[0], predicate=f.atoms[4]))
    assert result.value.role_refs[0] == address
    assert result.evidence['witnesses'] == (address,)
    assert all(f.store.location(witness) is not None for witness in result.evidence['witnesses'])


def test_native_arrival_without_synthetic_occurrence_joins_its_working_entry(monkeypatch):
    from test_reference_close import known_concept as mint
    resolver, store, atoms, words, _ = world(pending_atom=0)
    pending, _ = keep(resolver, described(atoms[0], words['x']))
    token = store.occurrence_of(pending)
    # A reference collection admits a native individual at the same order.
    individual = mint(resolver.components.owner, atoms[0], order=0)
    force_named(resolver, monkeypatch, individual)
    frame = SituationFrame(1, torch.stack((atoms[0], atoms[0]*0, atoms[0]*0)),
        torch.tensor([True, False, False]), -1, atoms[0], symbol_ids=(individual,))
    clause = equality(atoms[1], words['x'], words['wrong'], subject=atoms[0])
    plan = resolver.preview(clause, chunk_context(resolver, (SituationFrame.from_row(store.row(pending)), frame)))
    before, entries = snapshot(resolver), dict(resolver.components.owner.mtm.entries)
    prepared = resolver.prepare(clause, plan)
    assert not prepared.companions and individual not in plan['arrival_written']
    unchanged(resolver, before)
    assert resolver.components.owner.mtm.entries == entries
    written = {}
    row = store.write_clause(prepared, document_key='mtm', sentence_index=1, written_rows=written)
    before = snapshot(resolver)
    resolver.commit(plan, row, written, document='mtm', position=1, training=False)
    unchanged(resolver, before)
    owner = resolver.components.owner.mtm.token_of(token)
    entry = resolver.components.owner.mtm.entries[owner]
    assert entry.kind == individual and entry.occurrences == (token, owner) and entry.count == 2
    assert any(item.get('refinement') and item['selected'] == individual
               and item['reference'] == owner and item['missing_reason'] is None
               for item in resolver.last_audit)


def test_arrival_without_a_selection_reports_it_without_reopening_the_token(monkeypatch):
    resolver, store, atoms, words, _ = world()
    pending, _ = keep(resolver, named(atoms[0], words['x']))
    token = store.occurrence_of(pending)
    before = resolver.components.owner.mtm.entries[token]
    force_named(resolver, monkeypatch, -1000)
    frame = SituationFrame.from_row(store.row(pending))
    _, plan = keep(resolver, equality(atoms[1], words['x'], words['wrong']), (frame,), position=1)
    assert resolver.components.owner.mtm.entries[token] == before
    assert any(item['token'] == token and item['selected'] is None for item in plan['arrivals'])
    audit = next(item for item in resolver.last_audit
                 if item.get('refinement') and item['occurrence'] == token[-1])
    assert audit['reference'] == token and audit['missing_reason'] == 'no_selected_candidate'


def test_new_document_can_refine_a_same_stream_site_in_the_live_pool(monkeypatch):
    resolver, store, atoms, words, _ = world(pending_atom=0)
    force_named(resolver, monkeypatch, words['three'])

    def close(clause, document):
        plan = resolver.preview(clause, chunk_context(resolver, ()), document=document, position=0, stream=0)
        prepared = resolver.prepare(clause, plan)
        written = {}
        row = store.write_clause(prepared, document_key=document, sentence_index=0,
                                 stream=0, written_rows=written)
        retained = snapshot(resolver)
        resolver.commit(plan, row, written, document=document, position=0, training=False)
        unchanged(resolver, retained)
        return row

    pending = close(described(atoms[0], words['x']), 'old-document')
    token = store.occurrence_of(pending)
    pool = resolver.components.owner.mtm
    before = pool.entries[token]
    # D4: containment plus the live same-stream pool permits refinement.
    close(equality(atoms[0], words['x'], words['three'], subject=atoms[0]), 'new-document')
    retained = pool.entries[pool.token_of(token)]
    assert token in retained.occurrences and retained.kind == words['three']
    assert retained.count >= before.count


def test_refining_from_a_pooled_candidate_keeps_its_earlier_use_and_full_content(monkeypatch):
    resolver, store, atoms, words, _ = world(pending_atom=0)
    pending, _ = keep(resolver, described(atoms[0], words['x']))
    token = store.occurrence_of(pending)
    description = atoms[0] + atoms[2]
    from dataclasses import replace
    candidate_site = mention(description, reference=words['three'])
    candidate_site = replace(candidate_site, reference_request=dict(candidate_site.reference_request, leaves=(0, 1)))
    candidate, _ = keep(resolver, candidate_site, position=1)
    candidate_token = store.occurrence_of(candidate)
    frame = SituationFrame.from_row(store.row(candidate))
    used, _ = keep(resolver, mention(description, mode='bind'), (frame,), position=2)
    use = store.occurrence_of(used)
    pool = resolver.components.owner.mtm
    assert pool.token_of(use) == candidate_token
    before = pool.entries[candidate_token]
    force_named(resolver, monkeypatch, words['three'])
    keep(resolver, equality(atoms[1], words['x'], words['wrong'], subject=atoms[0]),
         (SituationFrame.from_row(store.row(pending)), frame), position=3)
    token = pool.token_of(token)
    assert pool.entries[token].kind == words['three']
    # Kind alone does not retain the chosen occurrence's distinguishing content.
    torch.testing.assert_close(resolver.token_value(token), description)
    assert pool.token_of(use) == candidate_token
    assert tuple(member for member in pool.entries[candidate_token].occurrences
                 if member in before.occurrences) == before.occurrences  # Merge interleaves the older introduction.


def test_arrival_readers_exclude_scoped_rows_before_a_new_stream_column_exists():
    resolver, store, atoms, words, _ = world()
    clause = named(atoms[0], words['x'])
    plan = resolver.preview(clause, chunk_context(resolver, ()), stream=0)
    prepared = resolver.prepare(clause, plan)
    written = {}
    pending = store.write_clause(prepared, document_key='initial-stream', stream=0,
                                 written_rows=written)
    resolver.commit(plan, pending, written, document='initial-stream', position=0, training=False)
    token = store.occurrence_of(pending)
    retained, entry = snapshot(resolver), resolver.components.owner.mtm.entries[token]
    assert store.stream.shape[1] == store.capacity and store.stream[pending, 0]
    arrival = equality(atoms[0], words['x'], words['three'])
    individuals = resolver.individuals(resolver.accessible(chunk_context(resolver, (SituationFrame.from_row(store.row(pending)),))))
    for stream in (1, 2, 7):
        assert resolver.pending_names(arrival, individuals, stream) == ()
        result = dict(items=[], records={token: (pending, 1.)})
        resolver._arrival_plan(arrival, result, individuals, stream=stream)
        assert result['arrivals'] == []
        assert resolver.components.owner.mtm.entries[token] is entry
        assert store.stream.shape[1] == store.capacity
        unchanged(resolver, retained)


def test_arrival_readers_keep_unscoped_rows_available_to_a_new_stream():
    resolver, store, atoms, words, _ = world()
    pending, _ = keep(resolver, named(atoms[0], words['x']))
    token = store.occurrence_of(pending)
    assert not store.stream[pending].any()
    arrival = equality(atoms[0], words['x'], words['three'])
    individuals = resolver.individuals(resolver.accessible(chunk_context(resolver, (SituationFrame.from_row(store.row(pending)),))))
    retained = snapshot(resolver)
    stream = store.stream.shape[1]
    assert resolver.pending_names(arrival, individuals, stream) == (words['x'],)
    result = dict(items=[], records={token: (pending, 1.)}, decisions=[], departure=-1,
                  cost=atoms[0].new_zeros(()))
    resolver._arrival_plan(arrival, result, individuals, stream=stream)
    assert len(result['arrivals']) == 1 and result['arrivals'][0]['token'] == token
    unchanged(resolver, retained)
