"""Retained red reproductions for the unresolved arrival review findings."""
import torch

from ReferenceContext import SituationFrame
from test_medium_term_memory import keep, mention, snapshot, unchanged
from test_reference_arrivals import equality, force_named, named, world


def test_refining_from_a_pooled_candidate_keeps_its_earlier_use_and_full_content(monkeypatch):
    resolver, store, atoms, words, _ = world()
    pending, _ = keep(resolver, named(atoms[0], words['x']))
    token = store.occurrence_of(pending)
    description = atoms[0] + atoms[2]
    candidate, _ = keep(resolver, mention(description, reference=words['three']), position=1)
    candidate_token = store.occurrence_of(candidate)
    frame = SituationFrame.from_row(store.row(candidate))
    used, _ = keep(resolver, mention(description, mode='bind'), (frame,), position=2)
    use = store.occurrence_of(used)
    pool = resolver.components.owner.mtm
    assert pool.token_of(use) == candidate_token
    before = pool.entries[candidate_token]
    force_named(resolver, monkeypatch, words['three'])
    keep(resolver, equality(atoms[1], words['x'], words['wrong']),
         (SituationFrame.from_row(store.row(pending)), frame), position=3)
    assert pool.entries[token].kind == words['three']
    # Kind alone does not retain the chosen occurrence's distinguishing content.
    torch.testing.assert_close(resolver.token_value(token), description)
    assert pool.token_of(use) == candidate_token
    assert pool.entries[candidate_token].occurrences[:len(before.occurrences)] == before.occurrences


def test_new_document_cannot_refine_a_same_stream_site_outside_its_frames(monkeypatch):
    resolver, store, atoms, words, _ = world()
    force_named(resolver, monkeypatch, words['three'])

    def close(clause, document):
        plan = resolver.preview(clause, (), document=document, position=0, stream=0)
        prepared = resolver.prepare(clause, plan)
        written = {}
        row = store.write_clause(prepared, document_key=document, sentence_index=0,
                                 stream=0, written_rows=written)
        retained = snapshot(resolver)
        resolver.commit(plan, row, written, document=document, position=0, training=False)
        unchanged(resolver, retained)
        return row

    pending = close(named(atoms[0], words['x']), 'old-document')
    token = store.occurrence_of(pending)
    pool = resolver.components.owner.mtm
    before = pool.entries[token]
    close(equality(atoms[0], words['x'], words['three']), 'new-document')
    assert pool.entries[token] == before
