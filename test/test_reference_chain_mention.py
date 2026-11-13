"""An open sentence chain and its MTM description have distinct addresses."""
from dataclasses import replace

import pytest
import torch

from ClauseRow import Clause, Composition
from Memory import SYMBOL
from Meaning import ConceptualMeaning
from chunk_fixtures import context
from test_clause_acceptance import SentenceFixture


def unresolved_description(f, *, lexical_kind=False):
    known = f.noun('known')
    word = f.cs.interpret.lookup_word([81], [], form='unbound')
    left = Composition(f.registry._payload(('sym', known)),
                       reference=f.words['known'][0] if lexical_kind else known,
                       order=1, leaves=(0,))
    right = Composition(torch.eye(8)[3], word_id=word, order=1, leaves=(1,))
    value = torch.maximum(left.point, right.point)
    term = Composition(value, (left, right), operation='union', formal_name='union',
                       order=1, leaves=(0, 1))
    request = dict(mode='bind', compose_bind=0, role=0, leaves=(0, 1), content=value,
                   cue=value, predicate=torch.zeros_like(value))
    clause = Clause(ConceptualMeaning.from_description(value), point=value, order=1,
                    composition=term, reference_request=request)
    return clause, known, word


def test_unresolved_root_description_keeps_chain_and_own_occurrence(monkeypatch):
    f = SentenceFixture(monkeypatch)
    resolver = f.cs.components.reference_close
    clause, known, word = unresolved_description(f)
    plan = resolver.preview(clause, context(resolver, ()))
    kept = resolver.prepare(clause, plan)
    # Models adds the sentence time after prepare; root address keys must also
    # work when that final dataclass replacement changes its Python identity.
    closed = replace(kept, when=kept.point.new_zeros(4))
    written = {}
    row = f.store.write_clause(closed, evidence=(.8, .1), document_key='chain-mention',
                               sentence_index=0, written_rows=written)
    assert row[0] == 'sym'
    assert f.store.parts.references(row[1]) == ((SYMBOL, known), (SYMBOL, word))
    assert written['root_occurrence'] == row
    occurrence = written['root_mention']
    assert occurrence[0] == 'ltm' and occurrence != row
    mention = f.store.location(occurrence)
    assert f.store.kind_of(mention) == 'unverified'
    assert f.store.row(mention)['evidence'] == (0., 0.)
    assert not f.store.entry_time(mention).any()
    torch.testing.assert_close(f.store.point_of_row(occurrence[-1]), kept.point, rtol=0, atol=0)
    resolver.commit(plan, row, written, document='chain-mention', position=0, training=False)
    token = f.cs.mtm.token_of(occurrence)
    assert token is not None and f.cs.mtm.token_of(row) is None
    torch.testing.assert_close(f.cs.mtm.value_of(token), kept.point, rtol=0, atol=0)
    assert f.cs.definitions.objects(word) == ()


def test_root_mention_occurrence_is_in_capacity_preflight(monkeypatch):
    f = SentenceFixture(monkeypatch)
    resolver = f.cs.components.reference_close
    clause, _, _ = unresolved_description(f)
    kept = resolver.prepare(clause, resolver.preview(clause, context(resolver, ())))
    f.store.count.fill_(f.store.capacity)
    before = f.cs._concept_allocator.next_id, len(f.store.parts)
    with pytest.raises(OverflowError, match='LTM0 construction capacity'):
        f.store.write_clause(kept)
    assert (f.cs._concept_allocator.next_id, len(f.store.parts)) == before


def test_description_does_not_turn_its_lexical_kind_into_a_word_reading(monkeypatch):
    f = SentenceFixture(monkeypatch)
    resolver = f.cs.components.reference_close
    clause, known, _ = unresolved_description(f, lexical_kind=True)
    plan = resolver.preview(clause, context(resolver, ()))
    kept = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(kept, written_rows=written)
    occurrence = written['root_mention']
    assert resolver._derivation_symbol(occurrence) == f.words['known'][0]
    resolver.commit(plan, row, written, document='lexical-kind', position=0, training=False)
    token = f.cs.mtm.token_of(occurrence)
    assert f.cs.mtm.reading_of(occurrence) is None
    torch.testing.assert_close(f.cs.mtm.value_of(token), kept.point, rtol=0, atol=0)
    assert not torch.equal(kept.point, f.registry._payload(('sym', known)))


def test_compressed_root_mention_reuses_its_own_occurrence(monkeypatch):
    f = SentenceFixture(monkeypatch)
    known = f.noun('known')
    point = f.registry._payload(('sym', known))
    clause = Clause(ConceptualMeaning.from_description(point), point=point, order=1,
                    composition=Composition(point, reference=known, order=1, leaves=(0,)),
                    reference_request=dict(mode='bind', leaves=(0,)))
    written = {}
    before = len(f.store)
    row = f.store.write_clause(clause, written_rows=written)
    assert isinstance(row, int) and len(f.store) == before + 1
    assert written.get('root_mention', written['root_occurrence']) == written['root_occurrence'] == f.store.occurrence_of(row)
