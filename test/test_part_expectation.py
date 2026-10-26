"""A Part close keeps its frozen forecast/arrival audit outside LTM2."""
import copy
import pytest
import torch

from ClauseRow import attach_clause_index
from Layers import ExpectationComparison, MeaningExpectation, LTM0
from test_clause_acceptance import SentenceFixture


@pytest.mark.parametrize('relative', [False, True])
def test_closing_audit_survives_bank_changes_and_checkpoint(monkeypatch, relative):
    f = SentenceFixture(monkeypatch)
    source = f.store.write_clause(f.clause('a'), evidence=(1., 0.))
    clause = f.clause(('part', 'a', 'b') if relative else ('lift', 'a', 'b'))
    identities = torch.tensor([f.noun('a'), f.noun('b')])
    keys = torch.stack([f.registry._payload(('sym', int(i))) for i in identities])
    logits = torch.tensor([[2., -1.], [-2., 1.]])
    prediction = MeaningExpectation(torch.zeros(3, 8), torch.zeros(3),
        symbol_keys=keys, symbol_logits=logits, symbol_ids=identities,
        symbol_orders=torch.zeros(2, dtype=torch.long))
    target = torch.tensor([[1., 0.], [0., 1.]])
    comparison = ExpectationComparison(prediction, clause.meaning.roles,
        clause.meaning.role_mask, target-logits.sigmoid(), document='audit',
        source_occurrences=(f.store.occurrence_of(source),), stream=0, activation=target)
    result = f.store.write_clause(clause, evidence=(.7, .1), expectation=comparison)
    pair = f.store.expectation_pair(result)
    torch.testing.assert_close(pair['residual'], target-logits.sigmoid())
    assert pair['source_occurrences'] == (f.store.occurrence_of(source),)
    assert pair['surprise'] == pytest.approx(comparison.surprise)
    assert len(f.store.parts.columns) == 7  # tagged references and percept, no separate clock
    # The mutable current bank is not the historical target's owner.
    with torch.no_grad():
        f.cs.similarity_codebook.W.fill_(.25)
    state, extras = copy.deepcopy(f.store.state_dict()), copy.deepcopy(f.store.semantic_extras())
    restored = LTM0(8, capacity=64, percept_width=0)
    attach_clause_index(f.model, restored, f.cs)
    restored.load_state_dict(state)
    restored.load_semantic_extras(extras)
    replay = restored.expectation_pair(result)
    torch.testing.assert_close(replay['residual'], pair['residual'], rtol=0, atol=0)
    assert replay['intended_occurrence'] == pair['intended_occurrence']


def test_part_expectation_capacity_failure_precedes_all_admission(monkeypatch):
    f = SentenceFixture(monkeypatch)
    source = f.store.write_clause(f.clause('a'))
    clause = f.clause(('part', 'a', 'b'))
    ids = torch.tensor([f.noun('a'), f.noun('b')])
    keys = torch.stack([f.registry._payload(('sym', int(i))) for i in ids])
    prediction = MeaningExpectation(torch.zeros(3, 8), torch.zeros(3),
        symbol_keys=keys, symbol_logits=torch.zeros(2, 2), symbol_ids=ids,
        symbol_orders=torch.zeros(2, dtype=torch.long))
    comparison = ExpectationComparison(prediction, clause.meaning.roles,
        clause.meaning.role_mask, torch.zeros(2, 2), document='full',
        source_occurrences=(f.store.occurrence_of(source),), activation=torch.ones(2, 2)*.5)
    f.store.capacity = len(f.store)
    before = f.cs._concept_allocator.next_id, len(f.store.parts), len(f.store)
    with pytest.raises(OverflowError):
        f.store.write_clause(clause, expectation=comparison)
    assert (f.cs._concept_allocator.next_id, len(f.store.parts), len(f.store)) == before
