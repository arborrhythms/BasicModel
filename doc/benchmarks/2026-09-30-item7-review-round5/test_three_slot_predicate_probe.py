"""AK follow-up: a first three-slot predicate occurrence needs no prior row."""
import pytest
import torch

from ClauseRow import predicate_identity
from reading_fixtures import finish_reading
from test_item7_acceptance import SentenceFixture
from Understanding import AnswerProgram


@pytest.mark.parametrize('relation', ['part', 'implies'])
@pytest.mark.parametrize('use_registry', [False, True])
def test_first_three_slot_predicate_occurrence(monkeypatch, relation, use_registry):
    f = SentenceFixture(monkeypatch)
    left = f.store.write_clause(f.clause(('lift', 'cat', 'runs')))
    right = f.store.write_clause(f.clause(('lift', 'dog', 'sleeps')))
    reference = f.registry.clause_reference(relation)
    point = f.registry._payload(reference)
    leaves = torch.stack((f.store.slots[left, 0], point, f.store.slots[right, 0]))
    refs = torch.tensor([int(f.store.row_ids[left]), reference[1], int(f.store.row_ids[right])])
    entry = AnswerProgram(rows=torch.full((3,), -1), word_rows=torch.full((3,), -1),
        word_ids=torch.full((3,), -1), activations=torch.ones(3), leaves=leaves,
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [0, -1, 2]]),
        targets=torch.tensor([-1]), end_state=leaves, concept_ids=refs)
    before = dict(f.cs._csw_rows)
    clause = finish_reading(f.language, entry, registry=f.registry if use_registry else None)
    row = f.store.write_clause(clause)
    expected = f.store.REL_PARTOF if relation == 'part' else f.store.REL_IMPLIES
    assert f.store.rel_type[row] == expected
    assert f.store.refs[row].tolist() == refs.tolist()
    assert int(f.store.refs[row, 1]) == predicate_identity(relation)
    assert torch.equal(f.store.slots[row, 1], point)
    assert f.cs._csw_rows == before
    assert f.cs._csw_row_of(reference[1]) is None
