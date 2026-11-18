"""The clause closing preserves prediction provenance and retrieval leaves."""
from types import SimpleNamespace

import torch
import pytest

from Layers import MeaningExpectation
from test_clause_storage import clause_store, idea_clause, part_clause




def test_idea_closing_initializes_activity_at_the_default_without_an_estimate():
    store, _ = clause_store()
    store.write_clause(idea_clause())
    target = store.write_clause(idea_clause())
    assert len(store) == 2 and target == 1
    assert float(store.activity[target]) == pytest.approx(.5)
    torch.testing.assert_close(store.slots[target], idea_clause().point)



def test_narrow_idea_retains_its_identity_without_a_predicate_operand():
    store, _ = clause_store()
    first = store.write_clause(idea_clause())
    from Occurrence import operand_name
    second = store.write_clause(idea_clause(refs=(operand_name(store.row_name(first)),3,-1)))
    assert not hasattr(store, 'refs')
    assert store.meaning_of(second).role_refs == (store.occurrence_of(first), None, None)
