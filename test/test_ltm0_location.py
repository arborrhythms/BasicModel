"""Percept location is not a stored occurrence coordinate."""
import copy
from dataclasses import fields

import pytest
import torch

from ReferenceContext import SituationAddress, SituationFrame
from test_clause_storage import clause_store, idea_clause, part_clause


@pytest.mark.parametrize('legacy_where', (False, True))
def test_location_is_not_retained_and_time_survives_restore_and_compaction(legacy_where):
    store, _ = clause_store()
    clause = idea_clause()
    location = clause.where.clone()
    row = store.write_clause(clause, evidence=(.6, .2))
    occurrence = store.occurrence_of(row)
    when = store.when[row].clone()
    original = store.row(row)
    frame = SituationAddress(occurrence).read(store)

    assert not hasattr(store, 'where')
    assert 'where' not in original
    assert 'where' not in {field.name for field in fields(SituationFrame)}
    assert not hasattr(frame, 'where')
    torch.testing.assert_close(clause.where, location, rtol=0, atol=0)
    torch.testing.assert_close(frame.detached().when, when, rtol=0, atol=0)

    relation = store.write_clause(part_clause(), evidence=(.7, 0.))
    assert 'where' not in store.row(relation)
    state = copy.deepcopy(store.state_dict())
    assert 'where' not in state
    if legacy_where:
        state['where'] = torch.arange(store.capacity * 4).reshape(store.capacity, 4).float()
    restored, _ = clause_store()
    restored.load_state_dict(state, strict=True)
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    assert not hasattr(restored, 'where')
    assert 'where' not in restored.state_dict()
    torch.testing.assert_close(restored.when[row], when, rtol=0, atol=0)
    torch.testing.assert_close(restored.row(row)['meaning'].roles, original['meaning'].roles,
                               rtol=0, atol=0)
    assert restored.row(row)['evidence'] == original['evidence']
    restored.compact_rows([row])
    assert restored.occurrence_of(0) == occurrence
    torch.testing.assert_close(restored.when[0], when, rtol=0, atol=0)
    restored.reset()
    assert len(restored) == 0 and not hasattr(restored, 'where')
    assert not bool(restored.when.any())
