"""Provenance bounds a claim; its content cannot certify itself (item 7)."""
import pytest
from test_clause_storage import clause_store, part_clause


def test_conflicting_claim_preserves_support_and_counterevidence():
    store, _ = clause_store()
    row = store.write_clause(part_clause(), trust=.8, evidence=(.8, 0.))
    assert store.write_clause(part_clause(polarity=False), trust=.6, evidence=(0., .6)) == row
    assert len(store) == 1
    assert store.row(row)['evidence'] == pytest.approx((.8, .6))
    assert store.row(row)['corners'][2] == pytest.approx(.6)


def test_zero_provenance_registers_without_asserting():
    store, _ = clause_store()
    row = store.write_clause(part_clause(), trust=0.)
    assert row == 0
    assert store.row(row)['evidence'] == (0., 0.)
    assert store.row(row)['corners'][3] == 1.
