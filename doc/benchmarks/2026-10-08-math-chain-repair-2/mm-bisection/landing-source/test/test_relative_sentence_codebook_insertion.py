"""Relation admission has one clause writer; WholeSpace owns no truth graph."""
import pytest
from Spaces import WholeSpace, ConceptualSpace
from test_clause_storage import clause_store, part_clause




def test_closing_reuses_relation_and_preserves_independent_evidence():
    from dataclasses import replace
    store, _ = clause_store()
    clause = part_clause()
    row = store.write_clause(clause, trust=.8, evidence=(.8, 0.))
    assert store.write_clause(replace(clause, meaning=replace(clause.meaning, polarity=False)), trust=.7, evidence=(0., .7)) == row
    assert len(store) == 1
    assert store.row(row)['evidence'] == pytest.approx((.8, .7))
