"""Relation admission has one clause writer; WholeSpace owns no truth graph."""
import pytest
from Spaces import WholeSpace, ConceptualSpace
from test_item7_storage import clause_store, part_clause


@pytest.mark.parametrize('name', ['insert_relation', 'insert_meta', 'meta_trust',
    'taxonomy_children', 'taxonomy_parent'])
def test_wholespace_truth_and_taxonomy_writer_is_deleted(name):
    assert not hasattr(WholeSpace, name)


def test_closing_reuses_relation_and_preserves_independent_evidence():
    from dataclasses import replace
    store, _ = clause_store()
    clause = part_clause()
    row = store.write_clause(clause, trust=.8, evidence=(.8, 0.))
    assert store.write_clause(replace(clause, meaning=replace(clause.meaning, polarity=False)), trust=.7, evidence=(0., .7)) == row
    assert len(store) == 1
    assert store.row(row)['evidence'] == pytest.approx((.8, .7))


@pytest.mark.parametrize('name', ['_maybe_learn_relation', '_route_learned_relation',
    '_learn_score_children_in_codebook', '_learn_score_is_truth_obvious'])
def test_gated_relation_routes_are_deleted(name):
    assert not hasattr(ConceptualSpace, name)
