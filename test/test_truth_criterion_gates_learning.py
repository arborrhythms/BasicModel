"""Item 7 retires content-based acceptance; grammar registers every clause."""
import pytest
from Spaces import ConceptualSpace
from test_item7_storage import clause_store, part_clause


@pytest.mark.parametrize('name', [
    '_compute_learn_score', '_learn_score_children_in_codebook',
    '_learn_score_is_truth_obvious', '_learn_score_resolves_contradiction',
    '_route_learned_relation', '_maybe_learn_relation', 'learn_relations_from_stm',
    '_relation_is_reducible', '_collapse_trust'])
def test_content_gate_and_alternative_relation_writer_are_deleted(name):
    assert not hasattr(ConceptualSpace, name)


@pytest.mark.parametrize('trust', [0., .3, 1., -.7])
def test_registration_does_not_depend_on_testimony_trust(trust):
    store, _ = clause_store()
    evidence = (.6, .4)
    row = store.write_clause(part_clause(), trust=trust, evidence=evidence)
    assert row == 0 and len(store) == 1
    assert store.row(row)['evidence'] == pytest.approx(evidence)
    assert store.row(row)['trust'] == pytest.approx(trust)
