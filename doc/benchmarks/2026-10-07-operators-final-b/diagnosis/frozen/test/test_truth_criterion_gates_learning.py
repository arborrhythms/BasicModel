"""Item 7 retires content-based acceptance; grammar registers every clause."""
import pytest
from Spaces import ConceptualSpace
from test_clause_storage import clause_store, part_clause




@pytest.mark.parametrize('trust', [0., .3, 1., -.7])
def test_registration_does_not_depend_on_testimony_trust(trust):
    store, _ = clause_store()
    evidence = (max(trust, 0.), max(-trust, 0.))
    row = store.write_clause(part_clause(), trust=trust)
    assert row == 0 and len(store) == 1
    assert store.row(row)['evidence'] == pytest.approx(evidence)
    assert store.row(row)['trust'] == pytest.approx(trust)
