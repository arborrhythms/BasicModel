"""TruthSet ingestion must use the same selected clause closing as observations."""
from test_ltm_consolidation import _make_model_provisioned, _SERIAL_CONFIG
from test_clause_storage import clause_store, part_clause
from ClauseRow import Clause
from part_fixtures import stored_pair
import pytest


@pytest.mark.usefixtures('eager_reading')
def test_provisioning_retains_completed_fields_and_shared_addresses():
    model = _make_model_provisioned(_SERIAL_CONFIG)
    store = model.symbolSpace.ltm_store
    asserted = store.trust[:len(store)].ne(0).nonzero().flatten().tolist()
    definitions = store.relations('definition').tolist()
    assert len(asserted) == 3
    assert tuple(definitions) == model._fixture_admitted_definitions, "reading cannot invent another DEF object"
    assert len(store) == len(asserted)
    for row in asserted:
        assert int(store.meaning_of(row).role_mask.sum()) == 1
        assert not hasattr(store, 'clause_derivation')
        assert store.row_name(row) >= 0
    for row in definitions:
        assert store.part_row(row)['evidence'] == (1., 0.)


def test_truthset_asserts_only_outer_idea_and_reasserts_declared_pairs(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    store = f.store
    child = f.clause(('part','a','b'))
    outer = f.clause(('lift','speaker',('verb','predicate',('part','a','b'))))
    with store.clause_assertions(trust=.7) as rows:
        parent = store.write_clause(outer)
    assert rows == [parent] and isinstance(parent,int)
    assert store.row(parent)['evidence'] == pytest.approx((.7,0.))
    child_id = store.parts.find(f.noun('a'),f.noun('b'))
    if child_id is not None:
        assert store.parts.evidence(child_id) == (0.,0.)
    with store.clause_assertions(trust=.8) as rows:
        child_row = store.write_clause(child)
    assert rows == [child_row]
    assert store.row(child_row)['evidence'] == pytest.approx((.8,0.))
    assert store.row(parent)['evidence'] == pytest.approx((.7,0.))


def test_external_assertion_context_cannot_declare_a_kind():
    store, _ = clause_store()
    with pytest.raises(TypeError, match='relation'):
        with store.clause_assertions(trust=1.,
                                     relation='implies'):
            store.write_clause(part_clause())
    assert len(store) == 0
