"""TruthSet ingestion must use the same selected clause closing as observations."""
from test_ltm_consolidation import _make_model_provisioned, _SERIAL_CONFIG
from test_clause_storage import clause_store, part_clause
from ClauseRow import Clause
import pytest


@pytest.mark.usefixtures('eager_reading')
def test_provisioning_retains_completed_fields_and_shared_row_ids():
    model = _make_model_provisioned(_SERIAL_CONFIG)
    store = model.symbolSpace.ltm_store
    asserted = store.rows_of_origin(store.ORIGIN_PROVISIONED).tolist()
    definitions = store.relations('definition').tolist()
    assert len(asserted) == 3
    assert tuple(definitions) == model._fixture_admitted_definitions, "reading cannot invent another DEF object"
    assert len(store) == len(asserted) + 3
    assert sum(store.row(i)['kind'] == 'cover' for i in range(len(store))) == 3
    for row in asserted:
        assert int(store.meaning_of(row).role_mask.sum()) == 1
        assert not hasattr(store, 'clause_derivation')
        assert int(store.row_ids[row]) not in (-1, 0)
        assert store.row(row)['kind'] == 'fact'
    for row in definitions:
        assert store.part_row(row)['kind'] == 'observation'
        assert store.part_row(row)['evidence'] == (1., 0.)


def test_truthset_asserts_only_outer_ends_including_deduplicated_roots():
    store, _ = clause_store()
    child = part_clause()
    outer = Clause(child.meaning, relation='operator', refs=(4, 3, ('clause', 0)),
                   children=(child,))
    with store.clause_assertions(trust=.7, origin=store.ORIGIN_USER,
                                text='he said cats are animals') as rows:
        parent = store.write_clause(outer, evidence=(.7, 0.))
    assert rows == [parent]
    child_row = ('sym', store.parts.find(1, 2))
    assert store.row(child_row)['kind'] == 'unverified'
    assert store.row(child_row)['evidence'] == (0., 0.)
    assert store.row(parent)['evidence'] == pytest.approx((.7, 0.))
    with store.clause_assertions(trust=.8, origin=store.ORIGIN_USER,
                                text='cats are animals') as rows:
        assert store.write_clause(child, evidence=(.8, 0.)) == child_row
    assert rows == [child_row] and len(store.parts) == 2
    assert store.row(child_row)['evidence'] == pytest.approx((.8, 0.))


def test_external_assertion_context_cannot_declare_a_kind():
    store, _ = clause_store()
    with pytest.raises(TypeError, match='relation'):
        with store.clause_assertions(trust=1., origin=store.ORIGIN_PROVISIONED,
                                    text='cats are animals', relation='implies'):
            store.write_clause(part_clause())
    assert len(store) == 0
