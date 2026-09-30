"""DEF symbols and ended clauses must share the grammar's identity owner."""
from test_ltm_consolidation import _make_model_provisioned, _SERIAL_CONFIG


def test_reading_and_closing_use_one_concept_allocator():
    model = _make_model_provisioned(_SERIAL_CONFIG)
    assert model._concept_owner() is model.grammatical_thoughts.space


def test_new_definition_symbols_do_not_alias_later_sentence_occurrences():
    model = _make_model_provisioned(_SERIAL_CONFIG)
    store = model.symbolSpace.ltm_store
    definitions = store.relations(store.REL_DEF)
    symbols = set(store.refs[definitions][:, (0, 2)].reshape(-1).tolist())
    assertions = store.rows_of_origin(store.ORIGIN_PROVISIONED)
    assert symbols.isdisjoint(store.row_ids[assertions].tolist())
