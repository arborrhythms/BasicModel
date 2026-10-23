"""A production model clock can stamp several kept closes with one tick."""
from test_reference_close import fixture
from test_medium_term_memory import mention
from test_memory_consolidation import kept


def test_kept_closes_with_one_model_tick_do_not_refresh_an_unmentioned_token():
    resolver, store, atoms = fixture()
    store._model_present = lambda: 7
    pool = resolver.components.owner.mtm
    first, _ = kept(resolver, mention(atoms[0]), position=0)
    token = store.occurrence_of(first)
    for position in range(1, 17):
        kept(resolver, mention(atoms[1]), position=position)
    assert token in pool.entries
    kept(resolver, mention(atoms[1]), position=17)
    assert token not in pool.entries
    assert store.location(token) is not None
