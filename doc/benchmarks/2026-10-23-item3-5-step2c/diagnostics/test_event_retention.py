"""Losing a token does not reclaim an event row owned by another token."""
import pytest
from MediumTermMemory import MediumTermMemory
from test_reference_close import fixture
from test_medium_term_memory import mention, snapshot, unchanged
from test_memory_consolidation import kept


@pytest.mark.parametrize('retirement', ('capacity', 'expiry'))
def test_retiring_a_token_keeps_another_tokens_event_until_row_reclamation(retirement):
    resolver, store, atoms = fixture()
    pool = resolver.components.owner.mtm = MediumTermMemory(capacity=2)
    arow, _ = kept(resolver, mention(atoms[0]), position=0)
    brow, _ = kept(resolver, mention(atoms[1]), position=1)
    crow = store.write_clause(mention(atoms[2]), document_key='consolidation', sentence_index=2)
    a, b, c = map(store.occurrence_of, (arow, brow, crow))
    pool.event(a, b, verb=1, role=2, when=50)
    events = pool.entries[a].events
    before = snapshot(resolver)
    if retirement == 'capacity':
        pool.open(c, kind=None, when=51, value=atoms[2])
    else:
        for _ in range(17):
            pool.close(store, stream=0, references=(a,))
    assert pool.token_of(b) is None
    assert pool.entries[a].events == events
    unchanged(resolver, before)
    pool.forget(b)
    assert not pool.entries[a].events
