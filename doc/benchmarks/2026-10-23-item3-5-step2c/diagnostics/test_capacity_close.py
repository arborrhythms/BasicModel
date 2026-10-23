"""The bounded pool can evict an earlier introduction in the same close."""
from MediumTermMemory import MediumTermMemory
from ClauseJournal import finish_clause
from test_clause_acceptance import SentenceFixture
from test_identity_threading import recorded_identity_reading
from test_medium_term_memory import snapshot, unchanged


def test_capacity_eviction_during_close_drops_only_the_evicted_tokens_history(monkeypatch):
    f = SentenceFixture(monkeypatch)
    f.cs.mtm = MediumTermMemory(capacity=2)
    tree = ('sum',
        ('lift', ('lower_unary', 'subject'), ('verb', 'move', ('lower', 'd', 'place'))),
        ('lift', ('lower_unary', 'alias'), ('verb', 'affect', ('lower', 'd', 'target'))))
    entry, _, _ = recorded_identity_reading(f, tree)
    resolver = f.cs.components.reference_close
    clause = finish_clause(f.language, entry, registry=f.registry)
    plan = resolver.preview(clause, ())
    prepared = resolver.prepare(clause, plan)
    written = {}
    row = f.store.write_clause(prepared, written_rows=written, document_key='capacity-close', sentence_index=0)
    before = snapshot(resolver)
    resolver.commit(plan, row, written, document='capacity-close', position=0, training=False)
    names = [written[('occurrence', id(item['written_node']))] for item in plan['items']]
    assert len(names) == 4 and len(f.cs.mtm.entries) == 2
    assert all(f.cs.mtm.token_of(name) is None for name in names[:2])
    assert all(f.cs.mtm.token_of(name) is not None for name in names[2:])
    assert all(store_row is not None for store_row in map(f.store.location, names))
    assert all(f.cs.mtm.entries[f.cs.mtm.token_of(name)].events for name in names[2:])
    unchanged(resolver, before)
