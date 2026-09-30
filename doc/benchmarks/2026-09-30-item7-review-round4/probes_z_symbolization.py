"""No inventory prefix remains after a closing cannot symbolize its parent."""
from probes_z_corrected import inventory
from Spaces import _concept_alloc_of
from test_item7_acceptance import SentenceFixture


def test_full_inventory_refuses_symbolization_and_still_writes_the_part_row(monkeypatch):
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'cat', 'animal'))
    alloc = _concept_alloc_of(f.cs)
    caps = f.cs._order_caps()
    for rung in range(len(caps)):
        alloc.layer(0)._row_next[sum(caps[:rung])] = caps[rung]
    alloc.layer(0)._row_next[sum(caps)] = f.cs.nVectors - sum(caps)
    before = inventory(f.cs)
    row = f.store.write_clause(clause)
    assert row >= 0 and f.store.rel_type[row] == f.store.REL_PARTOF
    assert inventory(f.cs) == before
    assert tuple(f.store.refs[row, [0, 2]].tolist()) == (f.noun('cat'), f.noun('animal'))
