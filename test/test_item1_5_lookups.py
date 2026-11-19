"""Exact ordered endpoint reads and index maintenance across reuse and reload."""
import copy

import pytest
import torch

from Memory import LTM1, OCCURRENCE, RELATION, SYMBOL


def reference(store):
    pairs, parts, wholes, references = {}, {}, {}, {}
    for row in store.indices().tolist():
        left, right = zip(store.ref_kinds[row].tolist(), store.refs[row].tolist())
        pairs[left, right] = row
        parts.setdefault(right, []).append(left)
        wholes.setdefault(left, []).append(right)
        for end in {left, right}:
            references.setdefault(end, []).append(row)
    return pairs, parts, wholes, references


def test_ordered_index_survives_duplicate_reuse_retain_and_checkpoint():
    store = LTM1(32)
    store.add(23, 27)
    store.add((OCCURRENCE, 5), 27)
    store.add(24, 27)
    store.add((RELATION, 1), 29)
    store.add(23, 29, truth_plus=.5)
    def check():
        expected = reference(store)
        assert (store._pairs, store._parts, store._wholes, store._references) == expected
        assert store.reference_pairs(store.indices()) == {v: k for k, v in expected[0].items()}
    check()
    prior = store._index_version
    assert store.add(23, 27, truth_plus=.3) == 1 and store._index_version == prior
    store.retain([1, 3, 4, 5])
    assert store.add(26, 27) == 2
    check()
    assert store.parts(27) == (23, 26, 24) and store.wholes(23) == (27, 29)
    assert store.rows_referencing(SYMBOL, 27) == (1, 2, 3)
    assert store.protected_references([4]) == {(RELATION, 1), (SYMBOL, 29), (SYMBOL, 23), (SYMBOL, 27)}
    assert store.stratum(1) == 1 and store.stratum(4) == 2
    assert store.contains_rows([-1, 0, 4, 6, 32]) == [False, False, True, False, False]
    state = copy.deepcopy(store.state_dict())
    store = LTM1(32)
    store.load_state_dict(state)
    check()
    store.ref_kinds[2, 1] = 0
    assert not store.contains(2)
    with pytest.raises(IndexError, match='unavailable relation'):
        store.reference_pairs([1, 2])
    with pytest.raises(ValueError, match='invalid LTM1 retention selection'):
        store.retain([1, 2])


def test_witness_cache_follows_allocation_witness_writes_and_retirement():
    from test_cs_sparse_weights import _cs, _mint_row
    owner = _cs()
    a = _mint_row(owner, 0, 9010)
    b = _mint_row(owner, 1, 9011)
    allocator = owner._concept_allocator
    layer = allocator.layer(0)
    def reference():
        return [r for r, cid in owner._row_to_concept().items()
                if cid not in allocator.retired and (owner._order0_inventory_row(r) or bool(layer.witnessed[r]))]
    assert owner._witnessed_rows() == reference()
    assert a in reference() and b not in reference()
    layer.witnessed[b] = True
    assert owner._witnessed_rows() == reference() and b in reference()
    allocator.retired.add(9010)
    assert owner._witnessed_rows() == reference() and a not in reference()
    returned = owner._witnessed_rows()
    returned.clear()
    assert owner._witnessed_rows() == reference()
