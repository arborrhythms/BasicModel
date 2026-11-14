from Memory import STM0
from dataclasses import replace
"""Reviewer probes for bounded nested occurrence retention and restore."""

import copy

import pytest
import torch

from Memory import LTM0
from Meaning import ConceptualMeaning


def _meaning(*, child=None, scope=(), twice=False, **metadata):
    return replace(ConceptualMeaning.from_description(torch.eye(6)[0]), role_refs=(child, None, None), scope=scope, **metadata)


def _chain(store, child, *, evidence=0.):
    from Memory import OCCURRENCE
    row = store.parts.add((OCCURRENCE, child[-1]), (OCCURRENCE, child[-1]), truth_plus=evidence)
    return ('sym', row)


def _index(store, reference):
    return next(
        (i for i in range(len(store)) if store.occurrence_of(i) == reference), None
    )


def _unchanged(store, before):
    for name, value in before.items():
        if name != 'activity':
            torch.testing.assert_close(store.state_dict()[name], value)






def test_missing_local_constituent_rejects_append_before_owner_mutation():
    store = STM0(6, capacity=3, relation_capacity=8192)
    row = store.append_meaning(_meaning())
    prefix = store.occurrence_of(row)[:2]
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="occurrence|constituent|reference"):
        store.append_meaning(_meaning(child=prefix + (99,)))
    _unchanged(store, before)


def test_repeated_edges_keep_role_order_without_evidence_inheritance():
    from part_fixtures import native_store
    store, _ = native_store(width=6, capacity=4)
    row = store.append_meaning(_meaning(scope={"quote": True}))
    child = store.occurrence_of(row)
    root = _chain(store, child, evidence=.9)
    view = store.read_structure(root, max_nodes=2, max_depth=1, max_records=4)
    assert view["incomplete"] == ()
    assert view["occurrences"] == (root, child)
    assert view["edges"] == ((root, 0, child), (root, 1, child))
    assert view["records_scanned"] == 2
    assert view["rows"][1]["trust"] == 0
    assert view["rows"][1]["meaning"].scope == (("quote", True),)


@pytest.mark.parametrize(
    "limit,reason",
    (({"max_nodes": 1}, "node_limit"), ({"max_depth": 0}, "depth_limit"),
     ({"max_records": 1}, "record_limit")),
)
def test_nested_read_limits_are_separate_and_explicit(limit, reason):
    from part_fixtures import native_store
    store, _ = native_store(width=6, capacity=4)
    child = store.occurrence_of(store.append_meaning(_meaning()))
    root = _chain(store, child)
    before = copy.deepcopy(store.state_dict())
    view = store.read_structure(
        root, **dict({"max_nodes": 4, "max_depth": 4, "max_records": 4}, **limit)
    )
    assert reason in view["incomplete"]
    _unchanged(store, before)


def test_retained_structure_and_reference_identity_survive_checkpoint_restore():
    store = STM0(6, capacity=4, relation_capacity=8192)
    child = store.occurrence_of(store.append_meaning(_meaning()))
    root = store.occurrence_of(store.append_meaning(_meaning(child=child)))
    restored = STM0(6, capacity=4, relation_capacity=8192)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    view = restored.read_structure(root)
    assert view["incomplete"] == () and view["occurrences"] == (root,) and view["edges"] == ()
    assert restored.meaning_of(_index(restored, root)).role_refs[0] == child
    assert restored.row(_index(restored, child))["trust"] == 0.
    assert not view["rows"][0]["meaning"].roles.requires_grad


def _boundary_meaning(ref=None):
    return replace(ConceptualMeaning.from_description(torch.eye(5)[0]), role_refs=(ref, None, None))


def _world():
    store = STM0(5, capacity=5, relation_capacity=8192)
    child = store.occurrence_of(store.append_meaning(_boundary_meaning()))
    root = store.occurrence_of(store.append_meaning(_boundary_meaning(child)))
    return store, child, root


def _corrupt_child(store, child, reference):
    from Memory import OCCURRENCE
    store.referent[store.location(child)] = store.referent.new_tensor((OCCURRENCE, reference[-1]))
    store.working.pop(child[-1], None)


def test_structural_restore_rejects_dangling_local_links_atomically():
    store, child, root = _world()
    wrong = child[:2] + (999,)
    _corrupt_child(store, child, wrong)
    saved = copy.deepcopy(store.state_dict())
    restored = STM0(5, capacity=5, relation_capacity=8192)
    before = copy.deepcopy(restored.state_dict())
    with pytest.raises(ValueError, match="unavailable|referent"):
        restored.load_state_dict(saved)
    _unchanged(restored, before)


def test_context_cycle_stays_finite_without_derivation_or_evidence_transfer():
    store, child, root = _world()
    _corrupt_child(store, child, root)
    view = store.read_structure(root, max_nodes=2, max_depth=2, max_records=2)
    # Identity bindings retain references, but an idea has no derivation to walk.
    assert view["incomplete"] == () and view["edges"] == ()
    assert view["occurrences"] == (root,) and view["records_scanned"] == 1
    assert store.raw_meaning_of(0).role_refs[0] == root
    assert store.raw_meaning_of(1).role_refs[0] == child
    restored = STM0(5, capacity=5, relation_capacity=8192)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    reread = restored.read_structure(root, max_nodes=2, max_depth=2, max_records=2)
    assert reread["incomplete"] == () and reread["occurrences"] == (root,)
    assert restored.row(0)["trust"] == 0.
    torch.testing.assert_close(restored.trust, store.trust, rtol=0, atol=0)




def test_derived_structure_read_does_not_expose_mutable_store_slots():
    store, child, root = _world()
    before = store.slots.clone()
    view = store.read_structure(root)
    view["rows"][0]["point"].zero_()
    view["rows"][0]["meaning"].roles.zero_()
    torch.testing.assert_close(store.slots, before)


def test_unavailable_owner_and_retired_reference_have_explicit_incomplete_results():
    store, child, root = _world()
    other = STM0(5, capacity=5, relation_capacity=8192)
    foreign = other.occurrence_of(other.append_meaning(_boundary_meaning(), name=999))
    assert store.read_structure(foreign)["incomplete"] == ("unavailable_reference",)
    store.reset()
    replacement = store.occurrence_of(store.append_meaning(_boundary_meaning(), ))
    assert replacement not in (child, root)
    assert store.read_structure(root)["incomplete"] == ("unavailable_reference",)




@pytest.mark.parametrize("field", ("bindings", "scope"))
def test_missing_local_context_reference_rejects_append_before_mutation(field):
    store = STM0(6, capacity=4, relation_capacity=8192)
    row = store.append_meaning(_meaning())
    missing = store.occurrence_of(row)[:2] + (99,)
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="reference|occurrence|context"):
        store.append_meaning(_meaning(**{field: {"referent": missing}}))
    _unchanged(store, before)
