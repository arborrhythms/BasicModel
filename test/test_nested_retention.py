from dataclasses import replace
"""Reviewer probes for bounded nested occurrence retention and restore."""

import copy

import pytest
import torch

from Layers import LTM0
from Meaning import ConceptualMeaning


def _meaning(*, child=None, scope=(), twice=False, **metadata):
    return replace(ConceptualMeaning.from_description(torch.eye(6)[0]), role_refs=(child, None, None), scope=scope, **metadata)


def _chain(store, child, *, evidence=0.):
    from LTM1 import OCCURRENCE
    row = store.parts.add((OCCURRENCE, child[-1]), (OCCURRENCE, child[-1]), truth_plus=evidence)
    return ('sym', row)


def _index(store, reference):
    return next(
        (i for i in range(len(store)) if store.occurrence_of(i) == reference), None
    )


def _unchanged(store, before):
    for name, value in before.items():
        torch.testing.assert_close(store.state_dict()[name], value)


def test_origin_replacement_retains_reachable_content_without_fact_authority():
    store = LTM0(6, capacity=8)
    leaf = store.append_meaning(_meaning(scope={"quote": "inner"}), trust=.8, sentence_index=len(store))
    store.set_origin(leaf, store.ORIGIN_USER, text="reported content")
    leaf_ref = store.occurrence_of(leaf)
    inner = store.append_meaning(
        _meaning(child=leaf_ref), kind="unverified"
    , sentence_index=len(store))
    store.set_origin(inner, store.ORIGIN_USER)
    inner_ref = store.occurrence_of(inner)
    root = store.append_meaning(
        _meaning(child=inner_ref, scope={"speaker": "outer"}), kind="observation"
    , sentence_index=len(store))
    root_ref = store.occurrence_of(root)
    orphan = store.append_meaning(_meaning(), trust=.7, sentence_index=len(store))
    store.set_origin(orphan, store.ORIGIN_USER)

    assert store.clear_origin(store.ORIGIN_USER) == 1
    assert len(store) == 3
    for reference in (leaf_ref, inner_ref, root_ref):
        assert _index(store, reference) is not None
    kept = store.row(_index(store, leaf_ref))
    assert kept["kind"] == "unverified" and kept["trust"] == 0
    assert kept["text"] == "reported content"
    assert kept["meaning"].scope == (("quote", "inner"),)
    assert store.row(_index(store, inner_ref))["kind"] == "unverified"
    assert store.row(_index(store, inner_ref))["evidence"] == (0., 0.)
    assert store.clear_origin(store.ORIGIN_USER) == 0


def test_external_live_root_pins_an_occurrence_through_origin_replacement():
    store = LTM0(6, capacity=3)
    row = store.append_meaning(_meaning(), trust=1, sentence_index=len(store))
    store.set_origin(row, store.ORIGIN_USER)
    ref = store.occurrence_of(row)
    assert store.clear_origin(store.ORIGIN_USER, retained_occurrences=(ref,)) == 0
    assert store.row(0)["occurrence"] == ref
    assert store.row(0)["kind"] == "unverified"
    assert store.clear_origin(store.ORIGIN_USER) == 1


def test_missing_local_constituent_rejects_append_before_owner_mutation():
    store = LTM0(6, capacity=3)
    row = store.append_meaning(_meaning(), sentence_index=len(store))
    prefix = store.occurrence_of(row)[:2]
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="occurrence|constituent|reference"):
        store.append_meaning(_meaning(child=prefix + (99,)), sentence_index=len(store))
    _unchanged(store, before)


def test_repeated_edges_keep_role_order_without_evidence_inheritance():
    from part_fixtures import native_store
    store, _ = native_store(width=6, capacity=4)
    row = store.append_meaning(_meaning(scope={"quote": True}), kind="unverified")
    child = store.occurrence_of(row)
    root = _chain(store, child, evidence=.9)
    view = store.read_structure(root, max_nodes=2, max_depth=1, max_records=4)
    assert view["incomplete"] == ()
    assert view["occurrences"] == (root, child)
    assert view["edges"] == ((root, 0, child), (root, 1, child))
    assert view["records_scanned"] == 2
    assert view["rows"][1]["kind"] == "unverified"
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
    child = store.occurrence_of(store.append_meaning(_meaning(), sentence_index=len(store)))
    root = _chain(store, child)
    before = copy.deepcopy(store.state_dict())
    view = store.read_structure(
        root, **dict({"max_nodes": 4, "max_depth": 4, "max_records": 4}, **limit)
    )
    assert reason in view["incomplete"]
    _unchanged(store, before)


def test_retained_structure_and_reference_identity_survive_checkpoint_restore():
    store = LTM0(6, capacity=4)
    child = store.occurrence_of(store.append_meaning(_meaning(), sentence_index=len(store)))
    store.set_origin(0, store.ORIGIN_USER)
    root = store.occurrence_of(store.append_meaning(_meaning(child=child), sentence_index=len(store)))
    store.clear_origin(store.ORIGIN_USER)
    restored = LTM0(6, capacity=4)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    view = restored.read_structure(root)
    assert view["incomplete"] == () and view["occurrences"] == (root,) and view["edges"] == ()
    assert restored.meaning_of(_index(restored, root)).role_refs[0] == child
    assert restored.row(_index(restored, child))["kind"] == "unverified"
    assert not view["rows"][0]["meaning"].roles.requires_grad


def _boundary_meaning(ref=None):
    return replace(ConceptualMeaning.from_description(torch.eye(5)[0]), role_refs=(ref, None, None))


def _world():
    store = LTM0(5, capacity=5)
    child = store.occurrence_of(store.append_meaning(_boundary_meaning(), kind="unverified", sentence_index=len(store)))
    root = store.occurrence_of(store.append_meaning(_boundary_meaning(child), kind="observation", sentence_index=len(store)))
    return store, child, root


def _corrupt_child(store, child, reference):
    identifier = child[2]
    store._semantic_rows[identifier]["bindings"] = (("_identity", reference),)
    store.metadata_required[0] = True
    store.semantic_fingerprint[0] = store.semantic_fingerprint.new_tensor(
        store._context_fingerprint(store._semantic_rows[identifier], store.text_of(0))
    )


def test_structural_restore_rejects_dangling_local_links_atomically():
    store, child, root = _world()
    good = copy.deepcopy(store.semantic_extras())
    wrong = child[:2] + (999,)
    _corrupt_child(store, child, wrong)
    saved, extras = copy.deepcopy(store.state_dict()), copy.deepcopy(store.semantic_extras())
    restored = LTM0(5, capacity=5)
    restored.load_state_dict(saved)
    before_context = copy.deepcopy(restored._semantic_rows)
    before_texts = list(restored._texts)
    with pytest.raises(ValueError, match="cycle|unavailable|constituent"):
        restored.load_semantic_extras(extras)
    assert restored._semantic_rows == before_context and restored._texts == before_texts
    assert dict(good["records"][0]["context"]["bindings"]).get("_identity") is None


def test_context_cycle_stays_finite_without_derivation_or_evidence_transfer():
    store, child, root = _world()
    _corrupt_child(store, child, root)
    view = store.read_structure(root, max_nodes=2, max_depth=2, max_records=2)
    # Identity bindings retain references, but an idea has no derivation to walk.
    assert view["incomplete"] == () and view["edges"] == ()
    assert view["occurrences"] == (root,) and view["records_scanned"] == 1
    assert store.raw_meaning_of(0).role_refs[0] == root
    assert store.raw_meaning_of(1).role_refs[0] == child
    restored = LTM0(5, capacity=5)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    reread = restored.read_structure(root, max_nodes=2, max_depth=2, max_records=2)
    assert reread["incomplete"] == () and reread["occurrences"] == (root,)
    assert restored.row(0)["kind"] == "unverified"
    torch.testing.assert_close(restored.c_plus, store.c_plus, rtol=0, atol=0)
    torch.testing.assert_close(restored.c_minus, store.c_minus, rtol=0, atol=0)


def test_invalid_survivor_closure_cannot_partly_withdraw_evidence():
    store, child, root = _world()
    store.set_origin(0, store.ORIGIN_USER)
    _corrupt_child(store, child, child[:2] + (999,))
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="unavailable|constituent"):
        store.clear_origin(store.ORIGIN_USER)
    _unchanged(store, before)


def test_derived_structure_read_does_not_expose_mutable_store_slots():
    store, child, root = _world()
    before = store.slots.clone()
    view = store.read_structure(root)
    view["rows"][0]["point"].zero_()
    view["rows"][0]["meaning"].roles.zero_()
    torch.testing.assert_close(store.slots, before)


def test_unavailable_owner_and_retired_reference_have_explicit_incomplete_results():
    store, child, root = _world()
    other = LTM0(5, capacity=5)
    foreign = other.occurrence_of(other.append_meaning(_boundary_meaning(), document_key='other-document', sentence_index=len(other)))
    assert store.read_structure(foreign)["incomplete"] == ("unavailable_reference",)
    store.reset()
    replacement = store.occurrence_of(store.append_meaning(_boundary_meaning(), document_key='replacement', sentence_index=len(store)))
    assert replacement not in (child, root)
    assert store.read_structure(root)["incomplete"] == ("unavailable_reference",)


@pytest.mark.parametrize("field", ("bindings", "scope"))
def test_withdrawal_preserves_occurrences_in_retained_semantic_context(field):
    store = LTM0(6, capacity=4)
    child = store.append_meaning(_meaning(), trust=.8, sentence_index=len(store))
    store.set_origin(child, store.ORIGIN_USER)
    ref = store.occurrence_of(child)
    outer = store.append_meaning(
        _meaning(**{field: {"referent": ref}}), kind="observation"
    , sentence_index=len(store))
    root = store.occurrence_of(outer)
    assert store.clear_origin(store.ORIGIN_USER) == 0
    assert store.occurrence_of(0) == ref
    assert store.row(0)["trust"] == 0 and store.row(0)["kind"] == "unverified"
    assert store.occurrence_of(1) == root
    assert getattr(store.meaning_of(1), field) == (("referent", ref),)


@pytest.mark.parametrize("field", ("bindings", "scope"))
def test_missing_local_context_reference_rejects_append_before_mutation(field):
    store = LTM0(6, capacity=4)
    row = store.append_meaning(_meaning(), sentence_index=len(store))
    missing = store.occurrence_of(row)[:2] + (99,)
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="reference|occurrence|context"):
        store.append_meaning(_meaning(**{field: {"referent": missing}}), sentence_index=len(store))
    _unchanged(store, before)
