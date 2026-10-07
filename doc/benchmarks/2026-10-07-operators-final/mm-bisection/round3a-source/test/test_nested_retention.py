"""Reviewer probes for bounded nested occurrence retention and restore."""

import copy

import pytest
import torch

from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning


def _meaning(*, child=None, mode="assertive", scope=(), twice=False, **metadata):
    return ConceptualMeaning(
        torch.eye(6)[:3], torch.ones(3, dtype=torch.bool), mode=mode,
        role_refs=(child if twice else None, ("sym", 17), child),
        scope=scope, **metadata,
    )


def _index(store, reference):
    return next(
        (i for i in range(len(store)) if store.occurrence_of(i) == reference), None
    )


def _unchanged(store, before):
    for name, value in before.items():
        torch.testing.assert_close(store.state_dict()[name], value)


def test_origin_replacement_retains_reachable_content_without_fact_authority():
    store = TernaryTruthStore(6, capacity=8)
    leaf = store.append_meaning(_meaning(scope={"quote": "inner"}), trust=.8)
    store.set_origin(leaf, store.ORIGIN_USER, text="reported content")
    leaf_ref = store.occurrence_of(leaf)
    inner = store.append_meaning(
        _meaning(child=leaf_ref, mode="interrogative"), kind="question"
    )
    store.set_origin(inner, store.ORIGIN_USER)
    inner_ref = store.occurrence_of(inner)
    root = store.append_meaning(
        _meaning(child=inner_ref, scope={"speaker": "outer"}), kind="observation"
    )
    root_ref = store.occurrence_of(root)
    orphan = store.append_meaning(_meaning(), trust=.7)
    store.set_origin(orphan, store.ORIGIN_USER)

    assert store.clear_origin(store.ORIGIN_USER) == 1
    assert len(store) == 3
    for reference in (leaf_ref, inner_ref, root_ref):
        assert _index(store, reference) is not None
    kept = store.row(_index(store, leaf_ref))
    assert kept["kind"] == "unverified" and kept["trust"] == 0
    assert kept["text"] == "reported content"
    assert kept["meaning"].scope == (("quote", "inner"),)
    assert store.meaning_of(_index(store, inner_ref)).mode == "interrogative"
    assert store.clear_origin(store.ORIGIN_USER) == 0


def test_external_live_root_pins_an_occurrence_through_origin_replacement():
    store = TernaryTruthStore(6, capacity=3)
    row = store.append_meaning(_meaning(), trust=1)
    store.set_origin(row, store.ORIGIN_USER)
    ref = store.occurrence_of(row)
    assert store.clear_origin(store.ORIGIN_USER, retained_occurrences=(ref,)) == 0
    assert store.row(0)["occurrence"] == ref
    assert store.row(0)["kind"] == "unverified"
    assert store.clear_origin(store.ORIGIN_USER) == 1


def test_missing_local_constituent_rejects_append_before_owner_mutation():
    store = TernaryTruthStore(6, capacity=3)
    row = store.append_meaning(_meaning())
    prefix = store.occurrence_of(row)[:2]
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="occurrence|constituent|reference"):
        store.append_meaning(_meaning(child=prefix + (99,)))
    _unchanged(store, before)


def test_repeated_edges_keep_role_order_without_evidence_inheritance():
    store = TernaryTruthStore(6, capacity=4)
    row = store.append_meaning(
        _meaning(mode="interrogative", scope={"quote": True}), kind="question"
    )
    child = store.occurrence_of(row)
    root_row = store.append_meaning(
        _meaning(child=child, twice=True), kind="fact", trust=.9
    )
    root = store.occurrence_of(root_row)
    view = store.read_structure(root, max_nodes=2, max_depth=1, max_records=4)
    assert view["incomplete"] == ()
    assert view["occurrences"] == (root, child)
    assert view["edges"] == ((root, 0, child), (root, 2, child))
    assert view["records_scanned"] == 2
    assert view["rows"][1]["kind"] == "question"
    assert view["rows"][1]["trust"] == 0
    assert view["rows"][1]["meaning"].scope == (("quote", True),)


@pytest.mark.parametrize(
    "limit,reason",
    (({"max_nodes": 1}, "node_limit"), ({"max_depth": 0}, "depth_limit"),
     ({"max_records": 1}, "record_limit")),
)
def test_nested_read_limits_are_separate_and_explicit(limit, reason):
    store = TernaryTruthStore(6, capacity=4)
    child = store.occurrence_of(store.append_meaning(_meaning()))
    root = store.occurrence_of(store.append_meaning(_meaning(child=child)))
    before = copy.deepcopy(store.state_dict())
    view = store.read_structure(
        root, **dict({"max_nodes": 4, "max_depth": 4, "max_records": 4}, **limit)
    )
    assert reason in view["incomplete"]
    _unchanged(store, before)


def test_retained_structure_and_reference_identity_survive_checkpoint_restore():
    store = TernaryTruthStore(6, capacity=4)
    child = store.occurrence_of(store.append_meaning(_meaning()))
    store.set_origin(0, store.ORIGIN_USER)
    root = store.occurrence_of(store.append_meaning(_meaning(child=child)))
    store.clear_origin(store.ORIGIN_USER)
    restored = TernaryTruthStore(6, capacity=4)
    restored.load_state_dict(copy.deepcopy(store.state_dict()))
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    view = restored.read_structure(root)
    assert view["incomplete"] == () and view["occurrences"] == (root, child)
    assert view["rows"][1]["kind"] == "unverified"
    assert not view["rows"][0]["meaning"].roles.requires_grad


def _boundary_meaning(ref=None):
    return ConceptualMeaning(
        torch.eye(5)[:3], torch.ones(3, dtype=torch.bool),
        role_refs=(None, None, ref),
    )


def _world():
    store = TernaryTruthStore(5, capacity=5)
    child = store.occurrence_of(store.append_meaning(_boundary_meaning(), kind="question"))
    root = store.occurrence_of(store.append_meaning(_boundary_meaning(child), kind="observation"))
    return store, child, root


def _corrupt_child(store, child, reference):
    identifier = child[2]
    store._semantic_rows[identifier]["role_refs"] = (None, None, reference)
    store.metadata_required[0] = True
    store.semantic_fingerprint[0] = store.semantic_fingerprint.new_tensor(
        store._context_fingerprint(store._semantic_rows[identifier], store.text_of(0))
    )


@pytest.mark.parametrize("corruption", ("cycle", "dangling"))
def test_structural_restore_rejects_invalid_local_links_atomically(corruption):
    store, child, root = _world()
    good = copy.deepcopy(store.semantic_extras())
    wrong = root if corruption == "cycle" else child[:2] + (999,)
    _corrupt_child(store, child, wrong)
    saved, extras = copy.deepcopy(store.state_dict()), copy.deepcopy(store.semantic_extras())
    restored = TernaryTruthStore(5, capacity=5)
    restored.load_state_dict(saved)
    before_context = copy.deepcopy(restored._semantic_rows)
    before_texts = list(restored._texts)
    with pytest.raises(ValueError, match="cycle|unavailable|constituent"):
        restored.load_semantic_extras(extras)
    assert restored._semantic_rows == before_context and restored._texts == before_texts
    assert good["records"][0]["context"]["role_refs"] == (None, None, None)


def test_cycle_is_reported_without_unbounded_read_or_evidence_transfer():
    store, child, root = _world()
    _corrupt_child(store, child, root)
    view = store.read_structure(root, max_nodes=2, max_depth=2, max_records=2)
    assert view["incomplete"] == ("cycle",)
    assert len(view["rows"]) == 2 and view["records_scanned"] == 2
    assert view["rows"][1]["kind"] == "question"


def test_invalid_survivor_closure_cannot_partly_withdraw_evidence():
    store, child, root = _world()
    store.set_origin(0, store.ORIGIN_USER)
    _corrupt_child(store, child, root)
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="cycle|constituent"):
        store.clear_origin(store.ORIGIN_USER)
    _unchanged(store, before)


def test_derived_structure_read_does_not_expose_mutable_store_slots():
    store, child, root = _world()
    before = store.slots.clone()
    view = store.read_structure(root)
    view["rows"][0]["np1"].zero_()
    view["rows"][0]["meaning"].roles.zero_()
    torch.testing.assert_close(store.slots, before)


def test_unavailable_owner_and_retired_reference_have_explicit_incomplete_results():
    store, child, root = _world()
    other = TernaryTruthStore(5, capacity=5)
    foreign = other.occurrence_of(other.append_meaning(_boundary_meaning()))
    assert store.read_structure(foreign)["incomplete"] == ("unavailable_owner",)
    store.reset()
    replacement = store.occurrence_of(store.append_meaning(_boundary_meaning()))
    assert replacement not in (child, root)
    assert store.read_structure(root)["incomplete"] == ("unavailable_reference",)


@pytest.mark.parametrize("field", ("bindings", "scope"))
def test_withdrawal_preserves_occurrences_in_retained_semantic_context(field):
    store = TernaryTruthStore(6, capacity=4)
    child = store.append_meaning(_meaning(), trust=.8)
    store.set_origin(child, store.ORIGIN_USER)
    ref = store.occurrence_of(child)
    outer = store.append_meaning(
        _meaning(**{field: {"referent": ref}}), kind="observation"
    )
    root = store.occurrence_of(outer)
    assert store.clear_origin(store.ORIGIN_USER) == 0
    assert store.occurrence_of(0) == ref
    assert store.row(0)["trust"] == 0 and store.row(0)["kind"] == "unverified"
    assert store.occurrence_of(1) == root
    assert getattr(store.meaning_of(1), field) == (("referent", ref),)


@pytest.mark.parametrize("field", ("bindings", "scope"))
def test_missing_local_context_reference_rejects_append_before_mutation(field):
    store = TernaryTruthStore(6, capacity=4)
    row = store.append_meaning(_meaning())
    missing = store.occurrence_of(row)[:2] + (99,)
    before = copy.deepcopy(store.state_dict())
    with pytest.raises(ValueError, match="reference|occurrence|context"):
        store.append_meaning(_meaning(**{field: {"referent": missing}}))
    _unchanged(store, before)
