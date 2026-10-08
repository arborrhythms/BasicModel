"""Complete descriptions, evidence provenance, and checkpoint ownership."""
from types import SimpleNamespace

import pytest
import torch

from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner, TRUE, UNKNOWN, BOTH


def _meaning(width=6, **metadata):
    from Meaning import ConceptualMeaning
    return ConceptualMeaning.from_payload(
        torch.eye(width)[:3], depth=3, layout="infix", **metadata)


def _evaluate(store, description):
    return TruthGroundedReasoner(store=store).evaluate(
        QuerySpec.from_surface("exist", description))


@pytest.mark.parametrize("tree,depth", [('cat', 1), (('part', 'cats', 'animals'), 3)])
@pytest.mark.parametrize("path", ["pending", "packed"])
def test_actual_observation_writers_keep_roles_without_asserting_world_facts(monkeypatch, tree, depth, path):
    from Models import BasicModel
    from reading_fixtures import finish_reading
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    entry = f.program(tree)
    clause = finish_reading(f.language, entry, registry=f.registry)
    from reading_fixtures import commit_reading
    model, store = f.model, f.store
    commit_reading(f.language, f.registry, entry, store, trust=.95, owner=model)
    assert int(store.rel_type[0]) == (store.REL_NONE if depth == 1 else store.REL_PARTOF)
    torch.testing.assert_close(store.meaning_of(0).roles[store.role_mask[0]], clause.slots)
    assert store.row(0).get("kind") == ("question" if depth == 3 else "observation")
    result = _evaluate(store, store.meaning_of(0))
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == result["support_false"] == 0


def test_legacy_checkpoint_does_not_invent_fact_status_for_conversation_rows():
    source = TernaryTruthStore(6, capacity=8)
    idea = torch.eye(6)[0]
    source.append_idea(idea, trust=.95, sentence_index=len(source))
    provisioned = source.append_idea(idea, trust=.6, sentence_index=len(source))
    source.set_origin(provisioned, source.ORIGIN_PROVISIONED)
    old_keys = {"slots", "rel_type", "timestamp", "trust", "count", "origin", "_next_ts"}
    old_state = {k: v.clone() for k, v in source.state_dict().items() if k in old_keys}
    restored = TernaryTruthStore(6, capacity=8)
    with pytest.warns(UserWarning, match='re-provisioned'):
        restored.load_state_dict(old_state, strict=True)
    assert len(restored) == 1
    assert restored.row(0)['origin'] == source.ORIGIN_CONVERSATION
    result = _evaluate(restored, idea)
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == result["support_false"] == 0
    assert result["candidates"] == []


@pytest.mark.parametrize("field", ["bindings", "scope", "role_refs"])
def test_full_description_metadata_changes_which_referent_is_requested(field):
    initial = {
        "bindings": {"subject": ("sym", 7)},
        "scope": {"where": ("sym", 11)},
        "role_refs": (("sym", 1), ("sym", 2), ("sym", 3)),
    }
    changed = {
        "bindings": {"subject": ("sym", 8)},
        "scope": {"where": ("sym", 12)},
        "role_refs": (("sym", 1), ("sym", 2), ("sym", 4)),
    }
    description = _meaning(**initial)
    store = TernaryTruthStore(6, capacity=8)
    store.append_meaning(description, kind="fact", trust=.85, sentence_index=len(store))
    assert _evaluate(store, description)["posture"] == TRUE
    initial[field] = changed[field]
    assert _evaluate(store, _meaning(**initial))["posture"] == UNKNOWN


@pytest.mark.parametrize("kind,mode", [
    ("question", "interrogative"),
    ("estimate", "assertive"),
    ("observation", "unspecified"),
])
def test_unaccepted_records_cannot_certify_their_own_referents(kind, mode):
    store = TernaryTruthStore(6, capacity=8)
    store.append_meaning(_meaning(mode=mode), kind=kind, trust=1.0, sentence_index=len(store))
    result = _evaluate(store, _meaning())
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == result["support_false"] == 0


def test_question_cannot_be_written_as_accepted_fact():
    store = TernaryTruthStore(6, capacity=8)
    with pytest.raises(ValueError, match="interrogative|question"):
        store.append_meaning(_meaning(mode="interrogative"), kind="fact", trust=1, sentence_index=len(store))
    assert len(store) == 0


def test_conflict_degrees_and_stable_provenance_survive_compaction_and_reset():
    store = TernaryTruthStore(6, capacity=8)
    first = store.append_meaning(_meaning(), kind="fact", trust=.8, sentence_index=len(store))
    second = store.append_meaning(_meaning(), kind="fact", trust=-.9, sentence_index=len(store))
    first_ref, second_ref = (store.row(i)["occurrence"] for i in (first, second))
    result = _evaluate(store, _meaning())
    assert result["posture"] == BOTH
    assert {c["occurrence"] for c in result["candidates"]} == {first_ref, second_ref}
    store.set_origin(first, store.ORIGIN_USER)
    store.clear_origin(store.ORIGIN_USER)
    assert store.row(0)["occurrence"] == second_ref
    assert _evaluate(store, _meaning())["support_false"] == pytest.approx(.9)
    store.reset()
    replacement = store.append_meaning(_meaning(), kind="fact", trust=.6,
                                       document_key='replacement', sentence_index=0)
    assert store.row(replacement)["occurrence"] not in (first_ref, second_ref)


def test_repeated_evidence_does_not_inflate_partial_support():
    store = TernaryTruthStore(6, capacity=8)
    for _ in range(4):
        store.append_meaning(_meaning(), kind="fact", trust=.4, sentence_index=len(store))
    result = _evaluate(store, _meaning())
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == pytest.approx(.4)
    assert result["support_false"] == 0


def test_live_description_owns_values_but_fact_storage_detaches():
    from Meaning import ConceptualMeaning
    source = torch.eye(6)[:3].clone().requires_grad_()
    description = ConceptualMeaning.from_payload(source, depth=3, layout="infix")
    gradient = torch.autograd.grad(description.roles.square().sum(), source)[0]
    torch.testing.assert_close(gradient, 2 * source)
    store = TernaryTruthStore(6, capacity=8)
    row = store.append_meaning(description, kind="fact", trust=.8, sentence_index=len(store))
    with torch.no_grad():
        source.zero_()
    torch.testing.assert_close(description.roles, torch.eye(6)[:3])
    stored = store.row(row)["meaning"]
    assert not stored.roles.requires_grad
    torch.testing.assert_close(stored.roles, description.roles)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_explicit_stm_layout_matches_predictor_adapter_and_retains_all_roles(depth):
    from Meaning import ConceptualMeaning
    from Layers import BracketExpectation
    payload = torch.eye(6)[:depth]
    meaning = ConceptualMeaning.from_payload(payload, depth=depth, layout="stm")
    expected, mask = BracketExpectation._canonical_meaning(
        SimpleNamespace(concept_dim=6), payload, depth, "stm")
    torch.testing.assert_close(meaning.roles, expected)
    torch.testing.assert_close(meaning.role_mask, mask)


def test_store_checkpoint_keeps_scoped_evidence_and_rejects_missing_sidecar():
    source = TernaryTruthStore(6, capacity=8)
    description = _meaning(scope={"where": ("sym", 11)}, bindings={"x": ("sym", 7)})
    row = source.append_meaning(description, kind="fact", trust=.85, sentence_index=len(source))
    source.set_origin(row, source.ORIGIN_USER, text="scoped witness")
    state = {k: v.clone() for k, v in source.state_dict().items()}
    extras = source.semantic_extras()
    restored = TernaryTruthStore(6, capacity=8)
    restored.load_state_dict(state, strict=True)
    # Required semantic context cannot be silently replaced by empty scope.
    incomplete = _evaluate(restored, _meaning())
    assert incomplete["posture"] == UNKNOWN
    assert incomplete["incomplete_evidence"]
    restored.load_semantic_extras(extras)
    result = _evaluate(restored, description)
    assert result["posture"] == TRUE
    assert result["support_true"] == pytest.approx(.85)
    assert result["candidates"][0]["occurrence"] == source.row(row)["occurrence"]
    assert result["candidates"][0]["text"] == "scoped witness"
    assert _evaluate(restored, _meaning())["posture"] == UNKNOWN


def test_actual_model_checkpoint_preserves_scoped_fact_identity(tmp_path):
    from test_ltm_consolidation import _make_model, _ON_CONFIG
    source = _make_model(_ON_CONFIG, stateless=False)
    restored = None
    try:
        store = source.symbolSpace.ltm_store
        store.reset()
        description = _meaning(width=store.nDim,
                               bindings={"x": ("sym", 7)},
                               scope={"where": ("sym", 11)})
        row = store.append_meaning(description, kind="fact", trust=-.8, sentence_index=len(store))
        store.set_origin(row, store.ORIGIN_USER, text="negative scoped witness")
        expected_ref = store.row(row)["occurrence"]
        path = str(tmp_path / "scoped-fact.pt")
        source.save_weights(path)
        restored = _make_model(_ON_CONFIG, stateless=False)
        assert restored.load_weights(path, strict=True, require_match=True)
        result = _evaluate(restored.symbolSpace.ltm_store, description)
        assert result["support_false"] == pytest.approx(.8)
        assert result["candidates"][0]["occurrence"] == expected_ref
        assert result["candidates"][0]["text"] == "negative scoped witness"
        assert _evaluate(restored.symbolSpace.ltm_store,
                         _meaning(width=store.nDim))["posture"] == UNKNOWN
    finally:
        source.End()
        if restored is not None:
            restored.End()


def test_checked_reader_keeps_both_sides_of_exist_evidence():
    store = TernaryTruthStore(6, capacity=8)
    store.append_meaning(_meaning(), kind="fact", trust=.8, sentence_index=len(store))
    store.append_meaning(_meaning(), kind="fact", trust=-.9, sentence_index=len(store))
    reasoner = TruthGroundedReasoner(store=store)
    result = reasoner.evaluate(QuerySpec.from_surface("exist", _meaning()))
    assert result['posture'] == 'BOTH'
    assert -result['support_false'] == pytest.approx(-.9)
    assert result['support_true'] == pytest.approx(.8)
    assert len(result['candidates']) == 2
