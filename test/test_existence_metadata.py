from Memory import STM0
from dataclasses import replace
"""Complete descriptions, evidence provenance, and checkpoint ownership."""
from types import SimpleNamespace

import pytest
import torch

from Memory import LTM0
from reasoning import QuerySpec, TruthGroundedReasoner, TRUE, UNKNOWN, BOTH


def _meaning(width=6, **metadata):
    from Meaning import ConceptualMeaning
    refs = metadata.get('role_refs', (None, None, None))
    if refs[0] is not None:
        metadata['bindings'] = {**metadata.get('bindings', {}), '_identity': refs[0]}
    return replace(ConceptualMeaning.from_description(torch.eye(width)[0]), **metadata)


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
    commit_reading(f.language, f.registry, entry, store, trust=0., owner=model)
    row = 0 if depth == 1 else ('sym', store.parts.find(f.noun('cats'), f.noun('animals')))
    meaning = store.meaning_of(row)
    if depth == 1:
        torch.testing.assert_close(meaning.roles[meaning.role_mask], clause.slots)
    else:
        assert store.row(row)['refs'].tolist() == [f.noun('cats'), f.noun('animals')]
    # All references are bound; (0, 0) is ignorance, not a free variable.
    assert store.row(row)["evidence"] == (0., 0.)
    from ThoughtReferences import open_slots
    assert not open_slots(meaning)
    result = _evaluate(store, meaning)
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == result["support_false"] == 0










def test_conflict_degrees_and_occurrences_survive_compaction_and_reset():
    from existence_fixtures import read_claim
    store = STM0(6, capacity=8, relation_capacity=8192)
    first = store.append_meaning(_meaning(), trust=.8)
    second = store.append_meaning(_meaning(), trust=-.9)
    first_ref, second_ref = (store.occurrence_of(i) for i in (first,second))
    assert read_claim(store,first)['support_true'] == pytest.approx(.8)
    assert read_claim(store,second)['support_false'] == pytest.approx(.9)
    store.compact_rows((second,))
    assert store.row(0)['occurrence'] == second_ref
    assert read_claim(store,0)['support_false'] == pytest.approx(.9)
    store.reset()
    replacement = store.append_meaning(_meaning(), trust=.6)
    assert store.row(replacement)['occurrence'] not in (first_ref,second_ref)


def test_repeated_evidence_does_not_mutate_stored_support():
    from existence_fixtures import read_claim
    store = STM0(6, capacity=8, relation_capacity=8192)
    for _ in range(4):
        row=store.append_meaning(_meaning(), trust=.4)
        assert read_claim(store,row)['support_true'] == pytest.approx(.4)
        assert read_claim(store,row)['support_false'] == 0.
    assert len(store) == 4


def test_live_description_owns_values_but_fact_storage_detaches():
    from Meaning import ConceptualMeaning
    source = torch.eye(6)[0].clone().requires_grad_()
    description = ConceptualMeaning.from_description(source)
    gradient = torch.autograd.grad(description.roles.square().sum(), source)[0]
    torch.testing.assert_close(gradient, 2 * source)
    store = STM0(6, capacity=8, relation_capacity=8192)
    row = store.append_meaning(description,  trust=.8)
    with torch.no_grad():
        source.zero_()
    torch.testing.assert_close(description.roles[0], torch.eye(6)[0])
    stored = store.row(row)["meaning"]
    assert not stored.roles.requires_grad
    torch.testing.assert_close(stored.roles, description.roles)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_explicit_stm_layout_matches_predictor_adapter_and_retains_all_roles(depth):
    from Meaning import ConceptualMeaning
    from Layers import SymbolExpectation
    payload = torch.eye(6)[:depth]
    meaning = ConceptualMeaning.from_payload(payload, depth=depth, layout="stm")
    expected, mask = SymbolExpectation._canonical_meaning(
        SimpleNamespace(concept_dim=6), payload, depth, "stm")
    torch.testing.assert_close(meaning.roles, expected)
    torch.testing.assert_close(meaning.role_mask, mask)


def test_store_checkpoint_keeps_signed_evidence_without_a_sidecar():
    from existence_fixtures import read_claim
    source = STM0(6, capacity=8, relation_capacity=8192)
    description = _meaning()
    row = source.append_meaning(description, trust=.85)
    state = {k:v.clone() for k,v in source.state_dict().items()}
    restored = STM0(6, capacity=8, relation_capacity=8192)
    restored.load_state_dict(state, strict=True)
    result = read_claim(restored,row)
    assert result['support_true'] == pytest.approx(.85)
    assert result['occurrence'] == source.occurrence_of(row)
    torch.testing.assert_close(result['meaning'].roles,description.roles)


def test_actual_model_checkpoint_preserves_signed_claim_identity(tmp_path):
    from test_ltm_consolidation import _make_model, _ON_CONFIG
    source = _make_model(_ON_CONFIG, stateless=False)
    restored = None
    try:
        store = source.symbolSpace.ltm_store
        store.reset()
        description = _meaning(width=store.nDim,
                               bindings={"x": ("sym", 7)},
                               scope={"where": ("sym", 11)})
        row = store.append_meaning(description,  trust=-.8)
        expected_ref = store.row(row)["occurrence"]
        description = store.meaning_of(row)
        path = str(tmp_path / "scoped-fact.pt")
        source.save_weights(path)
        restored = _make_model(_ON_CONFIG, stateless=False)
        assert restored.load_weights(path, strict=True, require_match=True)
        from existence_fixtures import read_claim
        result = read_claim(restored.symbolSpace.ltm_store, row)
        assert result["support_false"] == pytest.approx(.8)
        assert result["occurrence"] == expected_ref
        assert _evaluate(restored.symbolSpace.ltm_store,
                         _meaning(width=store.nDim))["posture"] == UNKNOWN
    finally:
        source.End()
        if restored is not None:
            restored.End()


def test_checked_reader_keeps_both_sides_of_addressed_evidence():
    from existence_fixtures import read_claim
    store = STM0(6, capacity=8, relation_capacity=8192)
    first=store.append_meaning(_meaning(),trust=.8)
    second=store.append_meaning(_meaning(),trust=-.9)
    positive,negative = read_claim(store,first),read_claim(store,second)
    assert -negative['support_false'] == pytest.approx(-.9)
    assert positive['support_true'] == pytest.approx(.8)
    assert len({positive['occurrence'],negative['occurrence']}) == 2
