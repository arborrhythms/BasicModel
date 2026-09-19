"""Reviewer probe for integrated plan §8.7 estimate/observation ownership."""

import copy
from types import SimpleNamespace

import torch

from Layers import InterSentenceLayer, TernaryTruthStore
from Meaning import ConceptualMeaning
from Models import _append_observed_meaning
from reasoning import QuerySpec, TruthGroundedReasoner, UNKNOWN


def _meaning(value, *, scope=()):
    roles = torch.tensor([
        [value, 0.0, 0.0, 0.0],
        [0.0, value, 0.0, 0.0],
        [0.0, 0.0, value, 0.0],
    ])
    return ConceptualMeaning(
        roles, torch.tensor([True, True, True]), mode="unspecified",
        scope=scope)


def _discourse():
    torch.manual_seed(91)
    return InterSentenceLayer(
        n_symbols=4, max_depth=4, n_dim=4, concept_dim=4,
        batch=1, expectation_scope="structured")


def _observe(discourse, meaning, document):
    discourse.predict_and_observe_stm_end_state(
        [3], [meaning.roles], documents=[document], layout="infix",
        role_masks=[meaning.role_mask])


def test_retained_estimate_links_its_external_observation_and_restores():
    """A prediction is a durable estimate, never extra observed evidence.

    The first external sentence starts the stream.  The second is the target
    for an estimate formed from that first retained occurrence.  Both records
    must remain distinct, cross-linked, scoped to the stream/document, and
    recoverable through the store's checkpoint sidecar.
    """
    store = TernaryTruthStore(4, capacity=8)
    discourse = _discourse()
    discourse._ltm_store = store
    document = "conversation-17"

    source = _meaning(1.0, scope={"where": ("sym", 3)})
    _observe(discourse, source, document)
    first = _append_observed_meaning(store, source.roles, 3, meaning=source)
    source_ref = store.row(first)["occurrence"]
    discourse.bind_observation_occurrence(0, source_ref)

    observed = _meaning(2.0, scope={"where": ("sym", 4)})
    _observe(discourse, observed, document)
    comparison = discourse.last_expectation_comparison(0)
    assert comparison is not None
    second = _append_observed_meaning(
        store, observed.roles, 3, meaning=observed, expectation=comparison)
    discourse.bind_observation_occurrence(0, store.row(second)["occurrence"])

    assert len(store) == 3
    estimate, observation = store.row(1), store.row(second)
    assert estimate["kind"] == "estimate"
    assert observation["kind"] == "observation"
    assert estimate["expectation"]["source_occurrences"] == (source_ref,)
    assert estimate["expectation"]["document"] == document
    assert estimate["expectation"]["intended_occurrence"] == observation["occurrence"]
    assert observation["expectation"]["estimate_occurrence"] == estimate["occurrence"]
    assert not estimate["meaning"].roles.requires_grad

    paired = store.expectation_pair(second)
    assert paired["estimate_occurrence"] == estimate["occurrence"]
    assert paired["observation_occurrence"] == observation["occurrence"]
    torch.testing.assert_close(
        paired["residual"],
        torch.where(observed.role_mask[:, None],
                    observed.roles - estimate["meaning"].roles,
                    torch.zeros_like(observed.roles)))
    # A high-confidence prediction remains an estimate, never self-support.
    result = TruthGroundedReasoner(store=store).evaluate(
        QuerySpec.from_surface("exist", estimate["meaning"]))
    assert result["posture"] == UNKNOWN

    restored = TernaryTruthStore(4, capacity=8)
    restored.load_state_dict(copy.deepcopy(store.state_dict()), strict=True)
    restored.load_semantic_extras(copy.deepcopy(store.semantic_extras()))
    replayed = restored.expectation_pair(1)
    assert replayed["source_occurrences"] == (source_ref,)
    assert replayed["document"] == document
    assert replayed["intended_occurrence"] == observation["occurrence"]
    torch.testing.assert_close(replayed["residual"], paired["residual"])


def test_estimate_never_displaces_the_last_available_external_observation():
    """Finite LTM capacity favors an understood input over its forecast."""
    store = TernaryTruthStore(4, capacity=2)
    source = _meaning(1.0)
    source_row = store.append_meaning(source, kind="observation")
    estimate, observed = store.append_expectation_pair(
        _meaning(.5), _meaning(2.0),
        presence_logits=torch.tensor([1.0, 0.0, -1.0]),
        source_occurrences=(store.row(source_row)["occurrence"],),
        stream=("external", "capacity"), document="capacity")
    assert estimate == -1
    assert observed == 1
    assert [store.row(i)["kind"] for i in range(len(store))] == [
        "observation", "observation"]


def test_retained_estimates_do_not_enter_the_generic_ltm_chain():
    """A forecast is available to its residual policy, not ordinary LTM reads."""
    store = TernaryTruthStore(4, capacity=8)
    source = _meaning(1.0)
    source_row = store.append_meaning(source, kind="observation")
    estimate_row, observed_row = store.append_expectation_pair(
        _meaning(.5), _meaning(2.0),
        presence_logits=torch.tensor([1.0, 0.0, -1.0]),
        source_occurrences=(store.row(source_row)["occurrence"],),
        stream=("external", "chain"), document="chain")

    chain = InterSentenceLayer.get_stm_chain(
        SimpleNamespace(_ltm_store=store))
    assert estimate_row >= 0
    assert len(chain) == 2
    torch.testing.assert_close(chain[0][1], source.roles)
    torch.testing.assert_close(chain[1][1], _meaning(2.0).roles)
    assert store.row(observed_row)["kind"] == "observation"
