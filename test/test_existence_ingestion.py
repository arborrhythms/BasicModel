"""Only the explicit evidence owner can admit assertions after kernel retirement."""
import pytest
import torch
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from reasoning import QuerySpec, TruthGroundedReasoner, UNKNOWN

def test_legacy_materialized_relation_without_a_vp_stays_unverified():
    left, right = torch.eye(6)[:2]
    store = TernaryTruthStore(6, capacity=8)
    row = TruthGroundedReasoner(store=store).legacy_materialize(left, right, .8)
    assert row >= 0
    assert store.row(row)["kind"] == "unverified"


def test_prediction_and_observation_occurrences_cannot_certify_their_content():
    description = ConceptualMeaning.from_description(torch.eye(6)[:3])
    store = TernaryTruthStore(6, capacity=8)
    for kind in ('estimate', 'observation', 'unverified'):
        store.append_meaning(description, kind=kind, trust=1)
    result = TruthGroundedReasoner(store=store).evaluate(QuerySpec.from_surface('exist', description))
    assert result['posture'] == UNKNOWN


def test_explicit_fact_admission_preserves_full_description_and_source():
    description = ConceptualMeaning.from_payload(torch.eye(6)[:3], depth=3,
        layout='infix', scope={'where': ('sym', 5)})
    store = TernaryTruthStore(6, capacity=8)
    row = store.append_meaning(description, kind='fact', trust=-.8)
    store.set_origin(row, store.ORIGIN_USER, text='named witness')
    reasoner = TruthGroundedReasoner(store=store)
    result = reasoner.evaluate(QuerySpec.from_surface('exist', description))
    assert result['support_false'] == pytest.approx(.8)
    assert result['candidates'][0]['text'] == 'named witness'
    unscoped = ConceptualMeaning.from_description(torch.eye(6)[:3])
    assert reasoner.evaluate(QuerySpec.from_surface('exist', unscoped))['posture'] == UNKNOWN
