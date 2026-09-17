"""Evidence ingestion must not turn prediction content into asserted truth."""
import pytest
import torch

from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner, UNKNOWN
from thinking import ThinkingKernel, Testimony


@pytest.mark.parametrize("value", [torch.ones(6), "an unparsed answer", float("nan")])
def test_nontruth_testimony_cannot_certify_a_referent(value):
    store = TernaryTruthStore(6, capacity=8)
    kernel = ThinkingKernel(TruthGroundedReasoner(store=store))
    testimony = Testimony(proposition=torch.eye(6)[0], value=value,
                          source="forecast", source_trust=.9)
    assert kernel.incorporate(testimony) == -1
    assert len(store) == 0


def test_legacy_part_testimony_without_a_vp_stays_unverified():
    left, right = torch.eye(6)[:2]
    store = TernaryTruthStore(6, capacity=8)
    kernel = ThinkingKernel(TruthGroundedReasoner(store=store))
    row = kernel.incorporate(Testimony(
        proposition=QuerySpec.from_surface("isPart", left, right),
        value=1, source="legacy part witness", source_trust=.8))
    assert row >= 0
    assert store.row(row)["kind"] == "unverified"


def test_legacy_materialized_relation_without_a_vp_stays_unverified():
    left, right = torch.eye(6)[:2]
    store = TernaryTruthStore(6, capacity=8)
    row = TruthGroundedReasoner(store=store).materialize(left, right, .8)
    assert row >= 0
    assert store.row(row)["kind"] == "unverified"


def test_scalar_prediction_is_identified_and_cannot_become_testimony_truth():
    from thinking import Frame
    store = TernaryTruthStore(6, capacity=8)
    kernel = ThinkingKernel(TruthGroundedReasoner(store=store))
    kernel.register_addressee("forecast", lambda _target: .95,
                              source_trust=1, evidence_kind="estimate")
    spec = QuerySpec.from_surface("exist", torch.eye(6)[0])
    result = kernel.query("forecast", spec)
    assert result.evidence_kind == "estimate"
    assert kernel.incorporate(result) == -1
    frame = Frame(spec, bindings={"testimony": [result]})
    interval = kernel._frame_interval(frame)
    assert interval.lower == interval.upper == 0
    assert len(store) == 0


def test_accepted_testimony_preserves_the_full_description_and_source():
    from Meaning import ConceptualMeaning
    description = ConceptualMeaning.from_payload(
        torch.eye(6)[:3], depth=3, layout="infix", scope={"where": ("sym", 5)})
    store = TernaryTruthStore(6, capacity=8)
    reasoner = TruthGroundedReasoner(store=store)
    kernel = ThinkingKernel(reasoner)
    request = QuerySpec.from_surface("exist", description)
    row = kernel.incorporate(Testimony(proposition=request, value=-1,
                                      source="named witness", source_trust=.8))
    assert row >= 0
    result = reasoner.evaluate(request)
    assert result["support_false"] == pytest.approx(.8)
    assert result["candidates"][0]["text"] == "named witness"
    unscoped = ConceptualMeaning.from_payload(torch.eye(6)[:3], depth=3, layout="infix")
    assert reasoner.evaluate(QuerySpec.from_surface("exist", unscoped))["posture"] == UNKNOWN
