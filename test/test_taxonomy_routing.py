"""PartOf must obtain evidence from the conceptual taxonomy."""
from Memory import STM0
import torch

from Memory import LTM0
from reasoning import QuerySpec, TruthGroundedReasoner, UNKNOWN


def test_partof_is_not_established_by_vector_containment():
    part = torch.tensor([1., 0., 0., 0.])
    whole = torch.ones(4)
    result = TruthGroundedReasoner(store=STM0(4, capacity=1024, relation_capacity=8192)).evaluate(
        QuerySpec.from_surface("isPart", part, whole))
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == 0


def test_partof_is_not_established_by_an_ltm1_description():
    part, whole = torch.eye(4)[:2]
    store = STM0(12, percept_width=0, capacity=1024, relation_capacity=8192)
    from Meaning import ConceptualMeaning
    store.append_meaning(ConceptualMeaning.from_payload(torch.stack((part, torch.eye(4)[2], whole)).flatten()[None],
        depth=1, layout='infix'),  trust=.9)
    result = TruthGroundedReasoner(store=store).evaluate(
        QuerySpec.from_surface("isPart", part, whole))
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == 0
