"""PartOf must obtain evidence from the conceptual taxonomy."""
import torch

from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner, UNKNOWN


def test_partof_is_not_established_by_vector_containment():
    part = torch.tensor([1., 0., 0., 0.])
    whole = torch.ones(4)
    result = TruthGroundedReasoner(store=TernaryTruthStore(4)).evaluate(
        QuerySpec.from_surface("isPart", part, whole))
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == 0


def test_partof_is_not_established_by_a_world_relation_row():
    part, whole = torch.eye(4)[:2]
    store = TernaryTruthStore(4)
    store.append_relation(part, torch.eye(4)[2], whole,
                          rel_type=store.REL_PARTOF, trust=.9)
    result = TruthGroundedReasoner(store=store).evaluate(
        QuerySpec.from_surface("isPart", part, whole))
    assert result["posture"] == UNKNOWN
    assert result["support_true"] == 0
