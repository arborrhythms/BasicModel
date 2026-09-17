"""Public lookup and trust updates preserve the complete evidence contract."""
import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import TernaryTruthStore
from reasoning import TruthGroundedReasoner
from thinking import ThinkingKernel


@pytest.mark.parametrize("structured", [True, False])
def test_public_kernel_accepts_complete_description_without_collapsing_roles(structured):
    roles = torch.eye(6)[:3]
    fact = ConceptualMeaning.from_description(roles)
    store = TernaryTruthStore(6, capacity=8)
    store.append_meaning(fact, trust=.8)
    kernel = ThinkingKernel(TruthGroundedReasoner(store=store))
    query = fact if structured else roles
    assert kernel.lookup(query).upper == pytest.approx(.8)
    altered = roles.clone()
    altered[1] = torch.eye(6)[3]
    other = ConceptualMeaning.from_description(altered) if structured else altered
    assert kernel.lookup(other).upper == 0


@pytest.mark.parametrize("trust", [float("nan"), float("inf"), -float("inf")])
def test_trust_update_rejects_nonfinite_degrees_without_mutating_evidence(trust):
    store = TernaryTruthStore(6, capacity=8)
    row = store.append_idea(torch.eye(6)[0], trust=.4)
    with pytest.raises(ValueError, match="finite"):
        store.set_trust(row, trust)
    assert float(store.trust[row]) == pytest.approx(.4)
