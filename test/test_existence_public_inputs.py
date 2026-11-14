"""Public lookup and trust updates preserve the complete evidence contract."""
from Memory import STM0
import pytest
import torch

from Meaning import ConceptualMeaning
from Memory import LTM0
from reasoning import TruthGroundedReasoner
from reasoning import QuerySpec




@pytest.mark.parametrize("trust", [float("nan"), float("inf"), -float("inf")])
def test_new_trust_rejects_nonfinite_degrees_without_mutating_evidence(trust):
    store = STM0(6, capacity=8, relation_capacity=8192)
    row = store.append_idea(torch.eye(6)[0], trust=.4)
    with pytest.raises(ValueError, match="finite"):
        store.append_idea(torch.eye(6)[0], trust=trust)
    assert float(store.trust[row]) == pytest.approx(.4)
