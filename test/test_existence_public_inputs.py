"""Public lookup and trust updates preserve the complete evidence contract."""
import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import TernaryTruthStore
from reasoning import TruthGroundedReasoner
from reasoning import QuerySpec


@pytest.mark.parametrize("structured", [True, False])
def test_checked_reader_accepts_complete_description_without_collapsing_roles(structured):
    roles = torch.eye(6)[:3]
    fact = ConceptualMeaning.from_description(roles)
    store = TernaryTruthStore(6, capacity=8)
    store.append_meaning(fact, trust=.8)
    reasoner = TruthGroundedReasoner(store=store)
    query = fact if structured else roles
    assert reasoner.evaluate(QuerySpec.from_surface('exist', query))['support_true'] == pytest.approx(.8)
    altered = roles.clone()
    altered[1] = torch.eye(6)[3]
    other = ConceptualMeaning.from_description(altered) if structured else altered
    assert reasoner.evaluate(QuerySpec.from_surface('exist', other))['support_true'] == 0


@pytest.mark.parametrize("trust", [float("nan"), float("inf"), -float("inf")])
def test_trust_update_rejects_nonfinite_degrees_without_mutating_evidence(trust):
    store = TernaryTruthStore(6, capacity=8)
    row = store.append_idea(torch.eye(6)[0], trust=.4)
    with pytest.raises(ValueError, match="finite"):
        store.set_trust(row, trust)
    assert float(store.trust[row]) == pytest.approx(.4)
