"""Only the explicit evidence owner can admit assertions after kernel retirement."""
from Memory import STM0
import pytest
import torch
from Memory import LTM0
from Meaning import ConceptualMeaning
from reasoning import QuerySpec, TruthGroundedReasoner, UNKNOWN

def test_part_without_supplied_evidence_stays_unverified():
    from part_fixtures import native_store, part
    left, right = torch.eye(6)[:2]
    store, _ = native_store(width=6)
    row = part(store, left, right, evidence=(0., 0.))
    assert store.row(row)['evidence'] == (0., 0.)




def test_explicit_fact_admission_preserves_full_description_and_trust():
    from existence_fixtures import read_claim
    description = ConceptualMeaning.from_payload(torch.eye(6)[:3].flatten()[None],depth=1,layout='infix')
    store = STM0(18,capacity=8,percept_width=0, relation_capacity=8192)
    row = store.append_meaning(description,trust=-.8)
    result = read_claim(store,row)
    assert result['support_false'] == pytest.approx(.8)
    torch.testing.assert_close(result['meaning'].roles,description.roles)
