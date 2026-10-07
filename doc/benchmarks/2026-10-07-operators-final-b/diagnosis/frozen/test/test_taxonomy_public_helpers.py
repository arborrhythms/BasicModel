"""The named PartOf helpers must share the public query evidence domain."""
from types import SimpleNamespace

import torch

from Layers import TernaryTruthStore
from reasoning import TruthGroundedReasoner
from test_cs_symbol_table import _cs


def test_public_part_helpers_cannot_accept_geometric_or_world_row_evidence():
    a, b = torch.eye(8)[:2]
    store = TernaryTruthStore(8)
    store.append_relation(a, torch.ones(8), b, rel_type=store.REL_PARTOF, trust=1)
    reasoner = TruthGroundedReasoner(store=store)
    assert reasoner.is_part_direct(a, torch.ones(8)) is None
    assert reasoner.is_part(a, b) == []
    assert reasoner.wholes(a) == reasoner.parts(b) == []


def test_public_part_helpers_follow_typed_links_without_world_materialization():
    cs = _cs()
    a, b, c = (("sym", cs.new_concept()) for _ in range(3))
    cs.add_whole(a[1], b)
    cs.add_whole(b[1], c)
    store = TernaryTruthStore(8)
    reasoner = TruthGroundedReasoner(SimpleNamespace(conceptualSpace=cs), store=store)
    assert reasoner.is_part_direct(a, b) == (1., "taxonomy")
    assert len(reasoner.is_part(a, c, materialize=True)[0]["path"]) == 2
    assert reasoner.wholes(a)[0]["reference"] == b
    assert reasoner.parts(c)[0]["reference"] == b
    assert len(store) == 0
