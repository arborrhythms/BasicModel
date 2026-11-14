"""The public legacy kernel must retain evidence beyond its lookup leaf."""
from Memory import STM0
from collections.abc import Mapping

import pytest
import torch

from Meaning import ConceptualMeaning
from Memory import LTM0
from reasoning import QuerySpec, TruthGroundedReasoner
from reasoning import BOTH as CONFLICTING, UNKNOWN


def _occurrences(value):
    found = set()
    if isinstance(value, Mapping):
        if "occurrence" in value:
            found.add(value["occurrence"])
        for item in value.values():
            found.update(_occurrences(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            found.update(_occurrences(item))
    return found


def test_checked_result_retains_addressed_occurrences_and_both_degrees():
    from existence_fixtures import read_claim
    meaning = ConceptualMeaning.from_payload(torch.eye(6)[:3].flatten()[None],depth=1,layout='infix')
    store = STM0(18,capacity=8,percept_width=0, relation_capacity=8192)
    rows = [store.append_meaning(meaning,trust=degree) for degree in (.8,-.9)]
    expected = {store.occurrence_of(i) for i in rows}
    results = [read_claim(store,i) for i in rows]
    assert results[0]['support_true'] == pytest.approx(.8)
    assert -results[1]['support_false'] == pytest.approx(-.9)
    assert _occurrences(results) == expected


def test_consolidated_prediction_reads_the_one_idea_presence():
    from types import SimpleNamespace
    from Layers import SymbolExpectation
    store = STM0(18, capacity=8, percept_width=0, relation_capacity=8192)
    roles = torch.eye(6)[:3].flatten()[None]
    store.append_meaning(ConceptualMeaning.from_payload(roles, depth=1, layout='infix'),
                         trust=.8)
    chain = SymbolExpectation.get_stm_chain(SimpleNamespace(_ltm_store=store))
    depth, payload, trust = chain[0]
    assert depth == 1
    torch.testing.assert_close(payload, roles)
    assert trust == pytest.approx(.8)
