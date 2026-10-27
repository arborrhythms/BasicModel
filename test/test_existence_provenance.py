"""The public legacy kernel must retain evidence beyond its lookup leaf."""
from collections.abc import Mapping

import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import LTM0
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


def test_checked_result_retains_matching_fact_occurrences_and_both_degrees():
    meaning = ConceptualMeaning.from_payload(torch.eye(6)[:3].flatten()[None], depth=1, layout="infix")
    store = LTM0(18, capacity=8, percept_width=0)
    rows = [store.append_meaning(meaning, trust=degree, sentence_index=len(store)) for degree in (.8, -.9)]
    expected = {store.row(i)["occurrence"] for i in rows}
    result = TruthGroundedReasoner(store=store).evaluate(
        QuerySpec.from_surface("exist", meaning))
    assert result['posture'] == CONFLICTING
    assert -result['support_false'] == pytest.approx(-.9)
    assert result['support_true'] == pytest.approx(.8)
    assert _occurrences(result['candidates']) == expected


def test_missing_semantic_context_remains_visible_in_the_checked_result():
    meaning = ConceptualMeaning.from_payload(torch.eye(6)[:3].flatten()[None], depth=1, layout="infix",
                                             scope={"where": ("sym", 11)})
    store = LTM0(18, capacity=8, percept_width=0)
    store.append_meaning(meaning, trust=.8, sentence_index=len(store))
    restored = LTM0(18, capacity=8, percept_width=0)
    restored.load_state_dict(store.state_dict(), strict=True)
    result = TruthGroundedReasoner(store=restored).evaluate(
        QuerySpec.from_surface("exist", meaning))
    assert result['posture'] == UNKNOWN
    assert result["incomplete_evidence"]


def test_consolidated_prediction_reads_the_one_idea_presence():
    from types import SimpleNamespace
    from Layers import SymbolExpectation
    store = LTM0(18, capacity=8, percept_width=0)
    roles = torch.eye(6)[:3].flatten()[None]
    store.append_meaning(ConceptualMeaning.from_payload(roles, depth=1, layout='infix'),
                         trust=.8, stream=0)
    chain = SymbolExpectation.get_stm_chain(SimpleNamespace(_ltm_store=store))
    depth, payload, trust = chain[0]
    assert depth == 1
    torch.testing.assert_close(payload, roles)
    assert trust == pytest.approx(.8)


@pytest.mark.parametrize("missing", ["record_kind", "metadata_required"])
def test_partial_new_checkpoint_cannot_forget_required_scope(missing):
    meaning = ConceptualMeaning.from_payload(torch.eye(6)[:3].flatten()[None], depth=1, layout="infix",
                                             scope={"where": ("sym", 11)})
    source = LTM0(18, capacity=8, percept_width=0)
    source.append_meaning(meaning, trust=.8, sentence_index=len(source))
    state = dict(source.state_dict())
    del state[missing]
    restored = LTM0(18, capacity=8, percept_width=0)
    with pytest.raises(ValueError, match="layout checkpoint column"):
        restored.load_state_dict(state, strict=False)
