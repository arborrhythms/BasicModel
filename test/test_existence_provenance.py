"""The public legacy kernel must retain evidence beyond its lookup leaf."""
from collections.abc import Mapping

import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import TernaryTruthStore
from reasoning import QuerySpec, TruthGroundedReasoner
from thinking import ThinkingKernel, CONFLICTING, UNKNOWN


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


def test_kernel_result_retains_matching_fact_occurrences_and_both_degrees():
    meaning = ConceptualMeaning.from_payload(torch.eye(6)[:3], depth=3, layout="infix")
    store = TernaryTruthStore(6, capacity=8)
    rows = [store.append_meaning(meaning, trust=degree) for degree in (.8, -.9)]
    expected = {store.row(i)["occurrence"] for i in rows}
    result = ThinkingKernel(TruthGroundedReasoner(store=store)).run(
        QuerySpec.from_surface("exist", meaning))
    assert result.value == CONFLICTING
    assert result.interval.lower == pytest.approx(-.9)
    assert result.interval.upper == pytest.approx(.8)
    assert _occurrences(result.provenance) == expected


def test_missing_semantic_context_remains_visible_in_the_kernel_result():
    meaning = ConceptualMeaning.from_payload(torch.eye(6)[:3], depth=3, layout="infix",
                                             scope={"where": ("sym", 11)})
    store = TernaryTruthStore(6, capacity=8)
    store.append_meaning(meaning, trust=.8)
    restored = TernaryTruthStore(6, capacity=8)
    restored.load_state_dict(store.state_dict(), strict=True)
    result = ThinkingKernel(TruthGroundedReasoner(store=restored)).run(
        QuerySpec.from_surface("exist", meaning))
    assert result.value == UNKNOWN
    assert "incomplete_evidence" in repr(result.provenance)


def test_consolidated_legacy_chain_reads_the_actual_two_role_presence():
    from types import SimpleNamespace
    from Layers import InterSentenceLayer
    store = TernaryTruthStore(6, capacity=8)
    roles = torch.eye(6)[:2]
    store.append_relation(roles[0], roles[1], None, trust=.8)
    chain = InterSentenceLayer.get_stm_chain(SimpleNamespace(_ltm_store=store))
    depth, payload, trust = chain[0]
    assert depth == 2
    torch.testing.assert_close(payload, roles)
    assert trust == pytest.approx(.8)


@pytest.mark.parametrize("missing", ["record_kind", "metadata_required"])
def test_partial_new_checkpoint_cannot_forget_required_scope(missing):
    meaning = ConceptualMeaning.from_payload(torch.eye(6)[:3], depth=3, layout="infix",
                                             scope={"where": ("sym", 11)})
    source = TernaryTruthStore(6, capacity=8)
    source.append_meaning(meaning, trust=.8)
    state = dict(source.state_dict())
    del state[missing]
    restored = TernaryTruthStore(6, capacity=8)
    with pytest.raises(RuntimeError, match="semantic checkpoint"):
        restored.load_state_dict(state, strict=False)
