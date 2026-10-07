"""Tensor and metadata restore must describe the same semantic occurrences."""
from copy import deepcopy

import pytest
import torch

from Meaning import ConceptualMeaning
from Layers import TernaryTruthStore


@pytest.mark.parametrize("text", [None, "a scoped assertion"])
@pytest.mark.parametrize("mutation", ["erase", "replace"])
def test_restored_sidecar_cannot_erase_or_replace_recorded_scope(text, mutation):
    source = TernaryTruthStore(6, capacity=8)
    meaning = ConceptualMeaning.from_payload(
        torch.eye(6)[:3], depth=3, layout="infix", scope={"where": ("sym", 11)})
    row = source.append_meaning(meaning, trust=.8)
    if text is not None:
        source.set_origin(row, source.ORIGIN_USER, text=text)
    extras = deepcopy(source.semantic_extras())
    extras["records"][0]["context"]["scope"] = (() if mutation == "erase"
                                                 else (("where", ("sym", 12)),))
    restored = TernaryTruthStore(6, capacity=8)
    restored.load_state_dict(source.state_dict(), strict=True)
    with pytest.raises(ValueError, match="semantic.*(fingerprint|content|scope)"):
        restored.load_semantic_extras(extras)
    assert restored.meaning_of(row) is None


def test_restored_sidecar_cannot_swap_contexts_between_existing_occurrences():
    source = TernaryTruthStore(6, capacity=8)
    for concept in (11, 12):
        meaning = ConceptualMeaning.from_payload(
            torch.eye(6)[:3], depth=3, layout="infix", scope={"where": ("sym", concept)})
        source.append_meaning(meaning, trust=.8)
    extras = deepcopy(source.semantic_extras())
    records = extras["records"]
    records[0]["context"], records[1]["context"] = records[1]["context"], records[0]["context"]
    restored = TernaryTruthStore(6, capacity=8)
    restored.load_state_dict(source.state_dict(), strict=True)
    with pytest.raises(ValueError, match="semantic.*(fingerprint|content|scope)"):
        restored.load_semantic_extras(extras)


def test_source_update_cannot_certify_missing_semantic_context():
    source = TernaryTruthStore(6, capacity=8)
    meaning = ConceptualMeaning.from_payload(
        torch.eye(6)[:3], depth=3, layout="infix", scope={"where": ("sym", 11)})
    source.append_meaning(meaning, trust=.8)
    restored = TernaryTruthStore(6, capacity=8)
    restored.load_state_dict(source.state_dict(), strict=True)
    with pytest.raises(ValueError, match="missing semantic"):
        restored.set_origin(0, source.ORIGIN_USER, text="new source")
    assert restored.meaning_of(0) is None
