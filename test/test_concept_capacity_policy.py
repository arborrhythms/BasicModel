"""Capacity-safe admission for the aligned conceptual inventory."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import pytest
import torch

from Spaces import ConceptualSpace
from test_basicmodel import _populate_test_config


_D = 8


def _cs(n_vectors=16):
    n_slots = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=n_slots, nPercepts=n_slots, nConcepts=n_vectors,
        nSymbols=n_vectors, nWords=n_vectors, nOutput=n_vectors,
        nWhere=0, nWhen=0,
    )
    cs = ConceptualSpace(
        [n_slots, _D], [n_vectors, _D], [n_slots, _D])
    object.__setattr__(cs, "_concept_binding", "aligned")
    return cs


def _property_ws(spans=None):
    return types.SimpleNamespace(
        property_basis=True,
        nVectors=8,
        _staged_analysis_spans=spans,
        property_rows_for_bytes=lambda _value: (0,),
    )


def _allocator_snapshot(cs):
    alloc = cs._concept_allocator
    layer = alloc.layer()
    return {
        "next_id": int(alloc.next_id),
        "placement": dict(alloc.placement),
        "word_obj_meta": dict(alloc.word_obj_meta),
        "relate_idx": dict(alloc.relate_idx),
        "chain_idx": dict(alloc.chain_idx),
        "joint": dict(alloc.joint),
        "constituents": {
            key: list(value) for key, value in layer._constituents.items()
        },
        "W": cs.similarity_codebook.getW().detach().clone(),
    }


def _assert_snapshot_equal(before, after):
    torch.testing.assert_close(after.pop("W"), before.pop("W"))
    assert after == before


def test_explicit_word_triple_capacity_failure_is_atomic():
    cs = _cs(8)                         # usable ids are 1..7
    for _ in range(5):
        cs.new_concept()                # next=6: only ids 6 and 7 remain
    before = _allocator_snapshot(cs)

    with pytest.raises(
            RuntimeError, match="word/object/META triple.*No concept"):
        cs.create_word_object_meta([10, 11], (0,), key="unseated")

    _assert_snapshot_equal(before, _allocator_snapshot(cs))


def test_explicit_chain_capacity_failure_is_atomic():
    cs = _cs(8)
    ids = [cs.new_concept() for _ in range(6)]  # next=7: one id remains
    before = _allocator_snapshot(cs)

    with pytest.raises(RuntimeError, match="concept chain.*No concept"):
        cs.conceptualize_chain(ids[:3])         # two missing pair identities

    _assert_snapshot_equal(before, _allocator_snapshot(cs))


def test_automatic_capacity_mode_reuses_known_identity_without_recycling():
    cs = _cs(8)
    known = cs.create_word_object_meta([10], (0,), key="known")
    fillers = [cs.new_concept() for _ in range(4)]  # ids 4..7; inventory full
    cs.retire_concept(fillers[-1])             # retirement does not free an id

    reused = cs._automatic_word_object_meta(
        [10, 11], (0,), key="known")
    assert reused == known
    assert set(cs.concept_parts(known[0])) == {10, 11}

    before = _allocator_snapshot(cs)
    assert cs._automatic_word_object_meta(
        [20], (0,), key="unseen") is None
    _assert_snapshot_equal(before, _allocator_snapshot(cs))

    stats = cs.concept_admission_stats()
    assert stats["lookup_only"] is True
    assert stats["remaining"] == 0
    assert stats["dropped"]["word/object/META"] == 1
    assert fillers[-1] in cs._concept_allocator.retired
    with pytest.raises(RuntimeError, match="No concept was minted"):
        cs.create_word_object_meta([20], (0,), key="explicit-unseen")


def test_serial_property_autobind_does_not_persist_sentence_chain():
    cs = _cs(32)
    object.__setattr__(cs, "_serial_object_meta", True)
    pid = torch.tensor([[10, 11, 20, 21]])
    groups = torch.tensor([[0, 0, 1, 1]])
    words = [["ab", "cd"]]

    cs._autobind_property_concepts(
        pid, torch.randn(1, 4, _D), groups, words, words,
        percept_where=None, percept_when=None, tile_spans=None,
        percept_store=None, ws=_property_ws())

    alloc = cs._concept_allocator
    assert set(alloc.word_obj_meta) == {"ab", "cd"}
    assert alloc.next_id == 7                    # two atomic A/B/C triples
    assert alloc.joint == {}
    assert alloc.chain_idx == {}
    assert alloc.relate_idx == {}


@pytest.mark.parametrize("remaining", [1, 2])
def test_rejected_word_does_not_consume_location_fallback_rows(remaining):
    cs = _cs(8)
    object.__setattr__(cs, "_serial_object_meta", True)
    # capacity-next_id == remaining
    for _ in range(7 - remaining):
        cs.new_concept()
    alloc = cs._concept_allocator
    before = _allocator_snapshot(cs)
    spans = torch.tensor([[[0, 2]]])

    cs._autobind_property_concepts(
        torch.tensor([[10, 11]]), torch.randn(1, 2, _D),
        torch.tensor([[0, 0]]), [["new"]], [["new"]],
        percept_where=torch.tensor([[0, 1]]), percept_when=None,
        tile_spans=None, percept_store=None, ws=_property_ws(spans))

    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    assert alloc.word_obj_meta.get("new") is None
    stats = cs.concept_admission_stats()
    assert stats["remaining"] == remaining
    assert stats["dropped"]["word/object/META"] == 1
    assert "location" not in stats["dropped"]


def test_rejected_mixed_type_word_suppresses_every_overlapping_ws_span():
    """A word group is not a WS span: ``abc1`` analyses as LETTER+DIGIT."""
    cs = _cs(8)
    object.__setattr__(cs, "_serial_object_meta", True)
    for _ in range(6):
        cs.new_concept()                    # next=7: one row remains
    before = _allocator_snapshot(cs)
    ws = _property_ws(torch.tensor([[[0, 3], [3, 4]]]))

    cs._autobind_property_concepts(
        torch.tensor([[10, 11, 12, 13]]), torch.randn(1, 4, _D),
        torch.tensor([[0, 0, 0, 0]]), [["abc1"]], [["abc1"]],
        percept_where=torch.tensor([[0, 1, 2, 3]]), percept_when=None,
        # Each constituent carries the complete enclosing word tile.
        tile_spans=[[(0, 4), (0, 4), (0, 4), (0, 4)]],
        percept_store=None, ws=ws)

    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    stats = cs.concept_admission_stats()
    assert stats["remaining"] == 1
    assert stats["dropped"]["word/object/META"] == 1
    assert "location" not in stats["dropped"]
