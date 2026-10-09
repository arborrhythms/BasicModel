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
from definition_fixtures import with_definitions

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
    object.__setattr__(cs, "_serial", True)
    return with_definitions(cs)


def _property_ws(spans=None):
    return types.SimpleNamespace(
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
        "definitions": dict(cs.definitions._rows),
        "relate_idx": dict(alloc.relate_idx),
        "constituents": {
            key: list(value) for key, value in layer._constituents.items()
        },
        "W": cs.similarity_codebook.getW().detach().clone(),
    }


def _assert_snapshot_equal(before, after):
    torch.testing.assert_close(after.pop("W"), before.pop("W"))
    assert after == before


def test_explicit_word_row_capacity_failure_is_atomic():
    cs = _cs(8)
    for row in range(8):
        assert cs._csw_concept_row(0, 1000 + row) == row
    before = _allocator_snapshot(cs)
    with pytest.raises(RuntimeError, match='capacity'):
        cs.interpret_word([10, 11], (0,), key='unseated')
    _assert_snapshot_equal(before, _allocator_snapshot(cs))




def test_automatic_capacity_mode_reuses_known_identity_without_recycling():
    cs = _cs(8)
    known = cs.interpret_word([10], (0,), key='known')
    for row in range(1, 8):
        assert cs._csw_concept_row(0, 1000 + row) == row
    cs.retire_concept(1007)
    assert cs.interpret_word([10], (0,), key='known') == known
    before = _allocator_snapshot(cs)
    with pytest.raises(RuntimeError, match='capacity'):
        cs.interpret_word([20], (0,), key='unseen')
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    assert 1007 in cs._concept_allocator.retired
    before = _allocator_snapshot(cs)
    with pytest.raises(RuntimeError, match='capacity'):
        cs.interpret_word([10, 11], (0,), key='known')
    _assert_snapshot_equal(before, _allocator_snapshot(cs))


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
    assert set(cs.definitions._forms) == {"ab", "cd"}
    assert alloc.next_id == 5                    # two atomic A/B/C triples
    assert not hasattr(alloc, "joint")
    assert not hasattr(alloc, "chain_idx")
    assert alloc.relate_idx == {}


@pytest.mark.parametrize("remaining", [1, 2])
def test_rejected_word_does_not_consume_location_fallback_rows(remaining):
    cs = _cs(8)
    for row in range(8 - remaining):
        cs._csw_concept_row(0, 1000 + row)
    store = cs.definitions._store()
    while len(store) < store.capacity:
        store.append_idea(torch.zeros(_D), sentence_index=len(store))
    before = _allocator_snapshot(cs)
    spans = torch.tensor([[[0, 2]]])
    cs._autobind_property_concepts(
        torch.tensor([[10, 11]]), torch.randn(1, 2, _D),
        torch.tensor([[0, 0]]), [['new']], [['new']],
        percept_where=torch.tensor([[0, 1]]), percept_when=None,
        tile_spans=None, percept_store=None, ws=_property_ws(spans))
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    assert cs.definitions.word(form='new') is None
    assert len(cs._concept_allocator.layer()._tensor_rows) == 8 - remaining


def test_rejected_mixed_type_word_suppresses_every_overlapping_ws_span():
    cs = _cs(8)
    for row in range(8):
        cs._csw_concept_row(0, 1000 + row)
    before = _allocator_snapshot(cs)
    ws = _property_ws(torch.tensor([[[0, 3], [3, 4]]]))
    cs._autobind_property_concepts(
        torch.tensor([[10, 11, 12, 13]]), torch.randn(1, 4, _D),
        torch.tensor([[0, 0, 0, 0]]), [['abc1']], [['abc1']],
        percept_where=torch.tensor([[0, 1, 2, 3]]), percept_when=None,
        tile_spans=[[(0, 4), (0, 4), (0, 4), (0, 4)]],
        percept_store=None, ws=ws)
    _assert_snapshot_equal(before, _allocator_snapshot(cs))
    assert cs.definitions.word(form='abc1') is None
