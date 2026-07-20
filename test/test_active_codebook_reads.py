"""Direct Codebook readers obey logical occupancy, not physical capacity."""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Spaces import Codebook, ConceptualSpace, WholeSpace


def _codebook(rows, active):
    rows = torch.as_tensor(rows, dtype=torch.float32)
    cb = Codebook()
    cb.create(
        nInput=1,
        nVectors=int(rows.shape[0]),
        nDim=int(rows.shape[1]),
        customVQ=True,
        monotonic=False,
    )
    with torch.no_grad():
        cb.W.copy_(rows)
        cb.vq.embed_avg.copy_(rows)
        cb.vq._b_norms_sq.copy_((rows * rows).sum(dim=-1))
    cb.vq.set_active_rows(active)
    return cb


def _whole_space(cb):
    ws = object.__new__(WholeSpace)
    torch.nn.Module.__init__(ws)
    object.__setattr__(ws, "subspace", types.SimpleNamespace(what=cb, basis=cb))
    object.__setattr__(ws, "_codebook", "quantize")
    ws.vq_chunk_budget = 1 << 20
    ws._next_position = 1
    ws._pos_kind = {}
    ws._ws_pos_to_row = {}
    ws._ws_row_to_pos = {}
    return ws


def test_active_prototypes_and_reverse_ignore_inactive_exact_match():
    cb = _codebook(
        [[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.8, 0.8]],
        active=2,
    )
    query = torch.tensor([[[0.8, 0.8]]])

    assert cb.active_row_count() == 2
    assert cb.active_prototypes().shape == (2, 2)
    assert cb.active_prototypes().data_ptr() == cb.getW().data_ptr()

    snapped = cb.reverse(query)
    assert any(torch.equal(snapped[0, 0], row) for row in cb.getW()[:2])
    assert not torch.equal(snapped[0, 0], cb.getW()[3])


def test_standalone_default_remains_all_active():
    cb = _codebook([[0.0, 0.0], [0.25, 0.5], [0.8, 0.8]], active=3)
    query = torch.tensor([[[0.8, 0.8]]])

    assert cb.active_row_count() == 3
    assert torch.equal(cb.reverse(query)[0, 0], cb.getW()[2])


def test_wholespace_nearest_and_terminal_targets_ignore_inactive_exact_match():
    cb = _codebook(
        [[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.8, 0.8]],
        active=2,
    )
    ws = _whole_space(cb)
    query = torch.tensor([0.8, 0.8])

    row, _distance = ws.nearest_ws_row(query)
    assert 0 <= row < 2

    target = ws._nearest_symbol_target(query.reshape(1, 1, 2))
    assert any(torch.equal(target[0, 0], active) for active in cb.getW()[:2])
    assert not torch.equal(target[0, 0], cb.getW()[3])

    snapped, indices = ws._snap_to_terminal_ste(query.reshape(1, 2), cb)
    assert int(indices.item()) < 2
    assert not torch.equal(snapped[0], cb.getW()[3])


def test_ensure_ws_position_rejects_inactive_row_without_taxonomy_mutation():
    cb = _codebook(
        [[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.8, 0.8]],
        active=2,
    )
    ws = _whole_space(cb)

    before = (ws._next_position, dict(ws._pos_kind), dict(ws._ws_row_to_pos))
    with pytest.raises(ValueError, match="cannot bind an inactive WS row 3"):
        ws.ensure_ws_position(3)
    assert (ws._next_position, ws._pos_kind, ws._ws_row_to_pos) == before

    pos = ws.ensure_ws_position(1)
    assert ws._ws_row_to_pos[1] == pos


def test_cs_direct_snap_cannot_score_or_ema_write_inactive_tail():
    cb = _codebook(
        [[0.0, 0.0], [0.5, 0.0], [-0.5, 0.0], [0.8, 0.8]],
        active=2,
    )
    cs = object.__new__(ConceptualSpace)
    torch.nn.Module.__init__(cs)
    cs.similarity_codebook = cb
    cs.nVectors = 4
    cs.outputShape = [4, 2]
    object.__setattr__(cs, "_concept_binding", "aligned")
    object.__setattr__(cs, "_serial", True)
    object.__setattr__(cs, "_symbolic_order", 0)
    cs.train()

    inactive_before = cb.getW()[2:].detach().clone()
    activation = cs.cs_snap_order0(
        torch.tensor([[[0.8, 0.8]]]), ema=True)

    assert activation.shape == (2, 1)
    assert torch.equal(cb.getW()[2:].detach(), inactive_before)


def test_topk_priming_scores_only_active_rows():
    cb = _codebook(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        active=1,
    )
    ws = _whole_space(cb)
    ws._intent_boosts = torch.ones(4)
    # Position 0 is an exact match only for inactive row 1. Position 1 is the
    # best match to the sole active row and therefore must survive top-k.
    act = torch.tensor([[[0.0, 1.0], [0.8, 0.2]]])

    mask = ws._topk_priming_mask(act)
    assert mask.reshape(-1).tolist() == [0.0, 1.0]


def test_semantic_arrangement_is_invariant_to_inactive_tail():
    cb = _codebook(
        [[1.0, 0.0], [0.0, 1.0], [10.0, 10.0], [-10.0, -10.0]],
        active=2,
    )
    ws = _whole_space(cb)
    ws.analysis_store = cb
    ws.semantic_arrangement_weight = 1.0

    first = ws.semantic_arrangement_loss(torch.tensor([0]))
    with torch.no_grad():
        cb.getW()[2:].copy_(torch.tensor([[1000.0, -3.0], [-500.0, 700.0]]))
    second = ws.semantic_arrangement_loss(torch.tensor([0]))

    assert torch.equal(first, second)
