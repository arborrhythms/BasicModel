"""Fixed-capacity ownership checks for the learned WholeSpace dictionary."""

from __future__ import annotations

import os
import sys
import types
import warnings

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor_fixture.xml")
_DEFAULTS = os.path.join(_PROJECT, "data", "model.xml")


def _make_model():
    import Language
    import Models
    from util import init_config

    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model, _ = Models.BasicModel.from_config(_CONFIG)
    return model


def test_wholespace_W_identity_is_frozen_and_optimizer_visible():
    model = _make_model()
    ws = model.wholeSpace
    cb = ws.subspace.what
    W = cb.getW()

    assert cb._capacity_frozen
    assert cb.vq.codebook is W
    assert any(param is W for param in ws.getParameters())
    optimizer = model.getOptimizer(lr=1e-3)
    assert sum(
        param is W
        for group in optimizer.param_groups
        for param in group["params"]
    ) == 1

    # A same-shape refresh may update values, but cannot replace the owner
    # Parameter (and therefore cannot orphan Adam state).
    replacement = torch.nn.Parameter(torch.zeros_like(W))
    cb.replace_W(replacement)
    assert cb.getW() is W
    assert cb.vq.codebook is W
    assert torch.count_nonzero(W).item() == 0
    assert any(param is W for param in ws.getParameters())
    assert any(
        param is W
        for group in optimizer.param_groups
        for param in group["params"]
    )


def test_wholespace_capacity_exhaustion_is_atomic_and_actionable():
    model = _make_model()
    ws = model.wholeSpace
    cb = ws.subspace.what
    W = cb.getW()
    cap = int(cb.nVectors)

    # Put the allocator exactly at capacity without mutating any taxonomy.
    ws._paired_next_row = cap
    before_W = W.detach().clone()
    before_next_position = ws._next_position
    before_pos_kind = dict(ws._pos_kind)
    before_row_map = dict(ws._ws_row_to_pos)
    had_pair_map = hasattr(ws, "_paired_orth_to_sem")

    with pytest.raises(RuntimeError, match="capacity exhausted.*No allocation"):
        ws.insert_whole()

    assert cb.getW() is W
    assert tuple(W.shape) == tuple(before_W.shape)
    assert torch.equal(W, before_W)
    assert ws._paired_next_row == cap
    assert ws._next_position == before_next_position
    assert ws._pos_kind == before_pos_kind
    assert ws._ws_row_to_pos == before_row_map
    assert hasattr(ws, "_paired_orth_to_sem") == had_pair_map

    with pytest.raises(RuntimeError, match="fixed capacity"):
        cb.grow_to(cap + 1)
    assert cb.getW() is W
    assert tuple(W.shape) == tuple(before_W.shape)


def test_legacy_fixed_reserve_is_rng_neutral_and_logically_lazy():
    """A physical WS reserve must not perturb later module initialization."""
    from Spaces import WholeSpace
    from util import init_config

    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    torch.manual_seed(1234)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        small = WholeSpace([8, 14], [8, 14], [8, 14])
    after_small = torch.rand(8)

    torch.manual_seed(1234)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        reserved = WholeSpace([8, 14], [64, 14], [8, 14])
    after_reserved = torch.rand(8)

    torch.testing.assert_close(
        reserved.subspace.what.getW()[:8],
        small.subspace.what.getW(), rtol=0, atol=0)
    torch.testing.assert_close(after_reserved, after_small, rtol=0, atol=0)
    assert reserved.subspace.what.active_row_count() == 8


def test_legacy_insert_reveals_reserved_row_without_replacing_parameter():
    model = _make_model()
    ws = model.wholeSpace
    cb = ws.subspace.what
    W = cb.getW()
    initial = cb.active_row_count()

    assert initial == int(ws.outputShape[0])
    # Place the structural cursor at the first inactive row so this call
    # exercises logical reveal rather than reusing historical atom slack.
    ws._paired_next_row = initial
    pos = ws.insert_whole()

    assert pos > 0
    assert cb.getW() is W
    assert cb.active_row_count() == int(ws._paired_next_row)
    assert cb.active_row_count() == initial + 1


def test_empty_wholespace_inventory_has_no_parameter_identity_to_freeze():
    """A zero-row shape-only model has no learned W or optimizer identity."""
    from Spaces import Codebook, WholeSpace

    ws = object.__new__(WholeSpace)
    torch.nn.Module.__init__(ws)
    cb = Codebook()
    cb.nVectors = 0
    object.__setattr__(ws, "subspace", types.SimpleNamespace(what=cb))
    ws.params = []

    ws._freeze_symbol_codebook_capacity()
    assert ws._symbol_codebook_parameter_id is None
    assert cb.getW() is None
    assert not cb._capacity_frozen
    ws._assert_symbol_codebook_optimizer_identity()
    assert ws.getParameters() == []
