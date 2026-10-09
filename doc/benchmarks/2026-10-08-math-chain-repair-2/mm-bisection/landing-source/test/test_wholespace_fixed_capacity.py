"""Fixed-capacity ownership checks for the learned WholeSpace property basis."""

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
    W = cb.W
    assert cb._capacity_frozen
    assert cb.vq is None
    assert any(param is W for param in ws.getParameters())
    optimizer = model.getOptimizer(lr=1e-3)
    assert sum(param is W for group in optimizer.param_groups for param in group['params']) == 1
    primitives = cb.primitive_properties.members
    assert sum(param is primitives for group in optimizer.param_groups for param in group['params']) == 1
    cb.replace_W(torch.nn.Parameter(torch.zeros_like(W)))
    assert cb.W is W
    assert torch.count_nonzero(W).item() == 0
    assert any(param is W for param in ws.getParameters())
    assert any(param is W for group in optimizer.param_groups for param in group['params'])


def _pending_split(ws):
    W = ws.subspace.what.W
    width = W.shape[-1]
    mean = torch.ones(width)
    ws._property_lbg = {0: dict(n=8, sum=mean * 8, sq=mean * 24,
        pulls=[(mean, b'a'), (-mean, b'b')])}
    ws._lbg_threshold, ws._lbg_min_count = .5, 8


def test_wholespace_capacity_exhaustion_is_atomic_and_actionable():
    model = _make_model()
    ws = model.wholeSpace
    cb, W = ws.subspace.what, ws.subspace.what.W
    cap = int(cb.nVectors)
    ws._property_rows_used = set(range(cap))
    _pending_split(ws)
    before_W = W.detach().clone()
    before_properties = cb.primitive_properties.members.detach().clone()
    assert ws.maybe_split_property_row(0) is None
    assert cb.W is W
    assert tuple(W.shape) == tuple(before_W.shape)
    assert torch.equal(W, before_W)
    assert torch.equal(cb.primitive_properties.members, before_properties)
    assert ws._property_rows_used == set(range(cap))
    assert not hasattr(ws, 'insert_whole')
    assert not hasattr(cb, 'grow_to')
    with pytest.raises(RuntimeError, match='fixed capacity'):
        cb.replace_W(torch.zeros(cap + 1, W.shape[1]))
    assert cb.W is W
    assert tuple(W.shape) == tuple(before_W.shape)


def test_property_reads_are_rng_neutral_and_admit_no_rows():
    """A property query cannot recreate a word/META allocator or hidden reserve."""
    model = _make_model()
    ws, cb = model.wholeSpace, model.wholeSpace.subspace.what
    W = cb.W.detach().clone()
    before = torch.get_rng_state().clone()
    for text in ('cat', 'dog', '123', 'cat'):
        ws.property_rows_for_bytes(text)
    torch.testing.assert_close(cb.W, W, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), before)
    assert cb.active_row_count() == cb.nVectors
    assert not hasattr(ws, '_paired_next_row')
    assert not hasattr(ws, 'taxonomy')


def test_property_split_uses_an_unused_row_without_replacing_parameters():
    from Spaces import WholeSpace
    from util import init_config
    init_config(path=_CONFIG, defaults_path=_DEFAULTS)
    ws = WholeSpace([8, 14], [64, 14], [8, 14])
    cb, W = ws.subspace.what, ws.subspace.what.W
    primitives = cb.primitive_properties.members
    initial = cb.active_row_count()
    _pending_split(ws)
    row = ws.maybe_split_property_row(0)
    assert row is not None and row > 0
    assert cb.W is W
    assert cb.primitive_properties.members is primitives
    assert cb.active_row_count() == initial == 64
    assert cb.primitive_properties.members[row, ord('b')] > 0


def test_empty_wholespace_inventory_has_no_parameter_identity_to_freeze():
    from Spaces import Codebook, WholeSpace
    ws = object.__new__(WholeSpace)
    torch.nn.Module.__init__(ws)
    cb = Codebook()
    cb.nVectors = 0
    object.__setattr__(ws, 'subspace', types.SimpleNamespace(what=cb))
    ws.params = []
    assert cb.W is None
    assert not cb._capacity_frozen
    assert ws.getParameters() == []
