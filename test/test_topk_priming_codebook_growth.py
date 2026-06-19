"""Regression: `WholeSpace._topk_priming_mask` must conform the cached
intent boosts to the CURRENT codebook size.

The WS codebook grows at runtime (words added) AFTER the §5 intent boosts
are cached (set at the §6c sentence-protocol prelude). The top-k
subsymbolic-attention gate (op 2 of the three cognitive operations) then
scored ``sim`` (shape ``[B, N, V_new]``) against ``boosts[:V_new]`` -- but a
stale ``boosts`` shorter than ``V_new`` stayed short, so the elementwise
``sim * bv`` raised "The size of tensor a (V_new) must match the size of
tensor b (len(boosts))". Reproduced independently on data/MM_grammar.xml
(serial -> sentenceProtocol ON) at the small space widths it uses; the LTM
serial fixture had to disable sentenceProtocol to dodge it.

Fix: pad new rows with the NEUTRAL weight 1.0 (intent_priming_weights'
multiplicative identity) and truncate a stale longer cache -- mirroring the
VQ selection's ``install_intent_priming._boosts``. Off-path (no intent) is
byte-identical (returns None).
"""

from __future__ import annotations

import os
import sys
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

_DATA = os.path.join(_PROJECT, "data")


def _ws():
    from util import init_config
    import Language
    import Models
    init_config(path=os.path.join(_DATA, "MM_grammar.xml"),
                defaults_path=os.path.join(_DATA, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(os.path.join(_DATA, "MM_grammar.xml"))
    return m.wholeSpace


def _grow(ss, v):
    cb = ss.subspace.what
    if int(cb.getW().shape[0]) < v:
        cb.grow_to(v)
    return int(cb.getW().shape[0])


def test_stale_short_boosts_after_codebook_growth_no_crash():
    # Codebook grew to V after boosts were cached at a SMALLER size -> the
    # pre-fix crash; the gate must now conform + return a valid [B,N,1] mask.
    ss = _ws()
    V = _grow(ss, 8)
    ss._intent_boosts = torch.ones(2)                     # stale, shorter than V
    act = torch.randn(1, 4, int(ss.subspace.what.getW().shape[1]))
    mask = ss._topk_priming_mask(act)
    assert mask is not None
    assert tuple(mask.shape) == (1, 4, 1)
    # top-k keeps k = N // 2 = 2 positions (0/1 mask).
    assert float(mask.sum().item()) == 2.0
    assert set(mask.reshape(-1).tolist()) <= {0.0, 1.0}
    assert V == 8


def test_stale_long_boosts_truncated_no_crash():
    ss = _ws()
    _grow(ss, 8)
    ss._intent_boosts = torch.ones(20)                   # stale, longer than V
    act = torch.randn(2, 4, int(ss.subspace.what.getW().shape[1]))
    mask = ss._topk_priming_mask(act)
    assert mask is not None
    assert tuple(mask.shape) == (2, 4, 1)


def test_no_intent_is_byte_identical_noop():
    ss = _ws()
    ss._intent_boosts = None
    act = torch.randn(1, 4, int(ss.subspace.what.getW().shape[1]))
    assert ss._topk_priming_mask(act) is None


def test_exact_length_boosts_unchanged():
    ss = _ws()
    V = _grow(ss, 8)
    ss._intent_boosts = torch.ones(V)
    act = torch.randn(1, 4, int(ss.subspace.what.getW().shape[1]))
    mask = ss._topk_priming_mask(act)
    assert mask is not None and tuple(mask.shape) == (1, 4, 1)


def test_n_equals_one_returns_none():
    # N == 1 -> k >= N -> no-op (the per-word serial step), regardless of
    # boosts/codebook sizing.
    ss = _ws()
    _grow(ss, 8)
    ss._intent_boosts = torch.ones(2)
    act = torch.randn(4, 1, int(ss.subspace.what.getW().shape[1]))
    assert ss._topk_priming_mask(act) is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
