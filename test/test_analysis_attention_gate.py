"""Analysis-side top-k attention gate (Architecture.md "three cognitive
operations", op 2).

Pi analysis over-produces codes; when an intent is set, WholeSpace keeps
the top-k analysed positions by the priming over the codes (best affinity
to the intent-boosted codebook), masking the rest to zero. Dark by default:
no intent => the gate is a no-op (byte-identical). Tested here as a unit on
the helper, bound to a lightweight stub (no full model build needed).
"""

import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import torch.nn as nn

from Spaces import WholeSpace, Codebook


def _stub(boosts, V=8, D=4):
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(V, D))
    s = types.SimpleNamespace()
    s.subspace = types.SimpleNamespace(what=cb)
    s.intent_boosts = (lambda: boosts)
    s._topk_priming_mask = types.MethodType(
        WholeSpace._topk_priming_mask, s)
    return s


def test_no_intent_is_a_noop():
    s = _stub(boosts=None)
    act = torch.randn(1, 6, 4)
    assert s._topk_priming_mask(act) is None


def test_mask_keeps_top_half_and_is_shape_safe():
    torch.manual_seed(0)
    boosts = torch.ones(8)
    s = _stub(boosts=boosts)
    act = torch.randn(2, 6, 4)
    mask = s._topk_priming_mask(act)
    assert mask.shape == (2, 6, 1)                  # shape preserved (N kept)
    # top-k = N//2 = 3 positions survive per row; the rest are zeroed.
    assert torch.allclose(mask.sum(dim=1).squeeze(-1), torch.tensor([3.0, 3.0]))
    assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_priming_selects_the_aligned_positions():
    # Boost exactly one codebook row; the analysed positions most aligned
    # with that row must be the ones kept.
    torch.manual_seed(1)
    V, D = 8, 4
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(V, D))
    boosts = torch.ones(V)
    boosts[3] = 50.0                                 # strongly prime row 3
    s = types.SimpleNamespace()
    s.subspace = types.SimpleNamespace(what=cb)
    s.intent_boosts = (lambda: boosts)
    s._topk_priming_mask = types.MethodType(WholeSpace._topk_priming_mask, s)

    row3 = cb.W[3].detach()
    # 6 positions: 0,1,2 aligned to row 3; 3,4,5 random/anti-aligned.
    act = torch.stack([
        row3, 1.1 * row3, 0.9 * row3,
        -row3, torch.randn(D), -2.0 * row3,
    ]).unsqueeze(0)                                  # [1, 6, 4]
    mask = s._topk_priming_mask(act)[0, :, 0]
    # The three row-3-aligned positions must survive.
    assert mask[0] == 1 and mask[1] == 1 and mask[2] == 1


def test_no_codebook_is_a_noop():
    s = types.SimpleNamespace()
    s.subspace = types.SimpleNamespace(what=None)
    s.intent_boosts = (lambda: torch.ones(8))
    s._topk_priming_mask = types.MethodType(WholeSpace._topk_priming_mask, s)
    assert s._topk_priming_mask(torch.randn(1, 6, 4)) is None
