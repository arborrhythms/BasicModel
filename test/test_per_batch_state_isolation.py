"""B>=2 per-row isolation for WordSpace.last_svo and WordSpace._stm_fired.

Task 2 of the microbatch AR refactor (see
basicmodel/doc/specs/2026-04-22-microbatch-ar-refactor-design.md).

WordSpace cannot be constructed in isolation -- it requires real
PerceptualSpace, ConceptualSpace, and SymbolicSpace objects.  We build the
minimal chain using _populate_test_config + direct Space constructors, the
same pattern used by the test_partition_* tests.
"""
import os
import sys

import pytest
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
_TEST = os.path.dirname(os.path.abspath(__file__))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
if _TEST not in sys.path:
    sys.path.insert(0, _TEST)

import Spaces
import Language
from test_basicmodel import _populate_test_config


def _make_ws(batch=2, nSymbols=3, symbolDim=4, conceptDim=4, nPercepts=3):
    _populate_test_config(
        inputDim=conceptDim,
        perceptDim=conceptDim,
        conceptDim=conceptDim,
        symbolDim=symbolDim,
        wordDim=symbolDim,
        outputDim=symbolDim,
        nInput=nPercepts,
        nPercepts=nPercepts,
        nConcepts=nSymbols,
        nSymbols=nSymbols,
        nWords=nSymbols,
        nOutput=nSymbols,
        nWhere=0,
        nWhen=0,
    )
    inputShape   = [nPercepts,  conceptDim]
    spaceShape   = [nSymbols,   symbolDim]
    outputShape  = [nSymbols,   symbolDim]
    percept_space   = Spaces.PerceptualSpace(inputShape, spaceShape, outputShape)
    concept_space   = Spaces.ConceptualSpace(inputShape, spaceShape, outputShape)
    symbolic_space  = Spaces.SymbolicSpace(inputShape, spaceShape, outputShape)
    Language.TheGrammar._configured = False
    ws = Language.WordSpace(
        perceptualSpace=percept_space,
        conceptualSpace=concept_space,
        symbolicSpace=symbolic_space,
        nPercepts=nPercepts,
        nConcepts=nSymbols,
        nSymbols=nSymbols,
        concept_dim=conceptDim,
        symbol_dim=symbolDim,
    )
    ws.ensure_batch(batch)
    return ws, symbolDim


# -- last_svo --------------------------------------------------------------

def test_last_svo_b2_isolation():
    ws, D = _make_ws(batch=2)
    s = torch.randn(D); v = torch.randn(D); o = torch.randn(D)
    ws.set_last_svo(0, s, v, o)
    assert ws.svo_valid(0) is True
    assert ws.svo_valid(1) is False
    sub, verb, obj = ws.get_last_svo(0)
    assert torch.equal(sub, s)
    assert torch.equal(verb, v)
    assert torch.equal(obj, o)


def test_last_svo_clear_per_row():
    ws, D = _make_ws(batch=2)
    ws.set_last_svo(0, torch.randn(D), torch.randn(D), torch.randn(D))
    ws.set_last_svo(1, torch.randn(D), torch.randn(D), torch.randn(D))
    assert ws.svo_valid(0) and ws.svo_valid(1)
    ws.clear_last_svo(1)
    assert ws.svo_valid(0) and not ws.svo_valid(1)
    ws.clear_last_svo()  # all rows
    assert not ws.svo_valid(0) and not ws.svo_valid(1)


def test_last_svo_ensure_batch_grows_clears():
    ws, D = _make_ws(batch=1)
    ws.set_last_svo(0, torch.randn(D), torch.randn(D), torch.randn(D))
    assert ws.svo_valid(0)
    ws.ensure_batch(3)
    # ensure_batch reallocates; valid mask is zero on every row.
    assert not ws.svo_valid(0)
    assert not ws.svo_valid(1)
    assert not ws.svo_valid(2)


# -- _stm_fired ------------------------------------------------------------

def test_stm_fired_b2_isolation():
    ws, _ = _make_ws(batch=2)
    assert not ws.stm_fired(0)
    assert not ws.stm_fired(1)
    ws.mark_stm_fired(0)
    assert ws.stm_fired(0)
    assert not ws.stm_fired(1)


def test_stm_arm_per_row():
    ws, _ = _make_ws(batch=2)
    ws.mark_stm_fired(0)
    ws.mark_stm_fired(1)
    assert ws.stm_fired(0) and ws.stm_fired(1)
    ws.arm_stm(1)
    assert ws.stm_fired(0) and not ws.stm_fired(1)
    ws.arm_stm()  # all rows
    assert not ws.stm_fired(0) and not ws.stm_fired(1)


def test_stm_fired_ensure_batch_resets():
    ws, _ = _make_ws(batch=1)
    ws.mark_stm_fired(0)
    assert ws.stm_fired(0)
    ws.ensure_batch(3)
    # ensure_batch reallocates; every row armed.
    assert not ws.stm_fired(0)
    assert not ws.stm_fired(1)
    assert not ws.stm_fired(2)


def test_reset_clears_both():
    ws, D = _make_ws(batch=2)
    ws.set_last_svo(0, torch.randn(D), torch.randn(D), torch.randn(D))
    ws.set_last_svo(1, torch.randn(D), torch.randn(D), torch.randn(D))
    ws.mark_stm_fired(0)
    ws.mark_stm_fired(1)
    ws.Reset()
    assert not ws.svo_valid(0) and not ws.svo_valid(1)
    assert not ws.stm_fired(0) and not ws.stm_fired(1)


# -- stm_residual_microbatch (per-source gating) ---------------------------

def _attach_discourse(ws, n_dim, concept_dim, batch):
    """Attach a minimal InterSentenceLayer to ws so predict()/prime()
    return non-None tensors.  Pre-populates the recent buffer with a
    deterministic snapshot per source row so predict() yields a real
    prediction instead of (None, None)."""
    import Layers
    n_sym = ws.subspace.outputShape[0] if hasattr(ws.subspace, 'outputShape') else 3
    disc = Layers.InterSentenceLayer(
        n_symbols=n_sym,
        max_depth=4,
        n_dim=n_dim,
        context_window=4,
        centroid_history=2,
        lam=1.01,
        concept_dim=concept_dim,
        batch=batch,
    )
    ws.discourse = disc
    s = torch.randn(batch, n_sym, n_dim)
    w = torch.randn(batch, 4, n_dim)
    disc.snapshot(s, w)
    return disc


def test_stm_residual_microbatch_b2_k3_per_source_gating():
    """Each source row contributes a bias to its K windows; rows
    already fired contribute zero; sources are marked fired after."""
    B, K = 2, 3
    ws, D = _make_ws(batch=B * K)
    # Reset _stm_fired to source-row width per ensure_microbatch contract.
    ws._stm_fired = torch.zeros(B, dtype=torch.bool)
    _attach_discourse(ws, n_dim=D, concept_dim=D, batch=B * K)
    # Pre-fire source row 1; source 0 should contribute, 1 should be zero.
    ws.mark_stm_fired(1)
    bias = ws.stm_residual_microbatch(B, K)
    assert bias is not None, "expected a bias tensor when source 0 unfired"
    assert bias.shape == (B * K, D), (
        f"expected [{B * K}, {D}], got {tuple(bias.shape)}")
    # Source 0's K windows are at indices [0, K); source 1's at [K, 2K).
    src0 = bias[:K]
    src1 = bias[K:]
    assert torch.any(src0 != 0), "source 0 should have nonzero bias"
    assert torch.all(src1 == 0), "source 1 was pre-fired; bias must be zero"
    # After this call, every source that contributed is marked fired.
    assert ws.stm_fired(0)
    assert ws.stm_fired(1)


def test_stm_residual_microbatch_returns_none_when_all_fired():
    """When every source has already fired, the call returns None and
    does not mutate state."""
    B, K = 3, 2
    ws, D = _make_ws(batch=B * K)
    ws._stm_fired = torch.ones(B, dtype=torch.bool)
    _attach_discourse(ws, n_dim=D, concept_dim=D, batch=B * K)
    bias = ws.stm_residual_microbatch(B, K)
    assert bias is None
    # Still all fired; no spurious unset.
    assert ws.stm_fired(0) and ws.stm_fired(1) and ws.stm_fired(2)


def test_stm_residual_microbatch_b1_k1_legacy_shape():
    """The degenerate B=K=1 case returns a [1, concept_dim] tensor and
    the broadcast at the call site (bias.unsqueeze(1)) yields [1,1,D]."""
    ws, D = _make_ws(batch=1)
    ws._stm_fired = torch.zeros(1, dtype=torch.bool)
    _attach_discourse(ws, n_dim=D, concept_dim=D, batch=1)
    bias = ws.stm_residual_microbatch(1, 1)
    assert bias is not None
    assert bias.shape == (1, D)
    assert ws.stm_fired(0)
