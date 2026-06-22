"""B>=2 per-row isolation for SymbolSpace.last_svo and SymbolSpace._stm_fired.

Task 2 of the microbatch AR refactor (see
basicmodel/doc/specs/2026-04-22-microbatch-ar-refactor-design.md).

SymbolSpace cannot be constructed in isolation -- it requires real
PartSpace, ConceptualSpace, and WholeSpace objects.  We build the
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
    percept_space   = Spaces.PartSpace(inputShape, spaceShape, outputShape)
    concept_space   = Spaces.ConceptualSpace(inputShape, spaceShape, outputShape)
    symbolic_space  = Spaces.WholeSpace(inputShape, spaceShape, outputShape)
    Language.TheGrammar._configured = False
    ss = Language.SymbolSubSpace(
        perceptualSpace=percept_space,
        conceptualSpace=concept_space,
        wholeSpace=symbolic_space,
        nPercepts=nPercepts,
        nConcepts=nSymbols,
        nSymbols=nSymbols,
        concept_dim=conceptDim,
        symbol_dim=symbolDim,
    )
    # ``ensure_microbatch`` is the per-row state entry point post-2026-05-13:
    # it sizes body-side state (B*K) AND the B-sized lifecycle fields
    # (``_stm_fired``, discourse) together.  Passing K=1 makes B*K == B
    # so both halves end up at ``batch`` rows -- the legacy contract these
    # tests check.  ``ensure_batch`` alone no longer touches ``_stm_fired``
    # (that wipe was the root cause of the K-change history loss bug).
    ss.ensure_microbatch(batch, 1)
    return ss, symbolDim


# -- last_svo --------------------------------------------------------------

def test_last_svo_b2_isolation():
    ss, D = _make_ws(batch=2)
    s = torch.randn(D); v = torch.randn(D); o = torch.randn(D)
    ss.set_last_svo(0, s, v, o)
    assert ss.svo_valid(0) is True
    assert ss.svo_valid(1) is False
    sub, verb, obj = ss.get_last_svo(0)
    assert torch.equal(sub, s)
    assert torch.equal(verb, v)
    assert torch.equal(obj, o)


def test_last_svo_clear_per_row():
    ss, D = _make_ws(batch=2)
    ss.set_last_svo(0, torch.randn(D), torch.randn(D), torch.randn(D))
    ss.set_last_svo(1, torch.randn(D), torch.randn(D), torch.randn(D))
    assert ss.svo_valid(0) and ss.svo_valid(1)
    ss.clear_last_svo(1)
    assert ss.svo_valid(0) and not ss.svo_valid(1)
    ss.clear_last_svo()  # all rows
    assert not ss.svo_valid(0) and not ss.svo_valid(1)


def test_last_svo_ensure_batch_grows_clears():
    ss, D = _make_ws(batch=1)
    ss.set_last_svo(0, torch.randn(D), torch.randn(D), torch.randn(D))
    assert ss.svo_valid(0)
    ss.ensure_batch(3)
    # ensure_batch reallocates; valid mask is zero on every row.
    assert not ss.svo_valid(0)
    assert not ss.svo_valid(1)
    assert not ss.svo_valid(2)


# -- _stm_fired ------------------------------------------------------------

def test_stm_fired_b2_isolation():
    ss, _ = _make_ws(batch=2)
    assert not ss.stm_fired(0)
    assert not ss.stm_fired(1)
    ss.mark_stm_fired(0)
    assert ss.stm_fired(0)
    assert not ss.stm_fired(1)


def test_stm_arm_per_row():
    ss, _ = _make_ws(batch=2)
    ss.mark_stm_fired(0)
    ss.mark_stm_fired(1)
    assert ss.stm_fired(0) and ss.stm_fired(1)
    ss.arm_stm(1)
    assert ss.stm_fired(0) and not ss.stm_fired(1)
    ss.arm_stm()  # all rows
    assert not ss.stm_fired(0) and not ss.stm_fired(1)


def test_ensure_microbatch_grows_stm_fired_to_new_B():
    """``ensure_microbatch(B', K)`` with a NEW source-row count B' grows
    ``_stm_fired`` to ``B'`` and resets it to zeros (the new rows are
    different streams; previous-stream fire flags don't carry over).

    Replaces the legacy ``test_stm_fired_ensure_batch_resets`` which
    assumed ``ensure_batch`` itself reallocated ``_stm_fired``.  Post
    2026-05-13 fix, ``ensure_batch`` is body-side-only; only
    ``ensure_microbatch`` reshapes the B-sized lifecycle fields, and
    only when the source-row count actually changes (K-change alone
    no longer wipes -- regression test in
    ``test_ensure_microbatch_cascade.py``).
    """
    ss, _ = _make_ws(batch=1)
    ss.mark_stm_fired(0)
    assert ss.stm_fired(0)
    ss.ensure_microbatch(3, 1)
    # B changed from 1 -> 3, so _stm_fired is a fresh [3] zero tensor.
    assert not ss.stm_fired(0)
    assert not ss.stm_fired(1)
    assert not ss.stm_fired(2)


def test_reset_clears_both():
    ss, D = _make_ws(batch=2)
    ss.set_last_svo(0, torch.randn(D), torch.randn(D), torch.randn(D))
    ss.set_last_svo(1, torch.randn(D), torch.randn(D), torch.randn(D))
    ss.mark_stm_fired(0)
    ss.mark_stm_fired(1)
    ss.Reset()
    assert not ss.svo_valid(0) and not ss.svo_valid(1)
    assert not ss.stm_fired(0) and not ss.stm_fired(1)


# -- stm_residual_microbatch (per-source gating) ---------------------------

def _attach_discourse(ss, n_dim, concept_dim, batch):
    """Attach a minimal InterSentenceLayer to ss so predict()/prime()
    return non-None tensors. Pre-populates the recent buffer with a
    deterministic snapshot per source row, then arms STM so the
    discourse-prediction cache (``_disc_pred`` / ``_disc_conf``) is
    populated. The D8 capture-gate refactor (2026-05-19) moved the
    ``disc.predict()`` call from ``stm_residual_microbatch`` to
    ``arm_stm``; tests must arm before they can read the cache."""
    import Layers
    n_sym = ss.subspace.outputShape[0] if hasattr(ss.subspace, 'outputShape') else 3
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
    ss.discourse = disc
    s = torch.randn(batch, n_sym, n_dim)
    w = torch.randn(batch, 4, n_dim)
    disc.snapshot(s, w)
    # Refresh the discourse-prediction cache so stm_residual_microbatch
    # can read ``_disc_pred`` / ``_disc_conf`` without falling back to
    # None at the cache miss.
    ss.arm_stm()
    return disc


# test_stm_residual_microbatch_* removed: these tests reached into
# ``SymbolSubSpace.subspace`` (no longer an attribute -- the SR-parser
# stack was retired into ConceptualSpace.stm 2026-05-20).
