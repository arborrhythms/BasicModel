"""Opportunistic (scored) STM reduce -- the 2b-2 refinement (Alec 2026-07-12).

A "normal" grammatical parse runs on STM DURING the serial phase: after
each word's push, a scored gate ``g`` (the reducer DP's reduce-vs-copy
marginal on the top-2 window) decides per row whether to fold. This keeps
the syntactic processing on the STM stack, distinct from the SymbolicLoop
(the symbolicOrder budget), which stays the forward processor of symbolic
activation. Grammar-free (substrate-only) configs have no reducer -> the
opportunistic step is a structural no-op there.

cpu/eager, seeded.
"""
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import functools
import sys

import torch

sys.path.insert(0, "bin")
from recon_bench import _build_model, _resolve_config


@functools.lru_cache(maxsize=None)
def _build(cfg):
    torch.manual_seed(3)
    model, *_ = _build_model(_resolve_config(cfg))
    return model


def _stm_with_depth2(m, B=2):
    """Fresh STM slab with two distinct ideas so every row has a
    reducible top-2 (begin_forward is the per-batch reset)."""
    stm = m.conceptualSpace.stm
    stm.begin_forward(B, device=torch.device("cpu"))
    D = int(stm.concept_dim)
    torch.manual_seed(11)
    for _ in range(2):
        vec = torch.randn(B, D)
        stm.push_step_masked(vec, torch.ones(B, 1, dtype=torch.bool))
    return stm


def test_gate_tau_bounds():
    """tau=0 folds every depth-2 row (g>0 always); tau=1 folds none
    (g<1 always) -- the marginal is a proper (0,1) score."""
    m = _build("data/MM_grammar.xml")
    assert m._stm_reducer() is not None, "grammar config must carry a reducer"
    stm = _stm_with_depth2(m)
    d0 = stm._depth.clone()
    m._stm_bounded_reduce_step(gate_tau=1.0)
    assert torch.equal(stm._depth, d0), "tau=1.0: nothing may fold"
    m._stm_bounded_reduce_step(gate_tau=0.0)
    assert bool((stm._depth == d0 - 1).all()), "tau=0.0: every row folds"


def test_gate_is_reduce_marginal():
    """The gate is the DP reduce marginal: folded rows are exactly those
    with g > tau (checked against the recorded routing)."""
    m = _build("data/MM_grammar.xml")
    stm = _stm_with_depth2(m)
    d0 = stm._depth.clone()
    m._stm_bounded_reduce_step(gate_tau=0.5)
    routing = getattr(m, "_stm_last_reduce_routing", None)
    assert routing is not None
    g = routing["reduce_marginal"][:, 0]
    folded = (stm._depth == d0 - 1)
    assert torch.equal(folded, g > 0.5), (g.tolist(), folded.tolist())


def test_no_reducer_is_noop():
    """Substrate-only grammar: no arity-2 op -> the scored step returns
    without touching STM (a mean-fold would be an unlicensed parse)."""
    m = _build("data/MM_masked_semantic.xml")
    assert m._stm_reducer() is None
    stm = _stm_with_depth2(m)
    d0 = stm._depth.clone()
    buf0 = stm._buffer.clone()
    m._stm_bounded_reduce_step(gate_tau=0.5)
    assert torch.equal(stm._depth, d0)
    assert torch.equal(stm._buffer, buf0)


def test_serial_loop_attempts_parse_per_word():
    """The serial per-word loop calls the scored step (a live incremental
    parse), with the configured <stmReduceTau> (default 0.5)."""
    m = _build("data/MM_grammar.xml")
    assert abs(float(getattr(m, "stm_reduce_tau")) - 0.5) < 1e-9
    calls = []
    orig = m._stm_bounded_reduce_step
    def spy(protect_depth=None, gate_tau=None, _o=orig, _c=calls):
        _c.append(gate_tau)
        return _o(protect_depth=protect_depth, gate_tau=gate_tau)
    m._stm_bounded_reduce_step = spy
    try:
        opt = m.getOptimizer(lr=0.01)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    finally:
        m._stm_bounded_reduce_step = orig
    scored = [t for t in calls if t is not None]
    assert scored, "no opportunistic parse step fired during serial reading"
    assert all(abs(t - 0.5) < 1e-9 for t in scored)
