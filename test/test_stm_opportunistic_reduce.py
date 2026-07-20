"""Opportunistic (scored) STM reduce -- the 2b-2 refinement (Alec 2026-07-12).

A "normal" grammatical parse runs on STM DURING the serial phase: after
each word's insertion, a scored gate ``g`` (the reducer DP's reduce-vs-copy
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


def test_grammar_confidence_is_neutral_to_reduce_rule_count():
    """Duplicating equally-scored grammatical operators must not turn a
    neutral INSERT/BINARY decision from 1/2 into 2/3, 10/11, ... ."""
    m = _build("data/MM_grammar.xml")
    copy = torch.zeros(3, 2, 1)
    for n_rules in (1, 2, 10):
        routing = {
            "copy_score": copy,
            "reduce_score": torch.zeros(3, 1, n_rules),
        }
        confidence = m._stm_grammar_reduce_confidence(routing)
        assert torch.allclose(confidence, torch.full((3,), 0.5))


def test_occupancy_pressure_is_monotonic_and_reaches_zero_threshold():
    m = _build("data/MM_grammar.xml")
    depth = torch.arange(2, 9)
    threshold, pressure = m._stm_occupancy_threshold(
        depth, capacity=8, base_tau=0.75)
    assert threshold[0].item() == 0.75
    assert threshold[-1].item() == 0.0
    assert pressure[0].item() == 0.0
    assert pressure[-1].item() == 1.0
    assert bool((threshold[1:] <= threshold[:-1]).all())
    assert bool((pressure[1:] >= pressure[:-1]).all())


def test_full_stm_demands_grammar_even_when_tau_disallows_soft_reduce():
    """At low occupancy tau=1 rejects; at capacity the same setting is
    overridden by the hard memory demand and the best grammar op fires."""
    m = _build("data/MM_grammar.xml")
    stm = _stm_with_depth2(m, B=2)
    low = stm._depth.clone()
    m._stm_bounded_reduce_step(
        gate_tau=1.0, occupancy_pressure=True)
    assert torch.equal(stm._depth, low)

    D = int(stm.concept_dim)
    while int(stm._depth.max().item()) < int(stm.capacity):
        stm.push_step_masked(
            torch.randn(2, D), torch.ones(2, 1, dtype=torch.bool))
    full = stm._depth.clone()
    reduced = m._stm_bounded_reduce_step(
        gate_tau=1.0, occupancy_pressure=True)
    assert bool(reduced.all())
    assert torch.equal(stm._depth, full - 1)
    routing = m._stm_last_reduce_routing
    assert bool(routing["stm_demand_mask"].all())


def test_capacity_demand_without_grammar_fails_loudly():
    m = _build("data/MM_masked_semantic.xml")
    assert m._stm_reducer() is None
    stm = _stm_with_depth2(m, B=1)
    D = int(stm.concept_dim)
    while int(stm._depth.max().item()) < int(stm.capacity):
        stm.push_step_masked(
            torch.randn(1, D), torch.ones(1, 1, dtype=torch.bool))
    try:
        m._stm_bounded_reduce_step(
            row_gate=torch.ones(1, dtype=torch.bool), demand=True)
    except RuntimeError as exc:
        assert "no binary grammatical operator" in str(exc)
    else:
        raise AssertionError("capacity demand silently used a non-grammar fold")


def test_serial_loop_attempts_parse_per_word():
    """The serial per-word loop calls the scored step (a live incremental
    parse), with the configured <stmReduceTau> (default 0.5)."""
    m = _build("data/MM_grammar.xml")
    assert abs(float(getattr(m, "stm_reduce_tau")) - 0.5) < 1e-9
    # The fixture's eight WS rows are stable upstream property classes.
    # Serial parsing must mint word/object/META relations downstream in CS,
    # never consume WholeSpace rows through the retired META allocator.
    assert m.wholeSpace.property_basis is True
    assert not hasattr(m.wholeSpace, "_paired_next_row")
    calls = []
    orig = m._stm_bounded_reduce_step
    def spy(protect_depth=None, gate_tau=None, _o=orig, _c=calls,
            **controller):
        _c.append(gate_tau)
        return _o(protect_depth=protect_depth, gate_tau=gate_tau,
                  **controller)
    m._stm_bounded_reduce_step = spy
    try:
        opt = m.getOptimizer(lr=0.01)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    finally:
        m._stm_bounded_reduce_step = orig
    assert not hasattr(m.wholeSpace, "_paired_next_row")
    assert not hasattr(m.wholeSpace, "meta_pair_to_idx")
    scored = [t for t in calls if t is not None]
    assert scored, "no opportunistic parse step fired during serial reading"
    assert all(abs(t - 0.5) < 1e-9 for t in scored)
