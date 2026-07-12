"""Unconditional priming + energy dissipation (Alec 2026-07-12).

Three laws on the SEEN surface:
  1. UNCONDITIONAL: priming is perception bookkeeping -- the SEEN writes
     fire on every batch, no ``<relevance>`` gate (the knob now gates only
     the hard-coded reading-scope consumer).
  2. DISSIPATION: each prime event decays the standing energy toward
     neutral AND transfers a ``primingSpread`` fraction of each connected
     row's energy to its neighbors (per-source normalized over |edge|
     weights) -- energy moves, it is never amplified.
  3. PROPAGATION: successive primes push energy further out -- after k
     events a source row's energy has reached its k-hop neighborhood in
     the concept store.

cpu/eager, seeded.
"""
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import sys

import torch

sys.path.insert(0, "bin")
from recon_bench import _build_model, _resolve_config


def _chain_cs():
    """Sparse-active CS with an order-0 chain 3 -> 5 -> 6 (two hops) via
    two order-1 relation rows (fixture style from test_cs_sparse_weights)."""
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=16, order=1)
    rA = _mint_row(cs, 1, 301)
    rB = _mint_row(cs, 1, 302)
    cs.add_concept_edge(rA, 3, weight=1.0)
    cs.add_concept_edge(rA, 5, weight=1.0)     # A bridges 3 <-> 5
    cs.add_concept_edge(rB, 5, weight=1.0)
    cs.add_concept_edge(rB, 6, weight=1.0)     # B bridges 5 <-> 6
    object.__setattr__(cs, "_priming_boosts", None)
    object.__setattr__(cs, "_priming_spread", 0.5)
    object.__setattr__(cs, "_priming_decay", 1.0)   # isolate the transfer law
    return cs, rA, rB


def test_edges_exposed_undirected():
    """The concept store's |edges| back the priming diffusion graph."""
    cs, rA, rB = _chain_cs()
    edges = cs._priming_edges()
    assert edges is not None
    src, dst, w = edges
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert (3, rA) in pairs and (rA, 3) in pairs, "edges must be undirected"
    assert (rB, 6) in pairs and (6, rB) in pairs
    assert bool((w > 0).all()), "diffusion weights are |magnitudes|"


def test_energy_transfers_not_amplifies():
    """One prime event moves a spread-fraction of standing energy to the
    neighbors; the surface total (deviation from neutral) never grows."""
    cs, rA, rB = _chain_cs()
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    e0 = float((b - 1.0).sum())
    assert abs(float(b[3]) - 2.0) < 1e-6, "first event: bump lands on 3"
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[rA]) > 1.0, "row 3's energy must reach its neighbor rA"
    assert float(b[3]) < 3.0, "the moved fraction leaves the source"
    e1 = float((b - 1.0).sum())
    assert e1 <= e0 + 1.0 + 1e-5, "transfer conserves; only the bump adds"


def test_successive_primes_propagate_further():
    """Energy reaches the 2-hop neighborhood only after repeated events."""
    cs, rA, rB = _chain_cs()
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[5]) == 1.0 and float(b[rB]) == 1.0, "no 2-hop on event 1"
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    hop1 = float(b[rA])
    assert hop1 > 1.0
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[5]) > 1.0, "event 3: energy crosses rA into row 5"
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[rB]) > 1.0, "successive primes keep propagating outward"


def test_zero_spread_is_pure_decay_bump():
    """spread=0 keeps the original decay+bump law byte-identical."""
    cs, _rA, _rB = _chain_cs()
    object.__setattr__(cs, "_priming_spread", 0.0)
    b = cs.prime_seen(torch.tensor([3, 5]), bump=1.0, decay=0.5)
    assert float(b[3]) == 2.0 and float(b[0]) == 1.0
    b = cs.prime_seen(torch.tensor([5]), bump=1.0, decay=0.5)
    assert abs(float(b[3]) - 1.5) < 1e-6
    assert abs(float(b[5]) - 2.5) < 1e-6


def test_priming_spread_knob_stamped():
    """<primingSpread> reaches the towers (default 0.25, live)."""
    model, *_ = _build_model(_resolve_config("data/MM_masked_semantic.xml"))
    for sp in list(model.conceptualSpaces) + list(model.wholeSpaces):
        assert abs(float(getattr(sp, "_priming_spread")) - 0.25) < 1e-9


# ---- CS->PS / CS->WS heat projection (Alec 2026-07-12) --------------------


def _projected_model(epochs=3):
    torch.manual_seed(7)
    m, *_ = _build_model(_resolve_config("data/MM_masked_semantic.xml"))
    opt = m.getOptimizer(lr=0.01)
    for e in range(epochs):
        torch.manual_seed(1000 + e)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    return m


def test_cs_to_ps_projection_live():
    """The PS surface warms through the projection ALONE (nothing else
    writes PS priming), on the word triples' pid rows."""
    m = _projected_model()
    cs0 = m.conceptualSpaces[0]
    bridge = getattr(cs0, "_priming_bridge", None)
    assert bridge, "the mint must record the projection bridge"
    b_ps = m.perceptualSpace.priming_weights()
    assert torch.is_tensor(b_ps), "PS surface must exist post-projection"
    assert float(b_ps.max()) > 1.0, "word pids must carry projected heat"
    # The heated rows are exactly bridge-mapped pids (no scatter elsewhere).
    store = cs0._relation_store()
    mapped = {store._ps_pos_to_row[p]
              for parts, _w in bridge.values() for p in parts
              if store._pos_kind.get(p) == 'ps' and p in store._ps_pos_to_row}
    hot = set((b_ps > 1.0).nonzero().reshape(-1).tolist())
    assert hot and hot <= mapped, (sorted(hot)[:8], sorted(mapped)[:8])


def test_cs_to_ws_projection_live():
    """The terminal WS surface (the reading scope's heat source on sO=0
    reading configs) carries projected word-whole heat."""
    m = _projected_model()
    cs0 = m.conceptualSpaces[0]
    store = cs0._relation_store()
    bridge = cs0._priming_bridge
    b_ws = m.wholeSpaces[-1].priming_weights()
    assert torch.is_tensor(b_ws)
    V = int(b_ws.shape[0])
    mapped = {store._ws_pos_to_row[w]
              for _p, w in bridge.values()
              if w is not None and store._pos_kind.get(w) in ('ws', 'meta')
              and w in store._ws_pos_to_row}
    in_range = {r for r in mapped if r < V}
    assert in_range, "at least one word-whole row must fit the WS surface"
    assert any(float(b_ws[r]) > 1.0 for r in in_range), (
        "projected word-whole heat must reach the WS surface")


def test_projection_bounded_across_epochs():
    """The destination decay event bounds the projected accumulation
    (steady state ~ gain * e / (1 - decay), not unbounded growth)."""
    m6 = _projected_model(epochs=6)
    b6 = m6.perceptualSpace.priming_weights()
    assert float(b6.max()) < 1000.0, "projection must not grow unbounded"


# ---- canonical order-indexed priming surface (Alec 2026-07-12) ------------


def test_one_canonical_surface_per_tower():
    """Per-stage WS/CS delegate to the ONE canonical surface (terminal WS,
    stage-0 CS): a write through any stage is visible through every other,
    and it is the same tensor object."""
    import torch as _t
    m = _projected_model(epochs=1)
    ws_c = m.wholeSpaces[-1]
    for ws in m.wholeSpaces[:-1]:
        assert ws._priming_target() is ws_c
    for cs in m.conceptualSpaces[1:]:
        assert cs._priming_target() is m.conceptualSpaces[0]
    object.__setattr__(ws_c, "_priming_boosts", None)
    m.wholeSpaces[0].prime_seen(_t.tensor([1]), decay=1.0)
    b0 = m.wholeSpaces[0].priming_weights()
    bc = ws_c.priming_weights()
    assert b0 is bc, "delegation must land on the one canonical tensor"
    assert float(bc[1]) > 1.0
    assert getattr(m.wholeSpaces[0], "_priming_boosts", None) is None, (
        "no per-stage surface may exist on a delegating tower")


def test_reading_heat_from_canonical_on_multistage():
    """The acceptance pin for the ws0-vs-terminal split: on a MULTI-STAGE
    config the reading scope follows heat on the CANONICAL surface (where
    the CS->WS projection lands), not a per-stage ws0 copy."""
    import torch as _t
    m = _projected_model(epochs=1)
    assert len(m.wholeSpaces) > 1, "the pin needs a multi-stage config"
    ws0, ws_c = m.wholeSpaces[0], m.wholeSpaces[-1]
    V = ws_c._priming_dim()
    spans = _t.tensor([[[0, 5], [6, 11]]], dtype=_t.float32)
    idx = _t.tensor([[2, 5]]).clamp(max=V - 1)
    object.__setattr__(ws0, "_staged_analysis_spans", spans)
    object.__setattr__(ws_c, "_stage0_indices", idx)
    object.__setattr__(ws_c, "_priming_boosts", None)
    ws_c.prime_seen(idx[0, 1:2], bump=5.0, decay=1.0)  # heat slot 1's row
    try:
        m._primed_reading_step()
        scope = getattr(ws0, "_passback_scope_where", None)
        assert _t.is_tensor(scope), "canonical heat must reach the scope"
        assert scope[0].tolist() == [6.0, 11.0]
    finally:
        for attr in ("_staged_analysis_spans", "_passback_scope_where"):
            object.__setattr__(ws0, attr, None)
        for attr in ("_stage0_indices", "_priming_boosts"):
            object.__setattr__(ws_c, attr, None)


def test_priming_rows_carry_abstraction_order():
    """The canonical surface is order-indexed: priming_row_orders reads
    the codebook's ramsification table, one order per row."""
    m = _projected_model(epochs=1)
    ws_c = m.wholeSpaces[-1]
    orders = ws_c.priming_row_orders()
    assert orders is not None and orders.dim() == 1
    assert int(orders.shape[0]) == ws_c._priming_dim()
    cb = ws_c.subspace.codebook()
    for r in range(min(8, int(orders.shape[0]))):
        assert int(orders[r]) == int(cb.abstraction_order(r))
    # Delegating stages see the SAME order vector.
    o0 = m.wholeSpaces[0].priming_row_orders()
    assert o0 is not None and bool((o0 == orders).all())
