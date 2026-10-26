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

cpu/eager, unseeded.
"""

import pytest
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import sys

import torch

sys.path.insert(0, "bin")
from recon_bench import _build_model, _resolve_config


def _chain_cs(*, batch_size=2):
    """Native Part chain 3 -> A -> 5 -> B -> 6, with disjoint content."""
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=16, order=1, batch_size=batch_size)
    for i in range(8):
        assert _mint_row(cs, 0, 100 + i) == i
    rA = _mint_row(cs, 1, 301)
    rB = _mint_row(cs, 1, 302)
    from Spaces import _concept_alloc_of
    parts = _concept_alloc_of(cs).parts
    for a, b in ((3, rA), (rA, 5), (5, rB), (rB, 6)):
        parts.add(cs.concept_id_at_row(a), cs.concept_id_at_row(b))
    with torch.no_grad():
        cs.similarity_codebook.W.zero_()
        for i, row in enumerate((3, rA, 5, rB, 6)):
            cs.similarity_codebook.W[row, i] = 1.
    cs._priming_boosts.fill_(1.)
    object.__setattr__(cs, "_priming_spread", 0.5)
    object.__setattr__(cs, "_priming_decay", 1.0)   # isolate the transfer law
    return cs, rA, rB


def test_edges_exposed_undirected():
    """The common Part pairs back the priming diffusion graph."""
    cs, rA, rB = _chain_cs()
    edges = cs._priming_edges()
    assert edges is not None
    src, dst, w = edges
    pairs = set(zip(src.tolist(), dst.tolist()))
    assert (3, rA) in pairs and (rA, 3) in pairs, "edges must be undirected"
    assert (rB, 6) in pairs and (6, rB) in pairs
    assert bool((w == 1).all()), "each retained Part pair contributes one edge"


def test_energy_transfers_not_amplifies():
    """One prime event moves a spread-fraction of standing energy to the
    neighbors; the surface total (deviation from neutral) never grows."""
    cs, rA, rB = _chain_cs()
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    e0 = float((b - 1.0).sum())
    assert abs(float(b[0, 3]) - 2.0) < 1e-6, "first event: bump lands on 3"
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[0, rA]) > 1.0, "row 3's energy must reach its neighbor rA"
    assert float(b[0, 3]) < 3.0, "the moved fraction leaves the source"
    e1 = float((b - 1.0).sum())
    assert e1 <= e0 + 1.0 + 1e-5, "transfer conserves; only the bump adds"


def test_successive_primes_propagate_further():
    """Energy reaches the 2-hop neighborhood only after repeated events."""
    cs, rA, rB = _chain_cs()
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[0, 5]) == 1.0 and float(b[0, rB]) == 1.0, "no 2-hop on event 1"
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    hop1 = float(b[0, rA])
    assert hop1 > 1.0
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[0, 5]) > 1.0, "event 3: energy crosses rA into row 5"
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    b = cs.prime_seen(torch.tensor([3]), bump=1.0)
    assert float(b[0, rB]) > 1.0, "successive primes keep propagating outward"


def test_zero_spread_is_pure_decay_bump():
    """The energy ledger keeps decay+bump; retrieval normalizes its boosts."""
    cs, _rA, _rB = _chain_cs()
    object.__setattr__(cs, "_priming_spread", 0.0)
    b = cs.prime_seen(torch.tensor([3, 5]), bump=1.0, decay=0.5)
    assert float(b[0, 3]) == 2.0 and float(b[0, 0]) == 1.0
    b = cs.prime_seen(torch.tensor([5]), bump=1.0, decay=0.5)
    assert abs(float(cs._priming_boosts[0, 3]) - 1.5) < 1e-6
    assert abs(float(cs._priming_boosts[0, 5]) - 2.5) < 1e-6
    assert abs(float(b[0, 3]) - (1 + .5/1.5)) < 1e-6
    assert float(b[0, 5]) == 2.


@pytest.mark.slow
def test_priming_spread_knob_stamped():
    """<primingSpread> reaches the towers (default 0.05, live)."""
    from configuration_fixtures import parallel_concepts
    with parallel_concepts(wide=True) as path:
        model, *_ = _build_model(_resolve_config(path))
    for sp in list(model.conceptualSpaces) + list(model.wholeSpaces):
        assert abs(float(getattr(sp, "_priming_spread")) - 0.05) < 1e-9


# ---- CS->PS / CS->WS heat projection (Alec 2026-07-12) --------------------


def _projected_model(epochs=3):
    from configuration_fixtures import parallel_concepts
    with parallel_concepts(wide=True) as path:
        m, *_ = _build_model(_resolve_config(path))
    # Projection precedes the document-end hard reset. Observe its live
    # outputs there, rather than requiring heat to leak into the next document.
    m._projection_fixture = []
    original = m._prime_seen_step
    def seen():
        original()
        m._projection_fixture.append(tuple(
            None if value is None else value.detach().clone()
            for value in (m.perceptualSpace.priming_weights(),
                          m.wholeSpaces[-1].priming_weights())))
    m._prime_seen_step = seen
    opt = m.getOptimizer(lr=0.01)
    for e in range(epochs):
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
    return m


def _surface_descriptions(cs):
    from Spaces import _concept_alloc_of
    return [_concept_alloc_of(cs).word_descriptions.get(word) or cs.interpret._pending[word]
            for word in cs._word_definition_ids()]


@pytest.mark.slow
def test_cs_to_ps_projection_live():
    """The PS surface warms through the projection ALONE (nothing else
    writes PS priming), on the word triples' pid rows."""
    m = _projected_model()
    cs0 = m._concept_owner()
    bridge = _surface_descriptions(cs0)
    assert bridge, 'admitted words must carry native surface references'
    b_ps = m._projection_fixture[-1][0]
    assert torch.is_tensor(b_ps), "PS surface must exist post-projection"
    assert float(b_ps.max()) > 1.0, "word pids must carry projected heat"
    # The heated rows are exactly bridge-mapped pids (no scatter elsewhere).
    mapped = {int(p) for definition in bridge for group in definition['parts'] for p in group}
    hot = set((b_ps > 1.0).any(0).nonzero().reshape(-1).tolist())
    assert hot and hot <= mapped, (sorted(hot)[:8], sorted(mapped)[:8])
    assert m.perceptualSpace.priming_weights().eq(1).all()


@pytest.mark.slow
def test_cs_to_ws_projection_live():
    """The terminal WS surface (the reading scope's heat source on sO=0
    reading configs) carries projected word-whole heat."""
    m = _projected_model()
    cs0 = m._concept_owner()
    bridge = _surface_descriptions(cs0)
    b_ws = m._projection_fixture[-1][1]
    assert torch.is_tensor(b_ws)
    V = int(b_ws.shape[1])
    mapped = {int(row) for definition in bridge for row in definition['wholes']}
    in_range = {r for r in mapped if r < V}
    assert in_range, "at least one word-whole row must fit the WS surface"
    assert any(float(b_ws[:, r].max()) > 1.0 for r in in_range), (
        "projected word-whole heat must reach the WS surface")


@pytest.mark.slow
def test_projection_bounded_across_epochs():
    """The destination decay event bounds the projected accumulation
    (steady state ~ gain * e / (1 - decay), not unbounded growth)."""
    m6 = _projected_model(epochs=6)
    assert all(value is not None and float(value.max()) < 1000.
               for value, _ in m6._projection_fixture), "projection must not grow unbounded"


# ---- canonical order-indexed priming surface (Alec 2026-07-12) ------------


@pytest.mark.slow
def test_one_canonical_surface_per_tower():
    """Per-stage WS/CS delegate to the ONE canonical surface (terminal WS,
    native concept owner): a write through any stage is visible through every other,
    and every stage shares the same stored tensor."""
    import torch as _t
    m = _projected_model(epochs=1)
    ws_c = m.wholeSpaces[-1]
    for ws in m.wholeSpaces[:-1]:
        assert ws._priming_target() is ws_c
    for cs in m.conceptualSpaces:
        assert cs._priming_target() is m._concept_owner()
    ws_c._priming_boosts.fill_(1.)
    m.wholeSpaces[0].prime_seen(_t.tensor([1]), decay=1.0)
    b0 = m.wholeSpaces[0].priming_weights()
    bc = ws_c.priming_weights()
    torch.testing.assert_close(b0, bc, rtol=0, atol=0)
    assert m.wholeSpaces[0]._priming_target()._priming_boosts is ws_c._priming_boosts
    assert float(bc[0, 1]) > 1.0
    assert bool(m.wholeSpaces[0]._priming_boosts.eq(1).all()), (
        "a delegating tower must leave its unused local slab neutral")




@pytest.mark.slow
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
