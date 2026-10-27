"""Seen decay, stored location propagation, and unchanged tower projections."""

import pytest
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import sys

import torch

sys.path.insert(0, "bin")
from recon_bench import _build_model, _resolve_config


def _chain_cs(*, batch_size=2):
    """One placed neighbourhood crosses orders; farther rows need another step."""
    from test_review13_subspaces import fixture, word
    from test_priming_diffusion import directions
    from ConceptIndex import index_part_row
    cs, cb, _, alloc = fixture(meaning_pairs=2, batch_size=batch_size)
    for i in range(8):
        assert word(cs, [(4, 1.)]) == i
    rA = cs._csw_row_of(index_part_row(cs, cs.concept_id_at_row(3), cs.concept_id_at_row(7)))
    rB = cs._csw_row_of(index_part_row(cs, cs.concept_id_at_row(5), cs.concept_id_at_row(4)))
    derived = cb.mereology
    selected = [3,rA,5,rB,6]
    derived.placement[selected] = directions([0,10,16,50,60], dtype=torch.float32)
    derived.placement_witnesses[selected] = 1
    derived.placement_valid[[rA,rB]] = True
    cs._priming_boosts.fill_(1.)
    cs._geometric_spread = .5
    cs._geometric_radius_rate = 0.
    cs._priming_decay = 1.
    return cs, rA, rB


def test_membership_cache_is_directed_and_weighted():
    from Priming import part_edges
    cs, rA, rB = _chain_cs()
    src, dst, w = part_edges(cs)
    assert set(zip(src.tolist(), dst.tolist())) == {(3,rA), (5,rB)}
    assert bool((w == 1).all())


def test_energy_transfers_not_amplifies():
    cs, rA, rB = _chain_cs()
    cs.prime_seen(torch.tensor([3]), bump=1.)
    assert cs._priming_boosts[0,3] == 1.5 and cs._priming_boosts[0,rA] == 1.5
    e0 = (cs._priming_boosts[0]-1).double().sum()
    cs.prime_seen(torch.tensor([3]), bump=1.)
    e1 = (cs._priming_boosts[0]-1).double().sum()
    assert e1 == pytest.approx(float(e0)+1, abs=1e-9*cs.nVectors)
    audit = cs._last_priming_diffusion
    torch.testing.assert_close(audit['before'],audit['after'],rtol=0,atol=1e-9*cs.nVectors)


def test_successive_primes_propagate_further():
    cs, rA, rB = _chain_cs()
    cs.prime_seen(torch.tensor([3]), bump=1.)
    assert cs._priming_boosts[0,5] == cs._priming_boosts[0,rB] == 1
    before = cs._priming_boosts.clone()
    cs.prime_seen(torch.tensor([-1]), bump=0., decay=.8)
    assert cs._priming_boosts[0,5] > 1 and cs._priming_boosts[0,rB] == 1
    assert cs._priming_boosts[0,5]-1 == pytest.approx(.8*.5, abs=1e-7)
    assert (cs._priming_boosts[0]-1).double().sum() == pytest.approx(float((before[0]-1).sum())*.8,abs=1e-6)


def test_zero_spread_is_pure_decay_bump():
    cs, _rA, _rB = _chain_cs()
    cs._geometric_spread = 0.
    b = cs.prime_seen(torch.tensor([3,5]), bump=1., decay=.5)
    assert float(b[0,3]) == 2. and float(b[0,0]) == 1.
    b = cs.prime_seen(torch.tensor([5]), bump=1., decay=.5)
    assert float(cs._priming_boosts[0,3]) == 1.5
    assert float(cs._priming_boosts[0,5]) == 2.5
    assert float(b[0,3]) == pytest.approx(1+.5/1.5)
    assert float(b[0,5]) == 2.


def test_geometric_controls_constructed_with_standing_defaults():
    from test_review13_subspaces import fixture
    import math
    cs, *_ = fixture()
    assert cs._geometric_spread == .05 and cs._geometric_neighbours == 8
    assert cs._geometric_radius.item() == pytest.approx(math.radians(15))
    assert cs._geometric_radius_rate == .02
    assert cs._geometric_radius_min == math.radians(1)
    assert cs._geometric_radius_max == math.radians(90)


# ---- CS->PS / CS->WS heat projection (Alec 2026-07-12) --------------------


def _projected_model(epochs=3):
    from slow_budget import slow_count
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
    for e in range(slow_count(epochs)):
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




def test_priming_row_order_read_is_retired():
    from Spaces import Space, ConceptualSpace
    assert not hasattr(Space, 'priming_row_orders')
    assert not hasattr(ConceptualSpace, 'priming_row_orders')
