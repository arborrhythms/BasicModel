"""The simplified relevance law (Architecture sec C, 2026-07-11):
ONE quadratic priming surface per space -- SEEN rows primed by being
perceived (bump + exponential decay), DESIRED/HATED rows by signed intent
(suppression floors at 0, never a veto). The CS surface feeds the pyramid's
top-K as a ranking score; readingAttention is hard-coded over the WS
surface. Priming is UNCONDITIONAL (Alec 2026-07-12): the SEEN writes and
the pyramid's priority read fire on every batch; ``<relevance>`` gates
only the hard-coded reading-scope consumer. cpu/eager.
"""

import pytest
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import sys

sys.path.insert(0, "bin")
import torch
from test_cs_sparse_weights import _evidence

from recon_bench import _build_model, _resolve_config


def _build(cfg):
    from configuration_fixtures import parallel_concepts
    with parallel_concepts() as path:
        model, *_ = _build_model(_resolve_config(path))
    return model


@pytest.mark.slow
def test_surface_contract_cold_at_build():
    """Every tower starts with its constructed neutral surface; consumers
    read the canonical bounded view before any perception write."""
    m = _build("data/MM_20M_xor.xml")
    towers = [m.perceptualSpace, m.wholeSpace]
    if m.symbolSpace is not None:
        assert m.symbolSpace._priming_dim() == 0
        assert m.symbolSpace.priming_weights() is None
        assert m.symbolSpace.relevance_weights() is None
    assert len(towers) == 2
    for t in towers:
        priming, relevance = t.priming_weights(), t.relevance_weights()
        assert priming.shape == t._priming_target()._priming_boosts.shape
        assert priming.eq(1).all(), type(t).__name__
        torch.testing.assert_close(priming, relevance, atol=0, rtol=0)
    assert getattr(m, "relevance_on", None) is False, "default must be off"


@pytest.mark.slow
def test_seen_bump_and_decay():
    """SEEN: bump the fired rows; the surface decays toward neutral."""
    m = _build("data/MM_20M_xor.xml")
    cs0 = m.conceptualSpaces[0]
    b = cs0.prime_seen(torch.tensor([3, 5]), bump=1.0, decay=0.5)
    assert b is not None and float(b[0, 3]) == 2.0 and float(b[0, 0]) == 1.0
    # A second write decays the old bump toward neutral before bumping.
    b = cs0.prime_seen(torch.tensor([5]), bump=1.0, decay=0.5)
    raw = cs0._priming_target()._priming_boosts
    assert float(raw[0, 3]) == 1.5, "old seen decays toward 1.0"
    assert float(raw[0, 5]) == 2.5, "refreshed seen re-bumps"
    assert abs(float(b[0, 3]) - (1 + .5/1.5)) < 1e-6
    assert float(b[0, 5]) == 2., "the consumer uses the unchanged bounded read"
    cs0._priming_boosts.fill_(1.)


@pytest.mark.slow
def test_desire_signed_floor():
    """DESIRED (+) boosts; HATED (-) suppresses with floor 0 (no veto)."""
    m = _build("data/MM_20M_xor.xml")
    cs0 = m.conceptualSpaces[0]
    b = cs0.prime_desire(torch.tensor([2]), valence=1.0)
    assert float(b[0, 2]) == 2.0
    b = cs0.prime_desire(torch.tensor([4]), valence=-1.0)
    assert float(b[0, 4]) == 0.0, "hate suppresses"
    b = cs0.prime_desire(torch.tensor([4]), valence=-5.0)
    assert float(b[0, 4]) == 0.0, "floor at 0 -- suppression, never negative"
    cs0._priming_boosts.fill_(1.)


@pytest.mark.slow
def test_unconditional_integration_end_to_end(monkeypatch):
    """No <relevance> needed: awareness primes (pyramid winners write the
    CS surface) on every batch, and the pyramid consumes the surface as
    its ranking score on the next one."""
    m = _build("data/MM_20M_xor.xml")
    opt = m.getOptimizer(lr=0.01)
    cs0 = m._concept_owner()
    assert m.relevance_on is False
    observed = []
    prime = m._prime_seen_step
    def capture():
        prime()
        heat = cs0.priming_weights(batch=m._priming_batch_size())
        priority = m._assemble_relevance_priority(cs0, 0, cs0, None)
        observed.append((heat.clone(), priority.clone()))
    monkeypatch.setattr(m, '_prime_seen_step', capture)
    try:
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
        assert len(observed) == 2
        for heat, priority in observed:
            assert float(heat.max()) > 1., "admitted rows prime before the document reset"
            torch.testing.assert_close(priority, (heat-1).T, atol=0, rtol=0)
            assert float(priority.max()) > 0., "the surface feeds the pyramid"
        assert cs0.priming_weights().eq(1).all()
    finally:
        cs0._priming_boosts.fill_(1.)
        object.__setattr__(cs0, "_relevance_priority", None)
        m.wholeSpaces[0]._priming_boosts.fill_(1.)
        for _ws in m.wholeSpaces:
            _ws._priming_boosts.fill_(1.)


@pytest.mark.slow
def test_native_attention_reads_canonical_whole_heat_by_stream():
    """The native attention reader sees canonical whole heat per stream."""
    from ModelCandidateAttention import _heat
    m = _build("data/MM_20M_xor.xml")
    ws0 = m.wholeSpaces[0]
    ws_c = m.wholeSpaces[-1]
    assert ws0._priming_target() is ws_c, "per-stage WS must delegate"
    V = ws0._priming_dim()                 # canonical dim via delegation
    assert V > 0, "WS must carry a codebook surface"
    idx = torch.tensor([[2, 5], [2, 5]]).clamp(max=V - 1)
    ws_c._priming_boosts.fill_(1.)  # fresh surface
    ws0.prime_desire(idx.gather(1, torch.tensor([[1], [0]])), valence=3.0)
    before = ws_c._priming_boosts.clone()
    try:
        heat = _heat(ws0, 2, V, before)
        assert heat.gather(1, idx).argmax(-1).tolist() == [1, 0]
        torch.testing.assert_close(heat, ws_c.relevance_weights(batch=2), atol=0, rtol=0)
        torch.testing.assert_close(ws_c._priming_boosts, before, atol=0, rtol=0)
    finally:
        ws_c._priming_boosts.fill_(1.)


def _minted_cs():
    """A sparse-active CS with two competing order-1 rows (fixture style
    from test_cs_sparse_weights: nVectors=16, K=1 -> caps (8, 4))."""
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=16, order=1)
    rA = _mint_row(cs, 1, 301)
    rB = _mint_row(cs, 1, 302)
    cs.add_concept_edge(rA, 3, weight=2.0)     # A: strong evidence
    cs.add_concept_edge(rB, 5, weight=0.5)     # B: weak evidence
    return cs, rA, rB


def test_priority_reranks_topk_selection():
    """A strong score on the weak row outranks raw |activation|."""
    cs, rA, rB = _minted_cs()
    a_0 = torch.zeros(8, 1)
    a_0[3, 0] = 0.9
    a_0[5, 0] = 0.9
    what = torch.randn(16, 32)
    cs.cs_forward_content(_evidence(a_0), what)
    assert int(cs._cs_level_rows[1][0, 0]) == rA
    prio = torch.zeros(16)
    prio[rB] = 100.0
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        cs.cs_forward_content(_evidence(a_0), what)
        assert int(cs._cs_level_rows[1][0, 0]) == rB
    finally:
        object.__setattr__(cs, "_relevance_priority", None)


def test_priority_never_distorts_activations():
    """Ranking bias only: winner ACTIVATION values are score-free."""
    cs, rA, rB = _minted_cs()
    a_0 = torch.zeros(8, 1)
    a_0[3, 0] = 0.9
    a_0[5, 0] = 0.9
    what = torch.randn(16, 32)
    _c, acts_plain = cs.cs_forward_content(_evidence(a_0), what)
    prio = torch.zeros(16)
    prio[rB] = 100.0
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        _c, acts_prio = cs.cs_forward_content(_evidence(a_0), what)
    finally:
        object.__setattr__(cs, "_relevance_priority", None)
    assert torch.allclose(acts_plain[rA], acts_prio[rA])
    assert torch.allclose(acts_plain[rB], acts_prio[rB])


def test_priority_spreads_through_edge_magnitudes():
    """Order-0 score reaches rung ranking via |W| (the hop); admitted rows
    only carry it upward."""
    cs, rA, rB = _minted_cs()
    a_0 = torch.zeros(8, 1)
    a_0[3, 0] = 0.9
    a_0[5, 0] = 0.9
    what = torch.randn(16, 32)
    cs.cs_forward_content(_evidence(a_0), what)
    assert int(cs._cs_level_rows[1][0, 0]) == rA
    prio = torch.zeros(16)
    prio[5] = 100.0                    # rB's CONSTITUENT (order-0 row 5)
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        cs.cs_forward_content(_evidence(a_0), what)
        assert int(cs._cs_level_rows[1][0, 0]) == rB, (
            "constituent priority must spread up through |W|")
    finally:
        object.__setattr__(cs, "_relevance_priority", None)


def test_symbol_history_projection():
    """Heat follows a genuine sigma reference; DEF is not a part edge."""
    from test_cs_symbol_table import _cs_sparse_active
    cs = _cs_sparse_active()
    _, obj = cs.interpret_word([1], 2, key="w1")
    parent = cs.singleton_concept(obj)
    row_object = cs._csw_row_of(parent)
    assert row_object is not None
    p = cs.symbol_history_priority({int(obj): 5.0})
    assert p is not None and float(p[row_object]) >= 5.0
    assert cs.symbol_history_priority({}) is None
    assert cs.symbol_history_priority(None) is None
