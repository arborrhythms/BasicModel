"""The simplified relevance law (Architecture sec C, 2026-07-11):
ONE quadratic priming surface per space -- SEEN rows primed by being
perceived (bump + exponential decay), DESIRED/HATED rows by signed intent
(suppression floors at 0, never a veto). The CS surface feeds the pyramid's
top-K as a ranking score; readingAttention is hard-coded over the WS
surface. Priming is UNCONDITIONAL (Alec 2026-07-12): the SEEN writes and
the pyramid's priority read fire on every batch; ``<relevance>`` gates
only the hard-coded reading-scope consumer. cpu/eager.
"""
import os
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
import functools
import sys

sys.path.insert(0, "bin")
import torch

from recon_bench import _build_model, _resolve_config


@functools.lru_cache(maxsize=None)
def _build(cfg):
    model, *_ = _build_model(_resolve_config(cfg))
    return model


def test_surface_contract_cold_at_build():
    """All three towers expose the surface APIs; None until a batch has
    primed them (writes are unconditional but perception-driven), and
    relevance_weights IS the priming surface. <relevance> still defaults
    off -- it now gates only the reading-scope consumer."""
    m = _build("data/MM_sparse_concept.xml")
    towers = [m.perceptualSpace, m.wholeSpace]
    if m.symbolSpace is not None:
        towers.append(m.symbolSpace)
    assert len(towers) == 3
    for t in towers:
        assert t.priming_weights() is None, type(t).__name__
        assert t.relevance_weights() is None, type(t).__name__
    assert getattr(m, "relevance_on", None) is False, "default must be off"


def test_seen_bump_and_decay():
    """SEEN: bump the fired rows; the surface decays toward neutral."""
    m = _build("data/MM_sparse_concept.xml")
    cs0 = m.conceptualSpaces[0]
    b = cs0.prime_seen(torch.tensor([3, 5]), bump=1.0, decay=0.5)
    assert b is not None and float(b[3]) == 2.0 and float(b[0]) == 1.0
    # A second write decays the old bump toward neutral before bumping.
    b = cs0.prime_seen(torch.tensor([5]), bump=1.0, decay=0.5)
    assert abs(float(b[3]) - 1.5) < 1e-6, "old seen decays toward 1.0"
    assert abs(float(b[5]) - 2.5) < 1e-6, "refreshed seen re-bumps"
    object.__setattr__(cs0, "_priming_boosts", None)


def test_desire_signed_floor():
    """DESIRED (+) boosts; HATED (-) suppresses with floor 0 (no veto)."""
    m = _build("data/MM_sparse_concept.xml")
    cs0 = m.conceptualSpaces[0]
    b = cs0.prime_desire(torch.tensor([2]), valence=1.0)
    assert float(b[2]) == 2.0
    b = cs0.prime_desire(torch.tensor([4]), valence=-1.0)
    assert float(b[4]) == 0.0, "hate suppresses"
    b = cs0.prime_desire(torch.tensor([4]), valence=-5.0)
    assert float(b[4]) == 0.0, "floor at 0 -- suppression, never negative"
    object.__setattr__(cs0, "_priming_boosts", None)


def test_unconditional_integration_end_to_end():
    """No <relevance> needed: awareness primes (pyramid winners write the
    CS surface) on every batch, and the pyramid consumes the surface as
    its ranking score on the next one."""
    m = _build("data/MM_sparse_concept.xml")
    opt = m.getOptimizer(lr=0.01)
    cs0 = m.conceptualSpaces[0]
    assert m.relevance_on is False
    try:
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
        b = cs0.priming_weights()
        assert torch.is_tensor(b) and float(b.max()) > 1.0, (
            "admitted rows must prime the surface unconditionally")
        m.runEpoch(optimizer=opt, batchSize=4, split="train", max_batches=1)
        prio = getattr(cs0, "_relevance_priority", None)
        assert torch.is_tensor(prio), "the surface must feed the pyramid"
        assert float(prio.max()) > 0.0
    finally:
        object.__setattr__(cs0, "_priming_boosts", None)
        object.__setattr__(cs0, "_relevance_priority", None)
        object.__setattr__(m.wholeSpaces[0], "_priming_boosts", None)
        for _ws in m.wholeSpaces:
            object.__setattr__(_ws, "_priming_boosts", None)


def test_primed_reading_scope():
    """Hard-coded readingAttention: scope = the span of the hottest-primed
    word-whole. Spans are staged on ws0 (the stem's surface); the slot
    selections and the heat live on the CANONICAL terminal codebook
    (synthetic state; the learned producer's contract)."""
    m = _build("data/MM_sparse_concept.xml")
    ws0 = m.wholeSpaces[0]
    ws_c = m.wholeSpaces[-1]
    assert ws0._priming_target() is ws_c, "per-stage WS must delegate"
    V = ws0._priming_dim()                 # canonical dim via delegation
    assert V > 0, "WS must carry a codebook surface"
    spans = torch.tensor([[[0, 5], [6, 11]], [[0, 3], [4, 9]]],
                         dtype=torch.float32)
    idx = torch.tensor([[2, 5], [2, 5]]).clamp(max=V - 1)
    object.__setattr__(ws0, "_staged_analysis_spans", spans)
    object.__setattr__(ws_c, "_stage0_indices", idx)
    object.__setattr__(ws_c, "_priming_boosts", None)  # fresh surface
    ws0.prime_desire(idx[0, 1:2], valence=3.0)  # delegates to the canonical
    try:
        m._primed_reading_step()
        scope = getattr(ws0, "_passback_scope_where", None)
        assert torch.is_tensor(scope)
        assert scope[0].tolist() == [6.0, 11.0], "hottest whole wins"
        assert scope[1].tolist() == [4.0, 9.0]
    finally:
        for attr in ("_staged_analysis_spans", "_passback_scope_where"):
            object.__setattr__(ws0, attr, None)
        for attr in ("_stage0_indices", "_priming_boosts"):
            object.__setattr__(ws_c, attr, None)


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
    cs.cs_forward_content(a_0, what)
    assert int(cs._cs_level_rows[1][0, 0]) == rA
    prio = torch.zeros(16)
    prio[rB] = 100.0
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        cs.cs_forward_content(a_0, what)
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
    _c, acts_plain = cs.cs_forward_content(a_0, what)
    prio = torch.zeros(16)
    prio[rB] = 100.0
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        _c, acts_prio = cs.cs_forward_content(a_0, what)
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
    cs.cs_forward_content(a_0, what)
    assert int(cs._cs_level_rows[1][0, 0]) == rA
    prio = torch.zeros(16)
    prio[5] = 100.0                    # rB's CONSTITUENT (order-0 row 5)
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        cs.cs_forward_content(a_0, what)
        assert int(cs._cs_level_rows[1][0, 0]) == rB, (
            "constituent priority must spread up through |W|")
    finally:
        object.__setattr__(cs, "_relevance_priority", None)


def test_symbol_history_projection():
    """Heat over symbols lands on the rows of the concepts that reference
    them through ('sym', id) constituent records (the SS->CS bridge)."""
    from test_cs_symbol_table import _cs_sparse_active
    cs = _cs_sparse_active()
    A1, _B1, C1 = cs.create_word_object_meta([1], 2, key="w1")
    row_meta = cs._csw_row_of(C1)
    assert row_meta is not None
    p = cs.symbol_history_priority({int(A1): 5.0})
    assert p is not None and float(p[row_meta]) >= 5.0
    assert cs.symbol_history_priority({}) is None
    assert cs.symbol_history_priority(None) is None
