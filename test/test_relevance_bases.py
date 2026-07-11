"""Three bases of relevance -- stub contract (Architecture sec C, 2026-07-11).

Relevance (per-tower weights) -> attention (the CS top-K readout) ->
awareness (the winners). The bases are stubbed default-None (byte-identical);
the CS readout consumes ``_relevance_priority`` as a RANKING bias.
cpu/eager.
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


def test_three_bases_contract_default_dark():
    """PS/WS/SS all expose relevance_weights(); default None (dark)."""
    m = _build("data/MM_sparse_concept.xml")
    towers = [m.perceptualSpace, m.wholeSpace]
    if m.symbolSpace is not None:
        towers.append(m.symbolSpace)
    assert len(towers) == 3, "symbolTower config must expose all three"
    for t in towers:
        assert hasattr(t, "relevance_weights"), type(t).__name__
        assert t.relevance_weights() is None, type(t).__name__


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
    """A strong priority on the weak row outranks raw |activation|."""
    cs, rA, rB = _minted_cs()
    a_0 = torch.zeros(8, 1)
    a_0[3, 0] = 0.9
    a_0[5, 0] = 0.9
    what = torch.randn(16, 32)
    cs.cs_forward_content(a_0, what)
    base_first = int(cs._cs_level_rows[1][0, 0])   # rank-1 winner, no priority
    assert base_first == rA, "raw |act| must rank the strong row first"
    prio = torch.ones(16)
    prio[rB] = 100.0
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        cs.cs_forward_content(a_0, what)
        assert int(cs._cs_level_rows[1][0, 0]) == rB, (
            "priority must rerank the top-K selection")
    finally:
        object.__setattr__(cs, "_relevance_priority", None)


def test_priority_never_distorts_activations():
    """Ranking bias only: winner ACTIVATION values are priority-free."""
    cs, rA, rB = _minted_cs()
    a_0 = torch.zeros(8, 1)
    a_0[3, 0] = 0.9
    a_0[5, 0] = 0.9
    what = torch.randn(16, 32)
    _c, acts_plain = cs.cs_forward_content(a_0, what)
    prio = torch.ones(16)
    prio[rB] = 100.0
    object.__setattr__(cs, "_relevance_priority", prio)
    try:
        _c, acts_prio = cs.cs_forward_content(a_0, what)
    finally:
        object.__setattr__(cs, "_relevance_priority", None)
    # Both rows stay selected (keep == n_alloc here); values identical.
    assert torch.allclose(acts_plain[rA], acts_prio[rA])
    assert torch.allclose(acts_plain[rB], acts_prio[rB])
