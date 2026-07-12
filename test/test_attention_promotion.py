"""Attention-to-relation promotion (2026-07-12 execution of
doc/plans/2026-07-04-attention-to-relation-promotion.md).

The pyramid's admitted field feeds a bounded candidate cache at the
sentence boundary; recurrent shared-context member sets that clear the
truth_criterion learn-score law mint a higher-order whole. Byte-identical
when the <attentionPromotion> gate is off.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
import torch

import Spaces
from test_basicmodel import _populate_test_config

_D = 8


def _cs(nS=64, order=3, promote=True):
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    cs = Spaces.ConceptualSpace([nP, _D], [nS, _D], [nS, _D])
    object.__setattr__(cs, "_symbolic_order", order)
    object.__setattr__(cs, "_serial", False)
    if promote:
        object.__setattr__(cs, "_promotion_enabled", True)
    return cs


def _mint_order0(cs, n):
    """Mint ``n`` order-0 concepts with reserved snap rows; returns
    ``[(cid, row)]`` in allocation order (rows 0, 1, 2, ...)."""
    alloc = Spaces._concept_alloc_of(cs)
    out = []
    for _ in range(n):
        cid = alloc.new_concept()
        row = cs._csw_concept_row(0, cid)
        assert row is not None
        out.append((cid, int(row)))
    return out


def _observe(cs, active, B_rows=None):
    """Stage a synthetic admitted field (one batch row) on the promotion
    stashes and run the collector. ``active`` maps global row -> signed
    activation; the level-rows stash is the full snap block (order-0 rows
    are always staged by the pyramid)."""
    N = int(cs.nVectors)
    a = torch.zeros(N, 1)
    for r, v in active.items():
        a[int(r), 0] = float(v)
    snap = torch.arange(cs._order_caps()[0]).unsqueeze(-1)
    object.__setattr__(cs, "_promo_last_acts", a)
    object.__setattr__(cs, "_cs_level_rows", [snap])
    cs.promotion_observe()


def _mock_promotion_score(cs, children, obvious, resolves):
    """Monkeypatch the three factor seams (the Task-6c test convention)
    so the promotion learn-score product is deterministic."""
    cs._learn_score_members_in_codebook = (
        lambda vecs, _c=children: float(_c))
    cs._learn_score_is_truth_obvious = (
        lambda rel, _o=obvious: float(_o))
    cs._learn_score_resolves_contradiction = (
        lambda rel, _r=resolves: float(_r))


def _royal_rounds(cs, members, context, rounds=1):
    """One observation per member: the member row active together with the
    shared context rows ({king|queen|prince} + {crown, palace})."""
    for _ in range(rounds):
        for (_cid, row) in members:
            active = {row: 1.0}
            for i, (_c, r) in enumerate(context):
                active[r] = 0.8 - 0.1 * i
            _observe(cs, active)


# -- gate ---------------------------------------------------------------------

def test_default_off_is_inert():
    cs = _cs(promote=False)
    assert getattr(cs, "_promotion_enabled") is False
    cs.promotion_observe()
    assert cs.promotion_pass() == []
    assert getattr(cs, "_promotion_cache_state", None) is None
    # The cutover does not stash evidence when the gate is off.
    cs.eval()
    D = int(cs.similarity_codebook.getW().shape[-1])
    cs.cs_symbolic_phase(torch.randn(2, int(cs.nVectors), D))
    assert getattr(cs, "_promo_last_acts", None) is None


# -- the collector + candidate cache ------------------------------------------

def test_collector_accumulates_shared_context_support():
    cs = _cs()
    members = _mint_order0(cs, 3)                      # king, queen, prince
    context = _mint_order0(cs, 2)                      # crown, palace
    _royal_rounds(cs, members, context)
    st = cs._promotion_state()
    key = frozenset(r for (_c, r) in context)
    e = st["cache"].get(key)
    assert e is not None and e["support"] == 3
    for (_cid, row) in members:
        assert float(e["member_w"][row]) > 0.0


def test_near_context_folds_by_cosine():
    cs = _cs()
    (king, k_row), = _mint_order0(cs, 1)
    ctx = _mint_order0(cs, 3)                          # crown, palace, ruling
    (c0, r0), (c1, r1), (c2, r2) = ctx
    _observe(cs, {k_row: 1.0, r0: 0.8, r1: 0.7})
    _observe(cs, {k_row: 1.0, r0: 0.8, r1: 0.7, r2: 0.6})   # near, not exact
    cache = cs._promotion_state()["cache"]
    # The king-focal observation folded into the existing {crown, palace}
    # entry by cosine (no NEW {crown, palace, ruling} entry); sibling
    # focal-context entries (crown-focal etc.) are expected and separate.
    assert cache[frozenset({r0, r1})]["support"] == 2
    assert frozenset({r0, r1, r2}) not in cache


def test_cache_capacity_evicts_weakest():
    cs = _cs()
    object.__setattr__(cs, "_promotion_cache_cap", 2)
    focal = _mint_order0(cs, 1)[0]
    rows = _mint_order0(cs, 6)
    # Three disjoint contexts (pairwise cosine 0) -> third insert evicts.
    for i in range(3):
        pair = rows[2 * i:2 * i + 2]
        _observe(cs, {focal[1]: 1.0,
                      pair[0][1]: 0.8, pair[1][1]: 0.7})
    assert len(cs._promotion_state()["cache"]) == 2


def test_nonfinite_acts_fail_loud():
    cs = _cs()
    N = int(cs.nVectors)
    a = torch.zeros(N, 1)
    a[0, 0] = float("nan")
    object.__setattr__(cs, "_promo_last_acts", a)
    object.__setattr__(cs, "_cs_level_rows",
                       [torch.arange(cs._order_caps()[0]).unsqueeze(-1)])
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        cs.promotion_observe()


# -- the acceptance law (todo.md: accept iff score >= tc AND tc < 1) ----------

def test_tc_one_promotes_nothing():
    cs = _cs()
    members = _mint_order0(cs, 3)
    context = _mint_order0(cs, 2)
    _royal_rounds(cs, members, context)
    _mock_promotion_score(cs, 1.0, 1.0, 1.0)           # perfect score
    cs.truth_criterion = 1.0
    assert cs.promotion_pass() == []                   # tc=1: NOTHING learned


def test_tc_zero_promotes_even_zero_score():
    cs = _cs()
    members = _mint_order0(cs, 3)
    context = _mint_order0(cs, 2)
    _royal_rounds(cs, members, context)
    _mock_promotion_score(cs, 0.0, 0.0, 0.0)           # zero score
    cs.truth_criterion = 0.0
    assert len(cs.promotion_pass()) == 1               # tc=0: everything


def test_score_product_gates_promotion():
    cs = _cs()
    members = _mint_order0(cs, 3)
    context = _mint_order0(cs, 2)
    _royal_rounds(cs, members, context)
    cs.truth_criterion = 0.5
    _mock_promotion_score(cs, 0.9, 0.9, 0.5)           # 0.405 < 0.5
    assert cs.promotion_pass() == []
    _mock_promotion_score(cs, 0.9, 0.9, 0.9)           # 0.729 >= 0.5
    assert len(cs.promotion_pass()) == 1


def test_below_min_support_never_scores():
    cs = _cs()
    members = _mint_order0(cs, 3)
    context = _mint_order0(cs, 2)
    # One round = support 1 per member entry... the shared-context entry
    # sees each member once -> support 3 with THREE members; drop to a
    # single member observation instead.
    _observe(cs, {members[0][1]: 1.0,
                  context[0][1]: 0.8, context[1][1]: 0.7})
    _mock_promotion_score(cs, 1.0, 1.0, 1.0)
    cs.truth_criterion = 0.0
    assert cs.promotion_pass() == []                   # support 1 < 3


# -- commit: mint, edges, intent, reuse ---------------------------------------

def _promote_royalty(cs, members=None, context=None):
    members = members if members is not None else _mint_order0(cs, 3)
    context = context if context is not None else _mint_order0(cs, 2)
    _royal_rounds(cs, members, context)
    _mock_promotion_score(cs, 1.0, 1.0, 1.0)
    cs.truth_criterion = 0.5
    minted = cs.promotion_pass()
    assert len(minted) == 1
    return minted[0], members, context


def test_promotion_mints_raised_whole_with_member_edges():
    cs = _cs()
    H, members, _context = _promote_royalty(cs)
    alloc = Spaces._concept_alloc_of(cs)
    assert H in alloc.raised
    parts = set(alloc.refs(H, "part"))
    for (cid, _row) in members:
        assert ("sym", int(cid)) in parts
    # Member edge values initialize from the NORMALIZED EWMA weights:
    # last-observed member carries weight 1.0, earlier ones decayed by beta.
    h_row = cs._csw_row_of(H)
    assert h_row is not None
    got = dict(cs.concept_weights(h_row))
    beta = float(cs._promotion_ewma)
    for age, (cid, row) in enumerate(reversed(members)):
        assert got[row] == pytest.approx(beta ** age, rel=1e-4)


def test_promotion_commits_weighted_intent_parts():
    cs = _cs()
    H, members, context = _promote_royalty(cs)
    alloc = Spaces._concept_alloc_of(cs)
    parts = set(alloc.refs(H, "part"))
    h_row = cs._csw_row_of(H)
    got = dict(cs.concept_weights(h_row))
    # crown (0.8) and palace (0.7) commit as sym_part intent, weight
    # normalized to the max context weight.
    (crown, c_row), (palace, p_row) = context
    assert ("sym", int(crown)) in parts
    assert ("sym", int(palace)) in parts
    assert got[c_row] == pytest.approx(1.0, rel=1e-4)
    assert got[p_row] == pytest.approx(0.7 / 0.8, rel=1e-4)


def test_resupport_strengthens_instead_of_reminting():
    cs = _cs()
    H, members, context = _promote_royalty(cs)
    h_row = cs._csw_row_of(H)
    before = dict(cs.concept_weights(h_row))
    _royal_rounds(cs, members, context)                # fresh support
    assert cs.promotion_pass() == []                   # no new mint
    e = cs._promotion_state()["cache"][
        frozenset(r for (_c, r) in context)]
    assert e["committed"] == H
    after = dict(cs.concept_weights(h_row))
    for col, v in before.items():
        assert after[col] == pytest.approx(min(4.0, v + 0.1), rel=1e-4)


def test_promotion_is_idempotent_per_member_set():
    cs = _cs()
    H, members, _context = _promote_royalty(cs)
    key = ("raise", frozenset(("sym", int(c)) for (c, _r) in members))
    alloc = Spaces._concept_alloc_of(cs)
    assert alloc.relate_idx.get(key) == H


# -- pyramid coupling (the plan's ablation criterion) --------------------------

def test_promoted_whole_enters_pyramid_and_ablates():
    cs = _cs()
    cs.eval()
    H, members, _context = _promote_royalty(cs)
    h_row = int(cs._csw_row_of(H))
    o1_start, o1_end = cs.order_slice(1)
    assert o1_start <= h_row < o1_end                  # order-1 block row
    caps0 = cs._order_caps()[0]
    a_0 = torch.zeros(caps0, 1)
    for (_cid, row) in members:
        a_0[row, 0] = 1.0
    W = cs.similarity_codebook.getW()
    _content, acts = cs.cs_forward_content(a_0, W)
    live = float(acts[h_row, 0].abs())
    assert live > 0.0                                  # composed from members
    # Ablation: disabling the whole's relation edges must change the
    # prediction-relevant activation (plan sec 6: functional connection).
    ly = Spaces._concept_alloc_of(cs).layer(0)
    cols = [c for (c, _w) in cs.concept_weights(h_row)]
    ly.remove_edges([(h_row, c if c != int(cs.nVectors)
                      else cs._bias_col()) for c in cols])
    _content2, acts2 = cs.cs_forward_content(a_0, W)
    assert float(acts2[h_row, 0].abs()) == pytest.approx(0.0, abs=1e-7)
    assert float(acts2[h_row, 0].abs()) < live


# -- prune / decay / retire ----------------------------------------------------

def test_stale_weak_candidate_is_dropped():
    cs = _cs()
    focal = _mint_order0(cs, 1)[0]
    ctx = _mint_order0(cs, 2)
    _observe(cs, {focal[1]: 1.0, ctx[0][1]: 0.8, ctx[1][1]: 0.7})
    st = cs._promotion_state()
    assert len(st["cache"]) == 3                       # one entry per focal
    st["obs"] += cs._promotion_stale_age + 1           # age them out
    cs.promotion_pass()
    assert len(st["cache"]) == 0


def test_unsupported_whole_decays_and_retires():
    cs = _cs()
    H, _members, _context = _promote_royalty(cs)
    object.__setattr__(cs, "_promotion_decay", 0.0)    # one-pass zeroing
    st = cs._promotion_state()
    st["obs"] += cs._promotion_stale_age + 1
    cs.promotion_pass()
    alloc = Spaces._concept_alloc_of(cs)
    assert H in alloc.retired                          # decayed -> retired
    assert alloc.records(H) == []
    assert len(st["cache"]) == 0


# -- the item-30 taper fix for _set_concept_edge_value -------------------------

def test_set_concept_edge_value_reaches_per_order_blocks():
    cs = _cs()
    (a, a_row), (b, b_row) = _mint_order0(cs, 2)
    H = cs.synthesize_higher_order((("sym", a), ("sym", b)))
    h_row = cs._csw_row_of(H)
    assert dict(cs.concept_weights(h_row))[a_row] == pytest.approx(1.0)
    # Pre-fix this silently no-oped: the concept row resolved via the
    # retired ("pool", cid) namespace, absent on rev-2 per-order blocks.
    cs._set_concept_edge_value(H, a, "sym_part", 0.25)
    assert dict(cs.concept_weights(h_row))[a_row] == pytest.approx(0.25)
