"""Tests for the symbol-learning policy layer.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md
§ Phase 2 deferred / Phase 3 closeout — "Symbol-learning policy +
extend_artifact integration".

The policy reads flushed snapshots from ``SymbolLearningStats`` and
emits ``NewRef`` candidates via MDL-flavored criteria:

  * **Zero-order / QE.** Online leader clustering accumulates per-cluster
    EMA of squared quantization error and a stability count. When a
    cluster's EMA exceeds ``qe_promote_threshold`` AND stability reaches
    ``stability_n``, that centroid becomes a new order-0 ref.

  * **Higher-order / PMI.** Pair counts gathered from REDUCE outcomes
    are turned into PMI × frequency scores. Pairs above
    ``pmi_threshold`` are promoted at the rule's LHS order/category.

``flush_and_promote`` orchestrates: flush the stats, run the policy,
call ``extend_artifact`` if there are candidates.
"""
import sys
from pathlib import Path

import pytest
import torch

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


# ---------------------------------------------------------------------------
# Extended SymbolLearningStats: pair_info + clusters
# ---------------------------------------------------------------------------


def test_observe_reduce_accepts_optional_parent_metadata():
    """``observe_reduce`` accepts ``parent_scalar`` / ``parent_category``
    / ``parent_order`` kwargs and tucks them into ``pair_info`` so the
    policy can promote at the rule's LHS metadata."""
    from symbol_learning import SymbolLearningStats
    sls = SymbolLearningStats(enabled=True)
    sls.observe_reduce(
        left_ref=1, right_ref=2, parent_ref=10,
        parent_scalar=0.7, parent_category='NP', parent_order=1)
    snap = sls.flush()
    assert (1, 2) in snap['pair_info']
    info = snap['pair_info'][(1, 2)]
    assert info['count'] == 1
    assert info['parent_category'] == 'NP'
    assert info['parent_order'] == 1
    assert info['parent_scalar_sum'] == pytest.approx(0.7)


def test_observe_qe_runs_leader_clustering():
    """When ``leader_radius`` is set, ``observe_qe`` performs online
    leader clustering and exposes per-cluster ``qe_ema`` + ``stability``
    via ``flush``."""
    from symbol_learning import SymbolLearningStats
    sls = SymbolLearningStats(enabled=True, leader_radius=0.5,
                              ema_alpha=0.5)
    # 5 observations at roughly the same activation
    for _ in range(5):
        sls.observe_qe(
            activation=torch.tensor([1.0, 1.0]),
            snapped=torch.tensor([0.0, 0.0]))  # QE = 2.0 each
    snap = sls.flush()
    assert len(snap['clusters']) == 1
    c = snap['clusters'][0]
    assert c['stability'] == 5
    assert c['qe_ema'] == pytest.approx(2.0, abs=1e-3)


def test_observe_qe_clusters_separate_far_activations():
    """Activations beyond ``leader_radius`` start new clusters."""
    from symbol_learning import SymbolLearningStats
    sls = SymbolLearningStats(enabled=True, leader_radius=0.5,
                              ema_alpha=0.5)
    # Cluster A
    for _ in range(3):
        sls.observe_qe(activation=torch.tensor([1.0, 1.0]),
                       snapped=torch.tensor([0.9, 0.9]))
    # Cluster B (far)
    for _ in range(2):
        sls.observe_qe(activation=torch.tensor([-1.0, -1.0]),
                       snapped=torch.tensor([-0.9, -0.9]))
    snap = sls.flush()
    assert len(snap['clusters']) == 2


def test_qe_raises_on_nan_input():
    """NaN activation must raise — never silently nan_to_num'd. Per the
    user's memory: fail loud on numerical divergence."""
    from symbol_learning import SymbolLearningStats
    sls = SymbolLearningStats(enabled=True, leader_radius=0.5)
    with pytest.raises(ValueError, match=r"NaN|nan"):
        sls.observe_qe(activation=torch.tensor([float('nan'), 1.0]),
                       snapped=torch.tensor([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Policy: propose_candidates
# ---------------------------------------------------------------------------


def test_policy_default_thresholds_construct():
    """``SymbolLearningPolicy()`` with defaults constructs and exposes
    threshold attributes."""
    from symbol_learning import SymbolLearningPolicy
    p = SymbolLearningPolicy()
    assert hasattr(p, 'qe_promote_threshold')
    assert hasattr(p, 'stability_n')
    assert hasattr(p, 'pmi_threshold')
    assert hasattr(p, 'count_threshold')


def test_policy_zero_order_promotion():
    """Cluster with stability_n observations and qe_ema above threshold
    yields an order-0 NewRef."""
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy)
    sls = SymbolLearningStats(enabled=True, leader_radius=2.0,
                              ema_alpha=0.5)
    for _ in range(8):
        sls.observe_qe(activation=torch.tensor([1.0, 1.0]),
                       snapped=torch.tensor([0.0, 0.0]))  # QE 2.0
    snap = sls.flush()
    policy = SymbolLearningPolicy(qe_promote_threshold=1.0,
                                  stability_n=5,
                                  default_zero_order_category='N',
                                  default_zero_order_parent_ref_id=0)
    candidates = policy.propose_candidates(snap)
    zero = [c for c in candidates if c.order == 0]
    assert len(zero) == 1
    assert zero[0].category == 'N'


def test_policy_no_zero_order_below_threshold():
    """A cluster whose qe_ema is below the threshold does NOT yield a
    NewRef even at high stability."""
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy)
    sls = SymbolLearningStats(enabled=True, leader_radius=2.0,
                              ema_alpha=0.5)
    for _ in range(20):
        sls.observe_qe(activation=torch.tensor([0.01, 0.01]),
                       snapped=torch.tensor([0.0, 0.0]))  # QE ≈ 0.0002
    snap = sls.flush()
    policy = SymbolLearningPolicy(qe_promote_threshold=1.0,
                                  stability_n=5)
    candidates = policy.propose_candidates(snap)
    assert all(c.order != 0 for c in candidates)


def test_policy_no_zero_order_below_stability():
    """A high-QE cluster below stability_n does NOT yield a NewRef."""
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy)
    sls = SymbolLearningStats(enabled=True, leader_radius=2.0,
                              ema_alpha=0.5)
    for _ in range(3):
        sls.observe_qe(activation=torch.tensor([1.0, 1.0]),
                       snapped=torch.tensor([0.0, 0.0]))
    snap = sls.flush()
    policy = SymbolLearningPolicy(qe_promote_threshold=1.0,
                                  stability_n=5)
    candidates = policy.propose_candidates(snap)
    assert all(c.order != 0 for c in candidates)


def test_policy_higher_order_pmi_promotion():
    """A pair whose PMI × count exceeds the threshold yields a
    higher-order NewRef at the rule's category/order.

    PMI for (1, 2) is positive when 1 and 2 don't co-occur with many
    other refs — i.e., the bound pair has tight marginals relative to
    overall corpus volume. We achieve that with one tight pair plus
    several unique distractor pairs that inflate the total without
    bumping (1, 2)'s marginals.
    """
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy)
    sls = SymbolLearningStats(enabled=True)
    # Tight (1, 2) co-occurrence
    for _ in range(10):
        sls.observe_reduce(
            left_ref=1, right_ref=2, parent_ref=100,
            parent_scalar=0.7, parent_category='NP', parent_order=1)
    # Unique distractor pairs inflate the total without touching (1, 2)
    for k in range(20):
        sls.observe_reduce(
            left_ref=100 + k, right_ref=200 + k, parent_ref=300 + k,
            parent_scalar=0.1, parent_category='S', parent_order=2)
    snap = sls.flush()
    # PMI(1,2) = log((10 * 30) / (10 * 10)) = log(3) ≈ 1.099
    policy = SymbolLearningPolicy(pmi_threshold=0.5,
                                  count_threshold=5)
    candidates = policy.propose_candidates(snap)
    promoted = [c for c in candidates if c.order > 0]
    assert len(promoted) >= 1
    cand = next(c for c in promoted if c.category == 'NP')
    assert cand.category == 'NP'
    assert cand.order == 1


def test_policy_higher_order_below_count_threshold():
    """A pair with too few observations does NOT promote."""
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy)
    sls = SymbolLearningStats(enabled=True)
    for _ in range(2):  # below count_threshold
        sls.observe_reduce(
            left_ref=1, right_ref=2, parent_ref=100,
            parent_scalar=0.7, parent_category='NP', parent_order=1)
    snap = sls.flush()
    policy = SymbolLearningPolicy(pmi_threshold=0.0,
                                  count_threshold=10)
    candidates = policy.propose_candidates(snap)
    assert candidates == []


# ---------------------------------------------------------------------------
# Orchestrator: flush_and_promote
# ---------------------------------------------------------------------------


def _tiny_grammar():
    """Minimal grammar mirror of test_knowledge_artifact_writer fixture."""
    from Language import Grammar
    g = Grammar()
    g.rules = [
        g._parse_rule("S4", "lift(NP3, VP1)", tier='S'),
        g._parse_rule("NP3", "lower(DET, NP4)", tier='S'),
    ]
    g._configured = True
    return g


def _build_minimal_artifact(tmp_path):
    """Build a tiny knowledge artifact via the production writer
    helpers, so extend_artifact has the schema it expects."""
    from embed import save_artifact, build_knowledge_section
    g = _tiny_grammar()
    ks = build_knowledge_section(g)
    path = str(tmp_path / 'knowledge.kv')
    save_artifact(path, knowledge=ks)
    return path


def test_flush_and_promote_calls_extend_artifact(tmp_path):
    """When the policy emits candidates, ``flush_and_promote`` invokes
    ``extend_artifact`` so the artifact grows by the candidate count."""
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy, flush_and_promote)
    from embed import load_artifact

    path = _build_minimal_artifact(tmp_path)
    initial = load_artifact(path)
    v_live_before = int(
        initial['knowledge']['reference_codebook']['v_ref_live'])

    sls = SymbolLearningStats(enabled=True, leader_radius=2.0,
                              ema_alpha=0.5)
    for _ in range(10):
        sls.observe_qe(activation=torch.tensor([1.0, 1.0]),
                       snapped=torch.tensor([0.0, 0.0]))
    policy = SymbolLearningPolicy(qe_promote_threshold=1.0,
                                  stability_n=5,
                                  default_zero_order_category='N',
                                  default_zero_order_parent_ref_id=0)

    promoted = flush_and_promote(sls, policy, path)
    assert len(promoted) >= 1

    after = load_artifact(path)
    v_live_after = int(
        after['knowledge']['reference_codebook']['v_ref_live'])
    assert v_live_after == v_live_before + len(promoted)


def test_flush_and_promote_noop_when_no_candidates(tmp_path):
    """When the policy emits no candidates, the artifact is unchanged."""
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy, flush_and_promote)
    from embed import load_artifact

    path = _build_minimal_artifact(tmp_path)
    initial = load_artifact(path)
    v_live_before = int(
        initial['knowledge']['reference_codebook']['v_ref_live'])

    sls = SymbolLearningStats(enabled=True)
    # No observations - flush yields empty snapshot
    policy = SymbolLearningPolicy()
    promoted = flush_and_promote(sls, policy, path)
    assert promoted == []

    after = load_artifact(path)
    v_live_after = int(
        after['knowledge']['reference_codebook']['v_ref_live'])
    assert v_live_after == v_live_before


def test_flush_and_promote_resets_stats(tmp_path):
    """After ``flush_and_promote``, the stats accumulator is reset."""
    from symbol_learning import (
        SymbolLearningStats, SymbolLearningPolicy, flush_and_promote)

    path = _build_minimal_artifact(tmp_path)
    sls = SymbolLearningStats(enabled=True, leader_radius=2.0,
                              ema_alpha=0.5)
    for _ in range(10):
        sls.observe_qe(activation=torch.tensor([1.0, 1.0]),
                       snapped=torch.tensor([0.0, 0.0]))
    policy = SymbolLearningPolicy(qe_promote_threshold=1.0,
                                  stability_n=5,
                                  default_zero_order_category='N')

    flush_and_promote(sls, policy, path)
    # Stats reset
    assert sls.qe_count == 0
    assert sls.pair_counts == {}
    assert sls._clusters == []
