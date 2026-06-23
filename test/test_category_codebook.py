"""MetaSymbol Category codebook scaffolding (Phase 1, increment B).

WholeSpace.enable_category_codebook allocates a small role-space VectorQuantize
plus a bounded pending learner for uncommitted MetaSymbols. assign_category is
the role-profile E-step, update_category_role updates the centroid prototype,
and category_role_of / category_role_for_meta gather the per-slot context.
With codebook_retire=False, unused centroids' cluster_size decays -> the basis
of natural collapse. Exercised in isolation on a synthetic grammar + vectors
(no parser/model dependency). doc/Language.md "Participation Categories".
"""

import os
import sys
from collections import namedtuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import Spaces
from Language import compute_role_vocabulary
from test_basicmodel import _populate_test_config

_Rule = namedtuple("_Rule", ["method_name", "rhs_symbols"])


class _G:
    def __init__(self, rules):
        self.rules_upward = rules


def _grammar():
    return _G([
        _Rule("isEqual", ["isEqual_I1", "isEqual_I2"]),
        _Rule("lift", ["lift_I1", "lift_I2"]),
        _Rule("exist", ["exist_I1"]),
    ])


def _whole_space(d=8):
    # Shapes mirror the non-flatten path (test_partition_pos_codebook); the
    # WholeSpace ctor reads TheXMLConfig, so populate it first.
    nP, nS = 4, 6
    _populate_test_config(
        inputDim=d, perceptDim=d, conceptDim=d, symbolDim=d,
        wordDim=d, outputDim=d,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    return Spaces.WholeSpace([nP, d], [nS, d], [nS, d])


def test_enable_allocates_codebook_and_role_table():
    ss = _whole_space()
    g = _grammar()
    _roles, _idx, n_roles = compute_role_vocabulary(g)
    assert not ss.category_codebook_enabled()      # off until enabled
    ok = ss.enable_category_codebook(g)
    assert ok and ss.category_codebook_enabled()
    assert ss._category_n_roles == n_roles
    assert tuple(ss._category_role.shape) == (n_roles, n_roles)   # K defaults to n_roles
    assert tuple(ss._category_vq.codebook.shape) == (n_roles, n_roles)
    torch.testing.assert_close(
        ss._category_role[:n_roles], torch.eye(n_roles))
    # idempotent
    assert ss.enable_category_codebook(g) is True


def test_enable_noop_without_roles():
    ss = _whole_space()
    assert ss.enable_category_codebook(_G([])) is False
    assert not ss.category_codebook_enabled()


def test_assign_returns_valid_centroid_indices():
    ss = _whole_space(d=8)
    ss.enable_category_codebook(_grammar())
    K = ss._category_role.shape[0]
    vecs = torch.randn(5, ss._category_n_roles)
    idx = ss.assign_category(vecs)
    assert idx is not None and tuple(idx.shape) == (5,)
    assert int(idx.min()) >= 0 and int(idx.max()) < K


def test_role_vector_ema_moves_toward_target():
    ss = _whole_space()
    ss.enable_category_codebook(_grammar())
    n_roles = ss._category_n_roles
    idx = torch.tensor([0])
    before = ss._category_role[0].clone()
    target = torch.ones(1, n_roles)
    for _ in range(20):
        ss.update_category_role(idx, target, ema=0.2)
    after = ss._category_role[0]
    # EMA toward all-ones: empty columns increased, none overshoots the target.
    assert torch.all(after[1:] > before[1:])
    assert torch.all(after <= 1.0 + 1e-5)
    torch.testing.assert_close(after, ss._category_vq.codebook[0])


def test_category_role_of_zeroes_negative_index():
    ss = _whole_space()
    ss.enable_category_codebook(_grammar())
    ss.update_category_role(torch.tensor([1]), torch.ones(1, ss._category_n_roles))
    got = ss.category_role_of(torch.tensor([1, -1]))
    assert got is not None and tuple(got.shape) == (2, ss._category_n_roles)
    assert torch.all(got[1] == 0.0)                 # composed/unassigned -> zero row
    assert torch.all(got[0] > 0.0)                  # assigned centroid -> its role vec


def test_disabled_methods_are_safe_noops():
    ss = _whole_space()
    assert ss.assign_category(torch.randn(3, ss.nDim)) is None
    assert ss.category_role_of(torch.tensor([0])) is None
    ss.update_category_role(torch.tensor([0]), torch.ones(1, 4))   # no error
    ss.enable_category_codebook(_grammar())
    assert ss.assign_category(
        torch.randn(3, ss._category_n_roles + 1)) is None           # wrong width


def test_pending_meta_symbol_commits_to_single_category():
    ss = _whole_space()
    ss.enable_category_codebook(_grammar())
    col = ss._category_role_index["isEqual_I1"]
    vec = torch.zeros(ss._category_n_roles)
    vec[col] = 1.0
    meta_pos = 123

    learner = ss._category_learner
    assert meta_pos not in ss._category_assign
    for _ in range(int(learner.min_mass) - 1):
        assert ss.observe_category_roles(meta_pos, vec) is None
    assert meta_pos in learner.pending
    pending_role = ss.category_role_for_meta(meta_pos)
    assert pending_role is not None and float(pending_role[col]) > 0.9

    cat = ss.observe_category_roles(meta_pos, vec)
    assert cat is not None
    assert ss._category_assign[meta_pos] == cat
    assert meta_pos not in learner.pending
    committed_role = ss.category_role_for_meta(meta_pos)
    assert committed_role is not None and float(committed_role[col]) > 0.9


def test_category_vq_assign_is_pure_lookup():
    # The category VQ has the in-forward EMA disabled (ema_update=False), so
    # update_category_role (the role-vector M-step) is the SOLE writer.
    # assign_category (the E-step read) must therefore be a pure lookup: it
    # still routes a tight cluster to its nearest centroid, but it does NOT
    # mutate the codebook or the cluster_size buffer (no EMA drift).
    torch.manual_seed(0)
    ss = _whole_space(d=8)
    ss.enable_category_codebook(_grammar())
    ss.train()
    assert ss._category_vq.ema_update is False
    cb_before = ss._category_vq.codebook.detach().clone()
    cs_before = ss._category_vq.cluster_size.detach().clone()
    cluster = torch.zeros(1, ss._category_n_roles)
    cluster[0, 0] = 10.0                              # far-out tight cluster
    winners = set()
    for _ in range(40):
        idx = ss.assign_category(cluster + 0.01 * torch.randn(8, ss.nDim))
        winners.update(int(i) for i in idx)
    assert len(winners) < ss._category_role.shape[0]   # not all centroids used
    # The repeated lookup left the codebook and EMA buffers untouched.
    torch.testing.assert_close(ss._category_vq.codebook.detach(), cb_before)
    torch.testing.assert_close(ss._category_vq.cluster_size.detach(), cs_before)
