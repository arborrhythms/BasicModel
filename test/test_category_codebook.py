"""MetaSymbol Category codebook scaffolding (Phase 1, increment B).

WholeSpace.enable_category_codebook allocates a small VectorQuantize over the
MetaSymbol vectors plus a per-centroid role-vector sidecar; assign_category is
the E-step (+ free recentroid M-step), update_category_role the role-vector
M-step, and category_role_of the per-slot context gather (Phase 2). With
codebook_retire=False, unused centroids' cluster_size decays -> the basis of
the natural collapse. Exercised in isolation on a synthetic grammar + vectors
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
    ws = _whole_space()
    g = _grammar()
    _roles, _idx, n_roles = compute_role_vocabulary(g)
    assert not ws.category_codebook_enabled()      # off until enabled
    ok = ws.enable_category_codebook(g)
    assert ok and ws.category_codebook_enabled()
    assert ws._category_n_roles == n_roles
    assert tuple(ws._category_role.shape) == (n_roles, n_roles)   # K defaults to n_roles
    # idempotent
    assert ws.enable_category_codebook(g) is True


def test_enable_noop_without_roles():
    ws = _whole_space()
    assert ws.enable_category_codebook(_G([])) is False
    assert not ws.category_codebook_enabled()


def test_assign_returns_valid_centroid_indices():
    ws = _whole_space(d=8)
    ws.enable_category_codebook(_grammar())
    K = ws._category_role.shape[0]
    vecs = torch.randn(5, ws.nDim)
    idx = ws.assign_category(vecs)
    assert idx is not None and tuple(idx.shape) == (5,)
    assert int(idx.min()) >= 0 and int(idx.max()) < K


def test_role_vector_ema_moves_toward_target():
    ws = _whole_space()
    ws.enable_category_codebook(_grammar())
    n_roles = ws._category_n_roles
    idx = torch.tensor([0])
    before = ws._category_role[0].clone()
    target = torch.ones(1, n_roles)
    for _ in range(20):
        ws.update_category_role(idx, target, ema=0.2)
    after = ws._category_role[0]
    # EMA toward all-ones: every column increased, none overshoots the target.
    assert torch.all(after > before)
    assert torch.all(after <= 1.0 + 1e-5)


def test_category_role_of_zeroes_negative_index():
    ws = _whole_space()
    ws.enable_category_codebook(_grammar())
    ws.update_category_role(torch.tensor([1]), torch.ones(1, ws._category_n_roles))
    got = ws.category_role_of(torch.tensor([1, -1]))
    assert got is not None and tuple(got.shape) == (2, ws._category_n_roles)
    assert torch.all(got[1] == 0.0)                 # composed/unassigned -> zero row
    assert torch.all(got[0] > 0.0)                  # assigned centroid -> its role vec


def test_disabled_methods_are_safe_noops():
    ws = _whole_space()
    assert ws.assign_category(torch.randn(3, ws.nDim)) is None
    assert ws.category_role_of(torch.tensor([0])) is None
    ws.update_category_role(torch.tensor([0]), torch.ones(1, 4))   # no error


def test_unused_centroids_decay_toward_collapse():
    # Feeding a single tight cluster repeatedly: the nearest centroid keeps
    # winning while the rest receive no assignments, so their cluster_size EMA
    # decays below the bootstrap 1.0 -- the mechanism behind natural collapse.
    torch.manual_seed(0)
    ws = _whole_space(d=8)
    ws.enable_category_codebook(_grammar())
    ws.train()
    cluster = torch.zeros(1, ws.nDim)
    cluster[0, 0] = 10.0                              # far-out tight cluster
    winners = set()
    for _ in range(40):
        idx = ws.assign_category(cluster + 0.01 * torch.randn(8, ws.nDim))
        winners.update(int(i) for i in idx)
    cs = ws._category_vq.cluster_size
    assert len(winners) < ws._category_role.shape[0]   # not all centroids used
    assert float(cs.min()) < 1.0                       # an unused centroid decayed
