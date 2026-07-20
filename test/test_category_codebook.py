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
import json
import pickle
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


def test_checkpoint_prewarm_materializes_requested_category_codebook(
        monkeypatch):
    """Autoload prewarm creates the lazy terminal category state."""
    import Language
    import Models
    from util import init_config, init_device

    project = os.path.dirname(_BIN)
    config = os.path.join(project, "data", "MM_xor.xml")
    defaults = os.path.join(project, "data", "model.xml")
    monkeypatch.setenv("BASIC_AUTOLOAD", "0")
    init_device("cpu")
    init_config(path=config, defaults_path=defaults)
    Language.TheGrammar._configured = False
    Models.TheData.load("xor")
    model, _ = Models.BasicModel.from_config(config, data=Models.TheData)
    ws = model.wholeSpace

    assert ws._category_codebook_requested is True
    assert not ws.category_codebook_enabled()
    model._prewarm_checkpoint_shapes()
    assert ws.category_codebook_enabled()
    n_roles = compute_role_vocabulary(Language.TheGrammar)[2]
    assert n_roles > 0
    assert tuple(ws._category_role.shape) == (n_roles, n_roles)

    state = model.state_dict()
    terminal = len(model.wholeSpaces) - 1
    expected = {
        f"wholeSpaces.{terminal}._category_role",
        f"wholeSpaces.{terminal}._category_vq._codebook",
        f"wholeSpaces.{terminal}._category_vq.cluster_size",
        f"wholeSpaces.{terminal}._category_vq.embed_avg",
        f"wholeSpaces.{terminal}._category_vq._b_norms_sq",
    }
    assert expected <= set(state)


def test_category_learning_vocab_extras_roundtrip_is_exact_and_json_safe():
    """Committed and pending terminal-category state resumes exactly."""
    source = _whole_space()
    source.enable_category_codebook(_grammar())
    learner = source._category_learner
    learner.max_pending = 17
    learner.min_mass = 3.0
    learner.min_confidence = 0.55
    learner.min_margin = 0.20
    learner.stable_updates = 2
    learner.evidence_decay = 1.0
    learner.prototype_ema = 0.23

    committed_vec = torch.zeros(source._category_n_roles)
    committed_vec[source._category_role_index["isEqual_I1"]] = 1.0
    pending_vec = torch.zeros(source._category_n_roles)
    pending_vec[source._category_role_index["lift_I1"]] = 1.0
    for _ in range(3):
        committed = source.observe_category_roles(101, committed_vec)
    assert committed is not None
    assert source.observe_category_roles(202, pending_vec) is None
    assert source.observe_category_roles(202, pending_vec) is None
    assert learner.step == 5

    extras = source.vocab_extras()
    assert extras["category_assign"] == {101: committed}
    assert 202 in extras["category_learner"]["pending"]
    # JSON is the stricter contract (no tensors or tuple keys); exercise a
    # pickle hop too because torch checkpoints use pickle internally.
    portable = json.loads(json.dumps(extras, allow_nan=False))
    portable = pickle.loads(pickle.dumps(portable))

    # Load before lazy category enablement to cover the non-BasicModel restore
    # ordering as well as the prewarmed production checkpoint path.
    restored = _whole_space()
    restored.load_vocab_extras(portable)
    assert not restored.category_codebook_enabled()
    deferred = restored.vocab_extras()
    assert deferred["category_learner"] == portable["category_learner"]
    restored.enable_category_codebook(_grammar())
    restored_learner = restored._category_learner

    assert restored._category_assign == source._category_assign
    for name in (
            "max_pending", "min_mass", "min_confidence", "min_margin",
            "stable_updates", "evidence_decay", "prototype_ema"):
        assert getattr(restored_learner, name) == getattr(learner, name)
    assert restored_learner.step == learner.step
    assert set(restored_learner.pending) == {202}
    source_row = learner.pending[202]
    restored_row = restored_learner.pending[202]
    assert restored_row["mass"] == source_row["mass"]
    assert restored_row["best"] == source_row["best"]
    assert restored_row["stable"] == source_row["stable"]
    assert restored_row["last"] == source_row["last"]
    torch.testing.assert_close(
        restored_row["evidence"], source_row["evidence"], rtol=0, atol=0)

    # The very next observation produces the same transition on both copies.
    source_result = source.observe_category_roles(202, pending_vec)
    restored_result = restored.observe_category_roles(202, pending_vec)
    assert restored_result == source_result
    assert restored._category_assign == source._category_assign
    assert restored_learner.step == learner.step == 6
    assert restored_learner.pending == learner.pending == {}


def test_category_learning_loads_legacy_vocab_extras_without_new_keys():
    """Pre-category sidecars retain the freshly initialized defaults."""
    source = _whole_space()
    legacy = source.vocab_extras()
    assert "category_assign" not in legacy
    assert "category_learner" not in legacy

    restored = _whole_space()
    restored.enable_category_codebook(_grammar())
    restored.load_vocab_extras(json.loads(json.dumps(legacy)))
    assert restored._category_assign == {}
    assert restored._category_learner.step == 0
    assert restored._category_learner.pending == {}
