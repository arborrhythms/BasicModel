"""MultiOptimizer exposes the torch-optimizer surface callers rely on."""
import os
import sys
from pathlib import Path

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch
import torch.nn as nn

_BIN = Path(__file__).resolve().parent.parent / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Optimizer import MultiOptimizer  # noqa: E402


def _two_family():
    dense = nn.Linear(3, 2)
    sparse = nn.Embedding(5, 3, sparse=True)
    adam = torch.optim.Adam(dense.parameters(), lr=1e-2)
    sadam = torch.optim.SparseAdam(list(sparse.parameters()), lr=1e-2)
    return dense, sparse, MultiOptimizer([adam, sadam])


def _step(dense, sparse, opt):
    opt.zero_grad()
    x = torch.randn(4, 3)
    idx = torch.tensor([0, 2, 2, 4])
    loss = dense(x).pow(2).mean() + sparse(idx).pow(2).mean()
    loss.backward()
    opt.step()


def test_state_view_covers_every_child():
    dense, sparse, opt = _two_family()
    _step(dense, sparse, opt)
    for p in list(dense.parameters()) + list(sparse.parameters()):
        assert p in opt.state
        assert "exp_avg" in opt.state[p]
    assert len(opt.state) == len(list(dense.parameters())) + len(list(sparse.parameters()))
    assert opt.state.get(torch.zeros(1), "missing") == "missing"


def test_add_param_group_goes_to_the_dense_child_and_flattens():
    dense, sparse, opt = _two_family()
    head = nn.Linear(2, 1)
    before = len(opt.param_groups)
    opt.add_param_group({"params": list(head.parameters())})
    assert len(opt.param_groups) == before + 1
    assert any(p is head.weight for g in opt.optimizers[0].param_groups
               for p in g["params"])
    assert not any(p is head.weight for g in opt.optimizers[1].param_groups
                   for p in g["params"])
    _step(dense, sparse, opt)


def test_state_dict_has_torch_views_and_round_trips():
    dense, sparse, opt = _two_family()
    _step(dense, sparse, opt)
    sd = opt.state_dict()
    assert set(sd) == {"optimizers", "state", "param_groups"}
    assert sd["state"] and len(sd["param_groups"]) == len(opt.param_groups)
    n_params = sum(len(g["params"]) for g in sd["param_groups"])
    assert sorted(i for g in sd["param_groups"] for i in g["params"]) == list(range(n_params))
    # Round trip through the per-child layout ...
    dense2, sparse2, opt2 = _two_family()
    opt2.load_state_dict({"optimizers": sd["optimizers"]})
    for a, b in zip(opt.optimizers, opt2.optimizers):
        assert a.state_dict()["param_groups"][0]["lr"] == b.state_dict()["param_groups"][0]["lr"]
    # ... and through the torch-style views.
    dense3, sparse3, opt3 = _two_family()
    opt3.load_state_dict({"state": sd["state"], "param_groups": sd["param_groups"]})
    for p_src, p_dst in zip(list(dense.parameters()) + list(sparse.parameters()),
                            list(dense3.parameters()) + list(sparse3.parameters())):
        assert torch.equal(opt.state[p_src]["exp_avg"].to_dense() if opt.state[p_src]["exp_avg"].is_sparse else opt.state[p_src]["exp_avg"],
                           opt3.state[p_dst]["exp_avg"].to_dense() if opt3.state[p_dst]["exp_avg"].is_sparse else opt3.state[p_dst]["exp_avg"])
