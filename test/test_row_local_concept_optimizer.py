"""Sparse-gradient and compact-moment contract for the shared CS codebook."""

from __future__ import annotations

import copy
import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

from Models import BaseModel, BasicModel, TheXMLConfig  # noqa: E402
from Optimizer import MultiOptimizer, RowLocalAdam  # noqa: E402
from Spaces import Codebook, ConceptualSpace  # noqa: E402
from test_basicmodel import _populate_test_config  # noqa: E402
from util import TheDevice, init_device  # noqa: E402


def _devices():
    values = ["cpu"]
    if torch.backends.mps.is_available():
        values.append("mps")
    return values


@pytest.mark.parametrize("device", _devices())
def test_sparse_cs_lookup_and_row_local_step_stay_compact(device):
    previous = str(TheDevice.get())
    init_device(device)
    try:
        rows, dim = 1024, 8
        cb = Codebook()
        cb.W = nn.Parameter(torch.randn(rows, dim, device=device))
        cb.sparse_lookup_grad = True
        before = cb.W.detach().clone()
        selected = torch.tensor([[3, 37, 3]], device=device)

        cb.lookup_rows(selected).square().sum().backward()
        assert cb.W.grad is not None
        assert cb.W.grad.layout == torch.sparse_coo

        optimizer = RowLocalAdam([cb.W], lr=1e-2)
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()

        state = optimizer.state[cb.W]
        assert state["step"] == 1
        assert tuple(state["exp_avg"].shape) == (64, dim)
        assert tuple(state["exp_avg_sq"].shape) == (64, dim)
        # Two compact moments together remain one eighth the size of the
        # single physical parameter in this fixture (and do not scale to V).
        moment_elements = sum(
            state[name].numel() for name in ("exp_avg", "exp_avg_sq"))
        assert moment_elements == 2 * 64 * dim
        assert moment_elements < cb.W.numel()

        changed = (cb.W.detach() - before).abs().sum(dim=1).nonzero().flatten()
        assert changed.cpu().tolist() == [3, 37]
    finally:
        init_device(previous)
        if device == "mps":
            torch.mps.empty_cache()


def test_row_local_adam_matches_sparse_adam_on_touched_rows():
    torch.manual_seed(7)
    row_local_param = nn.Parameter(torch.randn(32, 5))
    sparse_adam_param = nn.Parameter(row_local_param.detach().clone())
    row_local = RowLocalAdam([row_local_param], lr=3e-3)
    reference = torch.optim.SparseAdam([sparse_adam_param], lr=3e-3)

    for selected in ([1, 3, 3], [9, 1], [17, 9]):
        for param in (row_local_param, sparse_adam_param):
            param.grad = None
            F.embedding(torch.tensor(selected), param, sparse=True).square() \
                .sum().backward()
        row_local.step()
        reference.step()

    torch.testing.assert_close(row_local_param, sparse_adam_param)
    local_state = row_local.state[row_local_param]
    ref_state = reference.state[sparse_adam_param]
    n = int(local_state["exp_avg"].shape[0])
    assert n == 32
    torch.testing.assert_close(local_state["exp_avg"], ref_state["exp_avg"][:n])
    torch.testing.assert_close(
        local_state["exp_avg_sq"], ref_state["exp_avg_sq"][:n])


def test_compact_moments_roundtrip_and_normalizer_preserves_prefix():
    source = nn.Parameter(torch.randn(64, 4))
    source_opt = RowLocalAdam(
        [source], lr=1e-2, moment_dtype=torch.bfloat16)
    F.embedding(torch.tensor([2, 9]), source, sparse=True).sum().backward()
    source_opt.step()
    saved = copy.deepcopy(source_opt.state_dict())

    # A larger physical capacity must not cause checkpoint normalization to
    # zero-pad compact row-local moments to the complete parameter.
    restored = nn.Parameter(torch.randn(512, 4))
    restored_opt = RowLocalAdam(
        [restored], lr=1e-2, moment_dtype=torch.bfloat16)
    restored_opt.load_state_dict(saved)
    model = BaseModel()
    model.name = "RowLocalCheckpointTest"
    assert model._normalize_optimizer_state_shapes(restored_opt) == 0
    state = restored_opt.state[restored]
    assert state["step"] == 1
    assert tuple(state["exp_avg"].shape) == (16, 4)
    assert tuple(state["exp_avg_sq"].shape) == (16, 4)
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16

    restored.grad = None
    F.embedding(torch.tensor([260]), restored, sparse=True).sum().backward()
    restored_opt.step()
    assert tuple(state["exp_avg"].shape) == (512, 4)
    assert tuple(state["exp_avg_sq"].shape) == (512, 4)
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16


def test_pre_moment_dtype_checkpoint_adopts_configured_bfloat16_storage():
    source = nn.Parameter(torch.randn(32, 4))
    source_opt = RowLocalAdam([source], lr=1e-2)
    F.embedding(torch.tensor([2, 9]), source, sparse=True).sum().backward()
    source_opt.step()
    saved = copy.deepcopy(source_opt.state_dict())
    saved["param_groups"][0].pop("moment_dtype")

    restored = nn.Parameter(torch.randn(32, 4))
    restored_opt = RowLocalAdam(
        [restored], lr=1e-2, moment_dtype=torch.bfloat16)
    restored_opt.load_state_dict(saved)
    state = restored_opt.state[restored]
    assert restored_opt.param_groups[0]["moment_dtype"] == torch.bfloat16
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16


@pytest.mark.parametrize("device", _devices())
def test_bfloat16_moment_storage_uses_fp32_math_close_to_sparse_adam(device):
    previous = str(TheDevice.get())
    init_device(device)
    try:
        torch.manual_seed(7)
        compact_param = nn.Parameter(torch.randn(64, 5, device=device))
        reference_param = nn.Parameter(compact_param.detach().clone())
        compact = RowLocalAdam(
            [compact_param], lr=3e-3, moment_dtype=torch.bfloat16)
        reference = torch.optim.SparseAdam(
            [reference_param], lr=3e-3)
        selected = torch.tensor([1, 3, 3, 9, 17], device=device)

        # Exercise a broad gradient-magnitude range. BF16 retains fp32's
        # exponent range, while the touched-row calculations themselves are
        # promoted to fp32 by RowLocalAdam.
        for step in range(40):
            scale = 10.0 ** (-(step % 6))
            for param in (compact_param, reference_param):
                param.grad = None
                loss = F.embedding(
                    selected, param, sparse=True).square().sum() * scale
                loss.backward()
            compact.step()
            reference.step()
        if device == "mps":
            torch.mps.synchronize()

        torch.testing.assert_close(
            compact_param, reference_param, rtol=5e-4, atol=5e-4)
        state = compact.state[compact_param]
        assert state["exp_avg"].dtype == torch.bfloat16
        assert state["exp_avg_sq"].dtype == torch.bfloat16

        moment_bytes = sum(
            state[name].numel() * state[name].element_size()
            for name in ("exp_avg", "exp_avg_sq"))
        fp32_moment_bytes = sum(
            state[name].numel() * 4
            for name in ("exp_avg", "exp_avg_sq"))
        assert moment_bytes * 2 == fp32_moment_bytes
    finally:
        init_device(previous)
        if device == "mps":
            torch.mps.empty_cache()


def test_get_optimizer_routes_only_sparse_shared_cs_parameter_row_locally():
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(128, 6))
    cb.sparse_lookup_grad = True
    dense = nn.Parameter(torch.randn(6, 6))
    fake_space = types.SimpleNamespace(
        getParameters=lambda: [cb.W, dense])

    model = BaseModel()
    model.spaces = [fake_space]
    model.conceptualSpaces = [
        types.SimpleNamespace(similarity_codebook=cb)]
    optimizer = model.getOptimizer(lr=1e-3)

    assert isinstance(optimizer, MultiOptimizer)
    assert len(optimizer.optimizers) == 2
    row_local_leaves = [
        leaf for leaf in optimizer.optimizers if leaf.row_local_state]
    assert len(row_local_leaves) == 1
    assert row_local_leaves[0].param_groups[0]["params"] == [cb.W]
    assert (row_local_leaves[0].param_groups[0]["moment_dtype"]
            == torch.bfloat16)
    dense_leaves = [
        leaf for leaf in optimizer.optimizers if not leaf.row_local_state]
    assert len(dense_leaves) == 1
    assert any(dense is param
               for group in dense_leaves[0].param_groups
               for param in group["params"])


def test_sparse_decoder_leaves_shared_codebook_gradient_sparse():
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.nWhat, cs.nWhere, cs.nWhen = 8, 0, 0
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(64, 8))
    cb.sparse_lookup_grad = True
    cs.similarity_codebook = cb

    rows = torch.tensor([[2, 11]])
    activations = torch.ones(1, 6, 2)
    bands = torch.empty(1, 6, 2, 0)
    cs.decode_sparse_concept_rows(rows, activations, bands).square().sum() \
        .backward()

    assert cb.W.grad is not None
    assert cb.W.grad.layout == torch.sparse_coo
    assert sorted(cb.W.grad.coalesce().indices()[0].tolist()) == [2, 11]


def test_parallel_aligned_meta_model_does_not_enable_sparse_lookup_mode():
    """serialObjectMeta alone cannot opt a dense parallel CS into RowLocalAdam."""
    _populate_test_config(
        inputDim=8, perceptDim=8, conceptDim=8, symbolDim=16,
        outputDim=1, nInput=8, nPercepts=8, nConcepts=8,
        nSymbols=8, nOutput=4, conceptCodebook="quantize")
    TheXMLConfig._data["architecture"].update({
        "serial": False,
        "serialObjectMeta": True,
        "conceptBinding": "aligned",
        "subsymbolicOrder": 2,
    })
    TheXMLConfig._data["WholeSpace"].update({
        "propertyBasis": True,
        "nDim": 16,
    })
    model = BasicModel()
    model.concept_binding = "aligned"
    model.create(
        nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nOutput=4,
        subsymbolicOrder=2, model_type="embedding")

    assert not model.conceptualSpace.similarity_codebook.sparse_lookup_grad
    assert not any(
        cs.similarity_codebook.sparse_lookup_grad
        for cs in model.conceptualSpaces)


def test_concept_index_read_coexists_without_densifying_shared_gradient():
    """The optional known-concept read must use the same sparse row surface."""
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.nWhat, cs.nWhere, cs.nWhen = 8, 0, 0
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(64, 8))
    cb.sparse_lookup_grad = True
    cs.similarity_codebook = cb
    object.__setattr__(
        cs, "concept_codebook_row_of_percept",
        lambda pid: {3: 7, 5: 19}.get(int(pid)))

    # Ordinary PS/WS/SS sparse activation plus conceptIndexRead in the same
    # graph used to combine F.embedding(sparse=True) with W[row], causing
    # autograd to allocate a full strided W.grad before RowLocalAdam could
    # reject it.
    decoded = cs.decode_sparse_concept_rows(
        torch.tensor([[2, 11]]),
        torch.ones(1, 6, 2),
        torch.empty(1, 6, 2, 0))
    content, mask = cs.concept_row_content(torch.tensor([3, 99, 5]))
    assert mask.tolist() == [True, False, True]
    assert not bool(content[1].detach().ne(0).any())
    (decoded.square().sum() + content.square().sum()).backward()

    assert cb.W.grad is not None
    assert cb.W.grad.layout == torch.sparse_coo
    rows = sorted(cb.W.grad.coalesce().indices()[0].tolist())
    assert rows == [2, 7, 11, 19]


def test_model_debug_finite_check_accepts_sparse_codebook_gradient(
        monkeypatch):
    cb = Codebook()
    cb.W = nn.Parameter(torch.randn(16, 4))
    cb.sparse_lookup_grad = True
    cb.lookup_rows(torch.tensor([3])).sum().backward()
    model = BaseModel()
    model.probe = cb

    import util
    monkeypatch.setattr(util, "MODEL_DEBUG", True)
    model._assert_finite_train_state("sparse-test")


@pytest.mark.parametrize("device", _devices())
def test_zero_valued_sparse_placeholder_does_not_decay_row_zero(device):
    previous = str(TheDevice.get())
    init_device(device)
    try:
        param = nn.Parameter(torch.randn(8, 4, device=device))
        optimizer = RowLocalAdam(
            [param], lr=1e-2, moment_dtype=torch.bfloat16)

        # Seed nonzero row-0 moments first; this makes a later accidental Adam
        # visit observable as a parameter move even when its new gradient is 0.
        F.embedding(
            torch.tensor([0], device=device), param, sparse=True).sum() \
            .backward()
        optimizer.step()
        optimizer.zero_grad()
        before = param.detach().clone()
        step_before = optimizer.state[param]["step"]

        # This is the exact unknown-id masking shape: row 0 remains present in
        # the sparse COO structure, but every coordinate of its value is zero.
        (F.embedding(
            torch.tensor([0], device=device), param, sparse=True) * 0.0) \
            .sum().backward()
        assert param.grad.coalesce().indices()[0].tolist() == [0]
        assert not bool(param.grad.coalesce().values().ne(0).any().item())
        optimizer.step()
        if device == "mps":
            torch.mps.synchronize()

        torch.testing.assert_close(param, before, rtol=0.0, atol=0.0)
        assert optimizer.state[param]["step"] == step_before
    finally:
        init_device(previous)
        if device == "mps":
            torch.mps.empty_cache()
