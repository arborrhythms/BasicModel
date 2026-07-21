"""Fast contracts for the sparse ConceptualSpace compiler boundary.

The million-row conceptual dictionary must retain a sparse COO gradient for
``RowLocalAdam``.  A small eager island around ``F.embedding(...,
sparse=True)`` keeps that backward out of Inductor while allowing ordinary
dense work on either side to remain under Dynamo.  These tests deliberately
use Dynamo's lightweight eager backend; the slow platform Inductor gate lives
outside the default unit suite.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_ROOT = Path(__file__).resolve().parent.parent
_BIN = _ROOT / "bin"
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))

import Models  # noqa: E402
import util  # noqa: E402
from Models import BasicModel  # noqa: E402
from Optimizer import RowLocalAdam  # noqa: E402
from Spaces import Codebook, ConceptualSpace  # noqa: E402
from util import TheDevice, init_device  # noqa: E402


class _CompiledSparseSandwich(nn.Module):
    """Dense tensor work around one row-sparse conceptual dictionary read."""

    def __init__(self, rows=32, dim=6):
        super().__init__()
        self.codebook = Codebook()
        self.codebook.W = nn.Parameter(torch.randn(rows, dim))
        self.codebook.sparse_lookup_grad = True
        self.scale = nn.Parameter(torch.randn(dim))

    def forward(self, x, row_ids):
        dense_prefix = torch.sin(x * self.scale)
        selected = self.codebook.lookup_rows(row_ids)
        return torch.cos(dense_prefix + selected)


def _bare_concept_decoder(codebook, dim=6):
    cs = ConceptualSpace.__new__(ConceptualSpace)
    nn.Module.__init__(cs)
    cs.nWhat, cs.nWhere, cs.nWhen = dim, 0, 0
    cs.inputShape = [3, dim]
    cs.outputShape = [3, dim]
    cs.similarity_codebook = codebook
    return cs


def test_dynamo_eager_island_preserves_sparse_gradient_and_row_local_step():
    """The island may segment Dynamo, but it must not densify ``W.grad``."""
    torch.manual_seed(17)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    model = _CompiledSparseSandwich()
    before = model.codebook.W.detach().clone()
    row_ids = torch.tensor([2, 9, 2, 17])

    try:
        compiled = torch.compile(model, backend="eager", fullgraph=False)
        compiled(torch.randn(4, 6), row_ids).square().sum().backward()

        stats = dict(torch._dynamo.utils.counters.get("stats", {}))
        assert stats.get("unique_graphs", 0) > 0
        assert stats.get("calls_captured", 0) > 0
        assert model.scale.grad is not None

        grad = model.codebook.W.grad
        assert grad is not None
        assert grad.layout == torch.sparse_coo
        assert grad.coalesce().indices()[0].tolist() == [2, 9, 17]

        optimizer = RowLocalAdam([model.codebook.W], lr=1e-2)
        optimizer.step()
        changed = (model.codebook.W.detach() - before).abs().sum(dim=1) \
            .ne(0).nonzero().flatten().tolist()
        assert changed == [2, 9, 17]
    finally:
        torch._dynamo.reset()


def test_staged_sentence_bank_is_fullgraph_clean_and_gradient_equivalent():
    """One pre-graph gather must equal repeated sparse reads, duplicates too."""
    torch.manual_seed(23)
    weight = torch.randn(32, 6)

    staged_cb = Codebook()
    staged_cb.W = nn.Parameter(weight.clone())
    staged_cb.sparse_lookup_grad = True
    staged_cs = _bare_concept_decoder(staged_cb)

    direct_cb = Codebook()
    direct_cb.W = nn.Parameter(weight.clone())
    direct_cb.sparse_lookup_grad = True
    direct_cs = _bare_concept_decoder(direct_cb)

    bank_rows = torch.tensor([[9, 2, 9, -1]])
    bank_atoms = staged_cb.lookup_rows(bank_rows.clamp_min(0))
    bank_atoms = bank_atoms * bank_rows.ge(0).unsqueeze(-1)
    rows = torch.tensor([[9, 2, -1]])
    activations = torch.randn(1, 2, 3)
    bands = torch.empty(1, 2, 3, 0)

    class _StagedDecode(nn.Module):
        def __init__(self, cs):
            super().__init__()
            self.cs = cs

        def forward(self, row_ids, source_a, source_b, bank_ids, atoms):
            return self.cs.decode_sparse_concept_rows(
                row_ids, source_a, source_b,
                staged_rows=bank_ids, staged_atoms=atoms)

    torch._dynamo.reset()
    try:
        compiled = torch.compile(
            _StagedDecode(staged_cs), backend="eager", fullgraph=True)
        staged_out = compiled(
            rows, activations, bands, bank_rows, bank_atoms)
        direct_out = direct_cs.decode_sparse_concept_rows(
            rows, activations, bands)
        torch.testing.assert_close(staged_out, direct_out)

        staged_out.square().sum().backward()
        direct_out.square().sum().backward()
        assert staged_cb.W.grad.layout == torch.sparse_coo
        assert direct_cb.W.grad.layout == torch.sparse_coo
        torch.testing.assert_close(
            staged_cb.W.grad.coalesce().to_dense(),
            direct_cb.W.grad.coalesce().to_dense())
        nonzero_rows = (staged_cb.W.grad.coalesce().to_dense().abs()
                        .sum(dim=1).ne(0).nonzero().flatten().tolist())
        assert nonzero_rows == [2, 9]
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize(
    ("sparse_lookup_grad", "staged_serial_bank", "expected_fullgraph"),
    [(False, False, True), (True, False, False), (True, True, True)],
)
def test_compiled_step_relaxes_only_for_unstaged_sparse_concept_codebook(
        monkeypatch, sparse_lookup_grad, staged_serial_bank,
        expected_fullgraph):
    """The canonical staged sparse bank remains strict/fullgraph-clean."""
    previous_device = str(TheDevice.get())
    init_device("cpu")
    try:
        model = BasicModel()
        codebook = Codebook()
        codebook.sparse_lookup_grad = sparse_lookup_grad
        model.conceptualSpaces = [
            types.SimpleNamespace(similarity_codebook=codebook)]
        model.conceptualSpace = model.conceptualSpaces[0]
        if staged_serial_bank:
            model.inputSpace = types.SimpleNamespace(_per_word_enabled=True)
            model.serial = True
            model.serial_object_meta = True
            model.concept_binding = "aligned"
        model._prewarm_checkpoint_shapes = lambda: None
        model.forward = lambda input_data: input_data

        compile_calls = []

        def _capture_compile(callable_, **kwargs):
            compile_calls.append((callable_, kwargs))
            return callable_

        monkeypatch.setattr(Models, "ENUM_FULLGRAPH", True)
        monkeypatch.setattr(Models, "TheMessage", lambda *_args, **_kw: None)
        monkeypatch.setattr(util, "compile", _capture_compile)

        model.enable_compiled_step()

        assert len(compile_calls) == 1
        assert compile_calls[0][1]["fullgraph"] is expected_fullgraph
        if staged_serial_bank and sparse_lookup_grad:
            assert model._compiled_word_chunk_active is True
            assert model._compiled_word_chunk_step is compile_calls[0][0]
            assert model._compiled_step is model.forward
        else:
            assert model._compiled_word_chunk_active is False
            assert model._compiled_step is compile_calls[0][0]
    finally:
        init_device(previous_device)


def test_compile_none_preserves_legacy_sentence_loop(monkeypatch):
    """A skipped compile must not silently select the new chunk semantics."""
    previous_device = str(TheDevice.get())
    previous_backend = util.TheCompileBackend
    init_device("cpu")
    try:
        model = BasicModel()
        codebook = Codebook()
        codebook.sparse_lookup_grad = True
        space = types.SimpleNamespace(similarity_codebook=codebook)
        model.conceptualSpaces = [space]
        model.conceptualSpace = space
        model.inputSpace = types.SimpleNamespace(_per_word_enabled=True)
        model.serial = True
        model.serial_object_meta = True
        model.concept_binding = "aligned"
        model._prewarm_checkpoint_shapes = lambda: None
        model.forward = lambda input_data: input_data

        monkeypatch.setattr(Models, "ENUM_FULLGRAPH", True)
        monkeypatch.setattr(Models, "TheMessage", lambda *_args, **_kw: None)
        monkeypatch.setattr(util, "TheCompileBackend", "none")
        model.enable_compiled_step()

        assert model._compiled_word_chunk_active is False
        assert model._compiled_word_chunk_step is None
        assert model._compiled_step is model.forward
    finally:
        util.TheCompileBackend = previous_backend
        init_device(previous_device)
