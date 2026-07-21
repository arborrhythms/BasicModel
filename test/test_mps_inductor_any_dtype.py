"""Regression gate for PyTorch 2.12 MPS ``any`` reduction metadata."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

import util  # noqa: E402


class _AnyThenBitwiseMask(nn.Module):
    def forward(self, left, right, gate):
        # Reduction width eight is the smallest production geometry that
        # exercises Metal's threadgroup ``any`` accumulator path.
        return gate & (left & right).any(dim=-1)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_inductor_any_result_retains_torch_bool_metadata():
    previous_device = str(util.TheDevice.get())
    previous_backend = util.TheCompileBackend
    previous_mode = util.TheCompileMode
    util.init_device("mps")
    util.init_compile_backend("inductor")
    util.init_compile_mode("default")
    torch._dynamo.reset()
    try:
        left = torch.tensor(
            [[[True, False, True, False, True, False, True, False]]],
            device="mps")
        right = torch.tensor(
            [[[False, False, True, True, False, False, True, True]]],
            device="mps")
        gate = torch.tensor([[True]], device="mps")
        expected = _AnyThenBitwiseMask()(left, right, gate)

        compiled = util.compile(
            _AnyThenBitwiseMask(), verbose=False, fullgraph=True)
        actual = compiled(left, right, gate)
        torch.mps.synchronize()

        assert torch.equal(actual.cpu(), expected.cpu())
        from torch._inductor.codegen import mps as mps_codegen
        assert getattr(
            mps_codegen.MetalKernel._reduction_nocache,
            "_basicmodel_any_dtype_patch", False)
    finally:
        torch._dynamo.reset()
        util.init_compile_backend(previous_backend)
        util.init_compile_mode(previous_mode)
        util.init_device(previous_device)
        torch.mps.empty_cache()
