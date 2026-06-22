"""STM-residual microbatch is sync-free (§6a).

The brick-vectorization handoff §6a removed the ``not_fired.any().item()``
early-out from ``SymbolSpace.stm_residual_microbatch``: the gate was a
per-batch GPU->CPU sync that blocked CUDA-graph capture. The
replacement always calls ``disc.predict()`` and zeros out the bias
on already-fired rows via ``torch.where``-style multiplication.

These tests are source-inspection based so they run on any platform
(no CUDA required). The CUDA profiler-based equivalent
(``test_brick_no_sync.py``) gates on a CUDA device.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md §6a
"""
import sys
import inspect
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest


def test_stm_residual_microbatch_has_no_item_early_out():
    """The body of ``stm_residual_microbatch`` must not call ``.item()``.

    The early-out was the only ``.item()`` call in the function; the
    handoff replaced it with a host-sync-free vectorized gate.
    """
    from Language import SymbolicSubSpace
    src = inspect.getsource(SymbolicSubSpace.stm_residual_microbatch)
    # The function may mention ``.item()`` in comments (e.g. the
    # "Removed: ..." marker the handoff left as a regression
    # signpost), but no live call should remain.
    code_lines = [
        line.split("#", 1)[0]
        for line in src.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert ".item()" not in code_only, (
        "stm_residual_microbatch must not call .item() in the brick body; "
        "the handoff §6a replaced the early-out with a torch.where gate")


def test_stm_residual_microbatch_runs_when_all_fired():
    """When every row has already fired, the call returns a tensor of
    zeros (or None), but never raises and never executes a host sync.

    This mirrors the production behavior: with all rows fired, the
    bias is gated to zero before being returned, so the caller's
    ``y = y + bias.unsqueeze(1)`` is a no-op without any branch.
    """
    from data import TheData
    from Models import BaseModel

    config = str(_project / "data" / "MM_xor.xml")
    TheData.load("xor")

    model, _ = BaseModel.from_config(config, data=TheData)
    ss = model.symbolSpace

    if ss is None or ss.discourse is None:
        pytest.skip("model has no discourse layer; STM-residual is None")

    # Mark every row as fired -- under the legacy gate this would
    # have early-returned; under the new gate it should compute and
    # zero the bias.
    B = int(ss._stm_fired.shape[0])
    ss._stm_fired.fill_(True)
    K = 1
    bias = ss.stm_residual_microbatch(B, K)
    # Either None (discourse missing) or a tensor — never an exception.
    if bias is not None:
        # All-fired rows must return zero contributions.
        assert bias.abs().sum().item() == pytest.approx(0.0, abs=1e-6), (
            f"stm_residual_microbatch returned nonzero bias when every "
            f"row was already fired: max abs = {bias.abs().max().item()}")
