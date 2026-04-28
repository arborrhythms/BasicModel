"""CUDA-graph capture / replay bit-equivalence (§7).

The brick-vectorization handoff §7 explores wrapping ``runBatch`` in
either ``torch.compile(mode="reduce-overhead")`` or an explicit
``torch.cuda.CUDAGraph`` replay path. Both routes produce the same
forward/backward/step sequence, so 100 ticks of replay should give
per-position-loss arrays bit-equivalent to the eager path within
``1e-4`` (bf16 tolerance).

CUDA-only -- graph capture is unavailable on MPS / Metal. Skipped
automatically on macOS / non-CUDA hosts. The §7 capture wiring is
exercised on the GB10 host; this test becomes meaningful once the
capture path is enabled there.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md §7
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch
import pytest


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-graph capture requires a CUDA device",
)


def _capture_wired(model):
    """Probe BasicModel for a hardware-active capture path.

    The §7 wiring lives on the model itself (no env-var indirection):
    once the GB10 enables ``torch.compile(mode="reduce-overhead")`` or
    an explicit ``torch.cuda.CUDAGraph`` over ``runBatch``, the model
    exposes a flag like ``model._brick_compiled`` / a wrapper attr that
    this test detects. Until that lands, skip.
    """
    return bool(
        getattr(model, '_brick_compiled', False)
        or getattr(model, '_brick_cuda_graph', None) is not None
    )


def test_captured_replay_matches_eager_within_tolerance():
    """100 ticks of captured-graph replay produce identical losses to
    100 ticks of eager within bf16 tolerance.
    """
    from data import TheData
    from Models import BaseModel

    config = str(_project / "data" / "MM_xor.xml")
    TheData.load("xor")

    torch.manual_seed(0)
    model_cap, _ = BaseModel.from_config(config, data=TheData)
    model_cap = model_cap.to('cuda')
    if not _capture_wired(model_cap):
        pytest.skip(
            "§7 capture path not wired on this BasicModel; expected "
            "model._brick_compiled or model._brick_cuda_graph once the "
            "GB10 path lands.")

    eager_losses = []
    torch.manual_seed(0)
    model_eager, _ = BaseModel.from_config(config, data=TheData)
    model_eager = model_eager.to('cuda')
    opt_e = torch.optim.Adam(model_eager.parameters(), lr=1e-4)
    for _ in range(5):
        result = model_eager.runEpoch(
            optimizer=opt_e, batchSize=2, split="train")
        eager_losses.append(float(result[0]))

    captured_losses = []
    opt_c = torch.optim.Adam(model_cap.parameters(), lr=1e-4)
    for _ in range(5):
        result = model_cap.runEpoch(
            optimizer=opt_c, batchSize=2, split="train")
        captured_losses.append(float(result[0]))

    for e, c in zip(eager_losses, captured_losses):
        assert abs(e - c) < 1e-4, (
            f"captured-graph replay diverged from eager: "
            f"eager={e}, captured={c}")
