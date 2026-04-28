"""Brick body issues zero ``cudaMemcpyDtoH`` events (§6).

The brick-vectorization handoff §6 (a/b/c) removes every host sync
from ``runBatch``'s body so the brick is fit for CUDA-graph capture.
This test profiles one ``runBatch`` call and asserts the count of
``cudaMemcpyDtoH`` events between forward start and ``optimizer.step``
end is zero.

CUDA-only -- ``torch.profiler`` records device events that don't exist
on MPS/CPU. Skipped automatically on macOS / non-CUDA hosts.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md §6
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
    reason="CUDA profiler trace requires a CUDA device",
)


def test_brick_runBatch_zero_dtoh_events():
    """One ``runBatch`` tick on CUDA fires zero ``cudaMemcpyDtoH``.

    Profiles ``forward + backward + optimizer.step`` and asserts the
    aggregated event list contains no device-to-host copy events
    inside the brick body.
    """
    from data import TheData
    from Models import BaseModel
    import torch.profiler

    config = str(_project / "data" / "MM_xor.xml")
    TheData.load("xor")
    torch.manual_seed(0)
    model, _ = BaseModel.from_config(config, data=TheData)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warm-up: build the lazy state so the first profiled tick doesn't
    # incur init-time syncs (those don't reflect steady-state behavior).
    model.runEpoch(optimizer=optimizer, batchSize=2, split="train")

    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities,
                                record_shapes=False) as prof:
        model.runEpoch(optimizer=optimizer, batchSize=2, split="train")

    events = prof.events()
    dtoh = [e for e in events if 'memcpy' in e.name.lower()
            and 'dtoh' in e.name.lower()]
    assert len(dtoh) == 0, (
        f"brick body must not issue cudaMemcpyDtoH events; got {len(dtoh)}: "
        f"{[e.name for e in dtoh[:5]]}")
