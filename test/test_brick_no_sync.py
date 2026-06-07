"""Brick body issues zero ``cudaMemcpyDtoH`` events (§6).

The brick-vectorization handoff §6 (a/b/c) removes every host sync
from ``runBatch``'s body so the brick is fit for CUDA-graph capture.
This test profiles one ``runBatch`` call and asserts the count of
``cudaMemcpyDtoH`` events between forward start and ``optimizer.step``
end is zero.

Parametrized over representative configs so the contract is tracked on
the toy XOR path (``MM_xor``), the grammar path (``MM_grammar``), and
the production embedding path (``MM_20M`` on FineWeb shards). MM_20M is
skipped when the shard corpus is not present on the host.

CUDA-only -- ``torch.profiler`` records device events that don't exist
on MPS/CPU. Skipped automatically on macOS / non-CUDA hosts.

Plan reference: doc/plans/2026-04-27-brick-vectorization-and-legacy-removal-handoff.md §6
"""
import os
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


# (config, dataset, TheData.load kwargs). ``dataset`` mirrors the
# train.py ``--data`` override (the merged config's <data><dataset>
# defaults to "xor" for every config; the real corpus is selected at
# run time): MM_xor / MM_grammar run on the toy XOR set, MM_20M on the
# FineWeb-EDU shard corpus (kept small here -- this is a host-sync
# profile, not a convergence run).
_CONFIGS = [
    ("MM_xor.xml", "xor", {}),
    ("MM_grammar.xml", "xor", {}),
    ("MM_20M.xml", "text", {
        "shard_dir": str(_project / "data" / "fineweb"),
        "num_shards": 1,
        "max_docs": 64,
    }),
]


@pytest.mark.parametrize(
    "cfg_name,dataset,load_kw", _CONFIGS,
    ids=[c[0].replace(".xml", "") for c in _CONFIGS],
)
def test_brick_runBatch_zero_dtoh_events(cfg_name, dataset, load_kw):
    """One ``runBatch`` tick on CUDA fires zero ``cudaMemcpyDtoH``.

    Profiles ``forward + backward + optimizer.step`` and asserts the
    aggregated event list contains no device-to-host copy events
    inside the brick body.
    """
    from data import TheData
    from Models import BaseModel
    from util import init_config
    import torch.profiler

    config = str(_project / "data" / cfg_name)
    # Mirror ModelFactory.run: init the merged config before loading
    # data / building the model (needed for the embedding/text path).
    init_config(path=config,
                defaults_path=str(_project / "data" / "model.xml"))

    # MM_20M needs the FineWeb shard corpus; skip cleanly if it's not on
    # this host rather than failing on a missing-data load.
    shard_dir = load_kw.get("shard_dir")
    if dataset == "text" and (
            not shard_dir or not os.path.isdir(shard_dir)
            or not os.listdir(shard_dir)):
        pytest.skip(f"{cfg_name}: shard corpus {shard_dir!r} not present")

    TheData.load(dataset, **load_kw)

    model, _ = BaseModel.from_config(config, data=TheData)
    model = model.to('cuda')
    # Use the production optimizer factory: it sets capturable=True on
    # CUDA so Adam's step counter stays on-device (a hand-rolled
    # torch.optim.Adam issues a cudaMemcpyDtoH per param per step via
    # _multi_tensor_adam's _get_value(step).item(), which would defeat
    # the very contract this test asserts).
    optimizer = model.getOptimizer(lr=1e-4)

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
        f"[{cfg_name}] brick body must not issue cudaMemcpyDtoH events; "
        f"got {len(dtoh)}: {[e.name for e in dtoh[:5]]}")
