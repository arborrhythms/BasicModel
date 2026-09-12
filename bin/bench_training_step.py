"""Evidence-based training-step benchmark (Codex review, requirement 6).

Usage (from ``basicmodel/``)::

    BENCH_OUT=/tmp/step.json BASIC_BATCH_SIZE=8 BASIC_MAX_DOCS=400 \
        BASIC_MAX_BATCHES=6 BASIC_NUM_EPOCHS=1 \
        .venv/bin/python bin/bench_training_step.py data/BasicModel.xml

Leave ``MODEL_COMPILE`` unset for the production backend.  The JSON records
per brick the full-step wall-clock (forward, loss, backward, optimizer,
resets), dynamo recompile frames, peak accelerator memory and the executed
loss terms with their gradient presence; the first brick's time is the
compile.  See doc/benchmarks/2026-09-11-fold-ladder-throughput.md.

Runs ``runEpoch`` on a config and records, per optimizer brick: wall-clock of
the full step (forward, backward, optimizer), compile time (brick 0), dynamo
recompiles, peak accelerator memory, the executed loss terms and their
gradient presence, and the reconstruction/primary loss values.
"""
import sys, os, time, json
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch, Models
out_path = os.environ.get("BENCH_OUT", "/tmp/bench_protocol.json")
rec = {"bricks": [], "losses": {}, "config": sys.argv[1], "env": {k: v for k, v in os.environ.items() if k.startswith("BASIC_") or k.startswith("MODEL_") or k.startswith("BASICMODEL_")}}
def sync():
    if torch.backends.mps.is_available(): torch.mps.synchronize()
def mem():
    if torch.backends.mps.is_available():
        return int(torch.mps.driver_allocated_memory())
    return 0
orig_prim = Models.BasicModel._primary_loss
def prim(self, lossOut, lossIn=None, sbow=None):
    total = orig_prim(self, lossOut, lossIn, sbow)
    rec["losses"] = {"lossOut": float(lossOut) if torch.is_tensor(lossOut) else None,
                     "lossOut_grad": bool(torch.is_tensor(lossOut) and lossOut.requires_grad),
                     "lossIn": float(lossIn) if torch.is_tensor(lossIn) else None,
                     "lossIn_grad": bool(torch.is_tensor(lossIn) and lossIn.requires_grad),
                     "d3_active": bool(getattr(self, "_d3_active", False)),
                     "detached_reverse": bool(getattr(self, "detached_reverse", False))}
    return total
Models.BasicModel._primary_loss = prim
orig_rb = Models.BasicModel.runBatch
def rb(self, *a, **k):
    from torch._dynamo.utils import counters
    frames_before = sum(int(v) for v in counters.get("frames", {}).values())
    sync(); t = time.perf_counter()
    result = orig_rb(self, *a, **k)
    sync(); dt = time.perf_counter() - t
    frames_after = sum(int(v) for v in counters.get("frames", {}).values())
    rec["bricks"].append({"step_s": round(dt, 3), "recompile_frames": frames_after - frames_before,
                          "peak_mem_gb": round(mem() / 1e9, 3), "losses": dict(rec["losses"])})
    json.dump(rec, open(out_path, "w"), indent=1)
    return result
Models.BasicModel.runBatch = rb
try:
    Models.ModelFactory.run(sys.argv[1])
finally:
    json.dump(rec, open(out_path, "w"), indent=1)
