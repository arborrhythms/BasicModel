"""Throughput / optimal-batch-size sweep. NOT a pytest test.

Measures wall-clock training throughput at a range of batch sizes on
the current (eager) path -- the honest current speed (torch.compile
was a no-op pre-Task-1 and the compiled path still graph-breaks, so
eager is representative; optimal batch size is memory-bound and
compile-independent). Bypasses ModelFactory.run's strict pre-flight
(which would hard-abort while syncs remain) by driving bounded
runEpoch directly.

OOM-safe: ascending sweep, small bounded batch count per size, peak
memory + OOM caught, cache cleared between sizes, single foreground
run. Keep batch counts tiny on metalbaby (prior OOM/reboot).

Usage:
  .venv/bin/python test/bench_throughput.py CONFIG.xml DATASET \
      "8,16,32,64,128" [warmup] [timed] [max_docs]
e.g. MM_20M_legacy.xml text "8,16,32,64,128,256"
     MM_grammar.xml xor "8,16,32,64,128,256"
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MODEL_DEBUG", "0")
os.environ["MODEL_COMPILE"] = "none"   # eager: clean, compile-time-free

_p = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_p.parent / "bin"))
sys.path.insert(0, str(_p / "bin"))

import torch
from data import TheData
from Models import BaseModel
from util import init_config


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "MM_20M_legacy.xml"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "text"
    sizes = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3
                              else "8,16,32,64,128,256").split(",")]
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    timed = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    max_docs = int(sys.argv[6]) if len(sys.argv) > 6 else 512

    cfg = str(_p / "data" / cfg_name)
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    if dataset == "text":
        shard = str(_p / "data" / "fineweb")
        if not (os.path.isdir(shard) and os.listdir(shard)):
            print(f"SKIP {cfg_name}: shard corpus {shard!r} absent")
            return
        TheData.load("text", shard_dir=shard, num_shards=1,
                     max_docs=max_docs)
    else:
        TheData.load(dataset)

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = BaseModel.from_config(cfg, data=TheData)
    model = model.to(dev)
    opt = model.getOptimizer(lr=1e-4)

    print(f"[bench] cfg={cfg_name} dataset={dataset} device={dev} "
          f"warmup={warmup} timed={timed} max_docs={max_docs}")
    print(f"{'batch':>6} {'b/s':>8} {'samp/s':>10} {'ms/batch':>9} "
          f"{'peakGB':>7}  status")

    best = (None, 0.0)
    for bs in sizes:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model.runEpoch(optimizer=opt, batchSize=bs, split="train",
                           max_batches=warmup)
            _sync()
            t0 = time.perf_counter()
            model.runEpoch(optimizer=opt, batchSize=bs, split="train",
                           max_batches=timed)
            _sync()
            dt = time.perf_counter() - t0
            bps = timed / dt if dt > 0 else 0.0
            sps = bps * bs
            peak = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
            print(f"{bs:>6} {bps:>8.2f} {sps:>10.1f} "
                  f"{1000*dt/timed:>9.1f} {peak:>7.2f}  ok")
            if sps > best[1]:
                best = (bs, sps)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e)
            oom = ("out of memory" in msg.lower()
                   or isinstance(e, torch.cuda.OutOfMemoryError))
            print(f"{bs:>6} {'-':>8} {'-':>10} {'-':>9} {'-':>7}  "
                  f"{'OOM' if oom else 'ERR'}: {msg[:80]}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if oom:
                break          # ceiling reached; stop ascending
            else:
                continue

    print(f"[bench] optimal batch (max samples/s): "
          f"{best[0]} @ {best[1]:.1f} samp/s")


if __name__ == "__main__":
    main()
