#!/usr/bin/env python3
"""Measure steady compiled training throughput for one static word bucket.

Unlike ``bench_word_buckets.py``, this drives ``runBatch(train=True)`` with a
real optimizer and a fixed synthetic sentence.  It is intended to separate a
single bucket's forward/backward/step rate from corpus bucket switching and
one-time compilation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))


def _sentence(words):
    return " ".join(f"word{i}" for i in range(int(words)))


def _sync(torch, device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="data/BasicModel.xml")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1,
                    help="Unmeasured compiled training steps before timing")
    ap.add_argument("--seconds", type=float, default=7200.0,
                    help="Measured wall-clock budget, checked between steps")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--error-on-recompile", action="store_true")
    args = ap.parse_args(argv)
    if args.width < 1 or args.batch < 1 or args.warmup < 0 or args.seconds < 0:
        ap.error("width/batch must be positive; warmup/seconds must be non-negative")

    os.environ.setdefault("BASIC_AUTOLOAD", "0")
    os.environ.setdefault("BASIC_MAX_DOCS", "8")
    os.environ.setdefault("BASIC_NUM_SHARDS", "1")

    import torch
    from recon_bench import _build_model, _resolve_config

    model, device, _, _ = _build_model(_resolve_config(args.config))
    model.train()
    text = _sentence(args.width)
    optimizer = model.getOptimizer(lr=float(args.lr))
    target = model.outputSpace.prepOutput(model._stub_outputs(args.batch))

    # Populate host-side state before Dynamo sees the fixed training loop.
    with torch.no_grad():
        primer = model.inputSpace.prepInput([text] * args.batch)
        model._start_spaces_for_forward()
        model.forward(primer)
        model._end_step()
    model.enable_compiled_step()

    def step(index):
        x = model.inputSpace.prepInput([text] * args.batch)
        result, _ = model.runBatch(
            train=True, split="train", batchNum=index,
            batchSize=args.batch, optimizer=optimizer,
            batch_override=(x, target))
        _sync(torch, device)
        return result

    for index in range(args.warmup):
        step(-args.warmup + index)

    if args.error_on_recompile:
        import torch._dynamo as dynamo
        dynamo.config.error_on_recompile = True

    samples = []
    started = time.monotonic()
    index = 0
    while index == 0 or time.monotonic() - started < args.seconds:
        t0 = time.perf_counter()
        result = step(index)
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
        del result
        index += 1
        print(f"step={index}\tseconds={elapsed:.6f}", flush=True)

    report = {
        "config": str(_resolve_config(args.config)),
        "device": str(device),
        "torch_version": torch.__version__,
        "width": args.width,
        "batch": args.batch,
        "warmup_steps": args.warmup,
        "measured_steps": len(samples),
        "elapsed_s": time.monotonic() - started,
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "sentences_per_s": args.batch / statistics.median(samples),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
