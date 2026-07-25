#!/usr/bin/env python3
"""Measure steady compiled training throughput for synthetic or corpus input.

Unlike ``bench_word_buckets.py``, this drives ``runBatch(train=True)`` with a
real optimizer. By default every row is the same fixed-width synthetic
sentence; ``--corpus`` instead cycles through the configured training split's
contiguous-stream batches. On the tensor peer scheduler, ``--width`` is the
synthetic live word count while the configured W (normally 256) is only
storage capacity; the report includes both.
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
    ap.add_argument("--width", type=int, default=16,
                    help="Live words per synthetic sentence (ignored by --corpus)")
    ap.add_argument(
        "--corpus", action="store_true",
        help="Benchmark real configured training sentences instead of a fixed synthetic sentence")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1,
                    help="Unmeasured compiled training steps before timing")
    ap.add_argument("--seconds", type=float, default=7200.0,
                    help="Measured wall-clock budget, checked between steps")
    ap.add_argument("--steps", type=int,
                    help="Exact measured-step count (overrides --seconds)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--error-on-recompile", action="store_true")
    args = ap.parse_args(argv)
    if (args.width < 1 or args.batch < 1 or args.warmup < 0
            or args.seconds < 0 or (args.steps is not None and args.steps < 1)):
        ap.error(
            "width/batch/steps must be positive; warmup/seconds must be non-negative")

    os.environ.setdefault("BASIC_AUTOLOAD", "0")
    os.environ.setdefault("BASIC_MAX_DOCS", "8")
    os.environ.setdefault("BASIC_NUM_SHARDS", "1")

    import torch
    from recon_bench import _build_model, _resolve_config
    import util as model_util

    model, device, _, _ = _build_model(_resolve_config(args.config))
    model.train()
    text = _sentence(args.width)
    corpus_batches = None
    corpus_batch_max_words = None
    if args.corpus:
        import Meronomy

        loader = model.inputSpace.data.data_loader(
            split="train", num_streams=args.batch)
        corpus_batches = []
        corpus_batch_max_words = []
        for raw_batch in loader:
            texts = raw_batch[0] if isinstance(raw_batch, (tuple, list)) \
                and len(raw_batch) == 2 else raw_batch
            texts = list(texts)
            corpus_batches.append(texts)
            corpus_batch_max_words.append(max(
                len(Meronomy.word_spans(str(sentence).encode("utf-8")))
                for sentence in texts))
        if not corpus_batches:
            raise RuntimeError("configured corpus produced no training batches")
    optimizer = model.getOptimizer(lr=float(args.lr))
    target = model.outputSpace.prepOutput(model._stub_outputs(args.batch))

    # Legacy/static paths still need a complete eager primer. The functional
    # tensor loop initializes every mutable owner surface at runBatch's eager
    # boundary; priming it with the configured W=256 static fallback would
    # execute 256 expensive no-op word ticks before compilation.
    if not getattr(model, "_tensor_peer_while_enabled", False):
        with torch.no_grad():
            primer = model.inputSpace.prepInput([text] * args.batch)
            model._start_spaces_for_forward()
            model.forward(primer)
            model._end_step()
    try:
        from torch._dynamo.utils import counters as dynamo_counters
        dynamo_counters.clear()
    except Exception:
        dynamo_counters = None
    model.enable_compiled_step()

    corpus_cursor = 0

    def step(index):
        nonlocal corpus_cursor
        if corpus_batches is None:
            texts = [text] * args.batch
            batch_max_words = args.width
        else:
            slot = corpus_cursor % len(corpus_batches)
            texts = corpus_batches[slot]
            batch_max_words = corpus_batch_max_words[slot]
            corpus_cursor += 1
        x = model.inputSpace.prepInput(texts)
        result, _ = model.runBatch(
            train=True, split="train", batchNum=index,
            batchSize=args.batch, optimizer=optimizer,
            batch_override=(x, target))
        _sync(torch, device)
        return result, batch_max_words

    for index in range(args.warmup):
        step(-args.warmup + index)

    if args.error_on_recompile:
        import torch._dynamo as dynamo
        dynamo.config.error_on_recompile = True

    samples = []
    measured_batch_max_words = []
    started = time.monotonic()
    index = 0
    while (index < args.steps if args.steps is not None
           else (index == 0 or time.monotonic() - started < args.seconds)):
        t0 = time.perf_counter()
        result, batch_max_words = step(index)
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
        measured_batch_max_words.append(batch_max_words)
        del result
        index += 1
        print(f"step={index}\tseconds={elapsed:.6f}", flush=True)

    dynamo_report = {}
    if dynamo_counters is not None:
        for name in ("frames", "stats", "graph_break", "recompiles",
                     "aot_autograd", "inductor"):
            values = dynamo_counters.get(name)
            if values:
                dynamo_report[name] = {
                    str(key): int(value) for key, value in values.items()}
    runtime_trip = getattr(model, "_tensor_peer_trip_count", 0)
    if torch.is_tensor(runtime_trip):
        runtime_trip = int(runtime_trip.detach().to("cpu"))
    else:
        runtime_trip = int(runtime_trip or 0)
    report = {
        "config": str(_resolve_config(args.config)),
        "device": str(device),
        "torch_version": torch.__version__,
        "compile_backend": str(model_util.TheCompileBackend),
        "compile_mode_requested": str(model_util.TheCompileMode),
        "compile_mode_effective": str(model_util._effective_compile_mode(
            model_util.TheCompileMode, device)),
        "input_mode": "corpus" if args.corpus else "synthetic",
        "width": args.width,
        "staged_capacity": int(getattr(
            model.inputSpace, "_active_word_bucket", 0) or 0),
        "runtime_trip_count": runtime_trip,
        "measured_batch_max_words": measured_batch_max_words,
        "batch": args.batch,
        "warmup_steps": args.warmup,
        "requested_steps": args.steps,
        "measured_steps": len(samples),
        "elapsed_s": time.monotonic() - started,
        "samples_s": samples,
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "sentences_per_s": args.batch / statistics.median(samples),
        "aggregate_sentences_per_s": (
            args.batch * len(samples) / sum(samples)),
        "dynamo_counters": dynamo_report,
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
