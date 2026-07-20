#!/usr/bin/env python3
"""Benchmark the four fixed serial word-loop graphs on the live model.

Examples::

    BASICMODEL_DEVICE=mps .venv/bin/python bin/bench_word_buckets.py \
        --backend none --warmup 1 --iters 3

    BASICMODEL_DEVICE=mps .venv/bin/python bin/bench_word_buckets.py \
        --backend inductor --widths 16,32,64,128 --warmup 1 --iters 5

The eager stem is restaged before every sample and excluded from timing. Each
sample therefore measures the selected fixed-W forward graph, including its
masked padding iterations, PS/WS folds, aligned CS binding, and bounded STM
grammar.
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


def _parse_widths(raw):
    values = tuple(sorted(set(int(v.strip()) for v in raw.split(",")
                              if v.strip())))
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError("widths must be positive integers")
    return values


def _sentence(n):
    # Distinct suffixes keep the requested surface word count observable while
    # remaining well below InputSpace's raw-byte ceiling at W=128.
    return " ".join(f"word{i}" for i in range(int(n)))


def _sync(torch, device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="data/BasicModel.xml")
    ap.add_argument("--backend", choices=("none", "eager", "aot_eager",
                                           "inductor"), default="none")
    ap.add_argument("--widths", type=_parse_widths,
                    default=(16, 32, 64, 128))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--error-on-recompile", action="store_true",
                    help="raise after warmup if the same bucket retraces")
    args = ap.parse_args(argv)
    if args.batch < 1 or args.warmup < 0 or args.iters < 1:
        ap.error("batch/iters must be positive and warmup non-negative")

    # util.py reads compilation policy at import time.
    os.environ["MODEL_COMPILE"] = args.backend
    os.environ.setdefault("BASIC_AUTOLOAD", "0")
    os.environ.setdefault("BASIC_MAX_DOCS", "8")
    os.environ.setdefault("BASIC_NUM_SHARDS", "1")

    import torch
    from recon_bench import _build_model, _resolve_config

    model, device, _, _ = _build_model(_resolve_config(args.config))
    model.eval()
    # Prime host-side discourse/codebook state eagerly. Otherwise the first
    # compiled invocation is the cold-sentence variant (no AR prediction) and
    # the second necessarily specializes a warm-sentence variant, making a
    # one-warmup benchmark look like pathological same-bucket recompilation.
    with torch.no_grad():
        primer = model.inputSpace.prepInput([_sentence(2)] * int(args.batch))
        model._start_spaces_for_forward()
        model.forward(primer)
        model._end_step()
    model.enable_compiled_step()
    available = tuple(sorted(model._compiled_word_steps))
    requested = tuple(int(v) for v in args.widths)
    if available and any(v not in available for v in requested):
        raise RuntimeError(
            f"requested widths {requested} are not compiled choices "
            f"{available}")

    rows = []
    prior_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        for width in requested:
            text = _sentence(width)

            def _stage_once():
                # A forward consumes/mutates its carriers. Recreate the eager
                # stem before every sample, but start timing only afterward.
                model._end_step()
                x = model.inputSpace.prepInput([text] * int(args.batch))
                model._start_spaces_for_forward()
                staged_arg = model._begin_step(x)
                chosen = int(model.inputSpace._active_word_bucket)
                if chosen != width:
                    raise RuntimeError(
                        f"W={width} fixture selected W={chosen}; bucket "
                        "routing is not measuring the requested graph")
                fn = model._active_compiled_step or model._compiled_step
                return staged_arg, fn

            compile_wall = 0.0
            for i in range(int(args.warmup)):
                staged_arg, fn = _stage_once()
                t0 = time.perf_counter()
                result = fn(staged_arg)
                _sync(torch, device)
                if i == 0:
                    compile_wall = time.perf_counter() - t0
                del result

            samples = []
            if args.error_on_recompile:
                import torch._dynamo as _dynamo
                _dynamo.config.error_on_recompile = True
            for _ in range(int(args.iters)):
                staged_arg, fn = _stage_once()
                t0 = time.perf_counter()
                result = fn(staged_arg)
                _sync(torch, device)
                samples.append(time.perf_counter() - t0)
                del result
            model._end_step()

            median = statistics.median(samples)
            rows.append({
                "width": width,
                "backend": args.backend,
                "device": str(device),
                "batch": int(args.batch),
                "warmup_first_s": compile_wall,
                "median_s": median,
                "mean_s": statistics.fmean(samples),
                "min_s": min(samples),
                "words_per_s": (int(args.batch) * width / median
                                 if median > 0 else float("inf")),
                "samples_s": samples,
            })
    finally:
        torch.set_grad_enabled(prior_grad)
        model._end_step()

    print("W\tmedian_s\twords/s\tfirst_warmup_s")
    for row in rows:
        print(f"{row['width']}\t{row['median_s']:.6f}\t"
              f"{row['words_per_s']:.2f}\t{row['warmup_first_s']:.6f}")
    report = {
        "config": str(_resolve_config(args.config)),
        "device": str(device),
        "torch_version": torch.__version__,
        "backend": args.backend,
        "compiled_word_buckets": list(available),
        "n_concept_codes": int(getattr(model, "nConceptCodes", -1)),
        "n_symbols": int(getattr(model, "nSymbols", -1)),
        "n_symbol_slots": int(getattr(model, "nSymbolSlots", -1)),
        "stem_in_timing": False,
        "rows": rows,
    }
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
