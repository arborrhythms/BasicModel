#!/usr/bin/env python3
"""Evaluate mathematical thinking on the exact-arithmetic testbed.

Mathematical thinking spec section 9 / 10 (plan Phase 5): run one
checkpoint (or a fresh model) at several iteration budgets with an equal
primitive allowance per iteration, and report -- by depth and budget --
exact-answer accuracy, forced closures, mean iterations, latency, and the
memory ablation.  (The runtime carries no exact primitives -- Alec
2026-09-09 -- so there is no derivation to verify; the exact code is data
generation only.)

Usage:

    python bin/eval_math_thinking.py --config data/MM_math.xml \\
        --budgets 1,4,8,16 --seeds 1,2,3 --split test --out report.md

    # a trained checkpoint
    python bin/eval_math_thinking.py --config data/MM_math.xml \\
        --weights output/math.ckpt --budgets 1,8 --seeds 1

The report is Markdown; ``--json`` also writes the raw rows.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MODEL_COMPILE", "eager")
# Evaluation runs eager on the CPU unless the caller pins another device
# (the same pin the test suite uses); the model and its data must agree.
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = Path(__file__).resolve().parent
_ROOT = _BIN.parent
if str(_BIN) not in sys.path:
    sys.path.insert(0, str(_BIN))


def build_model(config_path, weights=None, seed=0):
    import torch
    import Language
    from util import init_config
    from data import TheData
    import Models

    config_path = str(config_path)
    init_config(path=config_path, defaults_path=str(_ROOT / "data" / "model.xml"))
    Language.TheGrammar._configured = False
    cfg = Models.BaseModel.load_config(config_path)
    dat = cfg.get("architecture", {}).get("data", {}) or {}
    TheData.load(dat.get("dataset", "math"), dat=dat)
    torch.manual_seed(int(seed))
    model, _ = Models.BaseModel.from_config(config_path, data=TheData)
    from util import TheDevice
    model = model.to(TheDevice.get())
    if weights:
        model.load_weights(str(weights), require_match=True)
    model.eval()
    return model


def _hide_memory(memory):
    """Memory ablation: the chooser context sees no interaction memory
    (representations, open question, latest output) while parity and the
    slot discipline are untouched."""
    original = memory.what_context

    def hidden(question=None, b=0):
        ctx = original(question=question, b=b)
        ctx = dict(ctx)
        ctx["input_representations"] = ()
        ctx["output_representations"] = ()
        ctx["open_question"] = None
        ctx["latest_output"] = None
        return ctx

    memory.what_context = hidden
    return original


def evaluate(model, *, split, budget, rows=None, batch=8, ablate_memory=False,
             illumination=False):
    """Run ``think()`` over ``split`` at ``budget`` iterations; return rows of
    per-problem measurements."""
    import torch
    from What import What

    data = model.inputSpace.data
    problems = data.math_problems[split]
    n = len(problems) if rows is None else min(int(rows), len(problems))
    memory = model._what_memory()
    restore = _hide_memory(memory) if (ablate_memory and memory is not None) else None
    del illumination                # retained for CLI compatibility
    out = []
    try:
        for start in range(0, n, int(batch)):
            idx = list(range(start, min(n, start + int(batch))))
            loader = data.data_loader(split=split, num_streams=len(idx))
            # data_loader yields the split in order; take the matching slice
            inp_items, out_items = None, None
            for k, (i_items, o_items) in enumerate(loader):
                if k == start // int(batch):
                    inp_items, out_items = i_items, o_items
                    break
            if inp_items is None:
                break
            x = model.inputSpace.prepInput(inp_items)
            y = model.outputSpace.prepOutput(out_items)
            questions = tuple(What.supervised(i, split=split) for i in idx)
            t0 = time.time()
            with torch.no_grad():
                result = model.think(questions, x, max_iterations=int(budget))
            latency = (time.time() - t0) / max(1, len(idx))
            target = y.reshape(len(idx), -1).argmax(-1)
            for b, i in enumerate(idx):
                problem = problems[i]
                answer = result.answers[b].what
                predicted = int(torch.as_tensor(answer).reshape(-1).argmax())
                out.append({
                    "row": i, "depth": problem.depth, "budget": int(budget),
                    "correct": int(predicted == int(target[b])),
                    "iterations": result.iterations,
                    "forced": result.forced_closures,
                    "latency_s": latency,
                })
    finally:
        if restore is not None:
            memory.what_context = restore
    return out


def summarize(rows):
    """Aggregate by (budget, depth) and by budget."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["budget"], r["depth"])].append(r)
        groups[(r["budget"], "all")].append(r)
    table = {}
    for key, items in groups.items():
        n = max(1, len(items))
        table[key] = {
            "n": len(items),
            "accuracy": sum(r["correct"] for r in items) / n,
            "forced": sum(r["forced"] for r in items) / n,
            "iterations": sum(r["iterations"] for r in items) / n,
            "latency_s": sum(r["latency_s"] for r in items) / n,
        }
    return table


REPORT_COLUMNS = ("n", "accuracy", "forced", "iterations", "latency_s")


def render(tables, *, config, weights, seeds, split, extra=None):
    lines = ["# Mathematical thinking evaluation", "",
             f"Config `{config}`; weights `{weights or 'fresh (untrained)'}`; "
             f"seeds {list(seeds)}; split `{split}`.", ""]
    for label, table in tables.items():
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| budget | depth | " + " | ".join(REPORT_COLUMNS) + " |")
        lines.append("|---|---|" + "|".join("---:" for _ in REPORT_COLUMNS) + "|")
        for (budget, depth) in sorted(table, key=lambda k: (k[0], str(k[1]))):
            row = table[(budget, depth)]
            cells = []
            for c in REPORT_COLUMNS:
                v = row[c]
                cells.append(f"{v}" if isinstance(v, int) else f"{v:.3f}")
            lines.append(f"| {budget} | {depth} | " + " | ".join(cells) + " |")
        lines.append("")
    if extra:
        lines.extend(extra)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=str(_ROOT / "data" / "MM_math.xml"))
    parser.add_argument("--weights", default=None)
    parser.add_argument("--budgets", default="1,4,8,16")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--ablate-memory", action="store_true")
    parser.add_argument("--illumination", action="store_true",
                        help="retained for compatibility; no effect")
    parser.add_argument("--out", default=None, help="Markdown report path")
    parser.add_argument("--json", default=None, help="raw rows path")
    args = parser.parse_args(argv)

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_rows = []
    tables = {}
    for seed in seeds:
        model = build_model(args.config, args.weights, seed=seed)
        for budget in budgets:
            rows = evaluate(model, split=args.split, budget=budget, rows=args.rows,
                            batch=args.batch, illumination=args.illumination)
            for r in rows:
                r["seed"] = seed
                r["ablation"] = "none"
            all_rows.extend(rows)
            if args.ablate_memory:
                ab = evaluate(model, split=args.split, budget=budget, rows=args.rows,
                              batch=args.batch, ablate_memory=True,
                              illumination=args.illumination)
                for r in ab:
                    r["seed"] = seed
                    r["ablation"] = "memory"
                all_rows.extend(ab)
    tables["thinking"] = summarize([r for r in all_rows if r["ablation"] == "none"])
    if args.ablate_memory:
        tables["memory ablation"] = summarize(
            [r for r in all_rows if r["ablation"] == "memory"])
    text = render(tables, config=args.config, weights=args.weights, seeds=seeds,
                  split=args.split)
    if args.out:
        Path(args.out).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=1))
    print(text)
    return all_rows


if __name__ == "__main__":
    main()
