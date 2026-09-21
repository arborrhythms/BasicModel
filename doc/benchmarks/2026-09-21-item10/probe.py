"""Fixed-seed native training baseline; emits measurements, no learning gate.

Run from basicmodel with PYTHONPATH=bin:test and the existing virtualenv.
The XML owns the model and data; only checkpoint writes and compilation are
disabled. Two warmup batches precede five timed batches, including epoch tails.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

os.environ.setdefault("MODEL_COMPILE", "eager")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("BASIC_AUTOLOAD", "false")

import numpy as np
import torch
import Language
from Models import BaseModel, _boundary_registry
from data import TheData
from util import init_config, init_device
from bounded_tests import source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="data/MM_ladder.xml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--thought", action="store_true", help="one explicit quantize request per composed sentence")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--parity", choices=("packed", "single"),
                        help="measure the same four validation sentences without updates")
    args = parser.parse_args()
    root = Path.cwd()
    torch.set_num_threads(1)
    for seed in (torch.manual_seed, random.seed, np.random.seed):
        seed(42)
    init_device("cpu")
    init_config(args.config, defaults_path="data/model.xml")
    cfg = BaseModel.load_config(args.config)
    TheData.load(cfg["architecture"]["data"]["dataset"], dat=dict(cfg["architecture"]["data"]))
    Language.TheGrammar._configured = False
    model, _ = BaseModel.from_config(args.config, data=TheData)
    model.set_sigma(0)
    model.checkpoint_every_batches = 0
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    optimizer = model.getOptimizer(lr=float(cfg["architecture"]["training"]["learningRate"]))
    source = source_snapshot(root)
    report = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "seed": 42, "config": args.config, "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
              "source": source, "torch": torch.__version__, "device": "cpu", "threads": 1,
              "batch_size": 2, "backend": "eager tensor loop", "phases": [],
              "thought_workload": "explicit quantize of a retained parsed leaf; no learned-query claim" if args.thought else None,
              "dictionary_rotation_rate": model.contextual_concept_learning_rate,
              "dictionary_parameter": isinstance(model.conceptualSpace.similarity_codebook.W, torch.nn.Parameter)}
    if args.parity:
        from parity import measure
        try:
            report["parity"] = measure(model, args.parity)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        finally:
            model.End()
            model.symbolSpace.soft_reset()
        return
    original_batch, original_thought, original_tail = model.runBatch, model.run_selected_thought, model.post_tick_compact
    thought_calls = 0
    steps = []
    completed = time.perf_counter()

    def thought(*a, **kw):
        nonlocal thought_calls
        thought_calls += 1
        return original_thought(*a, **kw)

    def batch(*a, **kw):
        before = thought_calls
        result = original_batch(*a, **kw)
        work = 0
        effects = 0
        if args.thought:
            registry = _boundary_registry(model)
            programs = model._last_understanding.answer_program
            for row, program in enumerate(programs):
                request = registry.form("quantize", program.leaves[0])
                outcome = model._run_public_thought(request, row=row, work_budget=128)
                work += outcome.work.spent
                effects += sum(record.kind == "thought" and record.operation == "quantize" for record in outcome.records)
        steps.append({"reconstruction": float(result[0].lossIn.detach()),
                      "answer": float(result[0].lossOut.detach()),
                      "thought_calls": thought_calls - before, "thought_work": work,
                      "execute_records": effects})
        return result

    def tail(*a, **kw):
        nonlocal completed
        result = original_tail(*a, **kw)
        now = time.perf_counter()
        steps[-1]["seconds"] = now - completed
        completed = now
        return result

    model.runBatch, model.run_selected_thought, model.post_tick_compact = batch, thought, tail
    try:
        phases = (("thought_evaluation", "validation", None, 12),) if args.eval_only else (
                                     ("before_training", "validation", None, 4),
                                      ("training", "train", optimizer, 7),
                                      ("after_training", "validation", None, 4))
        for name, split, opt, count in phases:
            steps = []
            completed = time.perf_counter()
            model.runEpoch(optimizer=opt, batchSize=2, split=split, max_batches=count)
            measured = steps[2:] if name in ("training", "thought_evaluation") else steps
            phase = {"name": name, "steps": steps,
                     "reconstruction_mean": statistics.mean(s["reconstruction"] for s in measured),
                     "sentences_per_second": 2 * len(measured) / sum(s["seconds"] for s in measured),
                     "thought_calls": sum(s["thought_calls"] for s in measured)}
            report["phases"].append(phase)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        model.runBatch, model.run_selected_thought, model.post_tick_compact = original_batch, original_thought, original_tail
        model.End()
    print(json.dumps({"phases": report["phases"], "dictionary_parameter": report["dictionary_parameter"]}))


if __name__ == "__main__":
    main()
