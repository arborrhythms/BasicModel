"""Predeclared seeds 0 through 7 native training baseline; emits measurements, no learning gate.

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
    parser.add_argument("--seed", type=int, choices=tuple(range(8)), required=True)
    parser.add_argument("--revision", required=True)
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
        seed(args.seed)
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
    report = {"revision": args.revision,
              "seed": args.seed, "config": args.config, "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
              "source": source, "torch": torch.__version__, "device": "cpu", "threads": 1,
              "batch_size": 2, "backend": "eager tensor loop", "phases": [],
              "thought_workload": "explicit quantize of a completed field; no learned-query claim" if args.thought else None,
              "dictionary_rotation_rate": model.contextual_concept_learning_rate,
              "dictionary_parameter": isinstance(model.conceptualSpace.similarity_codebook.W, torch.nn.Parameter),
              "initial_atom_prefix_sha256": hashlib.sha256(
                  model.conceptualSpace.similarity_codebook.W[:8].detach().cpu().numpy().tobytes()).hexdigest()}
    if args.parity:
        from parity import measure
        try:
            report["parity"] = measure(model, args.parity)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        finally:
            model.End()
            model.symbolSpace.soft_reset()
        return
    # Observe the identities actually supplied to each new predictor estimate.
    # This hook does not change the context or keep its autograd tensors alive.
    definition_reads = dict(estimates=0, rows=0, definitions=0)
    discourse = getattr(model.symbolSpace, 'discourse', None)
    if discourse is not None:
        original_expect = discourse.expect_next_meaning
        def expect(b=0, **kw):
            before = discourse._inter_last_meaning[b]
            result = original_expect(b, **kw)
            pending = discourse._inter_last_meaning[b]
            if pending is not None and pending is not before:
                definition_reads['estimates'] += 1
                store = discourse._ltm_store
                for occurrence in getattr(pending, 'source_occurrences', ()):
                    row = None if store is None else store._index_occurrences.get(occurrence)
                    if row is not None:
                        definition_reads['rows'] += 1
                        definition_reads['definitions'] += int(int(store.rel_type[row]) == getattr(store, 'REL_DEF', -1))
            return result
        discourse.expect_next_meaning = expect
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
            fields = model._last_understanding.sentence_states
            for row, field in enumerate(fields):
                request = registry.form("quantize", field.end_state[0])
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
            store = model.symbolSpace.ltm_store
            report['definition_context_reads'] = dict(definition_reads)
            report['store_owner_present'] = store is not None
            report['store_rows'] = 0 if store is None else len(store)
            report['definition_rows'] = 0 if store is None else int((store.rel_type[:len(store)] == getattr(store, 'REL_DEF', -1)).sum())
            report["phases"].append(phase)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        model.runBatch, model.run_selected_thought, model.post_tick_compact = original_batch, original_thought, original_tail
        model.End()
    print(json.dumps({"phases": report["phases"], "dictionary_parameter": report["dictionary_parameter"]}))


if __name__ == "__main__":
    main()
