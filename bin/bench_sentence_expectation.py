"""Reproducible sentence-expectation learning and full-training measurements.

See doc/benchmarks/2026-09-16-sentence-expectation.md for the preregistered
protocol, operating points and measured results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import threading
import time
import types

import torch
import torch.nn.functional as F


def meaning_documents(seed, count=64, length=24, width=8):
    """A fixed subject with independently varying predicate/object meanings."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    subject = torch.linspace(-0.3, 0.3, width, device="cpu")
    documents = []
    for _ in range(count):
        verb = torch.rand(width, generator=generator, device="cpu") * 1.2 - 0.6
        obj = torch.rand(width, generator=generator, device="cpu") * 1.2 - 0.6
        rows = []
        for _ in range(length):
            rows.append(torch.stack((subject, verb, obj)))
            verb, obj = verb.roll(1), 0.7 * obj.roll(2) + 0.3 * verb
        documents.append(torch.stack(rows))
    return documents


def meaning_windows(documents, context_window=4, occupancy=None):
    """Only predecessors from this document, with one cold start per document."""
    values, masks, targets, addresses, target_masks = [], [], [], [], []
    for document, rows in enumerate(documents):
        for index in range(1, len(rows)):
            prefix = rows[max(0, index-context_window):index].detach()
            padding = context_window - len(prefix)
            values.append(F.pad(prefix, (0, 0, 0, 0, padding, 0)))
            present = torch.zeros(context_window, 3, dtype=torch.bool, device=rows.device)
            if occupancy is None:
                present[padding:] = True
                target_masks.append(torch.ones(3, dtype=torch.bool, device=rows.device))
            else:
                present[padding:] = occupancy[document][max(0, index-context_window):index]
                target_masks.append(occupancy[document][index])
            masks.append(present)
            targets.append(rows[index].detach())
            addresses.append((document, index))
    if not values:
        return None
    return (torch.stack(values), torch.stack(masks), torch.stack(targets), addresses,
            torch.stack(target_masks))


def summarize_steps(steps, warmup=2):
    """Use observed counts, never configured B times the number of calls."""
    measured = steps[warmup:]
    if not measured:
        raise ValueError("no measured steps remain after warmup")
    seconds = sum(step["seconds"] for step in measured)
    observations = sum(step["observations"] for step in measured)
    inputs = sum(step["input_sentences"] for step in measured)
    pairs = sum(step["predicted_targets"] for step in measured)
    supplied = sum(step.get("supplied_answer_rows", 0) for step in measured)
    return {
        "steps": len(measured), "seconds": seconds,
        "warmup_seconds": sum(step["seconds"] for step in steps[:warmup]),
        "median_step_seconds": statistics.median(s["seconds"] for s in measured),
        "input_sentences": inputs, "observations": observations, "predicted_targets": pairs,
        "input_sentences_per_second": inputs / seconds,
        "predicted_targets_per_second": pairs / seconds,
        "supplied_answer_rows": supplied,
        "supplied_answer_rows_per_second": supplied / seconds,
        "document_boundary_fraction": sum(s["cold_starts"] for s in measured) / max(1, observations),
        "reconstruction_mean": statistics.mean(s["reconstruction"] for s in measured),
        "supplied_answer_mean": statistics.mean(s["answer"] for s in measured),
    }


def _controlled_inputs(values, masks, control, seed):
    if control == "context_free":
        return torch.zeros_like(values), masks
    if control == "shuffled":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(len(values), generator=generator, device="cpu")
        return values[order.to(values.device)], masks[order.to(masks.device)]
    return values, masks


def _score_head(head, values, masks, targets, root=False, target_masks=None):
    parameter = next(head.parameters())
    values, masks, targets = values.to(parameter), masks.to(parameter.device), targets.to(parameter)
    target_masks = (torch.ones(targets.shape[:2], dtype=torch.bool, device=parameter.device)
                    if target_masks is None else target_masks.to(parameter.device))
    with torch.no_grad():
        if root:
            prediction = head(values[:, :, 0], routing=None, parallel=False)
            roles = prediction[:, None].expand(-1, 3, -1)
            presence = None
        else:
            roles, presence = head(values, masks)
        squared = (roles - targets).square().mean(-1)
        per_role = (squared * target_masks).sum(0) / target_masks.sum(0).clamp(min=1)
        return {
            "feature_mse": float(((squared * target_masks).sum(-1)
                                  / target_masks.sum(-1).clamp(min=1)).mean()),
            "role_mse": per_role.tolist(),
            "root_objective_mse": float(per_role[0]),
            "presence_bce": None if presence is None else float(
                F.binary_cross_entropy_with_logits(presence, target_masks.to(presence.dtype))),
            "predicted_targets": len(targets),
            "target_role_variance": targets.var(0, unbiased=False).mean(-1).tolist(),
            "estimate_role_variance": roles.var(0, unbiased=False).mean(-1).tolist(),
        }


def synthetic_benchmark(seeds=(0, 1, 2), updates=300, batch_size=128):
    from Layers import SentenceExpectation, InterSentenceLayer
    from util import init_device
    init_device("cpu")
    report = {"workload": "fixed synthetic meanings", "device": "cpu",
              "encoder_training": False, "context_window": 4,
              "positive_gate": "20% lower held-out feature MSE than both trained controls for every seed",
              "runs": []}
    for seed in seeds:
        train_docs = meaning_documents(1000 + seed)
        heldout_docs = meaning_documents(2000 + seed, count=16)
        train_x, train_m, train_y, train_addresses, _ = meaning_windows(train_docs)
        val_x, val_m, val_y, val_addresses, _ = meaning_windows(heldout_docs)
        runs = {}
        for control in ("ordered", "shuffled", "context_free", "root"):
            torch.manual_seed(seed)
            if control == "root":
                head = InterSentenceLayer(4, 8, 8, concept_dim=8,
                                          ltm_capacity=4, expectation_scope="root")._inter_predictor
            else:
                head = SentenceExpectation(8, 4)
            x, mask = _controlled_inputs(train_x, train_m, control, 3000 + seed)
            vx, vm = _controlled_inputs(val_x, val_m, control, 4000 + seed)
            before = _score_head(head, vx, vm, val_y, root=control == "root")
            optimizer = torch.optim.Adam(head.parameters(), lr=0.003)
            generator = torch.Generator(device="cpu").manual_seed(5000 + seed)
            start = time.perf_counter()
            for _ in range(updates):
                indices = torch.randint(len(x), (batch_size,), generator=generator)
                optimizer.zero_grad(set_to_none=True)
                if control == "root":
                    predicted = head(x[indices, :, 0], routing=None, parallel=False)
                    loss = F.mse_loss(predicted, train_y[indices, 0])
                else:
                    predicted, logits = head(x[indices], mask[indices])
                    loss = F.mse_loss(predicted, train_y[indices]) + F.binary_cross_entropy_with_logits(
                        logits, torch.ones_like(logits))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"synthetic {control} seed {seed} diverged")
                loss.backward()
                optimizer.step()
            after = _score_head(head, vx, vm, val_y, root=control == "root")
            runs[control] = {
                "before": before, "after": after, "updates": updates,
                "training_examples_processed": updates * batch_size,
                "parameter_count": sum(p.numel() for p in head.parameters()),
                "training_seconds": time.perf_counter() - start,
                "train_pairs": len(train_addresses), "heldout_pairs": len(val_addresses),
            }
            if control == "ordered":
                for ablation in ("shuffled", "context_free"):
                    ax, am = _controlled_inputs(val_x, val_m, ablation, 4000 + seed)
                    runs[control]["checkpoint_" + ablation] = _score_head(head, ax, am, val_y)
        error = runs["ordered"]["after"]["feature_mse"]
        passed = all(error <= 0.8 * runs[c]["after"]["feature_mse"]
                     for c in ("shuffled", "context_free"))
        report["runs"].append({"seed": seed, "controls": runs, "gate_passed": passed})
    report["gate_passed"] = all(r["gate_passed"] for r in report["runs"])
    return report


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device):
    result = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if sys.platform != "darwin":
        result["process_peak_rss_bytes"] *= 1024
    if device.type == "mps":
        result["sampled_mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        result["sampled_mps_tensor_bytes"] = torch.mps.current_allocated_memory()
    elif device.type == "cuda":
        result["cuda_peak_tensor_bytes"] = torch.cuda.max_memory_allocated(device)
    return result


def _counter_snapshot():
    from torch._dynamo.utils import counters, compilation_time_metrics
    result = {group: dict(counters[group]) for group in (
        "frames", "stats", "aot_autograd", "graph_break")}
    result["compiler_timing_seconds"] = {
        name: list(durations) for name, durations in compilation_time_metrics.items()}
    return result


def observe_optimizer_steps(optimizer, callback):
    """Observe successful outer steps, including the native MultiOptimizer.

    Counting child Adam calls would inflate the model's update count. Restore
    the exact instance surface when the observer is removed.
    """
    previous_step = optimizer.step
    had_override = "step" in vars(optimizer)
    previous_override = vars(optimizer).get("step")

    def observed_step(*args, **kwargs):
        result = previous_step(*args, **kwargs)
        callback(optimizer, args, kwargs)
        return result

    optimizer.step = observed_step

    def remove():
        if had_override:
            optimizer.step = previous_override
        else:
            del optimizer.step

    return types.SimpleNamespace(remove=remove)


class MemorySampler:
    """Observed device high-water mark at 20 ms intervals, plus OS peak RSS."""

    def __init__(self, device):
        self.device = device
        self.peak = {}
        self.stop = threading.Event()
        self.thread = None

    def _sample(self):
        while True:
            for key, value in _memory(self.device).items():
                self.peak[key] = max(self.peak.get(key, 0), value)
            if self.stop.wait(0.02):
                break

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=1)


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruction_observations(model, batch_result):
    """Observe the completed result; never run another reconstruction.

    Event error is common to legacy evaluation and tied reconstruction. Byte
    and idea costs are reported only when the executed path published them.
    Output tensor dimensions are not an emitted-word count.
    """
    understanding = getattr(model, "_last_understanding", None)
    owned = getattr(understanding, "input_reconstruction", None)
    predicted = owned.event if owned is not None else batch_result.inputPred
    target = batch_result.forwardInput
    report = {"owned": owned is not None, "event_mse": None, "event_score": None,
              "byte_cost": None, "idea_mse": None, "truncated_rows": None}
    with torch.no_grad():
        if torch.is_tensor(predicted) and torch.is_tensor(target):
            width, count = min(predicted.shape[-1], target.shape[-1]), min(
                predicted.shape[1], target.shape[1])
            report["event_mse"] = float(F.mse_loss(
                predicted[:, :count, :width], target[:, :count, :width]))
            report["event_score"] = float(model._reverse_event_loss(predicted, target))
        if owned is not None:
            report.update(byte_cost=float(owned.byte_cost.mean()),
                          idea_mse=float(owned.idea_cost.mean()),
                          truncated_rows=int(owned.truncated.sum()))
        rows = getattr(model.inputSpace, "_ar_concept_lookup_rows", None)
        report["active_basis_rows"] = ((rows >= 0).sum(-1).cpu().tolist()
                                        if torch.is_tensor(rows) else None)
    return report


def native_benchmark(config, *, device="cpu", backend="eager", docs=24,
                     batch_size=1, train_steps=7, eval_steps=8, answers=False,
                     destination=None, basis_limit=None):
    """The native loader and epoch own all scheduling, training and resets."""
    from Models import BaseModel
    from data import TheData
    from util import TheDevice, TheXMLConfig, init_config, init_device, init_compile_backend
    import Language
    import util
    import numpy as np
    from What import WhatRelation

    config = Path(config).resolve()
    root = Path(__file__).resolve().parents[1]
    defaults = root / "data/model.xml"
    setup_started = time.perf_counter()
    init_device(device)
    init_compile_backend(backend)
    init_config(str(config), defaults_path=str(defaults))
    arch = copy.deepcopy(TheXMLConfig.data.get("architecture", {}))
    # Match ModelFactory's precedence: an explicit environment mode wins.
    if not os.environ.get("MODEL_AMP") and arch.get("amp"):
        os.environ["MODEL_AMP"] = str(arch["amp"])
    util.init_model_amp()
    data_options = arch.get("data", {})
    train_options = arch.get("training", {})
    for rng in (torch.manual_seed, random.seed, np.random.seed):
        rng(42)
    dataset = data_options.get("dataset", "text")
    TheData.load(dataset, num_shards=1, max_docs=docs,
                 shard_dir=data_options.get("shardDir"), dat=data_options,
                 max_sentence_words=int(data_options.get("maxSentenceWords", 0)) or None)
    Language.TheGrammar._configured = False
    target_device = TheDevice.get()
    if target_device.type == "mps":
        init_device("cpu")
    try:
        model, _ = BaseModel.from_config(str(config), data=TheData)
    finally:
        init_device(target_device)
    model.to(target_device)
    model.set_sigma(0)
    if basis_limit is not None:
        if basis_limit < 1:
            raise ValueError("basis_limit must be positive")
        model.reconstruction_basis_limit = basis_limit
    # A benchmark must not replace the user's training checkpoint.
    model.checkpoint_every_batches = 0
    model.enable_compiled_step()
    assert bool(model.inputSpace.data.has_supervised_outputs) == bool(answers)
    discourse = model.symbolSpace.discourse
    if discourse is None:
        raise ValueError("this benchmark requires explicit sentence expectation")
    assert discourse.expectation_scope == "structured"
    if train_steps < 7:
        raise ValueError("at least two warmup and five measured training steps are required")
    if bool(getattr(model, "two_pass_learning", False)):
        raise ValueError("two-pass learning requires separate timing attribution")
    if answers and not 0.0 < model.loss.reconstruction_scale < 1.0:
        raise ValueError("supplied-answer timing requires nonzero reconstruction and answer weights")
    amp_context, _ = util.amp_context()
    with amp_context:
        compute_dtype = str(util.autocast_compute_dtype(target_device))

    addresses = model.inputSpace.data.source_addresses
    doc_sets = {split: {row["document"] for row in rows}
                for split, rows in addresses.items() if split in ("train", "validation", "test")}
    if not answers:
        assert not (doc_sets["train"] & doc_sets["validation"])
    report = {
        "workload": "supplied answers" if answers else "FineWeb-Edu",
        "dataset": dataset, "device": str(target_device), "backend": str(util.TheCompileBackend),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "allocator_environment": {key: os.environ.get(key) for key in (
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")},
        "amp_requested": util.MODEL_AMP,
        "autocast_compute_dtype": compute_dtype,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "command": sys.argv,
        "config": str(config), "config_sha256": _fingerprint(config),
        "defaults_sha256": _fingerprint(defaults), "effective_architecture": arch,
        "runtime_sha256": {f: _fingerprint(root / "bin" / f) for f in (
            "Models.py", "Language.py", "Layers.py", "Understanding.py", "Spaces.py", "data.py", "Optimizer.py",
            "mps_memory.py", "bench_sentence_expectation.py")},
        "source_manifest": copy.deepcopy(TheData.source_manifest),
        "setup_seconds": time.perf_counter() - setup_started,
        "max_docs": docs, "batch_size": batch_size,
        "detached_reverse": bool(model.detached_reverse),
        "reconstruct_in_loop": bool(model.reconstruct_in_loop),
        "reconstruction_basis_limit": int(getattr(model, "reconstruction_basis_limit", 16)),
        "reconstruction_placement": model._recon_placement(),
        "reconstruction_backend": (
            None if not model.reconstruct_in_loop else
            "none" if model._recon_placement() == "eager" else
            "aot_eager" if model._recon_placement() == "compiled" and util.TheCompileBackend == "eager"
            else str(util.TheCompileBackend)),
        "overrides": {"reconstruction_basis_limit": basis_limit} if basis_limit is not None else {},
        "sigma": 0, "answer_synthesis": bool(model.answer_synthesis),
        "output_in_loop": bool(model.output_in_loop),
        "fullgraph_word_loop": bool(getattr(model, "_compiled_word_loop_fullgraph", False)),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "inter_weight": model.inter_loss_weight, "arma_weight": model.arma_scale,
        "reconstruction_weight": model.loss.reconstruction_scale,
        "answer_weight": 1.0 - model.loss.reconstruction_scale,
        "document_counts": {s: len(v) for s, v in doc_sets.items()},
        "phases": [],
        "timing_scope": "consecutive native step completions including next input staging, runBatch, reset/compaction and measurement overhead; epoch wall time also reported",
        "compiler_timing_scope": "PyTorch named compiler intervals in seconds; nested intervals overlap and must not be summed together",
    }
    shard = root / "data/fineweb/shard_00000.parquet"
    if not answers and shard.exists():
        report["shard_sha256"] = _fingerprint(shard)
    original_batch, original_compact = model.runBatch, model.post_tick_compact
    original_inter = model._discourse_inter_loss
    original_output_walk = model._compiled_output_walk
    active = None
    start_time = None
    previous_completed = None
    captured = {}
    target_detached = True
    trained_inter = None
    optimizer_steps = 0
    output_lengths = None
    output_truncated = None
    shared_gradients = {}
    original_observe = discourse._observe_meanings

    def capture_meanings(self, depths, payloads, tetralemmas, mask, layout, role_masks):
        nonlocal target_detached
        result = original_observe(depths, payloads, tetralemmas, mask, layout, role_masks)
        for row, payload in enumerate(payloads):
            if payload is None or not int(depths[row]) or (mask is not None and not bool(mask[row])):
                continue
            comparison = self.last_expectation_comparison(row)
            if comparison is not None:
                target_detached &= not comparison.observed.requires_grad
            if report["phases"] and report["phases"][-1]["name"].startswith("validation"):
                value, occupied = self._canonical_meaning(
                    payload, depths[row], layout, None if role_masks is None else role_masks[row])
                if bool(occupied.any()):
                    key = (row, self._expectation_documents[row])
                    captured.setdefault(key, []).append((value.detach().cpu(), occupied.cpu()))
        return result

    def write_report():
        if destination:
            Path(destination).write_text(json.dumps(report, indent=2) + "\n")

    def measured_inter(self):
        nonlocal trained_inter
        value = original_inter()
        trained_inter = ({"mean": float(value.detach()), "requires_grad": value.requires_grad}
                         if value is not None else None)
        return value

    def stepped(_optimizer, _args, _kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1

    def measured_batch(self, *args, **kwargs):
        nonlocal start_time, active, trained_inter, output_lengths, output_truncated
        before = discourse.expectation_metrics()
        steps_before = optimizer_steps
        trained_inter = None
        output_lengths = output_truncated = None
        shared_gradients.clear()
        _sync(target_device)
        start_time = time.perf_counter()
        result = original_batch(*args, **kwargs)
        _sync(target_device)
        after = discourse.expectation_metrics()
        batch_result = result[0]
        mask = self._last_answer_mask
        questions = [a.question for a in self._last_what_desired]
        supplied = sum(bool(mask[row]) and question.relation is WhatRelation.SUPERVISED
                       for row, question in enumerate(questions)) if mask is not None else 0
        word_mask = getattr(self.inputSpace, "_word_active_mask", None)
        sentence_mask = (self.inputSpace._packed_sentence_slot_mask
                         if self.inputSpace._sentence_pack_enabled
                         else word_mask.any(dim=1))
        active = {
            "input_sentences": int(sentence_mask.sum()),
            "observations": after["observations"] - before["observations"],
            "predicted_targets": after["predicted_targets"] - before["predicted_targets"],
            "cold_starts": after["cold_starts"] - before["cold_starts"],
            "reconstruction": float(batch_result.lossIn.detach()),
            "answer": float(batch_result.lossOut.detach()),
            "supplied_answer_rows": supplied,
            "training": bool(kwargs.get("train", False)),
            "optimizer_steps": optimizer_steps - steps_before,
            "trained_inter": trained_inter,
            "word_counts": word_mask.sum(-1).detach().cpu().tolist()
                           if torch.is_tensor(word_mask) else None,
            "sources": copy.deepcopy(kwargs.get("source_rows")),
            "reconstruction_fidelity": reconstruction_observations(self, batch_result),
            "output_word_counts": output_lengths,
            "output_truncated_rows": output_truncated,
            "output_tensor_shape": (list(batch_result.outputPred.shape)
                                    if torch.is_tensor(batch_result.outputPred) else None),
            "shared_compose_gradient_norms": {
                name: float(torch.stack(values).sum().sqrt())
                for name, values in shared_gradients.items()},
            "forward_backward_optimizer_seconds": time.perf_counter() - start_time,
            "context_detached": all(not p.requires_grad for chain in discourse._inter_context
                                     for _, p, _ in chain),
        }
        if not all(math.isfinite(active[key]) for key in ("reconstruction", "answer")):
            raise FloatingPointError("native benchmark produced a nonfinite reconstruction or answer loss")
        if trained_inter is not None and not math.isfinite(trained_inter["mean"]):
            raise FloatingPointError("native benchmark produced a nonfinite expectation loss")
        return result

    def measured_output_walk(self):
        walk = original_output_walk()

        def observe(*args, **kwargs):
            nonlocal output_lengths, output_truncated
            result = walk(*args, **kwargs)
            output_lengths = result[1].detach().cpu().tolist()
            output_truncated = int(result[2].detach().sum())
            return result

        return observe

    def measured_compact(self, *args, **kwargs):
        nonlocal active, start_time, previous_completed
        result = original_compact(*args, **kwargs)
        if active is not None:
            _sync(target_device)
            completed = time.perf_counter()
            active["seconds"] = completed - previous_completed
            active["batch_plus_tail_seconds"] = completed - start_time
            previous_completed = completed
            active["memory"] = _memory(target_device)
            active["compiler_counters"] = _counter_snapshot()
            report["phases"][-1]["steps"].append(active)
            active = start_time = None
            write_report()
        return result

    model.runBatch = types.MethodType(measured_batch, model)
    model.post_tick_compact = types.MethodType(measured_compact, model)
    model._discourse_inter_loss = types.MethodType(measured_inter, model)
    model._compiled_output_walk = types.MethodType(measured_output_walk, model)
    discourse._observe_meanings = types.MethodType(capture_meanings, discourse)
    optimizer = model.getOptimizer(lr=float(train_options.get("learningRate", 0.0005)))
    step_hook = observe_optimizer_steps(optimizer, stepped)
    # Only compose transforms, with true optimizer ownership. Hooks observe
    # every backward contribution, before the model's joint-gradient balance;
    # the final parameter delta separately proves an optimizer update.
    compose_parameters = {}
    seen = set()
    binary = model.languageSpace._tree_layer(2)
    for index, op in enumerate(binary.ops):
        for name, parameter in op.named_parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                compose_parameters[f"{binary.op_names[index]}.{name}"] = parameter
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    report["compose_optimizer_ownership"] = {
        name: id(p) in optimizer_ids for name, p in compose_parameters.items()}
    compose_before = {name: p.detach().cpu().clone() for name, p in compose_parameters.items()}
    gradient_hooks = []
    for name, parameter in compose_parameters.items():
        def observe_gradient(gradient, key=name):
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            shared_gradients.setdefault(key, []).append(values.detach().float().square().sum())
        gradient_hooks.append(parameter.register_hook(observe_gradient))
    head_before = {n: p.detach().cpu().clone() for n, p in discourse._inter_predictor.named_parameters()}
    fixed_heldout = None
    answer_before = []
    try:
        for name, split, opt, count in (
                ("validation_before", "validation", None, eval_steps),
                ("training", "train", optimizer, train_steps),
                ("validation_after", "validation", None, eval_steps)):
            report["phases"].append({"name": name, "split": split, "steps": []})
            captured.clear()
            _sync(target_device)
            start = time.perf_counter()
            previous_completed = start
            with MemorySampler(target_device) as memory:
                model.runEpoch(optimizer=opt, batchSize=batch_size, split=split, max_batches=count)
            _sync(target_device)
            phase = report["phases"][-1]
            phase["epoch_seconds"] = time.perf_counter() - start
            phase["peak_memory"] = memory.peak
            phase["memory_sampling_interval_seconds"] = 0.02
            phase["expectation"] = discourse.expectation_metrics()
            if name == "training":
                if len(phase["steps"]) < 7:
                    raise ValueError("dataset exhausted before five measured training steps")
                if any(s["optimizer_steps"] != 1 for s in phase["steps"]):
                    raise RuntimeError("native training did not take exactly one optimizer step per batch")
                phase["steady"] = summarize_steps(phase["steps"], warmup=2)
            elif captured:
                groups = list(captured.values())
                views = meaning_windows(
                    [torch.stack([v for v, _ in rows]) for rows in groups],
                    discourse._inter_chain_window,
                    [torch.stack([m for _, m in rows]) for rows in groups])
                if name == "validation_before":
                    fixed_heldout = views
                if views is not None:
                    phase["encoded_meaning_controls"] = {}
                    for control in ("ordered", "shuffled", "context_free"):
                        vx, vm = _controlled_inputs(views[0], views[1], control, 8042)
                        phase["encoded_meaning_controls"][control] = _score_head(
                            discourse._inter_predictor, vx, vm, views[2], target_masks=views[4])
                if fixed_heldout is not None and name == "validation_after":
                    phase["fixed_pretraining_encoding_score"] = _score_head(
                        discourse._inter_predictor, fixed_heldout[0], fixed_heldout[1],
                        fixed_heldout[2], target_masks=fixed_heldout[4])
            if name == "validation_before" and answers:
                answer_before = [(p, p.detach().cpu().clone()) for p in model.synthesis_parameters()]
            if name.startswith("validation"):
                assert all(s["optimizer_steps"] == 0 and s["trained_inter"] is None
                           for s in phase["steps"])
                assert discourse._inter_loss_count == 0
            write_report()
        report["head_parameter_delta_norm"] = sum(
            float((p.detach().cpu() - head_before[n]).square().sum())
            for n, p in discourse._inter_predictor.named_parameters()) ** 0.5
        report["answer_parameter_delta_norm"] = sum(
            float((p.detach().cpu() - before).square().sum()) for p, before in answer_before) ** 0.5
        report["compose_parameter_delta_norms"] = {
            name: float((p.detach().cpu() - compose_before[name]).norm())
            for name, p in compose_parameters.items()}
        report["target_detached"] = target_detached
        report["context_detached"] = all(step["context_detached"] for phase in report["phases"]
                                         for step in phase["steps"])
        training = next(p for p in report["phases"] if p["name"] == "training")
        report["predictor_update_verified"] = (
            training["expectation"]["predicted_targets"] > 0
            and report["head_parameter_delta_norm"] > 0)
        report["supplied_answer_update_verified"] = (
            answers and training["steady"]["supplied_answer_rows"] > 0
            and training["steady"]["supplied_answer_mean"] > 0
            and report["answer_parameter_delta_norm"] > 0)
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        step_hook.remove()
        for hook in gradient_hooks:
            hook.remove()
        model.runBatch, model.post_tick_compact = original_batch, original_compact
        model._discourse_inter_loss = original_inter
        model._compiled_output_walk = original_output_walk
        discourse._observe_meanings = original_observe
        write_report()
        model.End()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("synthetic", "fineweb", "answers"))
    parser.add_argument("--config", default="data/BasicModel.xml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="eager")
    parser.add_argument("--docs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=7)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--basis-limit", type=int)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.workload == "synthetic":
        result = synthetic_benchmark(updates=args.updates)
    else:
        result = native_benchmark(args.config, device=args.device, backend=args.backend,
                                  docs=args.docs, batch_size=args.batch_size,
                                  train_steps=args.train_steps, eval_steps=args.eval_steps,
                                  answers=args.workload == "answers", destination=args.out,
                                  basis_limit=args.basis_limit)
    root = Path(__file__).resolve().parents[1]
    result.setdefault("revision", subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    result.setdefault("command", sys.argv)
    result.setdefault("torch", torch.__version__)
    result.setdefault("threads", torch.get_num_threads())
    result.setdefault("runtime_sha256", {
        name: _fingerprint(root / "bin" / name)
        for name in ("Layers.py", "bench_sentence_expectation.py")})
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
