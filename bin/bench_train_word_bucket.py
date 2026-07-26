#!/usr/bin/env python3
"""Measure steady compiled training throughput for synthetic or corpus input.

Unlike ``bench_word_buckets.py``, this drives ``runBatch(train=True)`` with a
real optimizer. By default every row is the same fixed-width synthetic
sentence; ``--corpus`` instead cycles through the configured training split's
contiguous-stream batches. On the tensor peer scheduler, ``--width`` is the
synthetic live word count while the configured W (normally 512) is only
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


def _stage_peer_profile_batch(model, texts):
    """Run the same eager lexical boundary used before the fullgraph loop."""
    model._start_spaces_for_forward()
    raw = model.inputSpace.prepInput(texts)
    model._staged_in_sub = model._lex_embed_stem(raw)
    symbol = model.symbolSpace
    if not getattr(symbol, "_per_sentence_initialized", False):
        symbol.soft_reset()
        symbol._per_sentence_initialized = True
    model._stage_reconstruction_teacher()
    slab = model.inputSpace._ar_embedded_N
    model._prepare_reconstruction_choices(
        int(slab.shape[0]), int(slab.shape[1]), slab.device)
    model.conceptualSpace.stm.begin_forward(
        int(slab.shape[0]), device=slab.device, dtype=slab.dtype)
    model._stage_fixed_residual_part_capacity()
    model._stage_intersentence_seed()
    return raw


def _profile_peer_legs(torch, model, device, texts, repeats):
    """Time the exact compiled CSSub, CSSym, and CSLang word stages."""
    from Models import FunctionalPeerSTM
    from Layers import ShortTermMemory
    import util as model_util

    _stage_peer_profile_batch(model, texts)
    isp = model.inputSpace
    ps = model.perceptualSpace
    ws = model.wholeSpace
    cs = model.conceptualSpace
    ss = model.symbolSpace
    language = model.languageSpace
    stm = cs.stm
    fold_passes = tuple(range(max(0, int(model.subsymbolicOrder) - 1)))
    symbolic_passes = tuple(range(max(0, int(model.symbolicOrder))))
    batch = int(isp._ar_embedded_N.shape[0])
    capacity = int(stm.capacity)

    part_ids = isp._ar_word_part_ids[:, :1, :]
    part_mask = isp._ar_word_part_mask[:, :1, :]
    part_offsets = isp._ar_word_part_offsets[:, :1, :]
    property_weights = ws._staged_word_property_weights[:, 0, :, :]

    def ps_leg(ids, mask, offsets):
        return ps.compute_word_fold_sources(
            ids, mask, offsets, fold_passes)

    def ws_leg(weights):
        return ws.compute_word_property_fold_sources(
            weights, fold_passes)

    compiled_ps = model_util.compile(
        ps_leg, verbose=False, fullgraph=True)
    compiled_ws = model_util.compile(
        ws_leg, verbose=False, fullgraph=True)
    part_sources = tuple(
        value.detach() for value in compiled_ps(
            part_ids, part_mask, part_offsets))
    whole_sources = tuple(
        value.detach() for value in compiled_ws(property_weights))
    _sync(torch, device)

    row = isp._ar_word_concept_rows[:, 0]
    order = isp._ar_word_concept_orders[:, 0]
    object_row = isp._ar_word_object_rows[:, 0]
    object_order = isp._ar_word_object_orders[:, 0]
    object_atom = isp._ar_word_object_atoms[:, 0, :]
    row_gate = isp._word_active_mask[:, :1]
    commit = isp._word_last_slot_mask[:, :1]
    lookup_rows = isp._ar_concept_lookup_rows
    lookup_atoms = isp._ar_concept_lookup_atoms
    stm_state = tuple(value.detach().clone() for value in (
        stm._buffer, stm._depth, stm._orders, stm._grammar_orders,
        stm._concept_rows, stm._concept_activations))
    feedback = (
        isp._ar_embedded_N.new_zeros(batch, int(language._n_rules)),
        torch.zeros(batch, 1, dtype=torch.bool, device=device),
    )
    predictor = cs.intraSentenceLayer

    def cs_sub_leg(ids, mask, offsets, weights, gate):
        local_parts = ps.compute_word_fold_sources(
            ids, mask, offsets, fold_passes)
        local_wholes = ws.compute_word_property_fold_sources(
            weights, fold_passes)
        return cs.reduce_aligned_percept_peers(
            local_parts, local_wholes, row, order, gate.reshape(batch),
            staged_rows=lookup_rows, staged_atoms=lookup_atoms,
            part_n_what=int(ps.nWhat), whole_n_what=int(ws.nWhat))

    def cs_sub_reduce_leg(gate):
        return cs.reduce_aligned_percept_peers(
            part_sources, whole_sources, row, order, gate.reshape(batch),
            staged_rows=lookup_rows, staged_atoms=lookup_atoms,
            part_n_what=int(ps.nWhat), whole_n_what=int(ws.nWhat))

    compiled_cs_sub = model_util.compile(
        cs_sub_leg, verbose=False, fullgraph=True)
    compiled_cs_sub_reduce = model_util.compile(
        cs_sub_reduce_leg, verbose=False, fullgraph=True)
    sub_result = tuple(
        value.detach() for value in compiled_cs_sub(
            part_ids, part_mask, part_offsets, property_weights, row_gate))
    (event, orders, local_row, activation,
     _unused_symbol, _unused_validity,
     location_activations) = sub_result

    def cs_sym_leg(source_event, source_orders, source_row,
                   source_location_activations, gate):
        symbolic_event = source_event
        symbolic_orders = source_orders
        for _ in symbolic_passes:
            reference = ss.compute_symbolic_reference(
                symbolic_event, source_row, source_location_activations,
                symbolic_orders, gate.reshape(batch),
                n_what=int(cs.nWhat))
            symbolic_event, symbolic_orders = cs.promote_symbol_reference(
                *reference, prior_event=symbolic_event,
                staged_rows=lookup_rows, staged_atoms=lookup_atoms)
        return symbolic_event, symbolic_orders

    compiled_cs_sym = model_util.compile(
        cs_sym_leg, verbose=False, fullgraph=True)
    symbolic_event, symbolic_orders = (
        value.detach() for value in compiled_cs_sym(
            event, orders, local_row, location_activations, row_gate))

    def cs_lang_leg(symbol_event, symbol_orders, source_row,
                     source_activation, gate):
        word_idea = symbol_event[:, 0, :]
        prediction, loss_sum, loss_weight = FunctionalPeerSTM.predict(
            stm_state, word_idea, commit, predictor,
            routing=feedback[0], routing_valid=feedback[1])
        full_active = torch.logical_and(
            stm_state[1] >= capacity, commit.reshape(batch))
        pre_choice = language.choose_capacity_binary(
            stm_state, full_active, base_tau=model.stm_reduce_tau)
        (pre_state, pre_applied, pre_op,
         pre_valid, _pre_loss) = cs.apply_binary_language_choice(
            stm_state, pre_choice)
        word_order = symbol_orders[:, 0].to(dtype=torch.long)
        pushed_word = ShortTermMemory.functional_push_step_masked(
            *pre_state, word_idea, commit, word_order,
            torch.zeros_like(word_order), source_row, source_activation)
        object_content = (
            object_atom[:, :int(cs.nWhat)]
            * source_activation.reshape(batch, 1))
        object_idea = torch.cat(
            (object_content, word_idea[:, int(cs.nWhat):]), dim=-1)
        object_gate = torch.logical_and(
            commit.reshape(batch), object_row >= 0)
        resolved = FunctionalPeerSTM.resolve_top_reference(
            pushed_word, object_idea, object_row, object_order,
            source_activation, object_gate)
        post_choice = language.choose_post_binary(
            resolved, commit, pre_applied, base_tau=model.stm_reduce_tau)
        (post_state, _post_applied, post_op,
         post_valid, _post_loss) = cs.apply_binary_language_choice(
            resolved, post_choice)
        unary_choice = language.choose_unary(post_state, commit)
        (final_state, _unary_applied, unary_op,
         unary_valid, _unary_loss) = cs.apply_unary_language_choice(
            post_state, unary_choice)
        grammar = language.feedback_from_local_choices(
            (pre_op, post_op), (pre_valid, post_valid),
            unary_op, unary_valid, like=word_idea)
        resolved_idea = torch.where(
            object_gate.reshape(batch, 1), object_idea, word_idea)
        contribution = torch.where(
            gate, resolved_idea, torch.zeros_like(resolved_idea))
        return (
            *final_state, contribution, prediction,
            loss_sum, loss_weight, *grammar)

    compiled_cs_lang = model_util.compile(
        cs_lang_leg, verbose=False, fullgraph=True)
    compiled_cs_lang(
        symbolic_event, symbolic_orders, local_row, activation, row_gate)
    _sync(torch, device)

    def measure(fn, args):
        for _ in range(2):
            fn(*args)
        _sync(torch, device)
        samples = []
        for _ in range(int(repeats)):
            started = time.perf_counter()
            fn(*args)
            _sync(torch, device)
            samples.append(time.perf_counter() - started)
        return {
            "median_s": statistics.median(samples),
            "mean_s": statistics.fmean(samples),
            "min_s": min(samples),
            "samples_s": samples,
        }

    ps_timing = measure(compiled_ps, (part_ids, part_mask, part_offsets))
    ws_timing = measure(compiled_ws, (property_weights,))
    reduce_timing = measure(compiled_cs_sub_reduce, (row_gate,))
    cs_sub_timing = measure(
        compiled_cs_sub,
        (part_ids, part_mask, part_offsets, property_weights, row_gate))
    cs_sym_timing = measure(
        compiled_cs_sym,
        (event, orders, local_row, location_activations, row_gate))
    cs_lang_timing = measure(
        compiled_cs_lang,
        (symbolic_event, symbolic_orders, local_row, activation, row_gate))

    perceptual_name, perceptual_timing = max(
        (("PS", ps_timing), ("WS", ws_timing)),
        key=lambda item: item[1]["median_s"])
    stages = {
        "CSSub": cs_sub_timing,
        "CSSym": cs_sym_timing,
        "CSLang": cs_lang_timing,
    }
    slowest_stage, slowest_timing = max(
        stages.items(), key=lambda item: item[1]["median_s"])
    stage_sum = sum(
        timing["median_s"] for timing in stages.values())
    bottleneck = slowest_timing["median_s"]
    return {
        "batch": batch,
        "word_index": 0,
        "repeats": int(repeats),
        "ps": ps_timing,
        "ws": ws_timing,
        "cs_sub_reduce": reduce_timing,
        "cs_sub": cs_sub_timing,
        "cs_sym": cs_sym_timing,
        "cs_lang": cs_lang_timing,
        "longest_perceptual_leg": perceptual_name,
        "longest_perceptual_s": perceptual_timing["median_s"],
        "slowest_pipeline_stage": slowest_stage,
        "slowest_pipeline_stage_s": bottleneck,
        "stage_median_sum_s": stage_sum,
        "ideal_pipeline_tick_s": bottleneck,
        "serial_to_ideal_pipeline_ratio": stage_sum / bottleneck,
        "stage_ratios_to_bottleneck": {
            name: timing["median_s"] / bottleneck
            for name, timing in stages.items()
        },
        "scope": (
            "forward word bodies: CSSub includes word-local PS, word-local "
            "WS, and CS peer reduction; CSSym includes the fixed CS/SS "
            "promotion recurrence; CSLang includes prediction, reference "
            "deposit/resolution, and Language tree planning. Pipeline/state "
            "scatters, reconstruction-trace scatters, backward, and optimizer "
            "are excluded."),
    }


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
    ap.add_argument(
        "--empty-device-cache", action="store_true",
        help=(
            "release inactive CUDA/MPS allocator blocks after every complete "
            "training step; timed rates include this cache-trim cost"))
    ap.add_argument(
        "--profile-legs", action="store_true",
        help=(
            "after throughput timing, compile and time PS/WS plus the "
            "CSSub/CSSym/CSLang word stages"))
    ap.add_argument("--profile-repeats", type=int, default=10)
    args = ap.parse_args(argv)
    if (args.width < 1 or args.batch < 1 or args.warmup < 0
            or args.seconds < 0 or args.profile_repeats < 1
            or (args.steps is not None and args.steps < 1)):
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
    # boundary; priming it with the configured W=512 static fallback would
    # execute 512 expensive no-op word ticks before compilation.
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
        # Mirror runEpoch's complete trial-cursor boundary. Every corpus item
        # here is one complete sentence, so every row has hard EOS. Besides
        # preserving real training semantics, post_tick_compact severs the
        # completed autograd graph before the next dynamic while-loop tape is
        # allocated; omitting it retains one differently-sized tape per batch.
        model.flush_word_buffers()
        model.dispatch_per_row_reset([True] * len(texts))
        model.dispatch_soft_reset()
        model.post_tick_compact()
        _sync(torch, device)
        return result, batch_max_words

    def release_step(result):
        del result
        if not args.empty_device_cache:
            return
        if device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    for index in range(args.warmup):
        result, _batch_max_words = step(-args.warmup + index)
        release_step(result)

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
        release_step(result)
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)
        measured_batch_max_words.append(batch_max_words)
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
    peer_legs = None
    if args.profile_legs:
        peer_legs = _profile_peer_legs(
            torch, model, device, [text] * args.batch,
            args.profile_repeats)
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
        "empty_device_cache": bool(args.empty_device_cache),
        "input_mode": "corpus" if args.corpus else "synthetic",
        "sentence_boundary": "runEpoch-equivalent",
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
        "peer_leg_profile": peer_legs,
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
