#!/usr/bin/env python3
"""Frozen FineWeb-Edu next-word gate for the NanoChat grammar models.

The manifest contains document-held-out multiple-choice items.  Every item
has an intact prefix, a word-order-shuffled control with the same surface
pieces, and 16 frequency/length-matched candidate words.  BasicModel scores a
candidate with the MSE between the held ``IntraSentenceLayer`` prediction and
the whole-word idea perceived at the candidate boundary.

Examples::

    .venv/bin/python bin/eval_nanochat_grammar.py generate
    BASICMODEL_DEVICE=mps MODEL_COMPILE=none \
      .venv/bin/python bin/eval_nanochat_grammar.py score --limit 10
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
import warnings


PROJECT = Path(__file__).resolve().parent.parent
BIN = PROJECT / "bin"
DEFAULT_MANIFEST = PROJECT / "data" / "eval" / "nanochat_grammar_gate.json"
DEFAULT_MODEL = PROJECT / "data" / "MM_nanochat_grammar_gate.xml"
ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
SCHEMA_VERSION = 1


def surface_words(text):
    """Return the ASCII surface words used by the cross-model gate."""
    return [m.group(0) for m in ASCII_WORD_RE.finditer(text)]


def split_name(document_index):
    """Mirror ``Data.loadShards``' document-mod-10 8/1/1 split."""
    residue = int(document_index) % 10
    if residue < 8:
        return "train"
    return "validation" if residue == 8 else "test"


def _sentences(document):
    if str(BIN) not in sys.path:
        sys.path.insert(0, str(BIN))
    from util import parse
    return [text for text, _ in parse(document, lex="sentences")
            if text.strip()]


def _shuffle_prefix(prefix, rng):
    """Shuffle words while preserving every separator/punctuation piece."""
    matches = list(ASCII_WORD_RE.finditer(prefix))
    words = [m.group(0) for m in matches]
    if len(words) < 2 or len(set(w.casefold() for w in words)) < 2:
        return None
    shuffled = list(words)
    for _ in range(8):
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    if shuffled == words:
        shuffled = words[1:] + words[:1]
    pieces = []
    cursor = 0
    for match, replacement in zip(matches, shuffled):
        pieces.append(prefix[cursor:match.start()])
        pieces.append(replacement)
        cursor = match.end()
    pieces.append(prefix[cursor:])
    return "".join(pieces)


def _frequency_bucket(count):
    return int(math.floor(math.log2(max(1, int(count)))))


def _choose_distractors(target, prefix, counts, n, max_bytes, rng):
    """Choose deterministic frequency/length-nearest held-out words."""
    prefix_vocab = {word.casefold() for word in surface_words(prefix)}
    target_key = target.casefold()
    target_count = int(counts[target])
    target_bucket = _frequency_bucket(target_count)
    candidates = []
    for word, count in counts.items():
        if word == target or word.casefold() == target_key:
            continue
        if word.casefold() in prefix_vocab:
            continue
        if len((prefix + word).encode("utf-8")) > max_bytes:
            continue
        metric = (
            abs(len(word) - len(target)),
            abs(_frequency_bucket(count) - target_bucket),
            abs(int(count) - target_count),
            rng.random(),
            word,
        )
        candidates.append((metric, word))
    candidates.sort(key=lambda pair: pair[0])
    selected = [word for _, word in candidates[:n]]
    if len(selected) != n:
        return None
    return selected


def _manifest_digest(items):
    payload = json.dumps(
        items, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest_from_documents(
        documents, *, num_items=500, seed=42, heldout_split="test",
        min_prefix_words=4, max_prefix_words=16, num_choices=16,
        max_completion_bytes=32, max_items_per_sentence=2,
        shard_names=None, max_docs=None):
    """Build a deterministic manifest from an ordered document sequence.

    Candidate words and their matching frequencies come only from the selected
    held-out split.  Targets are lowercase ASCII words, which removes the
    capitalization shortcut.  Prefixes are byte-capped so even a fresh radix
    store (one percept per byte) reaches the candidate within the 32-slot gate.
    """
    if num_choices < 2:
        raise ValueError("num_choices must be at least 2")
    if max_items_per_sentence < 1:
        raise ValueError("max_items_per_sentence must be at least 1")
    if min_prefix_words < 1 or max_prefix_words < min_prefix_words:
        raise ValueError("invalid prefix word bounds")
    docs = list(documents)
    if max_docs is not None:
        docs = docs[:int(max_docs)]
    rng = random.Random(int(seed))
    counts = Counter()
    heldout_sentences = []
    split_doc_indices = []
    for doc_index, document in enumerate(docs):
        if split_name(doc_index) != heldout_split:
            continue
        split_doc_indices.append(doc_index)
        for sentence_index, sentence in enumerate(_sentences(document)):
            heldout_sentences.append((doc_index, sentence_index, sentence))
            for word in surface_words(sentence):
                if word.isascii() and word.isalpha() and word == word.lower():
                    counts[word] += 1

    opportunities = []
    for doc_index, sentence_index, sentence in heldout_sentences:
        matches = list(ASCII_WORD_RE.finditer(sentence))
        target_positions = list(range(min_prefix_words, len(matches)))
        rng.shuffle(target_positions)
        sentence_items = 0
        for target_pos in target_positions:
            chosen = None
            target = matches[target_pos].group(0)
            if (not target.isascii() or not target.isalpha()
                    or target != target.lower() or not (2 <= len(target) <= 14)):
                continue
            upper = min(max_prefix_words, target_pos)
            prefix_lengths = list(range(min_prefix_words, upper + 1))
            # Prefer more context when it still fits the fresh-radix cap.
            prefix_lengths.sort(reverse=True)
            for prefix_len in prefix_lengths:
                start = matches[target_pos - prefix_len].start()
                end = matches[target_pos].start()
                prefix = sentence[start:end]
                if len((prefix + target).encode("utf-8")) > max_completion_bytes:
                    continue
                prefix_vocab = {w.casefold() for w in surface_words(prefix)}
                if target.casefold() in prefix_vocab:
                    continue
                shuffled = _shuffle_prefix(prefix, rng)
                if shuffled is None or shuffled == prefix:
                    continue
                chosen = {
                    "doc_index": doc_index,
                    "sentence_index": sentence_index,
                    "target_word_index": target_pos,
                    "prefix": prefix,
                    "shuffled_prefix": shuffled,
                    "prefix_word_count": prefix_len,
                    "target": target,
                }
                break
            if chosen is not None:
                opportunities.append(chosen)
                sentence_items += 1
                if sentence_items >= int(max_items_per_sentence):
                    break

    rng.shuffle(opportunities)
    items = []
    seen = set()
    for opportunity in opportunities:
        target = opportunity["target"]
        prefix = opportunity["prefix"]
        key = (prefix, target)
        if key in seen:
            continue
        distractors = _choose_distractors(
            target, prefix, counts, num_choices - 1,
            max_completion_bytes, rng)
        if distractors is None:
            continue
        candidates = distractors + [target]
        rng.shuffle(candidates)
        answer_index = candidates.index(target)
        candidate_counts = [int(counts[word]) for word in candidates]
        item = {
            "id": f"fineweb-{heldout_split}-{len(items):04d}",
            **opportunity,
            "candidates": candidates,
            "candidate_frequencies": candidate_counts,
            "answer_index": answer_index,
        }
        items.append(item)
        seen.add(key)
        if len(items) == int(num_items):
            break
    if len(items) != int(num_items):
        raise RuntimeError(
            f"Only built {len(items)} of {num_items} requested items from "
            f"{len(split_doc_indices)} {heldout_split} documents; increase "
            "max_docs or relax the prefix/byte constraints.")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "fineweb-edu-100b-shuffle",
        "shards": list(shard_names or []),
        "max_docs": len(docs),
        "heldout_split": heldout_split,
        "split_rule": "document_mod10_8_1_1",
        "seed": int(seed),
        "num_items": len(items),
        "num_choices": int(num_choices),
        "min_prefix_words": int(min_prefix_words),
        "max_prefix_words": int(max_prefix_words),
        "max_completion_bytes": int(max_completion_bytes),
        "max_items_per_sentence": int(max_items_per_sentence),
        "item_sha256": _manifest_digest(items),
    }
    return {"metadata": metadata, "items": items}


def validate_manifest(manifest):
    """Validate invariants required by both BasicModel and NanoChat scorers."""
    metadata = manifest.get("metadata")
    items = manifest.get("items")
    if not isinstance(metadata, dict) or not isinstance(items, list):
        raise ValueError("manifest must contain metadata and items")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    n_choices = int(metadata["num_choices"])
    max_bytes = int(metadata["max_completion_bytes"])
    ids = set()
    for index, item in enumerate(items):
        item_id = item.get("id")
        if not isinstance(item_id, str) or item_id in ids:
            raise ValueError(f"item {index}: id is missing or duplicated")
        ids.add(item_id)
        prefix = item.get("prefix")
        shuffled = item.get("shuffled_prefix")
        candidates = item.get("candidates")
        answer = item.get("answer_index")
        if not isinstance(prefix, str) or not isinstance(shuffled, str):
            raise ValueError(f"{item_id}: prefixes must be strings")
        if Counter(surface_words(prefix)) != Counter(surface_words(shuffled)):
            raise ValueError(f"{item_id}: shuffled prefix changed the word multiset")
        if prefix == shuffled:
            raise ValueError(f"{item_id}: shuffled prefix is unchanged")
        if not isinstance(candidates, list) or len(candidates) != n_choices:
            raise ValueError(f"{item_id}: expected {n_choices} candidates")
        if len({word.casefold() for word in candidates}) != n_choices:
            raise ValueError(f"{item_id}: candidates are not unique")
        if not isinstance(answer, int) or not 0 <= answer < n_choices:
            raise ValueError(f"{item_id}: invalid answer_index")
        if candidates[answer] != item.get("target"):
            raise ValueError(f"{item_id}: answer does not name target")
        if len(surface_words(prefix)) != int(item["prefix_word_count"]):
            raise ValueError(f"{item_id}: prefix_word_count mismatch")
        for candidate in candidates:
            if len((prefix + candidate).encode("utf-8")) > max_bytes:
                raise ValueError(f"{item_id}: completion exceeds byte cap")
    digest = metadata.get("item_sha256")
    if digest and digest != _manifest_digest(items):
        raise ValueError("manifest item_sha256 mismatch")
    return manifest


def write_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def load_manifest(path):
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def generate_manifest(args):
    if str(BIN) not in sys.path:
        sys.path.insert(0, str(BIN))
    from data import get_shard_paths, iter_documents
    shard_paths = get_shard_paths(
        str(args.shard_dir), num_shards=args.num_shards,
        random_select=False)
    if not shard_paths:
        raise RuntimeError(f"No FineWeb shards found in {args.shard_dir}")
    documents = list(iter_documents(shard_paths, max_docs=args.max_docs))
    manifest = build_manifest_from_documents(
        documents,
        num_items=args.items,
        seed=args.seed,
        heldout_split=args.split,
        min_prefix_words=args.min_prefix_words,
        max_prefix_words=args.max_prefix_words,
        num_choices=args.choices,
        max_completion_bytes=args.max_completion_bytes,
        max_items_per_sentence=args.max_items_per_sentence,
        shard_names=[Path(path).name for path in shard_paths],
        max_docs=args.max_docs,
    )
    validate_manifest(manifest)
    write_manifest(args.output, manifest)
    print(json.dumps(manifest["metadata"], indent=2))
    print(f"Wrote {len(manifest['items'])} frozen items to {args.output}")
    return 0


def last_gated_intra_scores(trace):
    """Return the last active per-row MSE, target, prediction, and count."""
    import torch
    if not trace:
        raise RuntimeError("model produced no captured intra-sentence targets")
    # The sentence prelude may emit a parallel ``[B,N,D]`` prediction before
    # the serial word loop.  It is useful training telemetry but is not a
    # candidate-word prediction; score only the word-grain ``[B,D]`` entries.
    serial_trace = [
        entry for entry in trace
        if entry["prediction"].dim() == 2 and entry["target"].dim() == 2
    ]
    if not serial_trace:
        shapes = [tuple(entry["prediction"].shape) for entry in trace]
        raise RuntimeError(
            f"model produced no serial [B,D] intra targets; saw {shapes}")
    first = serial_trace[0]["prediction"]
    batch, dim = first.shape
    scores = torch.full(
        (batch,), float("nan"), device=first.device, dtype=first.dtype)
    targets = torch.zeros(batch, dim, device=first.device, dtype=first.dtype)
    predictions = torch.zeros_like(targets)
    counts = torch.zeros(batch, device=first.device, dtype=torch.long)
    for entry in serial_trace:
        prediction = entry["prediction"]
        target = entry["target"]
        if prediction.shape != first.shape or target.shape != first.shape:
            raise RuntimeError("intra capture changed shape within one forward")
        gate = entry.get("row_gate")
        if gate is None:
            active = torch.ones(batch, device=first.device, dtype=torch.bool)
        else:
            active = gate.reshape(batch).to(
                device=first.device, dtype=torch.bool)
        mse = (prediction - target).square().mean(dim=-1)
        scores = torch.where(active, mse, scores)
        targets = torch.where(active.unsqueeze(-1), target, targets)
        predictions = torch.where(
            active.unsqueeze(-1), prediction, predictions)
        counts = counts + active.long()
    if bool(torch.isnan(scores).any().item()):
        missing = torch.isnan(scores).nonzero().reshape(-1).tolist()
        raise RuntimeError(f"no active next-word target for rows {missing}")
    return (scores.detach().cpu(), targets.detach().cpu(),
            predictions.detach().cpu(), counts.detach().cpu())


def _reset_model(model):
    for space in model.spaces:
        reset = getattr(space, "Reset", None)
        if reset is not None:
            reset(hard=True)


@contextmanager
def frozen_online_learning(model):
    """Disable vocabulary/taxonomy growth while held-out choices are read."""
    owners = [getattr(model, "perceptualSpace", None)]
    owners.extend(list(getattr(model, "conceptualSpaces", []) or []))
    saved = []
    for owner in owners:
        if owner is None:
            continue
        saved.append((owner, getattr(owner, "_online_learning_frozen", False)))
        object.__setattr__(owner, "_online_learning_frozen", True)
    try:
        yield
    finally:
        ws = getattr(model, "wholeSpace", None)
        if ws is not None and hasattr(ws, "_category_role_obs"):
            ws._category_role_obs = None
        for owner, previous in saved:
            object.__setattr__(owner, "_online_learning_frozen", previous)


def _seed_from_xml(path):
    root = ET.parse(path).getroot()
    raw = root.findtext("./architecture/training/seed")
    return int(raw) if raw not in (None, "") else 0


def _autoload_from_xml(path):
    root = ET.parse(path).getroot()
    raw = root.findtext("./architecture/training/autoload")
    return str(raw or "").strip().casefold() in ("1", "true", "yes", "on")


def _prewarm_checkpoint_shapes(model, target):
    """Materialize every lazy module whose shape is saved by training.

    Training's ``enable_compiled_step`` does two shape-affecting prewarms
    before optimizer construction: it builds the STM reducer and grows the
    terminal WholeSpace symbol codebook from its well-known-atom seed to the
    configured inventory. Evaluation stays eager, but strict checkpoint
    loading still needs the same destination shapes.
    """
    reducer_factory = getattr(model, "_stm_reducer", None)
    if callable(reducer_factory):
        reducer_factory()

    ws = getattr(model, "wholeSpace", None)
    sub = getattr(ws, "subspace", None)
    codebook = (sub.codebook()
                if sub is not None and hasattr(sub, "codebook") else None)
    budget = int(getattr(ws, "nVectors", 0) or 0)
    if (codebook is not None and budget > 0
            and int(getattr(codebook, "nVectors", 0) or 0) < budget):
        codebook.grow_to(budget)

    model.to(target)


def build_eval_model(model_path, checkpoint=None, *, autoload=True):
    """Construct, prewarm, and optionally load an evaluation model.

    Autoload is performed explicitly *after* the lazy STM reducer has been
    registered.  Training checkpoints contain that reducer's learned weights;
    letting ``BaseModel`` autoload during construction would otherwise see
    those keys before the child module exists and silently treat them as stale.
    """
    import numpy as np
    import torch
    if str(BIN) not in sys.path:
        sys.path.insert(0, str(BIN))
    import Language
    from data import TheData
    from Models import BaseModel
    from util import TheDevice, init_device

    model_path = Path(model_path).resolve()
    seed = _seed_from_xml(model_path)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    TheData.processLM({
        "train": {"text": ["placeholder"], "label": []},
        "validation": {"text": ["placeholder"], "label": []},
        "test": {"text": ["placeholder"], "label": []},
    })
    Language.TheGrammar._configured = False
    target = TheDevice.get()
    previous_autoload = os.environ.get("BASIC_AUTOLOAD")
    os.environ["BASIC_AUTOLOAD"] = "0"
    try:
        if target.type == "mps":
            init_device("cpu")
            try:
                model, _ = BaseModel.from_config(str(model_path), data=TheData)
            finally:
                init_device(target)
            model.to(target)
        else:
            model, _ = BaseModel.from_config(str(model_path), data=TheData)
    finally:
        if previous_autoload is None:
            os.environ.pop("BASIC_AUTOLOAD", None)
        else:
            os.environ["BASIC_AUTOLOAD"] = previous_autoload

    # Mirror training's shape-affecting prewarm even though scoring stays
    # eager. Strict loading requires destinations for both the lazy reducer
    # and the configured WholeSpace inventory.
    _prewarm_checkpoint_shapes(model, target)

    load_path = Path(checkpoint).resolve() if checkpoint is not None else None
    explicit_checkpoint = load_path is not None
    if load_path is None and autoload and _autoload_from_xml(model_path):
        candidate = Path(model._checkpoint_path()).resolve()
        if candidate.exists():
            load_path = candidate
    loaded_path = None
    if load_path is not None:
        loaded = model.load_weights(str(load_path), require_match=True)
        if not loaded and explicit_checkpoint:
            raise FileNotFoundError(checkpoint)
        if loaded:
            loaded_path = str(load_path)
    object.__setattr__(model, "_eval_checkpoint_path", loaded_path)
    if not bool(getattr(model, "serial_object_meta", False)):
        raise ValueError(
            "next-word evaluation requires <serialObjectMeta>true</...> so "
            "the candidate is scored before any of its radix spelling leaks")
    model.eval()
    model.set_sigma(0)
    return model, TheData


def _validate_candidate_reached(model, items, control, choices):
    """Fail if the outer word cap or raw-constituent staging cuts a candidate."""
    import torch
    word_index = getattr(model.inputSpace, "_word_index_N", None)
    active = getattr(model.inputSpace, "_word_active_mask", None)
    if word_index is None or active is None:
        raise RuntimeError("serialObjectMeta word-index tensors were not built")
    expected = []
    for item in items:
        prefix = item["prefix" if control == "intact" else "shuffled_prefix"]
        final_word_id = len(surface_words(prefix))
        expected.extend([final_word_id] * choices)
    expected_t = torch.tensor(
        expected, device=word_index.device, dtype=word_index.dtype)
    reached = ((word_index == expected_t.unsqueeze(1)) & active).any(dim=1)
    if not bool(reached.all().item()):
        missing = (~reached).nonzero().reshape(-1).tolist()
        raise RuntimeError(
            "candidate word was truncated before its commit boundary in "
            f"flattened rows {missing}; regenerate with a smaller byte cap")
    part_truncated = getattr(
        model.inputSpace, "_ar_word_truncated_mask", None)
    if (torch.is_tensor(part_truncated)
            and tuple(part_truncated.shape) == tuple(word_index.shape)):
        cut = ((word_index == expected_t.unsqueeze(1))
               & part_truncated).any(dim=1)
        if bool(cut.any().item()):
            rows = cut.nonzero().reshape(-1).tolist()
            raise RuntimeError(
                "candidate word's raw constituents were truncated before "
                f"in-loop synthesis in flattened rows {rows}")


def trace_serial_grammar(model, text):
    """Run one sentence and audit the occupancy-driven STM grammar.

    A reduction counts only when ``stm._depth`` decreases.  Online folds are
    separated into low/medium-occupancy soft-pressure decisions and hard
    capacity demands; sentence-boundary closure and any legacy/unlicensed
    fallback are reported separately.  Device tensors are accumulated during
    the forward and transferred in a few bulk slabs afterward, avoiding the
    per-word ``.item()`` synchronization that made an MPS trace misleadingly
    slow.
    """
    import torch

    _reset_model(model)
    stm = model.conceptualSpace.stm
    in_sweep = False
    depth_samples = []
    events = []
    original_reduce = model._stm_bounded_reduce_step
    original_sweep = model._stm_reduce_to_single_S
    original_push = stm.push_step_masked

    def sample_depth():
        depth_samples.append(stm._depth.detach().clone())
        return len(depth_samples) - 1

    def reduce_spy(*args, **kwargs):
        nonlocal in_sweep
        before_idx = sample_depth()
        reduced_mask = original_reduce(*args, **kwargs)
        after_idx = sample_depth()
        routing = getattr(model, "_stm_last_reduce_routing", None)
        B = int(stm._depth.shape[0])
        device = stm._depth.device

        confidence = threshold = pressure = None
        routed_demand = None
        chosen = torch.full(
            (B,), -1, dtype=torch.long, device=device)
        if isinstance(routing, dict):
            confidence = routing.get("grammar_reduce_confidence")
            threshold = routing.get("stm_reduce_threshold")
            pressure = routing.get("stm_occupancy_pressure")
            routed_demand = routing.get("stm_demand_mask")
            rmo = routing.get("reduce_marginal_op")
            if torch.is_tensor(rmo) and rmo.dim() == 3 and rmo.shape[-1]:
                chosen = rmo[:, 0, :].argmax(dim=-1).detach().clone()

        row_gate = kwargs.get("row_gate")
        if torch.is_tensor(row_gate):
            row_gate = row_gate.detach().to(
                device=device, dtype=torch.bool).reshape(B).clone()
        else:
            row_gate = torch.ones(B, dtype=torch.bool, device=device)
        explicit_demand = bool(kwargs.get("demand", False))
        if explicit_demand:
            demand_rows = row_gate
        elif torch.is_tensor(routed_demand):
            demand_rows = routed_demand.detach().to(
                device=device, dtype=torch.bool).reshape(B).clone()
        else:
            demand_rows = torch.zeros(B, dtype=torch.bool, device=device)

        def optional_sample(value, dtype=torch.float32):
            if torch.is_tensor(value) and value.numel() == B:
                return value.detach().to(device=device, dtype=dtype).reshape(B).clone()
            fill = float("nan") if dtype.is_floating_point else -1
            return torch.full((B,), fill, dtype=dtype, device=device)

        events.append({
            "kind": ("boundary" if in_sweep else
                     "pressure" if kwargs.get("occupancy_pressure", False) else
                     "demand" if explicit_demand else
                     "scored" if kwargs.get("gate_tau") is not None else
                     "unlicensed"),
            "before": before_idx,
            "after": after_idx,
            "confidence": optional_sample(confidence),
            "threshold": optional_sample(threshold),
            "pressure": optional_sample(pressure),
            "demand_rows": demand_rows,
            "chosen": chosen,
        })
        return reduced_mask

    def sweep_spy(*args, **kwargs):
        nonlocal in_sweep
        in_sweep = True
        try:
            return original_sweep(*args, **kwargs)
        finally:
            in_sweep = False

    def push_spy(*args, **kwargs):
        result = original_push(*args, **kwargs)
        events.append({
            "kind": "push",
            "after": sample_depth(),
        })
        return result

    model._stm_bounded_reduce_step = reduce_spy
    model._stm_reduce_to_single_S = sweep_spy
    stm.push_step_masked = push_spy
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepared = model.inputSpace.prepInput([str(text)])
            model.forward(prepared)
    finally:
        model._stm_bounded_reduce_step = original_reduce
        model._stm_reduce_to_single_S = original_sweep
        stm.push_step_masked = original_push

    # Bulk device-to-host collection: one depth slab plus one slab per
    # reducer field, rather than synchronizing Metal inside every word tick.
    depth_rows = (torch.stack(depth_samples).detach().cpu()
                  if depth_samples else torch.zeros(0, 0, dtype=torch.long))
    reduce_events = [event for event in events if event["kind"] != "push"]
    for key in ("confidence", "threshold", "pressure", "demand_rows", "chosen"):
        if reduce_events:
            slab = torch.stack([event[key] for event in reduce_events]).detach().cpu()
            for index, event in enumerate(reduce_events):
                event[key + "_cpu"] = slab[index]

    stats = {
        "pressure_calls": 0,
        "soft_pressure_reductions": 0,
        "capacity_demand_calls": 0,
        "capacity_demand_reductions": 0,
        "boundary_reductions": 0,
        "unlicensed_calls": 0,
        "unlicensed_reductions": 0,
        "legacy_scored_calls": 0,
        "legacy_scored_reductions": 0,
        "peak_depth": 0,
    }
    timeline = []
    online_operators = Counter()
    boundary_operators = Counter()
    reducer = model._stm_reducer()
    op_names = list(getattr(reducer, "op_names", None) or [])
    pressure_confidences = []
    pressure_thresholds = []
    for event in events:
        if event["kind"] == "push":
            depth = int(depth_rows[event["after"]].max().item())
            stats["peak_depth"] = max(stats["peak_depth"], depth)
            timeline.append({
                "word": len(timeline) + 1,
                "after_push": depth,
                "after_pressure": depth,
                "after_licensed": depth,
            })
            continue

        before = depth_rows[event["before"]]
        after = depth_rows[event["after"]]
        actually = after < before
        count = int(actually.sum().item())
        demand_rows = event["demand_rows_cpu"].bool()
        demanded = actually & demand_rows
        soft = actually & ~demand_rows
        kind = event["kind"]

        if kind == "pressure":
            stats["pressure_calls"] += 1
            stats["soft_pressure_reductions"] += int(soft.sum().item())
            stats["capacity_demand_reductions"] += int(demanded.sum().item())
            if bool(demand_rows.any().item()):
                stats["capacity_demand_calls"] += 1
            finite_c = event["confidence_cpu"][
                torch.isfinite(event["confidence_cpu"])]
            finite_t = event["threshold_cpu"][
                torch.isfinite(event["threshold_cpu"])]
            pressure_confidences.extend(float(v) for v in finite_c.tolist())
            pressure_thresholds.extend(float(v) for v in finite_t.tolist())
            if timeline:
                post = int(after.max().item())
                timeline[-1]["after_pressure"] = post
                timeline[-1]["after_licensed"] = post
                timeline[-1]["reduced"] = bool(actually[0].item())
                timeline[-1]["demanded"] = bool(demand_rows[0].item())
                if finite_c.numel():
                    timeline[-1]["grammar_confidence"] = float(
                        event["confidence_cpu"][0].item())
                    timeline[-1]["reduce_threshold"] = float(
                        event["threshold_cpu"][0].item())
                    timeline[-1]["occupancy_pressure"] = float(
                        event["pressure_cpu"][0].item())
        elif kind == "demand":
            stats["capacity_demand_calls"] += 1
            stats["capacity_demand_reductions"] += count
        elif kind == "boundary":
            stats["boundary_reductions"] += count
        elif kind == "unlicensed":
            stats["unlicensed_calls"] += 1
            stats["unlicensed_reductions"] += count
        else:  # legacy scored controller
            stats["legacy_scored_calls"] += 1
            stats["legacy_scored_reductions"] += count

        chosen = event["chosen_cpu"]
        target_counter = (boundary_operators
                          if kind == "boundary" else online_operators)
        for row in actually.nonzero().reshape(-1).tolist():
            op_idx = int(chosen[row].item())
            name = (op_names[op_idx]
                    if 0 <= op_idx < len(op_names) else f"op_{op_idx}")
            target_counter[name] += 1
    total_reductions = (
        stats["soft_pressure_reductions"]
        + stats["capacity_demand_reductions"]
        + stats["boundary_reductions"]
        + stats["unlicensed_reductions"]
        + stats["legacy_scored_reductions"])

    part_ids = getattr(model.inputSpace, "_ar_word_part_ids", None)
    part_mask = getattr(model.inputSpace, "_ar_word_part_mask", None)
    word_cut = getattr(model.inputSpace, "_ar_word_truncated_mask", None)
    sentence_cut = getattr(
        model.inputSpace, "_sentence_word_truncated_mask", None)
    active = getattr(model.inputSpace, "_word_active_mask", None)
    result = {
        "surface_words": (int(active.sum().item())
                          if torch.is_tensor(active)
                          else len(surface_words(str(text)))),
        "outer_word_capacity": int(getattr(
            model, "serial_word_capacity", 0) or 0),
        "part_field_width": int(model.perceptualSpace.outputShape[0]),
        "whole_field_width": int(model.wholeSpace.inputShape[0]),
        "concept_field_width": int(model.conceptualSpace.outputShape[0]),
        "stm_capacity": int(stm.capacity),
        "raw_part_width": (int(part_ids.shape[-1])
                           if torch.is_tensor(part_ids) else 0),
        "max_raw_parts_per_word": (int(part_mask.sum(dim=-1).max().item())
                                   if torch.is_tensor(part_mask) else 0),
        "word_constituents_truncated": (
            bool(word_cut.any().item()) if torch.is_tensor(word_cut) else False),
        "sentence_truncated": (
            bool(sentence_cut.any().item())
            if torch.is_tensor(sentence_cut) else False),
        "final_depth": int(stm._depth.detach().max().cpu().item()),
        "total_reductions": total_reductions,
        "operators": dict(sorted(online_operators.items())),
        "boundary_operators": dict(sorted(boundary_operators.items())),
        "grammar_confidence_range": (
            [min(pressure_confidences), max(pressure_confidences)]
            if pressure_confidences else []),
        "reduce_threshold_range": (
            [min(pressure_thresholds), max(pressure_thresholds)]
            if pressure_thresholds else []),
        "timeline": timeline,
        **stats,
    }
    return result


def _score_control_batch(model, data, items, control, choices):
    import torch
    texts = []
    prefix_key = "prefix" if control == "intact" else "shuffled_prefix"
    for item in items:
        prefix = item[prefix_key]
        texts.extend(prefix + candidate for candidate in item["candidates"])
    _reset_model(model)
    cs = model.conceptualSpace
    with torch.no_grad(), data.runtime_batch(texts), \
            cs.capture_intra_predictions() as trace:
        input_tensor = model.inputSpace.prepInput(list(data.train_input))
        model.forward(input_tensor)
    _validate_candidate_reached(model, items, control, choices)
    result = last_gated_intra_scores(trace)
    _reset_model(model)
    return result


def _rank(scores, answer_index):
    ordered = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return ordered.index(int(answer_index)) + 1


def _latent_statistics(targets):
    import torch
    if not targets:
        return {"target_variance": 0.0, "target_effective_rank": 0.0}
    matrix = torch.stack(targets).float()
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    variance = float(centered.square().mean().item())
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum()
    if not bool((total > 0).item()):
        effective_rank = 0.0
    else:
        probs = energy / total
        entropy = -(probs * probs.clamp_min(1e-30).log()).sum()
        effective_rank = float(entropy.exp().item())
    return {
        "target_variance": variance,
        "target_effective_rank": effective_rank,
    }


def score_manifest(model, data, manifest, *, limit=None, item_batch_size=1):
    """Score intact and shuffled controls, returning aggregate + item rows."""
    import torch
    items = list(manifest["items"])
    if limit is not None:
        items = items[:max(0, int(limit))]
    if not items:
        raise ValueError("no manifest items selected")
    choices = int(manifest["metadata"]["num_choices"])
    if item_batch_size < 1:
        raise ValueError("item_batch_size must be at least 1")
    rows = []
    target_vectors = []
    started = time.perf_counter()
    with frozen_online_learning(model):
        for start in range(0, len(items), int(item_batch_size)):
            batch_items = items[start:start + int(item_batch_size)]
            intact = _score_control_batch(
                model, data, batch_items, "intact", choices)
            shuffled = _score_control_batch(
                model, data, batch_items, "shuffled", choices)
            intact_scores, intact_targets, _, intact_counts = intact
            shuffled_scores, _, _, shuffled_counts = shuffled
            for local, item in enumerate(batch_items):
                lo = local * choices
                hi = lo + choices
                scores_i = [float(x) for x in intact_scores[lo:hi]]
                scores_s = [float(x) for x in shuffled_scores[lo:hi]]
                if not all(math.isfinite(x) for x in scores_i + scores_s):
                    raise RuntimeError(f"{item['id']}: non-finite candidate score")
                answer = int(item["answer_index"])
                rank_i = _rank(scores_i, answer)
                rank_s = _rank(scores_s, answer)
                target_vectors.append(intact_targets[lo + answer])
                rows.append({
                    "id": item["id"],
                    "answer_index": answer,
                    "intact_scores": scores_i,
                    "shuffled_scores": scores_s,
                    "intact_rank": rank_i,
                    "shuffled_rank": rank_s,
                    "intact_true_mse": scores_i[answer],
                    "intact_prediction_steps": int(
                        intact_counts[lo + answer]),
                    "shuffled_prediction_steps": int(
                        shuffled_counts[lo + answer]),
                })
            done = min(len(items), start + len(batch_items))
            print(f"Scored {done}/{len(items)} items", flush=True)
    n = len(rows)
    metrics = {
        "items": n,
        "choices": choices,
        "chance_top1": 1.0 / choices,
        "intact_top1": sum(row["intact_rank"] == 1 for row in rows) / n,
        "intact_mrr": sum(1.0 / row["intact_rank"] for row in rows) / n,
        "shuffled_top1": sum(row["shuffled_rank"] == 1 for row in rows) / n,
        "shuffled_mrr": sum(1.0 / row["shuffled_rank"] for row in rows) / n,
        "intact_true_mse": sum(row["intact_true_mse"] for row in rows) / n,
        "wall_seconds": time.perf_counter() - started,
        **_latent_statistics(target_vectors),
    }
    metrics["intact_minus_shuffled_top1"] = (
        metrics["intact_top1"] - metrics["shuffled_top1"])
    return {"metrics": metrics, "items": rows}


def run_score(args):
    if args.fresh and args.checkpoint is not None:
        raise ValueError("--fresh and --checkpoint are mutually exclusive")
    manifest = load_manifest(args.manifest)
    model, data = build_eval_model(
        args.model, checkpoint=args.checkpoint, autoload=not args.fresh)
    result = score_manifest(
        model, data, manifest, limit=args.limit,
        item_batch_size=args.item_batch_size)
    result["manifest"] = {
        "path": str(Path(args.manifest).resolve()),
        "item_sha256": manifest["metadata"]["item_sha256"],
    }
    result["model"] = {
        "path": str(Path(args.model).resolve()),
        "checkpoint": getattr(model, "_eval_checkpoint_path", None),
        "fresh_initialization": (
            getattr(model, "_eval_checkpoint_path", None) is None),
        "fresh_requested": bool(args.fresh),
        "device": str(next(model.parameters()).device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    if (args.require_top1 is not None
            and result["metrics"]["intact_top1"] < args.require_top1):
        return 1
    return 0


def run_trace(args):
    model, _ = build_eval_model(
        args.model, checkpoint=args.checkpoint, autoload=False)
    if args.text is not None:
        text = args.text
    else:
        text = " ".join(f"w{i}" for i in range(int(args.words)))
    result = trace_serial_grammar(model, text)
    if not getattr(args, "timeline", False):
        result.pop("timeline", None)
    print(json.dumps(result, indent=2))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="create the frozen manifest")
    gen.add_argument("--shard-dir", type=Path,
                     default=PROJECT / "data" / "fineweb")
    gen.add_argument("--num-shards", type=int, default=1)
    gen.add_argument("--max-docs", type=int, default=200)
    gen.add_argument("--split", choices=("validation", "test"), default="test")
    gen.add_argument("--items", type=int, default=500)
    gen.add_argument("--choices", type=int, default=16)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--min-prefix-words", type=int, default=4)
    gen.add_argument("--max-prefix-words", type=int, default=16)
    gen.add_argument("--max-completion-bytes", type=int, default=32)
    gen.add_argument("--max-items-per-sentence", type=int, default=2)
    gen.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)

    score = sub.add_parser("score", help="score a BasicModel checkpoint")
    score.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    score.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    score.add_argument("--checkpoint", type=Path)
    score.add_argument("--fresh", action="store_true",
                       help="ignore the XML autoload checkpoint")
    score.add_argument("--device", choices=("cpu", "gpu", "mps", "cuda"),
                       default="gpu")
    score.add_argument("--limit", type=int)
    score.add_argument("--item-batch-size", type=int, default=1,
                       help="items per forward pair; device rows = value * 16")
    score.add_argument("--output", type=Path)
    score.add_argument("--require-top1", type=float)

    trace = sub.add_parser(
        "trace", help="trace licensed vs emergency STM reductions")
    trace.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    trace.add_argument("--checkpoint", type=Path)
    trace.add_argument("--device", choices=("cpu", "gpu", "mps", "cuda"),
                       default="cpu")
    trace.add_argument("--words", type=int, default=64)
    trace.add_argument("--text")
    trace.add_argument(
        "--timeline", action="store_true",
        help="include the per-word depth/confidence timeline")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command in ("score", "trace"):
        # util.py resolves the process device at import time.  Set these before
        # the lazy Models/data imports in ``build_eval_model``.
        os.environ["BASICMODEL_DEVICE"] = args.device
        os.environ.setdefault("MODEL_COMPILE", "none")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        return run_score(args) if args.command == "score" else run_trace(args)
    os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
    return generate_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
