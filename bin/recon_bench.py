"""Shared fidelity+timing harness (2026-07-03 recon plan): every debug run
is a benchmark sample. CLI: recon_bench.py <config.xml> [--epochs N]
[--seed S] [--out output/] [--profile] [--compiled-step]."""
import argparse
import dataclasses
import json
import math
import os
import random
import socket
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import numpy as np
import torch

# Memory guard: cap the single eval batch used for the decode pass.
_MAX_EVAL_BATCH = 512

# JSON notes key for decode-path degradation (consumed by later plan tasks).
DECODE_NOTE = "decode_note"


@dataclasses.dataclass
class RunRecord:
    config: str
    seed: int
    epochs: int
    wall_s_per_epoch: float
    output_loss: float
    recon_loss: float
    exact_match_rate: float
    where_recovery: float
    channel_losses: dict
    device: str
    compile_mode: str
    host: str
    timestamp: str
    notes: dict = dataclasses.field(default_factory=dict)

    @property
    def filename(self):
        base = os.path.splitext(os.path.basename(self.config))[0]
        return f"recon_{base}_{self.timestamp}_{os.getpid()}.json"


def exact_match_rate(targets, decoded):
    """Fraction of rows whose decoded reconstruction string is IDENTICAL."""
    if not targets:
        return 0.0
    hits = sum(1 for t, d in zip(targets, decoded) if t == d)
    return hits / len(targets)


def _norm(s):
    """NUL/whitespace-normalize a rendered string for exact comparison."""
    return " ".join(s.replace("\x00", " ").split())


def where_recovery_rate(raw_targets, meta):
    """Fraction of decoded percepts whose recovered span (start, len) exactly
    matches the true word-tile span (ordinal-aligned; extra/missing percepts
    count against)."""
    import Meronomy
    tokens = (meta or {}).get("tokens") or []
    hits = total = 0
    for b, raw in enumerate(raw_targets):
        tiles = Meronomy.word_tiling(str(raw).rstrip("\x00").encode("utf-8"))
        toks = tokens[b] if b < len(tokens) else []
        rec = [(off, off + len(w.encode("utf-8"))) for (w, off) in toks
               if off is not None and w not in ("", "\x00")]
        total += max(len(tiles), len(rec))
        hits += sum(1 for t, r in zip(tiles, rec) if t == r)
    if total == 0:
        return -1.0
    return hits / total


def _finite(name, value):
    """Fail loud on Inf/NaN losses (house rule: never mask divergence)."""
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {name}: {value}")
    return value


def _resolve_config(config):
    """Resolve a config path against cwd first, then the project root."""
    if os.path.exists(config):
        return os.path.abspath(config)
    candidate = os.path.join(_PROJECT, config)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"config not found: {config}")


def _build_model(config_path):
    """Build model + data from a config path the way bench_throughput does."""
    import Language
    from data import TheData
    from Models import BaseModel
    from util import init_config, TheXMLConfig, TheDevice

    init_config(path=config_path,
                defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
    # Allow rebuilds in a warm process (mirrors the fixture-model tests).
    Language.TheGrammar._configured = False
    arch = TheXMLConfig.data.get("architecture", {})
    dat = arch.get("data", {})
    trn = arch.get("training", {})
    dataset = os.environ.get("BASIC_DATASET", dat.get("dataset"))
    TheData.load(
        dataset,
        num_shards=int(os.environ.get("BASIC_NUM_SHARDS",
                                      dat.get("numShards", 1))),
        max_docs=int(os.environ.get("BASIC_MAX_DOCS",
                                    dat.get("maxDocs", 10000))),
        shard_dir=dat.get("shardDir"),
        dat=dat)
    model, _ = BaseModel.from_config(config_path, data=TheData)
    dev = TheDevice.get()
    model = model.to(dev)
    lr = float(trn.get("learningRate", 1e-4))
    batch_size = int(trn.get("batchSize", 10))
    return model, dev, lr, batch_size


def _raw_targets(model):
    """The last staged eval batch's raw input strings (pre-normalization)."""
    sents = getattr(model.inputSpace, "_last_sentences", None)
    if sents is None:
        # Byte-cursor configs bypass prepInput; use the _lex_batch stash.
        slab = getattr(model.inputSpace, "_last_host_slab", None)
        if slab is None:
            raise RuntimeError(
                "no staged eval-batch inputs to align targets against")
        sents = [model._bytes_to_text(slab[i])
                 for i in range(int(slab.shape[0]))]
    return [str(s) for s in sents]


def _decode_texts(model):
    """Render the staged reverse() batch vs that SAME batch's raw inputs."""
    # Route (b): only the LAST staged eval batch is renderable, so targets come from that batch's own stashed inputs (1:1 row-aligned).
    decoded = model.perceptualSpace.reconstruct_data(text=True)
    targets = [_norm(s) for s in _raw_targets(model)]
    decoded = [_norm(s) for s in decoded]
    # Alignment invariant: a row-count mismatch must never silently truncate.
    if len(decoded) != len(targets):
        raise RuntimeError(f"decode/target row mismatch: {len(decoded)} "
                           f"decoded vs {len(targets)} targets")
    return targets, decoded


def run_config(config, epochs, seed, out_dir, profile=False,
               max_batches=None, compiled_step=False):
    """Run `epochs` timed training epochs on `config`; write a JSON record.

    The `seed` argument deliberately overrides BASIC_SEED / the XML
    <training><seed>: harness records must be reproducible standalone.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    config_path = _resolve_config(config)
    model, dev, lr, batch_size = _build_model(config_path)
    if compiled_step:
        # Production parity (Task 8): ModelFactory.run's torch.compile enlistment; default-off keeps the deterministic eager test path.
        model.enable_compiled_step()
    optimizer = model.getOptimizer(lr=lr)

    epoch_times = []
    last_losses = {}

    def _epoch_loop():
        for _ in range(int(epochs)):
            t0 = time.perf_counter()
            out_loss, rec_loss, _, _ = model.runEpoch(
                optimizer=optimizer, batchSize=batch_size, split="train",
                max_batches=max_batches)
            epoch_times.append(time.perf_counter() - t0)
            last_losses["output"] = _finite("output_loss", float(out_loss))
            last_losses["recon"] = _finite("recon_loss", float(rec_loss))

    prof = None
    if profile:
        from torch.profiler import profile as torch_profile, ProfilerActivity
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with torch_profile(activities=activities) as prof:
            _epoch_loop()
    else:
        _epoch_loop()

    # Decode pass: one bounded eval epoch stages the reverse() state.
    test_input, _ = model.inputSpace.getTestData()
    n = (int(test_input.shape[0]) if torch.is_tensor(test_input)
         else len(test_input))
    model.set_sigma(0)
    model.train(False)
    with torch.no_grad():
        model.runEpoch(batchSize=max(1, min(n, _MAX_EVAL_BATCH)),
                       split="test", max_batches=max_batches)

    notes = {}
    where_recovery = -1.0
    try:
        targets, decoded = _decode_texts(model)
        if len(decoded) < n:
            notes["exact_match_note"] = (
                f"decode covers the last eval batch only ({len(decoded)} "
                f"of {n} rows); targets are that batch's own inputs")
        # Real span metric (Task 5.5): decoded spans vs true word-tile spans.
        meta = model.perceptualSpace._materialize_recovered_input()
        where_recovery = where_recovery_rate(_raw_targets(model), meta)
    except Exception as e:
        # Degrade to the timing-only record; the note keeps the failure loud.
        targets, decoded = [], []
        notes[DECODE_NOTE] = (f"text decode unavailable: "
                              f"{type(e).__name__}: {e}")
    if where_recovery < 0.0:
        notes["where_recovery_note"] = "span metric unavailable (no decode)"
    notes["decoded_rows"] = len(decoded)

    # Record the step routing + torch.compile mode so records self-describe.
    notes["compiled_step"] = bool(compiled_step)
    import util as _u
    notes["torch_compile_mode"] = str(_u.TheCompileMode)

    # channel_losses stays {}: ModelLoss sums what/where/when inline (bin/Layers.py).
    channel_losses = {}
    notes["channel_losses_note"] = (
        "per-channel what/where/when terms are summed inline in "
        "ModelLoss and not exposed on the model")

    # Steady-state timing: drop the warm-up epoch when there is more than one.
    notes["epoch_times_s"] = [float(t) for t in epoch_times]
    if len(epoch_times) > 1:
        wall = sum(epoch_times[1:]) / (len(epoch_times) - 1)
    else:
        wall = epoch_times[0] if epoch_times else 0.0
        notes["wall_note"] = "single epoch includes warm-up"

    import util as _util
    rec = RunRecord(
        config=config,
        seed=int(seed),
        epochs=int(epochs),
        wall_s_per_epoch=float(wall),
        output_loss=last_losses.get("output", 0.0),
        recon_loss=last_losses.get("recon", 0.0),
        exact_match_rate=float(exact_match_rate(targets, decoded)),
        where_recovery=float(where_recovery),
        channel_losses=channel_losses,
        device=str(dev),
        compile_mode=str(_util.TheCompileBackend),
        host=socket.gethostname(),
        timestamp=timestamp,
        notes=notes,
    )

    payload = dataclasses.asdict(rec)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, rec.filename)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    if prof is not None:
        table = prof.key_averages().table(sort_by="self_cpu_time_total",
                                          row_limit=25)
        with open(out_path + ".profile.txt", "w") as f:
            f.write(table)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="model XML config path")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap runEpoch batches per epoch")
    ap.add_argument("--compiled-step", action="store_true",
                    help="route runBatch through the torch.compiled step "
                         "(production parity; MODEL_COMPILE picks backend)")
    args = ap.parse_args(argv)
    rec = run_config(args.config, epochs=args.epochs, seed=args.seed,
                     out_dir=args.out, profile=args.profile,
                     max_batches=args.max_batches,
                     compiled_step=args.compiled_step)
    print(json.dumps(dataclasses.asdict(rec), indent=2))
    print(f"[recon_bench] wrote {os.path.join(args.out, rec.filename)}")


if __name__ == "__main__":
    main()
