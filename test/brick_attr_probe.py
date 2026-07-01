"""Host-sync attribution probe -- the *truth signal* for brick sync work.

The raw ``torch.profiler`` "Memcpy DtoH" device-event count is noisy /
non-monotone (see doc/BrickHostSyncStatus.md "Reliable metric caveat").
This probe is the signal of truth: it monkeypatches the tensor methods
that force a GPU->host round-trip
(``item/tolist/nonzero/__bool__/__int__/__float__/__index__``), counts
every call made *inside ``runEpoch``* by ``file:line`` callsite via
``traceback.extract_stack``, and prints a histogram.

Device-independent: the Python callsite that issues ``.item()`` is the
same on CPU/MPS/CUDA, so the attribution is identical on any device
(only the CUDA *device-event* count -- the noisy metric -- needs a GPU).
Run on metalbaby (CUDA) for the authoritative read; it also runs on
CPU/MPS as a harness self-check.

Usage:
    .venv/bin/python test/brick_attr_probe.py [CONFIG.xml] [batchSize]
Defaults: MM_xor.xml, batchSize=2. MM_20M_legacy.xml uses the FineWeb shard
corpus (skips cleanly if absent), mirroring test_brick_no_sync.py.

NOT a pytest test (no ``test_`` prefix -> not collected): it needs an
explicit run and, for the authoritative read, a CUDA host.
"""
import os
import sys
import threading
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("MODEL_DEBUG", "0")  # doc recipe: MODEL_DEBUG=0

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch

# Tensor methods that force a GPU->host transfer. The doc's core set
# plus the *non-method-call* D2H paths (`.cpu()`, `.numpy()`) the
# original attribution recipe missed -- attribution can read 0 while
# the profiler still sees `Memcpy DtoH` from these (the doc's "device
# count is the final gate" tail). `.to()` is handled separately
# (conditional: only a CUDA->host `.to` is a sync).
_SYNC_METHODS = (
    "item", "tolist", "nonzero",
    "__bool__", "__int__", "__float__", "__index__",
    "cpu", "numpy",
)

_state = threading.local()
_counts = Counter()       # (method, file, line, func) -> count
_enabled = [False]

# Bound every probed runEpoch: a few steady batches expose the same
# sync callsites as a full epoch (sites fire per batch), and MM_20M's
# epoch is 358k sentences. Overridable via env.
_PF = int(os.environ.get("BRICK_PROBE_MAX_BATCHES", "6"))


_VENV = str(_project / ".venv")
_SELF = str(Path(__file__).resolve())


def _record(method, is_sync):
    # ``is_sync``: caller already decided this call is a real CUDA->host
    # transfer. For value methods (item/tolist/cpu/...) that means the
    # receiver was non-CPU; a `.tolist()` on a CPU tensor is host-only
    # (no cudaMemcpyDtoH) and must NOT be attributed -- otherwise a fix
    # that moves a tensor host-side (residual A: lex the host byte slab)
    # would look un-fixed even though the device sync is gone. For
    # `.to()` it means CUDA-receiver -> CPU-result. This is strictly
    # more faithful than the doc's naive callsite count.
    if getattr(_state, "busy", False) or not _enabled[0]:
        return
    if not is_sync:
        return
    _state.busy = True
    try:
        stack = traceback.extract_stack()[:-2]  # drop wrapper + _record
        if not any(fr.name == "runEpoch" for fr in stack):
            return  # scope to the brick body runEpoch profiles
        # Attribute to the nearest *project* frame, not a torch-internal
        # shim. MPS interposes a TorchFunctionMode (_device.py) and the
        # Adam non-capturable path syncs inside torch/optim; walking to
        # the last bin/ frame makes the callsite device-independent so
        # the fast local MPS loop matches the authoritative CUDA read.
        # Skip lexer-internal pass-through frames so a sync is
        # attributed to the *caller* that drove it (the actionable
        # callsite), not to _to_text's own .tolist() line.
        _passthru = {"_to_text", "_token_stream", "_char_stream",
                     "tokenize", "_record"}
        caller = None
        for fr in reversed(stack):
            if (fr.filename.startswith(str(_project))
                    and not fr.filename.startswith(_VENV)
                    and fr.filename != _SELF
                    and fr.name not in _passthru):
                caller = fr
                break
        if caller is None:
            caller = stack[-1]  # pure torch-internal under runEpoch
        try:
            fname = str(Path(caller.filename).relative_to(_project))
        except ValueError:
            fname = caller.filename
        _counts[(method, fname, caller.lineno, caller.name)] += 1
    finally:
        _state.busy = False


def _is_cuda_tensor(x):
    return isinstance(x, torch.Tensor) and x.device.type != "cpu"


def _install_probes():
    originals = {}
    for name in _SYNC_METHODS:
        orig = getattr(torch.Tensor, name)
        originals[name] = orig

        def make(method, original):
            def wrapper(self, *a, **k):
                _record(method, _is_cuda_tensor(self))
                return original(self, *a, **k)
            return wrapper

        setattr(torch.Tensor, name, make(name, orig))

    # `.to()` is a sync only when it actually moves a CUDA tensor to
    # the host (`.to('cpu')` / `.to(cpu_tensor)`); `.to(dtype)` /
    # `.to(cuda)` are not. Decide post-call by comparing devices.
    _to_orig = torch.Tensor.to
    originals["to"] = _to_orig

    def _to_wrapper(self, *a, **k):
        result = _to_orig(self, *a, **k)
        if (_is_cuda_tensor(self) and isinstance(result, torch.Tensor)
                and result.device.type == "cpu"):
            _record("to", True)
        return result

    torch.Tensor.to = _to_wrapper
    return originals


def _report():
    total = sum(_counts.values())
    print(f"\n=== host-sync attribution (inside runEpoch) ===")
    print(f"total recorded host-sync calls: {total}\n")
    if not total:
        print("  (none -- runEpoch brick body is host-sync-free)")
        return
    width = max(len(m) for m, *_ in _counts)
    for (method, fname, line, func), n in sorted(
            _counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:4d}  {method:<{width}}  {fname}:{line}  ({func})")


def main():
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else "MM_xor.xml"
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    from data import TheData
    from Models import BaseModel
    from util import init_config

    config = str(_project / "data" / cfg_name)
    init_config(path=config,
                defaults_path=str(_project / "data" / "model.xml"))

    if cfg_name == "MM_20M_legacy.xml":
        shard_dir = str(_project / "data" / "fineweb")
        if not (os.path.isdir(shard_dir) and os.listdir(shard_dir)):
            print(f"SKIP {cfg_name}: shard corpus {shard_dir!r} absent")
            return
        TheData.load("text", shard_dir=shard_dir, num_shards=1, max_docs=64)
    else:
        TheData.load("xor")

    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    model, _ = BaseModel.from_config(config, data=TheData)
    model = model.to(dev)
    # BRICK_PROBE_FREEZE=1: freeze the BPE vocab + enable the GPU
    # tokenizer path, so the attribution reflects the frozen-vocab
    # GPU-train regime (what remains to drive to 0 DtoH).
    if os.environ.get("BRICK_PROBE_FREEZE") == "1":
        ps = getattr(model, "perceptualSpace", None)
        cl = getattr(ps, "chunk_layer", None) if ps is not None else None
        if cl is not None:
            cl.word_learning = 0
            ps._bpe_gpu_enabled = True
    optimizer = model.getOptimizer(lr=1e-4)

    print(f"[probe] cfg={cfg_name} device={dev} batch={batch} "
          f"cuda={torch.cuda.is_available()}")

    # Sync-debug mode (BRICK_PROBE_SYNCDEBUG=1): catches *implicit* C++
    # syncs the method monkeypatch structurally cannot see -- most
    # importantly boolean-mask indexing ``x[cuda_mask]`` /
    # ``masked_select`` / functional ``torch.nonzero`` /
    # data-dependent ops, which D2H-copy to size their output without
    # any Python `.item()`/`.tolist()` call. This is the doc's "device
    # count is the final gate" blind spot. CUDA-only.
    # Sync-ERROR mode (BRICK_PROBE_SYNCERROR=1): raise on the *first*
    # synchronizing op and print its full traceback. Anomaly mode is
    # on, so when the sync is autograd-internal (during backward) the
    # traceback is augmented with the *forward* op that created the
    # offending grad_fn -- which is what we actually need to fix. The
    # probe builds the model eagerly (no torch.compile), so frames are
    # real Python lines, not fused inductor kernels.
    if os.environ.get("BRICK_PROBE_SYNCERROR") == "1" and dev == "cuda":
        # Aux-term bisection: BRICK_PROBE_DROP=name1,name2 (or
        # cat:<category>) suppresses those Error.add terms so they leave
        # the loss graph -- if dropping a term removes the backward
        # sync, that term's subgraph is the culprit. Every (name,
        # category) seen is recorded and printed so the candidate set
        # is known.
        import Layers
        _seen = set()
        _drop = set(s for s in os.environ.get(
            "BRICK_PROBE_DROP", "").split(",") if s)
        _orig_add = Layers.Error.add

        def _patched_add(self, name, value, *, weight=1.0,
                         space=None, category="other"):
            _seen.add((name, category))
            if name in _drop or ("cat:" + str(category)) in _drop:
                return
            return _orig_add(self, name, value, weight=weight,
                             space=space, category=category)

        Layers.Error.add = _patched_add
        model.runEpoch(optimizer=optimizer, batchSize=batch,
                       split="train", max_batches=_PF)  # warm-up
        # Anomaly augments a backward sync's traceback with the forward
        # op -- invaluable for attribution, but it re-runs forward per
        # node and is very slow once the forward is sync-free (the sync
        # is reached deep in backward). Off by default so the DROP
        # bisection is fast; BRICK_PROBE_ANOMALY=1 for attribution.
        _anom = os.environ.get("BRICK_PROBE_ANOMALY") == "1"
        if _anom:
            torch.autograd.set_detect_anomaly(True)
        torch.cuda.set_sync_debug_mode("error")
        try:
            model.runEpoch(optimizer=optimizer, batchSize=batch,
                           split="train", max_batches=1)
            print("\n=== no synchronizing op raised "
                  "(brick is sync-free at this shape) ===")
        except RuntimeError as e:
            torch.cuda.set_sync_debug_mode("default")
            if _anom:
                torch.autograd.set_detect_anomaly(False)
            print(f"\n=== FIRST synchronizing op (anomaly={_anom}) ===")
            print(f"{type(e).__name__}: {e}\n")
            print(traceback.format_exc())
        finally:
            torch.cuda.set_sync_debug_mode("default")
            if _anom:
                torch.autograd.set_detect_anomaly(False)
            Layers.Error.add = _orig_add
        print(f"\n[aux terms seen] dropped={sorted(_drop)}")
        for nm, ct in sorted(_seen):
            print(f"  term={nm!r:32} category={ct!r}")
        return

    if os.environ.get("BRICK_PROBE_SYNCDEBUG") == "1" and dev == "cuda":
        import warnings
        sync_sites = Counter()
        model.runEpoch(optimizer=optimizer, batchSize=batch,
                        split="train", max_batches=_PF)  # warm-up

        def _showwarning(message, category, filename, lineno,
                         file=None, line=None):
            if "synchroni" not in str(message).lower():
                return
            for fr in reversed(traceback.extract_stack()[:-1]):
                if (fr.filename.startswith(str(_project))
                        and not fr.filename.startswith(_VENV)
                        and fr.filename != _SELF):
                    try:
                        fn = str(Path(fr.filename).relative_to(_project))
                    except ValueError:
                        fn = fr.filename
                    sync_sites[(fn, fr.lineno, fr.name)] += 1
                    return

        old_show = warnings.showwarning
        warnings.showwarning = _showwarning
        torch.cuda.set_sync_debug_mode("warn")
        try:
            model.runEpoch(optimizer=optimizer, batchSize=batch,
                           split="train", max_batches=_PF)
        finally:
            torch.cuda.set_sync_debug_mode("default")
            warnings.showwarning = old_show
        print("\n=== implicit-sync attribution "
              "(set_sync_debug_mode, inside runEpoch) ===")
        total = sum(sync_sites.values())
        print(f"total project sync callsites: {total}\n")
        for (fn, ln, fnc), n in sorted(sync_sites.items(),
                                       key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:4d}  {fn}:{ln}  ({fnc})")
        if not total:
            print("  (none in project frames)")
        return

    _install_probes()  # patched but disabled until after warm-up

    model.runEpoch(optimizer=optimizer, batchSize=batch, split="train",
                   max_batches=_PF)
    _enabled[0] = True
    model.runEpoch(optimizer=optimizer, batchSize=batch, split="train",
                   max_batches=_PF)
    _enabled[0] = False

    _report()


if __name__ == "__main__":
    main()
