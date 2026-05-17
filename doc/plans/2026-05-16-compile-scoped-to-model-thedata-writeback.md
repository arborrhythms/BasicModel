# Compile-scoped-to-model + TheData write-back — design spec

**Date.** 2026-05-16
**Owner.** Alec
**Status.** Spec for review (not started). Supersedes the "chase every
DtoH out of `runEpoch`" approach in `doc/BrickHostSyncStatus.md`.

> **For agentic workers:** this is a *design spec* for alignment, not a
> bite-sized TDD plan. Once the design is approved, a detailed
> implementation plan (per-task TDD steps) is written from it.
> **metalbaby is ONLINE** — GPU phases are runnable; keep every GPU
> run tiny/bounded/single (a prior over-aggressive loop OOM'd the box
> and required a reboot).

**Goal.** Make the model's per-batch compute (`forward + backward +
optimizer.step`) a single `torch.compile`d, CUDA-graph-capturable unit
that reads inputs from and writes outputs/errors to persistent
`TheData` device buffers, with all Python data-streaming and host
staging *outside* the compiled region — instead of trying to eliminate
every host sync from the whole `runEpoch` loop.

**Architecture (one paragraph).** `runEpoch` stays an eager Python loop
that streams data and stages it into persistent device buffers on
`TheData` (batch-scoped, in/around `runBatch`). The model's compute
step becomes a compiled callable that takes only those device buffers
and writes its outputs (predictions) and error/loss terms back into
`TheData` device buffers — no Python accumulation across the epoch,
no host round-trips inside the compiled region. Device policy
(placement, the synchronous-vs-async transfer rule, the capture
boundary) is centralized in a new `bin/Device.py`.

---

## Why (problem statement)

1. **"No DtoH anywhere in `runEpoch`" was the wrong frame.** It forced
   correctness-hostile tricks (async `pin_memory().to(non_blocking=True)`
   on ephemeral buffers) that caused a data race → NaN, and pinned-
   memory exhaustion → OOM (2026-05-16; reverted to synchronous).
2. **`torch.compile` is currently mis-scoped.** `ModelFactory.run`
   does `m = compile(m)` (`util.compile` → `torch.compile(model)`),
   then calls `m.run(...)` → `runTrial` → `runEpoch` → `runBatch` →
   `self.forward(...)`. `torch.compile(model)` only intercepts the
   module's `__call__`/`forward`; `m.run` delegates to the **original
   eager module**, so `self` inside `runBatch` is the eager module and
   the compiled callable is (almost certainly) never invoked. **Open
   question O1 — must verify on a GPU.** Either way, the captured unit
   today is conflated with the data loop.
3. **`torch.set_default_device(str(TheDevice.get()))`** (`util.py:247`)
   silently places *every* bare `torch.tensor(...)`/`torch.zeros(...)`
   on the GPU — the root cause of the staging-tensor footguns this
   session (`stringTensor`, `where_idx`, `encode_tokens` had to force
   `device='cpu'`). The capture boundary must own this explicitly.
4. **Accumulation leaks outside a batch.** `runEpoch`'s `record()`
   accumulates `outErr`/`inErr` as Python scalars across the epoch;
   error terms live in a global `TheError` registry + per-subspace
   `Error` instances. The owner's intent: nothing should accumulate
   outside an epoch; per-batch read/write belongs in `runBatch`.
5. **Device-type logic is scattered** (`Models.py:735`
   `any(p.is_cuda …)`, `data_loader` pin check, the `device.type ==
   "cuda"` branches). It should live only in `Device.py`.

## Goals / Non-goals

**Goals**
- One compiled, static-shape, capturable unit = the per-batch compute.
- Inputs read from / outputs+errors written to persistent `TheData`
  device buffers, batch-scoped in `runBatch`.
- All host staging + Python streaming *outside* the compiled region;
  transfers there are **synchronous** (correct, race-free).
- `bin/Device.py` is the single home for device policy.
- Divergence still fails loud (`ModelLoss` raise — already landed).
- Zero correctness regression (CPU representative-order net).

**Non-goals**
- Eliminating host syncs from the *eager* streaming loop (they're fine
  there — they don't break capture of the model unit).
- Multi-stream / async prefetch overlap (possible later; not now —
  needs the safe double-buffered-pinned + events pattern).
- Changing model math, the residual A–F fixes, or the dense
  `compute_masked` / `torch.where` forward rewrites (all kept).

---

## Current-state facts (grounding)

- `util.compile(model)` (`util.py:567`) = `torch.compile(model,
  mode=TheCompileMode, backend=TheCompileBackend)`; modes via
  `MODEL_COMPILE` / `MODEL_COMPILE_MODE` / `--compile-mode`.
- **VERIFIED — only the *model* is compiled, never `TheData`.**
  `ModelFactory.run` (`Models.py` ~5289): `m = compile(m)` (the
  `BasicModel` nn.Module) then `m.run(numTrials, numEpochs, …)`.
  `TheData` is a separate global `Data()` loaded earlier
  (`TheData.load(...)`), is not an nn.Module, and is never passed to
  `torch.compile`. ⇒ the model can stay a **persistent GPU resident**
  and `TheData` is the sole disk/CPU↔GPU boundary. `_brick_preflight`
  profiles `runEpoch` (bounded, budget-exempt — landed this session).
- `runBatch(… batch_override=(inputTensor, outputTensor) …)`
  (`Models.py:2534`) → `self.forward(inputTensor)` (2640) → returns
  `result` with `.lossOut`/`.lossIn` tensors (+ `batchNum`).
- `runEpoch` (`Models.py` ~3440): eager cursor loop; `record(result)`
  accumulates `outErr`/`inErr` via `.detach()` Python-side across the
  epoch; stages input via `prepInput`/byte-cursor (now synchronous).
- `TheData.toDevice()` (`data.py:549`) already pre-stages whole-dataset
  *tensor* splits to device once (list splits stay CPU, staged per
  batch by `prepInput`). **This is the existing "buffered device read"
  pattern to mirror for outputs/errors.**
- Errors: global `TheError = Error()` (`Layers.py:8211`) + per-subspace
  `SubSpace.errors = Error()` shared via `copy_context`;
  `Error.total()` = weighted sum for backprop; `.breakdown()` /
  `.snapshot()` diagnostic.
- Device API lives in `util.py` (`DeviceHandle`, `resolve_device`,
  `auto_device`, `_DeviceHolder`, `TheDevice`, `init_device`,
  `torch.set_default_device`). No `bin/Device.py` yet. Only ~4
  scattered `is_cuda`/`device.type==` sites remain post-revert.
- **Prefetch / `numWorkers` (verified):** `_TickPrefetcher`
  (`Models.py:3296`) is a **single daemon thread** calling
  `ds.next_tick()` ahead of the consumer. `<numWorkers>` (→
  `self._num_workers`, `Models.py:540`) sets only the **queue depth**
  (`Queue(maxsize)`), explicitly *"not thread count … next_tick is
  Python-bound and the GIL serializes Python execution"*. Overlap
  relies on GPU/C++ kernels releasing the GIL.
- **`data_loader` `pin_memory` is currently inert.** `runEpoch` drives
  `ds.next_tick()` directly; the `DataLoader` is built but **never
  iterated** (only for `loader.dataset`), so its
  `pin_memory=(cuda)` / `num_workers` never take effect. Post-revert,
  host→device staging is plain **synchronous** `.to(device)`; there is
  today **no pinned-buffer fast path and no multiprocess CPU
  parallelism** for tokenization.

---

## Target architecture

```
runEpoch  (eager Python, OUTSIDE compile)
  └─ per tick:
       cursor.next_tick()                      # host
       Device.stage_input(TheData, host_slab)  # ONE synchronous H2D
       runBatch():
         step = Model.compute_step             # the COMPILED callable
         step(TheData.in_buf)                   #   reads device buffers
            → writes TheData.out_buf,           #   predictions
                     TheData.err_buf            #   per-term loss scalars
         # backward + optimizer.step INSIDE step (capturable)
       Device.drain(TheData.err_buf)            # batch-scoped readout
                                                #   (one sync, eager)
```

- **Compiled unit (`Model.compute_step`)**: pure function of device
  tensors → device tensors. Static shapes (fixed `batchSize`, fixed
  `nObj`). No Python data-dependent control flow, no host syncs, no
  `.item()`. Contains `forward + loss + backward + optimizer.step`.
  Reads `TheData` input buffer(s); writes `TheData` output buffer +
  error buffer (a fixed `[n_terms]` device tensor of per-term scalars).
- **`TheData` buffers**: persistent, pre-allocated device tensors
  (registered once, sized from config): `in_buf` (already effectively
  this for tensor splits), `out_buf` (predictions), `err_buf`
  (`[n_error_terms]` float). The model **writes** these in-place
  (`copy_`/index-assign) instead of returning Python objects /
  appending to registries.
- **`runEpoch` / `runBatch` (eager)**: stream + stage (synchronous
  `.to(device)`), invoke the compiled step, then do *batch-scoped*
  readout of `err_buf` (one synchronous D2H per batch — fine, eager,
  outside capture) to produce logs/metrics. No cross-epoch Python
  accumulation: epoch metrics are reduced from per-batch `err_buf`
  reads, scoped in `runBatch`.
- **`bin/Device.py`**: owns `resolve/auto/handle`, the default-device
  policy, `is_cuda(x)` / `device_type()`, `stage(host→device)` (the
  single synchronous-transfer helper), and the capture-boundary
  markers. `util.py` device API moves here (or re-exports for compat).
  All `is_cuda`/`device.type==` callers route through `Device`.

---

## Key design decisions

- **D1. Compile the step, not the module-as-called.** Introduce an
  explicit compiled callable for the per-batch compute and invoke
  *that* from `runBatch` (so compilation actually takes effect and the
  captured region is exactly the brick). Resolve O1 first (verify
  whether today's `m=compile(m)` ever runs compiled).
- **D2. Transfers at the boundary, into *persistent static* buffers.**
  Never inside the compiled step. **Default = synchronous**
  `static_buf.copy_(host)` / `.to(device)`. The *only* sanctioned
  async form is the researched-safe one (see boundary section):
  copy into a **prefetcher-/`TheData`-owned, lifetime-managed,
  double-buffered pinned** host tensor, then `non_blocking=True` into
  a persistent device buffer, guarded by a `torch.cuda.Event` before a
  slot is refilled. **Banned forever:** `x.pin_memory().to(dev,
  non_blocking=True)` on an ephemeral per-call tensor (the
  NaN-race / OOM footgun this session). CUDA-graph inputs must be
  updated via `static.copy_(new)` — never reassigned/reallocated
  during replay.
- **D3. Write-back, don't return/accumulate.** The step writes
  predictions/errors into `TheData` device buffers in-place; eager
  code reads them per batch. Kills the `runEpoch` `record()` cross-
  epoch Python accumulation and keeps the compiled region output-pure.
- **D4. Explicit default-device discipline.** `Device` owns the
  default-device decision; host-staging construction is always
  explicit `device='cpu'`. Re-evaluate whether
  `torch.set_default_device(GPU)` should remain global.
- **D5. Divergence stays loud.** The `ModelLoss` raise (landed) is the
  net; the compiled step must not reintroduce silent `nan_to_num`.
  (A per-batch finiteness check on `err_buf` at the eager boundary is
  the natural, capture-safe place if an in-step check is undesirable.)
- **D6. Keep the landed forward fixes.** Residual A–F, dense
  `compute_masked`, IR-mask `torch.where`, `torch.zeros((),…)`,
  host-build+single-sync staging, `ImpenetrableLayer` MODEL_DEBUG gate
  — all retained (they're on-device/race-free and reduce in-step
  syncs, which still matters for capture).

---

## Disk ↔ CPU ↔ GPU boundary (researched)

The model is a persistent GPU resident (compiled, never moved);
`TheData` owns every host transfer. Canonical pattern (CUDA-graph +
data-loading best practice — sources below):

**Input (disk → CPU → GPU), per tick, eager, OUTSIDE the step:**
1. **Disk → CPU:** shard read + (today) Python lex/encode on the
   producer (the `_TickPrefetcher` thread or, see next section, real
   worker processes). Prefer pre-tokenised / memory-mapped shards so
   the hot path carries tensors, not Python tokenisation.
2. **CPU → pinned:** producer writes into a **persistent,
   double-buffered pinned host staging tensor it owns** (≥2 slots so
   slot *k+1* fills while slot *k*'s copy is in flight). Owner-managed
   lifetime is what makes `non_blocking` *safe* here (vs the banned
   ephemeral-pin footgun).
3. **pinned → GPU:** `static_in_buf.copy_(pinned_slot,
   non_blocking=True)` into the **persistent static device input
   buffer** the captured step reads. A `torch.cuda.Event` recorded
   after the copy gates reuse of that pinned slot (producer waits on
   the event before refilling). Synchronous `.copy_()` is the safe
   default; the async form is the *only* sanctioned `non_blocking`.

**Output / errors (GPU → CPU → disk), per batch, eager:**
4. The step writes predictions + per-term loss scalars into persistent
   **static device** `out_buf` / `err_buf` (in-place; no Python
   return, no registry append inside the step).
5. **GPU → CPU:** one batch-scoped D2H of `err_buf` (small `[n_terms]`)
   — synchronous by default, or async into an owned pinned readback
   buffer + event. This is the natural place for the D5 finiteness
   check (raise on non-finite).
6. **CPU → disk:** checkpoints / logs are written off the hot path
   (background thread/process), never inside the step or the
   per-tick critical path.

**Why this is safe where the session's edits were not:** ownership +
lifetime + a CUDA event guard the pinned buffer across the in-flight
copy; ephemeral `x.pin_memory().to(non_blocking=True)` had none of
these (source freed mid-copy → NaN; per-tick pin allocs → OOM).

## `numWorkers` evaluation & recommendation

**Current:** `<numWorkers>` only sizes the `_TickPrefetcher` queue;
production is a **single GIL-bound thread**. The `DataLoader`'s real
`num_workers`/`pin_memory` are inert (loader never iterated). So:

- Tokenisation (byte/word lexer in `_lex_batch`/`_token_stream`,
  `encode_tokens`, `tokens_to_decoded`, `_embed` OOV/index) is
  **Python/GIL-bound and single-threaded**. The single prefetch thread
  can only overlap it with GPU compute *while CUDA kernels hold the
  GIL released*.
- Raising `<numWorkers>` past ~2–3 buys nothing for throughput (queue
  depth only — it does not add a producer). If per-tick lex cost ≳ GPU
  step (likely for **MM_5M**: real vocab, byte lexer, 358k sentences),
  the producer is the bottleneck and the GPU starves regardless of
  `<numWorkers>`.

**Recommendation (to fold into the plan):**
- **Short term:** keep the single overlap thread; document
  `<numWorkers>` as *queue depth only*; size it 2–3. Make the pinned
  double-buffer (boundary step 2/3) the actual win.
- **Real fix for the CPU bottleneck (measure first on metalbaby):**
  either (a) **pre-tokenise shards to a binary/mmap format** the
  cursor reads directly (no Python lex on the hot path — best per the
  data-loading guidance), or (b) move tokenisation to **true
  multiprocess producers** (each its own GIL) feeding a host-tensor
  queue, and redefine `<numWorkers>` as producer-process count
  (`persistent_workers`-style: spawned once per run, not per epoch).
  (a) is preferred — it removes the Python lexer from the steady
  state entirely and makes the input path a pure tensor copy.
- Decide (a) vs (b) by a metalbaby profile of producer time vs step
  time at the real `batchSize` (added as a Phase-0 measurement).

---

## Phase 0 findings (2026-05-16) — RESOLVED

- **O1 — PIVOTAL. `torch.compile` is currently a complete no-op for
  the training/pre-flight path.** Verified locally (device-independent;
  `/tmp/o1_compile_probe.py`): `m = compile(m)` →
  `torch._dynamo.eval_frame.OptimizedModule` wrapping `_orig_mod`;
  `m.run`/`m.runEpoch` bind to the **eager `_orig_mod`** (`.__self__
  is _orig_mod`, not the wrapper), and `runBatch` calls
  `self.forward()` on that eager module. After a bounded `runEpoch`
  through the "compiled" wrapper, `torch._dynamo.utils.counters
  ['frames']` is **`{}`** — dynamo never traced anything. ⇒ every
  prior "Model compiled (inductor, max-autotune)" run, **including the
  pre-flight's 58/430-`cudaMemcpyDtoH` measurements, was fully
  eager**; `--compile-mode` did nothing. **Consequence:** "actually
  build and invoke a compiled per-batch step" is *foundational*, not
  Phase 4 — it moves to **Phase 1**. Sync elimination only matters
  once a real compiled step exists to capture.
- **O2 — runBatch boundary.** In: `batch_override=(inputTensor,
  outputTensor)` (device). Out: `BatchResult = namedtuple(
  outputPred, symbols, lossOut, lossIn, inputPred, forwardInput)`.
  `lossOut`/`lossIn` are 0-dim device scalars; `aux_total =
  outputSpace.subspace.errors.total()` (the `category="symbol"` etc.
  terms) is folded into `totalLoss` *inside* the step. `runEpoch`'s
  `record()` already **overwrites** (not accumulates) `outErr/inErr`
  as 0-dim tensors and `.item()`s once at epoch end; only the **eval**
  path clones `outputPred`/`inputPred` into chunk lists (training
  discards them). ⇒ `err_buf` = a fixed `[n_terms]` device float
  vector (lossOut, lossIn, + named `TheError` terms via a stable
  term→index map); `out_buf` = predictions, **needed only for eval**
  (training write-back is scalars only). No cross-epoch training
  accumulation to remove beyond moving the per-batch scalars into
  `err_buf`.
- **O3 — buffer model.** `TheData.toDevice()`: *tensor* splits become
  persistent whole-dataset **device** residents (pre-shaped once);
  *list* splits stay **CPU** only because the streaming DataLoader
  pickles them across worker processes. `out_buf`/`err_buf` are model
  outputs (never pickled across workers) ⇒ allocate them as
  **persistent device tensors at load/first-use, sized from config**
  (B, nObj, D / n_terms); update in-place via `copy_` (CUDA-graph
  static-buffer rule). Fixed within a run (config-static) ⇒ no
  per-replay realloc. (Also: `data.py:584` `t.device.type != "cpu"`
  is another scattered device-type site for Phase 2/Device.py.)
- **O4 — producer-vs-step profile**: still needs metalbaby (real GPU
  step timing at the real `batchSize`, bounded MM_5M). Decides the
  `numWorkers` path (pre-tokenised/mmap vs multiprocess producers).
  Deferred into its phase; not blocking the reprioritised Phase 1.

## Phased plan (revised after Phase 0)

**Phase 1 — Actually invoke a compiled per-batch step (was Phase 4;
O1 made this foundational).**
- Factor the per-batch compute (`forward + loss + backward +
  optimizer.step`) into an explicit callable; `torch.compile` *that*
  and **invoke the compiled object** from `runBatch` (not via the
  eager `_orig_mod`). Streaming/staging stays eager outside it.
- Update `_brick_preflight` to profile the **compiled step** and
  assert (re-measure DtoH; the prior numbers were eager and moot).
- Verify: O1 probe now shows dynamo frames > 0; CPU representative
  suite 0 regr.; tiny bounded metalbaby run.

**Phase 2 — `bin/Device.py` (no GPU needed).**
- Create `bin/Device.py`; move the `util.py` device API there;
  `util` re-exports for back-compat. Add `Device.is_cuda(x)`,
  `Device.stage(host, *, dtype=None)` (synchronous), `Device.type()`.
- Route the ~5 scattered sites (`Models.py:735`, `data.py:584`,
  `data_loader` pin check, residual `device.type==`) through `Device`.
- Verify: CPU representative-order suite, 0 regressions.

**Phase 3 — `TheData` output/error buffers + write-back (CPU-testable).**
- Add persistent device `out_buf` / `err_buf` (+ a stable term→index
  map) to `TheData`, sized from config; allocate at load/first-use.
- `runBatch`: write per-term loss scalars into `err_buf` in-place
  (`out_buf` only on the eval path); replace `record()`'s per-batch
  0-dim handoff with a batch-scoped `err_buf` read. Epoch metrics
  reduced from per-batch reads. The D5 divergence raise reads
  `err_buf` here (eager boundary).
- Verify: CPU representative-order suite + metrics parity vs current
  (`outErr`/`inErr` numerically unchanged), 0 regressions.

**Phase 4 — Persistent static buffers + safe staging.**
- Promote `in_buf`/`out_buf`/`err_buf` to **persistent static**
  tensors updated via `static.copy_(new)` (never reassigned). Default
  synchronous. Add the *owned, double-buffered pinned* host staging +
  `torch.cuda.Event` reuse-guard as the single sanctioned async path
  (boundary section). No ephemeral pin anywhere.
- Verify: CPU suite; metalbaby tiny bounded run — no NaN, stable
  memory (no pinned-alloc growth).

**Phase 5 — `numWorkers` resolution (gated on O4 measurement).**
- Per O4: implement (a) pre-tokenised/mmap shards (preferred) or
  (b) multiprocess producers; redefine/​document `<numWorkers>`
  accordingly; keep queue depth 2–3 for the overlap thread otherwise.
- Verify: CPU suite + metalbaby producer-vs-step profile improved.

**Phase 6 — verify + tune on metalbaby.**
- `make sync HOST=mb`; pre-flight passes (DtoH=0 in the step) at the
  real `batchSize`; bf16 + `max-autotune` capture works; MM_xor +
  bounded MM_5M; representative-order CPU stays green. Update
  `BrickHostSyncStatus.md`.

---

## Risks / open questions

- **O1 — RESOLVED**: compile is a no-op today (eager `_orig_mod`
  delegation); prior "compiled"/pre-flight DtoH numbers are eager and
  moot. Phase 1 (build+invoke a real compiled step) is now the crux.
- **Static shapes.** Capture needs fixed `batchSize`/`nObj`; the
  cursor's last partial batch and N-halving/conceptualOrder shape
  variation must be padded/bucketed or excluded from the captured
  step. Design detail for the implementation plan.
- **In-place write-back vs autograd.** Writing predictions into a
  buffer that participates in grad needs care (write detached
  outputs; keep the loss graph internal to the step, only scalars
  out). Spec the buffer as *outputs/metrics only*, grads stay in-step.
- **Error registry semantics.** `.breakdown()`/`.snapshot()`
  diagnostics currently read the Python registry; map them onto
  `err_buf` + the term-index map (or keep diagnostics eager,
  MODEL_DEBUG-gated, as already done for bucket-2/3).
- **metalbaby OOM risk** — every GPU run must be tiny/bounded/single
  (a prior over-aggressive iterative loop OOM'd it → reboot). No
  unbounded epochs, no piled-up background runs.

## Verification (every phase)

CPU representative order — `test_mm_xor test_mm_boolean
test_universality test_invertibility test_xor_spaces
test_basicmodel::TestNormalizeFlag test_lexicon_ownership
test_conceptual_bivector test_perceptual_bivector
test_phase2_pipeline_primitives` (full suite needs
`PYTORCH_ENABLE_MPS_FALLBACK=1` for the unrelated MPS `_cdist_backward`
gap) → metalbaby pre-flight / step-scoped device-event assert
(tiny bounded runs only).

## Sources (research grounding)

- PyTorch — *A guide on good usage of non_blocking and pin_memory()*
  (ephemeral-pin corruption; safe owner-managed pinned + non_blocking).
- NVIDIA — *CUDA Graph Best Practice for PyTorch* / PyTorch
  *CUDAGraph Trees* (preallocated static in/out buffers; update via
  `static.copy_(new)`, never reallocate during replay).
- PyTorch DataLoader guidance (`num_workers` ≈ cores,
  `persistent_workers`, `pin_memory`, `prefetch_factor`;
  CPU-bound tokenisation → pre-tokenise / mmap or multiprocess
  producers, not a single GIL-bound thread).
