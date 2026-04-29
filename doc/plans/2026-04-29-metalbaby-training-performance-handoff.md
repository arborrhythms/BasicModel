# Handoff: Training-throughput performance on `metalbaby` (GB10)

**Target machine:** `admin@metalbaby.local:~/WikiOracle/basicmodel`
**GPU:** NVIDIA GB10 (Grace Blackwell, unified memory)
**Date:** 2026-04-29
**Status:** GPU is severely under-utilized (~25-33% active fraction). Two fixes already landed; ~2-3× more speedup is on the table once the residual main-thread CPU gap is closed.

## TL;DR

A single training loop on `metalbaby` runs at ~25-30% GPU duty cycle: ~1 s of compute followed by ~2-3 s of CPU-bound housekeeping per batch. Two upstream fixes have been merged:

1. **Async tick prefetch** ([Spaces.py](../../bin/Models.py) `BasicModel._TickPrefetcher`) hides `ds.next_tick()` cost on a single background thread; XML `<numWorkers>` now controls queue depth (in-flight tick budget).
2. **Span-table sweep skipped in byte mode** ([Spaces.py:1552](../../bin/Spaces.py)) — the per-doc tokenizer pass over all 426K source docs at `Embedding.create()` is now a no-op when `lexer=byte`.

The residual gap is in **other main-thread per-batch work**: `prepInput`/`prepOutput`, `runBatch` internals (forward/backward + optimizer), and tail dispatch (`flush_word_buffers`, `dispatch_per_row_reset`, `dispatch_soft_reset`, `post_tick_compact`). Closing that gap is the next handoff target.

## Observed performance (post-fix snapshot)

From `nvidia-smi dmon -s u -d 1` while training under `make train_micro`:

```
SM% column samples (1/sec):
70  6  6  96  96  6  6  72  28  6  6  45  6  6  96  63  6  6  96  7  7  96  96  6  6  ...
```

- **Active fraction:** ~25-33% — one ~70-96% sample followed by 2-3 dead idle samples (~6%).
- **Power draw:** 13-35 W on a GPU whose real working envelope is 90-150+ W. Corroborates the duty cycle.
- **Throughput:** at `<batchSize>128</batchSize>` and ~3 s/batch, ~43 sentences/sec end-to-end.
- **Per-batch wall:** ~1 s GPU compute + ~2 s main-thread CPU work between batches.

For the original sweep target (`--max-docs 1000000 --num-shards 10`), the run will take ~10-12 h at this rate. Saturating the GPU should cut it to **~4-5 h**.

## What's already done

### 1. Async tick prefetch ([Models.py](../../bin/Models.py))

Added `BasicModel._TickPrefetcher`: single daemon thread, bounded queue, calls `ds.next_tick()` ahead of consumption.

- **Why one thread:** `next_tick` is Python-bound; the GIL serializes Python execution, so additional prefetch threads contend without speedup. The GPU side releases the GIL during CUDA kernels, which is exactly the window the one prefetch thread uses.
- **`<numWorkers>` semantic change:** previously read but ignored (PyTorch `DataLoader.num_workers=0` was hard-coded because the loader was never iterated). Now controls queue depth — at most one tick in-flight at the consumer + `numWorkers - 1` buffered ahead. `0` preserves the legacy synchronous path.
- **Cleanup:** `prefetcher.close()` runs after the with-block; thread is `daemon=True`, so abnormal-exit paths don't leak.

To enable, set in `data/MM_5M.xml` (or any model XML):
```xml
<numWorkers>4</numWorkers>
```

### 2. Span-table sweep gated on byte mode ([Spaces.py:1552](../../bin/Spaces.py))

`Embedding.create()` previously walked all source documents on every fresh model load, calling `_token_stream(doc)` per doc and storing byte tensors + spans into `self.doc_sources` / `self.doc_spans`. In byte mode this:

- Tokenized 426K docs in Python (slow; "Building span table: N/426804 docs" lines)
- Held a list of byte tensors in RAM
- Called `self.insert(token_text)` per token — **no-op in byte mode** because all 256 byte values are already inserted at construction time
- Produced caches that **no downstream consumer indexes into**

Loop now wrapped in `if not self.byte_mode:`. Word/BPE mode is unaffected. Saves several minutes of startup and the held byte-tensor list per fresh model load.

### 3. Removed dead emergency-checkpoint path ([Models.py](../../bin/Models.py))

`_autosave_on_exception` / `_save_exception_checkpoint` / `.emergency.ckpt` writing path deleted. Existing `.emergency.ckpt` files on disk are inert; safe to delete.

## Diagnostics — measure utilization

### Live time-series

```bash
# Tabular running display, one row per second (best for verifying gap pattern)
nvidia-smi dmon -s u -d 1

# CSV log to file for later analysis
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,memory.used \
           --format=csv -l 1 -f /tmp/gpu_log.csv

# Interactive UI with charts (recommended)
sudo apt install -y nvtop && nvtop
```

The "good" signature: SM% column consistently ≥80%, no dead-idle gap rows.

### Find the residual hot frame (Python side)

```bash
# Get the training PID
ps -ef | grep "Models.py" | grep -v grep

# Live top-style sampler at 200 Hz for 30 s
sudo /home/admin/WikiOracle/basicmodel/.venv/bin/pip install py-spy
sudo /home/admin/WikiOracle/basicmodel/.venv/bin/py-spy top --pid <PID> --rate 200
```

The top frames by `%TotalTime` are the bottleneck. Likely candidates given the architecture:

| Hot frame | Mitigation |
|---|---|
| `flush_word_buffers`, `post_tick_compact`, `dispatch_per_row_reset` | Tail dispatch is Python-bound and serial with the next batch. Move into the prefetch window or batch them across multiple ticks. |
| `prepInput`, `prepOutput`, `.to(device)` | Host→device transfer + dtype convert. Pin memory in the data loader; use `non_blocking=True` on the copy. |
| `optimizer.step()`, `.item()`, `.cpu()` | Forced GPU syncs. Collapse `.item()` reads to once per N batches; ensure no host scalar reads in the inner loop. |
| `runBatch` forward/backward | Real GPU work. The "gap" might be sub-second compute that 1 s polling under-samples — confirm with `nvidia-smi dmon -s u`. |

### CPU/IO snapshot

```bash
# CPU fan-out: if only 1-2 threads pegged, a single Python thread is the limiter
top -p <PID> -H -d 1

# Disk: should be near-zero for byte-mode training (data fits in pagecache)
iostat -x 2
```

## Recommended next steps (ranked by ROI)

### 1. Move tail dispatch into the prefetcher window — **highest ROI, medium effort**

The tail dispatch (`flush_word_buffers`, `dispatch_per_row_reset(hard_eos)`, `dispatch_soft_reset`, `post_tick_compact`) currently runs on the main thread *between* batches, holding the GIL. Possible refactors:

- **Batch them** across multiple ticks: many of these are bookkeeping that doesn't need to fire every tick. If reset-state can accumulate and flush every K ticks, K-1 ticks save the dispatch cost.
- **Async dispatch:** if any of these can run on a worker thread (after dependency analysis), they become invisible to the main loop.
- **Vectorize:** if `dispatch_per_row_reset(hard_eos)` is per-row Python looping, batched tensor ops would help.

Start by running `py-spy top` to confirm tail dispatch is the actual hot frame before touching any of these. Don't optimize what isn't hot.

### 2. Larger `<batchSize>` — **easy win if not already tried**

`<batchSize>32</batchSize>` is small for GB10 (128 GB unified memory). Larger batches amortize fixed per-tick CPU overhead. Try `<batchSize>256</batchSize>` and watch util.

Constraint: AMP / gradient accumulation considerations — verify the LR schedule still makes sense at the new batch size. With `<learningRate>0.0005</learningRate>` and `<l1Lambda>0.01</l1Lambda>` it's likely fine to scale up by 2-4× without retuning.

### 3. Pinned memory + non-blocking host→device — **easy, modest ROI**

The DataLoader is built with `pin_memory=(TheDevice.get().type == "cuda")` already, but the actual `.to(device)` calls inside `prepInput`/`prepOutput` may not use `non_blocking=True`. Audit those paths and add `non_blocking=True` where the next read of the tensor is also on GPU.

### 4. `runBatch` instrumentation — **medium effort, finds where compute time goes**

Add an optional `MODEL_PROF=1` env path that wraps the inner-loop sections in `torch.cuda.Event` timers (start/end events around forward, backward, optimizer.step, dispatch). Print a 32-batch rolling average of each. Cheap when off, surgical when on.

This lets future debuggers see "forward = X ms, backward = Y ms, opt = Z ms, dispatch = W ms" without external tools.

### 5. Bigger swings (if 1-4 plateau) — **high effort**

- **Vectorize `_lex_batch`** — currently per-row Python; could become a single `torch` byte-decode op for byte mode.
- **CUDA Graphs** — `--compile-mode reduce-overhead` enables CUDAGraph capture, which removes per-launch overhead. Helps when launch overhead is significant relative to per-kernel compute.
- **DDP / multi-GPU** — N/A on single-GPU GB10, but worth flagging if the box ever gains a second GPU.

## How to verify a fix worked

1. **Rerun:** `make train_micro` (or the equivalent command).
2. **Watch:** `nvidia-smi dmon -s u -d 1` in another terminal.
3. **Pass criterion:** SM% column shows sustained ≥70% (rather than the current sawtooth between 96% and 6%). Power draw should rise into the 60-100+ W range.
4. **Throughput sanity:** the `batch = N (Δ=Xs)` line in the train log should show smaller `Δ` for a given batch-stride. Roughly: half the gap = double the rate.

## Hypothetical perf envelope

| Stage | Per-batch wall | Throughput | 1 M-doc run |
|---|---|---|---|
| Pre-fix (sync `next_tick`) | ~1.0 s | ~32 sentences/sec | ~20 h |
| Post-prefetch + bigger batch (current) | ~3.0 s @ batch 128 | ~43 sentences/sec | ~10-12 h |
| After tail-dispatch fix | ~1.3 s @ batch 128 (estimate) | ~100 sentences/sec | ~4-5 h |
| GPU-bound limit (estimate) | ~0.5-0.8 s @ batch 128 | ~150-250 sentences/sec | ~2-3 h |

The third row is the realistic next milestone. The fourth is where compute-saturation should land if all CPU side-channels are addressed.

## Open questions

- **Is `dispatch_per_row_reset` per-row Python or a vector op?** Inspect; if Python, vectorize.
- **Can `post_tick_compact` skip ticks where no compaction is needed?** It currently runs every tick.
- **Do `.item()` reads inside `runBatch` force per-step GPU syncs?** Audit; collapse to per-epoch where possible.
- **Is `<numWorkers>4</numWorkers>` actually being read in the active config?** Verify with `grep numWorkers data/MM_5M.xml` on `metalbaby`.
- **Does `--compile-mode reduce-overhead` (CUDAGraphs) help here?** Worth a one-off A/B test once the CPU side is closer to balanced.

## Pointers

- Prefetcher: [bin/Models.py](../../bin/Models.py), search `class _TickPrefetcher`
- Prefetcher wiring in `runEpoch`: [bin/Models.py](../../bin/Models.py), search `prefetcher = (BasicModel._TickPrefetcher`
- Span-table skip: [bin/Spaces.py:1552](../../bin/Spaces.py)
- Tail dispatch sites: search `flush_word_buffers`, `dispatch_per_row_reset`, `post_tick_compact` in [bin/Models.py](../../bin/Models.py)
- Source handoff that triggered this work: `~/.claude/plans/test-test-mm-boolean-py-http-boolean-py-hidden-tulip.md` (DNF teardown — performance work piggybacked on its training-soak validation)

## Reproduction recipe

On `metalbaby`:

```bash
cd ~/WikiOracle/basicmodel

# Archive any stale checkpoints
mv data/MM_5M.ckpt data/MM_5M.ckpt.bak-$(date +%Y%m%d) 2>/dev/null

# Set numWorkers ≥ 1 in the XML
grep numWorkers data/MM_5M.xml          # should be >0; if not, edit
# <numWorkers>4</numWorkers>

# Start training
make train_micro

# In another terminal, watch utilization
nvidia-smi dmon -s u -d 1

# Or capture a profile
sudo /home/admin/WikiOracle/basicmodel/.venv/bin/py-spy top \
     --pid $(pgrep -f Models.py) --rate 200
```

Look for: SM% sustained ≥70% (means CPU-side gap is closed). If still sawtooth, follow the diagnostics above.
