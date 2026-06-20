# Handoff: Vectorize the compute brick, capture CUDA graphs, retire legacy paths

**Date.** 2026-04-27
**Owner.** Alec
**Status.** Not started.

**Predecessor.** [2026-04-26-rolling-cursor-doc-streaming-handoff.md](2026-04-26-rolling-cursor-doc-streaming-handoff.md)
(landed). That handoff relocated the in-loop Reset gate, introduced
the per-row cursor on `TheData`, added soft / hard reset semantics, and
flagged §6 (vectorize residual sync points) and §7 (CUDA-graph capture)
as required follow-ups. This handoff executes those follow-ups and
enumerates the legacy paths that become removable once they land.

---

## Context

After the rolling-cursor handoff, `runBatch` no longer fires the Reset
cascade and the truth-layer compact has moved to the outer
doc-streaming loop. But the brick body still issues GPU→host syncs at
three sites identified in the predecessor's masked-brick audit. CUDA-
graph capture requires zero `cudaMemcpyDtoH` events between
`forward(...)` start and `optimizer.step()` end; the brick is not yet
graph-capturable.

The legacy paths that survive in tree are also a known smell — the
two-path runEpoch (cursor vs. legacy DataLoader), the dead
`InputSpace.batch_advances_sentence` stub property, the `Reset()`
zero-arg fallback, and the silent `nObj-1` byte truncation in
`_lex_batch`. These exist either because the predecessor handoff kept
them for back-compat with the existing test suite or because removing
them required vectorization work that wasn't in scope.

---

## Goals

1. **Sync-free brick.** `runBatch`'s body issues zero GPU→host
   transfers in the AR training path. Profiler trace between
   `forward()` start and `optimizer.step()` end shows zero
   `cudaMemcpyDtoH` events on a CUDA device.
2. **CUDA-graph capture works.** Either `torch.compile(mode="reduce-
   overhead")` succeeds end-to-end on the brick, or a hand-written
   `torch.cuda.CUDAGraph.replay()` over the brick produces bit-
   identical results to the eager path on the same input.
3. **One data path, one Reset path.** The legacy `__iter__` of
   `SentenceStreamDataset` is removed; the non-cursor branch of
   `runEpoch` is removed; every space's `Reset` requires the
   `(batch, hard)` signature; the dead `batch_advances_sentence`
   property is removed; `_end_of_stream` becomes a host-side
   `list[bool]` only (no scalar/tensor variants).

---

## Architecture

### The remaining sync points (from §"Masked-brick audit" in the predecessor)

| # | Site | What it does | Vectorization strategy |
|---|------|-------------|------------------------|
| 1 | `WordSpace.stm_residual_microbatch` (Language.py ~2520) | `not_fired.any().item()` early-out | Always compute `disc.predict()`; zero the bias on already-fired rows via `torch.where(_stm_fired, 0, bias)`. Cost: one matmul that was sometimes skippable. |
| 2 | `SymbolicSpace` truth-layer record loop (Spaces.py ~6760) | Per-cell `.item()` mask check + Python `should_store` / `record` calls | **Drop the `should_store` mask entirely.** Encode `truth_vec * DoT` via the codebook; low-DoT entries become near-zero vectors that the codebook's nearest-neighbor lookup matches against the existing zero/empty prototype, claiming no new slot. `record_many(vecs, dots)` becomes a pure batched insert. No mask, no per-cell gate, no later pruning. |
| 3 | `SyntacticLayer.compose` Python loops (Language.py ~1257–1915) | Many `.tolist()` / `.item()` calls drive per-row rule selection and per-cell `subspace.add_word(b, pos, gid)` Python list mutations | **Path B** from the predecessor: compose writes to a tensor word buffer; `flush_word_buffer` runs once per tick *outside* the brick to materialize `subspace.word` for legacy consumers. The compose body itself becomes sync-free. |

Item 3 is the largest piece of work. Items 1 and 2 are independent and
can land in any order.

### Tensor word buffer (Path B, recap from predecessor §"Tensor buffer for subspace.word")

`SubSpace` ([Spaces.py:2390]) gains:
```python
self.word_records  # [B*K, max_depth, ENTRY_WIDTH] long
self.word_count    # [B*K] long, current depth per cell
```
`ENTRY_WIDTH = 7`: `(batch, vec_idx, rule, order, leaf1, leaf2, leaf3)`.

`add_word` becomes a tensor scatter:
```python
def add_word(self, b_indices, vec_idxs, rule_ids, order=None,
             leaf1=None, leaf2=None, leaf3=None):
    depths = self.word_count[b_indices]
    self.word_records[b_indices, depths, RULE]    = rule_ids
    self.word_records[b_indices, depths, VEC_IDX] = vec_idxs
    if order  is not None: self.word_records[b_indices, depths, ORDER]  = order
    if leaf1  is not None: self.word_records[b_indices, depths, LEAF1]  = leaf1
    if leaf2  is not None: self.word_records[b_indices, depths, LEAF2]  = leaf2
    if leaf3  is not None: self.word_records[b_indices, depths, LEAF3]  = leaf3
    self.word_count[b_indices] += 1
```

Outer loop calls `flush_word_buffer` once per tick after `runBatch`:
```python
def flush_word_buffer(self, subspace):
    counts = self.word_count.tolist()        # one sync; [B*K] ints
    records = self.word_records.tolist()     # one sync; nested ints
    for bk, depth in enumerate(counts):
        for d in range(depth):
            subspace.word.append(WordEntry(*records[bk][d]))
    self.word_count.zero_()
```

Brick body is now sync-free; downstream consumers (`decompose`,
`reconstruct`, SVO walker, derivation-trace tests) see
`subspace.word` populated as before.

### Legacy paths to retire

| # | Path | Why it exists today | Removable when |
|---|------|--------------------|---------------|
| L1 | `SentenceStreamDataset.__iter__` | Used by tests that build the dataset directly | After cursor is universal *and* tests migrate to `next_tick` |
| L2 | Non-cursor branch in `runEpoch` ([Models.py:2585+]) | Numeric data (MNIST, XOR-with-labels), non-byte lexers | After cursor mode handles all input shapes (byte/word/sentence lex) |
| L3 | `InputSpace.batch_advances_sentence` property | Predecessor handoff stub, returns constant `True` | Now (no caller reads it after `runBatch` was relocated) |
| L4 | `_end_of_stream` as scalar / tensor | Diagnostic artifact; cursor uses host-side `list[bool]` | After every reader migrates to the list form |
| L5 | `Reset()` zero-arg fallback in `_reset_call` | Compat shim for any space that hasn't migrated | After every space's `Reset` accepts `(batch, hard)` natively |
| L6 | Fast-path `if all(hard_eos): space.Reset()` global form in `dispatch_per_row_reset` | Restores pre-handoff dispatch overhead | Optional — keep for perf parity, mark as such |
| L7 | `_lex_batch` `nObj - 1` byte truncation | Reserves the empty-word EOS sentinel slot | After cursor sets `slab_bytes = nObj - 1` everywhere (already the case in `runEpoch`); makes the truncation a deliberate slab boundary, not silent drop |
| L8 | `MODEL_DEBUG` `_end_of_stream` paranoia assert | Predecessor handoff sanity check | Now (the assert site is gone from `runBatch`) |

L1, L2 are the load-bearing pair: removing them collapses the two
data paths into one. The rest are smaller.

---

## Sequencing (one PR per step preferred)

1. **§6a — Vectorize STM-residual gate.** Smallest change. Drop the
   `.item()` early-out; always call `disc.predict()`; zero the bias
   on already-fired rows via `torch.where`. Run the existing
   discourse and STM tests. *Independent.*

2. **§6b — Replace the truth-layer record loop with DoT-scaled
   batched insert.** No mask. `TruthLayer.record_many(vecs, dots)`
   stores `vecs * dots.unsqueeze(-1)` via the existing codebook
   path; low-DoT rows resolve to the zero/empty prototype and don't
   pollute the store. Delete `should_store`. Rewrite the
   `SymbolicSpace.forward` per-cell loop as a single
   `record_many(act_flat, dots_flat)` call. Run the truth-layer and
   SymbolicSpace tests; the codebook nearest-neighbor invariant
   ("near-zero vec → existing zero prototype, no growth") is the
   correctness gate. *Independent of 1; blocking 7.*

3. **§6c — Tensor word buffer (Path B).** Largest change.
   - SubSpace: register `word_records` / `word_count` buffers; new
     vector-typed `add_word` overload; `clear_word_buffer`.
   - SyntacticLayer: rewrite `_compose_vector_chart` (and any other
     compose path that calls `add_word` per-row) to write to the
     tensor buffer.
   - Outer loop: `wordSpace.syntacticLayer.flush_word_buffer(subspace)`
     after `runBatch`, alongside `dispatch_soft_reset` and
     `post_tick_compact`.
   - Keep the legacy scalar `add_word(int, int, int, ...)` overload
     for tests and for `_compose_activation` which uses different
     plumbing.
   - Run grammar / chart compose / SVO / partition tests. *Blocking
     7.*

4. **§7 — CUDA-graph capture.** Two-stage exploration:
   - 7a. Try `torch.compile(mode="reduce-overhead")` on the brick.
     Set `torch._dynamo.config.cache_size_limit` high enough that
     the warm-up doesn't recompile. Run the streaming AR test on
     CUDA; assert no `cudaMemcpyDtoH` between forward start and
     `optimizer.step` end via `torch.profiler.profile`.
   - 7b. If 7a doesn't capture (graph breaks, dynamic shapes), build
     an explicit `torch.cuda.CUDAGraph` over `runBatch`'s body with
     a static-shape capture warmup pass. Replay path runs on every
     subsequent tick. Bit-equivalence test: 100 steps eager vs
     captured produce identical losses to within fp tolerance.
   - Mark both `@pytest.mark.skipif(not has_cuda)` since macOS / MPS
     can't capture graphs.

5. **§8 — Legacy removal.** Order matters because some removals
   depend on §6:
   - 8a (now): Delete `InputSpace.batch_advances_sentence` (L3).
     Update `test_no_eos_sync.py::test_batch_advances_sentence_is_python_bool`
     to assert the property is *gone*.
   - 8b (now): Delete the `MODEL_DEBUG` `_end_of_stream` paranoia
     assert in any docstrings / comments referencing it (L8).
   - 8c (now): Convert `_end_of_stream` to a `list[bool]` always;
     remove the scalar / tensor branches in `Reset` and
     `dispatch_per_row_reset` (L4). Update the diagnostic test.
   - 8d (after §6c): Drop the `Reset()` zero-arg fallback in
     `_reset_call`; require every space's `Reset` to accept
     `(batch, hard)` (L5). The fallback was added because the
     predecessor handoff didn't migrate every internal Layer; with
     §6c done, every Reset-capable site is under our control.
   - 8e (last): Make cursor universal and remove the non-cursor
     branch in `runEpoch` + `SentenceStreamDataset.__iter__` (L1, L2).
     Numeric-data path goes through a trivial cursor wrapper that
     yields one tensor slab per tick with `hard_eos=[True]*B`.
   - 8f (optional): Keep the `if all(hard_eos): global Reset` fast
     path in `dispatch_per_row_reset` (L6); just rename / comment
     it as the legacy-parity hot path.
   - 8g (last): Remove `_lex_batch`'s `nObj - 1` truncation (L7);
     replace with `assert n_tokens <= nObj` and let the cursor be
     responsible for sizing the slab. Cursor mode already does this
     (`slab_bytes = nObj - 1`).

§6 lands first; §7 depends on §6; §8 mostly depends on §6c (8d, 8e);
8a–8c can land any time after §6a.

---

## Files this handoff will touch

**Code (§6 vectorization):**
- [basicmodel/bin/Language.py](../../bin/Language.py) `WordSpace.stm_residual_microbatch` (§6a) — drop `.item()` early-out.
- [basicmodel/bin/Layers.py](../../bin/Layers.py) `TruthLayer` (§6b) — add `record_many(vecs, mask)`; rewrite append path to be vectorized.
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `SymbolicSpace.forward` (~6700) (§6b) — replace per-cell loop with tensor scatter.
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `SubSpace.__init__` (~2390) (§6c) — register `word_records` / `word_count` buffers.
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `SubSpace.add_word` (~3010) (§6c) — vector-typed overload; keep scalar overload alive for `_compose_activation` and tests.
- [basicmodel/bin/Language.py](../../bin/Language.py) `SyntacticLayer.flush_word_buffer` (new) (§6c).
- [basicmodel/bin/Language.py](../../bin/Language.py) `SyntacticLayer._compose_vector_chart` (~1602) (§6c) — replace `.tolist()` / Python `add_word` with tensor scatter.
- [basicmodel/bin/Models.py](../../bin/Models.py) `runEpoch` outer loop (§6c) — call `flush_word_buffer` once per tick.

**Code (§7 CUDA graphs):**
- [basicmodel/bin/Models.py](../../bin/Models.py) `BasicModel.runBatch` (§7a) — wrap in `torch.compile(mode="reduce-overhead")` when `MODEL_COMPILE=reduce_overhead` env is set.
- [basicmodel/bin/Models.py](../../bin/Models.py) (§7b, fallback) — explicit `CUDAGraph` warmup + capture + replay path.

**Code (§8 legacy removal):**
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `InputSpace.batch_advances_sentence` — delete (L3).
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `_reset_call` helper — drop the `TypeError` fallback; require new signature (L5).
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `_end_of_stream` — make it `list[bool]` always (L4).
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `_lex_batch` — replace `min(len(stream), nObj - 1)` with assert (L7).
- [basicmodel/bin/data.py](../../bin/data.py) `SentenceStreamDataset.__iter__` — delete (L1).
- [basicmodel/bin/Models.py](../../bin/Models.py) `runEpoch` — delete the non-cursor branch (L2). Numeric inputs flow through a 1-tick cursor wrapper.
- [basicmodel/bin/Models.py](../../bin/Models.py) `dispatch_per_row_reset` — keep the `all(hard_eos)` fast-path; comment it as the legacy-parity hot path (L6).

**Tests (new):**
- `basicmodel/test/test_brick_no_sync.py` (§6) — `torch.profiler.profile` over one tick of `runBatch` on CUDA; assert zero `cudaMemcpyDtoH` between forward start and `optimizer.step` end. Marked `@pytest.mark.skipif(not has_cuda)`.
- `basicmodel/test/test_word_buffer_flush.py` (§6c) — feed a controlled grammar derivation through chart compose; assert `subspace.word` after `flush_word_buffer` matches the legacy per-cell `add_word` output entry-for-entry.
- `basicmodel/test/test_truth_layer_record_many.py` (§6b) — `record_many(vecs, dots)` on a `[B*K, D]` input scales by `dots`; low-DoT rows produce a near-zero vec that the codebook nearest-neighbor matches to the existing zero prototype (codebook size unchanged); high-DoT rows match expected prototypes; same final store contents as a reference Python implementation that loops `record(vec * dot)`.
- `basicmodel/test/test_stm_residual_no_sync.py` (§6a) — profiler trace asserts no `.item()` event in the `stm_residual_microbatch` call.
- `basicmodel/test/test_cuda_graph_capture.py` (§7) — bit-equivalence test for the captured-graph replay path. Marked `@pytest.mark.skipif(not has_cuda)`.
- `basicmodel/test/test_cursor_universal.py` (§8e) — numeric-data path (XOR, MNIST) flows through `next_tick`; one tick == one batch; `hard_eos=[True]*B` always.

**Tests (existing — gates):**
- All from the predecessor handoff.
- `basicmodel/test/test_streaming_ar_training.py`, `test_serial_mode_integration.py`, `test_mental_model.py`, `test_grammar_derivation.py`, `test_grammar_cfg_dispatch.py`, `test_compose_chart.py`, `test_partition_reconstruction_stack.py`, `test_per_batch_stack_isolation.py`, `test_discourse_per_batch_snapshot.py`, `test_discourse_space.py`, `test_truth_layer*` (whatever exists), `test_svo_extraction.py`, `test_invertibility.py`, `test_hierarchical.py`.
- `test_data_no_byte_loss.py`, `test_per_row_hard_reset.py`, `test_soft_reset_at_sentence.py`, `test_grammar_start_rule.py`, `test_grammar_sugar_rules.py` (predecessor handoff).
- `test_no_eos_sync.py` — update once `batch_advances_sentence` is deleted.
- `test_reset_at_batch_boundary.py` — update for the cursor-universal path.

**Docs (updated as part of this handoff):**
- [basicmodel/doc/Architecture.md](../Architecture.md) — pipeline-as-unit + brick contract: now sync-free, CUDA-graph-capturable.
- [basicmodel/doc/Spaces.md](../Spaces.md) — `Reset(batch, hard)` is required signature; `_end_of_stream: list[bool]`; `_lex_batch` byte cap is assert-not-truncate.
- [basicmodel/doc/Language.md](../Language.md) — `subspace.word` is now tensor-backed inside the brick, materialized post-tick.

---

## Verification

### Unit tests (new) — see "Tests (new)" above.

### Performance gates

On a CUDA host with bf16:
```bash
MODEL_AMP=bf16 MODEL_COMPILE=reduce_overhead \
basicmodel/.venv/bin/python basicmodel/bin/train.py \
    --model basicmodel/data/MM_5M.xml --data text --num-epochs 1
```

Pass criteria:
- `torch.profiler` over 100 ticks reports zero `cudaMemcpyDtoH` events
  between `forward()` start and `optimizer.step()` end.
- `nvidia-smi dmon` shows `pwr` consistent ≥ predecessor-handoff baseline.
- Wall-clock per tick drops vs predecessor by ≥ 1.5× (kernel launch
  overhead reduction from graph replay).

### Bit-equivalence (CUDA only)

100 ticks captured-graph replay vs. eager produce per-position-loss
arrays identical to within `1e-4` (bf16 tolerance). Larger drift
indicates a graph-capture bug.

### Legacy-removal gates

After §8 lands:
- `grep -rn "batch_advances_sentence" basicmodel/` returns zero hits
  outside `doc/plans/`.
- `grep -rn "def __iter__" basicmodel/bin/data.py` returns zero hits.
- `grep -rn "_end_of_stream" basicmodel/bin/` shows only the
  `list[bool]` form; no `torch.is_tensor(eos)` branches.
- Every `Reset` definition in `basicmodel/bin/` accepts
  `(self, batch=None, hard=True)`.
- `runEpoch` has one branch, not two.

---

## Risks and known unknowns

1. **Compose vectorization correctness.** The chart compose path mixes
   tensor-scored merges with Python-side trace recording
   (`self._derivation_trace[b].append(...)`). Path B keeps the trace
   list as Python — only the `subspace.word` mutation moves to tensor.
   The trace is an off-brick consumer (SVO walker reads it after
   compose returns), so it doesn't need to be sync-free. Verify the
   SVO walker still works.

2. **`torch.compile` graph breaks on data-dependent control flow.**
   The chart compose has `if compat.sum() == 0: break` and
   `if pair_tensor.shape[1] == 0: break`. These are conditional on
   tensor values and will break the graph. Two options:
   (a) replace with masked-zero (do max_depth iterations always,
       mask out done rows),
   (b) accept the graph break and fall back to per-doc graph segments.
   The plan defers this to §7b's measurement.

3. **`record_many` storage growth.** Today's `TruthLayer` appends one
   entry at a time and compacts after the tick. The vectorized path
   needs to either preallocate a max-size buffer (memory cost) or
   handle dynamic resize on append (potential graph break). The
   `compact()` already runs out-of-brick so the buffer can grow within
   a tick and shrink at the post-tick compact.

4. **§8e (cursor universal) for numeric data.** XOR with numeric
   labels and MNIST images don't have a "byte stream" model. The
   1-tick cursor wrapper has to handle that: doc = whole tensor,
   `hard_eos=[True]*B` every tick, slab_shape = whatever the original
   tensor row was. This is straightforward but touches every dataset
   loader path.

5. **CUDA-graph capture on DGX GB10 (Grace + Blackwell).** Native
   CUDA path; `torch.compile(mode="reduce-overhead")` and explicit
   `torch.cuda.CUDAGraph` both work without ROCm caveats. Blackwell
   adds fp8 autocast as an additional knob — out of scope for this
   handoff but worth noting once the brick is graph-capturable.

6. **MPS / Metal does not support graph capture.** §7 is skipped on
   Mac. The brick contract from §6 still applies (sync-free is good
   regardless of capture availability), and MPS still benefits from
   reduced launch overhead via fused kernels under inductor.

---

## Sidebar: pipeline-package considerations

Pipeline parallelism (`torch.distributed.pipeline.sync.Pipe`,
FSDP-PP, DeepSpeed-PP) is GPU-count parallelism — it splits the model
into stages on separate devices and overlaps stage-i forward of
microbatch t+1 with stage-i+1 forward of microbatch t. Two relevant
observations:

1. **MPS / single-device architectures (Mac, GB10) don't benefit.**
   Apple Silicon's GPU is one device from PyTorch's perspective; many
   cores are SIMD/tensor-parallel inside that one device, not pipeline
   boundaries. GB10 is similarly a single Blackwell device. The cost
   levers that matter on these targets are kernel-launch reduction
   (`torch.compile` + CUDA graphs), memory-pressure relief
   (activation checkpointing), and AMP — none of which involve PP.

2. **Multi-device clusters are where PP earns its keep.** Once the
   model outgrows single-device memory (typically >100M params) and
   spans ≥2 devices, PP + TP + FSDP/ZeRO are the standard axes. At
   MM_5M's 5M-param scale PP would be net-negative because per-stage
   compute is too small to amortize cross-stage activation transfers.
   Revisit when scaling up the model itself, not as part of this
   handoff.

For the immediate roadmap: §6 + §7 are the right speedup levers on
the GB10 target. Pipeline parallelism is an axis to revisit alongside
a model-size scale-up, not as a brick-level optimization.

---

## What is NOT in this handoff

- **Architectural changes to the AR window (sliding-window stem,
  K = T cap).** The window stays; this handoff only changes what
  happens inside the body.
- **Discourse / sentence-prediction redesign.** The soft-reset
  contract from the predecessor handoff is now correct; further
  changes are out of scope.
- **A second-tier mask design for cross-slab byte context.** The
  user has confirmed the cursor + structural-prior bridge is the
  intended design.
- **Switching to BPE / subword tokenization.** Byte lex stays. Token
  efficiency is a separate concern.
- **Numeric data redesign.** §8e absorbs numeric data into the cursor
  wrapper but does not change how MNIST / XOR are represented.
