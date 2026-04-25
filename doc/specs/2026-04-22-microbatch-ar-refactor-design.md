# Integrated Refactor: Per-Batch State + Stem/Body/Head + Microbatch AR

## Context

Three interlinked bugs/refactors must ship together as one cutover:

1. **Batch-handling bug.** Six pieces of state currently single-sample, shared across a batch: `wordSpace`/`error`, `pos_stack`, `reconstruction_stack`, `last_svo`, `_stm_fired`, and discourse `recent`/`prev_centroids`/`counts`. Each needs a leading `[B]` dimension. Today `ensure_batch(B)` exists on `WordSubSpace` (Spaces.py:3647) but not on these stacks/flags/buffers, so at `B>1` each batch element corrupts the others' state.

2. **Pipeline structure.** Today `pipeline_fwd = nn.Sequential(perceptualSpace, conceptualSpace, symbolicSpace, symbol_cache, outputSpace)` (Models.py:1507) with InputSpace driving a serial AR while-loop outside it (Models.py:1763). The single chain blurs the boundary between stateful ingest, pure transforms, and output projection. Reorganizing into stem/body/head makes the execution regions explicit and creates the seam the pipelining work needs.

3. **Microbatch AR.** Training currently serializes AR: one position per outer iteration through the whole pipeline. With teacher forcing, we already know all positions, so we can produce `[B, K, N, D]` windows in one stem call and let body/head process the flattened `[B*K, N, D]` in parallel. This gives distributed pipelining enough microbatches to overlap compute.

**User directional choices (confirmed):** one integrated plan; stem = `[inputSpace, perceptualSpace]` running once per step with `K` windows emitted from a single call; tensor-first `[B, ...]` state representation; microbatch replaces the serial AR while-loop — no config flag, no legacy branch kept.

**Why no legacy coexistence here:** the microbatch path is strictly more general. Teacher-forced training becomes a single forward (no loop); inference is a degenerate `K=1` microbatch driven by the external ARIR generation loop. The while-loop at Models.py:1761-1792 has no role left and is removed outright. (This diverges from the usual "keep legacy alongside new" guidance because the user explicitly directed removal for this refactor.)

## Target Architecture

**Three pipelines, assembled in `build_pipelines()` (Models.py:1472):**

```python
self.pipeline_stem = nn.Sequential(self.inputSpace, self.perceptualSpace)
self.pipeline_body = nn.Sequential(self.conceptualSpace, self.symbolicSpace, self.symbol_cache)
self.pipeline_head = nn.Sequential(self.outputSpace)
self.pipeline_fwd  = nn.Sequential(self.pipeline_stem, FlattenKWrapper(self.pipeline_body), self.pipeline_head)
```

The old assembly at Models.py:1488-1519 is replaced by this. `pipeline_rev` and `pipeline_rt` (round-trip) rebuild analogously across the three groups.

**Stem output contract.** Stem runs once per step and returns a single subspace whose event is `[B, K, N, D]`. A new `SubSpace.k_axis: bool` flag (default False) signals the K-extended shape. This preserves the uniform `forward(subspace) -> subspace` contract and keeps `copy_context` (Spaces.py:2393) intact. Returning `K` separate subspaces would fork every downstream API — rejected.

**Body consumption of `[B*K, N, D]`.** A new `FlattenKWrapper(nn.Module)` in Pipeline.py wraps `pipeline_body`. On forward: read `[B, K, N, D]` event, `view(B*K, N, D)`, invoke inner body, restore `[B, K, N, D']`. Reshape is contained entirely in the wrapper; `model.forward` stays clean and autograd handles the back-view automatically. `ensure_batch(B*K)` is called on body-side state (wordSpace, stacks, discourse) since each of the `B*K` rows needs isolated state. Each flattened row `i ∈ [0, B*K)` has origin-batch index `b = i // K`, used for `_stm_fired` gating and discourse snapshotting.

**Head and loss.** Head runs on unreshaped `[B, K, N, D']` and produces `[B, K, N, predDim]`. `runBatch` (Models.py:2108-2156) flattens to `[B*K, N, predDim]` for the existing loss code; `_ar_valid_pos` becomes `[B, K, N]` built by the stem.

## Per-Batch State Design

| Location | File:line | New representation |
|---|---|---|
| `PoSStack` | Language.py:1777-1799 | `_entries: [B, max_depth, 4]`, `_top: [B] long`; push/pop via scatter/gather |
| `ReconstructionStack` | Language.py:1802-1822 | `_entries: [B, max_depth, 2] long`, `_top: [B] long` |
| `last_svo` | Language.py:332, 2023 | `[B, 3, svo_dim]` tensor + `_svo_valid: [B] bool` (replaces `None`) |
| `_stm_fired` | Language.py:2027 | `[B] bool`; `stm_residual` vectorizes |
| Discourse `_recent`/`_prev_centroids`/`_recent_count`/`_prev_count` | Layers.py:3230-3250 | Leading `[B, ...]`; drop mean-pool at Layers.py:3293; add `ensure_batch(B)` on `InterSentenceLayer` |
| `wordSpace.errors` | Spaces.py:2385-2386 | No structural change. Error terms become `[B]` tensors through upstream writers; `.total()` mean-reduces. `Error()` is already a batched accumulator once callers put tensors in. |

State under microbatch: sized `B*K` inside body (stack push/pop is row-local on the flattened dim). `_stm_fired` is `[B]` with origin-mapping `b = i // K` inside body gates. Discourse snapshots still fire at sentence boundaries (Spaces.py:4572 `_end_of_stream`) — not per-window — so centroid history stays at `[B, ctx, rows, dim]`.

## AR via Microbatching (No Config, No Loop)

**K is always auto-derived.** `K = T - N + 1` where `T` is the lexed-sequence length and `N = inputSpace.outputShape[0]` (the context window). No config key. No master flag.

**Window construction.** `InputSpace.forward()` is rewritten: reuse `_lex_and_embed` (Spaces.py:4552) to get `_ar_embedded: [B, T, D]`, then `tensor.unfold(1, N, 1)` → `[B, K, N, D]` (prefer `unfold` over `as_strided` — autograd-safe). Build `[B, K]` validity mask from NULL-sentinel detection (vectorized version of Spaces.py:4568-4583). Emit a single subspace with `k_axis=True` whose event is `[B, K, N, D]`.

**Inference (K=1 degenerate case).** At ARIR inference, each external step feeds the current buffer as one window: `T = N`, so `K = 1` and the emitted shape is `[B, 1, N, D]`. The `FlattenKWrapper` reshapes `[B, 1, N, D]` → `[B, N, D]` — a no-cost view. Head restores `[B, 1, N, predDim]`. The caller consumes one prediction per `forward()` call; cross-call state rides on `_ar_buffer` (Spaces.py:4306) unchanged.

**`Models.forward()` (replacing Models.py:1734-1804).** The entire while-loop is deleted. The new body is straight-line:

```python
def forward(self, inputData):
    if isinstance(inputData, torch.Tensor):
        inputData = inputData.to(TheDevice.get())
    self.inputSpace.masked_prediction = self.masked_prediction
    result = self.pipeline_fwd(...)          # stem -> FlattenKWrapper(body) -> head
    pred = result.materialize(...)           # [B, K, N, predDim]
    self.inputs = self.pipeline_stem[0].subspace  # last stem subspace (for legacy callers)
    # bookkeeping: increment nWhere, denormalize, etc.
    return last_input_state, symbols, pred, None
```

No `is_ar_mode` branch (AR is always on via K). No `is_runtime_arir` break (K=1 handles that naturally). The `predictions` list accumulation in the old loop becomes the K axis of the output tensor — no `torch.stack` needed post-hoc.

## Files to Modify

| File | Change | Lines |
|---|---|---|
| `basicmodel/bin/Spaces.py` | Rewrite `InputSpace.forward` to emit `[B, K, N, D]` via `unfold`; add `SubSpace.k_axis` flag; verify `WordSubSpace.ensure_batch` at B*K scale | 3647, 4306, 4478-4605 |
| `basicmodel/bin/Language.py` | Tensor-ize `PoSStack`/`ReconstructionStack`; `last_svo` `[B, 3, D]`; `_stm_fired` `[B]`; 10+ reader/writer sites | 332, 1010, 1245-1249, 1433, 1491-1517, 1777-1822, 1946-1950, 2023-2050, 2382-2391 |
| `basicmodel/bin/Layers.py` | `InterSentenceLayer` per-B buffers; drop mean-pool in `_fit_rows`; add `ensure_batch`; per-B `snapshot`/`_recent_centroid` via gather | 3230-3302, 3323-3361 |
| `basicmodel/bin/Models.py` | `build_pipelines` assembles stem/body/head; replace `forward()` (delete while-loop); loss handles K in `runBatch`; same for MentalModel | 1472-1520, 1734-1804, 2108-2156, 3148-3210 |
| `basicmodel/bin/Pipeline.py` | Add `FlattenKWrapper(nn.Module)` | end of file |

## Critical Files

- `basicmodel/bin/Models.py` — `build_pipelines`, `forward`, `runBatch`
- `basicmodel/bin/Spaces.py` — `InputSpace`, `WordSubSpace.ensure_batch`, `SubSpace`
- `basicmodel/bin/Language.py` — `PoSStack`, `ReconstructionStack`, `SyntacticLayer.last_svo`, `WordSpace._stm_fired`
- `basicmodel/bin/Layers.py` — `InterSentenceLayer`, `Error`
- `basicmodel/bin/Pipeline.py` — glue modules, target for `FlattenKWrapper`

## Implementation Order

Steps 1-3 are additive (no pipeline changes) — existing tests stay green. Steps 4-10 land together as the pipeline cutover; during this window the while-loop is torn out and the microbatch path takes over. Step 11 is the correctness gate before committing.

1. Tensor-backed `PoSStack`/`ReconstructionStack` (API preserved). Per-B isolation tests at B=2. [Language.py]
2. `last_svo` → `[B, 3, D]` + `_svo_valid`; `_stm_fired` → `[B]` bool. Audit all readers via Grep. [Language.py]
3. Discourse buffers → `[B, ctx, rows, dim]`; `ensure_batch` on `InterSentenceLayer`; drop mean-pool. [Layers.py]
4. `FlattenKWrapper` in Pipeline.py; isolated round-trip + backward tests (before wiring into pipeline).
5. Rewrite `InputSpace.forward` to emit `[B, K, N, D]` via `unfold`; add `SubSpace.k_axis`; `[B, K]` validity mask. Handles `K=1` degenerate case for inference.
6. `build_pipelines` assembles stem/body/head; replace the old `pipeline_fwd`; update `pipeline_rev`/`pipeline_rt` analogously.
7. Rewrite `Models.forward`: delete the while-loop at 1761-1792; straight-line stem → `FlattenKWrapper(body)` → head.
8. `runBatch` loss handles K axis; flatten `[B, K, N, predDim]` → `[B*K, N, predDim]` for the existing loss call; build `_ar_valid_pos` as `[B, K, N]`; scale per-token mean.
9. `ensure_batch(B*K)` cascade at `Start()` — wordSpace, stacks, discourse sized to `B*K`. `_stm_fired` stays `[B]` with origin mapping `b = i // K` at read sites inside body.
10. Apply the same `forward()` rewrite to `MentalModel` (Models.py:3148-3210).
11. Equivalence test: before-cutover capture of predictions on a fixed-seed tiny config; after cutover must match within 1e-5.
12. Run `make sync_local` and verify on ArborMini (never train locally, per memory).

## Verification

Run under `basicmodel/.venv/bin/python -m pytest basicmodel/test/`:

- **`test_per_batch_state_isolation.py`** — B=2 divergent inputs, state per element distinct across all 6 locations.
- **`test_microbatch_equivalence.py`** — pre-cutover captured predictions vs post-cutover microbatch, match within 1e-5 on a fixed-seed tiny config. Main correctness gate.
- **`test_stem_body_head_shapes.py`** — intermediate shape asserts parameterized over (B, K, N, D); verify K=1 degenerate case works.
- **`test_flatten_k_wrapper_grad.py`** — end-to-end backward identical to manual `view` path.
- **`test_inference_k1_degenerate.py`** — ARIR inference with K=1 produces same token stream as pre-cutover baseline.
- **`test_discourse_per_batch_snapshot.py`** — B=3 with varying sentence ends; `_recent_count` distinct per sample.

End-to-end: real training validation via `make sync_local` then run on ArborMini (never `make train` locally, per memory).

## Non-Goals / Not in This Plan

- Distributed pipelining backend choice (torch.distributed.pipeline vs fairscale) — out of scope; this plan sizes the seam.
- A config flag for microbatching — explicitly rejected by the user. Microbatch is the only path.
- A config for K — always auto-derived from input length and window size.
