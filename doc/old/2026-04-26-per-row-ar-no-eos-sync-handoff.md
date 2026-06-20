# Handoff: Per-row AR — drop EOS sync, propagate masks correctly

**Date.** 2026-04-26
**Owner.** Alec
**Status.** Not started.

---

## Context

**The problem.** During AR training of MentalModel (`MM_5M.xml`, `<maskedPrediction>ARIR</maskedPrediction>`), the GPU on a DGX Spark sits at ~32 W while `nvidia-smi dmon` reports `sm = 80–96%` and `mem = 0%`. That signature is **Python-orchestration-bound**: the SMs activate for microsecond bursts then idle waiting for the next launch. Real compute is around 15–20% of what the chip can do.

The smoking gun is at [bin/Models.py:2436-2443](../../bin/Models.py):

```python
eos = self.inputSpace._end_of_stream
if torch.is_tensor(eos):
    eos = bool(eos.all().item())                # ← per-batch GPU sync
if (not is_ar_mode) or eos:
    for space in self.spaces:
        if hasattr(space, 'Reset'):
            space.Reset()                       # ← global wipe
```

The `.all().item()` reduction forces a host sync every batch, and the global Reset cascade prevents CUDA-graph capture of the AR step.

**The architecture as it actually works** (verified during planning):

- **One sentence per batch row.**  [bin/data.py:153-170](../../bin/data.py) `SentenceStreamDataset.__iter__` yields `[inputs[b * L + t] for b in range(B)]` per timestep — one document per row, batch rectangular.
- **K = max sentence length in the batch.**  [bin/Spaces.py:4613](../../bin/Spaces.py) `K = T` where T is the embedded buffer width set by the lex pass to the longest sentence in the batch.
- **NULL-padded shorter sentences.**  [bin/Spaces.py:4616-4617](../../bin/Spaces.py): `valid_mask = (embedded.abs().sum(-1) > 0)`. Shorter sentences have `valid_mask_bk[b, k] = False` at their tail.

So under the existing setup, EOS at batch boundary is **deterministic** — host-side, no GPU reduction needed. The `.all().item()` sync is checking something the host already knows. Removing it requires only a small code change once we verify state mutations honor the per-cell mask.

**The intended outcome.**

1. Eliminate the per-batch `.item()` sync at the EOS gate.
2. Make every per-cell state mutation in the forward pass honor `valid_mask_bk` so a NULL-padded cell is a true no-op (no parser advance, no codebook write, no cache update, no loss contribution).
3. Verify the per-row state model (the parse stack `_top` etc.) is consistent with K-serial state accumulation within a sentence.

Expected effect on hardware: GPU power should climb from ~32 W toward 100+ W; `mem%` becomes nonzero in `dmon`. Per-epoch wall time should drop materially.

---

## What is already correct (do not re-implement)

- **`valid_mask_bk: [B, K]` per-cell validity mask** — [bin/Spaces.py:4622](../../bin/Spaces.py). Already encodes "is (sentence b, position k) a real token?". Loss path consumes it at [bin/Models.py:2188-2196](../../bin/Models.py). **Reuse this signal everywhere.**
- **`_end_of_stream: [B]` bool** — [bin/Spaces.py:4626](../../bin/Spaces.py). Already per-row. Currently consumed via `.all().item()` for host control flow; the fix is to stop reading it for control flow, not to reshape it.
- **`WordSpace._top, _svo_valid, _stm_fired`** — [bin/Language.py:2129, 2392, 2398](../../bin/Language.py). Already per-row tensors with per-row helpers (`arm_stm(b)`, `clear_last_svo(b)`).
- **Existing per-batch `Reset()` cascade** ([bin/Spaces.py](../../bin/Spaces.py) at 2407, 2619, 2975, 3075, 4525; [bin/Language.py:2871](../../bin/Language.py)). Stays as the global Reset; just gets called unconditionally at AR-batch end instead of being gated on a tensor reduction.

---

## What this handoff changes

### 0. Verify K-serial state accumulation (READ-ONLY pre-step)

Before any code changes, **trace `WordSpace.ensure_microbatch` at [bin/Language.py:2913](../../bin/Language.py)** and the FlattenKWrapper path at [bin/Pipeline.py:102-144](../../bin/Pipeline.py). Determine:

- Does `ensure_microbatch(B, K)` resize per-row state (`_top`, `_svo_valid`, `_stm_fired`, the parse stack) to shape `[B*K]`, or does it keep state at `[B]`?
- Under FlattenKWrapper's `[B, K, N, D]` → `[B*K, N, D]` reshape, which body operations carry state across K positions for a given source row, and which treat each (b, k) cell as independent?

This determines the correct interpretation of the per-cell mask:

- **If state is `[B*K]` (per-cell independent):** each (b, k) cell stands alone. Per-cell masking via `valid_mask_bk` flattened to `[B*K]` is sufficient. NULL-padded cells get masked at every state-mutating op. **No reshape change needed.**
- **If state is `[B]` and accumulates K-serially within a row:** the `[B*K, N, D]` flatten is incompatible with that semantics. Switch the per-percept flatten in serial mode to `[B, N*K, D]` so PerceptualSpace's `[B, 1, D]` emissions accumulate along a per-B time axis. This is the **flatten-dim change** under discussion: necessary if and only if the K-serial state-accumulation semantics is the intended one.

Document the finding at the top of the implementation commit. If state is `[B*K]`, skip §1.5 below; if state is `[B]` and K-serial, do §1.5.

### 1. Remove the per-batch EOS sync

**File.** [bin/Models.py:2436-2443](../../bin/Models.py).

**Change.** In AR mode, drop the `.all().item()` reduction entirely. With one sentence per batch row, the loader knows host-side that every batch is "this sentence's K positions for each row," and the next batch starts fresh sentences for every row. So `Reset()` (global) fires unconditionally at AR-batch end:

```python
# Old:
eos = self.inputSpace._end_of_stream
if torch.is_tensor(eos):
    eos = bool(eos.all().item())
if (not is_ar_mode) or eos:
    for space in self.spaces:
        if hasattr(space, 'Reset'):
            space.Reset()
    self.inputSpace._end_of_stream = False

# New:
if not is_ar_mode or self.inputSpace.batch_advances_sentence():
    for space in self.spaces:
        if hasattr(space, 'Reset'):
            space.Reset()
    # Don't read _end_of_stream as Python bool. If MODEL_DEBUG is on,
    # do an after-the-fact assertion that _end_of_stream agrees with
    # the loader's host-side expectation; otherwise leave it alone.
```

Where `batch_advances_sentence()` is a host-side `@property` on InputSpace that returns `True` whenever the data loader has yielded a fresh batch (which under the simple-version contract is every AR batch). For the simple version, this can be a no-op constant `True` — every AR batch is a sentence boundary by construction. The `MODEL_DEBUG` assertion path runs `_end_of_stream.all().item()` only under the debug flag.

### 1.5. (Conditional, only if §0 found per-`[B]` K-serial state) Switch serial-mode flatten to `[B, N*K, D]`

**File.** [bin/Pipeline.py:102-144](../../bin/Pipeline.py) FlattenKWrapper, plus the cold-path entry in [bin/Spaces.py:5424+](../../bin/Spaces.py) PerceptualSpace.forward.

**Change.** Under the K-serial-state interpretation, the `[B, K, N, D]` → `[B*K, N, D]` reshape is wrong because it loses per-row state continuity across K. Replace with `[B, K, N, D]` → `[B, K*N, D]` (or `[B, N*K, D]` with whatever ordering the body expects). PerceptualSpace's `[B, 1, D]` per-step emission is then the natural unit; the body sees a per-B sequence whose layers can accumulate state per row.

This is a significant downstream change for any layer that depends on the FlattenKWrapper output shape. Audit the body sequence (`[bin/Models.py](../../bin/Models.py) self._body_inner`) for consumers of the flattened input.

**Skip this section if §0 finds state is `[B*K]` (per-cell independent).**

### 2. Per-cell mask propagation through state-mutating ops

**Files.** Multiple — see touch list below.

**Change.** Every per-cell state mutation in the forward chain must skip cells where `valid_mask_bk` is False. The mask must be threaded through `forward(...)` calls (or read from `subspace.valid_mask_bk` which is already attached at [bin/Spaces.py:4622](../../bin/Spaces.py)).

Specific call sites to gate:

- **PerceptualSpace.forward** ([bin/Spaces.py:5424+](../../bin/Spaces.py)) — chunk-layer state, `_bpe_word_mask` update.
- **PerceptualSpace serial warm path** ([bin/Spaces.py:5402-5421](../../bin/Spaces.py)) — the `rolled[:, -1, :] = new_out[:, 0, :]` write at line 5413 should skip rows where the current (b, k) cell is invalid. Pattern: `rolled[active_rows, -1, :] = new_out[active_rows, 0, :]` or `where(active, new_out, old_cache)`.
- **ConceptualSpace.forward** ([bin/Spaces.py:5888+](../../bin/Spaces.py)) — VQ codebook usage updates, sparsity regularizer accumulation.
- **SymbolicSpace.forward** ([bin/Spaces.py:6700+](../../bin/Spaces.py)) — codebook usage updates, any per-cell write into the symbolic substrate.
- **WordSpace.forward** ([bin/Language.py](../../bin/Language.py)) — parse-stack push at the per-cell rule-application path. STM-arming, SVO update, discourse history append. Each must `where(valid, new_state, old_state)` or skip via index gather of valid cells only.

**Pattern.** For state-mutating tensor ops on `[B*K, ...]` shape (post-flatten), use:

```python
mask = subspace.valid_mask_bk.flatten()           # [B*K] bool
# instead of: state = state.scatter_add(0, indices, updates)
state = state.scatter_add(0, indices[mask], updates[mask])
```

For Python-list / dict mutations (e.g. WordSpace's per-row token append): gate the loop body on the per-row aggregate (`if any cell in this row is valid`) or — preferably — re-shape the operation to a tensor pattern and use the mask uniformly.

### 3. Reset is unconditional at AR-batch end

**File.** [bin/Models.py:2436-2443](../../bin/Models.py).

**Change.** Per §1, the `Reset()` cascade fires at every AR batch boundary — no host-side decision needed. The Reset itself is unchanged (it's the existing global cascade). Only the gate changes from "tensor reduction sync + branch" to "host-side property check + branch."

No `Reset(batch=b)` API is needed under the one-sentence-per-batch-row contract: every row finishes its sentence at the same batch boundary by construction. Surgical per-row reset would only matter if we adopted a rolling-sentences design, which is explicitly out of scope.

---

## Files this handoff will touch

**Code:**
- [bin/Models.py](../../bin/Models.py) lines 2436-2443 — drop EOS sync; unconditional Reset at AR-batch end.
- [bin/Spaces.py](../../bin/Spaces.py) PerceptualSpace.forward (~5424+) and serial warm path (5402-5421) — propagate `valid_mask_bk` to chunk-layer state and serial_cache write.
- [bin/Spaces.py](../../bin/Spaces.py) ConceptualSpace.forward (~5888+) — gate codebook / sparsity updates on per-cell mask.
- [bin/Spaces.py](../../bin/Spaces.py) SymbolicSpace.forward (~6700+) — gate codebook / sortNetwork updates on per-cell mask.
- [bin/Language.py](../../bin/Language.py) WordSpace.forward — gate parse-stack push, STM, SVO, discourse on per-cell mask.
- [bin/Pipeline.py:102-144](../../bin/Pipeline.py) FlattenKWrapper — only changes if §0 finds per-`[B]` state needing K-serial accumulation; else leave alone.

**No data-loader changes.** The simple version assumes the existing per-batch contract.

**Optional:**
- Add an `InputSpace.batch_advances_sentence()` host-side property that returns `True` (per the simple contract) — placeholder for future per-row rolling logic if ever needed.

---

## Verification

### Unit tests (new)

- `basicmodel/test/test_no_eos_sync.py`:
  - Run one AR batch under `torch.profiler.profile(...)` accounting.
  - Assert zero `cudaMemcpyDtoH` events between `optimizer.step` and the next loader fetch in the recorded trace.
  - Confirms §1 removed the per-batch sync.

- `basicmodel/test/test_padded_rows_no_op.py`:
  - Build a B=2 AR batch where row 0 is a 5-token sentence and row 1 is a 12-token sentence (so K=12, row 0 has positions 5..11 NULL-padded with `valid_mask_bk[0, 5:] = False`).
  - Snapshot all per-row state (parse stack `_top`, codebook usage counters, serial cache rows for row 0, etc.) immediately before the K-positions-5..11 portion of the body runs.
  - Run the body.
  - Assert row 0's per-row state is bit-identical pre/post — the NULL-padded cells made no contribution.
  - Assert row 1's state advanced normally.
  - Confirms §2 mask propagation.

- `basicmodel/test/test_reset_at_batch_boundary.py`:
  - Run two consecutive AR batches.
  - Between them, assert that `Reset()` was called and that all per-row state for both rows is cleared at the start of batch 2.
  - Confirms §3 unconditional Reset.

### Existing-test gates

All of the following must remain green after changes:

```
basicmodel/test/test_streaming_ar_training.py
basicmodel/test/test_serial_mode_integration.py
basicmodel/test/test_mental_model.py
basicmodel/test/test_pi_sigma_ownership.py
basicmodel/test/test_conceptual_bivector.py
basicmodel/test/test_grammar_derivation.py
basicmodel/test/test_grammar_cfg_dispatch.py
basicmodel/test/test_hierarchical.py
basicmodel/test/test_invertibility.py
```

Run with:
```bash
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_streaming_ar_training.py basicmodel/test/test_serial_mode_integration.py basicmodel/test/test_mental_model.py basicmodel/test/test_pi_sigma_ownership.py basicmodel/test/test_conceptual_bivector.py basicmodel/test/test_grammar_derivation.py basicmodel/test/test_grammar_cfg_dispatch.py basicmodel/test/test_hierarchical.py basicmodel/test/test_invertibility.py
```

### End-to-end hardware verification (the actual hypothesis test)

On a DGX Spark with BF16:

```bash
MODEL_AMP=bf16 python bin/train.py --model data/MM_5M.xml --data text --num-epochs 1
```

In a parallel shell:
```bash
nvidia-smi dmon -s pucvmet -c 60
```

Observe:
- `pwr` climbs from ~32 W to >80 W (target ~150 W).
- `mem` becomes nonzero (>10%).
- Wall-clock per epoch drops 3–10×.

Loss-curve regression: 1-epoch loss curve under the new design tracks the old design within ~5% at every checkpoint. (Loss need not be bit-identical because state-mutation gating changes the order of codebook updates etc., but training dynamics should not regress.)

If `pwr` does not climb, the Python-orchestration-bound hypothesis is wrong (or another sync is the dominant cost). In that case, profile and identify the next bottleneck before declaring the work complete.

---

## Risks and known unknowns

1. **`ensure_microbatch` semantics (the §0 question).** The plan branches based on whether per-row state is `[B*K]` (per-cell independent) or `[B]` (K-serial accumulating). If the body's actual behavior is more nuanced (some state per `[B]`, some per `[B*K]`), the plan needs per-state-attribute treatment, not a global decision. Trace carefully before committing to §1.5 yes/no.

2. **`serial_cache` aliasing.** Cache entries at [bin/Spaces.py:2409](../../bin/Spaces.py) are keyed on `id(self)` and shaped `[B, N, D]`. Per-cell mask gating in §2 requires the cache write at line 5413 to know which (b, k) cell it's writing for. If the warm path is only called per-AR-step (not per K-cell), this is per-row gating only, and `valid_mask_bk[:, k]` at the current step k suffices.

3. **Loss-curve drift.** Bit-identical loss curves are not guaranteed because state-update ordering changes when invalid cells are masked. Within ~5% drift is acceptable; larger drift suggests a state-mutation site was missed in §2.

4. **CUDA-graph capture is the *next* step, not this one.** This handoff removes the per-batch sync but does not introduce CUDA graphs. Once the sync is gone, a follow-up commit can attempt `torch.compile(mode="reduce-overhead")` or explicit `torch.cuda.CUDAGraph` capture of the AR step. That work is out of scope here; flag it as a follow-up.

---

## Sequencing

1. **§0 verification first** — read `ensure_microbatch` and the FlattenKWrapper path; document whether state is `[B*K]` or `[B]`. Decide whether §1.5 is in scope.
2. **§1: drop the EOS sync** — smallest blast radius. Land it. Run the test sweep. Confirm no regression.
3. **§2: mask propagation** — work outward from PerceptualSpace through ConceptualSpace, SymbolicSpace, WordSpace. After each layer, re-run the test sweep. Bit-equivalence is the existing-test gate; new-test bits are gated by the §2 unit tests.
4. **§1.5 if applicable** — flatten reshape change. Highest blast radius; isolate as its own commit.
5. **§3 verification at AR-batch boundary** — confirm Reset is firing. Should be trivial since the cascade itself is unchanged.
6. **End-to-end hardware verification on DGX Spark** — measure GPU power, mem%, wall-clock. This is the success criterion, not the test sweep.

All five steps land in a single PR or series of small PRs (preferred: one per §). Each is independently revertible.

---

## What is NOT in this handoff

- **`Reset(batch=b)` API.** Unnecessary under the one-sentence-per-batch-row contract.
- **Per-row rolling-sentences data loader.** Considered; rejected as out of scope for the simple version. Would be required only if measured `valid_mask_bk == False` waste exceeds ~30%.
- **CUDA graph capture.** A natural follow-up once the per-batch sync is gone, but a separate commit.
- **Architectural changes to PiLayer / SigmaLayer / Ops.*.** Unrelated to the sync issue.
- **Data-loader changes.** None. The existing `SentenceStreamDataset` contract is preserved.
