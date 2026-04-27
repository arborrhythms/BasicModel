# Handoff: Rolling-cursor document streaming with two-tier reset

**Date.** 2026-04-26
**Owner.** Alec
**Status.** Not started.

**Predecessor.** [2026-04-26-per-row-ar-no-eos-sync-handoff.md](2026-04-26-per-row-ar-no-eos-sync-handoff.md) (landed). This handoff *un-rejects* the "per-row rolling-sentences data loader" item that the predecessor explicitly listed as out of scope, and supersedes parts of its Reset gating with a cleaner design.

---

## Context

**The problem.** Today `SentenceStreamDataset` ([basicmodel/bin/data.py:153-170](../../bin/data.py)) yields one whole document per row per AR batch. Documents longer than `nOutput` (= 1024 bytes for [MM_5M.xml](../../data/MM_5M.xml)) are silently truncated by the lex at [basicmodel/bin/Spaces.py:4493-4501](../../bin/Spaces.py) (`n_tokens = min(len(stream), nObj - 1)`). On the FineWeb-EDU corpus (`maxDocs=10000`), that hard truncation drops:

| cap   | docs truncated | % |
|-------|----------------|----------|
| 1024  | 8910 / 10000   | 89.10% |
| 4096  | 3329 / 10000   | 33.29% |
| 8192  | 1116 / 10000   | 11.16% |
| 16384 | 345 / 10000    |  3.45% |

The training run is missing the bulk of long-tail documents. Raising the cap to 8192 helps but still loses 11% of docs; even at 16K we lose 3.5%. Widening the cap further blows up the AR stem's per-batch event tensor (memory grows linearly with K).

**The intended outcome.** Replace the doc-as-atomic-row contract with a per-row byte cursor:

1. `TheData` walks each row's document one ≤1024-byte slab at a time.
2. The pipeline (forward → backward → optimizer.step) becomes a pure compute brick, called repeatedly until all rows have exhausted their docs.
3. Reset is **two-tiered**:
   - **Hard reset** (deterministic, from `TheData`): a row's cursor crosses a document boundary. Wipes all per-row state. Fires *outside* the compute brick.
   - **Soft reset** (structural, from the grammar): a row's parse derivation completes against `<start>`. Re-arms `_stm_fired[b]`, clears `_last_svo[b*K..]`, resets parse stack to start state. Fires *outside* the compute brick (signal accumulated during the brick, dispatched after).
4. Empty-slot handling stays as `valid_mask` propagation (already in place from the predecessor handoff).
5. **No data is skipped.** Concatenating per-tick slabs for any row reproduces the original document byte-exact.

**Expected effects.**
- Coverage: 100% of corpus bytes seen by the model, vs. 11–89% under truncation.
- Compute brick: forward+backward+step is gate-free (no in-loop Reset, no in-loop tensor sync) — preconditions for CUDA-graph capture become real.
- Architectural cleanup: the runBatch tail-Reset block (and the `batch_advances_sentence` property + `MODEL_DEBUG` paranoia assert added by the predecessor) all delete.

---

## FineWeb paragraph metadata: not available

Inspection of `basicmodel/data/fineweb/shard_00000.parquet`:
```
schema: text: string
```
One column. No paragraph delimiters, no document metadata beyond raw text. Single newlines (`\n`) are present (typical 8–50 per doc) but `\n\n` is absent in spot-checks.

**Decision: hard reset = document boundary only.** No paragraph splitting in scope. If future corpora carry paragraph metadata, the hard-reset trigger is the natural extension point.

---

## Architecture

### Two-tier reset

```
                  ┌─────────────────────────────────────┐
                  │  Outer doc-streaming loop (runEpoch)│
                  │                                     │
                  │  while not all_streams_done:        │
                  │    batch, hard_eos = data.next_tick()
                  │    soft_eos = wordSpace.drain_soft_eos()
                  │    runBatch(batch)  ◄── compute brick
                  │    for b in range(B):               │
                  │       if hard_eos[b]:               │
                  │           cascade.Reset(batch=b,    │
                  │                        hard=True)   │
                  │       elif soft_eos[b]:             │
                  │           wordSpace.soft_reset(b)   │
                  └─────────────────────────────────────┘
                              │
                              ▼ runBatch (pure)
                  ┌─────────────────────────────────────┐
                  │  forward(input)                     │
                  │  loss = compute_loss(...)           │
                  │  loss.backward()                    │
                  │  optimizer.step()                   │
                  │  ─ no Reset, no .item(), no branch ─│
                  └─────────────────────────────────────┘
```

| Trigger | Authority | Effect | Frequency (typical) |
|---------|-----------|--------|----------------------|
| Hard reset | `TheData` cursor | Full row-state wipe (parse stack, SVO, STM, codebook commit, discourse history) | Once per document end |
| Soft reset | `SyntacticLayer` derivation completion | Parse stack to start state, `_stm_fired[b]` re-armed, `_last_svo` cleared. **Discourse history preserved** | Once per natural sentence (driven by `<start>` reduction) |

### Compute brick contract

`runBatch` is pure compute. Specifically: **no `.item()`, no `.tolist()`, no Python-conditional on a tensor value, no tensor->host copies** within the call. All control flow is host-side from values that were host-side before the brick started (e.g., `valid_mask` is a tensor but never reduced inside the brick).

This contract is **not** met by today's code. See [§ Masked-brick audit](#masked-brick-audit) below for the residual call sites that must be vectorized for the contract to hold.

### Per-row cursor in TheData

```python
class SentenceStreamDataset(IterableDataset):
    def __init__(self, inputs, num_streams, slab_bytes=1024):
        self.inputs = inputs                       # list[str]
        self.num_streams = num_streams
        self.slab_bytes = slab_bytes
        self.stream_length = len(inputs) // num_streams
        # Per-row cursors. doc_idx[b] = current doc index in row b's stream;
        # offset[b] = byte offset within that doc.
        self.doc_idx = [b * self.stream_length for b in range(num_streams)]
        self.offset  = [0] * num_streams

    def next_tick(self):
        """Return ([B, slab_bytes] byte tensor, [B] host-side bool list).
        bool[b] is True iff this slab consumed the rest of row b's current doc."""
        slab = torch.zeros(self.num_streams, self.slab_bytes, dtype=torch.uint8)
        hard_eos = [False] * self.num_streams
        for b in range(self.num_streams):
            if self.doc_idx[b] >= (b + 1) * self.stream_length:
                # Row exhausted its slab; epoch-end for this row.
                continue
            doc = self.inputs[self.doc_idx[b]].encode('utf-8')
            remaining = len(doc) - self.offset[b]
            advance = min(self.slab_bytes, remaining)
            slab[b, :advance] = torch.frombuffer(
                doc[self.offset[b]:self.offset[b] + advance], dtype=torch.uint8)
            self.offset[b] += advance
            if self.offset[b] >= len(doc):
                hard_eos[b] = True
                self.doc_idx[b] += 1
                self.offset[b] = 0
        return slab, hard_eos
```

Bytes past `advance` stay zero → `valid_mask[b, advance:] = False` after the lex/embed pass, exactly the existing NULL-padding contract. **Zero data lost; zero spurious gradient.**

### Soft reset signal from the grammar

`Grammar` reads `<start>S</start>` from the XML config and stores `Grammar.start_symbol = "S"`. `SyntacticLayer.compose` ([Language.py:1257](../../bin/Language.py)) gains a per-row check: at the end of compose for batch row `b_flat`, if the row's parse stack reduced to a single node of category `start_symbol`, set a host-side bool `wordSpace._sentence_completed[b_source] = True` (where `b_source = b_flat // K`).

The outer loop drains `wordSpace._sentence_completed` after `runBatch` returns, dispatches `wordSpace.soft_reset(batch=b)` for True rows, and clears the flags.

`<start>` is a clean naming convention rather than a behavioral change to the parser — the parser already has a notion of "complete derivation"; this just makes the start symbol nameable and the boundary explicit.

---

## Syntactic sugar consumption

**The concern (raised during planning):** the current S-tier grammar has 17 rules, all of the form `S -> op(S)` or `S -> op(S, S)`. There's no rule like `S -> punct(S)` or `S -> ws(S)` or `S -> identity(S)`. Every input position is a candidate for some op, but no op explicitly says "this position is just syntactic noise (a comma, a space, a quote) — absorb it."

**Why it matters under rolling cursors.** With the current 1-byte-per-position byte lex routed through BPE chunking, raw-text sentences carry plenty of sugar: `,`, `.`, `;`, `:`, `"`, `'`, `(`, `)`, `\n`, runs of whitespace. After BPE, these become their own word slots. The grammar has to reduce them somehow to get a clean `S`. Today's grammar can only do this by abusing one of the existing binary rules (e.g. `union(S, S)` learns to pass-through one operand when the other is sugar) — bandaid, not contract.

Under rolling cursors, sentences run start-to-end with their punctuation; the grammar will see more sugar more often than under doc-as-atomic-row training. The lack of an explicit absorption rule is more visible.

**Proposed mitigation (in scope of this handoff):**

Add an explicit identity / sugar-absorption rule to the grammar:
```xml
<S>identity(S)</S>            <!-- no-op: pass-through one operand -->
<S>absorb(S, S)</S>            <!-- left-pass: discard right operand (sugar) -->
```

`identity` is a unary projection (S' = S). `absorb` is a binary rule that returns the left operand and discards the right — the rule predictor learns to assign it high probability when the right operand looks like sugar (low semantic activation magnitude, codebook prototype matching whitespace/punct, etc.).

These rules go in `MM_5M.xml` and are discoverable via training. No changes to `SyntacticLayer.project`'s rule dispatch are needed beyond adding the two methods to `_RULE_METHODS`.

**Alternative considered, not chosen:** strip syntactic sugar in `TheData` before the lex sees it. Loses information (model can't reconstruct original bytes), and breaks the byte-level lex contract. Rejected.

**Alternative considered, deferred:** a learned "sugar codebook" — a separate small codebook for low-content tokens that the rule predictor can route to. Cleaner long-term but requires PerceptualSpace changes; out of scope here.

---

## Masked-brick audit

To make `runBatch` a true masked compute brick, the following residual sync points must be removed. Each is independently revertible.

### Required for true-brick contract

1. **`stm_residual_microbatch.any().item()`** at [Language.py:2480](../../bin/Language.py).
   ```python
   not_fired = ~self._stm_fired
   if not bool(not_fired.any().item()):    # ← sync
       return None
   ```
   Replace: always run `disc.predict()` and zero the result for already-fired rows via `torch.where(_stm_fired.unsqueeze(-1), 0, bias)`. Cost: one matmul that wasn't strictly needed when no rows fired. Probably negligible.

2. **Truth-layer record loop** at [Spaces.py:6760-6770](../../bin/Spaces.py).
   ```python
   for i in range(act.shape[0]):
       for j in range(act.shape[1]):
           ...
           score = truth_layer.should_store(vec, ...)        # Python call
           if self.accumulateTruth * score > 0.5:            # tensor->bool branch
               truth_layer.record(vec, ...)                  # Python call
   ```
   Vectorize: `should_store` and `record` accept `[B*K, D]` tensors; the record op is gated by a `[B*K]` mask. The truth-layer's internal storage append needs a vectorized append-many path. This is the largest piece of work in the brick.

3. **`SyntacticLayer.compose` Python loops** ([Language.py:1257-1915](../../bin/Language.py)).
   The compose function has many `.tolist()` and `.item()` calls in its parse loops:
   - Line 1346: `query_mask_list = query_mask.tolist()`
   - Line 1348: `best_list = best.tolist()`
   - Line 1422: `alive_list = alive.tolist()`
   - Line 1644: `category_list = category.tolist()`
   - Line 1802: `cb_indices_list = cb_indices.tolist()`
   - Line 1880: `query_mask_list = query_mask.tolist()`
   - Line 1988: `probs_d.argmax(dim=-1)[0].item()`
   These drive Python branches that decide which rule to apply per row, then call `subspace.add_word(b, pos, gid)` per cell. To vectorize: replace per-row Python rule selection with a `[B*K, num_rules]` weighted blend (already partially in place via `probs_bcast`), and replace `add_word` per-row append with a tensor scatter into a fixed-size word-record buffer.

4. **Per-cell mask check in SymbolicSpace truth-layer** at [Spaces.py:6764](../../bin/Spaces.py) (added by the predecessor handoff):
   ```python
   if mflat is not None and not bool(mflat[i].item()):    # ← sync per cell
       continue
   ```
   Becomes a no-op once item (2) is vectorized: the mask is applied to the `[B*K, D]` tensor input to `should_store` directly, no per-cell branch.

### Already host-side (non-issue)

- `wordSpace._sentence_completed: list[bool]` (proposed) — set by appending in compose, drained in the outer loop. Never tensorized; no sync.
- `data.next_tick()` returning `hard_eos: list[bool]` — pure host-side cursor logic.
- `valid_mask` reads inside the brick — read as tensor, used in `torch.where`, never bool-coerced.

### Confirmation

**Under this handoff alone, the brick is *not* fully clean.** The handoff *removes* the in-loop Reset sync (via the outer-loop relocation) and adds *new* clean signaling paths for hard/soft reset. But it **does not vectorize** items 2–3 above; those remain as known follow-ups.

**To answer the user's literal question:** moving Reset to the outer loop is the right move and eliminates the in-loop Reset gate. After this handoff lands, the residual non-brick calls in runBatch's body are: `stm_residual_microbatch` (1), truth-layer loop (2), and `SyntacticLayer.compose` Python branches (3). Removing those gets us to a fully gate-free brick suitable for CUDA-graph capture. They're flagged here as required follow-ups; without them the brick still has 2–3 host syncs per tick.

---

## Files this handoff will touch

**Code:**
- [basicmodel/bin/data.py](../../bin/data.py) — `SentenceStreamDataset.next_tick()` per-row cursor; rename `__iter__` → `next_tick` or wrap so it returns `(slab, hard_eos)`.
- [basicmodel/bin/Models.py](../../bin/Models.py) — `runEpoch` becomes the outer doc-streaming loop. Delete the runBatch tail-Reset block at [Models.py:2436-2443](../../bin/Models.py) (and the `batch_advances_sentence` reads + MODEL_DEBUG assert added by predecessor).
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `InputSpace` — delete `batch_advances_sentence` property; leave `_end_of_stream` as a host-side `[B] list[bool]` propagated from `data.next_tick()` (no tensor reduction).
- [basicmodel/bin/Spaces.py](../../bin/Spaces.py) `Subspace.Reset`, `Space.Reset`, `InputSpace.Reset`, `PerceptualSpace.Reset`, `ConceptualSpace.Reset` — gain `(self, batch=None, hard=True)` signature. `batch=None` keeps current global-Reset behavior. `batch=b` clears row b only.
- [basicmodel/bin/Language.py](../../bin/Language.py) `WordSpace.Reset` — gain `(self, batch=None, hard=True)`. Add `WordSpace.soft_reset(batch=b)` (re-arm `_stm_fired[b]`, clear `_last_svo[b*K..(b+1)*K]`, reset parse stack rows for b).
- [basicmodel/bin/Language.py](../../bin/Language.py) `SyntacticLayer.compose` — at end of compose, scan per-row "did the stack reduce to start?" and append to `wordSpace._sentence_completed`.
- [basicmodel/bin/Language.py](../../bin/Language.py) `Grammar` — parse `<start>S</start>` from XML config, store as `Grammar.start_symbol`.

**Config:**
- [basicmodel/data/MM_5M.xml](../../data/MM_5M.xml) — add `<start>S</start>` and the sugar-absorption rules `<S>identity(S)</S>`, `<S>absorb(S, S)</S>`.
- [basicmodel/data/MentalModel.xml](../../data/MentalModel.xml) and [basicmodel/data/MM_xor.xml](../../data/MM_xor.xml) — same additions for consistency.

**Docs (updated as part of this handoff):**
- [basicmodel/doc/Architecture.md](../Architecture.md) — pipeline-as-unit invocation, two-tier reset.
- [basicmodel/doc/Spaces.md](../Spaces.md) — `Reset(batch=b)` semantics, removal of in-loop Reset gate.
- [basicmodel/doc/Language.md](../Language.md) — `<start>` rule, soft-reset detection, sugar rules.

---

## Sequencing (one PR per step preferred)

1. **TheData cursor (no behavioral change to model):** Implement `SentenceStreamDataset.next_tick()` returning `(slab, hard_eos)`. Add unit test asserting concatenation of per-tick slabs == original doc bytes for 100 random docs. Keep the old `__iter__` interface working for non-cursor callers.

2. **Outer-loop relocation in runEpoch:** Move the Reset cascade from runBatch tail to the new outer loop. Delete `batch_advances_sentence`. At this point Reset still fires on every tick (via `hard_eos[b] = True` for all b after one tick) — the cursor isn't yet driving the timing.

3. **Per-row Reset(batch=b) plumbing:** Each space's Reset gains `(batch=None, hard=True)`. Existing global-Reset callers stay correct. New per-row paths added; tested via direct calls.

4. **Cursor drives Reset timing:** `next_tick()` returns true `hard_eos` per row; the outer loop dispatches per-row Reset only when `hard_eos[b]`. Existing tests update to expect per-row state spanning multiple ticks within a doc.

5. **Soft reset from grammar:** `<start>` rule in Grammar; `SyntacticLayer.compose` emits `_sentence_completed`; outer loop dispatches `soft_reset(batch=b)`. Sugar rules `identity`, `absorb` added to grammar.

6. **(Follow-up PR) Vectorize residual sync points** — items 1–3 from the masked-brick audit. Independently revertible.

7. **(Follow-up PR) CUDA-graph capture** of the now-clean brick — `torch.compile(mode='reduce-overhead')` or explicit `torch.cuda.CUDAGraph` of `runBatch`.

---

## Verification

### Unit tests (new)

`basicmodel/test/test_data_no_byte_loss.py`:
- For 100 random docs of varying lengths (100 B to 100 KB), feed them through `SentenceStreamDataset(slab_bytes=1024)`.
- Run `next_tick()` repeatedly until `hard_eos[b] = True`.
- Concatenate all per-tick slabs (excluding NULL-padded tails) for row 0.
- Assert `concatenated == original_doc.encode('utf-8')`. Confirms no byte loss.

`basicmodel/test/test_per_row_hard_reset.py`:
- B=2 streams. Row 0 doc is 500 bytes (one tick + hard_eos). Row 1 doc is 2500 bytes (three ticks: 1024+1024+452, hard_eos on tick 3).
- Spy on `space.Reset(batch=...)`; assert row 0's Reset fires after tick 1, row 1's after tick 3, and they don't fire on each other's ticks.

`basicmodel/test/test_soft_reset_at_sentence.py`:
- Build a controlled grammar where derivation completes at a known input position.
- Run a tick; assert `wordSpace._sentence_completed[b]` is True for the row whose derivation completed and False for others.
- Assert `soft_reset(batch=b)` on True rows re-arms `_stm_fired[b]` (now False) and clears `_last_svo[b*K..]` (now zero).

`basicmodel/test/test_masked_brick.py`:
- Wrap `runBatch` in `torch.profiler.profile(...)`.
- Run one tick.
- Assert no `cudaMemcpyDtoH` events fire between forward start and optimizer.step end.
- Mark this test `@pytest.mark.skipif(not has_cuda)` and `@pytest.mark.slow` since it requires a CUDA device and is intended to gate the follow-up vectorization work.

`basicmodel/test/test_grammar_start_rule.py`:
- Parse a config with `<start>S</start>`.
- Assert `Grammar.start_symbol == "S"`.
- Parse a config without `<start>` (legacy). Assert `Grammar.start_symbol` defaults sensibly (probably `"S"` with a warning, or whatever the existing implicit default is).

`basicmodel/test/test_grammar_sugar_rules.py`:
- Build a config with the sugar rules `identity(S)` and `absorb(S, S)`.
- Construct a SyntacticLayer; assert both rules appear in `_RULE_METHODS`.
- Run a forward where the right operand of `absorb` is a near-zero vector; assert the output equals the left operand (within a small tolerance for the rule's softmax mixing).

### Existing-test gates

All tests listed in the predecessor handoff plus:
```
basicmodel/test/test_no_eos_sync.py            # may need updates: gate moved out of runBatch
basicmodel/test/test_padded_rows_no_op.py      # mask propagation unchanged
basicmodel/test/test_reset_at_batch_boundary.py # rename to test_reset_at_doc_boundary; semantics shift
```

### End-to-end hardware verification

On DGX Spark with BF16:
```bash
MODEL_AMP=bf16 basicmodel/.venv/bin/python basicmodel/bin/train.py \
    --model basicmodel/data/MM_5M.xml --data text --num-epochs 1
```
Pass criteria:
- `pwr` climbs to >80 W (target ~150 W).
- `mem%` becomes nonzero (>10%).
- Wall-clock per epoch drops 3–10× compared to pre-predecessor baseline.
- Loss curve: training proceeds; loss decreases monotonically over the first epoch (within noise). Bit-exact match to the pre-cursor version is *not* expected because the model now sees full long-tail documents that were previously truncated.

---

## Risks and known unknowns

1. **Compose state across ticks.** Within one tick, all (b, k) cells are computed independently (`[B*K]` per-cell state, established by the predecessor §0 finding). But across ticks, the per-cell parse stack is reused — does that compose correctly when row b's tick t+1 starts where tick t left off? Audit `WordSpace.ensure_microbatch` and the ReconstructionStack's behavior under repeated calls. Specifically: does `ensure_batch(B*K)` zero state on each call, or preserve it? If it zeroes, soft-reset between ticks does nothing; if it preserves, the cross-tick continuity works for free. (Best-guess: preserves on resize-to-same-shape, but verify.)

2. **`<start>` reduction detection.** The proposed signal is "stack reduced to single node of `start_symbol`." But the current parse-stack representation may not directly expose category labels per stack frame. Audit `ReconstructionStack` and `category_stack` to find the right read; possibly need a small extension to store the top frame's category alongside its vector.

3. **Sugar rules might over-fire.** If `absorb(S, S)` becomes the predicted rule for too many positions during early training, the grammar collapses to "discard everything but the first token." Mitigation: initial rule logits biased *against* `absorb` (prior); rely on the loss to upweight when actually beneficial.

4. **Soft reset frequency vs. hard reset frequency.** Under healthy training, soft resets should fire more often than hard resets (multiple sentences per doc). If the parser never fires soft resets, the per-row state would accumulate to doc-end and reset only on hard. That's not catastrophic (current behavior with no soft reset at all) but means soft reset adds nothing. Track soft-reset firing rate as a training metric to catch this.

5. **Discourse history meaning.** Today discourse accumulates across "sentences" (= docs) within a stream, and clears at hard reset (= now: doc boundary). Under the new design discourse will accumulate across *true* sentences within a doc, which is what discourse should mean. But existing trained checkpoints expect the old semantics; replaying training under the new design will produce different (better, in principle) discourse representations. Not a bug, but a checkpoint compatibility note.

---

## Tensor buffer for `subspace.word` (compose vectorization, in-scope)

The remaining `.tolist()` / `.item()` syncs in `SyntacticLayer.compose`
([Language.py:1257-1915](../../bin/Language.py)) are **all** driven by
`subspace.add_word(b, pos, rule_id, ...)` Python list mutations.
Vectorizing them in isolation is impossible because they're paired with
host-side appends to `subspace.word: list[tuple]`. The fix is to replace
that Python list with a tensor buffer.

### Current state

`SubSpace` ([Spaces.py:2390](../../bin/Spaces.py)):
```python
self.word = []  # list of (batch, vector, rule, ...) tuples
```
Mutated per-cell in compose:
```python
subspace.add_word(b, pos, rule_id, order=d, leaf1=..., leaf2=...)
# implementation: self.word.append(self.wordEncoding.encode(...))
```
Consumed by `decompose`, `reconstruct`, the SVO extractor, and the
derivation-trace walkers.

### Target state

Two tensor buffers on `SubSpace`:
```python
self.word_records  # [B*K, max_depth, ENTRY_WIDTH] long
self.word_count    # [B*K] long, current depth per cell
```
`ENTRY_WIDTH` covers the existing tuple shape: `(batch, vec_idx, rule,
order, leaf1, leaf2, leaf3)` = 7 longs (vec_idx is the position in the
word codebook; the actual vector is recovered by lookup).

`add_word` becomes a tensor scatter:
```python
def add_word(self, b_indices, vec_idxs, rule_ids,
             order=None, leaf1=None, leaf2=None, leaf3=None):
    # b_indices: [N_active] long
    depths = self.word_count[b_indices]
    self.word_records[b_indices, depths, RULE]    = rule_ids
    self.word_records[b_indices, depths, VEC_IDX] = vec_idxs
    if order  is not None: self.word_records[b_indices, depths, ORDER]  = order
    if leaf1  is not None: self.word_records[b_indices, depths, LEAF1]  = leaf1
    if leaf2  is not None: self.word_records[b_indices, depths, LEAF2]  = leaf2
    if leaf3  is not None: self.word_records[b_indices, depths, LEAF3]  = leaf3
    self.word_count[b_indices] += 1
```

`compose` then computes per-cell tensors directly (no `.tolist()`):
```python
# All [B] long, on device:
b_alive   = torch.arange(B, device=...)
positions = ...                          # from active_positions[d]
rule_gids = composable_global[best.long()]   # if composable_global tensorized
subspace.add_word(b_alive, positions, rule_gids, order=d)
```

### Migration strategy (Path A vs Path B from earlier discussion)

**Path B chosen: hybrid with deferred materialization.** Inside the
brick, compose writes to the tensor buffer. Outside the brick (after
`runBatch` returns), the outer loop calls a flush:
```python
def flush_word_buffer(self, subspace):
    """Drain the tick's tensor buffer into subspace.word (Python list)
    so legacy consumers (decompose, reconstruct, SVO walker) keep
    working unchanged. One sync per tick, outside the compute brick.
    """
    counts = self.word_count.tolist()      # one sync; [B*K] ints
    records = self.word_records.tolist()   # one sync; nested ints
    for bk, depth in enumerate(counts):
        for d in range(depth):
            entry = records[bk][d]
            subspace.word.append(WordEntry(*entry))
    self.word_count.zero_()                # ready for next tick
```

This keeps the wide blast radius of changing every consumer **out of
scope**. Consumers see `subspace.word` populated as before, just
materialized once per tick instead of mutated per-cell. The brick body
itself is sync-free.

Path A (full tensor-only `subspace.word`, eliminating the flush by
updating every consumer to read tensors) is the proper end state but is
a larger PR; if Path B causes downstream pain it can replace it later.

### Touch list for Path B

- [Spaces.py](../../bin/Spaces.py) `SubSpace`:
  - `__init__`: register `word_records`, `word_count` buffers; keep
    `self.word = []` for materialization.
  - `add_word(...)`: rewrite as tensor scatter, with the legacy
    `(int, int, int, ...)` overload kept for the Models.py / test
    callers that pass scalars.
  - Add `clear_word_buffer()` for the inter-tick reset.
- [Language.py](../../bin/Language.py) `SyntacticLayer`:
  - `flush_word_buffer(subspace)`: call from outer loop post-runBatch.
  - `compose`, `_compose_vector`, `_compose_to_target`,
    `_compose_vector_chart`: replace the per-row `.tolist()` +
    Python-loop-`add_word` patterns with tensor `add_word(b_indices,
    ...)` calls. Delete the residual `.tolist()` / `.item()` sites
    flagged in the masked-brick audit.
- [Models.py](../../bin/Models.py) `runEpoch` outer loop: call
  `wordSpace.syntacticLayer.flush_word_buffer(subspace)` after each
  `runBatch` (next to the per-row Reset and `truth_layer.compact()`).

### Tests

- `basicmodel/test/test_word_buffer_flush.py`: feed a controlled grammar
  derivation; assert `subspace.word` after flush matches the legacy
  per-cell `add_word` output entry-for-entry.
- `basicmodel/test/test_compose_no_sync.py`: `torch.profiler.profile`
  one tick of compose; assert no `cudaMemcpyDtoH` events fire between
  forward start and forward end (excluding the post-tick flush).

## What is NOT in this handoff

- **CUDA-graph capture.** Becomes feasible once compose-vectorization (above) lands. Separate PR.
- **Real natural-language sentence splitting** (e.g. via `pysbd`). The grammar's `<start>` reduction is the canonical sentence boundary; punctuation-driven splitting in `TheData` was considered and rejected as a noisy proxy.
- **Paragraph-level reset.** FineWeb has no paragraph metadata. If a future corpus carries it, add a paragraph cursor between doc and slab; the hard-reset trigger generalizes naturally.
- **Per-document positional re-zeroing.** Today positional encodings are continuous across batches. Under rolling cursors, positional encodings should reset to 0 on hard reset (so doc 2 doesn't continue doc 1's `where`/`when` encoding). Audit `WhereEncoding.p` increments at [Spaces.py:1863](../../bin/Spaces.py) and reset on hard reset. (This may already work via the `Reset()` cascade if `WhereEncoding.Reset` zeros the cursor; verify.)
