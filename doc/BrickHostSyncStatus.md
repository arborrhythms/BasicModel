# Brick host-sync elimination — status & residual

> Goal: make `runEpoch`'s brick body issue **zero `cudaMemcpyDtoH`**
> events so it is CUDA-graph-capturable, and gate all training on it.
> Driver: `test/test_brick_no_sync.py` (now parametrized over
> MM_xor / MM_grammar / MM_5M) + a strict `train.py` pre-flight.

> **Update (2026-05-19) — new deferred residual on the per-word IR path.**
> The two-loop word-at-a-time IR-reconstruction objective (Reworks A/B,
> see `doc/plans/2026-05-18-two-loop-pipeline-architecture.md` §MISSION
> COMPLETION STATUS) introduces a **per-word path that is not
> fullgraph-capturable**: the variable-length `while (w:=next_word())`
> loop + `next_word`'s `bool(any_pos.any())` host-sync
> (`bin/Spaces.py:6508`). This is the **Phase-5 capture-closeout
> concern** — deep (variable per-word-loop capture), explicitly deferred
> (spec decision D6); the metalbaby perf gate (shift-reduce vs
> CKY+resize) is gated behind it, CKY+resize is the documented fallback
> meanwhile. The eager training path is verified working (objective
> trains end-to-end). `test_compiled_step_invoked` failing on the
> grammar/per-word path is this known, accepted residual — not a
> regression.

> **Status (2026-05-16):** the enumerated residual (`_snap_content`,
> `runEpoch`-return, F, C, D, E, A, B) is **fully landed** — MM_xor
> `runEpoch` **attribution = 0** on metalbaby CUDA ("brick body is
> host-sync-free"); full CPU suite 1084 passed, 0 regressions. The
> strict gate is **not yet open**: `test_brick_no_sync.py[MM_xor]`
> still reports ~20 implicit `Memcpy DtoH` (autograd / boolean-index
> internals) that the attribution methodology cannot see. See
> "Device-event gate: the implicit-sync tail".

## Pre-flight gate (added)

`ModelFactory._brick_preflight` ([Models.py](bin/Models.py)) profiles one
short `runEpoch` before substantial training and **hard-aborts** if any
`cudaMemcpyDtoH` is issued. CUDA-only (no-op on CPU/MPS — the MPS dev
path and CPU test suite are unaffected; verified). It is wired before
all training dispatch in `ModelFactory.run`, so **every** model trained
via `train.py` is gated. This is intentionally strict (owner's choice:
"strict hard-abort now") — it is a forcing function and will keep MM_5M
(and every CUDA config) blocked until the residual below reaches 0.

**Pre-flight batch-budget fix (2026-05-16).** The pre-flight's two
`runEpoch(split="train")` calls used to be bound by *and* increment
`BASIC_MAX_BATCHES` (the `--batches` budget). On a large corpus the
warm-up consumed the entire budget, so the profiled pre-flight epoch
*and* the real training each ran **zero** batches — training did
nothing, and the gate reported a **false `OK: 0`** (it had profiled an
empty epoch). Fixed: `_brick_preflight` sets `m._preflight_active`,
runs **bounded** throw-away epochs (`max_batches` = 4 warm-up / 2
profiled) **at the real configured `batchSize`** (it was hard-capped
at 2 — meaningless, since a CUDA graph is shape-specialized; cheapness
now comes from the batch-*count* bound, not a shrunk batch), and
`runEpoch` skips all `BASIC_MAX_BATCHES` accounting while
`_preflight_active`. Verified on metalbaby: `--batches 10` is fully
available to training; the profiled epoch runs real-shape batches.
This exposed the truth — MM_5M at the real **`bs=64`** issues **430
`cudaMemcpyDtoH`** under bf16 + `max-autotune` (the old `bs=2` cap
under-reported 58; the original budget bug reported a false `0`). Real
training is unaffected (`m.run` uses the config batchSize in a
separate call). The gate now works honestly at the production shape.

## Measured impact (MM_xor, MODEL_DEBUG off, metalbaby CUDA)

**245 → 24 cudaMemcpyDtoH (~90% reduction), zero correctness
regressions** (CPU: 189 passed in the representative order).

| Fix | Site | Mechanism | ~syncs cut |
|---|---|---|---|
| Bucket 1 | `BaseModel.getOptimizer` (+ brick test uses it) | `Adam(capturable=True)` on CUDA — stock `_multi_tensor_adam` does `_get_value(step).item()` per param otherwise | ~136 |
| Bucket 2 | `SubSpace.normalize` range/finite check | gated behind `MODEL_DEBUG` (no functional effect; mirrors `_assert_finite_train_state`); `test_normalize_false_does_not_modify` contract updated to match | ~88 |
| Bucket 3 | `TheError.snapshot()` call in `runBatch` | gated behind `MODEL_DEBUG` (diagnostic-only; feeds `covariance()` which has no training-loop consumer) | ~30 |
| Bucket 5a | `WhatEncoding.decode_tokens` | one bulk `buf.tolist()` instead of B·N per-slot copies (identical data) | ~60 |
| #6 | `Error`/loss `total()` | `torch.isfinite(lossIn).all()` Python branch → on-device `{0.,1.}` gate + `nan_to_num` (same semantics, no `__bool__` sync) | ~few |
| Lazy reverse | `PerceptualSpace.reverse` word recovery | `decode_reverse_meta` deferred to first consumer via `_materialize_recovered_input()` — report-only, never gradient; **computation unchanged, only its timing** | ~64 |

### Reverted (correctness > perf — concrete lesson)

**Bucket 4** (vectorize `decode_reverse_meta`'s per-slot
`_nearest_idx().item()`): **reverted.** `_wrapped_mse_score(a,b)`
([embed.py](bin/embed.py)) is **broadcasted elementwise subtraction**,
not a pairwise `[M,V]` score; the batched `[M,D]` vs `[V,D]` form
produced garbage indices → silently broken reconstruction.
`test/test_xor_spaces.py` caught it (4 reconstruction failures);
reverted, 20/20 green again. The *correct* elimination for this 64-sync
bucket was the **lazy** approach above (timing change, not computation
change) — which is in.

## Landed since

Contained, verified-safe items from the residual, completed (CPU: 175
passed, 0 regressions; attribution confirms their sites are gone):

- **`_snap_content` guard** ([Spaces.py](bin/Spaces.py) `_snap_content`):
  dropped `if torch.any(nonzero):`. Every op in the body already masks
  via `flat[nonzero]`, so all-zero input is an empty-tensor no-op —
  bit-identical in all cases (all-zero / mixed / all-nonzero), minus
  the `torch.any().__bool__` host sync.
- **`runEpoch` return** ([Models.py](bin/Models.py)): returns `outErr`/
  `inErr` as **tensors**; the 3 `runTrial` consumer sites materialize
  with `.item()` immediately after the unpack. The host sync now lands
  in `runTrial`, *outside* the pre-flight's profiled `runEpoch`. (Any
  new `runEpoch` caller that formats/accumulates must `.item()` first —
  documented at the return site.)
- **Residual F — `TruthLayer.__len__` empty-truth early-out**
  ([Layers.py](bin/Layers.py) `TruthLayer`, [Language.py](bin/Language.py)
  `truth_modulated_loss`): the one in-brick consumer needs *emptiness*,
  not the count. Added a host-side `_nonempty` bool + `is_empty()`
  (sync-free), maintained co-located with **every** `count` write
  (`record`/`compact`/restore/`prune`/`orthogonalize`) from an
  already-host-side value (Python int / `tensor.shape[0]`) so it adds no
  sync and cannot drift; added `TruthLayer.clear()` to make the external
  reset (was `Models.py` reaching in to zero `count`/`_sources`/
  `_trusts`) atomic. `truth_modulated_loss` now calls `is_empty()`.
  `__len__` unchanged (correct `len()` semantics for the out-of-brick
  callers). Attribution: MM_xor 32 → **28** on CUDA (metalbaby), the
  `Layers.py __len__` site gone, no new sites.
- **Residual C — `create_ir_mask` empty-mask early-out**
  ([Models.py](bin/Models.py) `create_ir_mask`): dropped
  `if not bool(mask.any()): return`. An all-False boolean-mask write
  (`new_event[mask, :nWhat] = null_vec`) selects zero rows, so the body
  is a content-identical no-op when nothing is masked (`new_event ==
  event`); running it unconditionally is bit-identical in all cases
  (same shape as the `_snap_content` fix), minus the
  `mask.any().__bool__` sync. Attribution: 28 → **24** on CUDA, site
  gone, no new sites; CPU 175 passed, 0 regressions.
- **Residual D — `runBatch` IR-pred / recon guards (×2)**
  ([Models.py](bin/Models.py) `runBatch`): the None checks stay Python
  (no sync); only the `bool(mask_pos.any())` term was a sync. It guarded
  the empty-mask case where the masked gather is empty and
  `F.mse_loss`'s mean-of-empty is **NaN**. Dropped the bool term, run
  the gather unconditionally, and sanitize with
  `torch.nan_to_num(..., nan=0.0)` (the sanctioned #6 on-device-gate
  pattern). Semantics preserved: finite real losses pass through
  unchanged; empty-mask → 0.0, and `TheError.total()` is a *weighted
  sum* so a 0.0 term is identical to a skipped one; `nan_to_num`'s
  backward zeroes the NaN position so no bad gradient leaks.
  Attribution: 24 → **16** on CUDA (−8, both sites), gone, no new
  sites; CPU 175 passed, 0 regressions.
- **Residual E — `PerceptualSpace.reverse` substrate-fold magnitude
  branch** ([Spaces.py](bin/Spaces.py) `reverse`): the old
  `y_post.detach().abs().max().item() > 1.01` chose
  fold-vs-passthrough (a host sync) when the SyntacticLayer dispatcher
  short-circuited. Compute the fold and select with an **on-device**
  0-dim bool gate via `torch.where` (scalar condition → all-or-nothing
  select), **bit-identical** to the old branch in every case (the
  shape-mismatch branch stays a Python check). The doc sanctioned an
  unconditional fold here; selecting is strictly safer (zero output
  change). Attribution: 16 → **12** on CUDA, site gone, no new sites;
  CPU 175 passed, 0 regressions (incl. `test_xor_spaces` /
  `test_invertibility`, E's named net).
- **Residual A + B — lex→host-token-carry refactor**
  ([data.py](bin/data.py) `stringTensor`, [Spaces.py](bin/Spaces.py)
  `WhatEncoding.tokens_to_decoded` / `prepInput` / `_lex_batch` /
  `_embed`, [Models.py](bin/Models.py) byte-cursor branch). The lexer
  had the token strings on the host in `_lex_batch`, encoded them to a
  device byte buffer, then `_embed` decoded them back (B) — and the
  lexer itself `.tolist()`-ed a *device* byte tensor (A).
  - **B:** new `WhatEncoding.tokens_to_decoded` reproduces
    `decode_tokens(encode_tokens(...))` purely host-side (replicates
    the `nWhat-1` UTF-8 truncation + null-terminated decode incl. the
    embedded-NUL early stop), carried on `subspace._host_tokens`;
    `_embed` consumes it (decode_tokens kept as the fallback for
    direct-construction callers/tests). **Bit-identical** OOV keys /
    indices.
  - **A:** `stringTensor` now builds the byte tensor on **CPU** (a
    default-device mode was placing it on MPS/CUDA; it is a host
    string→bytes encode); `prepInput` / the byte cursor stash that
    host copy as `_host_input_slab`, consumed once in `_lex_batch`, so
    `_to_text`'s `.tolist()` is a CPU op. Byte-identical to the device
    copy (it *is* the tensor that becomes `inputTensor` via
    `.to(device)`; `_to_text` masks `& 0xFF` so int8/uint8 agree).
  - **Result:** MM_xor CUDA **attribution = 0** — *“runEpoch brick
    body is host-sync-free”* (authoritative metalbaby read). Full CPU
    suite **1084 passed, 0 regressions** (representative order 175,
    0 regr.). Tradeoff: `prepInput` now does one **HtoD** stage-copy
    of the host slab (pageable → a synchronizing H2D, *not* a DtoH, in
    data-prep outside the forward/backward/step brick) — it replaced a
    per-row lexer **DtoH**; net DtoH ↓.

## Reliable metric caveat (read before continuing)

The raw `torch.profiler` "Memcpy DtoH" **device-event count is noisy /
non-monotone** run-to-run (262→245→96→64→34→24→36 across this work) —
not every `.item()`/`__bool__`/`.tolist()` emits a distinct pageable
memcpy (allocator/coalescing dependent). **Use the monkeypatch
*attribution* (recorded host-sync calls by `file:line` callsite) as the
signal of truth when eliminating syncs** — wrap
`torch.Tensor.{item,tolist,nonzero,__bool__,__int__,__float__,__index__}`,
count by `traceback.extract_stack` callsite, run one `MODEL_DEBUG=0`
`runEpoch` on CUDA. The strict pre-flight itself asserts on the device-
event count (so the gate only truly opens when the attributed sites are
all gone *and* the device count hits 0).

## Residual — path to 0 (ALL LANDED — attribution = 0)

The doc's enumerated residual (the attribution-visible host syncs) is
**fully eliminated**. Every item (`_snap_content`, `runEpoch`-return,
F, C, D, E, A, B) landed deliberately, one at a time, each verified:
CPU representative-order regression (175 passed, 0 regr.) → `make sync
HOST=mb` → metalbaby CUDA re-profile. On the authoritative metalbaby
read, **MM_xor `runEpoch` attribution = 0** (262… → 32 → 28 → 24 → 16
→ 12 → **0**): *“runEpoch brick body is host-sync-free”*. Full CPU
suite: **1084 passed, 0 regressions**.

This is the FIRST of the doc's two "done" conditions. The SECOND — the
strict pre-flight's device-event assert (`cudaMemcpyDtoH == 0`) — is
**not yet met**: see next section. The doc always anticipated this
(“the gate only truly opens when the attributed sites are all gone
*and* the device count hits 0”).

## Device-event gate: the implicit-sync tail (remaining work)

The implicit C++ syncs (data-dependent op output sizes; pageable
host↔device staging; per-element device writes) are *invisible* to the
attribution monkeypatch. `brick_attr_probe.py` gained two modes that
*do* see them: `BRICK_PROBE_SYNCDEBUG=1` (`set_sync_debug_mode("warn")`,
aggregated callsites) and **`BRICK_PROBE_SYNCERROR=1`**
(`set_sync_debug_mode("error")` + `set_detect_anomaly(True)`: raise on
the *first* sync; anomaly augments a backward sync's traceback with the
*forward* op that built the grad_fn). All probed `runEpoch`s are now
bounded (`BRICK_PROBE_MAX_BATCHES`, default 6) so MM_5M doesn't run a
358k-sentence epoch.

**Session 2026-05-16 — the forward brick is now sync-free.** Peeled,
one at a time on metalbaby (CPU representative-order 175 passed, 0
regr. after the batch):

- **Pageable host→device staging** (synchronizing) made async via
  `pin_memory().to(dev, non_blocking=True)`: `prepInput` /
  `prepOutput` (`Spaces.py`), byte-cursor `inp_items.to(device)`
  (`Models.py` runEpoch).
- **Per-element device writes in Python loops** (each a sync H2D) →
  build on **CPU** (`device='cpu'` explicit, dodging the
  default-device-mode trap) then one async stage: `where_idx`
  (`_lex_batch`), `encode_tokens` `buf`, `_embed` `what_indices`,
  `embed.py` sbow/cbow `idx`.
- **`torch.tensor(0.0, device=cuda)`** scalar constructors (host
  float → device copy) → `torch.zeros((), device=cuda)` (memset
  kernel, no copy) — 4 sites in the runBatch loss path.
- **`create_ir_mask` boolean-mask assignment** `new_event[mask,
  :nWhat] = …` → dense `torch.where` (static shape; bit-identical).
- **Residual-D boolean-mask gathers** `pred[mask]` (×3: IR-pred,
  recon-C, recon-S) → new **`ModelLoss.compute_masked(pred, target,
  mask)`** (one source of truth): masked-sum / (mask_count ·
  seg_width) ≡ `compute(pred[mask], target[mask])`, on-device
  reductions only (`mask_count` never `.item()`-ed), empty-mask → 0.0
  with no 0/0 (replaces the `nan_to_num` gate). Value-equivalent
  (fp reduction order differs at ULP scale, not the value); CPU
  regression confirms 0 functional change.

**Remaining (next phase): one autograd-internal D2H in the aux-loss
backward.** With the forward sync-free, `BRICK_PROBE_SYNCERROR` +
anomaly now points precisely at **`Models.py` `runBatch`
`totalLoss = totalLoss + aux_total`** (`AddBackward0`).
`aux_total = self.outputSpace.subspace.errors.total()` — the summed
auxiliary error terms every SymbolicSpace stage writes during forward.
One of those terms has a backward that synchronizes (a data-dependent
op in a symbolic/auxiliary forward). **Next step:** bisect
`pipeline_errors.terms()` (disable/zero terms one at a time, or scan
each contributing `SymbolicSpace.forward` aux computation for
boolean-mask / `nonzero` / `.item()` in its backward), then convert
the offender to a dense on-device form (the `compute_masked` /
`torch.where` shape). Bucket-4 discipline: one change → CPU
representative order → metalbaby `SYNCERROR` re-probe.

**MM_5M confirmation (2026-05-16, metalbaby, bf16 + max-autotune).**
With the pre-flight budget + real-batchSize fix the profiled epoch
runs real-shape batches: **`bs=64` → 430 `Memcpy DtoH`**
(Device→Pageable / →Pinned) — scales per-row (the throttled `bs=2`
read was 58). The sync-debug probe
(`BRICK_PROBE_SYNCDEBUG=1 … MM_5M.xml`) attributes them to the same
two project frames as MM_xor: `Models.py runBatch totalLoss.backward()`
(autograd-internal D2H, dominant) and `Models.py runEpoch`
byte-cursor `inp_items.to(device)` (HtoD input staging, *not* a DtoH;
pre-existing, byte-lexer path). So MM_5M needs no new attribution work
beyond MM_xor's — it is the **same shared backward-D2H tail**, just
more events on the larger vocab.

Prime suspects for the backward D2H: the **boolean-mask gathers** that
survive in the brick (`pred_clip[mask_clip]` and the recon-C/S
`pred[..][mc]` in residual-D's blocks; the codebook OOV/index paths in
`_embed`). Residual D removed the *attributed* `bool(mask.any())`
guard, but the gather `x[cuda_bool_mask]` is *itself* an implicit D2H
(its output size is host-decided) whose forward **and** backward sync —
invisible to the attribution monkeypatch, counted by the profiler. The
elimination is to rewrite them as **dense masked ops** (compute over
all positions, multiply by the float mask, `sum` / `clamp` — the same
on-device-no-op shape used for C/D), preserving numerics
(bucket-4-risky; `test_xor_spaces` / `test_invertibility` /
`test_perceptual_bivector` are the net).

The device-event count is *also* the doc's known-noisy metric
(non-monotone run-to-run; `set_sync_debug_mode` reports few callsites
for many profiler events — the mapping is not 1:1, because one
`.backward()` line fans out to many C++ autograd D2H nodes). Closing
it to exactly 0 is a deeper investigation than the attribution
residual and is **not** covered by the original A–F table.

## Handoff — how to resume cold (next session)

1. Read this file + `test/test_brick_no_sync.py` + `brick_attr_probe.py`
   + `ModelFactory._brick_preflight`.
2. **Truth signals** (both on metalbaby; local MPS is a fast proxy for
   #a only): build the latest tree (`make sync HOST=mb` from the
   `WikiOracle` parent), then
   (a) explicit-method attribution:
   `MODEL_DEBUG=0 .venv/bin/python test/brick_attr_probe.py MM_xor.xml 2`
   — expect **0** (regression check; the probe is device-aware and
   attributes past lexer pass-through frames);
   (b) implicit-sync attribution:
   `MODEL_DEBUG=0 BRICK_PROBE_SYNCDEBUG=1 … test/brick_attr_probe.py
   MM_xor.xml 2` — drives the device-event tail;
   (c) final gate:
   `pytest test/test_brick_no_sync.py -k MM_xor` (asserts profiler
   `cudaMemcpyDtoH == 0`).
3. The remaining work is the **implicit-sync tail** above, not the
   A–F residual (landed). Start with the `backward()` D2H: bisect the
   runBatch forward graph (the boolean-mask gathers in the residual-D
   recon blocks and the codebook OOV/index paths are prime suspects —
   each `x[cuda_mask]` is an implicit D2H whose backward can also
   sync). Prefer masked/dense on-device forms over boolean-index
   gathers (compute densely, weight by the float mask, sum / clamp —
   the on-device-no-op shape used for C/D).
4. After each fix: CPU **representative order**
   (`test_mm_xor test_mm_boolean test_universality test_invertibility
   test_xor_spaces test_basicmodel::TestNormalizeFlag
   test_lexicon_ownership test_conceptual_bivector
   test_perceptual_bivector test_phase2_pipeline_primitives`; full
   suite needs `PYTORCH_ENABLE_MPS_FALLBACK=1` for the unrelated
   MPS `_cdist_backward` gap) → `make sync HOST=mb` → probe (a)+(b)
   → brick test (c).
5. Done when MM_xor *and* MM_5M: attribution = 0 **and** the
   pre-flight device-event assert passes (0). Then the strict gate
   opens and MM_5M trains.

## MM_5M status

MM_5M is **still gated** by the strict pre-flight, now solely on the
**implicit-sync device-event tail** (the A–F attribution residual is
landed and shared code with MM_xor, so MM_xor attribution = 0 carries
over directly; MM_5M only exercises a *bigger* `decode_tokens`/`_embed`
lexicon-decode — same host-token-carry path, already fixed). Full
MM_5M attribution profiling on metalbaby is heavy (FineWeb + 5M
model); the shared-path analysis is authoritative. To unblock MM_5M
training: close the implicit-sync tail (drive the pre-flight
`cudaMemcpyDtoH` to 0) per the section above.

## Note: pre-existing test-ordering pollution (separate issue)

`test/test_invertibility.py` passes 82/82 standalone and in most
orderings, but 4 `TestConceptualSpace*` tests fail when run *after*
`test_xor_spaces`/`test_mm_xor` in a particular file order. Same code,
different order ⇒ global-state leakage not reset by the conftest
`_reset_global_singletons` fixture for those classes. **Not** caused by
the brick changes (verified by isolation). Pre-existing test-isolation
fragility; worth a follow-up to extend the conftest reset or seed those
classes.
