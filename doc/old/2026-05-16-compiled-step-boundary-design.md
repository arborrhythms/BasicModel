# Compiled per-batch step — boundary design

**Date.** 2026-05-16
**Owner.** Alec
**Status.** Design approved; Phase-1 execution **partially landed +
blocked on an architecture fork** (see "Phase 1 progress & pivotal
finding" at end). Companion to / detailed Phase 1 of
[2026-05-16-compile-scoped-to-model-thedata-writeback.md](2026-05-16-compile-scoped-to-model-thedata-writeback.md).

> **✅ FORK RESOLVED (owner, 2026-05-16): path B**, with concrete
> per-class tactics (tractable, not a vague rearchitecture):
> - **#1,2,3** (lexer `_lex_and_embed`/`_lex_batch`/`_to_text`):
>   write into **pre-allocated buffers** (no dynamic alloc / host
>   round-trip in the traced region).
> - **#0,5,6,7** (`empty_state`/`copy_context` SubSpace plumbing):
>   each `Space` **owns its SubSpace(s)**, allocated in `__init__`
>   (or a per-batch `reset()`-style call invoked eagerly at batch
>   start); the traced `forward` reuses the owned subspace
>   (tensor-mutates in place) — no `SubSpace(...)`/`copy_context`
>   in the traced path.
> - **#4** (`_embed` codebook/OOV): **pre-allocate the codebook at
>   config `<nVectors>` size**, use only the live portion via a
>   **view/slice** (no dynamic resize).
>
> Resume the recon→eliminate loop with these; one class at a time,
> CPU representative suite + `test_compiled_step_invoked` per step.

**Goal.** Make the model's per-batch compute a real, invoked,
`torch.compile`d unit (today it is **not** — O1: `m = compile(m)` then
`m.run()` delegates to the eager `_orig_mod`; dynamo traces 0 frames),
with the **non-grammar forward path entirely break-free** so it is
CUDA-graph-capturable per shape-bucket.

---

## Decided design (from brainstorming)

1. **Boundary.** One compiled callable = `forward + loss + backward`.
   `optimizer.step()` / `zero_grad()` stay **eager** after it. Params
   are module buffers; compiled backward populates `.grad`; the eager
   optimizer consumes them. Any Adam-capturable `.item()` is in the
   eager region (allowed — only the captured graph must be sync-free;
   consistent with the prior bucket-1 `capturable=True`).

2. **What is compiled = the whole `BasicModel.forward`, made
   break-free**, by moving every data-dependent Python construct out
   of the compiled region:
   - **Relocate to the eager `TheData` producer** (input-prep, not
     model math): byte/word lexing, `_token_stream`/`_to_text`,
     OOV/index resolution, the cursor, host staging. The producer
     writes static-shape tensors into persistent `TheData` device
     buffers (the write-back architecture in the parent spec).
   - **Rewrite as static on-device tensor ops** (model *compute*
     that is currently data-dependent Python): the residual-style
     fixes — boolean-mask gathers → `ModelLoss.compute_masked` /
     `torch.where`; `.item()`/`bool()` guards → on-device; fixed-trip
     loops → unrolled/masked static form.
   - The compiled forward then consumes only `TheData` tensors and
     emits loss + a per-term `err_buf`; backward populates `.grad`.

3. **Static shapes = bucketed.** A **bounded, finite** set of static
   shapes keyed by structural level (conceptualOrder / N-halving step
   / a small fixed set of token-slab sizes — the cursor already sizes
   to `nObj`). One compiled+captured graph per bucket
   (CUDAGraph-trees specializes per shape); the design work is
   *bounding* the bucket set to a handful, not making it unbounded.

4. **Vocab = non-issue (resolved with owner).** At training time all
   codebooks are config-`<nVectors>`-fixed and the BPE codebook is
   **frozen** (`wordLearning=0`, carried in the `.ckpt`). The only
   vocab growth is a separate **offline codebook-build** stage
   (`wordLearning`→1) that is *not* in the training/capture loop.
   **Precondition:** training runs from a populated `.ckpt`. No
   freeze machinery, no dynamic-vocab handling in the compiled step.

5. **AMP.** bf16 autocast wraps the compiled forward+backward (the
   owner trains bf16 + max-autotune).

## Scope & acceptance criterion (the key refinement)

The forward splits into a **non-grammar path** and a **grammar
path**:

- **Non-grammar path** — the dense subsymbolic numeric compute:
  stem → body → head; `PerceptualSpace` embed/lift; the
  Conceptual/Symbolic *tensor* ops (Sigma/Pi folds, codebook
  snap/commit, reconstruction/IR losses) **excluding** the grammar
  chart. **Requirement: entirely break-free. Mandatory. Not
  deferrable.**
- **Grammar path** — `SyntacticLayer` compose / chart parser /
  grammar-CFG dispatch / `_apply_rule_forward` / derivation-trace:
  inherently data-dependent (parse trees, rule selection).
  **Best-effort break-free; any residual grammar-path breaks may be
  left for later work.**

**Acceptance:** `torch.compile(fullgraph=True)` succeeds for the
non-grammar path on every shape-bucket (0 graph breaks); the only
permitted remaining breaks are inside the grammar path, each
explicitly attributed and logged as deferred. "Everything else should
be done."

The precise grammar/non-grammar line is finalized by the recon step
(below): each enumerated break is tagged *grammar* or *non-grammar*
from its source frame; non-grammar ⇒ must-fix, grammar ⇒ fix-if-cheap
else defer with a recorded reason.

## Method — measured, iterative break-elimination

1. **Make it actually invoked.** Factor the per-batch compute into an
   explicit callable; `torch.compile` *that*; invoke the **compiled
   object** from `runBatch` (not the eager `_orig_mod`). Confirm via
   the O1 probe that dynamo now traces > 0 frames.
2. **Recon.** Run `torch.compile(fullgraph=True)` (and/or
   `torch._dynamo.explain`) over the forward; enumerate every graph
   break with its source line; tag each *grammar* / *non-grammar*.
   Produces the elimination backlog.
3. **Eliminate by class**, one at a time, non-grammar first: each
   break → choose *relocate-to-producer* vs *rewrite-static* vs
   *unroll*; implement; verify (CPU representative suite, 0 regr. +
   tiny bounded metalbaby check); repeat. Bucket-4 discipline (one
   change, full verification, never batch correctness-critical edits).
4. **Gate.** `fullgraph=True` on the non-grammar path is the strict
   acceptance contract (matches the project's "strict hard-abort now"
   stance). The pre-flight is updated to profile the **compiled
   step** (prior 58/430-DtoH numbers were eager and are moot).

## Risks

- **Symbolic non-grammar compute** (truth layer, SVO, mereonomy,
  Sigma/Pi codebook ops) may have data-dependent control flow that
  needs real static rewrites (à la `compute_masked`). These are
  in-scope (non-grammar) and must be done; recon sizes the effort and
  any genuinely hard one is escalated to the owner *with data*, not
  silently deferred (only *grammar* breaks are silently deferrable).
- **Bucket explosion.** conceptualOrder × N-halving × token-slab
  could be large; bounding it (pad/quantize token slabs; cap levels)
  is a design task in the implementation plan.
- **`optimizer.step` eager** keeps a per-step host sync possible in
  the eager loop — acceptable (outside the captured graph), but means
  the captured unit is fwd+bwd only (not the doc's original
  fwd→step). Owner-approved.
- **metalbaby**: every GPU run tiny/bounded/single (prior OOM/reboot).

## Verification (every elimination step)

CPU representative order — `test_mm_xor test_mm_boolean
test_universality test_invertibility test_xor_spaces
test_basicmodel::TestNormalizeFlag test_lexicon_ownership
test_conceptual_bivector test_perceptual_bivector
test_phase2_pipeline_primitives` (full suite needs
`PYTORCH_ENABLE_MPS_FALLBACK=1` for the unrelated MPS
`_cdist_backward` gap) → tiny bounded metalbaby: O1 probe shows
dynamo frames > 0, then `fullgraph=True` non-grammar acceptance +
step-scoped `cudaMemcpyDtoH` profile.

---

## Phase 1 progress & pivotal finding (2026-05-16, autonomous run)

### Landed + verified (CPU 176 passed, 0 regressions; tree green)

- **O1 fixed.** `BasicModel.enable_compiled_step()` `torch.compile`s
  the `forward` callable and `runBatch` invokes it (not the eager
  `_orig_mod`). `ModelFactory.run` calls it; `_brick_preflight` now
  profiles the compiled step. MPS falls back to eager (torch.compile
  MPS fake-tensor device-propagation gap — same class as
  `_cdist_backward`); CUDA/CPU compile. Pinned by
  `test/test_compiled_step_invoked.py` (order-independent: forces
  `init_device("cpu")`; asserts dynamo frames > 0).
- **Recon harness** `test/brick_recon.py` (device-independent;
  `torch._dynamo.explain`) → tagged backlog
  `doc/plans/recon-breaks-MM_xor.md`.
- **Break #6 eliminated** (`str(util.DeviceHandle)` C-str in
  `_vq_chunk_budget`): now an import-time-memoized module constant
  (`_compute_vq_chunk_budget`), so the traced path returns a Python
  int — no device-str. Recon 33→32 breaks, non-grammar 9→8.

### Benchmarks (eager, GB10, metalbaby; honest current speed)

- **MM_5M** ≈ **140 row-ticks/s** (≈140K byte-tokens/s; *not*
  sentences/s — `samp/s = batchSize × batches/s`). Compute-bound:
  flat ~127–144 across bs16–256; mem linear 0.6→9.3 GB; **bs512 =
  hard perf cliff** (30 s/batch). Optimal **bs 32–64**.
- **MM_grammar** (toy XOR): launch-bound, fixed ~2.4 ms/batch,
  ~20 MB; throughput scales ~linearly with batch (bs512 ≈ 219K
  row-ticks/s). Not scale-matched to MM_5M (no stock 5M-param grammar
  config exists).

### PIVOTAL FINDING — the approved target is empirically blocked

Recon proves the **8 remaining non-grammar breaks are all
`BasicModel.forward` calling `Space`/`SubSpace` object orchestration**,
not data-dependent *tensor* compute:

| recon # | site | nature |
|---|---|---|
| 1,2,3 | `_lex_and_embed` / `_lex_batch` / `_to_text` | lexing: byte→str, host data-dependent input prep |
| 4 | `_embed` | codebook/OOV index resolution (host) |
| 0 | `empty_state` (Space) | SubSpace object construction |
| 5,6,7 | `copy_context` (Space.forward) | SubSpace object/registry plumbing |

`torch._dynamo.disable` does **not** help: a disabled call is still a
graph break under `fullgraph=True`. Satisfying the approved "whole
`BasicModel.forward` non-grammar break-free" therefore requires one
of:

- **(A) Inner-numeric-brick narrowing** — compile the dense tensor
  kernel; keep `Space`/lexer/`_embed` orchestration eager (it feeds /
  consumes `TheData` buffers). *This is the brainstorming option the
  owner explicitly rejected*, but recon shows it is the only
  near-term-feasible boundary. ~days.
- **(B) Rearchitect the Space pipeline** so `forward` is a pure
  tensor function (no SubSpace construction / `copy_context` /
  host lexing in the traced region — push all of it into the eager
  `TheData` producer, per the parent spec's write-back design).
  Correct end-state, but a large multi-file rearchitecture (the
  lexer/`_embed` relocation alone is parent-spec Phase 3). ~weeks;
  must be done deliberately + verified, not unsupervised.

**Recommendation:** (B) is the true target and subsumes the
parent-spec producer/write-back work, but it is too large/risky to do
unsupervised in one session. Suggest: adopt **(A) now** as an
intermediate (delivers a real captured numeric kernel + the speed
win, low risk, CPU-verifiable) while (B) proceeds incrementally
behind it (each lexer/`_embed`/Space-plumbing relocation is an
independent, verifiable step that also moves toward (B)). The
"whole-forward" acceptance criterion should be revised to **"the
compiled numeric kernel is `fullgraph` break-free; Space/lexer
orchestration is the eager producer"** — functionally the design's
own "relocate-to-producer" principle, just acknowledging the
compiled boundary sits at the numeric kernel, not `BasicModel
.forward`'s outer Space-orchestration shell.

### Stopped here deliberately

Per the bucket-4 discipline and the owner's repeated correction of
over-aggressive unsupervised change: the remaining work is an
architecture fork (A vs B) the owner must choose, and (B) is a
weeks-scale rearchitecture unsafe to attempt unsupervised. All landed
work is verified green; this is a truthful handoff, not a flail.
**Next action = owner picks A or B**, then resume the
recon→eliminate loop (`test/brick_recon.py` →
`doc/plans/recon-breaks-MM_xor.md`; CPU representative suite +
`test_compiled_step_invoked` as the per-step net).

---

## Path-B execution progress (2026-05-16, owner-directed tactics)

**Breaks 33 → 6 total; 9 → 4 non-grammar classes. CPU representative
suite 176 passed, 0 regressions after every step. Tree green.**

Eliminated (one class at a time, CPU-verified + recon-delta):

- **#6** `str(util.DeviceHandle)` in `_vq_chunk_budget` → import-time
  memoized module constant (`_compute_vq_chunk_budget`); traced path
  returns a Python int.
- **#5/6/7** (the keystone — dominated the *count*: per-stage
  `SubSpace(...)`+`copy_context` repeated across N-halving sub-graphs)
  → `ConceptualSpace` owns a **persistent `_subspaceForPS`** allocated
  in `__init__`; `forward` reuses it via `set_event(lifted)`. Dropped
  `copy_context` (the consumer `PerceptualSpace.forward` only
  `.is_empty()`/`.materialize()`s it and it is never the returned
  vspace, so the pipeline copy_context invariant does not apply).
  Recon 32 → 7.
- **#0** `SymbolicSpace.empty_state` → removed the
  `device=TheDevice.get()` kwarg from its `torch.zeros` (same root
  cause as #6: `DeviceHandle` in a traced factory). Default-device +
  the caller's existing `.to(inputData.device)` keep placement
  identical. Recon 7 → 6.

**Remaining: 4 non-grammar breaks, one cohesive cluster** —
`_lex_and_embed` / `_lex_batch` / `_to_text` / `_embed` (host
tokenisation byte→str/parse/OOV + codebook index) still executing
*inside* the compiled `forward` (via `_forward_per_stage` →
`InputSpace.forward`). This is precisely the **parent-spec Phase 3
eager-producer / pre-allocated-buffer relocation**: lex+embed must run
**eagerly in `runBatch` before** the compiled step, writing results
into persistent device buffers the compiled forward then *reads*
(tensor-only). Owner tactic confirmed correct (#1,2,3 pre-alloc
buffers; #4 codebook pre-alloc + view). The proven pattern from this
session (residual A/B host-token carry; the #5/6/7 owned-subspace
reuse) applies directly.

**Checkpoint rationale.** This last cluster is the largest, most
correctness-critical change (the entire input path) and is a
deliberate multi-step refactor the spec itself says must not be done
unsupervised. Stopping here is the responsible boundary: maximal
verified value delivered (33→6, all green), remainder precisely
scoped with a proven pattern. **Next:** execute the lex+embed →
eager-producer/pre-alloc-buffer relocation as its own careful pass
(parent Phase 3), `test/brick_recon.py` + the representative suite +
`test_compiled_step_invoked` as the per-step net; then Task 4
(`fullgraph=True` non-grammar acceptance gate + grammar isolation)
and a tiny bounded metalbaby capture/`cudaMemcpyDtoH` re-measure.

---

## Phase 3 COMPLETE (2026-05-16) — non-grammar fullgraph break-free

**recon (MM_xor, Phase-3-aware harness): `total graph breaks: 0;
graphs: 1`.** Cumulative 33 → 0 non-grammar breaks. CPU representative
suite 176 passed, 0 regressions. Tree green.

**Phase-3 relocation (the final #1-4 lexer/embed cluster):**
`runBatch` now runs the lex+embed STEM **eagerly** (`self._staged_in_sub
= self.inputSpace.forward(inputTensor)`) *before* invoking the compiled
step, then clears it post-call (consume-once; all attr-writes eager,
none traced). `_forward_per_stage` only **reads**
`getattr(self,"_staged_in_sub",None)` and falls back to inline
`inputSpace.forward` when not staged (eager/uncompiled/tests →
behaviour unchanged; verified). This moves host tokenisation / OOV /
codebook-index (`_lex_and_embed`/`_lex_batch`/`_to_text`/`_embed`)
entirely OUT of the traced region — exactly the parent-spec
eager-producer principle. `test/brick_recon.py` updated to mirror the
staging so it measures the real compiled path.

**Status of the design's acceptance criterion:** non-grammar path is
`fullgraph`-clean for MM_xor (no grammar path there). Remaining
follow-ups (Task 4): (1) flip `enable_compiled_step` to
`fullgraph=True` as the strict non-grammar gate; (2) grammar isolation
(`@torch._dynamo.disable` on the chart/`SyntacticLayer` entry) so
MM_5M / MM_grammar enforce non-grammar-only while grammar stays eager
(deferred per design); (3) update `_brick_preflight` assertion; (4)
tiny bounded metalbaby: confirm dynamo frames>0 on CUDA + re-measure
step-scoped `cudaMemcpyDtoH` (the prior 58/430 were eager & moot —
expect a large drop now the step is real & break-free, enabling
CUDA-graph capture).

---

## CORRECTION + COMPLETION (2026-05-16) — real fullgraph achieved; the explain recon was unfaithful

**Critical tooling correction.** Every "0 breaks / graphs: 1"
milestone above was measured with `test/brick_recon.py` using
`torch._dynamo.explain`. **`explain` is NOT a faithful
`fullgraph=True` oracle**: it traces *through* graph breaks and
silently tolerates `nn.Module.__setattr__` / SubSpace construction /
`copy_context` plumbing that a real `torch.compile(fullgraph=True)`
*rejects*. A real fullgraph compile of MM_xor still failed at
`_forward_per_stage` `inputData.to(TheDevice.get())`, then
`copy_context` (`self.wordSpace = other.wordSpace` →
`nn.Module.__setattr__` → `remove_from` → untraceable `isinstance`).
The "PIVOTAL FINDING / (B) is weeks-scale" assessment was an artifact
of working the wrong gate.

**Keystone fix (the owner's reset/owned-object instinct, surgically
applied).** `SubSpace.__setattr__` now routes the 5 *non-owning*
context back-channels (`wordSpace`, `errors`, `serial_cache`,
`valid_mask`, `stem_embedded`) through `object.__setattr__`, bypassing
`nn.Module.__setattr__`'s registration/`remove_from` path. This is
also a latent-bug fix (registering `wordSpace` as a SubSpace submodule
was an ownership cycle + state_dict pollution). One chokepoint cleared
**all ~14 `copy_context` call sites at once** — (B) was a surgical
change, not weeks. Plus: input device-placement moved to the eager
producer in `runBatch` (skip in-trace `.to(TheDevice.get())` when
staged).

**Real-gate result (faithful: real `fullgraph=True` + bounded
`runEpoch`, `Unsupported` raises on any break):**

| config | status | unique_graphs | calls_captured |
|---|---|---|---|
| MM_xor | FULLGRAPH-CLEAN | 2 | 1216 |
| MM_5M (production) | FULLGRAPH-CLEAN | 2 | 1833 |
| MM_grammar | FULLGRAPH-CLEAN | 2 | 1216 |

Grammar included — the keystone made the **whole** forward fullgraph,
so the design's grammar/non-grammar split and `@torch._dynamo.disable`
isolation are moot (a disabled call is itself a break under
fullgraph; not needed — no breaks remain).

**Landed (all CPU-verified, 207 representative tests pass, 0
regressions):** discourse-predict eager staging; codebook
`device=x.device`; `bincount`→static `index_add_`; input-device eager
placement; `SubSpace.__setattr__` context-attr chokepoint;
`enable_compiled_step` → `fullgraph=True` **unconditional**;
`_brick_preflight` gated behind `MODEL_DEBUG` (fullgraph is now the
*static* guarantee; skip the redundant Kineto preflight in production
→ faster compiles); `test/brick_recon.py` rewritten to the faithful
real-compile gate; `test_compiled_step_invoked` fixed to the correct
counter (`stats.unique_graphs`, not the absent `frames`).

**Follow-ups (not blockers):** (1) `unique_graphs: 2` over 2 batches
= one warm-up→steady recompile (guard specialization); fine for the
O1 "is it captured" gate, worth a look for steady-state CUDA-graph
capture. (2) XOR_grammar errors in any full runEpoch driver
(`_forward_head`/OutputSpace `reshape '[2,-1,12]'` vs size 64) — a
pre-existing eager IR-head/config-shape bug, orthogonal to graph
breaks (its own structural tests pass). (3) Pre-existing grammar
test-isolation: `test_chart_wordspace_wiring` fails after other
grammar tests share the `TheGrammar` singleton (reproduces with all
session changes reverted). (4) Bounded metalbaby: sync + one tiny run
to confirm fullgraph capture on CUDA + re-measure `cudaMemcpyDtoH`.

---

## REVISION (2026-05-16) — keystone reworked per owner directive (no __setattr__ override / no defensive getattr)

The `SubSpace.__setattr__` override (intercepting all attribute sets)
and the defensive `getattr(obj, "x", default)` probing were removed at
the owner's direction. Replaced with the project's **own existing
convention** for non-owning Module back-references:

- **`SubSpace.attach_wordSpace(ws)`** — mirrors the pre-existing
  `Space.attach_wordSpace` (Spaces.py): one named helper,
  `object.__setattr__(self,'wordSpace',ws)`, so WordSpace is not
  registered as a SubSpace submodule. Plain `self.wordSpace = ws` was
  verified to infinitely recurse `model.to(device)` (the
  `subspace→wordSpace→…→subspace` cycle the existing helper's docstring
  warns about), so bypassing Module registration is mandatory here, not
  stylistic; this matches the codebase's `_model` / `rule_codebook_host`
  back-ref idiom.
- **`copy_context` no longer touches `wordSpace`** — it is build-stable
  and stamped ONCE eagerly: `BasicModel` build loop calls
  `space.subspace.attach_wordSpace(self.wordSpace)` for every space.
  The traced forward never assigns the Module → no fullgraph break, no
  `.to()` recursion. The other 4 context attrs (Error/dict/tensor/bool)
  stay in `copy_context` (plain values, no Module registration).
- **Explicit attributes, no `getattr`** — `self._staged_in_sub`
  initialized in `BasicModel` build; `discourse` accessed directly
  (WordSpace always defines it; InterSentenceLayer always has
  stage_/clear_staged_prediction). Dead `_stack_text` (getattr-on-torch-
  internals) deleted from `brick_recon.py`.

**Final verification (faithful real `fullgraph=True` gate):** MM_xor /
MM_5M / MM_grammar all FULLGRAPH-CLEAN (1216/1833/1216 ops, 2 graphs,
no `Unsupported`). `model.to()` no longer recurses; wordSpace not a
submodule of any subspace. 207 representative tests pass; the lone
one-off `test_invertibility::test_training_preserves_invertibility`
blip is pre-existing training-RNG flakiness — passes 5/5 in isolation
(3 with changes, 2 with all session changes stashed) and in both
full-suite runs; uncorrelated with this work.

---

## METALBABY VERIFY (2026-05-16, CUDA GB10) — fullgraph confirmed; DtoH attributed

Synced (`make sync HOST=mb`), GPU idle, bounded run (bs=8, 2 warm-up
+ 2 profiled batches, `MODEL_COMPILE_MODE=default`).

- **Real `fullgraph=True` on CUDA: CONFIRMED.** `Model compiled
  (inductor, mode=default, fullgraph=True)`; dynamo
  `{'calls_captured': 1833, 'unique_graphs': 2}`; no `Unsupported`.
  The earlier stale-code crash (`bincount` unbacked-symint
  `IndexError`) is resolved.
- **Step-scoped `cudaMemcpyDtoH` re-measure: 64/step (post-warmup).**
  Attribution probe (`brick_attr_probe.py`, eager, MM_5M, CUDA,
  bounded) — 224 host-sync calls / 4 batches, ALL in eager components,
  **none in the compiled `_forward_per_stage`**:
  - 64 `__bool__` `Layers.py:6054` + 64 `item` `:6063` + 64 `item`
    `:6071` — `InterSentenceLayer.observe`/`_push_row` per-row
    `for b in range(B)` ARMA ring update. This is
    `@torch.compiler.disable`'d and runs **eager, post-body in
    runBatch by design** (the write-side counterpart of the staged
    `predict()`).
  - 8 each: `observe` `:6046`, `Layer.forward` `:7980`/`:6416`,
    eager stem `_embed_bpe` `Spaces.py:7107` — eager producer host
    work (tokenisation), by design.

**Conclusion:** the compiled numeric kernel is fullgraph-clean AND
host-sync-free on CUDA (no sync callsite attributes to it). The
residual ~64 DtoH/step are the eager, compiler-disabled discourse
ARMA ring update (`observe`→`_push_row`) — a known, bounded,
by-design eager component, not the compiled step.

**Follow-up opportunity (not a blocker for the fullgraph mandate):**
vectorize `InterSentenceLayer._push_row`/`observe`'s per-row
`for b in range(B)` (`.item()`/`bool()` ×3/row) into batched tensor
ops — same de-sync pattern used elsewhere. That removes the last
~64 DtoH/step and makes the *entire* runEpoch (eager producer
included) host-sync-free → eligible for full CUDA-graph capture
(`reduce-overhead`/`max-autotune`).

---

## DISCOURSE DE-SYNC — Step A landed (2026-05-16, metalbaby-verified)

`InterSentenceLayer.observe`'s per-row `for b in range(B)` ring update
+ `_push_row` (the `_s_count[b].item()` / `bool(active[b])` syncs) was
**vectorized** into masked tensor ops (`torch.roll` for the at-capacity
shift, `scatter_` for the fill slot, `torch.where` to gate per row).
Numerically identical: same fill-then-shift ring layout, same per-row
gating, same counts; the loss block is untouched (zero training-signal
risk). `_push_row` removed (sole caller was `observe`).

**metalbaby attribution probe (MM_5M, CUDA, 4 bounded batches):**
224 → **32** recorded host-sync calls (192 eliminated, 86%). The
dominant `observe`/`_push_row` 64×3 cluster — the ~64
`cudaMemcpyDtoH`/step — is gone. Residual 8/batch are all **outside
the compiled brick**: `observe:6046 bool(primed.any())` (loss gate,
left intact for exact loss semantics), two eager `Layer.forward`
lines, eager stem `_embed_bpe` (by-design producer host work). Brick
stays fullgraph-clean (MM_xor/MM_5M/MM_grammar). CPU representative
suite 207 passed, 0 regressions.

**Status:** the measurable perf goal (kill the ~64 DtoH/step → full
CUDA-graph-capture eligibility) is **met by Step A alone**, low-risk
and verified. The remaining plan items — split `observe` loss vs
commit, relocate commit+predict into `soft_reset`, SubSpace
pre-allocation — are now **architectural cleanliness only** (no
further perf delta; syncs already gone) and touch the
training-critical ARMA loss/backward-timing + cross-`runBatch`/
`soft_reset` plumbing. Sequenced as a deliberate follow-up rather
than bundled, per bucket-4 (one verified change at a time on a
training-critical path).

---

## ARCHITECTURE REFACTOR — runBatch<->model boundary (2026-05-16, metalbaby-verified)

Owner directive: "runBatch should interact only with the model, not
the model contents." Correctness finding surfaced first: the grammar
`soft_reset` (`_sentence_completed` / `_signal_sentence_completed_chart`)
fires only on chart sentence-completion and **never per-batch on the
IR/MM_5M path** -- relocating discourse there would silently stop ARMA
on MM_5M. So the boundary is a **model-level per-step lifecycle** at
per-batch timing (identical to the prior inline code), NOT the grammar
soft_reset.

Landed:
- **Step A** -- `InterSentenceLayer.observe`/`_push_row` per-row loop
  vectorized (masked `roll`/`scatter_`/`where`); numerically identical,
  loss untouched. metalbaby attribution: **224 -> 32** host-sync calls
  (192 / the ~64 DtoH/step eliminated). `_push_row` removed.
- **Step B1** -- model-level `_begin_step` / `_end_step` /
  `_discourse_arma_loss`. `runBatch` now calls only `self._begin_step`
  / `self._compiled_step` / `self._discourse_arma_loss` /
  `self._end_step` / `self.forward` -- zero reach into
  `self.wordSpace.discourse` / `self.inputSpace`. `_staged_in_sub` /
  `_compiled_step` / `_current_discourse_s` explicitly initialized at
  build, removing the `getattr(self,...,None)` defensive reads.
  Pure encapsulation: behaviour + per-batch timing identical.
- **Step C-safe** -- `getattr(cs,"_subspaceForPS",
  self._empty_subspace())` -> explicit `cs._subspaceForPS` (the owned
  subspace is always allocated in `ConceptualSpace.__init__`). Kills a
  flagged getattr AND the per-loop-iteration `_empty_subspace()`
  construction; traced graph leaner (`calls_captured` 1216->1096 /
  1833->1713), still FULLGRAPH-CLEAN.

Verification: CPU representative+discourse+grammar 207 passed
behaviour-identical at every step; MM_xor/MM_5M/MM_grammar
FULLGRAPH-CLEAN (faithful real-compile gate); metalbaby 32 host-syncs
(all outside the compiled brick: loss-gate `primed.any()` kept for
exact loss semantics, two eager `Layer.forward` lines, eager stem
`_embed_bpe`); `model.to()` no recursion.

Deliberately NOT done (rationale):
- **B2** (split `observe` loss vs commit; relocate commit to
  post-backward `_end_step`): risk on the training-critical ARMA path
  for **zero** perf gain -- ring-update already vectorized/sync-free
  (A) and encapsulated (B1). Not recommended.
- **Seed pre-alloc** (the 2 `_empty_subspace()` at `_forward_body`
  top, 2/forward, not per-stage): fullgraph-clean already; a shared
  persistent read-only seed needs proof it is never mutated -- a
  deliberate separate pass if desired.
- **`getattr(cs,"_subspaceForSS",CS_sub)`**: `_subspaceForSS` is set
  mid-forward (not `__init__`), so the `CS_sub` default is genuinely
  needed; removing requires deeper restructuring (real risk, no gain).

---

## DtoH reduction #1 + #3 (2026-05-16, metalbaby-verified)

Owner asked to pursue 0 DtoH ("OK to remove the failure, just don't
silently succeed").

- **#1 observe loss-gate** — replaced `if bool(primed.any()):
  diff[primed]...` (host gate + boolean-mask gather = per-step DtoH)
  with a fully-masked form (`(s_hat - pooled.detach()) * primed_mask`,
  `/ primed.sum().clamp(min=1)`). Numerically equivalent when >=1 row
  primed; cold-start now returns a training-neutral **0.0 tensor**
  instead of `None` (diff is exactly zero -> no gradient; logs
  `arma=0` rather than skipping). Contract change recorded; the 2
  pinning tests updated (`test_observe_pushes_and_returns_zero_on_
  cold_start`, `test_snapshot_alias_calls_observe`).
- **#3 divergence guard** — `if not bool(torch.isfinite(lossIn)
  .all()): raise` (deliberate per-step DtoH) -> `torch._assert_async(
  torch.isfinite(lossIn).all())`. Still **fails loud, never silent**:
  CUDA enqueues the assert on the stream (no DtoH) and a non-finite
  loss aborts the process at the next kernel launch; CPU asserts
  immediately (tests still catch divergence). Honors the fail-loud
  memory; only the *mechanism* (sync raise -> async abort) changed,
  per owner's explicit "OK to remove the failure" sanction.

**metalbaby:** compiled-path `cudaMemcpyDtoH` 16 -> **12**
(64 at session start); attribution 32 -> 16 calls. The 2 eliminated
sources are gone; the **only remaining syncs are the 2 inherent host
BPE tokenization `.tolist()`** (`ChunkLayer.forward`, `_embed_bpe`)
in the eager producer -- the design's accepted compile-scoped
boundary. FULLGRAPH-CLEAN (MM_xor/MM_5M/MM_grammar); 207 tests green.

**To reach literal 0** only the tokenization rearchitecture remains
(host-side token-id precompute / device tokenizer) -- a larger,
higher-risk input-path change, not in this scope. The compiled
numeric kernel is and stays sync-free; these 12 are eager-producer
host tokenization outside the captured graph.

---

## GPU BPE tokenizer (2026-05-16) — bit-identical but a perf regression; opt-in default-off

Goal: eliminate the remaining BPE-tokenization DtoH (`.tolist()` in
`ChunkLayer.forward` / `_embed_bpe`) by an on-GPU tokenizer over a
null-separated buffer, frozen-vocab (CPU-pretrain -> freeze ->
GPU-train) contract.

Built (`bin/bpe_gpu.py` + `_embed_bpe` dispatch in Spaces.py):
- `build_static_tables` (one-time, frozen): padded id->bytes, lengths,
  per-length sorted poly-hash index, `is_boundary`, `chunk_to_cb`
  (collapses the per-call latin1/key_to_index dict chain).
- `gpu_longest_match`: parallel per-position longest match via
  per-length window hash + `searchsorted` + **byte-verify**
  (collision-proof, exact; `_max_merge_len=9` so no single int64 pack).
- `gpu_chunk_ids`: greedy consumption as a bounded on-device scan.
- `segment_words` + `_bpe_emit_gpu`: **fully static** (no `.item()`,
  no boolean compaction) -- scatter into `[B*nObj]` buffers; shared
  `_bpe_finalize` tail.
- Trie path kept as `_embed_bpe_trie` (verified reference + non-frozen
  fallback). Bit-identical gate: `test/bpe_gpu_equiv.py` (inline
  GPU-vs-trie, frozen) PASS; keystone `test/bpe_gpu_match_check.py`
  PASS (chunk-ids == `ChunkLayer.forward`). 134 representative/BPE
  tests pass, fullgraph-clean unchanged (GPU tokenizer is in the eager
  stem, not the compiled forward).

**Honest measured outcome (metalbaby, frozen MM_5M, real
fullgraph=True):**

| | DtoH | throughput |
|---|---|---|
| trie BPE (baseline) | 20 | 23.7 rows/s |
| GPU BPE | 16 | 9.9 rows/s |

The premise did **not** hold: `_embed_bpe` was not the dominant sync
source (only ~4 of the residual DtoH; others are byte-cursor
`_lex_batch`/`_to_text`, `ChunkLayer.forward` on non-`_embed_bpe`
paths, the 2-graph boundary), so this does not reach 0 DtoH; and
tensorizing an inherently-sequential byte algorithm (per-length
hash-match + an N-iteration consumption scan = many tiny kernel
launches) is **2.4x slower** than the host trie at this batch scale.

**Decision:** opt-in, **default off** (`_bpe_gpu_enabled=False`).
Production stays on the verified trie path -- no regression. The GPU
tokenizer + the bit-identical gate are kept as reusable, correct
infrastructure for future perf work (the consumption scan is the hot
spot to rethink), but shipping it now would regress training 2.4x for
no DtoH win. Reaching true 0 DtoH requires attributing & eliminating
the *other* eager-producer host syncs first, not just `_embed_bpe`.

---

## 2-graph recompile — precise attribution + 2/3 causes eliminated (2026-05-16, metalbaby TORCH_LOGS=recompiles)

`TORCH_LOGS="recompiles,cudagraphs"` on frozen MM_5M / reduce-overhead
named the recompile guards exactly. The `unique_graphs=2` (which
prevents stable CUDAGraph capture) has THREE root causes:

1. **FIXED** — `self.inputs/percepts/concepts/symbols/outputs =
   <SubSpace>` at the end of `_forward_per_stage` (both the normal and
   empty-input blocks). `nn.Module.__setattr__` registered them in
   `self._modules` *inside the traced forward* → guard
   `len(self._modules)==13` failed → recompile. Now stored via
   `object.__setattr__` (the codebase's non-owning-ref idiom; reads
   work via `__dict__` precedence). Guard gone (verified: no longer in
   the recompile log).
2. **FIXED** — `self._subspaceForSS = vspace` (and the degenerate
   `_subspaceForPS` reassignment) in `ConceptualSpace.forward`: same
   class (adds a key to `cs._modules` on first forward → guard
   `len(cs._modules)==8` fails). Now `object.__setattr__`. Guard gone.
3. **NOT fixed (deeper)** — `SyntacticLayer._cursor_compose_gen`
   (`Language.py:_next_rule_name`): `if gen != self._cursor_compose_gen:`
   is a data-dependent control-flow branch on a per-batch-mutating
   nn.Module integer, *inside the traced grammar dispatch*. PyTorch's
   recommended `torch._dynamo.config.allow_unspec_int_on_nn_module =
   True` was applied (in `util.compile`) but does NOT resolve it: that
   flag unspecializes int *inputs*; a Python `if` on a per-batch-
   changing value still forces a guard/recompile. Eliminating it needs
   the grammar rule-cursor advance OUT of the traced numeric region
   (or a non-specializing representation) — a grammar-path change, not
   a config tweak.

Hot-path `getattr/hasattr` defensive lookups in the compiled-forward
path were also removed (direct access, per the no-defensive-protection
directive) — verified no regression, but measured to NOT affect
recompiles/DtoH/throughput (the dispatch-overhead hypothesis was not
supported by metalbaby data).

**State:** 212 representative/grammar/discourse/BPE tests pass;
MM_xor/MM_5M/MM_grammar fullgraph-clean; equivalence gate
bit-identical; `object.__setattr__` fixes + `util.compile` config +
getattr cleanup all verified non-regressing. Cause #3 still forces
`unique_graphs=2` on metalbaby (reduce-overhead), so CUDAGraph capture
still does not engage and throughput is unchanged — the remaining
blocker is precisely localized to the traced grammar cursor.
