# Substrate Refactor: Integrated Multi-Stage Plan

> **Single canonical plan** integrating the substrate refactor, the cross-position-mixing butterfly capability, and the signal-router-only parser commitment.
>
> Supersedes / consolidates:
> - [`doc/specs/2026-05-21-cross-position-mixing-via-nInputDim.md`](../specs/2026-05-21-cross-position-mixing-via-nInputDim.md) — butterfly impl details kept there as reference; architectural framing lives here.
> - [`doc/plans/2026-05-02-signal-router.md`](2026-05-02-signal-router.md) — detailed task-by-task implementation reference for Stage 3 (parser cleanup + signal router promotion).

---

## Goal

Refactor the substrate so that:

- **PS** is a single-direction input processor (pi + sigma + MPHF/codebook lookup).
- **CS** is the STM container and the **main grammatical CPU** — STM bookkeeping plus signal-router dispatch of read-only grammar ops.
- **SS** owns the **unified word lexicon codebook** and hosts grammar ops needing codebook write access.
- **Lift / Lower** retire as substrate operators; they become binary `GrammarLayer` subclasses dispatched by the signal router over STM pairs.
- **The signal router (`LanguageLayer`) is the canonical parser.** The CKY `Chart` class and the STM shift-reduce path retire entirely.
- **`GrammarLayer` gains butterfly mode** — packed-Parameter cascade for efficient O(N · log N · D²) cross-STM pairwise mixing. Inherited by all subclasses.

End state: STM is the basis; GrammarLayer ops are the operators; the signal router's per-position copy / reduce marginals are the coefficients of a soft superposition over compositions.

## Motivation (three drivers)

1. **Loop / arrival bottleneck dissolves.** With Lift / Lower as substrate ops, lowering had to run concurrently with new words still arriving — a pipeline hazard. Moving them into pairwise grammar ops separates input arrival (PS fills STM) from composition (grammar dispatches over STM contents).
2. **No inter-percept or inter-symbol mixing today.** `test/test_mm_xor.py::TestMMXorConvergence::test_convergence` plateaus at ~0.25 (target 0.20). The new design remediates via PiLayer butterfly mode on PS (inter-percept) and pairwise GrammarLayer ops over STM with butterfly cascades (inter-symbol / cross-STM soft superposition).
3. **Three parsers, one substrate is wasteful.** The CKY chart's `[B, N+1, N+1, C, D]` lattice is overkill for STM-sized N; STM shift-reduce is purely hard with no soft mixing for gradient flow. The signal router (`LanguageLayer`) was designed as the GPU-friendly replacement: per-position copy / reduce scoring with forward-backward DP marginals at training (soft), Viterbi at eval (hard).

Scope: substrate APIs, two-mode contract, parser consolidation, butterfly capability on GrammarLayer. Grammar rule-content design (specific Lift/Lower/Intersection semantics, chart of binary rules) is a separate later overlay.

---

# Design summary

## Two operating modes

| Mode | PS.forward argument | Iterations | STM behavior |
|---|---|---|---|
| **SERIAL** (= GRAMMATICAL) | `IS_t` per word | one per word; PS pushes one idea per word | shift-and-push (oldest dropped, newest at slot 7); signal router dispatches over STM per word or at sentence boundary |
| **PARALLEL** | `IS` once, then `CS` for `T-1` iterations | T = `<conceptualOrder>` | parallel write of T slots; signal router dispatches after STM population |

`PS.forward` takes one positional argument. SERIAL passes `IS_t` per word; PARALLEL passes `IS` on the first call then `CS` thereafter. Same layer, same signature, different inputs. SERIAL and GRAMMATICAL collapse at the architectural level — grammar dispatch is a chart/rule-catalog config, not a substrate mode.

## Forward APIs

```python
class PerceptualSpace:
    # Owns: self.pi, self.sigma, MPHF + index table → SS.codebook
    def forward(self, x_subspace):
        # pi + sigma + (MPHF → SS.codebook lookup when x is IS)
        ...

class ConceptualSpace:
    # Owns: STM (depth 7±2, Miller). Main grammatical CPU.
    def forward(self, new_idea_subspace):
        # STM shift / swap / push.
        # Dispatch read-only GrammarLayer ops via the signal router.
        ...

class SymbolicSpace:
    # Owns: codebook (unified word lexicon).
    def forward(self, stm_snapshot):
        # Invoke GrammarLayer ops via the signal router.
        # MUST host ops that need codebook write access.
        ...
```

`subspace.materialize()` returns `subspace.what[subspace.activation]` — nDim-wide selected codebook entry. SS codebook is ~1M today (verbal vocab), likely doubling.

## Per-Space ownership

| Space | Owns |
|---|---|
| **PerceptualSpace** | `self.pi`, `self.sigma`, MPHF + index table (referencing SS.codebook) |
| **ConceptualSpace** | STM (depth ~7), grammatical CPU (signal-router dispatch site for read-only ops) |
| **SymbolicSpace** | unified word lexicon codebook, signal-router dispatch site for codebook-write-required ops |
| **InputSpace / OutputSpace** | unchanged |

## Grammar op dispatch via the signal router

The signal router (`LanguageLayer` at [`bin/Language.py:1243`](../../bin/Language.py)) is the canonical parser.

- Slab `[B, N, D]` throughout (N = STM length).
- Per-layer scorers emit `copy_score: [B, N, R_copy]` and `reduce_score: [B, N-1, R_reduce]`.
- `binary_tiling_soft_dp` ([line 2101](../../bin/Language.py)) gives marginals at training (soft superposition); `binary_tiling_viterbi` ([line 2201](../../bin/Language.py)) gives the best tiling at eval.
- Multi-layer cascade builds hierarchical composition; each layer's hard-compacted slab feeds the next; the soft slab carries gradient.
- `Grammar.rule_probability(body)` ([line 857](../../bin/Language.py)) generalizes to the per-position, per-op score head producing copy/reduce scores. Dormant defaults (fold ops fire, negation ops don't) carry over as initial biases.
- Single-application enforcement (`_fired_bodies` / `reset_derivation`) carries over — row-local, fits the per-position routing model.

Two dispatch sites:
- **CS-side** routes read-only ops (most ops live here).
- **SS-side** routes codebook-write-required ops (lexicon entry creation, EMA updates, codebook expansion).

The router itself is shared; CS and SS each register their tier-specific op lists via `attach_unary_ops` / `attach_layer_ops` ([line 1307+](../../bin/Language.py)).

## STM details

- Capacity: 7 (Miller). `<ConceptualSpace><stmCapacity>` may override.
- SERIAL: shift-and-push per word; signal router dispatches per word or at sentence boundary.
- PARALLEL: parallel write of T slots from T iterations of PS; signal router dispatches after population.

## Butterfly mode on `GrammarLayer`

`GrammarLayer` base class gains a butterfly cascade mode for efficient cross-STM pairwise computation **within a single op's invocation**. Orthogonal to the signal-router's layer cascade (sequential composition); butterfly is the per-op all-pairs mixing.

- API: `GrammarLayer(... , butterfly=True, N=N)`. Inherited by all subclasses.
- Internal: packed `nn.Parameter[n_levels, N // 2, 2D, 2D]` (single Parameter), bit-reversal perm buffer, LDU per node via `InvertibleLinearLayer`, identity init.
- Each subclass implements `_butterfly_pair_op(x_pair, W_node)` for its per-pair op (PiLayer: atanh→matmul→tanh; SigmaLayer similar; Lift/Lower: sigma/pi semantics on the pair; Intersection/Union/Conjunction/Disjunction: tetralemma ops).
- Parameter savings: 8/10 → 0.38× single matrix; 32/10 → 0.16×; 128/10 → 0.05×.

## Reverse pipeline

No per-stage forward caches. Existing unrolled reverse chain `inputEstimate = reverse(forward(input))` on the **terminal STM contents** at the end of the forward chain. Signal router's soft slab is preserved for the inverse / generate pass so gradient reaches every op. `_ss_cache` / `_cs_cache` retire entirely.

---

# Stages

Stages 1 – 3 are baseline (the architectural shift). Stages 4 – 6 layer on the GrammarLayer-side migrations and butterfly capability. Stage 6 is optional.

Each stage has a goal, prerequisites, files modified, tests, and acceptance check.

## Stage 1: PS single-arg input processor, CS as STM holder, SS as codebook owner

**Goal.** Establish the new per-Space contracts. PS becomes a single-arg input processor; CS holds STM + does shift/push bookkeeping (no atomic forward operator); SS owns the unified word lexicon (migrated from InputSpace). Both modes (SERIAL/GRAMMATICAL and PARALLEL) wired into the forward dispatch. `_ss_cache` / `_cs_cache` retired.

**Prerequisites.** None.

**Files modified.**

- `bin/Spaces.py`:
  - `PerceptualSpace` ([line 7161](../../bin/Spaces.py)): collapse `pi_input` + `pi_concept` ModuleLists (lines 7407–7430) into `self.pi` + `self.sigma`. Add MPHF + index table referencing `SS.codebook`. `forward(x_subspace)` — single argument — does pi + sigma + (MPHF lookup when x is IS).
  - `ConceptualSpace` ([line 9358](../../bin/Spaces.py)): retire `sigma_percept` + `sigma_percept_1` / `sigma_percept_2` (lines 9473–9491). Retire `self.blend` if present from earlier drafts. `forward(new_idea_subspace)` does STM shift / swap / push.
  - `SymbolicSpace`: receive `self.codebook` (unified word lexicon, migrated from `InputSpace`'s text-mode codebook / BPE).
- `bin/Models.py`:
  - `_forward_body_per_word` ([line 5237](../../bin/Models.py)): SERIAL-mode entry — per-word PS.forward(IS_t); push to STM.
  - `_forward_per_stage` ([line 5836](../../bin/Models.py)): PARALLEL-mode entry — PS.forward(IS), then T-1 × PS.forward(CS); parallel STM write.
  - `_forward_body`: top-level dispatch between SERIAL and PARALLEL.
  - `_run_pipeline_rev` ([line 5606](../../bin/Models.py)), `_reverse_body` ([line 5569](../../bin/Models.py)): keep the unrolled reverse chain; reduce to terminal STM read.
  - **Retire** `_ss_cache` ([line 5953](../../bin/Models.py)), `_cs_cache`, any sibling per-stage capture lists.
- `data/*.xml`: enforce `IS.outputSize == CS.outputSize == PS.inputSize` at config load (loud error on mismatch).

**Tests.**

- New: `test_ps_single_arg_forward.py` — PS.forward(IS) and PS.forward(CS) shapes, identity-of-output-at-init.
- New: `test_cs_stm_shift_push.py` — STM transitions (shift in SERIAL, parallel-write in PARALLEL), capacity 7 enforced.
- New: `test_ss_codebook_ownership.py` — SS owns the unified lexicon; PS dereferences via MPHF.
- Existing forward / reverse roundtrip tests must still pass (operating on terminal STM, no per-stage caches).

**Acceptance.**

- Minimal config in each of SERIAL and PARALLEL modes trains end-to-end without crashing (a few epochs, loss-decreases-monotonically not required).
- `_run_pipeline_rev(forward(IS)) ≈ IS` (approximate, codebook-mediated).
- `grep -rn "sigma_percept\|pi_input\|pi_concept\|_ss_cache\|_cs_cache" bin/` returns only comments / dead-code-marker lines.
- Fail-loud numerical check (per project memory rule): Inf/NaN in loss raises with stack trace.

---

## Stage 2: `PiLayer` / `SigmaLayer` → `GrammarLayer` inheritance

**Goal.** Resolve the existing docstring / code mismatch at [`bin/Layers.py:1541`](../../bin/Layers.py): the docstring says PiLayer / SigmaLayer inherit from GrammarLayer, but the actual class declarations say `(Layer)`. Make the docstring true.

**Prerequisites.** Stage 1 doesn't strictly require this; both can happen in parallel, but cleaner to do Stage 1 first.

**Files modified.**

- `bin/Layers.py`:
  - `class SigmaLayer(Layer):` ([line 1762](../../bin/Layers.py)) → `class SigmaLayer(GrammarLayer):`.
  - `class PiLayer(Layer):` ([line 3387](../../bin/Layers.py)) → `class PiLayer(GrammarLayer):`.
  - Verify the `super().__init__(nInput, nOutput)` call signatures align with `GrammarLayer.__init__(self, nInput=0, nOutput=0)`.
  - Set `rule_name`, `arity`, `invertible`, `lossy`, `tier`, `reads_activation` class attrs if needed (likely Pi/Sigma stay anonymous as substrate folds — `rule_name=""` is the GrammarLayer base default and means the chart authority won't auto-register them).

**Tests.**

- New: `test_pi_sigma_grammarlayer_isinstance.py` — `isinstance(PiLayer(...), GrammarLayer)` and `isinstance(SigmaLayer(...), GrammarLayer)` both True.
- Run existing isinstance / GrammarLayer-consuming tests; ensure no behavioral change.

**Acceptance.**

- All existing tests pass.
- The docstring at [`bin/Layers.py:1541`](../../bin/Layers.py) is now accurate.
- `isinstance(layer, GrammarLayer)` consumers handle the broader set (no spurious behavior on PiLayer/SigmaLayer instances).

---

## Stage 3: Parser cleanup — retire Chart and STM shift-reduce; promote `LanguageLayer`

**Goal.** Remove the chart and STM-S/R parser paths; promote the signal router (`LanguageLayer`) to canonical, attached directly to `WordSpace`. Generalize `Grammar.rule_probability` into the per-position copy / reduce score head.

**Prerequisites.** Stage 1 (the substrate must accept whatever parser is wired in).

**Reference.** [`doc/plans/2026-05-02-signal-router.md`](2026-05-02-signal-router.md) has detailed Tasks 0–13 for the signal router's own implementation. That plan was written before this consolidation; this stage absorbs / extends it. Use its task list as the implementation breakdown for the router itself; this stage adds the retirement work that completes the consolidation.

**Files modified.**

- `bin/Language.py`:
  - **Retire** the `Chart` class ([line 3005](../../bin/Language.py)) and its CKY methods: `_chart_inside`, `_chart_outside`, `_compose_chart_cky`, `_compose_chart_cky_viterbi`, `_ensure_soft_chart_built`, `_chart_score` / `_chart_vec` allocations.
  - **Retire** the STM shift-reduce greedy left-corner code path ([line 7052](../../bin/Language.py)) and `_try_reduce` / `reduce_admissibility` helpers — **except** any shift / reduce primitives the signal router's compaction directly reuses.
  - **Retire** `<parserBackend>` / `<routerKind>` dispatch logic.
  - **Retire** chart-specific packed rule-table machinery (`_ensure_packed_table` at [line 1032](../../bin/Language.py)) if it's chart-only.
  - **Promote `LanguageLayer`** ([line 1243](../../bin/Language.py)) to a direct attribute on `WordSpace`. Remove the `Chart._ensure_signal_router` indirection.
  - **Generalize `Grammar.rule_probability`** ([line 857](../../bin/Language.py)) into the per-position, per-op score head producing copy/reduce scores. Dormant defaults carry over as initial biases on the score head.
  - **Keep** `binary_tiling_soft_dp` ([line 2101](../../bin/Language.py)) and `binary_tiling_viterbi` ([line 2201](../../bin/Language.py)).
  - **Keep** `_fired_bodies` / `reset_derivation` single-application enforcement.
- `bin/Models.py`: retire any chart-dispatch helpers (`_chart_generate_from_stm`, `_chart_compose_at_C`, references to `chart_score` / `chart_vec`).
- `bin/parse_state.py`: keep `ParseState` carrier; remove the dead chart-side branches.
- `data/*.xml`: retire `<parserBackend>`, `<routerKind>`, `<chartTau>`, `<chartTopK>`, `<chartNoiseEps>` knobs. Repurpose `<wMax>` as the signal router's STM-pair span bound or retire.

**Tests.**

- Retire `test_chart_*` files and chart-specific tests in `test_mm_*`.
- Retire STM shift-reduce parser tests.
- Extend `test_signal_router_*` tests per the existing signal router plan's Task structure.
- Add `test_signal_router_dispatch_routing.py` — read-only ops via CS-side router; codebook-write ops via SS-side router.
- Add `test_rule_probability_as_score_head.py` — score head produces `copy_score` / `reduce_score` consistent with the dormant defaults; learned predictors override.

**Acceptance.**

- `grep -rn "class Chart\b\|_chart_inside\|_chart_vec\|parserBackend\|routerKind" bin/` returns nothing live (only retired-doc breadcrumbs).
- Signal router trains end-to-end via `LanguageLayer` at WordSpace.
- `Grammar.rule_probability` consumers updated to read from the score head.
- ParseState populated correctly by the signal router; downstream consumers (SVO extraction, derivation trace) unaffected.

---

## Stage 4: `LiftLayer` / `LowerLayer` as binary `GrammarLayer` subclasses, attached to the signal router

**Goal.** Migrate substrate-level `Lift` (sigma synthesis) and `Lower` (pi lowering) into binary `GrammarLayer` subclasses; attach to the signal router as reduce ops.

**Prerequisites.** Stages 1 + 2 + 3.

**Files modified.**

- `bin/Layers.py`:
  - `LiftLayer` ([line 2297](../../bin/Layers.py)): rework as `class LiftLayer(GrammarLayer)` with `arity=2`, `rule_name="lift"`, `tier='C'`, `forward(idea_a, idea_b)` applying sigma-style synthesis to the pair, `reverse` symmetric. Drop the substrate-borrowed-from-PerceptualSpace.sigma pattern.
  - `LowerLayer` ([line 2440](../../bin/Layers.py)): rework as `class LowerLayer(GrammarLayer)` with `arity=2`, `rule_name="lower"`, `tier='C'`, `forward(idea_a, idea_b)` applying pi-style lowering. Drop the substrate-borrowed-from-ConceptualSpace.pi pattern.
- `bin/Language.py` / WordSpace: attach LiftLayer and LowerLayer as reduce ops on the signal router's C-tier `attach_layer_ops`.

**Tests.**

- New: `test_lift_lower_as_grammar_ops.py` — instantiation, forward / reverse roundtrip, attachment to signal router, dispatch via reduce_score.
- Confirm existing parser tests still pick up Lift / Lower correctly (now via signal router rather than substrate).

**Acceptance.**

- Lift / Lower fire as reduce-op alternatives in the signal router's binary tiling, weighted by `reduce_marginal_op[lift]` and `reduce_marginal_op[lower]`.
- No remaining call sites that reach into `PerceptualSpace.sigma` or `ConceptualSpace.pi` from `LiftLayer` / `LowerLayer`.
- Soft superposition over Lift vs Lower at a given pair site visible in the marginals during training.

---

## Stage 5: `GrammarLayer` butterfly mode

**Goal.** Add butterfly cascade mode to `GrammarLayer` base class. Available to all subclasses via `butterfly=True, N=N` constructor flag. Wire into `PerceptualSpace.pi` as the immediate use site; close the XOR convergence test as the acceptance signal.

**Prerequisites.** Stage 2 (`PiLayer` / `SigmaLayer` must already inherit from `GrammarLayer`).

**Files modified.**

- `bin/Layers.py`:
  - `GrammarLayer.__init__` ([line 1591](../../bin/Layers.py)): accept `butterfly=False, N=None` kwargs. If `butterfly=True`:
    - Validate `N` is a power of 2 (or pad strategy documented).
    - Set `self.butterfly = True`, `self.N`, `self.n_levels = int(log2(N))`.
    - Allocate `self.butterfly_W: nn.Parameter` of shape `[n_levels, N // 2, 2D, 2D]`, init via per-node LDU (`L=I, d=1, U=I`).
    - Register `self.butterfly_perms` buffer with bit-reversal indices per level.
    - Allocate matching LDU bookkeeping if needed (or apply LDU to per-slice `butterfly_W`).
  - `GrammarLayer.forward` / `reverse`: dispatch to `_butterfly_forward` / `_butterfly_reverse` when `self.butterfly`. Each iterates over levels, reshapes pairs, applies `_butterfly_pair_op` (or its reverse), and permutes.
  - Define a base `_butterfly_pair_op(x_pair, W_node)` returning `torch.einsum('bmi,mij->bmj', x_pair, W_node)`.
  - `PiLayer` / `SigmaLayer` / `LiftLayer` / `LowerLayer`: override `_butterfly_pair_op` with their per-pair semantics.
- `bin/Spaces.py`:
  - `PerceptualSpace`: instantiate `self.pi = PiLayer(D, D, invertible=True, butterfly=True, N=N)` where N is the configured per-position count.
- `data/MM_xor.xml` (and any other config that needs butterfly): add `<PerceptualSpace><butterfly>true</butterfly><butterflyN>8</butterflyN>` or wire defaults.

**Tests.**

- New: `test_grammar_layer_butterfly_forward_reverse.py` — at N = 2, 4, 8, 16, the cascade has `forward(reverse(y)) ≈ y` and `reverse(forward(x)) ≈ x`.
- New: `test_grammar_layer_butterfly_identity_init.py` — `forward(x) == x` at init within tolerance.
- New: `test_grammar_layer_butterfly_parameter_count.py` — count matches `n_levels · (N/2) · 4D²`.
- New: `test_grammar_layer_butterfly_subclass_dispatch.py` — PiLayer / SigmaLayer / LiftLayer / LowerLayer each apply their per-pair op semantics correctly.
- Existing `test_mm_xor.py::TestMMXorConvergence::test_convergence` and `::test_learns_xor_signal` should converge below 0.20 / 0.15 thresholds in the seeded `for seed in (42, 123, 7)` loop.

**Acceptance.**

- All butterfly unit tests pass.
- XOR convergence tests pass at the original thresholds without flakiness.
- No regression in MM_xor / MM_xor_loopback / MM_5M / MM_grammar test suites.
- `test/test_active_payload_audit.py` baselines unchanged.

---

## Stage 6 (optional): Butterfly on additional `GrammarLayer` subclasses

**Goal.** Extend butterfly mode to `IntersectionLayer`, `UnionLayer`, `ConjunctionLayer`, `DisjunctionLayer` for efficient cross-STM soft-superposition aggregation. Attach as additional reduce ops on the signal router.

**Prerequisites.** Stages 1 – 5.

**Files modified.**

- `bin/Layers.py`:
  - `IntersectionLayer` ([line 2142](../../bin/Layers.py)), `UnionLayer` ([line 2203](../../bin/Layers.py)), `ConjunctionLayer` ([line 2545](../../bin/Layers.py)), `DisjunctionLayer` ([line 2613](../../bin/Layers.py)): add `_butterfly_pair_op` overrides applying each op's tetralemma semantics on the pair input.
- `bin/Language.py` / WordSpace: attach the relevant ops as additional reduce-op alternatives on the signal router where the grammar wants them.

**Tests.**

- Per-op butterfly cascade unit tests (mirror Stage 5's tests).
- Soft-superposition end-to-end tests: a config with multiple competing reduce ops trains and shows the marginals shifting toward the correct op family.

**Acceptance.**

- Each op's butterfly mode passes its unit tests.
- A config exercising cross-STM aggregation (e.g. union of all STM contents) trains, with the relevant op's `reduce_marginal_op` dominating where expected.

---

# Cross-cutting concerns

## Reverse pipeline

The reverse chain is the existing unrolled `inputEstimate = reverse(forward(input))` over all Spaces, operating on the **terminal STM** cached at the end of the forward chain. Each fired GrammarLayer op's reverse is invoked in the reverse of dispatch order, weighted by the same soft marginals from the signal router's forward pass. PS.reverse turns the resulting STM contents into IS_hat surface reconstruction.

No per-stage caches. `_ss_cache` / `_cs_cache` retired in Stage 1.

## XML config migration

Stages 1 and 3 together produce these XML changes:

- **New / repurposed:**
  - `<PerceptualSpace><inputSize>` enforces single-knob input dim for both IS and CS sides.
  - `<PerceptualSpace><butterfly>` / `<butterflyN>` (Stage 5).
  - `<ConceptualSpace><stmCapacity>` keeps its meaning.
  - Optional: a signal-router top-K / noise knob if `<chartTopK>` / `<chartNoiseEps>` are replaced rather than retired.
- **Retired:**
  - `<WordSpace><parserBackend>`.
  - `<WordSpace><routerKind>`.
  - `<WordSpace><chartTau>`, `<chartTopK>`, `<chartNoiseEps>`.
  - `<WordSpace><wMax>` (optional: repurpose for signal router's STM-pair span bound).
  - The `<ConceptualSpace><blend>` knob from earlier drafts (never landed but listed here for completeness).

Sweep `data/*.xml` after Stage 3; loud error on retired knobs to catch drift.

## Documentation updates

After each stage:

- `doc/Architecture.md` — update the "Sigma / Pi ownership" table (lines 48–55), the cross-tier fold contract (line 56+), the modes section, and the parser-backend section.
- `doc/Spaces.md` — update "Sigma / Pi ownership (2026-05-13 rebalance)" section, `SymbolicSpace` section (now owns unified lexicon), `ConceptualSpace` section (STM + grammatical CPU), Lift / Lower factorization section (now binary grammar ops).
- `doc/specs/2026-05-21-cross-position-mixing-via-nInputDim.md` — already has absorption note pointing to this plan.
- `doc/plans/2026-05-02-signal-router.md` — annotate that it's been consolidated into this master plan (its Tasks 0–13 are the detailed implementation reference for Stage 3 of this plan).
- `doc/plans/*` — older plans referencing the chart or `parserBackend=stm` need superseded-by notes.

## Test infrastructure

Per the project's memory rule, prefer **targeted pytest node IDs** during iteration. Full `test/` suite only for explicit gates (before declaring a stage done, before merging).

Per-stage test additions are listed in each stage. Cross-stage: forward / reverse roundtrip and IR loss tests should pass after every stage.

## Numerical fail-loud rule

Per project memory: any Inf / NaN in loss MUST raise with a stack trace. Never `nan_to_num`'d silently. Verify after Stage 5 (butterfly cascade introduces new sources of numerical sensitivity).

---

# Whole-plan acceptance

Plan is complete when:

1. All six stages' acceptance criteria hold.
2. `grep -rn "Chart\|_chart_inside\|_chart_vec\|parserBackend\|routerKind\|sigma_percept\|pi_input\|pi_concept\|_ss_cache\|_cs_cache" bin/` returns only retired-doc breadcrumbs and intentional callsite comments — no live code.
3. `test_mm_xor.py::TestMMXorConvergence::test_convergence` passes at the original 0.20 threshold across seeded inits.
4. MM_xor / MM_xor_loopback / MM_5M / MM_grammar test suites pass.
5. Forward / reverse roundtrip approximates input across all four configs.
6. Documentation reflects the new architecture (Architecture.md, Spaces.md, this plan, the absorption-noted subordinate docs).
7. XML configs cleaned up; loud error on retired knobs.

---

# Risks / open items

1. **Signal-router maturity.** The signal router is younger code than the chart. Expect rough edges during Stage 3; allow time for hardening before declaring Stage 3 done. The existing 2026-05-02 plan's TDD task list helps.
2. **LDU + butterfly composition.** LDU per butterfly node within a packed Parameter is not how `InvertibleLinearLayer` is used today (it's per-layer, not per-slice). Stage 5 will need to verify the LDU-per-slice composition works; may need a small refactor of `InvertibleLinearLayer` to support batched per-slice LDU.
3. **Power-of-2 N constraint.** STM = 7 isn't a power of 2. Stage 5 needs a pad strategy (e.g. pad-to-8 with identity-init on the extra slot, mask it from the loss). Document the strategy explicitly.
4. **MPHF + index table on PS.** PS dereferences SS.codebook via MPHF; the MPHF code already exists ([Architecture.md](../Architecture.md) mentions it under "per-percept gaussian-attentional input → MPHF→table percept mapping"). Stage 1 needs to lift the MPHF infrastructure to PS-owned (currently lives elsewhere).
5. **`<conceptualOrder>` semantics shift.** Used to mean "T iterations of the two-loop body." Now means "T iterations of PS.forward(CS) in PARALLEL mode." Mode-conditional semantics; documented in the two-modes table.
6. **Some grammatical artifacts still live on `WordSpace`.** Chart-time caches, parse-state buffers. They stay as orthogonal cleanup; this plan does not retire them.
7. **GRAMMATICAL-mode pipelined parallelism.** Earlier-draft notion of "PS / SS Lift / Lower in parallel with CS reduce." Now subsumed by the signal router's layer cascade — sequential composition decisions per layer; no separate pipelined model needed.

---

# Resolved design decisions (locked)

1. Two operating modes: SERIAL/GRAMMATICAL and PARALLEL.
2. `PS.forward(x_subspace)` takes one argument.
3. CS is the main grammatical CPU (STM + read-only op dispatch via the signal router).
4. SS hosts codebook-write-required ops; owns the unified word lexicon codebook.
5. `self.blend` weight retired; replaced by signal router's per-position copy / reduce marginals.
6. `<conceptualOrder>` knob drives PARALLEL-mode iteration count.
7. Lift / Lower are binary `GrammarLayer` subclasses attached to the signal router as reduce ops.
8. PiLayer / SigmaLayer inherit from `GrammarLayer` (Stage 2 makes the code match the docstring).
9. `GrammarLayer` butterfly mode lives on the base class as a constructor flag, with packed `nn.Parameter`, bit-reversal perms, LDU per node, identity init.
10. `_ss_cache` / `_cs_cache` retire entirely; reverse operates on terminal STM.
11. Signal router (`LanguageLayer`) is the canonical parser. CKY `Chart` and STM shift-reduce retire.
12. `Grammar.rule_probability` generalizes to the signal router's per-position copy / reduce score head; dormant defaults carry over as initial biases.
