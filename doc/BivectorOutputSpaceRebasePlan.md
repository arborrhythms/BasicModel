# Bivector Retirement — completion record (was: OutputSpace re-base fix)

> Status: **implemented** (Phases 0–5) on the bivector-retirement
> branch. This file was the original "MM_5M_bivector OutputSpace
> re-base + fail-fast checkpoint guard" design; it is rewritten here as
> the as-built record because **Bug #1 / Fix #1 were structurally
> eliminated** (not patched) and **Fix #2 was retained and reframed**
> as the migration-cliff catcher. See `BivectorRetirementPlan.md` for
> the original handoff plan.

## Outcome

**Bug #1 (OutputSpace re-base `[B,V,2]` vs stale-width crash) — gone,
structurally.** The crash existed only because a `bivectorOutput=true`
ConceptualSpace emitted a catuskoti `[B, V_C, 2]` slab while the
OutputSpace head was sized from the stale declared `outputShape[1]`.
The bivector terminal head was **retired**: every inter-component
interface now carries a single signed Degree-of-Truth scalar, so the
terminal C stage emits scalar and `.nOutputDim == .outputShape[1]` for
**every** config. The 1-line "Fix #1" (read `nOutputDim` at the re-base
site) was therefore **not applied** — there is no longer a width to
reconcile.

**Fix #2 (fail-fast on autoload architecture mismatch) — retained &
reframed.** `BaseModel.load_weights` gained `require_match: bool =
False`. On a genuine mismatch (`mismatches or missing or
fatal_unexpected`) it now `raise ValueError("\n".join(lines))` when
`require_match` is set, instead of soft-warn + later crash on fresh
weights. The autoload caller passes `require_match=True`; all
non-autoload callers keep the soft-warn (vocab-grow / benign-stale-key
loads still work). This is now the **bivector-retirement
migration-cliff catcher**: pre-refactor checkpoints (e.g.
`data/MM_5M_bivector.ckpt`, owner-managed, untouched) are invalidated
by the scalarization, so autoloading one now fails fast with the
already-actionable "correct the XML … or delete/move" message.
Regression-tested by
`test_basicmodel.py::TestWeightShapeMismatch::test_require_match_raises_on_shape_mismatch`.

## The fixed invariant

The 4-valued bivector `[aP, aN]` (affirm / deny) exists **only** as
(1) an internal accumulator inside `TruthLayer`, and (2) a terminal
client-facing truth assessment. It is **never** a transport between
components. Every inter-component interface carries a single signed
Degree-of-Truth in `[-1, +1]` (Strong Kleene per `../doc/Logic.md`:
`and=min`, `or=max`, `not(a)=−a`, `non(a)=1−2|a|`). Evidence arrives
as signed scalars (per-source `trust`); `TruthLayer` accumulates them —
paraconsistency is the accumulator's job, not the wire's.

Two owner amendments to the original plan were folded in:

* **Tri-state `<codebook>`** — `none` / `quantize` / `project` (was a
  boolean). Migration is behaviour-preserving: `true → quantize`,
  `false → none` (all 29 `data/*.xml` + the `model.xml` default).
  `project` selects a scalar `ProjectionBasis` (the size-changing,
  LDU-invertible codebook) — kept, not deleted; it emits a signed
  scalar `[B, N, 1]`, not the `[B, N, 2]` bivector.
* **Keep the bivector operations for future use.** `NotLayer` /
  `NonLayer` / `TrueLayer` / `FalseLayer`, `Ops`'
  negation/non/conjunction/disjunction monotonic kernels,
  `_check_bivector_shape`, and the `TruthLayer` accumulator are
  retained as library ops (still exercised by their op-level tests).
  Only the bivector-activation **paths through SubSpace / the Space
  classes** were deleted (not merely made obsolete). Verified
  precondition: bivector ops are **not** needed for the positive
  monotonicity of the sigma/pi mereological order preservation — that
  rests on W≥0 (softplus) maps and the scalar clipped-cosine
  `Ops._part_kernel`, and `test_parthood_orders.py` (15/15) exercises
  every order-preservation invariant in `scalar=True` mode.

## Phases as executed

* **Phase 0** — scalar-substitute generic test configs to `MM_xor.xml`;
  honest regression floor recorded (1079 passed / 5 pre-existing
  failures: `test_impenetrable::TestOverlapPenalty` ×3,
  `test_subspace_context`, `test_util_device`).
* **Phase 1** — consolidate truth computation into `TruthLayer`
  (`luminosity` / `isConsistent` / `ground` / `extrapolate` moved in;
  `Mereology` / `Models` become thin delegators). Pure extract/move,
  zero regressions.
* **Phase 2** — retire the `bivectorOutput` flag + the Q2
  promote/lower substrate; tri-state `<codebook>`; `ProjectionBasis`
  kept but scalar (`forward → [B,N,1]`, `reverse` accepts
  scalar/legacy); delete the bivector configs/tests
  (`MM_5M_bivector.xml`, `MM_xor_bivector.xml`,
  `test_bivector_basis` / `test_conceptual_bivector` /
  `test_perceptual_bivector`, `test_csbp::TestBivectorEndToEnd`,
  `test_invertibility::test_C_to_S_to_C_chain`); add the
  `MM_xor_loopback.xml` fixture; `XOR_exact` / `XOR_recon` → `project`
  with their bivector-width params scalarized `2→1`.
* **Phase 3** — delete the bivector-activation paths in the Spaces
  substrate: `ActiveEncoding.nDim 2→1`; `SubSpace.set_activation` drops
  the `nd==2` lift; `resolve` / `_compute_active` /
  `set_activation_from_event` / `SymbolicSpace.resolve` carry/derive a
  signed scalar (no `pos − neg` pole collapse); `activation_presence`
  returns `|DoT|`; the inert `ShortTermMemory._truth_tags` buffer +
  accessors deleted. Bivector-substrate-contract tests
  retired/rewritten to the scalar contract.
* **Phase 4** — `TruthLayer.falsity_penalty` is representation-agnostic
  (no `[...,:2]`; folds via the kept `Ops._disjunction_kernel`);
  `truth_modulated_loss`'s retired `symbol_acts[..., :2]` pole slice
  replaced by the `TruthLayer`-internal accumulator read; `load_weights`
  Fix #2 (above).
* **Phase 5** — `TruthLayer.assess()` exposes the terminal
  paraconsistent triple `support = aP·(1−aN)`,
  `conflict = aP·R_N + aN·R_P`,
  `ignorance = (1−max(aP,aN))·(1−max(R_P,R_N))` from the accumulator;
  cached on `store_truths` and surfaced as `serve.py`'s
  `response["truth_assessment"]` (terminal output, not the token
  stream). Gated by
  `test_clarifications.py::test_assess_contested_vs_silent` — a
  TruthSet that splits on a proposition reports high `conflict`; one
  silent on it reports high `ignorance`; the two are provably distinct
  (the degeneracy a scalar `aP − aN` collapse loses).

## Resolved symbol_states gate

The Phase 2→4 seam (`materialize("active")` must be scalar before
`falsity_penalty` / `tetralemma_balance_penalty` are rescalarized) was
honoured: Phase 2 made `.what` scalar, Phase 3 scalarized the
substrate, and only then (Phase 4) did `truth_modulated_loss` stop
slicing `symbol_acts[..., :2]` and read the accumulator instead.

## Verification

Per-phase targeted gates green; full suite at the Phase-2 boundary:
1047 passed / 5 pre-existing failures (the 1047-vs-1079 delta is
exactly the legitimately-retired bivector tests/configs — zero
non-bivector regressions). `project` (e.g. `XOR_exact`) trains an
epoch with no `[B,V,2]` reshape crash. Run with
`BASICMODEL_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python
-m pytest test/ -q`.
