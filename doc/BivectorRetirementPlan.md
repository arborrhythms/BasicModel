# Bivector Retirement & TruthLayer Consolidation

> Handoff plan for a fresh implementation conversation. Architecture is **decided** —
> this is execution, not design. Do not re-litigate or propose "bridge" variants.

## Context

**Why this change.** The model stores a 4-valued "bivector" truth representation
`[aP, aN]` (affirming / denying evidence — catuṣkoṭi corners TRUE `[1,0]`,
FALSE `[0,1]`, BOTH `[1,1]`, NEITHER `[0,0]`). Investigation established that the
bivector is *stored* but **not utilized**: every site that scores a proposition
against the truth region collapses it to a signed scalar (`SubSpace.resolve()` →
`pos − neg`, inside every forward; `falsity_penalty` → union L2 norm;
`_luminosity_truth_fold` → `pos−neg` dot). So "the region contradicts p" and
"the region is silent on p" both read as low truth — the exact degeneracy the
bivector was meant to prevent. The WikiOracle social spec
(`../doc/Truth.md`, `../doc/Logic.md`) *requires* the BOTH corner ("the same
Truth is held to be both True and False, each contextualized in its own
reference frame") yet specifies the wire algebra as ternary signed scalar — the
spec-level statement of the same collapse.

**The decision (fixed invariant).** The bivector exists ONLY as an internal
accumulator inside `TruthLayer`, and as a terminal client-facing truth
assessment. It is **never** a transport between components. Every
inter-component interface carries a single signed scalar Degree-of-Truth in
`[-1,+1]` (Strong Kleene per `../doc/Logic.md`: `and=min`, `or=max`,
`not(a)=−a`, `non(a)=1−2|a|`). Evidence arrives as signed scalars (per-source
`trust`); `TruthLayer` accumulates them into `[aP,aN]` internally — paraconsistency
is the accumulator's job, not the wire's.

**Intended outcome.** (1) Retire the subspace bivector substrate and the
`bivectorOutput` head — this *structurally eliminates* "Bug #1" of
`basicmodel/doc/BivectorOutputSpaceRebasePlan.md` (the `[B,V,2]` vs stale-width
crash) rather than patching it. (2) Consolidate all truth computation into
`TruthLayer`. (3) Keep & reframe that doc's "Fix #2" (fail-fast on autoload
arch mismatch) — now the migration-cliff catcher. (4) Expose the
support/conflict/ignorance assessment to the client.

`isPart`/`isEqual` stay vector-returning geometric cone ops — consumed by
TruthLayer's consistency logic but **not themselves paraconsistent**. No change.

## Resolved correctness gate (verified)

`symbol_states[-1]` (built via `SubSpace.materialize()` default `mode="active"`,
[Spaces.py:4554](bin/Spaces.py:4554)/4569 → muxed event, bivector at `[...,:2]`;
cache at [Models.py:4196](bin/Models.py:4196)/4259, consumed
[Models.py:2956](bin/Models.py:2956)) is a **pre-collapse bivector layout
today**. Therefore `falsity_penalty`/`tetralemma_balance_penalty`
rescalarization MUST be Phase 4 — after Phase 2 makes `.what` scalar. This is
the Phase 2→4 seam; do not rewrite `falsity_penalty` before Phase 2 lands and
`materialize("active")` is re-confirmed scalar on a live forward.

## Critical files

- [bin/Spaces.py](bin/Spaces.py) — `ActiveEncoding`, `SubSpace.resolve`/`set_activation`/`materialize`, `ProjectionBasis`, `_bivector_output`, `_build_what_basis`, `ShortTermMemory._truth_tags`, accumulateTruth recording block.
- [bin/Layers.py](bin/Layers.py) — `TruthLayer` (4642–~5650), grammar ops `NotLayer`/`NonLayer`/`TrueLayer`/`FalseLayer`, `Ops._negation/_non/_conjunction/_disjunction_kernel`, `_check_bivector_shape`.
- [bin/Models.py](bin/Models.py) — `isConsistent`/`ground`/`extrapolate`/`store_truths`, `load_weights` (Fix #2), `truth_modulated_loss` caller.
- [bin/Mereology.py](bin/Mereology.py) — `Luminosity`/`_luminosity_truth_fold` (move into TruthLayer).
- [bin/Language.py](bin/Language.py) — `truth_modulated_loss`.
- [bin/serve.py](bin/serve.py) — truth ingestion + client response (Phase 5).

## Phased execution

### Phase 0 — Baseline + scalar substitute (no logic change)
- Repoint every generic reference of `data/MM_xor_bivector.xml` to the existing
  non-bivector **`data/MM_xor.xml`** (grammar, scalar — already the analogue per
  the rebase doc's config table). Affected: `test_invertibility.py`,
  `test_brick_no_sync.py`, `test_xor_spaces.py`, `test_csbp.py`,
  `test_conceptual_stm.py`, `test_lift_lower_factorization.py`,
  `test_parthood_orders.py`, `test_perceptual_loopback.py` (grep `MM_xor_bivector`
  to confirm the full set). Strip `<bivectorOutput>` lines from `data/XOR_exact.xml`
  in place (general config, keeps its purpose).
- **Gate:** full suite `BASICMODEL_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=1
  .venv/bin/python -m pytest test/ -q`. **Record pass count — the regression
  floor.** Rollback: trivial.

### Phase 1 — TruthLayer consolidation (give the accumulator a home)
*Before substrate retirement: the `[aP,aN]` accumulator must exist in one place
before Phase 2/3 removes the substrate that currently carries 4-valued state.*
- Move computation INTO `TruthLayer` ([bin/Layers.py:4642](bin/Layers.py:4642)+):
  `Mereology.Luminosity`/`_luminosity_truth_fold`
  ([Mereology.py:861](bin/Mereology.py:861)/970); disjunction fold from
  `Models.isConsistent()` ([Models.py:939](bin/Models.py:939)); `Models.ground()`
  ([Models.py:986](bin/Models.py:986)); `Models.extrapolate()`
  ([Models.py:1100](bin/Models.py:1100)-1202). Reuse the existing internal
  accumulator in `record()`/`record_batch()` bivector path
  ([Layers.py:4719](bin/Layers.py:4719)/4798) — it already pole-routes via
  `Ops._negation_kernel` for `degree<0`; this is where the bivector stays.
- Call sites become thin delegators (NOT moved): `truth_modulated_loss`
  ([Language.py:5419](bin/Language.py:5419), caller
  [Models.py:2920](bin/Models.py:2920)); accumulateTruth block
  ([Spaces.py:10114](bin/Spaces.py:10114)-10131); `store_truths`
  ([Models.py:1914](bin/Models.py:1914)); `serve.py` 218/251.
- Do **not** touch `falsity_penalty` internals yet (still bivector until Phase 2).
- **Gate:** `test_truth_layer_record_many.py test_resolve_luminosity.py
  test_clarifications.py test_universality.py test_mereology.py
  test_quaternary_corners.py test_reasoning.py` green. Rollback: medium
  (pure extract/move).

### Phase 2 — Retire `bivectorOutput` (terminal scalar emission)
*Flag → substrate order: `_build_what_basis` runs inside `__init__`
([Spaces.py:5461](bin/Spaces.py:5461)). Kill the flag first so `ProjectionBasis`
is never constructed → Phase 3 deletes dead, not live, code.*
- Delete the 3 soft reads + attribute: Perceptual
  [Spaces.py:6500](bin/Spaces.py:6500)-6503, Conceptual 8407-8410, Symbolic
  9033-9036. Collapse every `if self._bivector_output:` to its non-bivector
  branch (7674, 7710, 8440, 8759, 8784, 8805, 8845, 9356, 10158, 10308 — grep
  `_bivector_output` for the full set).
- Delete `ProjectionBasis` class ([Spaces.py:2124](bin/Spaces.py:2124)-2290)
  once constructors (8554, 9357) are gone; bivector `_build_what_basis` branches
  collapse to the legacy/`None` path. Remove PerceptualSpace Q2
  promote/lower (`_q2_promote_activation` + inverse + call sites ~7356-7714).
- Terminal CS now emits scalar ⇒ `.nOutputDim == .outputShape[1]` for ALL
  configs ⇒ Bug #1 structurally gone. **Do NOT apply the old 1-line "Fix #1".**
- **Delete bivector configs:** `data/MM_5M_bivector.xml`,
  `data/MM_xor_bivector.xml`. **Delete bivector-specific tests:**
  `test/test_bivector_basis.py`, `test/test_conceptual_bivector.py`,
  `test/test_perceptual_bivector.py` (delete *in this phase* — earlier masks
  Phase-1 regressions, later leaves the suite red).
- Re-confirm the symbol_states gate: `materialize("active")` now returns scalar
  `.what` (this enables Phase 4).
- **Gate:** `test_invertibility.py test_brick_no_sync.py test_xor_spaces.py`
  (on `MM_xor.xml`), `test_basicmodel.py`; smoke `python bin/train.py --model
  data/MM_5M_bivector.xml ...` — file is deleted, so instead smoke a surviving
  grammar config end-to-end (1 epoch, 3 batches) with no reshape crash.
  Rollback: HIGH — keep Phase 2 as one revertable commit.

### Phase 3 — Retire the subspace bivector substrate
*After Phase 2: no `[B,N,2]` is produced, so these become identities, not
mid-pipeline shape breaks.*
- `ActiveEncoding` ([Spaces.py:207](bin/Spaces.py:207)/221) → scalar width.
- `SubSpace.resolve()` ([Spaces.py:4492](bin/Spaces.py:4492); callers ~10015/
  10150/10236): `.what` already scalar DoT → `pos−neg` collapse becomes
  identity (keep the `_apply_active_selection` wrapper).
- `set_activation` ([Spaces.py:4042](bin/Spaces.py:4042)): delete the `nd==2`
  lift (~4061-4064 `relu(±x)` stack); store scalar `[B,N]`;
  `activation_presence` `.max(-1)` branch → scalar passthrough.
- Grammar ops → Logic.md scalar Kleene: `NotLayer`
  ([Layers.py:2070](bin/Layers.py:2070)/2082) flip→`−a`; `NonLayer`
  (2122/2134) `1−x`→`1−2|a|`; `TrueLayer` (2785/2819) / `FalseLayer`
  (2835/2860). Delete `_check_bivector_shape`
  ([Layers.py:1693](bin/Layers.py:1693)) + callers. `Ops` kernels lose
  monotonic/bivector branches: `_negation_kernel` (7225→`−x`),
  `_non_kernel` (7245→`1−|clamp|`), `_conjunction`/`_disjunction_kernel`
  (7165/7177→scalar min/max); `intersection`/`union` aliases follow.
- `ShortTermMemory._truth_tags` ([Spaces.py:8106](bin/Spaces.py:8106)-8127;
  accessors 8322/8339): **confirmed inert** (only `.zero_()` writes;
  `get_truth_tag` never called) — delete buffer + both accessors.
- **Gate:** Phase-1 truth set + `test_quaternary_corners.py
  test_partition_resolve.py test_reasoning.py`, then FULL suite vs Phase-0
  floor. Rollback: HIGH (grammar semantics) — Layers and Spaces independently
  revertable.

### Phase 4 — Rescalarize falsity_penalty + fail-fast guard
- `TruthLayer.falsity_penalty` ([Layers.py:5478](bin/Layers.py:5478)-5530):
  operands now scalar `[B,N,D]`; drop the `[...,:2]` pole assumption; fold via
  the now-scalar `Ops._disjunction_kernel` (= `max`). In `truth_modulated_loss`
  ([Language.py:5503](bin/Language.py:5503)-5512) the `>=2` / `[...,:2]` guard
  no longer denotes poles — replace it with the TruthLayer-internal accumulator
  read feeding Phase 5 (the only legitimate remaining bivector surface).
- **Keep & reframe Fix #2:** `load_weights`
  ([Models.py:1346](bin/Models.py:1346)) add `require_match=False`; mismatch
  block (~1426-1442) → `raise ValueError("\n".join(lines))` when
  `require_match and (mismatches or missing or fatal_unexpected)`; autoload
  caller (~[Models.py:672](bin/Models.py:672)) passes `require_match=True`.
  Non-autoload callers keep soft-warn. The refactor invalidates
  `data/MM_5M_bivector.ckpt` (owner-managed, untouched) → this guard is now the
  migration-cliff catcher.
- **Gate:** FULL suite vs floor; a deliberately shape-mismatched state_dict
  raises ValueError; `require_match=False` still soft-warns; autoload against
  the stale ckpt now fails fast with the actionable message. Rollback: LOW.

### Phase 5 — Client-facing truth assessment (in scope)
- Add a paraconsistent read on `TruthLayer` (internal accumulator → terminal
  output): per proposition vs region, expose
  `support = aP·(1−aN)`, `conflict = aP·R_N + aN·R_P`,
  `ignorance = (1−max(aP,aN))·(1−max(R_P,R_N))`. Build on the existing
  `consistency(return_report=True)` / `_detect_contradictions` /
  `suggest_clarifications` path ([Layers.py:5257](bin/Layers.py:5257)/5405) —
  it already does cross-source opposite-polarity detection; generalize it to
  return the triple.
- Surface via [serve.py:251](bin/serve.py:251) response (extend the existing
  `clarifications` field, or add a `truth_assessment` field) — terminal output,
  NOT the inference token stream.
- **Gate:** extend `test_clarifications.py` with a contested-vs-silent case
  (a proposition the TruthSet splits on must report high `conflict`; a
  proposition the TruthSet is silent on must report high `ignorance` — they
  must NOT be equal). FULL suite vs floor.

## Order rationale (one line)
1 before 2/3 (accumulator needs a home). 2 before 3 (kill flag → `ProjectionBasis`
never built → safe delete). 3 before 4 (scalar Kleene kernels final before
`falsity_penalty` rescalarized). 4 before 5 (assessment reads the consolidated
accumulator). symbol_states gate sits at the 2→4 seam.

## Verification (end to end)
- Per-phase gate command: `BASICMODEL_DEVICE=cpu PYTORCH_ENABLE_MPS_FALLBACK=1
  .venv/bin/python -m pytest test/ -q` (subset per phase; FULL at Phases 0, 3,
  4, 5 — must meet/exceed the Phase-0 pass-count floor).
- Real repro: a surviving grammar config trains 1 epoch / 3 batches with no
  `[B,V,2]` reshape crash (Bug #1 gone).
- Fix #2: stale `data/MM_5M_bivector.ckpt` + default autoload now fails fast
  with the actionable "correct the XML … or delete/move" message;
  `require_match=False` path still soft-warns; a benign extra-key ckpt loads.
- Phase 5: client response distinguishes contested (high `conflict`) from
  unknown (high `ignorance`) — the degeneracy this whole refactor removes.

## Doc to update on completion
Rewrite `basicmodel/doc/BivectorOutputSpaceRebasePlan.md`: Bug #1/Fix #1
deleted (structurally eliminated by Phase 2), Fix #2 retained/reframed, the
invariant + phases + the resolved symbol_states gate as the new body.
