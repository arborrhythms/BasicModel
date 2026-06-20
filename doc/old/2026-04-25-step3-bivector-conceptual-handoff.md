# Handoff: Step 3 — Bivector activation in ConceptualSpace

**Date.** 2026-04-25
**Owner.** Alec
**Status.** Not started.  Steps 1 and 2 of the parent plan have landed.

---

## What this hands off

Step 3 of the lift / lower / bivector refactor:
- Plan: [2026-04-24-lift-lower-bivector-refactor.md §Step 3](2026-04-24-lift-lower-bivector-refactor.md)
  (lines 259–327).
- Spec: [2026-04-24-lift-lower-bivector-design.md](../specs/2026-04-24-lift-lower-bivector-design.md)
  §B1–B7 govern the layout; §Q2 (percept promotion) and §O4
  (tetralemma sharing) bear directly.

This is the **largest blast radius** in the parent plan and the
only step that materially changes a tensor shape exposed across
spaces.  The plan calls out a signed-collapse shim
(`ConceptualSpace.activation_signed()`) as the rollback safety
valve.  Land Step 3 with the shim in place; migrate consumers off
in Step 7.

---

## Pre-state (verified 2026-04-25)

Steps 1 and 2 landed.

- `Ops.lift` / `Ops.lower` (`bin/Layers.py:4350`/`4419`) carry the
  unified mode dispatch and the `kind` parameter
  (`strict` default, `smooth`, `radial`).
- `Ops.conjunction` / `Ops.disjunction` (`bin/Layers.py:4205`/`4217`)
  are thin forwarders.
- `Ops._radmin` / `Ops._radmax` (`bin/Layers.py:4192`/`4197`) hold
  the bitonic same-sign magnitude bodies.
- `Basis.lift` / `Basis.lower` (`bin/Spaces.py:965`/`987`) carry
  the matching `kind` parameter and pass it through.
- Test file: [test/test_ops_lift_lower.py](../../test/test_ops_lift_lower.py) — 40/40 pass.
- Broader sweep at acceptance time: 173/173 pass across the
  conjunction/disjunction-using files.

`ConceptualSpace.subspace.activation` is still a single signed
scalar per concept (not yet bivector).  `SymbolicSpace.subspace.what`
is already bivector (`nWhat = 2`) — the precedent to mirror.

---

## What Step 3 changes

Per the parent plan §Step 3 and spec §B1–B7:

1. **Shape change** at `ConceptualSpace.subspace`:
   - `nWhat` becomes 2 (mirrors `SymbolicSpace`).
   - `subspace.activation` shape becomes `[batch, N, 2]` (or
     `[batch, N, 2 + nWhere + nWhen]` when positional content is
     included).  The two `.what` dims are `[aP, aN]` per concept.
   - Layout is **leading-bivector** per spec B7
     (`subspace.what` carries the bivector at the front; TruthLayer
     uses paired-index storage with explicit slicing at the
     boundary).

2. **Producer side** (writes into conceptual activation):
   - **PerceptualSpace → ConceptualSpace** (Pi forward path,
     currently writes a signed scalar).  Promote to bivector via
     `[max(0, x), max(0, -x)]` so positive evidence routes to `aP`
     and negative evidence to `aN`.  See spec Q2 for the exact
     mapping and B3 for whether PerceptualSpace itself adopts
     bivector activation.
   - **SymbolicSpace → ConceptualSpace** (Sigma reverse / lower
     path).  Currently a signed-sum projection.  Should preserve
     each pole pair through the lower's pooled product so
     contradiction (`[1, 1]`) is distinguishable from ignorance
     (`[0, 0]`).

3. **Consumer side** (reads conceptual activation):
   - Anything that read a signed scalar must now derive
     `signed = aP − aN` explicitly (or use the
     `activation_signed()` shim during transition).
   - Where contradiction-awareness matters, derive
     `contradiction = aP · aN` as a sibling.
   - `ImpenetrableLayer` already operates on bivector codebook
     rows for SymbolicSpace; the same path applies to the
     ConceptualSpace codebook under bivector activation.
   - `tetralemma_balance_penalty` ([bin/Layers.py:2464](../../bin/Layers.py))
     is shape-agnostic — no internal change, just a different
     upstream caller.

4. **Backward-compat shim**:
   - Add `ConceptualSpace.activation_signed()` returning `aP − aN`
     for callers that haven't migrated.  This is the rollback
     safety valve.
   - Removed in Step 7 (consumer migration).

5. **TetralemmaPolicy config** (spec §O4):
   - **shared default with per-space override**.  Add the
     `<TetralemmaPolicy>` block to the XML config, with
     `<tetralemmaOverride enabled="false" />` blocks under
     `<ConceptualSpace>` and `<SymbolicSpace>`.  Schema is in
     spec §O4 (lines 153–170).

---

## File map (verified at handoff time)

Files Step 3 will touch:

- `bin/Spaces.py:5781` — `class ConceptualSpace(Space)`.  Constructor
  is the entry point: set `nWhat = 2` and update the activation
  shape.
- `bin/Spaces.py:5945` — `class SymbolicSpace(Space)`.  Already
  has `nWhat = 2` — read it as the precedent for layout / accessor
  patterns.
- `bin/Spaces.py:872`, `2387–2389`, `2503–2538`, `2668–2731` —
  generic `subspace` event/what/where/when machinery and bivector
  derivation (`aP, aN` formulas already exist near line 2533 for
  SymbolicSpace; mirror them for ConceptualSpace).
- `bin/Layers.py:2464` — `tetralemma_balance_penalty`.  Read-only
  for Step 3; called by Language.py:2454 already on a bivector
  symbol activation.
- `bin/Models.py:2970, 3389, 3489` — `self.conceptualSpace.subspace`
  reads.  Audit each for shape assumptions; most are likely
  attribute reads (e.g. `wordSpace`) and unaffected.
- `bin/Pipeline.py` — if it carries activation through stages, the
  shape change propagates here.  Grep `conceptualSpace.subspace`
  before deciding scope.
- `bin/Language.py` — has consumer code; `Ops.lift(left, right)` /
  `Ops.lower(left, right)` legacy calls at lines 926/934 are the
  current high-volume callers (~244K deprecation warnings in the
  test sweep).  Step 3 doesn't migrate these but the bivector shape
  may require them to switch to `kind='smooth'` explicitly with
  bivector inputs — verify per-axis broadcast.

Test files that read `model.conceptualSpace.subspace` directly
(audit for shape assumptions before changing):

- `test/test_grammar_derivation.py:125`
- `test/test_toy_grammar.py:421, 485`
- `test/test_subspace_context.py:247`
- `test/test_head_divergence.py:103, 105, 117`
- `test/test_serial_mode_integration.py:58, 63`

---

## New tests required (per parent plan)

`test/test_conceptual_bivector.py` covering:

1. `[1, 0]` activations route correctly (positive evidence).
2. `[0, 1]` activations route correctly (negative evidence).
3. `[0, 0]` distinguishable from `[1, 1]` downstream:
   `signed = 0` for both, `contradiction = 0` vs `contradiction = 1`.
4. Signed-collapse shim returns the same numbers as the pre-Step-3
   single-scalar activation for `[max(0, x), max(0, -x)]` inputs.

All existing `model.conceptualSpace.subspace.activation` readers
must continue to pass via the shim, then be migrated to
bivector-aware reads in Step 7.

---

## Open spec resolutions to honor

The spec already resolves the following — no fresh decisions
needed before starting:

- **B1** (shape `[N, 2]`): committed.
- **B2** (lift/lower preserve bivector): committed.
- **B3** (PerceptualSpace bivector adoption): committed —
  PerceptualSpace also goes bivector.  Flag during implementation
  if percept N is large enough to make this expensive.
- **B4** (codebook `.what` widths): committed —
  high-dim conceptual, bivector symbolic.
- **B5** (`monotonic=True` propagates): committed.
- **B6** (category vectors are bivector in their own codebook):
  committed (but not exercised until Step 6).
- **B7** (leading-bivector layout for `subspace.what`,
  paired-index for TruthLayer): committed.
- **Q2** (percept promotion): committed — `[max(0, x), max(0, -x)]`.
- **O4** (TetralemmaPolicy sharing): committed — shared default
  with per-space override.

Re-read the spec sections above before starting; the
implementation decisions are mechanical from there.

---

## Sequencing within Step 3

A reasonable order, smallest blast first:

1. Add `ConceptualSpace.activation_signed()` shim.  Returns the
   current single-scalar activation unchanged at this point —
   purely a stub the migration will lean on.  No tests break.
2. Flip `nWhat` and the activation shape.  Audit the producers and
   make them write `[aP, aN]` instead of a signed scalar.  All
   existing consumers break here.
3. Patch every consumer to call the shim
   (`x = subspace.activation_signed()`) so the existing test suite
   goes green on the new shape via signed-collapse.
4. Add the new `test_conceptual_bivector.py` tests for the four
   bivector-distinguishability cases.
5. Land the TetralemmaPolicy XML config block (spec §O4).  Wire the
   parser for `<tetralemmaOverride>`; default to inherited.

If a consumer is ambiguous about whether it wants the signed
scalar or the bivector, lean on the shim and defer the call to
Step 7.  Step 3's job is "shape works, behavior unchanged."

---

## Verification commands

Per-memory: tests via `basicmodel/.venv/bin/python -m pytest`.
Never run `make train` locally.  User manages git commits.

```bash
# unit-level
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_conceptual_bivector.py -v
basicmodel/.venv/bin/python -m pytest basicmodel/test/test_ops_lift_lower.py -v   # should still be 40/40

# acceptance sweep — files that read conceptualSpace.subspace
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_grammar_derivation.py \
  basicmodel/test/test_toy_grammar.py \
  basicmodel/test/test_subspace_context.py \
  basicmodel/test/test_head_divergence.py \
  basicmodel/test/test_serial_mode_integration.py \
  basicmodel/test/test_quaternary_corners.py \
  basicmodel/test/test_partition_grammar_rewrite.py \
  basicmodel/test/test_mask_dispatch.py \
  basicmodel/test/test_mental_model.py
```

The pre-Step-3 baseline of the acceptance sweep (run during the
Step 2 verification) was 173 pass.  Step 3 acceptance is
"existing test suite green via shim, new bivector tests pass" —
i.e. the same 173 plus the new `test_conceptual_bivector.py`
cases.

---

## Risks / things to watch

- **Producer audit completeness.**  Any writer to
  `subspace.activation` that is missed will leave a half-bivector
  layout that the shim can't repair.  Grep for every assignment
  to `subspace.activation` and `set_event` on the conceptual
  subspace before declaring producer migration done.
- **Broadcast quirks under bivector.**  `Ops.lift(x, y, kind='smooth')`
  on a `[B, N, 2]` tensor returns `(x + y) / 2` elementwise
  including across the bivector axis — this is what Step 3 wants.
  `kind='strict'` returns `torch.max(x, y)` axis-wise, which on
  bivector inputs produces a per-pole max, not a tetralemma-aware
  comparison.  Any caller that relied on a single-scalar
  comparison will need explicit signed-collapse first.
- **Pipeline.py wiring.**  Not surveyed in detail at handoff time —
  if Pipeline carries activation through stages, the shape change
  propagates and may require additional consumer migration.
- **Language.py legacy callers (lines 926, 934).**  Currently
  emitting ~244K deprecation warnings during the test sweep.
  Bivector inputs to the legacy 2-arg form route through
  `kind='smooth'`, which is element-wise and fine on bivector
  tensors — but verify with a small fixture.
- **TruthLayer / ImpenetrableLayer storage layout.**  Spec B7
  commits leading-bivector for `subspace.what`, paired-index for
  TruthLayer.  Slicing at the boundary needs to be explicit; if
  the contracts get out of sync the symptom will be silent
  per-pole permutation.
