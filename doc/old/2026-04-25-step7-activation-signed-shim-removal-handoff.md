# Handoff: Step 7 — Migrate `ConceptualSpace.activation_signed()` consumers and remove the shim

**Date.** 2026-04-25
**Owner.** Alec
**Status.** Not started.  Step 6 (`grammar.cfg`-driven dispatch) and
the Step 6.5 follow-up (`SymbolicSpace.layer` shim removal,
`# XXX deprecated alias` cleanup, `[layer2]` section drop) landed; the
deferred decisions from Step 6 are all closed in
[2026-04-25-step6.5-deferral-resolution-closure.md](2026-04-25-step6.5-deferral-resolution-closure.md).

---

## What this hands off

Step 7 of the lift / lower / bivector refactor as defined in the
parent plan:

- Plan: [2026-04-24-lift-lower-bivector-refactor.md §Step 7](2026-04-24-lift-lower-bivector-refactor.md)
  (lines 775–798).
- Predecessors: Step 6 ([2026-04-25-step6-grammar-cfg-handoff.md](2026-04-25-step6-grammar-cfg-handoff.md))
  and the Step 6.5 deferral resolutions
  ([2026-04-25-step6.5-deferral-resolution-closure.md](2026-04-25-step6.5-deferral-resolution-closure.md)).

The parent-plan Step 7 is a separate workstream from the Step 6.5 work
that already landed.  Step 6.5 migrated `SymbolicSpace.layer`
consumers off the `@property` shim that Step 4 introduced; Step 7
migrates `ConceptualSpace.activation_signed()` consumers off the
signed-collapse shim that Step 3 introduced.  The two shims are
unrelated; they share only the broader pattern of "Step N adds a
shim, a later step removes it."

---

## Pre-state

### What landed (Steps 3 / 6 / 6.5 outcome relevant to this step)

- **`ConceptualSpace.activation_signed()`** ([bin/Spaces.py:5881-5897](../../bin/Spaces.py))
  — backward-compat shim that collapses a `[B, N, 2]` bivector
  `[aP, aN]` to a `[B, N]` signed scalar `aP - aN`.  Step 3 of the
  refactor introduced the bivector storage on
  `ConceptualSpace.subspace.activation`; the shim let legacy callers
  keep reading the signed scalar form unchanged.
- **`test/test_conceptual_bivector.py`** exercises the shim directly
  ([lines 170–210](../../test/test_conceptual_bivector.py)) — three
  tests assert the shim returns `aP - aN`, returns `None` for an
  empty subspace, and matches the equivalent direct derivation.
- **No production consumers** — a fresh
  `grep -rn "\.activation_signed()" basicmodel/bin/` returns zero
  matches; the only callers live in the test file.  This makes
  Step 7 nearly free: the shim has no behavioral debt, only an
  unused surface waiting to be removed.

### What is unchanged

- The bivector storage on `ConceptualSpace.subspace.activation`
  (Step 3's deliverable) is in place and load-bearing.
- The Step 6 cfg-driven grammar dispatch
  (`<grammarCfg>data/grammar.cfg</grammarCfg>` in MentalModel.xml).
- The Step 6.5 cleanup state — `SymbolicSpace.layer` is gone; the
  4 legacy positional `Ops.lift`/`Ops.lower`/`liftReverse`/`lowerReverse`
  aliases stay with permanent `DeprecationWarning`.
- All Step 4 / Step 5 ownership and binary-mode wiring.

---

## What Step 7 changes

### 1. Remove the `activation_signed()` shim

The shim has no production consumers (verified via `grep`).  Delete
the method body at [bin/Spaces.py:5881-5897](../../bin/Spaces.py).

### 2. Update or delete the shim tests

The three tests in [test/test_conceptual_bivector.py](../../test/test_conceptual_bivector.py)
that exercise `activation_signed()` directly need to be either:

- (a) **Deleted outright.** The shim is gone; no behavior to test.
  Cleanest cut.
- (b) **Reframed** as a "shim removed" regression test that asserts
  the attribute is no longer present (the same pattern Step 6.5 used
  for `TestLayerShim` → `TestLayerShimRemoved` in
  `test_pi_sigma_ownership.py`).  Useful guard if a future commit
  might accidentally re-add the shim.

Recommend (b) for symmetry with Step 6.5.

### 3. Audit any indirect shim consumers (defensive)

The parent plan §Step 7 line 794 calls for "a grep gate that fails
CI if `activation_signed` is referenced outside the `ConceptualSpace`
definition itself."  The current state already satisfies this gate
(zero non-test references); after removal, the gate becomes "zero
references anywhere."

If the audit surfaces unexpected indirect callers (e.g. a future
commit added a caller after the parent plan was written), choose one
of the parent plan's three migration patterns per consumer:
- `aP - aN` (signed scalar — preserves the old number);
- `(aP - aN, aP * aN)` (signed + contradiction mask);
- direct bivector use (if the consumer can take both poles).

Run the audit with:
```
grep -rn "activation_signed" basicmodel/
```

### 4. Documentation cleanup

Remove the two parent-plan-references to `activation_signed` from
[doc/Spaces.md](../Spaces.md) and any other markdown that mentions
the shim (search with
`grep -rn "activation_signed" basicmodel/doc/`).

---

## Files Step 7 will touch

**Code:**
- [bin/Spaces.py](../../bin/Spaces.py) — delete the
  `activation_signed()` method on `ConceptualSpace`.

**Tests:**
- [test/test_conceptual_bivector.py](../../test/test_conceptual_bivector.py)
  — either delete the three shim-exercising tests (option a) or
  reframe them as a single shim-removed regression test (option b,
  recommended).

**Docs (sync as part of Step 7 implementation):**
- [doc/Spaces.md](../Spaces.md) — remove or update any
  `activation_signed` references.
- Search the rest of `basicmodel/doc/` for stale mentions and clean.

---

## New tests required

If option (b) above is chosen, one regression-style test in
`test/test_conceptual_bivector.py`:

```python
def test_activation_signed_shim_removed(self):
    """Step 7 removed ``ConceptualSpace.activation_signed()``;
    callers must derive ``aP - aN`` themselves from the bivector
    storage at ``subspace.activation``."""
    holder = ConceptualSpace(...)
    self.assertFalse(hasattr(holder, 'activation_signed'),
                     "activation_signed shim should be removed")
```

Plus a one-liner CI gate (or an existing test extended) that asserts
`activation_signed` does not appear in `basicmodel/bin/`:

```python
def test_no_shim_references_in_bin(self):
    import os, subprocess
    bin_dir = os.path.join(os.path.dirname(__file__), '..', 'bin')
    out = subprocess.run(
        ['grep', '-rn', 'activation_signed', bin_dir],
        capture_output=True, text=True,
    )
    self.assertEqual(out.stdout.strip(), '',
                     f"unexpected activation_signed references: "
                     f"{out.stdout}")
```

---

## Acceptance

- `grep -rn "activation_signed" basicmodel/bin/` returns no matches.
- The shim regression test (option b) passes.
- Acceptance sweep stays green:
  - `test/test_grammar_derivation.py`
  - `test/test_toy_grammar.py`
  - `test/test_subspace_context.py`
  - `test/test_head_divergence.py`
  - `test/test_serial_mode_integration.py`
  - `test/test_quaternary_corners.py`
  - `test/test_partition_grammar_rewrite.py`
  - `test/test_mask_dispatch.py`
  - `test/test_mental_model.py`
  - `test/test_reasoning.py`
  - `test/test_pi_sigma_ownership.py`
  - `test/test_ops_layer_wiring.py`
  - `test/test_grammar_cfg_dispatch.py`
  - `test/test_grammar_split.py`
  - `test/test_conceptual_bivector.py`
- No new deprecation warnings.

---

## Risks / things to watch

- **Shim has zero production consumers — but that is "as of Step 6.5
  audit."**  A commit landing between Step 6.5 and Step 7 may have
  added a new caller.  Re-run the audit
  (`grep -rn "activation_signed" basicmodel/bin/`) before deleting
  the method body.

- **Bivector vs scalar shape mismatch is the main bug class to
  watch.**  The shim's purpose was hiding the
  `[B, N, 2]` → `[B, N]` shape change from legacy callers.  Any
  caller that reads `subspace.get_activation()` directly and expects
  a `[B, N]` signed scalar will silently get a `[B, N, 2]` bivector
  after the shim is gone, with subsequent broadcasting bugs that
  may not surface as test failures (e.g. `act.mean()` is finite for
  both shapes, but the meaning differs).  Spot-check the existing
  `subspace.get_activation()` callers for shape-naive arithmetic
  before declaring done.

- **The test file may have non-shim tests worth keeping.**
  `test/test_conceptual_bivector.py` exercises bivector storage end
  to end — only the three shim-targeted tests should be touched.
  Read the file before editing to avoid collateral damage.

- **Step 7 is small enough to be done in one commit.**  Resist
  scope creep into a parent-plan Step 8 (none currently planned —
  see "Last task" below).

---

## Sequencing within Step 7

Smallest blast first:

1. **Re-audit** with `grep -rn "activation_signed" basicmodel/bin/
   basicmodel/test/`.  Record the current set of references.
2. **Delete the shim** at
   [bin/Spaces.py:5881-5897](../../bin/Spaces.py).
3. **Update the test file** — choose option (a) or (b); apply.
4. **Run the acceptance sweep.**  Expect green.
5. **Doc sweep** — clean stale references.
6. **Optional CI gate** — add the
   `test_no_shim_references_in_bin` regression test if option (b)
   is chosen.

All five steps are reversible with a single revert if anything
surprising surfaces.

---

## Verification commands

Per memory: tests via `basicmodel/.venv/bin/python -m pytest`.
Never run `make train` locally.  User manages git commits.

```bash
# unit-level
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_conceptual_bivector.py -v

# acceptance sweep
basicmodel/.venv/bin/python -m pytest \
  basicmodel/test/test_grammar_derivation.py \
  basicmodel/test/test_toy_grammar.py \
  basicmodel/test/test_subspace_context.py \
  basicmodel/test/test_head_divergence.py \
  basicmodel/test/test_serial_mode_integration.py \
  basicmodel/test/test_quaternary_corners.py \
  basicmodel/test/test_partition_grammar_rewrite.py \
  basicmodel/test/test_mask_dispatch.py \
  basicmodel/test/test_mental_model.py \
  basicmodel/test/test_reasoning.py \
  basicmodel/test/test_pi_sigma_ownership.py \
  basicmodel/test/test_ops_layer_wiring.py \
  basicmodel/test/test_grammar_cfg_dispatch.py \
  basicmodel/test/test_grammar_split.py \
  basicmodel/test/test_conceptual_bivector.py

# regression-greps
grep -rn "activation_signed" basicmodel/bin/      # expect: empty
grep -rn "activation_signed" basicmodel/doc/      # expect: empty
```

Pre-Step-7 baseline: Step 6 + Step 6.5 totals (241 passed + 6
xfailed across the 14-test sweep, plus the 31 / 6 / 8 result on the
`reasoning + pi_sigma_ownership + universality` subset that landed
the Step 6.5 cleanup).  Step 7 acceptance: same totals, with the
shim-removed regression test added (or three shim-exercising tests
removed if option a).

---

## Last task: close out the lift / lower / bivector refactor

Step 7 is the final step in the parent plan
([2026-04-24-lift-lower-bivector-refactor.md](2026-04-24-lift-lower-bivector-refactor.md))
— there is no Step 8 currently planned.  After Step 7 lands and the
acceptance sweep passes, write a short close-out note (not a handoff)
at `doc/plans/<date>-lift-lower-bivector-refactor-complete.md`
covering:

1. What the refactor accomplished end-to-end (Steps 1–7), with
   pointers to the per-step handoffs:
   - Steps 1–2: unified `Ops.lift` / `Ops.lower` signature with mode
     dispatch.
   - Step 3: bivector `[aP, aN]` storage on
     `ConceptualSpace.subspace.activation`
     ([2026-04-25-step3-bivector-conceptual-handoff.md](2026-04-25-step3-bivector-conceptual-handoff.md)).
   - Step 4: ownership flip — Pi/Sigma now owned by ConceptualSpace
     ([2026-04-25-step4-pi-sigma-ownership-handoff.md](2026-04-25-step4-pi-sigma-ownership-handoff.md)).
   - Step 5: STE top-2 selector + binary forward mode on Pi/Sigma
     ([2026-04-25-step5-ops-wiring-handoff.md](2026-04-25-step5-ops-wiring-handoff.md)).
   - Step 6: `data/grammar.cfg` explicit-op dispatch
     ([2026-04-25-step6-grammar-cfg-handoff.md](2026-04-25-step6-grammar-cfg-handoff.md))
     plus Step 6.5 deferral resolutions
     ([2026-04-25-step6.5-deferral-resolution-closure.md](2026-04-25-step6.5-deferral-resolution-closure.md)).
   - Step 7: this handoff — `activation_signed()` shim removal.
2. Final acceptance numerics (sweep totals after Step 7).
3. Open follow-ups intentionally left out of scope:
   - The 4 legacy positional `Ops` aliases stay with permanent
     `DeprecationWarning` (Step 6.5 decision).
   - Pi/Sigma do not delegate through `Ops.lower`/`Ops.lift`
     (Step 6.5 decisions on Step 5 deferrals 1 / 2 / 3).
   - `bind` / `scale` not promoted to first-class ops (Step 6.5
     decision on Step 6 deferral 4).
   - The "Layer 1 / Layer 2" rule split survives only as parent-plan
     terminology; the runtime grammar.cfg has a flat `[upward]`
     section (Step 6.5 decision on deferral 8).
4. Any unrelated workstreams that surfaced during the refactor and
   should be tracked separately (e.g. the rule predictor's
   `max_depth * pos_dim` widening, the `WordSpace.category_codebook`
   capacity guard, the `[ADAM-BUG-PROBE]` log noise during testing).

The close-out note is the deliverable that closes the parent plan;
write it after Step 7 lands and the sweep is green.
