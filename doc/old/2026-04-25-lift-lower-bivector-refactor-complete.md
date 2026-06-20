# Close-out: Lift / Lower / Bivector Refactor — Steps 1–7 Complete

**Date.** 2026-04-25
**Owner.** Alec
**Status.** Complete.  All seven steps of the parent plan
([2026-04-24-lift-lower-bivector-refactor.md](2026-04-24-lift-lower-bivector-refactor.md))
landed.  No Step 8 planned.

---

## What the refactor accomplished

Three substantive changes shipped end-to-end, organized around the
synthesis / analysis lift / lower axis:

1. **Pi / Sigma ownership flip.**  Synthesis is now owned by the
   receiving "one" layer and analysis by the producing "many" layer;
   the canonical home for the C → S Pi is `conceptualSpace.pi`
   (not `symbolicSpace.layer`, which no longer exists).
2. **Unified `Ops.lift` / `Ops.lower` signature with mode dispatch.**
   `Y = f(X1, X2=None, mode='AND'|'OR'|'NOT', inverse=False)` covers
   point and region scales; the four legacy positional aliases are
   warn-only indefinite.
3. **Bivector ConceptualSpace activation.**  `subspace.activation` is
   now `[B, N, 2]` carrying `[aP, aN]`; the S → C lower preserves
   contradiction (`[1, 1]`) vs. ignorance (`[0, 0]`) instead of
   collapsing both to `0` under signed-sum readout.

Plus two structural changes that fell out:

4. **`data/grammar.cfg` explicit-op dispatch** (Step 6) replaced the
   inline `<grammar>` XML block in `MentalModel.xml`.
5. **`SymbolicSpace.layer` shim removal** (Step 6.5) and
   **`ConceptualSpace.activation_signed()` shim removal** (Step 7)
   close out the two transitional surfaces the refactor introduced.

### Per-step handoffs

| Step | Title | Handoff |
|---|---|---|
| 1–2 | Unified `Ops.lift` / `Ops.lower` signature with mode dispatch | (no separate handoff — landed in initial Ops rename pass) |
| 3 | Bivector `[aP, aN]` storage on `ConceptualSpace.subspace.activation` | [step3-bivector-conceptual-handoff.md](2026-04-25-step3-bivector-conceptual-handoff.md) |
| 4 | Ownership flip — Pi/Sigma now owned by ConceptualSpace | [step4-pi-sigma-ownership-handoff.md](2026-04-25-step4-pi-sigma-ownership-handoff.md) |
| 5 | STE top-2 selector + binary forward mode on Pi/Sigma | [step5-ops-wiring-handoff.md](2026-04-25-step5-ops-wiring-handoff.md) |
| 6 | `data/grammar.cfg` explicit-op dispatch | [step6-grammar-cfg-handoff.md](2026-04-25-step6-grammar-cfg-handoff.md) |
| 6.5 | Deferred-decision resolutions (Step 5/6 deferrals; `SymbolicSpace.layer` shim removal; `[layer2]` drop) | [step6.5-deferral-resolution-closure.md](2026-04-25-step6.5-deferral-resolution-closure.md) |
| 7 | `ConceptualSpace.activation_signed()` shim removal | [step7-activation-signed-shim-removal-handoff.md](2026-04-25-step7-activation-signed-shim-removal-handoff.md) |

---

## Final acceptance numerics

Step 7 acceptance sweep (15 test files,
`basicmodel/.venv/bin/python -m pytest`):

```
253 passed, 6 xfailed in 607.76s (0:10:07)
```

Files in the sweep:

- `test_grammar_derivation.py`
- `test_toy_grammar.py`
- `test_subspace_context.py`
- `test_head_divergence.py`
- `test_serial_mode_integration.py`
- `test_quaternary_corners.py`
- `test_partition_grammar_rewrite.py`
- `test_mask_dispatch.py`
- `test_mental_model.py`
- `test_reasoning.py`
- `test_pi_sigma_ownership.py`
- `test_ops_layer_wiring.py`
- `test_grammar_cfg_dispatch.py`
- `test_grammar_split.py`
- `test_conceptual_bivector.py`

The 6 xfailed are pre-existing in `test_reasoning.py` and unchanged
across the refactor.

**Regression-greps:**
- `grep -rn "activation_signed" basicmodel/bin/` → empty.
- `grep -rn "symbolicSpace\.layer\|SymbolicSpace\.layer" basicmodel/bin/ basicmodel/test/` → empty (or only `TestLayerShimRemoved` assertion strings).

**Post-sweep regression caught and fixed:**
`test_hierarchical.py::test_pair_pi_layers_created` was using the
removed `model.symbolicSpaces[i].layer` shim.  Migrated to
`model.conceptualSpaces[i].pi` (the Step 4 ownership home).
`test_hierarchical.py` was not in the Step 6 / Step 7 acceptance
sweep — adding it (and any other butterfly-mode tests) to a future
sweep would have caught this earlier.  Also fixed a stale
`symbolic_space.layer` reference in a `bin/Language.py` docstring.

---

## Open follow-ups intentionally out of scope

Per the Step 6.5 closure decisions:

- **The 4 legacy positional `Ops` aliases stay with permanent
  `DeprecationWarning`.**  `Ops.lift(left, right)` /
  `Ops.lower(left, right)` / `Ops.liftReverse` / `Ops.lowerReverse`
  are warn-only indefinite; no escalation to FutureWarning / TypeError,
  no hard delete.
- **Pi / Sigma do NOT delegate through `Ops.lower` / `Ops.lift`.**
  PiLayer / SigmaLayer keep their existing log-domain matmul body
  (Step 5 deferrals 1 & 2).  Layer reverses use the linear-algebraic
  LDU inverse, not the codebook-search `Ops.*Reverse`.
- **Grammar dispatch goes through explicit two-operand `Ops.lower` /
  `Ops.lift` calls** (Step 5 deferral 3 — option (a)).  No N-element
  STE packing, no per-rule annotation.
- **`bind` / `scale` not promoted to first-class ops.**  The
  `intersection(...)` fallback covers `PP = intersection(P, NP)` and
  `AP = intersection(DEG, AP)`; promote later if a linguistic
  motivation surfaces.
- **The "Layer 1 / Layer 2" rule split survives only as parent-plan
  terminology.**  The runtime `grammar.cfg` has a flat `[upward]`
  section; `Grammar._CFG_SECTION_LAYER2` and `rules_layer2` are gone.

---

## Unrelated workstreams surfaced during the refactor

These are tracked here so they don't get lost; none of them belong to
the lift / lower / bivector refactor.

- **Rule predictor input feature width.**  Currently
  `max_depth * pos_dim`; widens linearly with `nPercepts`.  Revisit if
  `nPercepts` ever climbs into the hundreds.
- **`WordSpace.category_codebook` capacity.**  Currently
  `max(64, len(categories))`; the cfg surfaces ~19 categories so the
  floor of 64 dominates.  No action needed unless the cfg grows
  beyond ~50 categories.
- **`ChunkLayer` / perceptual chunking** lives outside grammar
  dispatch and was unaffected by the refactor.
- **`[ADAM-BUG-PROBE]` log noise during testing.**  Each test run
  prints ~190 `[ADAM-BUG-PROBE] rebuild #N pretrain opt_id=...` lines.
  Diagnostic, harmless, but worth silencing or threshold-gating
  separately.

---

## Closing note

The parent plan
([2026-04-24-lift-lower-bivector-refactor.md](2026-04-24-lift-lower-bivector-refactor.md))
should be marked **Status: Complete** with a pointer to this note.
The seven step-handoffs and the 6.5 deferral closure remain as the
historical record of the work as it landed.
