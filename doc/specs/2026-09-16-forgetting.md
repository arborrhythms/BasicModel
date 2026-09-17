# Forgetting: keeping LTM under its limit by value

> **Status:** specification, 2026-09-16, written by Claude from Alec's
> decisions in conversation on 2026-09-16. Simple by intent. Owns
> [FutureWork.md §2](../FutureWork.md#2-forgetting-and-consolidation-urgent).
> Builds on the [two-truths spec](2026-09-16-two-truths-ideas-and-relations.md)
> (every S writes a row; rows are referenced by index) and on the
> [operation profile](../FutureWork.md#1-operation-profile-optimal-versus-human).
> Everything marked **(decided)** is Alec's decision.

## 1. The problem

The unified LTM store (`TernaryTruthStore`, `symbolSpace.ltm_store`) is
append-only to a fixed capacity (`<ltmCapacity>`, default 1024). Under the
two-truths contract every sealed S writes a row, so the capacity is
reached in ordinary training and the store then refuses writes. Something
must be forgotten, and what is forgotten must be chosen by value, not by
age alone.

## 2. Decisions (decided)

- **A limit in the model file.** `<ltmCapacity>` remains the hard limit.
  Forgetting keeps the store under it.
- **Periodic, not too often.** A forgetting pass runs only when the store
  is nearly full, never mid-sentence, and reduces the store to a target
  occupancy so that many writes fit before the next pass.
- **Delete by value.** The pass deletes the lowest-valued rows. Value is a
  combination of three terms:
  1. **Trust.** High-trust rows are preferred.
  2. **Utility.** A row that can be easily deduced from other rows is not
     worth remembering.
  3. **Contribution to luminosity.** Dissonant rows, those that push a
     region toward the *both* corner by contradicting a better-supported
     row, are forgotten. Full luminosity contribution is expensive, so the
     pass uses the cheap dissonance test and computes coverage gain only
     when enabled.

This replaces the earlier direction in FutureWork.md that isolated rows
should be forgotten: an isolated row may be high in value on all three
terms, and a well-integrated row may be exactly the deducible one.

## 3. Value

For a row `i`, at pass time:

```text
V(i) = w_t · T(i) + w_u · U(i) + w_l · L(i)          (profile: optimal)
V(i) = the same − w_a · A(i)                           (profile: human)
```

- **T(i) = |trust(i)|.** A confidently false row is knowledge as much as a
  confidently true one; the sign is not the value. (If Alec means positive
  trust only, change this to `max(0, trust)`; the difference is one line
  and one test.)
- **U(i) = 1 − deducibility(i)**, with deducibility in `[0, 1]`:
  - For an **idea row**: the expectation's discrepancy when the row was
    observed, recorded at write time as a per-row `surprise` column
    (normalised occupied-role MSE plus presence loss of the local-role
    expectation; `-1` when no expectation was staged, treated as
    surprise `1`). A well-expected sentence was deducible from its
    context and is low utility. No extra computation at pass time.
  - For a **relation row**: one step of the relation readers over the
    other rows. A part row `A ⊑ C` is deducible if some `B` has `A ⊑ B`
    and `B ⊑ C` with both trusts at least the row's own. An implies row
    is deducible if modus ponens over other rows yields it in one step.
    Deducibility is `1` when found, else `0`. One step only: the pass
    is not a theorem prover.
- **L(i) = 1 − dissonance(i)**, with dissonance in `[0, 1]`:
  - **Cheap test, always run.** Rows are dissonant in pairs: two idea rows
    whose fused points fall in the same region (cosine above
    `<forgetSimilarity>`) with opposite trust signs, or two relation rows
    with the same kind and operands and opposite signs (the dedup path
    already collapses same-sign duplicates). Of a dissonant pair, the row
    with the lower `|trust|` carries dissonance `1`, the other `0`; a row
    in no dissonant pair has `0`. This is `TruthLayer.consistency`'s
    report path applied to the store rather than the view.
  - **Coverage gain, optional.** When `<forgetCoverage>` is on, `L(i)` is
    instead the drop in catuṣkoṭi coverage of the truth view if row `i`
    were removed, normalised to `[0, 1]`. Off by default; too expensive
    to run at every pass at scale.
- **A(i)**, human profile only: age since the row was last used
  (written, read as expectation context, resolved as a relation operand,
  or returned by a query), on an Ebbinghaus-shaped curve
  `1 − exp(−age / <forgetHalfLife>)`. Under the optimal profile age does
  not enter value at all.

Weights `w_t, w_u, w_l, w_a` are `model.xml` elements (§6). Terms are in
`[0, 1]` so the weights are comparable.

## 4. The pass

1. **Trigger.** After a document boundary's host-side work (never inside
   a sentence, never inside a compiled region), if
   `count ≥ <forgetHighWater> · capacity`. Also at the end of an epoch
   regardless of occupancy when `<forgetAtEpoch>` is on. Not more often
   than `<forgetMinInterval>` sentences since the last pass.
2. **Protection.** Rows whose origin is in `<forgetProtectOrigins>`
   (default: provisioned and user) are never deleted; they are supplied
   truths with their own replace-on-resubmit and re-provision rules. If
   protected rows alone exceed the target, the pass logs and does nothing.
3. **Value.** Compute `V` for every unprotected row (§3).
4. **Selection.** Delete the lowest-`V` unprotected rows until
   `count ≤ <forgetLowWater> · capacity`. Ties break oldest first.
5. **Cascade.** A relation row whose operand row is deleted is deleted with
   it, whatever its own value, because a relation over a missing row is
   meaningless. Cascade repeats until stable. The cascade is counted
   toward the target, so step 4 may delete fewer rows than it planned.
6. **Compaction.** Rows are compacted in place (the mechanism
   `clear_origin` already uses) and the `refs` column of every surviving
   row is remapped to the new indices. No dangling reference survives a
   pass; a debug assertion checks it.
7. **Dependents.** After compaction: the truth view (`sync_from_ltm`) is
   rebuilt; the concept-level taxonomy index is rebuilt from surviving
   part rows and META bindings; the discourse observation view is
   untouched (it holds tensors, not indices); pending expectation state
   is untouched.
8. **Record.** The pass appends one line to the training log: rows
   before, rows after, the value cut-off, the count deleted per origin,
   and the count deleted by cascade. A checkpoint records the number of
   passes and the last pass's cut-off, nothing about deleted rows.

## 5. What is not forgotten and what is not done

- Concept rows in the shared index are not the store's rows and are not
  forgotten by this pass. A fused idea row that is deleted keeps its
  concept row if any surviving row or binding references it; otherwise
  the concept row's release is the codebook's own business and out of
  scope here.
- Derivation decay (dropping parts of a surviving row's reconstruction
  trace) is FutureWork.md §3, not this spec; the two share the human
  profile's age curve when both exist.
- No merging of near-duplicate rows into a coarser row. Deletion only.
- No forgetting inside a request in stateless serving; the pass runs in
  training and in stateful serving only.

## 6. Configuration

All under `<architecture>`, next to `<ltmCapacity>`:

| Element | Default | Meaning |
|---|---|---|
| `forgetting` | `true` | run the pass at all |
| `forgetHighWater` | `0.9` | occupancy fraction that triggers a pass |
| `forgetLowWater` | `0.7` | occupancy fraction the pass reduces to |
| `forgetMinInterval` | `256` | minimum sentences between passes |
| `forgetAtEpoch` | `false` | also run at epoch end |
| `forgetTrustWeight` | `1.0` | `w_t` |
| `forgetUtilityWeight` | `1.0` | `w_u` |
| `forgetLuminosityWeight` | `1.0` | `w_l` |
| `forgetAgeWeight` | `1.0` | `w_a`, human profile only |
| `forgetHalfLife` | `4096` | sentences, human profile only |
| `forgetSimilarity` | `0.9` | cosine for the dissonant-pair test |
| `forgetCoverage` | `false` | compute coverage gain instead of the cheap test |
| `forgetProtectOrigins` | `provisioned user` | origins never deleted |

Schema entries with these defaults; Params.md rows; the comment in
`model.xml` states technique only.

## 7. Acceptance tests

1. **Trigger and target.** A store at high water runs one pass at the next
   document boundary and ends at or below low water; no pass runs below
   high water; no pass runs within the minimum interval; no pass runs
   mid-sentence.
2. **Order of deletion.** With weights isolating each term: the
   lowest-`|trust|` rows go first; the lowest-surprise idea rows go
   first; a transitively deducible part row goes before an underivable
   one; of a dissonant pair the lower-`|trust|` row goes and the other
   stays.
3. **Protection.** Provisioned and user rows survive a pass that deletes
   every conversation row; a store of only protected rows logs and does
   nothing.
4. **Cascade and references.** Deleting an idea row deletes the relation
   rows over it; after compaction every `refs` entry points at the row it
   pointed at before, or is `-1` because that row was deleted; the debug
   assertion passes.
5. **Dependents.** After a pass, the truth view, the taxonomy index and
   `relations()` agree with the surviving rows; the discourse observation
   view is byte-identical before and after.
6. **Profiles.** Under `optimal`, two rows differing only in age have equal
   value; under `human`, the older has lower value.
7. **Surprise column.** An idea row observed without a staged expectation
   has surprise `-1` and is treated as maximally useful; a row observed
   with a near-zero discrepancy is treated as deducible.
8. **Packed and single-sentence parity.** The same corpus in both cursors
   yields the same surviving rows.
9. **Checkpoint.** The pass count and last cut-off round-trip; a
   checkpoint saved between passes reloads with the same occupancy.
10. **Determinism.** Two runs with the same seed delete the same rows.

## 8. Documentation required with implementation

- STM.md's LTM consolidation section: the pass, its trigger and the value.
- Reasoning.md: deducibility as the utility term; one step only.
- Philosophy.md discrepancy 3 marked as specified here; the luminosity
  term's relation to the catuṣkoṭi coverage.
- Params.md: the elements in §6.
- FutureWork.md §2: replaced by a pointer to this spec.
