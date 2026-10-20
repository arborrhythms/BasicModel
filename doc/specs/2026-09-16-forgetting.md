# Forgetting: keeping LTM under its limit by value

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](../plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](2026-10-08-expectation-at-every-level.md).
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.


> **Status:** implementation candidate, amended 2026-10-19 after Alec's
> instruction to begin Item 5.0 with age retained for LTM turnover. Original
> specification, 2026-09-16, written by Claude from Alec's
> decisions in conversation on 2026-09-16. Simple by intent. Owns
> [FutureWork.md §2](../FutureWork.md#2-forgetting-and-consolidation).
> Builds on the [two-truths spec](2026-09-16-two-truths-ideas-and-relations.md)
> (completed readings write rows; references now use stable addresses) and on the
> [operation profile](../FutureWork.md#1-operation-profile-optimal-versus-human).
> Everything marked **(decided)** is Alec's decision. §4a (detail before rows)
> added 2026-09-20 from his ruling that day.

## 1. The problem

The unified LTM store (`TernaryTruthStore`, `symbolSpace.ltm_store`) is
append-only to a fixed capacity (`<ltmCapacity>`, default 1024). Under the
two-truths contract every ended S writes a row, so the capacity is
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
V(i) = w_t · T(i) + w_u · U(i) + w_l · L(i) − w_a · A(i)
```

- **T(i) = |source_trust(i)|.** This separately stored provenance scalar is
  independent of the two evidence poles. The legacy `trust` accessor still
  returns their signed difference for existing readers; forgetting does not
  use it. Checkpoints without provenance load unknown trust as zero, never
  infer it from the evidence pair. An older explicit scalar is migrated.
- **U(i) = 1 − deducibility(i)**, with deducibility in `[0, 1]`:
  - For an **idea row** (`REL_NONE` or ordinary-sentence `REL_OPERATOR`):
    the expectation's discrepancy when the row was
    observed, recorded at write time as a per-row `surprise` column
    (normalised all-role squared residual of the local-role
    expectation; `-1` when no expectation was staged, treated as
    surprise `1`). A well-expected sentence was deducible from its
    context and is low utility. No extra computation at pass time.
  - For a **part or implication relation row**: one step of the relation readers over the
    other rows. A part row `A ⊑ C` is deducible if some `B` has `A ⊑ B`
    and `B ⊑ C` with both source trusts and positive evidence at least the
    row's own. Modus ponens requires both an implication and its actual
    antecedent, and makes the consequent expendable.
    Deducibility is `1` when found, else `0`. One step only: the pass
    is not a theorem prover. Only positive-only evidence in the same scope
    supplies these certificates. A deleted conclusion's supporting rows are
    pinned for the rest of that pass. Loss of a support raises a pending
    candidate's utility before selection.
  - Other relation kinds have utility `1`.
- **L(i) = 1 − dissonance(i)**, with dissonance in `[0, 1]`:
  - **Cheap test, always run.** Rows are dissonant in pairs: two idea rows
    whose fused points fall in the same region (cosine above
    `<forgetSimilarity>`) with opposing single-pole evidence, or two relation rows
    with the same kind and operands and opposing poles (the dedup path
    already collapses same-sign duplicates). Of a dissonant pair, the row
    with the lower `|trust|` carries dissonance `1`, the other `0`; a row
    in no dissonant pair has `0`. This is `TruthLayer.consistency`'s
    report path applied to the store rather than the view. Scope and order
    must match. Intrinsic *both* within one row is heterogeneous content,
    not conflicting sources, and is not penalized by this test.
  - **Coverage gain, optional.** When `<forgetCoverage>` is on, `L(i)` is
    instead the drop in catuṣkoṭi coverage of the truth view if row `i`
    were removed, normalised to `[0, 1]`. Off by default; too expensive
    to run at every pass at scale.
- **A(i)**, both profiles under capacity pressure: rows back from the tail
  of the row's witnessing stream, on the bounded curve
  `1 − exp(−age / <forgetHalfLife>)`. A row shared by streams uses the
  smallest distance. Unknown legacy history receives zero age penalty.
  Re-witnessing does not move a row; a read does not refresh it. This is
  5.5's stream history, not a document index, wall clock or last-use clock.
  The retained parameter name denotes an exponential scale, not a literal
  half-life. There is no free-running age-decay pass in either profile.

Weights `w_t, w_u, w_l, w_a` are `model.xml` elements (§6). Terms are in
`[0, 1]` so the weights are comparable.

## 4. The pass

1. **Trigger.** After a document boundary's host-side work (never inside
   a sentence, never inside a compiled region), if
   `count ≥ <forgetHighWater> · capacity`. Also at the end of an epoch
   regardless of occupancy when `<forgetAtEpoch>` is on. Not more often
   than `<forgetMinInterval>` sentences since the last pass.
2. **Protection.** Rows whose origin is in `<forgetProtectOrigins>`
   (default: provisioned and user) are never coarsened or deleted; they are supplied
   truths with their own replace-on-resubmit and re-provision rules. Retained
   thought results, returned understanding and pending expectation references
   also protect their rows and semantic dependencies. If
   protected rows alone exceed the target, the pass logs and does nothing.
   *Noted (Alec, 2026-09-29).* A definition row, `REL_DEF`, ties a word
   concept to its object concept
   ([two truths §17](2026-09-16-two-truths-ideas-and-relations.md#17-definitions-word-def-object-decided-alec-2026-09-29)).
   It is **not** protected: "I don't see any need to keep it out of
   recency or forgetting, but perhaps the forgetting algorithm can be
   altered so that it biases forgetting or relative/ultimate truth." An
   earlier answer of the same day, "forgetting should probably not
   forget definitions", is replaced by this one. *Claude's reading, to
   confirm when item 5 is taken up:* the bias is a weight in `V` (§3) by
   the kind of truth, relative rows against ideas, with definitions
   among the relative; a definition's use, each reading of its word, is
   counted in its utility; and step 7 rebuilds the definitions table
   after compaction, since the rows it indexes have moved. This extra
   truth-kind/use weighting was not decided and is not added here; DEF rows
   use the ordinary relation value and origin protections.
3. **Value.** Compute `V` for every unprotected row (§3).
4. **Selection.** Delete the lowest-`V` unprotected rows until
   `count ≤ <forgetLowWater> · capacity`. Ties break oldest first.
5. **Cascade.** A relation row whose operand row is deleted is deleted with
   it, whatever its own value, because a relation over a missing row is
   meaningless. Cascade repeats until stable. The cascade is counted
   toward the target, so step 4 may delete fewer rows than it planned.
6. **Compaction.** Rows are compacted in store order. Stable occurrence
   addresses in `refs` do not change. Physical postings, definitions,
   adjacency, semantic contexts and fingerprints are rebuilt or filtered.
   Semantic references are validated before mutation. Coarsened rows are
   reindexed from their surviving content. Expectation-pair links are
   unlinked when an endpoint is removed, preserving a surviving observation.
7. **Dependents.** After compaction: the truth view (`sync_from_ltm`) is
   rebuilt; testimony-owned taxonomy edges lose deleted supports. An edge
   disappears only when it has no surviving support and no independent
   native definition. Provenance rides the allocator sidecar; legacy edges
   without provenance are conservatively preserved. Surviving learned
   coefficients and optimizer moments stay intact. Predictor tensors and
   pending expectation state are untouched; their live sources are protected.
8. **Record.** The pass appends one line to the training log: rows
   before, rows after, the value cut-off, the count deleted per origin,
   and the count deleted by cascade. A checkpoint records the number of
   passes, completed-sentence interval counter and last cut-off. Logging also
   distinguishes dropped wording, coarsening and an unreachable target.

## 4a. Detail before rows (Alec, 2026-09-20)

**Decision revised (Alec, 2026-09-28).** Detail can be lost before a row is
deleted, but [two truths §11.1](2026-09-16-two-truths-ideas-and-relations.md#111-a-row-holds-structure-never-a-derivation-decided)
removes the proposed derivation-decay stage: a row never stores a derivation.
The open reading's operation record is discarded at its closing, and the
retrieval index is built immediately by unfolding the row with semantic
priming. No later re-indexing transition from a trace is needed.

The remaining proposed order is wording before clauses before gist
(Sachs 1967; Reyna & Brainerd 1995):

1. **Wording.** Drop retained provenance source text, where present. The row
   already regenerates from its stored structure and remains cueable through
   terms obtained by unfolding. This frees sidecar memory, not store rows.
2. **Subordinate rows.** A row that survives only as an operand of another
   row — a clause under `NP → S` or `NP → REF(S)`, a link in a chained
   episode — goes before the row that refers to it. This frees store rows
   and counts toward the target.
3. **The row itself**, by value (§4 steps 3–4).

**Coarsening, not cascade, where the gist survives.** When a subordinate
*idea* row is dropped, the row referring to it is kept: its slot already
holds that operand's fused point, so its `refs` entry becomes `-1` and it
reads thereafter as a relation over a point with no inner structure. Step 5's
cascade still applies where there is no point to keep — an operand that was
itself a relation, since fusion is fair only where operands have points
([two truths](2026-09-16-two-truths-ideas-and-relations.md)).

**Gradual.** A pass takes each row at most one stage further, lowest value
first, and stops as soon as the target occupancy is met. A row's main idea,
its trust and its `surprise` are unchanged by stages 1 and 2.

**Profiles.** Pressure-driven detail loss and the bounded age preference
apply under both profiles. Neither profile retains a derivation or performs
age-driven decay without pressure. The explicit epoch option remains available.

The host hook runs after the final backward and reset at a document boundary,
or after a stateful inference request. A document can still exhaust capacity
before reaching that boundary; atomic admission then raises. Protected roots
and one-stage detail loss can also prevent reaching low water. These cases
warn or fail visibly instead of deleting live content or partially admitting
a graph. Capacity must include room for a complete document transaction.

## 5. What is not forgotten and what is not done

- Concept rows have a separate capacity recovery pass, after all streams
  close and backward completes. Reaching maximum preallocation warns,
  including an allocation that exactly fills it. The pass groups identical
  nonempty definitions, orders, sparse coefficients and feature groups;
  it ranks redundant candidates by meaning magnitude and keeps a stronger
  representative (a referenced one takes precedence). Smallness or zero
  meaning alone is insufficient. **Also decided, 2026-10-19:** unused mints
  with no definition, reference/posting evidence or sparse placement may be
  reclaimed, smallest magnitude first. They need not be the same concept.
  Native records, definition rows, witness/reference metadata, feature groups
  and zero-valued sparse edges exclude this unused-mint route. A referenced
  or defined concept is never reclaimed merely for being small.
  Distinct order-one individuals are never
  merged by matching descriptions. Independently, unreferenced identity/change
  columns with zero relevance can be removed. Native references, definitions,
  postings, component coordinates, live thoughts and frozen concepts protect
  their targets. If nothing is safe, a warning reports no reclaimed seat.
  Physical seats can be reused with cleared moments; retired handles remain
  retired, and new handles increase monotonically. Allocator reclamation
  counts and maps survive checkpoints. Existing `n/(n+4)` definedness keeps
  an unwitnessed derived meaning zero and grows its magnitude with witnesses;
  it is a certainty proxy, not a calibrated probability.
- Derivation decay is vacuous under the September 28 row contract (§4a).
  Learning experiments during an open sentence are outside this forgetting
  pass; [FutureWork.md §3](../FutureWork.md#3-lossy-reconstruction-trace-inversion-as-learning) records the revised decision.
- No merging of near-duplicate rows into a coarser row. A row is coarsened
  only by losing its own detail (§4a), never by being merged with another.
- No forgetting inside a request in stateless serving; the pass runs in
  training and in stateful serving only.

## 6. Configuration

All under `<architecture>`; the hard limit remains `SymbolSpace.ltmCapacity`:

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
| `forgetAgeWeight` | `1.0` | `w_a`, both profiles under pressure |
| `forgetHalfLife` | `4096` | exponential age scale in retained stream rows |
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
4. **Cascade and references.** Point-bearing idea operands can coarsen their
   parents; missing relational operands cascade. Surviving addresses keep
   their identity and stream order; no reference names a replacement row.
5. **Dependents.** After a pass, the truth view, the taxonomy index and
   `relations()` agree with the surviving rows; the discourse observation
   view is byte-identical before and after.
6. **Age.** In both profiles, older rows rank lower at otherwise equal value.
   Trust/utility can outweigh age; shared and unknown histories follow §3.
7. **Surprise column.** An idea row observed without a staged expectation
   has surprise `-1` and is treated as maximally useful; a row observed
   with a near-zero discrepancy is treated as deducible.
8. **Packed and single-sentence parity.** The same ordered stream and document
   boundaries, with presentations grouped differently, retain the same rows.
9. **Checkpoint.** The pass count and last cut-off round-trip; a
   checkpoint saved between passes reloads with the same occupancy.
10. **Determinism.** The same state deletes the same rows, with stable-address
    tie breaking; no random selection or test seed is required.
11. **Detail before rows.** Under pressure a pass drops wording from the
    lowest-value rows, then subordinate rows, and deletes a standalone row
    only when that has not reached low water; after its trace is dropped a
    row is retrieved by a code its point regenerates and not by a leaf code
    it no longer regenerates.
12. **Coarsening, not cascade.** Dropping a subordinate idea row leaves the
    referring row with that operand's point and `refs = -1`, its trust and
    `surprise` unchanged; the referring row is deleted by cascade only when
    the dropped operand had no point.
13. **Gradual.** Successive passes take the same row one stage further
    before it is deleted; no pass takes a row more than one stage.
14. **No free-running decay.** With no pressure or explicit epoch trigger,
    neither profile drops detail at any age.

The implementation certificates live in `test_forgetting.py` and
`test_concept_forgetting.py`, alongside the existing occurrence, taxonomy,
expectation-retention and structural-checkpoint tests. Learning and long-run
capacity sizing remain separate gates; this mechanism makes no learning claim.

## 8. Documentation required with implementation

- STM.md's LTM consolidation section: the pass, its trigger and the value.
- Reasoning.md: deducibility as the utility term; one step only.
- Philosophy.md discrepancy 3 marked as specified here; the luminosity
  term's relation to the catuṣkoṭi coverage.
- Params.md: the elements in §6.
- FutureWork.md §2: replaced by a pointer to this spec; §3's schedule is
  §4a here.
