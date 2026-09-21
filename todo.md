# TODO

> **Keeping this file useful (Alec, 2026-09-19).** This file lists what is
> left; it is not a log. A landed item is one line with its commit hash under
> "Done"; its evidence and narrative live in the document the line links to
> ([Testing](doc/Testing.md) for receipts, the topic doc for the design). An
> item leaves the list when its exit criterion is met, not when work on it
> starts. When a spec or plan leaves residue, that residue is a line here or
> it is lost. Reconcile this file at every landing, before the commit.

## NEXT — the integrated production/thinking spec

Work follows
[2026-09-15-next-sentence-as-the-production-objective.md](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md)
§10 and its completion gates. **Item 1's bounded supervised wording gate is
complete; items 1b, 1c and 1d are complete. Next: item 2, after review of 1d.**
Natural word → operator associations use the existing compose/generate grammars.
Structural-preference and routing-share measurements
remain design goals; learned questioning utility remains unproven under item 4.
Current contracts,
limits and receipts are in
[SelectedMeaning](doc/SelectedMeaning.md),
[KernelRetirement](doc/KernelRetirement.md) and [Testing](doc/Testing.md).
The September 17 handoff and preserved candidates are in
[the checkpoint](doc/checkpoints/2026-09-17-production-spec/README.md);
only the generation catalogue (item 3) remains uninstalled.

### Done (newest first)

- `6bf211a` Item 1d restores rotation-owned concept codes and completes the 1b/1c review corrections ([GradientFlow](doc/GradientFlow.md), [receipt](doc/Testing.md#item-1d-review-corrections-september-21)).
- `101dc22` Item 1c complete in mechanism: checked subsystem effects, indexed cued retrieval and retained frames; compound recovery remains unproven ([AccessibleMind](doc/AccessibleMind.md), [receipt](doc/Testing.md#accessible-mind-effects-september-20)).
- `7c2fa5a` Item 1b complete: objective-local state gradients and measured shared operator/codebook credit ([GradientFlow](doc/GradientFlow.md), [receipt](doc/Testing.md#gradient-factorization-september-20)).
- `6a6bb21` Item 1 complete: grammar-owned wording on real parsed text, held-out complete wordings/nouns, converse/sense controls and full-vocabulary generation/recomposition ([evidence and receipt](doc/Testing.md#working-grammar-wording-gate-september-20)).
- `d1d8b5b` Ordered compose-MLP inputs and production generate wiring (item 1 increment; the wording gate remains open, [SelectedMeaning](doc/SelectedMeaning.md), [Testing](doc/Testing.md#grammar-owned-wording-architecture-september-20)).
- `02db7ef` Review corrections: removed the standalone codec and legacy policy; completed chooser context, bounded memory and nested-controller mechanisms ([SelectedMeaning](doc/SelectedMeaning.md), [Testing](doc/Testing.md)).
- `288b56b` Initial controller replacement; review reopened item 1's wording gate ([current design and limits](doc/SelectedMeaning.md), [original receipt and review corrections](doc/Testing.md#selected-meaning-and-one-controller-september-20), [test dispositions](doc/KernelRetirement.md)).
- `60b1497` Forward owns anchored lexical forms by WORD row; capture reads retained provenance ([Testing](doc/Testing.md#forward-owned-lexical-forms-september-19); item 1 increment).
- `a70c77d` Direct `concept`-unary selected meaning (`what(quantize(x))`)
  preserves its live leaf into the normal controller; description operands
  still require owned occurrences. Ordinary pytest failures now complete their
  selected receipt while resource/protocol failures remain bounded/fail-fast
  ([Testing](doc/Testing.md#complete-diagnostic-receipts-september-19)).
- `943a64f` Lexical converse forms: `whole` vs `part` provenance frozen at
  program capture ([Testing](doc/Testing.md#forward-owned-lexical-forms-september-19)
  records the later forward-ownership correction).
- `fb5611c` Normal thought resolution grammar-owned; `resolveAnswer()` no
  longer calls `answer_query()` from a raw prompt (item 1 increment).
- `ea92851` **Item 0 complete:** model-declared `<thought>` allow-list between
  `<compose>` and `<generate>`; `<Queries>`, the `is`-aliases and
  `query="false"` retired. [Spec](doc/specs/2026-09-18-thought-operations-in-compose.md),
  [plan](doc/plans/2026-09-18-thought-operator-unification.md),
  [receipts](doc/Testing.md#explicit-thought-catalogue-september-19).
- `573fe0c` Bounded tests fast: no background QoS, `cpu−4` one-thread pool,
  28 GiB aggregate cap; 4,649 cases in 197 s (was 4,165 s serial).
- `6a34bbf` Sentence expectations retained as `estimate` occurrences without
  LTM leakage ([ExpectationRetention](doc/ExpectationRetention.md); item 2 foundation).
- `c3918a2` `9dc5c58` Typed `ThoughtResult` and nested evidence persist through
  history checkpoints (sidecar v3; [ThoughtHistory](doc/ThoughtHistory.md)).
- `e8b2f43` Catalogue refinement preserves mode, polarity, bindings, scope.
- `b705a48` `69233d1` Selected truth and `arma` results become the row's answer
  seed; `arma` emits `[3, D]`, never a fact.
- `04b34a3` Normal What runs on the selected thought trace.
- `4518db0` Structural and thought operators unified (predecessor of `ea92851`).
- `7923845` Shared selected-query work meter ([QueryWork](doc/QueryWork.md)).
- `3ab3c4b` Nested grammatical occurrences retained ([NestedRetention](doc/NestedRetention.md)).
- `3a5703e` Checked query execution phases ([QueryPhases](doc/QueryPhases.md)).
- `985b594` Replayable ordinary thought history ([ThoughtHistory](doc/ThoughtHistory.md)).
- `3ffb465` Bounded runner foundations, 512 MiB VQ tile, MPS routing receipt.

### Open

2. **Expectation as a negative image** (Alec 2026-09-20; design:
   [spec §2.6](doc/specs/2026-09-20-accessible-mind-subsystems.md#26-expectation);
   meaning: [Philosophy](doc/Philosophy.md#expectation-as-a-negative-image-attention-as-exclusion-2026-09-20)).
   Composition stays pure: no estimate enters compose, and the
   comprehension-time additive prior (`_c_prior`, `sentencePrimingScale`) is
   deleted. At the seal the sign-reversed estimate is added to the composed
   idea per role, `c = o − g·(1 − m)·κ·ê` — `κ` the predicted presence, `m`
   the open roles of the active question (the object of observation is
   spared), `g` a `model.xml` gain whose zero is beginner's mind. The
   surprise `r = o − ê` is what is learned, target detached, identical at
   every gain; its magnitude is the row's `surprise`. An expected role left
   empty is a conceived absence: evidence for the chooser, concluded as
   `not X` only in thought, never written by compose. The retained
   estimate/observation pair is the pointer and tag; no third record.
   Residual credit never trains the gain, the object of observation or
   reading attention. Then, as before: retained estimates extended to
   checked bindings/scope metadata without copying the arriving target;
   prior-view isolation from arriving/unseen input and other rows; residual
   query credit with its separate baseline and parameter-version-safe
   trajectories. Mechanism probes alone do not satisfy the learning gates.
   Exit: spec tests 3, 19, 23–30 and 32, plus the plan's §10 expectation
   gates (test 31 needs the forgetting pass and belongs to that item).
3. **Generation ownership and end-to-end output.** Rebase the preserved
   generation-catalogue candidate (it no longer applies to primary); validate
   checkpoint/optimizer migration and normal supervised output under item
   1b's contract (output error stops at the concluded idea; shared operators
   train); keep GradientFlow.md current.
4. **Evidence/design goals and documentation.** Prefer understandable structural
   operators when
   they carry the meaning; any opaque operator must be an ordinary grammar-MLP
   choice, with the structural face preferred at equal fit. Measure the opaque
   routing share and its decrease as structural coverage grows on the same
   corpus. These preference/routing measurements remain design goals.
   Held-out causal utility against direct-answer and equal-compute baselines
   across seeds,
   reconstruction/discrimination controls, warmed training throughput; the
   preserved arbitrary-symbol poison probes and renamed-vocabulary controls.
   Numerical values or symbol IDs never supply learner arithmetic or answer
   seeds. Learned utility stays explicitly unproven until these comparisons
   pass, even when mechanism and regression tests pass. Record null results honestly.
5. **Publish each item:** failing probe → fix → affected files → one
   source-matched full receipt → BasicModel commit/push → WikiOracle bump/push,
   with the co-author trailer. Do not remove unused reasoning methods without
   Alec's review.

**Exit:** the §10 gates have evidence and both repositories are pushed. Then
the pre-FineWeb list below.

## Before the long FineWeb run

### Codex

1. **Two-truths spec** ([doc/specs/2026-09-16-two-truths-ideas-and-relations.md](doc/specs/2026-09-16-two-truths-ideas-and-relations.md)),
   new session. One S = one LTM row: an absolute S fuses to one point and
   writes an idea row with derivation and `refs`; a relative S (generic
   subject, or any S referencing a relation) stays three slots and writes a
   relation row of kind part / implies / operator over row references.
   Clause-level seal as grammar (`NP → S`, `NP → REF(S)`), one relation
   writer at the seal, `REL_OTHER` and the reducible/ineffable routing
   deleted, the WholeSpace META taxonomy retired for a concept-level index,
   luminosity restricted to idea rows, the sentence never setting its own
   trust; do not assume one word per META (§3.4); **object permanence by
   reference** (§3.5, Alec 2026-09-21): translating a word to its object may
   tie it to an earlier noun or sentence in the recency buffer; then the
   distributional context widens from the sentence to the **situation** the
   predictor anchors, under three `model.xml` variables (plan §8.4 point 2:
   situation weight, anchor bound, expectation weight). Exit: the
   twenty-one §7 tests,
   the §8 docs, the item 6 reconstruction baseline unchanged, **and the
   `true` operator over the sealed clause declared in `<thought>` and
   executable** (deferred here by the unification plan). Claude reviews.
2. **Forgetting spec** ([doc/specs/2026-09-16-forgetting.md](doc/specs/2026-09-16-forgetting.md)),
   after item 1 (needs `refs` and every S writing a row). Document-boundary
   pass from the high-water to the low-water mark deleting the lowest-value
   unprotected rows; value = `|trust|` (Alec, September 18) + utility
   (1 − deducibility) + luminosity contribution; cascade, reference remap,
   dependents rebuilt, the human profile's age term; and **detail before
   rows** (§4a, Alec 2026-09-20): wording, then subordinate rows, then the
   row, gradually, coarsening the referring row instead of cascading where
   the operand's point survives. Exit: the fourteen §7 tests, the
   accessible-mind spec's test 31 (retention by surprise),
   the §6 elements in schema/`model.xml`/Params.md, the §8 docs.
3. **Expectation review** (plan [§11](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md#11-code-review-2026-09-16-local-role-expectation-implementation)):
   discourse `Reset` honours `hard`; rename to "expectation", on in
   `model.xml`, staging tolerant of unaddressed rows; one owner for the What
   interaction memory; the seal-layout assertion; stale references. Exit:
   the §11.1–§11.4 tests and the suite green.

### Claude

4. **Reconstruction baseline** before Codex merges item 1: reconstruction
   loss and packed/single-sentence parity at a fixed seed and config,
   recorded in the plan's §8.2 with the commit; gate the merge on matching.
5. **Run harness and resume test:** one logger per interval (reconstruction
   loss; expectation discrepancy; LTM occupancy, forgetting passes, rows
   deleted per origin, value cut-off; luminosity of provisioned truths;
   the per-shared-operator gradient cosine of NEXT item 1b — reconstruction
   vs expectation, and vs output where answers are supplied — with the
   operators showing persistent negative cosine named in the report;
   held-out two-truths §7 test-12 probe plus a fixed reconstruction sample)
   and a resume test proving a mid-epoch checkpoint restores cursor, stream
   count, `refs`/surprise columns and forgetting counters with the next batch
   byte-identical. Exit: one command on a small config; the test in the suite.
6. **Corpus at target size:** raise `maxDocs`, exercise multi-shard if needed,
   measure sentence-list/address-table memory and loader time, confirm
   `resume_skip` with the run's stream count. Exit: the load recorded and
   `maxDocs`/`shardDir`/stream count in the run config.

### Then

7. **Compiler work:** forward chooser split/lift once per slot; backward's
   launch-bound kernel count per brick; B24 brick +25% vs pre-ladder. Exit:
   sentences/s and peak memory at the run's batch and brick size against the
   July baseline.
8. **The long FineWeb run**, only with 1–6 done and 7 measured. Expectation on
   in `model.xml`; the `BasicModel.xml` flip follows the plan's §10 gates.
   Stop on rising expectation discrepancy, a reconstruction regression against
   item 4, or a forgetting pass deleting protected rows.

### Done

- Xcode licence accepted (Alec, September 18).
- Trust term sign: `|trust|` (Alec, September 18).
- Thought operations as compose/thought rules (`ea92851`, NEXT item 0).

## Residues of earlier specs (audit 2026-09-19)

Small, real, and not on the critical path. Fold each into the nearest Codex
change; delete the line with the commit.

- **Stored-idea generativity — not small: forgetting §4a's dropping of
  derivations depends on it.** Item 1c's probe reports zero compound recovery
  ([measurements](doc/AccessibleMind.md#measured-limits)). It trains for 8
  small updates, so it mostly measures a split/stop policy that has not learned
  to split (it does within ~100). The deeper limit is the operator: with the
  correct split actions *forced*, the tied inverse of `lower` returns children
  about 65% from their codes at depth 1, they project to the wrong code, and
  400 updates of the probe's training do not move that. Report the two
  separately (forced actions = the operator's inverse; free-running = the
  policy, trained to convergence), and evaluate clean-up decoding for lift /
  lower: bounded candidate search through the forward kernel
  (`_bounded_binary_reconstruction`) in place of the reference-free affine
  inverse. Until a recovery rate is measured, forgetting must not drop
  derivations.
- **Bounded test batches:** default 256-case/16-file and 32-case/4-file
  workers exceeded the 8 GiB cap; eight-case/one-file batches pass the same
  default selection. Restore safe, faster batching without raising caps or
  omitting cases; reconsider [fixture reuse](doc/plans/2026-09-17-bounded-fixture-reuse.md)
  if useful ([evidence](doc/Testing.md#forward-owned-lexical-forms-september-19)).
- **Fold ladder:** delete the radix-backed non-word-major meronomy path (no
  legacy paths); confirm the `dispatch_per_row_reset` note in
  [the plan](doc/plans/2026-09-10-meronomy-fold-ladder.md#open-defects-found-on-the-way)
  is stale (`taxonomy_parent_map` is now initialised) and remove it.
- **Plan status hygiene:** [compiled reverse loops](doc/plans/2026-09-12-compiled-reverse-loops.md)
  still lists contract 6 and the future predictor as open (both decided /
  landed); [answer-path §6](doc/plans/2026-09-14-answer-path-ownership-and-training.md)
  is titled "Open questions" but records answers — retitle; §3.7 closed by
  `69233d1`/`b705a48`.

### Superseded — no work (recorded so nobody re-derives them)

- Mathematical thinking's lexical `{ANSWER, OPEN, EXECUTE}` policy, its §12
  Q1/Q2/Q5 and self-cloning from verifier traces: replaced by the one grammar
  of thought (NEXT item 1). Q3/Q4 (numeral representation, clause lexing) and
  "multi-digit wholes, then counting" remain an evaluation target for the
  unified controller — in [FutureWork](doc/FutureWork.md), not here.
- Fold-ladder Q1–Q3 (admission knobs, coverage schedule, category-utility
  floor): tuning questions with no evidence until the long run produces some.
- The answer path and owned output programs, queries-as-prediction with
  residual credit, surface prediction, derivation decay, cross-document
  context, the n-ary META chooser: [FutureWork](doc/FutureWork.md).

## Deferred Teacher-to-LTM persistence

- When the deferred NP-VP transition cache is implemented, retain only
  detached, row-local student-produced records. Do not describe that cache as
  already implemented in Teacher v1 or commit reconstructions/predictions
  directly to the persistent/global truth store.
- Define an admission policy for student-generated memories with provenance,
  confidence, `asserted_at`, represented/target-event support, revision, and
  contradiction handling.
- Prevent Teacher-only clean targets and future information from entering
  student-visible LTM before evaluation.
- Add leakage, replay, correction, and document-boundary tests before enabling
  persistent writes.

## Objective-address conditioning

- Follow the [unified Teacher specification](doc/specs/2026-07-27-teaching-modes-and-next-iteration.md)
  and its gated milestones. The [What and spacetime design](doc/WhatSpacetimeDesign.md)
  describes the interface ownership and chooser/stack context.
- Keep corpus/snapshot/document/sentence/span coordinates on the Teacher query
  seam; never reuse or overwrite the model's subjective `.where`/`.when`.
- Split target observation/event time from source snapshot validity; retain the
  latter as provenance rather than overloading objective `when`.
- After the clean Teacher throughput gate, add a student-side encoder that
  embeds corpus/snapshot/split IDs categorically and document/sentence/span
  positions as ordered coordinates. Raw hash magnitude must have no meaning.
- Prove that an addressed clean input round-trips through the privileged
  `Teacher.Data(address)`, keeping `Teacher.What(where, when)` only as a
  controller-side compatibility adapter. Measure whether the separate
  `model.what(address)` uses addresses on partial and blank lessons without
  accessing private clean content.
- Treat DOI and source date as optional aliases/provenance. FineWeb has neither;
  retain shard SHA-256 and corpus release metadata instead of fabricating them.
