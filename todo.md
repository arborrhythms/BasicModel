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
§10 and its completion gates. Validation: the fresh, source-matched default
selection completed 4,668/4,668 with a green receipt in 1,040 s
(`output/tests/20260919-125200-894ef3`), using eight-case/one-file batches
under the unchanged resource caps. Default batch memory stability remains
under Residues below. The September 17 handoff and preserved
candidates are in
[doc/checkpoints/2026-09-17-production-spec/README.md](doc/checkpoints/2026-09-17-production-spec/README.md);
of its six candidates only the generation catalogue (item 3) remains uninstalled.

### Done (newest first)

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

1. **Selected linguistic meaning and the normal thought controller** over the
   `<thought>` catalogue. One grammar of thought in conceptual space: the
   `ThinkingKernel`, `NeuralToolUser` and the thought/query MLP are three
   drafts of it — merge them, add no fourth selector. Exit, all required:
   - one controller; `ThinkingKernel`'s addressee table and `Testimony`,
     `NeuralToolUser.run_legacy_world` and the legacy frame kernel deleted;
     `reason_about` / `think_about` / `answer_query` route through
     `_query_boundary_scope`, opened at each completed thought boundary;
   - typed adapters for set, code and subgoal results (truth and prediction
     landed); `what(Q)`'s `continuation` is the controller's child push;
   - converse/paraphrase and nested-reference meaning across all three writers
     (only the anchored `whole` converse is done; non-anchor and nested cases
     open);
   - full-width mandatory roles, actual shared work costs recorded in the
     controller's own episode meter, causal child evidence, bounded return;
   - replay/checkpoint isolation and policy credit (candidates detached,
     gradient through the chooser — the one hard-choice credit contract, one
     entry in [GradientFlow](doc/GradientFlow.md));
   - the prepared selected-meaning probes red first, then green.
2. **Expectation and residual learning.** Extend retained estimates to checked
   bindings/scope metadata without copying the arriving target; prior-view
   isolation from arriving/unseen input and other rows; residual query credit
   with its separate baseline and parameter-version-safe trajectories.
   Mechanism probes alone do not satisfy the learning gates.
3. **Generation ownership and end-to-end output.** Rebase the preserved
   generation-catalogue candidate (it no longer applies to primary); validate
   checkpoint/optimizer migration and normal supervised output; keep the §8.4
   downstream/reconstruction gradient contract and GradientFlow.md current.
4. **Evidence gates and documentation.** Held-out causal utility,
   reconstruction/discrimination controls, warmed training throughput; the
   preserved arbitrary-symbol poison probes and renamed-vocabulary controls.
   Numerical values or symbol IDs never supply learner arithmetic or answer
   seeds. Record null results honestly.
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
   trust; do not assume one word per META (§3.4). Exit: the sixteen §7 tests,
   the §8 docs, the item 6 reconstruction baseline unchanged, **and the
   `true` operator over the sealed clause declared in `<thought>` and
   executable** (deferred here by the unification plan). Claude reviews.
2. **Forgetting spec** ([doc/specs/2026-09-16-forgetting.md](doc/specs/2026-09-16-forgetting.md)),
   after item 1 (needs `refs` and every S writing a row). Document-boundary
   pass from the high-water to the low-water mark deleting the lowest-value
   unprotected rows; value = `|trust|` (Alec, September 18) + utility
   (1 − deducibility) + luminosity contribution; cascade, reference remap,
   dependents rebuilt, the human profile's age term. Exit: the ten §7 tests,
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
  `69233d1`/`b705a48`. [ExistenceEvidence](doc/ExistenceEvidence.md)'s
  "nested-meaning traversal not implemented" line to be reconciled against
  `3ab3c4b`.

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
