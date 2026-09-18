# TODO

## NEXT — finish this session and the integrated production/thinking spec

**This is the next work to do, before the later-session tasks below.** Resume
and complete the work started in this round against
[2026-09-15-next-sentence-as-the-production-objective.md](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md),
following its §10 order and completion gates. The September 14 ownership fixes
are already recorded as complete; the integrated spec is **not** complete.

Alec requested an OS-update checkpoint on September 17. That checkpoint's
single full receipt timed out with 2,907/4,458 cases completed. On September
18, under Alec's no-rerun instruction, a composite bounded coverage record
completed all 4,502 current default node IDs; the two failures found during
that record were fixed and re-run green. This is not a replacement for a fresh
single-snapshot full receipt after later source changes. The current
runner/device selection passed 29/29 and explicit significant training routed
to MPS. See [Testing](doc/Testing.md#validation) for durable receipts and limits.
[Handoff, preserved candidates and recovery instructions](doc/checkpoints/2026-09-17-production-spec/README.md).

The subsequent thought-operator snapshot completed all 4,623 current default
node IDs in bounded fresh workers. One assertion still expected an unwrapped
unary router kernel after both unary and binary faces adopted the common
structural adapter; the assertion was corrected and that exact node passed
1/1. Per Alec's no-rerun instruction, this is composite current coverage, not
a fresh single-snapshot all-green receipt. The exact receipts are in
[Testing](doc/Testing.md#thought-operator-catalogue-record-september-18).

### Completed in this session

- **Bounded-test performance and MPS routing.** The runner keeps finite
  memory/deadline limits and fresh-process recycling; heavy integration tests
  are slow-gated rather than removed; the VQ distance tile has a 512 MiB bound;
  and supervised training has an explicit successful MPS-routing receipt.
  Published in BasicModel `3ffb465` and WikiOracle `905fec7`.

- **Ordinary thought history and live occurrence reads.** One existing
  `WhatInteractionMemory` owner now stores replayable ordinary transitions,
  preserves the shared budget/forced drain and checkpoint/gradient boundaries,
  and serves row-local live thought occurrences. Its reviewer evidence is
  recorded in [Thought history](doc/ThoughtHistory.md). This is a foundation,
  not normal-controller or learned-utility completion.

- **Sentence/query phase permission.** Checked execution now opens only for
  completed owned rows at answer resolution, while sentence work and compiled
  tracing are masked. The rebased reviewer probes were red first, then 23/23
  and 280/280 affected cases passed; see [Query phases](doc/QueryPhases.md).

- **Bounded nested occurrence retention.** The existing LTM and ordinary
  history owners now retain reachable request content without retaining its
  fact authority, validate local semantic structure atomically, and defer
  stateless compaction until checkpoint owners are restored. The preserved
  reviewer probes were red first; the focused 68/68 and broader 309/309
  bounded affected gates are green. See [Nested retention](doc/NestedRetention.md).

- **Shared selected-query work accounting.** One transient meter now covers
  selected VP/operand preparation, execution, native reads, taxonomy
  capture/traversal, prediction context, and nested callbacks without creating
  a second semantic owner or controller. The rebased probes were red first;
  13/13 focused and a non-overlapping 372/372 affected bounded selection are
  green. The normal controller still must create and record its episode meter.
  See [Query work](doc/QueryWork.md).

0. **Complete — thought operations as compose rules** — before the remaining
   controller and before the two-truths and forgetting specs below, so that
   none of them is built on the `<Queries>` catalogue and then redone
   ([doc/specs/2026-09-18-thought-operations-in-compose.md](doc/specs/2026-09-18-thought-operations-in-compose.md),
   Alec 2026-09-18). One grammar: a compose or generate rule has a structural face and,
   when an executor of the same name is registered, an executable face the
   thought controller runs at a boundary. `<Queries>` and `query="false"` are
   rejected in production; historical `is` aliases are quarantined rather than
   authoritative. Arity, roles, open-role variants and the converse permutation
   derive from the rule; the executor table keeps only
   domain, kinds, scopes, evidence kind and the executor. The controller's
   action set is the compose operators with executors plus conclude; `arma`
   is its synthesis face and emits the `[3, D]` end state. Any controller
   work already in progress under item 1 aligns to this catalogue rather
   than to `<Queries>`. `true` remains reserved for the separately deferred
   two-truths sealed-clause representation. Exit evidence: the adjusted
   spec §9 catalogue/context/phase tests and the no-rerun default coverage
   record above; documentation is in spec §10.
1. **Finish selected linguistic meaning and the normal thought controller**
   over the item 0 catalogue.
   Preserve selected signed operands and native middle VP across all three
   writers; cover mode, polarity, paraphrase/converse and nested references.
   Integrate full-width mandatory roles, actual shared work costs, causal child
   evidence, bounded return, replay/checkpoint isolation and policy credit.
   Obtain the prepared selected-meaning probes' red before implementing them.
   The kernel, `NeuralToolUser` and the thought/query MLP are three drafts of
   one grammar of thought in conceptual space; merge them, do not add a
   fourth selector (`run_legacy_world` and the addressee/`Testimony` route
   are retired with item 0).
2. **Finish expectation and residual learning.** Complete owned estimates and
   observation links, prior-view isolation from arriving/unseen input and other
   rows, residual query credit, the separate baseline and parameter-version-safe
   trajectories. Mechanism probes alone do not satisfy the learning gates.
3. **Finish generation ownership and end-to-end output.** Rebase the preserved
   generation-catalog candidate after the earlier integrations; validate actual
   checkpoint/optimizer migration and normal supervised output. Maintain the
   combined downstream/reconstruction gradient contract from §8.4 and keep
   [the architecture-wide gradient map](doc/GradientFlow.md) current.
4. **Close the evidence gates and documentation.** Measure held-out causal
   utility, reconstruction/discrimination controls and current warmed training
   throughput. Run the preserved arbitrary-symbol runtime poison probes and
   renamed-vocabulary learning controls. Numerical values or symbol IDs must
   not supply learner arithmetic or answer seeds. Record incomplete or null
   results honestly. Update the canonical spec and this list as items land.
5. **Publish each completed item:** failing probe → fix → affected files → full
   default suite green in the background → BasicModel commit/push → WikiOracle
   submodule bump/push, with the required co-author trailer. Preserve user-owned
   files. Do not remove unused reasoning methods without Alec's review.

**Exit:** the remaining §10 items and their semantic, gradient, lifecycle,
learning and validation gates have evidence; the completed work, current spec
and todo are committed and pushed in both repositories. Then proceed to the
separately reserved two-truths and forgetting work below. The earlier notes are
preserved verbatim and should be reconciled with landed work before acting on
their older status statements.

## Before the long FineWeb run (2026-09-16)

Ordered. Each item names its owner, its spec or plan section, and its
exit test. Written 2026-09-16 from the review session with Alec; keep
this file current as items land (move them to "Done" with the commit).

### Codex

0. **Thought operations as compose rules** (NEXT item 0 above) precedes
   items 1–3 here: the two-truths seal (`NP → REF(S)`) and the `true`
   operator, and the expectation review's query paths, are specified against
   the compose-listed catalogue.

1. **Implement the two-truths spec**
   ([doc/specs/2026-09-16-two-truths-ideas-and-relations.md](doc/specs/2026-09-16-two-truths-ideas-and-relations.md)),
   in a new session. One grammatical S is one LTM row: an absolute S fuses
   to one point (the existing depth-1 reduce) and writes an idea row with
   its derivation and `refs`; a relative S (generic subject, or any S that
   references a relation) stays three slots and writes a relation row of
   kind part, implies or operator over row references. Clause-level seal as
   grammar (`NP → S`, `NP → REF(S)`), one relation writer at the seal,
   `REL_OTHER` and the reducible/ineffable routing deleted, the WholeSpace
   META taxonomy retired for a concept-level index, luminosity restricted
   to idea rows, the sentence never setting its own trust. Do not assume
   one word per META (§3.4). Exit: the sixteen tests in §7, the
   documentation in §8, and the reconstruction baseline of item 4 below
   unchanged. Claude reviews afterwards.

2. **Implement the forgetting spec**
   ([doc/specs/2026-09-16-forgetting.md](doc/specs/2026-09-16-forgetting.md)),
   after item 1, since it depends on the `refs` column and on every S
   writing a row. A pass at a document boundary when the store passes the
   high-water mark, down to the low-water mark, deleting the lowest-value
   unprotected rows: value = trust magnitude + utility (one minus
   deducibility: the per-row surprise column for ideas, one-step
   derivability for relations) + luminosity contribution (the dissonant-pair
   test; coverage gain behind a knob). Cascade over deleted operands,
   reference remap on compaction, dependents rebuilt, the human profile's
   age term. Exit: the ten tests in §7, the elements in §6 in schema,
   `model.xml` and Params.md, the documentation in §8.

3. **Address the expectation review** (integrated plan
   [§11](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md#11-code-review-2026-09-16-local-role-expectation-implementation)).
   In order of importance: the discourse layer's Reset honours `hard` so
   the packed loop's per-brick soft reset keeps the observation view (until
   then expectation scores only inside a brick); rename the September 16
   surface to "expectation", turn it on in `model.xml`, and make document
   staging tolerant of unaddressed rows; one owner for the What interaction
   memory with the delegates deleted; the seal-layout assertion; the stale
   references. Exit: the tests named in §11.1 through §11.4 and the running
   suite green.

### Alec

4. **Xcode licence accepted (September 18).** Clang, `/usr/bin/git` and
   `/usr/bin/python3` are no longer blocked by this prerequisite. Compiler and
   long-run validation still require their separate measured gates.

5. **Trust term sign decided (September 18).** The forgetting specification
   uses `|trust|`: a strongly distrusted row is knowledge and must remain
   protected alongside a strongly trusted row. Preserve this when the deferred
   forgetting implementation begins.

### Claude, before Codex merges item 1

6. **Take the reconstruction baseline on the current tree.** The integrated
   plan §8.2 requires retaining the measured reconstruction baseline across
   any prediction change, and item 1 moves the seal and changes what the
   packed drain and LTM sink write. Record reconstruction loss and the
   packed/single-sentence parity numbers at a fixed seed and config, and
   gate the item 1 merge on matching them. Exit: the numbers in the plan's
   §8.2 with the commit they were taken at.

### Claude, alongside items 1 to 3

7. **Build the run harness and the resume test.** The piece that decides
   whether the long run can be judged. One logger emitting, per interval:
   reconstruction loss; expectation discrepancy (occupied-role MSE and
   presence loss, and the kind logit once item 1 lands); LTM occupancy,
   forgetting passes, rows deleted per origin and the value cut-off;
   luminosity of the provisioned truths; and a held-out probe consisting
   of the two-truths §7 test 12 sentences (generic subject versus token
   subject) plus a fixed reconstruction sample. And a resume test proving
   that a mid-epoch checkpoint restores the cursor position, the stream
   count check, the store's new columns (`refs`, surprise) and the
   forgetting counters, with the next batch byte-identical to an
   uninterrupted run. Exit: one command that runs the harness on a small
   config and prints the interval report; the resume test in the suite.

8. **Verify the corpus at the target size.** Production loads 2,000
   documents from shard 0 into memory with one address per sentence
   (`BasicModel.xml` `maxDocs`). Raise `maxDocs` to the run's size and, if
   the run spans shards, exercise the multi-shard path; measure the
   sentence list and address table memory, the loader time, and that
   `resume_skip` still maps rows to ticks with the run's stream count.
   Exit: a recorded load at the target size with its memory and time, and
   the chosen `maxDocs`, `shardDir` and stream count written into the run
   config.

### Then

9. **Compiler work.** The levers recorded in memory and the fold-ladder
   plan: the forward chooser split/lift once per slot; the backward's
   launch-bound tiny-kernel count per brick; the B24 brick's remaining
   overhead versus pre-ladder. Exit: sentences per second and peak memory
   at the run's batch and brick size, recorded against the July baseline.

10. **The long FineWeb run.** Start only with items 1 to 8 done and item 9
    measured. Expectation on in `model.xml`; `BasicModel.xml`'s flip
    follows the plan's §10 gates, for which the run itself is the
    evidence. Watch the item 7 report; stop on a rising expectation
    discrepancy, a reconstruction regression against item 6, or a
    forgetting pass that deletes protected rows.

### Not needed for the run (recorded so nobody waits on them)

- The answer path and its owned output programs; FineWeb is
  self-supervised.
- Queries moved to prediction with residual credit (plan §8.10); surface
  prediction; derivation decay; context across documents; the n-ary META
  chooser. All in [doc/FutureWork.md](doc/FutureWork.md).
- The `BasicModel.xml` expectation flip; see item 10.

### Done

(none yet)


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
