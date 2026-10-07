# TODO

> **Keeping this file useful (Alec, 2026-09-19).** This file lists what is
> left; it is not a log. A landed item is one line with its commit hash under
> "Done"; its evidence and narrative live in the document the line links to
> ([Testing](doc/Testing.md) for receipts, the topic doc for the design). An
> item leaves the list when its exit criterion is met, not when work on it
> starts. When a spec or plan leaves residue, that residue is a line here or
> it is lost. Reconcile this file at every landing, before the commit.

## Countdown to the full training session

Items count **down**: the next item to do has the highest number and **item 0
is the full training session**. (They are bullets, not a numbered list, because
Markdown renumbers a numbered list upward.) The order is the intended order; an item may
be taken early when it does not depend on a higher-numbered one. Codex builds;
Claude writes the specs and reviews each landing (Alec, 2026-09-21). "Done"
lines below keep the numbers their items had when they landed.

**Next:** **query and ask**, under [thought-loop §§5–7](doc/plans/2026-10-07-thought-loop.md#5-the-query-rename-claude-for-codex-2026-10-07), then **6.2, thinking**, whose specification follows from Claude. Alec accepted the [6.5 mechanism landing](doc/benchmarks/2026-10-07-item6-5/acceptance.json) on October 7 after spec §10 reviewed the §9 choices; its learning gates remain pending.

**Current sequence (6.5 mechanism accepted on October 7; no conference freeze):**
query and ask → 6.2 → 6 → 5.5 → 5 → 4 → 3 → 2 → 1 → 0.
The word-level evaluator built in 6.8 waits for item 4's trained checkpoint;
acceptance does not authorize fresh-model bulk scoring. Items 9 and 8 are implemented and
reviewed; what remains of them is empirical and waits for the
million-sentence checkpoint that item 0's run provides, so they are not
in the implementation queue.

Work follows
[2026-09-15-next-sentence-as-the-production-objective.md](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md)
§10 and its completion gates. Current contracts, limits and receipts are in
[SelectedMeaning](doc/SelectedMeaning.md),
[KernelRetirement](doc/KernelRetirement.md),
[ExpectationRetention](doc/ExpectationRetention.md) and
[Testing](doc/Testing.md). The September 17 handoff and preserved candidates
are in [the checkpoint](doc/checkpoints/2026-09-17-production-spec/README.md).

**Publish rule, every item (updated by Alec, September 24):** failing probe →
fix → affected files → one source-matched full receipt → **stop for Claude's
code review** → resolve review → BasicModel commit/push → WikiOracle bump/push,
with the co-author trailer. Do not commit or push before the review.
Natural word →
operator associations use the existing compose/generate grammars. Mechanism
probes do not satisfy learning gates, and a null result is recorded as null.
**No test pins a random seed in order to pass** (Alec, 2026-09-21): a fixed
seed may make a *measurement* reproducible, never an *assertion* true. A test
that fails at some seed has found a defect; fix the defect, or let it fail.
Do not remove unused reasoning methods without Alec's review.

**The grammatical operations and their inverses are still in development
(Alec, 2026-09-26).** The compose, thought and generate catalogs are a
fragment, their routing is learned with sparse credit, and their tied
inverses are exercised mostly by reconstruction. No item's gate, and no
conference number, should be read as a claim about the performance of the
grammar as a set yet; a poor result on wording, parse depth or free
generation is expected at this stage and is recorded, not tuned away.

- **9. Mature-checkpoint learning evidence.** The mechanism, checkpoint exposure
   counter and quality-evaluation machinery are reviewed and implemented
   ([receipt](doc/benchmarks/2026-09-26-item9-followup/README.md)). Learning
   acceptance remains deferred until a checkpoint has at least **one million
   completed FineWeb training sentences**; missing exposure skips, never passes,
   the quality evaluation. Correctness checks stay unconditional. No qualifying
   checkpoint has been evaluated. Then run the loaded-model wording, prediction
   and predictive-thought checks; the full causal gate still needs separately
   trained, equal-update controls across at least three seeds, with tasks, run
   length and thresholds declared first. Ordered prediction must beat shuffled
   and context-free controls without worsening reconstruction/discrimination
   against reconstruction-only; thought must improve error at matched work or
   reduce work at matched error. Same-checkpoint ablations do not establish this
   causal result, and fused AMP conservatively undercounts exposure. Preserve the
   tiny-run nulls, incomplete reverse programs and 5/56 wording failures as
   development diagnostics ([original receipt](doc/benchmarks/2026-09-26-item9/README.md));
   old context-free and assertion-thought scores are invalid controls.
   **6.5's learning gates** — held-out anaphora against `206a0146`, verb
   reuse, prediction control, shuffled order, renamed vocabulary and
   determiner control — remain pending the million-sentence checkpoint,
   across **seeds 0/1/2**, and are **not claimed** by its mechanism acceptance
   ([receipt](doc/benchmarks/2026-10-07-item6-5/README.md#learning-limits)).
- **8. Evidence: learned utility and structural preference.** Prefer
   understandable structural operators when they carry the meaning; any opaque
   operator must be an ordinary grammar-MLP choice, with the structural face
   preferred at equal fit. Measure the opaque routing share and its decrease
   as structural coverage grows on the same corpus. Held-out causal utility
   against direct-answer and equal-compute baselines across seeds;
   reconstruction/discrimination controls; warmed training throughput; the
   preserved arbitrary-symbol poison probes and renamed-vocabulary controls.
   Numerical values or symbol IDs never supply learner arithmetic or answer
   seeds. Learned utility stays explicitly unproven until these comparisons
   pass. The seed audit of `d4dc385` left the unseeded MM-grammar XOR gate
   failing (.21757 after 900 epochs, bar <.20), and the two `XOR_grammar.xml`
   CLI gates stopped before training at unsupported W=6. Keep these failures
   visible; no passing-seed selection or expected-failure waiver
   ([audit](doc/benchmarks/2026-09-21-item10/README.md#validation-and-limits)).
   *Compatibility:* `XOR_grammar.xml` composes `not` / `conjunction` /
   `disjunction` over symbols on the serial path and never used the deleted
   tower folds, so its gate stands as written. Reconstruction comparisons use
   the [reviewed 9b baseline](doc/benchmarks/2026-09-26-item9b-occurrence-fix/README.md):
   serial before/during/after .1005906649 / .0948241442 / .0928765051;
   packed/single byte reconstruction .6839025617 with exact parity. Learning
   quality gates follow the million-sentence prerequisite in item 9; mechanism
   and regression checks remain unconditional.
   The [item-8 implementation](doc/benchmarks/2026-09-26-item8-review/README.md) implements
   exact-score tie preference and committed-program routing measurements;
   its [protocol](doc/benchmarks/2026-09-26-item8/PROTOCOL.md) predeclares the
   qualified causal study. Learned utility and declining opaque routing remain
   unproven. W=6 scheduling now reaches the next unchanged
   CLI blocker: the six-row symbol inventory exhausts during the first epoch's
   reset/autobind, after batch execution. Keep both CLI assertions and the
   historical MM failing-seed result visible. All three declared diagnostic
   seeds were attempted; seed 1 hit the 8 GiB guard and remains incomplete.
   The initial 4,968-case full sweep exits 1 on an unchanged strict-XPASS
   reconstruction marker. Claude's [review corrections](doc/benchmarks/2026-09-26-item8-review/README.md)
   use the marginal uniformly and require complete reconstructed positions;
   the marker and threshold remain intact. The corrected source passes its
   single 4,980-case full sweep: 4,653 passed, 326 skipped and one expected
   failure. Reconstruction exactly matches reviewed 9b. The review corrections
   are validated for the requested commit; item 8's empirical gates remain open.
- **7.5. Deferred compose follow-ups (non-blocking).** Specify the nonzero
   training temperature and sentence parsimony/work term accepted for a later
   spec; evaluation remains deterministic. Restate the shared-operator report
   assertion around closing contributions or forced parametric selection, and
   rename the native context-read `exploration_trial` flag in housekeeping.
   **Found 2026-09-29 (6.9 plan §3.12):** the sentence pair costs its second
   trial after stepping on the first, so the comparison favours the second
   whatever its derivation; the correction is 6.9's step 5 (decided by
   Alec, 2026-09-30: "equal comparison").
   The unchanged depth-three campaign remains red; retain its assertion and
   the XOR/MM evidence ([accepted landing](doc/benchmarks/2026-09-27-item7-5-landing/README.md)).
Remaining operators work stays with its assigned items: surface, tense,
morphology, aspect, null and situation codes with 5.5; forgetting's truth-kind
bias with 5; trained-checkpoint evaluation with 4; the concept-face negative
image with 2; and the open-read/teacher-loss host island with 1. The
[FutureWork list](doc/FutureWork.md#carried-from-the-68-1315-rounds-2026-10-04)
retains the dynamic stop, operator renames and other deferred work. The
[6.8 plan](doc/plans/2026-09-27-item-6-8-one-attention.md) retains the reading
residue around forced `interpret` and digit boundaries. Historical operators
measurements and decisions are in the accepted receipt and its predecessors.

**Deferred from 6.5 (non-blocking):** retire the detached role view carried
beside the native occurrence anchor; learned-column pruning remains with
item 5 ([limits](doc/FutureWork.md)).

- **Query and ask.** Keep `query` as LTM lookup returning the best match;
   rename `what` to `ask` throughout the API, executor, grammars, tests and
   current documents, preserving the shared controller and history. The
   [thought-loop plan §§5–7](doc/plans/2026-10-07-thought-loop.md#5-the-query-rename-claude-for-codex-2026-10-07)
   owns the corrected names and the open-reference meaning of asking.
- **6.2. Thinking.** Next after query and ask, before item 6. Claude's
   specification follows from the [thought-loop audit and Alec's rulings](doc/plans/2026-10-07-thought-loop.md).
   Forgetting depends on thought conclusions and their references for
   deducibility; do not infer implementation details ahead of the specification.
- **6. Stored-idea generativity.** Forgetting's dropping of derivations depends
   on it. Item 1c's probe reports zero compound recovery
   ([measurements](doc/AccessibleMind.md#measured-limits)). It trains for 8
   small updates, so it mostly measures a split/stop policy that has not
   learned to split (it does within ~100). The deeper limit is the operator:
   with the correct split actions *forced*, the tied inverse of `lower` returns
   children about 65% from their codes at depth 1, they project to the wrong
   code, and 400 updates of the probe's training do not move that. Exit:
   recovery reported separately for forced actions (the operator's inverse)
   and free running (the policy, trained to convergence), by depth and chain
   length; clean-up decoding evaluated for lift / lower — bounded candidate
   search through the forward kernel (`_bounded_binary_reconstruction`) in
   place of the reference-free affine inverse. Until a recovery rate is
   measured, item 5 must not drop derivations. The `d4dc385` audit also records
   the existing MM_20M grammar free-derivation harness at 0/4 exact recovery
   after three epochs; its acceptance assertion now requires recovery rather
   than preserving that zero ([audit](doc/benchmarks/2026-09-21-item10/README.md#validation-and-limits)).
   *Compatibility:* this is the serial grammar's `lift`/`lower` with their
   tied chart inverses, unchanged by 11c and by item 10's drop. Recovery is
   measured from forward artifacts — the stored derivation and activations —
   never a saved input trace (Alec, 2026-09-25); and the 11c rule for
   descent applies: reverse sigma is a choice of case, reverse pi is
   attribution against the field, so clean-up decoding through the forward
   kernel is the intended form of that choice.
- **5.5. Where and when a sentence occurred; tense, aspect, the preposition
   and surface form** ([spec](doc/specs/2026-09-30-occurrence-tense-aspect.md)),
   after item 6 and needing item 6.5's verb columns; one check-in (Alec,
   2026-09-30). **Relative `.when` and address upsert moved to operators
   landing (accepted October 7; plan §45)**:
   source-relative `[i, i+1]`, stable document/content address keys,
   re-witnessing and fail-loud capacity. See its
   [receipt](doc/benchmarks/2026-10-07-operators-final-b/).
   The 10-03 amendment keeps `.when` in percepts and concepts; the absolute
   model clock remains in the timestamp column for recency. Remaining here:
   documents as situation codes in content; learned before/after; tense,
   aspect, prepositions and surface transformations. Every text
   configuration interleaves; tense and aspect are prepositions of the
   verb phrase with none written, the catalogue's compound selecting among
   the verb phrase's phases; the preposition is one operation with two
   attachments; markers are leaves, minted on recurrence, and `surface`
   is one operator with four suboperations, trained by reconstruction
   under parsimony. Removes the old tense, aspect and morphology code and
   its table of English forms. Exit: the ten mechanism tests of spec §8;
   learning measurements after 6.5, recorded and not tuned. Replace the
   legacy `get_stm_chain` recency across batch rows and test lookup by the
   row's occurrence address (two truths §21). Open for Alec: spec §10.
- **5. Forgetting** ([spec](doc/specs/2026-09-16-forgetting.md)), after item 7
   (needs `refs` and every S writing a row). Document-boundary pass from the
   high-water to the low-water mark deleting the lowest-value unprotected
   rows; value = `|trust|` + utility (1 − deducibility) + luminosity
   contribution; cascade, reference remap, dependents rebuilt, the human
   profile's age term; and **detail before rows** (§4a): wording, then
   subordinate rows, then the row, gradually, coarsening the referring row
   instead of cascading where the operand's point survives. Exit: the fourteen
   §7 tests, the accessible-mind spec's test 31 (retention by surprise), the
   §6 elements in schema/`model.xml`/Params.md, the §8 docs, and the reviewed 9b
   reconstruction measurements unchanged unless explicitly re-baselined.
   *Compatibility (corrected, Alec, 2026-09-28):* rows carry the evidence
   pair `(c⁺, c⁻)` (§1.1) **and, separately, a univalent trust scalar**.
   `|trust|` in the value is that scalar; it is **not** the collapse
   `|c⁺ − c⁻|`, which this line previously said and which was applying the
   evidence algebra to trust. The collapse `c⁺ − c⁻` and the *both* mass
   `min(c⁺, c⁻)` are the row's **evidence**, and the latter is the dissonance
   in the luminosity term, so a heterogeneous row is protected, not
   discarded. Open for Alec: forgetting
   of the concept inventory itself — order-0 definitions, alternatives and
   feature groups — is not in the spec; discovered rows are never recycled
   (item 11), so their retirement needs a rule here or in FutureWork.
   *Age (Alec, 2026-09-30):* with no clock across documents (item 5.5), the
   age term "can probably just be an increasing document index: more
   important within a document will be its salience"
   ([5.5 spec §9](doc/specs/2026-09-30-occurrence-tense-aspect.md#9-what-it-touches)).
- **4. Run harness and resume test.** *Once pulled forward for the
   conference (Alec, 2026-09-27; no freeze since 2026-10-03, so it is built
   in 6.8-1 and evaluated here):* the word-level predictor as the NanoChat gate's
   evaluator — the model's own expectation of the next word scored on the
   frozen item manifest (top-1, reciprocal rank, shuffled-prefix control),
   living in `eval_nanochat_grammar.py` rather than the training loop; it
   is the first level of item 6.8's expectation at every bracket. Also logged
   per bracket level once 6.8 lands: the both-rate and the
   categorical-discrimination index. One logger per interval: reconstruction
   loss; expectation discrepancy; LTM occupancy and forgetting passes;
   [categorical discrimination](bin/CategoricalDiscrimination.py) from the
   fixed four XOR and 68 FineWeb probes, reporting CP and within/between
   distances without pass thresholds. Reuse captured readings and budget their
   collection cost; the three-seed experiment is archived in FutureWork.
   Also log rows deleted per origin, value cut-off; luminosity of provisioned truths; the
   per-shared-operator gradient cosine and norm ratio — reconstruction against
   expectation, and against output where answers are supplied — with the
   operators in persistent opposition named; the held-out two-truths §7
   test-12 probe and a fixed reconstruction sample; and from 11–11c: order-0
   inventory rows used against `nVectors`, provisional-pool occupancy and
   exhaustion warnings, raises per order with their run counts and stalled
   patience, refinement requests, lexical references left unknown per order.
   And a resume test proving
   a mid-epoch checkpoint restores cursor, stream count, `refs` / surprise
   columns and forgetting counters with the next batch byte-identical —
   including the concept sidecar (feature groups, located brackets, sparse
   context, refinement buffers), the radix part groups and pending groups,
   the word-form index and the understanding's conceptual field. Exit:
   one command on a small config; the test in the suite.
- **3. Corpus at target size.** Raise `maxDocs`, exercise multi-shard if needed,
   measure sentence-list/address-table memory and loader time, confirm
   `resume_skip` with the run's stream count. Measure admitted PartSpace rows
   on the target corpus and raise the current 32,768-row reserve before the
   million-sentence run. The required capacity depends on admitted parts,
   not one row per sentence; exhaustion now stops training. Record the chosen
   physical `nVectors`, occupancy and dictionary/optimizer memory cost.
   Target about 200,000 English word forms plus one associated object per
   word, with further room for non-verbal concepts: roughly one million
   physical ConceptualSpace rows (about 4.3 GB at width 1032 in float32,
   without Adam moments on the rotation-owned dictionary). Reserve a few
   hundred thousand PartSpace rows and measure WholeSpace admission for the
   same vocabulary and non-verbal coverage. The current CS 65,536 / PS 32,768
   settings are small-run values, not production capacity estimates.
   Set `SymbolSpace.ltmCapacity` for the intended million-row run; its default
   1,024 also sizes the shared `.when` ladder. Record retained-row memory, the
   actual subjective-clock range needed by the run and the ladder's resulting
   periods, so a long training history does not silently reuse temporal phases.
   Exit: the load and capacity measurement recorded, and
   `maxDocs`/`shardDir`/stream count/PS-CS-WS `nVectors`/`ltmCapacity` in the run config.
- **2. Housekeeping.** Restore safe, faster bounded-test batching: the default
   256-case/16-file and 32-case/4-file workers exceeded the 8 GiB cap, and
   eight-case/one-file batches pass the same selection; do not raise caps or
   omit cases; reconsider [fixture reuse](doc/plans/2026-09-17-bounded-fixture-reuse.md)
   if useful ([evidence](doc/Testing.md#forward-owned-lexical-forms-september-19)).
   Delete the radix-backed non-word-major meronomy path (no legacy paths), and
   remove the stale `dispatch_per_row_reset` note in
   [the fold-ladder plan](doc/plans/2026-09-10-meronomy-fold-ladder.md#open-defects-found-on-the-way)
   (`taxonomy_parent_map` is now initialised).
   From the 11–11c refactor, verify legacy before deleting (no-legacy rule;
   reasoning methods need Alec's review): `overlap_where_tiling` and
   `where_tiling_for_pass`, `intent_boosts`, WholeSpace `part_chain`,
   `_automatic_synthesize_higher_order` against the pool's discovery path,
   and the META `insert_meta` / `taxonomy_parent` bindings once item 7's
   concept-level taxonomy owns them. Move the 2.9 MB
   `evaluation-source.tar.gz` out of the repository (the receipt keeps its
   hash) unless Alec wants it in-tree.
- **1. Compiler work.** *Added 2026-10-04:* the open read of 6.8-1 made the
   whole forward 2.6× slower (.141 s against .055 s on the XOR fixture;
   [profile](doc/benchmarks/2026-10-03-operators-attention/attention-profile-review/measurement.json));
   Alec: "the slowdown will have to be addressed." The decoder's two-word
   walk (6.8 plan §10.2) may pay part of it back. The throughput levers for the serial loop — batch
   across sentences, the known-word lookup concession under the serial
   flag, subsampled reconstruction, closing the host islands and moving
   the sparse stores to device CSR, single passes — are listed in
   [FutureWork](doc/FutureWork.md#throughput-levers-for-the-serial-loop-item-1-candidates)
   and belong here (Alec, 2026-09-25). Forward chooser split/lift once per slot; backward's
   launch-bound kernel count per brick; B24 brick +25% against pre-ladder.
   Exit: sentences/s and peak memory at the run's batch and brick size against
   the July baseline, re-taken on `99207a3` since the tower fold ladders are
   gone. New eager hot spots from 11–11c to measure and vectorise: the
   per-row loop in `cs_read_memberships`, the per-row composition in
   `_compose_order0`, the edge loops in `_reverse_field`, the per-reading
   descent in `refine_over_collected`, candidate growth in
   `_prepare_part_learning`, and the per-column loops in
   `promotion_observe`.
   The explicit slow reconstruction-cache probe fails its nonzero
   compose-gradient assertion on the current baseline; the cache claim remains
   unverified by that probe ([audit](doc/benchmarks/2026-09-25-item9-parity/README.md#validation)).
- **0. The full training session**: the long FineWeb run, only with items 11–2
   done and item 1 measured. Expectation on in `model.xml`; the
   `BasicModel.xml` flip follows the plan's §10 gates. Stop on rising
   expectation discrepancy, a reconstruction regression against the `99207a3`
   baseline, a forgetting pass deleting protected rows, a provisional-pool
   exhaustion warning, or order-0 inventory approaching `nVectors`.

Everything that is decided in direction but not on this path is in
[FutureWork](doc/FutureWork.md).

### Done (newest first)

- **Item 6.5 mechanism** (Alec accepted October 7; one landing from `f4a68404e`): definedness, native identity/change columns and global bind/mint; learning gates remain pending the million-sentence checkpoint ([acceptance](doc/benchmarks/2026-10-07-item6-5/acceptance.json)).
- `f4a68404e` **Operators update, final-b** (Alec accepted October 7): addressed rows, bipolar meanings, centroids, priming, catalogue and two-lane repair; all standing counts 10/10, R/E zero, green sweep ([acceptance](doc/benchmarks/2026-10-07-operators-final-b/acceptance.json)).
- `cce3a4f7b` Operators round 3a: identity by construction ([receipt](doc/benchmarks/2026-10-07-operators-round3a/README.md)).
- `3de37eef5` Operators round 2: accepted credit repair ([receipt](doc/benchmarks/2026-10-06-operators-round2e/README.md)).
- `73cd7b71b` Operators round 1 accepted and pushed, with WikiOracle bumped at `c9670b5` ([receipt](doc/benchmarks/2026-10-05-operators-update/README.md)).
- `42daf96f` Item 6.8 accepted: one attention, two spaces and one index; tag `6.8-landing`; class **7/10**, reconstruction **9/10**, joint **6/10**, MM_xor **10/10**, sum **10/10**, sweep **4,917 passed / 0 failed**, zero ownership conflicts ([receipt](doc/benchmarks/2026-10-03-operators-attention/README.md)); carried work: [FutureWork](doc/FutureWork.md#carried-from-the-68-1315-rounds-2026-10-04) and the [operators update plan](doc/plans/2026-10-05-operators-update.md).
- `bfe6a0d7` Item 6.9: the baseline of the composition mechanism (6.8 §12.1), one reconstruction/output decoder and one writer per weight; accepted class MSE .11475 and reconstruction 0/4 remain historical, with zero ownership conflicts ([closing receipt](doc/benchmarks/2026-10-03-item6-9-closing/README.md), [round history](doc/benchmarks/2026-10-03-item6-9-closing/todo-history.md)).
- `9810fc7` Item 7: two truths, indexed definitions, ended clause state and shared row-free predicates; accepted under review §25 after all three fixture ports and required checks ([landing receipt](doc/benchmarks/2026-09-30-item7-review-round5/landing/README.md)).
- `6906727` Item 7.5: one-operation exploit/explore derivations trained at each sentence closing, reduction pressure/deadlines and closing-gradient reporting; accepted with the unchanged depth-three campaign red ([receipt](doc/benchmarks/2026-09-27-item7-5-landing/README.md)).
- `8bc710a` Item 9b: shared fields, parallel-first context, association-first interpretation, fixed capacities and corrected occurrence/time objectives; Claude accepted the source-matched 4,948-case sweep ([receipt](doc/benchmarks/2026-09-26-item9b-occurrence-fix/README.md)).
- `8bc710a` Item 9 evaluation machinery: verified FineWeb exposure and maturity-gated wording/prediction/predictive-thought checks; empirical acceptance remains open above ([receipt](doc/benchmarks/2026-09-26-item9-followup/README.md)).

- `0e70001` Item 9 parity landing: sentence-owned reconstruction candidates, final saved roots, first-sight bank checks without scalar host reads, and the historical-answer warning ([receipt](doc/benchmarks/2026-09-25-item9-bank-sync/README.md)); expectation-learning gates remain open.

- `99207a3` Item 10 decided: dropped; normalized means deleted, native XOR and forward-evidence reconstruction pass unseeded, with mode exclusions asserted ([receipt](doc/benchmarks/2026-09-25-item10-forward/README.md), [evaluation](doc/benchmarks/2026-09-24-item10/README.md)).
- `99207a3` Item 11c residue: accepted three-update refine-before-raise patience, reset by strict improvement and cleared by pure/unknown reads ([receipt](doc/benchmarks/2026-09-25-item10-forward/README.md)).
- `d5f2e1a` Item 11c: native perception, located XOR, per-pole witnessing, nonzero focus and grammatical reference across orders ([receipt](doc/benchmarks/2026-09-24-item11c/README.md)).
- `15c9bde` Item 11b extent correction: containment at the subject extent, observed raw spans and the restored positive-only `MM_sparse_concept` smoke ([receipt](doc/benchmarks/2026-09-24-item11b-extent/README.md)).
- `aa235d2` Item 11b accepted-review corrections: recurrent parts, containment over canonical ids, and order-0 fields bound per turn with persistent concept identity ([receipt](doc/benchmarks/2026-09-24-item11b-corrections/README.md)).
- `db73581` Item 11b: native membership definitions, located fused parts and pervading wholes, with alternatives preserved through refinement; normalization residue closed ([review receipt](doc/benchmarks/2026-09-24-item11b-review/README.md), [design](doc/Architecture.md#item-11b-membership-read-and-extent-truth-corners-september-23)).

- `4b097dc` Item 11a learns WholeSpace properties over byte primitives, retains grounded extent evidence and passes six unseeded native XOR runs ([receipt](doc/Testing.md#item-11a-primitive-properties-and-grounded-extents-2026-09-23)).
- `e917d9c` Item 11 review corrections carry paired evidence through scoped dual folds, symbols, thought and checkpoints; calibrated background rejection and the full receipt pass, learned XOR remains null for item 11a ([receipt](doc/Testing.md#item-11-review-corrections-paired-evidence-september-23)).
- `eb45ae4` Item 11 lands the union default, optional conjunction pass and context-backed provisional rows; unseeded XOR learning remains null ([receipt](doc/Testing.md#item-11-concept-parts-and-provisional-rows-september-22)).
- `d4dc385` Item 10 (the first of that number) records the reconstruction baseline and packed/single parity gap, fixes MentalModel compaction overflow, and removes selected passing seeds (taken early from item 2; [receipt](doc/Testing.md#item-10-reconstruction-baseline-and-seed-audit-september-21)).
- `f8aa23c` Item 11 completes generation catalogue ownership, scoped output dispatch, checkpoint/Adam migration and supervised gradient boundaries ([receipt](doc/Testing.md#item-11-generation-ownership-september-21)).
- `afdcdfa` Item 12 completes the expectation review corrections: role-normalized retention surprise, indexed pairs, reading purity, bounded cache recovery and gradient norm ratios ([receipt](doc/Testing.md#item-12-review-corrections-september-21)).
- `7d7dc4f` Item 2 implements negative-image expectation and residual credit; measured joint/useful-query learning remains open under item 9 ([design](doc/ExpectationRetention.md), [receipt](doc/Testing.md#negative-image-expectation-september-21)).
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
- Expectation review (plan §11): implemented and validated on September 16 ([record](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md#116-implementation-and-validation-september-16)).
- Xcode licence accepted; trust term sign `|trust|` (Alec, September 18).
