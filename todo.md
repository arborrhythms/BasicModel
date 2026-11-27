# TODO

> **Keeping this file useful (Alec, 2026-09-19).** This file lists what is
> left; it is not a log. A landed item is one line with its commit hash under
> "Done"; its evidence and narrative live in the document the line links to
> ([Testing](doc/Testing.md) for receipts, the topic doc for the design). An
> item leaves the list when its exit criterion is met, not when work on it
> starts. When a spec or plan leaves residue, that residue is a line here or
> it is lost. Reconcile this file at every landing, before the commit.

## Countdown to the full training session

Items count **down**: the next item to do has the highest number and **item 0 is the full training
session**. An item may be taken early when it does not depend on a higher-numbered one. "Done" lines
keep the numbers their items had when they landed. Items 1.6 and above are integrated (Alec,
2026-11-26); the text this file held until then is in
[DevelopmentHistory](doc/DevelopmentHistory.md#work-recorded-in-todomd-until-2026-11-26). What is
decided in direction but not on this path is in [FutureWork](doc/FutureWork.md).

Standing rules:

- **Who builds, who reviews.** Codex builds; Claude writes the specs and reviews each landing (Alec,
  2026-09-21). Exception (Alec, 2026-11-09): Codex builds item 0.5 while Claude builds item 1. Items
  1.4 and 1.5 are code-reviewed before they are combined (Alec, 2026-11-26).
- **Tests until the long run (Alec, 2026-11-18).** "We need to run only targetted test from now on
  during the point build rounds (e.g. 1.x), and the complete sweep only at the item numbers (e.g.
  1.0). No Run_Slow until after all architecture is baked"; corrected the same day: each item gets
  one complete default sweep, "but not before its potential push round"
  ([Testing](doc/Testing.md#which-tier-runs-when-until-the-long-run-alec-2026-11-18)).
- **Bounded test runs.** The default sweep runs at ten workers under the 28 GiB ceiling; six are
  allowed, recorded, if ten cross it (Alec, 2026-11-05). Do not raise caps or omit cases
  ([Testing](doc/Testing.md)).
- **Publish rule (Alec, 2026-10-11).** Repair, one regression per defect, scoped measurement,
  candidate commit on a branch, Claude's code review, one full sweep at landing, BasicModel push,
  WikiOracle bump and push, with the co-author trailer. Every linked file exists before the sweep
  (2026-11-07).
- **Commits and pushes (Alec, 2026-11-18).** Claude may commit its work locally; pushing stays rare
  and on Alec's word; the superproject's pointer is his. `nanochat` carries no changes of ours
  (Alec, 2026-11-03).
- **Claims.** No test pins a random seed in order to pass (Alec, 2026-09-21). Mechanism probes do
  not satisfy learning gates, and a null result is recorded as null. The grammatical operations are
  still in development (Alec, 2026-09-26): a poor result on wording, parse depth or free generation
  is recorded, not tuned away.
- **Code.** No legacy paths; verify legacy before deleting it, and do not remove unused reasoning
  methods without Alec's review. Constants set by hand are powers of two (Alec, 2026-11-04).

- **1.5. Computation by wiring; then lookups as one matmul, whole/part folds as scatters, the
   percept pass per sentence** (Alec, 2026-10-30; and 2026-11-18: "Please move all 1.6 work into
   1.5"; [plan](doc/plans/2026-11-14-item1-5.md)). Before item 1, which compiles what this builds.
   Being finished by another agent; then code-reviewed and combined with item 1.4.
   - **Current priority (Alec, 2026-11-26).** Pause the other 1.5 optimizations. Make exploration
     faster in ordinary training and testing: one percept read per sentence; the opening bank and
     greedy prefix shared across a sentence's alternatives; each path's parse and provisional memory
     private; one clock per sentence; only the winner committed.
   - First experiment, status not confirmed: four sentences with three explorations each, the twelve
     lanes run together against the same departures run serially; revert if it needs much
     bookkeeping or gives no speedup
     ([plan](doc/plans/2026-11-14-item1-5.md#the-explore-paths-as-rows-of-the-batch-alec-2026-11-19)).
   - Open since 2026-11-24: the input ladder reads learned span endpoints on the host, which
     prevents whole-sentence capture ([handoff](doc/plans/2026-11-24-item1-5-handoff.md); [remaining
     gates](doc/plans/2026-11-25-item1-5-claude-completion.md)).
   - Paused, status not confirmed: part 1, wiring steps B, D and E
     ([document](doc/plans/2026-11-17-item1-6-zero-data-movement.md)); the five steps from the
     profile, with item 1.8's list of capture breaks
     ([plan](doc/plans/2026-11-14-item1-5.md#where-a-sentences-time-goes-and-five-steps-for-it-added-2026-11-19));
     the percept lookup on the device by the set of a form's parts
     ([plan](doc/plans/2026-11-14-item1-5.md#the-radix-trie-as-a-tensor-and-the-pi-in-its-place-alecs-question-2026-11-18));
     item 1.75's part looked at again now that item 1.6's second modality is in; a fragment's words
     read at one time, measured ([sigma-pi](doc/plans/2026-11-14-sigma-pi-layers.md) §21, rule 58).
   - Paused, status not confirmed: part 2, lookups, folds and the percept pass. Gates: `fullgraph`
     compile of every rewritten brick, equal results against the current path, one before/after
     timing on lesson-13 and the math chain with the July baseline retaken, no resize
     ([FutureWork](doc/FutureWork.md#throughput-levers-for-the-serial-loop)).
   - Carried 2026-11-03 (items 2.8, 2.7), status not confirmed: the atom property fill in
     `refresh_property_fills` (`bin/WordIdentity.py`), with a certificate.
   - Carried from item 2.0, status not confirmed: the basic-percept feature layout at order 0, and
     the port of
     `test/test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`.
- **1.4. Verbs in the field of concepts** (after items 1.6 and 1.5; [the item 1.4
   plan](doc/plans/2026-11-23-item1-4-verbs-in-the-field.md), §10 to §14 for what follows).
   - Built on local branch `item1-4`, uncommitted: evidence at order zero is positive (the first Pi
     reads "for" only; no Not at order zero); MTM writes a link at zero from a held row's concept to
     its holder's row and the Sigma learns it; refinement's "both" is read one order up; intensity
     levels are linear and fixed by default (`architecture.intensityLevelsFixed`), with a slow
     linear self-organizing map when it is false; the first Pi reads a word form as the set (OR) of
     its parts again (`WordIdentity.LIVE_TABLE_CODES` false), which restored discrimination that
     item 1.5's graded table codes had lost.
   - Two production bugs from item 1.5 found and fixed on this branch: native references were not
     bound at sentence close (`bin/ClauseJournal.py`); the word pool could nominate a word onto a
     row already taken (`bin/Models.py`).
   - Not yet built: the delayed perceptual field and verbs as changes (plan §5.8); episodic memory
     as percepts (plan §6).
   - To do at 1.4 integration (Alec, 2026-11-26); these three tests are left failing on purpose,
     unweakened: (a)
     `test/test_item17_looks.py::test_32_a_known_word_is_not_read_inside_a_longer_known_word`: with
     the against lane gone, decide how a look uses a link below zero (a look for `the` still returns
     `there`); (b)
     `test/test_item18_expect.py::test_81_expectation_restores_the_omitted_native_word`: evaluate
     together with the Meta-Concept learning; (c)
     `test/test_item17_word_said_back.py::test_the_answer_depends_on_the_spoken_request`: the same.
   - Open for Alec: the map's two rates; whether to keep the Sigma links between neighbouring
     intensity levels; `OrderZero.read_word` still reads a signed match;
     `test/test_item18_intuit.py::test_84_intuit_reads_once_at_observation_and_no_relation_row`
     failed once and could not be reproduced;
     `test/test_compiled_lesson_step.py::test_one_lesson_sentence_trains_with_the_compiled_output_walk`
     needs more than 9.6 GiB and is left out of sweeps.
   - Waiting on this item: the XOR of intensities, the name formed by MTM, the order of two words
     through grammar training ([item 1.6 plan](doc/plans/2026-11-18-item1-6-felt-math-world.md),
     §44).
   - Documentation (Alec, 2026-11-24), status not confirmed: what positive concepts mean at each
     order, in [Logic](doc/Logic.md), [Orders](doc/Orders.md) and [Philosophy](doc/Philosophy.md).
   - The complete default sweep ran on 2026-11-27 (plan §16.4): 7,807 tests, 7 failing; three
     were new and are settled (two certificates restated for the set codes, one GPU hang), which
     leaves the three tests above and the pinned-banks bound. Before the branch is pushed: code
     review of 1.4 and of 1.5, then the two combined.
- **1. Compiler work.** Built by Claude, side by side with item 0.5 on Codex (Alec, 2026-11-09).
   Detail of each line: [the block as it
   stood](doc/DevelopmentHistory.md#live-items-15-to-0-as-they-stood-in-todomd).
   - Exit: sentences/s and peak memory at the run's batch and brick size against the July baseline,
     retaken on `99207a3`, with a thought episode's step and the per-sentence time at the math
     corpus.
   - Compiled and eager runs of the generated math corpus (item 0.5's rows) under `RUN_SLOW` agree
     in actions, probabilities, costs and updates.
   - Four runtime failure groups from the item 3.0 landing (2026-10-26):
     `test/test_objective_conflicts_slow.py` at the 28 GiB guard; `test/test_testpoint.py` (GPU
     placement at 120 s, three server queries at 30 s); three tests at their 45-minute caps.
   - Slowdowns to explain: the open read of 6.8-1, 2.6× (Alec, 2026-10-04: "the slowdown will have
     to be addressed"); the rising per-presentation runtime in the 6.1 curriculum. The slow
     reconstruction-cache probe fails its compose-gradient assertion; the cache claim is unverified.
   - The throughput levers ([FutureWork](doc/FutureWork.md#throughput-levers-for-the-serial-loop))
     and the eager hot spots: `cs_read_memberships`, `_compose_order0`, `_reverse_field`,
     `refine_over_collected`, `_prepare_part_learning`, `FamiliarTier.observe_field`; the hook on
     every saved tensor.
   - Generate walk (Alec, 2026-11-11): `Generative.MenuStore` exists only as uncommitted work on the
     local branch `item1-generate-walk` of the first machine; next, batching inside a menu and the
     walk under `fullgraph`.
   - Carried from item 2.0 (2026-11-11): a thought store's growth happens outside the compiled
     regions, or its recompilation is accepted. Carried 2026-11-03 (items 3.5, 2.7): time the
     nested-admission path, the stationary-walk stop and the eager decoder's selected-inverse path;
     exercise the three `torch._assert_async` sites once on MPS.
- **Item 0 binding evidence:** `test_live_pending_premise_closes_in_both_orders` keeps its strict
  arrival assertions, with `open_names=('x',)` and no named closure forcing; its
  `pending_premise_learning` artifact is unregistered. Measure live closure at the trained
  checkpoint ([as it stood](doc/DevelopmentHistory.md#live-items-15-to-0-as-they-stood-in-todomd)).
- **0.5. Test development pass** (Alec, 2026-11-09; built by Codex, side by side with item 1, which
   Claude builds; [spec](doc/plans/2026-11-09-item0-5-test-development.md)). On the day item 0
   starts, every measurement it reads has a harness that has run once on a small model, and the
   training corpus is frozen under one manifest.
   - Scope (spec §5 to §10): the test and harness work from items 2, 1 and 0; transformation games;
     the math corpus as frozen graded rows; the staged training corpus; lessons and registered
     artifacts; the harnesses item 0 reads. What is built is recorded in spec §20.
   - Exit (spec §12) still requires, as of 2026-11-13: byte-exact checkpoint/update restoration; the
     missing historical whole-pullback capture; runtime control definitions and lesson/study bars;
     the joint default/weekly exit with item 1.
   - Game learning is stopped at its two signal nulls. As games supply evidence, revisit
     restrictions such as the one on several unary operations at one location (Alec, 2026-11-09).
   - Moved from item 2 (2026-11-09), status not confirmed: bounded-test batching and fixture reuse,
     timed before any speedup is claimed; D5 (the two decomposition snapshots regenerated from their
     seeds, then both arms at budgets 512 and 768); the deferred slow marks once under `RUN_SLOW`;
     `make test_all` green once on one hash; the default sweep green at the ten-worker ceiling. Also
     the per-interval logger; the resume certificate (`test/test_corpus_checkpoint.py` on the real
     `runBatch`); confirm the 7.5 shared-operator report assertion is gone.
- **0. The full training session**: the long FineWeb run, only with items 11–1.5 done and item 1
   measured. Every record is kept in [the block as it
   stood](doc/DevelopmentHistory.md#live-items-15-to-0-as-they-stood-in-todomd).
   - Before the run: expectation on (`sentenceExpectation`); reconstruction re-baselined on
     `f4a68404e`; reconstruction is an audit only where the audit reads zero at the run's width
     (confirmed by Alec).
   - Stop on rising expectation discrepancy, a reconstruction regression, a forgetting pass deleting
     protected rows, a provisional-pool exhaustion warning, or order-0 inventory approaching
     `nVectors`.
   - It opens with three analyzable stages (words; simple sentences and synthetics; math), a model
     saved after each; the million-plus sentence corpus starts after Alec has read the operator
     table (Alec, 2026-11-09).
   - Declared before the run (Alec, 2026-11-03): item 8's causal study (256 held-out tasks, four
     arms, seeds 0, 1 and 2), 6.5's six learning gates and the held-out cost set; and the measured
     constants taken to powers of two (`batchSize` 28, `maxDocs` 11,207, concept width 1256,
     `conceptLayers` 5, the thresholds at 3).
   - Before the run, from item 2.7: the capacity search of `test/test_corpus_large_slow.py` with the
     R == 0 audit at the chosen width; the sentence-132 capture, repaired at the producer.
   - At the trained checkpoint: register it under `fineweb` at 1,000,000 completed sentences and run
     the five `fineweb` certificates (item 9); the MM_math_chain measurement under its four
     corrections (unimplemented); the NanoChat next-word manifest (item 4); the occurrence
     measurements (item 5.5); the modal-corpus retry from operator choice (item 3.5).
   - Learning gates still open, each with its strict assertion, seed and original budget: item 3.0's
     companion residual (6/8) and sweep remainders; 6.1's; item 4's; item 3.5's iteration 1 and 2
     carries; item 6's (stage 4e first); item 2.7's (the unforced 1-to-2 raise, mass, count and
     kind-verb, order-2 kinds, silence). Slow budgets (Alec, 2026-10-26): `RUN_SLOW` repeated work
     defaults to half; learning claims use `RUN_SLOW_FRACTION=1`.

### Done

Items 1.6 and above are done and integrated (Alec, 2026-11-26). Their entries, and the full text of
what this file held before it was trimmed, are in
[DevelopmentHistory](doc/DevelopmentHistory.md#work-recorded-in-todomd-until-2026-11-26).
