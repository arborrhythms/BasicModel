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
- **1.4. Verbs in the field of concepts** ([the item 1.4
   plan](doc/plans/2026-11-23-item1-4-verbs-in-the-field.md); §9 to §9.9 hold Alec's decisions of
   2026-11-27 in his words, §15 the meta-concept, §16 the defects fixed). On `main` since
   2026-11-27 (`bd766c1596`), merged with item 1.5 and its compiler work (`4664cd0563`).
   - **Built.**
     - Evidence at order zero is positive: the first Pi reads "for" only, there is no Not at order
       zero, and what tells against a concept is an entry below zero on another row at the same
       place, on a link MTM writes at zero. "Both" is read one order up. Intensity levels are
       linear and fixed by default (`intensityLevelsFixed`). A word form is the set of its parts
       again (`WordIdentity.LIVE_TABLE_CODES` false).
     - The unit delay (`perceptDelay`, on): each basic percept one step back is a second input
       of the order-zero Sigma, on the numeric stem too; each concept one step back is a column
       of the concept store, and a kind over such columns is reached from what was active then.
       A hard reset clears it; it is saved with a checkpoint.
     - The look cycle (plan §9.5): the opening is look zero and openings are compared with
       openings; at each look the kept trace is decayed, the field rises met by expectation,
       descends masked by `attend`'s request, and what is left is what `attend` reads next.
     - `attend`'s request reads the whole STM stack, through a weight that starts at zero.
     - The links between kept ideas (`ideaTransitions`, off): in the one `LTM1`, marked, read as
       neither a part nor a kind, saved with a checkpoint.
     - `make test` writes the log `make push` reads; `test/failures_ok.txt` holds the written oks.
   - **To do, in this order** (all required before 1.4 is submitted; Alec, 2026-11-27).
     1. *One loop* (plan §9.9). "attend() is one of thoughts operations, as is output, and it can
        decide to read as many words as it likes. We should not have a reading loop independent
        of the thinking loop." Each `attend` is one look and reads one word, a shift; the reduce
        after it informs the next, through the STM stack. The loop of `Looks.run`,
        `attentionBudget` and `hardwireReading` go, and the walk ends with the thinking loop.
        Taught by a curriculum: "single-word reading until we get that reliable, then ... longer
        sentences in word order". The measure: one word said back (`attend`, then `output`)
        reliably; then as many `attend`s as words, each on the next word, before `output`.
        `hardwireReading` 0 (one look a sentence) is no target.
     2. *Two kinds of input and two of output*, text and intensity, "and parameters on the model
        to specify which to enable" (plan §9.8). Today an intensity is an output only on the
        numeric stem (XOR), and the text model refuses one as an answer.
     3. *Membership read off the matrices* (plan §9.9). No one in the symbol table: "the spares PI
        matrix that operates on the symbols at every turn should have its weight matrix adjusted
        so that its weights are not sparse for that input", with a learned weight. The Sigma the
        same: "we dont have a member list independent of the weights, we just add an edge to the
        sparse matrix (in this case the LDU)". The symbol table's member list and
        `order_zero_members` go. The spread then carries a symbol's members by graded weights
        (it carries each at one today); a binary salience, where one is needed, is made at the
        perceptual input.
     4. *Verbs as operations* (plan §9.6, 3). "verbs will generalize an operation, presumably into
        24 verb classes or so, and be capable of transforming their noun. So the word is placed on
        the stack, but the operation specified by the verb is performed", trained from the
        corpus. A verb's extension is "the map of running() to the extension of 'things that
        run'". Then the measures of plan §8, and what follows for the other parts of speech.
     5. *The meta-concept* (plan §15.1, §9.6, §9.8). "the meta concept is a result of identity,
        when a implies b and b implies a (where that implication is a link)." Both get a parent
        above the greater of their orders; "if one was a superset of the other, then it is lowered
        with respect to its element when they are both placed into the same class." It replaces
        the meta-concept `createMetaConcepts` makes on a word's first reading. Testpoint:
        `test/test_item14_meta_concept.py` (A passes; B and C fail until this is built; B also
        waits on 2, the intensity as an answer).
     6. *Behaviour from LTM by way of the basic level* (plan §9.3). "LTM can be used to train the
        basic level network, and ... recursive application (with some measure of previous state
        from output or thought) will yield the behavioral chains that we are hoping for." To
        build: episodic memory as percepts (plan §6; `episodicConcepts`, 4, percepts to an
        entry); a replay of kept ideas in the order of their links; a sentence's result as an
        input at the next ("we write to LTM and clear STM"). The Markov count waits: "the idea
        store has a graph of transitions which is not yet mined".
   - **Standing rules from this item.** State is in STM, MTM or LTM and nowhere else, except the
     unit delay's kept activations; STM is the state within a sentence and LTM between sentences
     (plan §9.4). A link is an entry of a sparse Pi or Sigma, made implicitly in MTM's pools or
     explicitly by a stated type relation (plan §9.8). M0 is the tier of ideas and M1 of
     relations, in each memory.
   - **Open for Alec.** The decay of the trace: "we should have a look at psych literature, and
     also decide if we want a two-parameter gamma memory" (it uses `primingDecay`, 0.9, today).
     Whether the same `attend` serves reading and thought's own attention (implied by 1; not
     said). The slow map's two rates; the Sigma links between neighbouring intensity levels;
     `OrderZero.read_word` still reads a signed match.
   - **Tests.** Failing and left so, each with Alec's written ok in `test/failures_ok.txt`:
     `test_32_a_known_word_is_not_read_inside_a_longer_known_word` (how a look uses a link below
     zero), `test_81_expectation_restores_the_omitted_native_word` and
     `test_the_answer_depends_on_the_spoken_request` (with the meta-concept). Failing with no ok
     yet: the meta-concept stages B and C. Marked `intermittent`: `test_partition_isolation`,
     `test_n7_the_corpus`, `test_84`, the MPS timing. `slow`: the compiled lesson step
     (unmeasured) and the gate over several looks. Last run, 36 files through `make test` on the
     merged tree: 392 passed, 12 skipped, 5 failed (those five). No complete `make test` has been
     run since the first sweep of 2026-11-27 (7,807 tests), so there is no log and `make push`
     will refuse: run `make test` first on the faster machine.
   - **Waiting on this item:** the XOR of intensities, the name formed by MTM, the order of two
     words through grammar training ([item 1.6
     plan](doc/plans/2026-11-18-item1-6-felt-math-world.md), §44).
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
