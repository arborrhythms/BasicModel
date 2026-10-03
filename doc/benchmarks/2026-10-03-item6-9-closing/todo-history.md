# Item 6.9: todo history at acceptance

This is the complete former todo entry, preserved at the October 3 landing.
The closing receipt and plan §26 supersede the historical open statuses.

- **6.9. XOR_grammar: the baseline of grammatical learning — CLOSED**
   ([plan](../../../doc/plans/2026-09-29-item-6-9-xor-grammar.md); Alec, 2026-09-29:
   "let's make item 6.9 an effort to fix xor-grammar ... so that we can
   prevent any grammatical regression"). Item 7 is accepted; 6.9 precedes
   6.8. The first operators update is now part of 6.9 under §§21–22.
   **October 3, §25 closing baseline:** one decoder is shared by reconstruction
   and output; the generate chooser infers operations without a journal.
   Reconstruction owns it and the antipode term; output owns its reader or
   conditioner. Echoic priming includes the sentence's words, after the seen
   write; the §24.4 pre-write proposal is withdrawn. The
   [closing receipt](../../../doc/benchmarks/2026-10-03-item6-9-closing/README.md)
   records one shared XOR_grammar training: all **4/4 class labels** correct,
   but **MSE .1147481948**, so the unchanged **class bar is red**; complete
   read-backs **0/4**, so the unchanged **reconstruction bar is red**. The
   decoder chose STOP immediately in every training row/trial, emitting one
   word. Both gates used the same model. Ownership conflicts **0**; decoder,
   code and anchor gradients/displacements are live. **29 focused cases pass**;
   collection has **5,019 cases**. No repeated gates, attribution, sweep,
   MM_grammar or native run this round. The prior §22 measurement remains
   **9/10 class and 5/10 reconstruction**; it was not repeated.
   **MM_xor::test_convergence stays red by §17**: meronomy removed the former
   cross-word percept lookup. Its bar/test stay unchanged; word-level XOR is
   6.8's acceptance test. **6.9 closes as a baseline, not an all-green claim.**
   The standing XOR rule is no regression against this record, including its
   explicit red gates and the preserved earlier measurements. The catalog
   (§§20.3, 24.5, 25.2, 25.4) assigns conceptual identities/remaining decoder
   learning to the operators update, open reading and MM_xor to 6.8, and
   operand order to surface markers. Next: **operators update → 6.8**
   (no conference freeze; Alec, 2026-10-03). Nothing committed; stop for review.
   **Historical rounds below retain their status at the time; §25 supersedes
   their open/deferred status.**
   **October 3, §23 corrections:** the acceptance campaign is deferred to
   the next check-in. The completed [§22 receipt](../../../doc/benchmarks/2026-10-03-item6-9-free-readback/README.md)
   remains the measurement baseline. This round repairs the log-space
   byte read-back, operand-unary replay, sum generation, truth union and
   clause-journal writes, and ports the reviewed observers/fixtures.
   Its [correction receipt](../../../doc/benchmarks/2026-10-03-item6-9-corrections/README.md)
   contains focused regressions and one shared XOR training with both bars
   and the gradient/displacement audit. No ten-run campaign, attribution,
   full sweep, MM_grammar measurement or native run is authorized this round.
   Nothing is committed; 6.9 remains open for review.
   **October 3, shared-gate amendment:** both XOR_grammar bars now read one
   module-scoped trained model, also kept together by bounded and weekly
   dispatch. The bars are unchanged. Future campaigns use ten trainings and
   report class, reconstruction and joint success per run, reusing run 1 for
   both table rows. Alec waived retesting after this consolidation; the
   already-started repeat was stopped and its partial outcomes retained in
   the [amendment receipt](../../../doc/benchmarks/2026-10-03-item6-9-shared-gates/README.md).
   The following completed receipt remains the measurement baseline.
   **October 3, §§21–22 measured; stopped for review:** free read-back alone reconstructs the
   input from its root, recorded rules and primed bank; inverse witnesses are removed.
   Conjunction uses product binding, disjunction the mean, and min/max remain
   separate catalogue operators. Only strictly lower reconstruction selects
   explore; the reader trains on kept rows. Reconstruction uses momentum SGD,
   with displacement and derivation-stability audits. The
   [current receipt](../../../doc/benchmarks/2026-10-03-item6-9-free-readback/README.md)
   records class **9/10**, reconstruction **5/10**, sum control **10/10** and
   the named XOR table **33/34**, with only the §17 MM_xor proof red.
   Class errors: nine at 0, one between; MM_grammar: nine at 0, one at ¼,
   median ending training MSE **4.10104e-7**. Attribution is not triggered.
   Both audits record zero ownership conflicts, but the audited XOR codes
   and chooser anchors have zero displacement at float32 precision. The
   native batch-28 stage-1 test passes at **21.91 GiB** within its unchanged
   24 GiB slow-only ceiling. The ten-worker source-matched sweep completes
   **5,000 cases in 19.45 minutes**: 4,706 pass, 8 fail, 285 skip and one
   expected failure. Moved cases: **159 pass, 6 fail, no stops**. All eight
   prior sweep failures pass; three prior moved cases remain red. Current
   production regressions affect sum generation, truth penalties and compiled
   byte-score parity; observer ports and learning failures remain. The journal
   also retains numerical clause-closing frames, so its rule-only scope is
   incomplete despite removal of inverse witnesses. **6.9 stays open;
   nothing is accepted or committed by this receipt.**
   **October 2 night, §20 measured; not accepted (§21):** reconstruction alone trains the
   unconstrained concept dictionary. VQ EMA and contextual rotation are off;
   witnessed and free read-back are separate relative reconstruction terms.
   XOR_exact's evidence coefficients are answer-owned. The
   [reconstruction receipt](../../../doc/benchmarks/2026-10-02-item6-9-reconstruction/README.md)
   contains the failing probes, complete old/new port bodies and closing results:
   class **0/10**, reconstruction **2/10**, sum control **10/10**; named XOR
   table **31/34**, with only the two first grammar gates and the deferred
   MM_xor proof red. Both XOR_exact checks and the single MM_20M exact round
   trip pass. MM_grammar's median ending training MSE is **2.20011e-7**,
   with one known .2500589 stop. Both ownership audits have zero conflicts;
   the native production batch passes at **23.63 GiB**, within its existing
   native-only 24 GiB ceiling. The conditional attribution is recorded.
   The ten-worker source-matched sweep attempted all **4,980 cases in 22.23
   minutes**: 4,686 pass, 8 fail, 285 skip and one non-strict XPASS (raw reports
   contain three additional passing subtests). All 23 prior full failures pass.
   The 165 newly dispatched moved cases give **157 passes, 5 failures and
   3 stops**; seven further cases reuse their named-table results. Remaining
   fixture ports, a reconstruction capture-limit failure and two learning
   failures are unwaived; nothing is committed and 6.9 is not closed.
   Deferred mechanisms are in
   [plan §20.3](../../../doc/plans/2026-09-29-item-6-9-xor-grammar.md#203-catalog-set-aside-now-to-return-once-reconstruction-and-xor-hold).
   **Previous ownership round, §§15–17 (not accepted, §18):** reconstruction, expectation and the
   supplied answer now have separate parameter owners. Both trial consumers
   read one understanding record, including row-local diffused priming. The
   uncut answer and its gradient projection are retired. The configuration
   cleanup and contract ports precede the once-only closing measurements in
   [the ownership receipt](../../../doc/benchmarks/2026-10-02-item6-9-ownership/README.md).
   **MM_xor stays red by §17 decision:** its old convergence used promoted
   cross-word chunks (`hello wo`, `hello th`, `loving w`, `loving t`) as a
   sentence lookup. Meronomy removes that shortcut. Its convergence test and
   bar are unchanged, without a marker; word-level XOR is an item 6.8 §8
   acceptance test. Saved geometry and promotion probes remain in the receipt.
   **Previous ownership closing measurements:** class 1/10, reconstruction 1/10,
   sum control 10/10; the named XOR table is 31/34. MM_xor happened to pass
   this single unseeded attempt without repair; its §17 deferral is unchanged.
   `XOR_exact` is a new, unwaived ownership regression (all answers zero,
   MSE .5): its named concept output has no trainable answer reader, and the
   answer no longer updates the reconstruction-owned concept coefficients.
   The source-matched sweep attempted all 4,976 cases in 19.37 minutes:
   4,662 passed, 23 failed, 290 skipped and one non-strict XPASS; no process
   stops. The 170 extra slow/moved cases yielded 110 passes, 48 failures and
   12 stopped attempts (four memory stops and eight aborted peers). The receipt
   preserves the conditional attribution, ownership audits, every failure and
   the unfinished fixture/contract ports. Nothing is committed; 6.9 stays open
   for review, with the new XOR regression and remaining test work unresolved.
   **Previous candidate, §14 stopped for review (2026-10-02):** sentence journals
   use the existing static word buckets; the default and explicit old reading
   modes have moved to meronomy, so XOR_grammar now reconstructs through its
   understanding. The detached student runtime and legacy reporting are
   retired, with the current BasicModel checkpoint's one-way migration kept.
   The [§14 receipt](../../../doc/benchmarks/2026-10-01-stage1-and-suite-trim/review14/README.md)
   retains the failing probes, complete ports, configuration first batches
   (32 complete, 18 exceptions, one unchanged 8 GiB guard stop), and triage of
   the completed first weekly run. Closing measurements are complete:
   class 8/10, reconstruction 3/10, sum control 10/10, the table 30/34 and
   MM_grammar median ending MSE 4.36e-11. The source-matched sweep attempted
   all 5,060 cases in 214.66 minutes: 4,672 passed, 314 skipped, 71 reported
   failures and three stopped without a result, against 5,219 cases / 122
   minutes. The receipt discloses ten case IDs inadvertently repeated by
   its continuation, retains first outcomes and includes the interruption
   in wall time; the corrected continuation adds no further duplicates.
   Nothing is committed; item 6.9 is still open and
   the venv rebuild remains held.
   Alec accepts the observed 8/10 class passes as the intended demonstration
   of learning; consistency is measured, not an additional acceptance bar.
   Reconstruction's 3/10 result and possible gradient conflict remain for
   review, with all assertions and thresholds unchanged.
   The structural ownership direction in
   [FutureWork](../../../doc/FutureWork.md#separate-gradient-ownership) was subsequently
   approved and taken up in §15; its older optimizer proposal remains deferred.
   **Previous §13 candidate, submitted for review:** the repairs and cost function
   are implemented, with scoped tied reconstruction, uninformed relative errors,
   penalties apart, and reconstruction precedence for supplied-answer gradients
   and trial selection. The [closing receipt](../../../doc/benchmarks/2026-10-01-stage1-and-suite-trim/README.md)
   records class 9/10, reconstruction 0/10, strict half-control 0/10, the XOR
   table 33/34 with one passing exact round trip, and MM_grammar median .0625.
   The unchanged XOR_grammar fixture remains in the old-mode exemption.
   The source-matched sweep completed 5,090 cases in 110.70 minutes: 4,760
   passed, eleven failed, 318 skipped and one expected failure. The saved
   failures include unported fixtures/contracts, an extra graph capture and a
   zero compose result. The earlier-source weekly run continues in the
   background. Nothing is committed; this item is not closed, and the venv
   rebuild and old-mode migration remain held.
   **Earlier step-5 status, 2026-10-01**
   ([repair and measurement receipt](../../../doc/benchmarks/2026-10-01-item6-9/README.md)):
   all six full-sweep failures are repaired without changing their assertions;
   the twelve affected checks pass. The candidate-only XOR table is 33/35,
   with both grammar gates still red and the one exact round trip passing.
   Ten unseeded runs of each isolated probe meet the settled bar in **0/10**
   for detached answer comparison, **0/10** with four explore trials, and
   **9/10** with the answer gradient reaching the codes and chooser. These
   are measurements only; the candidate retains the answer's gradient cut.
   Steps 6–8 and part 4 have not started. Nothing is committed.
   **Reviewed and decided (Claude and Alec, 2026-10-01):** the six repairs
   are right. The configuration is not the cause: trained as the gate
   trains, MM_grammar meets the bar in 0/5 runs, as do XOR_grammar variants
   with MM_grammar's dimensions and order; MM_grammar's ten-run table
   succeeds only because its harness never runs the sentence trials
   ([plan §3.13](../../../doc/plans/2026-09-29-item-6-9-xor-grammar.md#313-after-step-5-what-holds-xor-back-measured-2026-09-30-and-10-01)).
   Alec: "Let's just implement the answer training the whole path, as you
   said, and have codex continue with all of 6.9." For a sentence with a
   supplied answer, the answer's error trains the reading map, the chooser
   and the object codes (plan §4 step 5a; it amends the 2026-09-20 cut and
   the 2026-09-21 codes rule for such sentences). **Hand-off:**
   [plan §9](../../../doc/plans/2026-09-29-item-6-9-xor-grammar.md#9-hand-off-to-codex-continued-2026-10-01):
   step 5a, then steps 6 to 9, then part 4 of the October 1 hand-off.
   **The cost function is part of this item** (Alec, 2026-10-01: "No
   separate item, it would take too long. Add it to the current work, or
   even better, the next work"; the former item 6.85,
   [spec](../../../doc/specs/2026-10-01-objective-conflicts.md)). Alec decided:
   "A mind has to see what is there before it can make good decisions
   about it"; "keep the trial if it's better overall (subject to 1)";
   every configuration reconstructs its input through the understanding,
   with the answer added when supplied and expectation always learning
   ("We should be reconstructing input in all cases"); the terms of a
   cost have "a similar norm" before weighting, taken from the
   literature, described in the docs and implemented in the `Error`
   class of `Layers.py`. Design: spec §8 and §10 (relative errors against
   a trivial predictor of each term's own targets; reconstruction's
   precedence in the gradient and in the choice between trials). The
   current gradient is described in
   [GradientFlow](../../../doc/GradientFlow.md#the-training-step-october-1).
   **The October 1 sequence (plan §12):** repair item 7's operation-record
   memory (the production benchmark does not fit at its own batch); the
   production stage-1 measurement as a `RUN_SLOW` test; the suite trim,
   items 1 to 10; then this item's cost function; then both gates and the
   control under it.
   The [September 30 receipt](../../../doc/benchmarks/2026-09-30-item6-9/README.md)
   remains historical evidence:
   0/10 fresh 400-epoch class runs meet the settled bar; the current
   reconstruction gate fails 10/10. Steps 6–8 have not been taken. The
   source-matched full sweep completes all 5,161 cases: 4,832 passed,
   six failed, 322 skipped, one expected failure, with no resource stops.
   The six failures are closed by the October 1 targeted checks; no new full
   sweep is claimed. The September 30 step-5 exact
   table is 11/15 on the candidate and 12/15 on fresh HEAD; the accepted
   historical 12/15 record remains intact. The item is not complete.
   Found: a
   sentence's two trials are compared across an optimizer step, so the
   explore trial is kept in most rows whatever its derivation (a control
   that repeats the greedy derivation "wins" 88–93%), and the answer is
   trained mostly on understandings that evaluation never sees (plan
   §3.12). Decided (Alec, 2026-09-30): "equal comparison", done efficiently,
   with a snapshot where the two trials branch left to future work. Decided
   (Alec, 2026-09-29): the class gate asks for all four answers right with
   an error below .05, "but let's make sure it's theoretically possible"
   (it is, when the grammar composes object concepts, and not for symbols
   composed the same way for every sentence: plan §3.11); "Answers begin
   with the understanding left in the 1 or 3 slot representation"; "The
   grammar operations are conducted over the object concepts, not the word
   concepts"; reconstruction is "a lower bar", its only admitted errors
   transpositions, since "the grammar is symmetric"; the rules stay
   `conjunction` and `disjunction`, "since meaning is not significant".
   Measured (plan §3): the class
   gate reads a placeholder zero for every `embedding` configuration and
   cannot pass; read correctly, its bar is met by chance nine times in
   sixteen; the answer reads one concept per word position, not the root
   the grammar composed; the grammar is given each word as its own event,
   with no object and two numbers that differ between words; the
   reconstruction gate never calls the grammar's reverse. The operator set
   is not the obstacle: `intersection` and `union` change nothing, and in
   isolation any operator that is not a sum learns XOR from full codes.
   With the answer reading the root and each word given as its object's
   code (two probes), the four roots are exactly separable; the remaining
   gap is the answer map's convergence (plan §3.7). Also here: the slow
   MM_20M_xor exact round trip fails 1 of 15 unseeded runs at HEAD (3 of 15
   on the item 7 candidate), decoding half the inputs; an XOR proof that is
   not reliable, whose cause is to be found (two truths §19 AA). Exit: both gates pass
   without a seed at the settled bar, a `sum`-only negative control fails,
   and no other XOR proof, nor MM_grammar's ten-run table, is worse.
