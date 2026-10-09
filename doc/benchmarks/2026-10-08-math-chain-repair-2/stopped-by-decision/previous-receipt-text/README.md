# MM_math_chain repair pass 2 — October 8, 2026

Declared under thinking spec §14.8: ten fresh starts paired across the three
conditions, seventeen epochs each, attentionBudget 32, ltmCapacity 131,072,
graded by chain length alone, sentence departure judged by R + A.
`protocol.json` records the declaration and the dry-run numbers. Source and
helpers are frozen. The full sweep is green (5,351 passed, 284 skipped, one
XPASS; 5,636 unique cases completed), and the thinking gate passed 57/57. The
standing thirty are complete: sum 10/10, XOR class and reconstruction 10/10,
raw MM 8/10. Both MM misses reproduce bit-for-bit on the 6.2 landing for all
200 steps; the §20 unchanged-path evidence and original misses are retained.
The thirty declared math trainings are now running.

The development certificate is accepted on the corrected criterion: mean work
per declarative episode fell from 30.333 to 18.942 across the two halves of the
1,449-sentence epoch. Episode share is reported, not gated. The original failed
share-based result stays intact; `development-acceptance-14-8.json` records the
review acceptance and exact computation. No learned-chain result is claimed.
Both earlier receipts remain immutable. Stop for Claude review before any commit.

The notes below retain the development history, including superseded findings
and the certificate questions answered in §14.1. Current §14 details follow
under “Section 14 continuation”; `development-status.json` records run status.

## Development history before §14

The repair is still in development. The ordinary initial-opening, answer-fill,
bound-declarative and pending-row certificates are not established. A clarification is pending about whether the answer-fill and pending-row
mechanism certificates may control legal grammar choices while retaining the real
training driver, frozen observer and a live explore suffix. The measurement will
remain unforced. All seven files in `frozen-contracts.json` still match.

The isolated, unforced initialization probe opened **0 of 8 questions** through
the real training driver (`development-question-initialization-01/`). Making mint,
open and all retained candidates available does not make a greedy argmax sample
the near-uniform distribution. A second clarification asks whether certificate
(b) should verify the actual probabilities and report the observed openings, or
whether its required nonzero rate needs a revised rule. No opening rule or
certificate bar has been substituted.

A separate observation-only probability probe (`development-binding-probabilities-01/`)
used one new unforced start, without replacing the first result. Its 378 legal
mint/open pairs had conditional open probabilities 0.49905–0.50002. Six of eight
sentences opened episodes, but none of the eight outer source rows remained
open after the closing. These observations do not establish the requested open
referent or answer filling. They are retained alongside the initial 0/8 result.

Candidate changes so far:

- Compose retains a per-row reservoir snapshot and joins a detached suffix at the
  sampled round. The rebuilt reference bank is detached with the fork. Thought
  uses an episode continuation at its sampled step; the prefix is not executed
  again. Both comparisons precede updates, keep uses reconstruction, and departure
  credit uses reconstruction plus answer.
- The bind menu has fixed masked candidate, mint, open and unchanged slots;
  recency retention precedes enumeration, and `found` is a feature. A selected
  determiner's binding survives enclosing operations. Later native bindings clear
  stale open flags in the clause journal.
- Openness includes missing relational operands. The saved answer/start-2 next
  batch was reproduced before changing that guard: **`query` produced the two-role
  result** (`[true, true, false]`) whose missing NP2 was absent from the old open
  metadata. See `diagnostic-answer02.json` and its unchanged-baseline replay log.
- `isPart` proposals require two accepted concept references and exclude open
  grammar forms. Episode-local knowing, gain and prediction effects are restored;
  pending rows are held in LTM, with history and credit retained in their owners.
- Reference variants of an operation run as one numerical batch where its
  contract allows it. Large alternative menus use exact compact state keys;
  addresses remain int64, and equality is not approximated or hashed.

Development checks include the eight real corpus sentences through `present()`
with optimization and the frozen observer, distinct row departure rounds, and an
episode-state diff. The latest focused compose/fork/reference check was **52/52**
(`development-menu-and-fork-02.log`). Separate thought tests verify that the
explore episode starts at its detached departure and calls `begin` only once.
These are development results, not the requested complete sweep or thinking gate.

The 57 standing thinking selectors passed in `development-thinking-01/` under
the existing observer that rejects explicit seed calls. This is a development
check, before declaration and freeze. The complete second development sweep
ran 5,620 cases and found eleven failures. All eleven passed in the focused
13-case check after two controller guards and test ports. A separate unit also
checks that an empty operation menu finishes with its reference open and no
`conclude`. The third development sweep is green: **5,336 passed, 284 skipped,
one XPASS**, all 5,621 selected cases completed in 1,268.37 seconds. This is the
development source, before the missing certificates, declaration and freeze;
the standing thirty have not been launched for repair pass 2.
`development-source-03.zip` contains the exact 738-file source manifest validated
by that green sweep. The post-sweep integrity check again found zero mismatches
in either prior receipt (2,393 and 1,967 files respectively).

The first partial epoch timing runs (`development-epoch-*-01`) were interrupted
after the exact state comparison was compacted. Their logs, profiles, process
outcomes and `interruption.json` records are retained. The revised full-epoch
development runs (`development-epoch-*-02`) use budget 32 (zero for the control),
capacity 32,768, graded document order and a two-hour development timeout. They
import the original corpus, driver, verifier and observer unchanged. Epoch count
and final capacity will be declared only after the completed development timings.

The exact revision loaded by these epochs is preserved in
`development-epoch-02-source.zip`; every one of its 737 entries was verified
against their starting manifests. Subsequent development edits to three source
files and six tests are why the live-source comparison in the epoch results says
`source_unchanged: false`. These development runs are not a source freeze.

The expectation-only epoch completed 1,332 sentences in 2,058.56 seconds,
occupancy 6,863. It opened 3/117 questions, with zero correct bindings, and
1,094/1,215 declaratives. The answer and zero-budget epochs are still running.
`capacity-profile.json` supplements the ordinary episode profiles with a small
constructed two-row query at three reserve sizes. Its menu and score costs are
similar; its query charges two record reads at every size. Median query times
were 0.252, 0.259 and 0.472 ms respectively, so the small timing probe alone does
not certify that reserve size has no per-step cost. It is not an
ordinary-path certificate or a capacity declaration.

`development-status.json` records the running development workers and their
output folders. Each has its own two-hour timeout and 8 GiB memory guard; all
outcomes remain development evidence. Neither worker can launch measurement.
The pending certificate rulings precede any protocol declaration or freeze.

The first broad development sweep stopped after a source edit: 384 cases had
run, with one stale lasting-knowing assertion failing. That test now checks the
required restoration of episode-local knowing while preserving retrieval history;
the original text is in `starting-source.zip`. The later focused check passed it.
The incomplete sweep remains in `development-sweep-01/`; it is not green coverage.

Ordinary-path diagnostics also remain intact. They show that the untrained
chooser frequently selects a projection, union or another operation instead of
the intended copula. Consequently some answer lines still open their own
references, and the unforced tiny-document probes do not yet certify the requested
answer fill or the two pending-reference arrival orders. No correctness or learned
chain result is inferred from construction-only unit tests or these timings.

## Section 14 continuation

Alec’s two certificate clarifications are resolved: c/d/e may force grammar and binding choices while exercising the real driver, frozen observer and live explore suffix; b checks the distribution and reports greedy openings without a per-start rate bar. The live work is governed by §14.7. No protocol is declared or frozen.

The previous answer dry epoch completed in 4,510.97 seconds, 1,449 sentences, occupancy 18,766. The zero-budget development worker outlived its vanished guard and its two-hour timeout; it was stopped, with the partial progress and reason retained in `development-epoch-zero-02/interruption.json`. It is not a completed epoch. The preceding notes about running workers and pending decisions are historical.

The new development changes remove evidence pseudo-slots, allow answer cost on a filled referent regardless of its pair, expose addressed open columns to binding, add configurable exhausted-search conclusion, and record formation choices/probabilities. BindingAnswers.cost is changed only as specifically required by §14.7.2; the original contract hashes and both earlier receipts remain intact. All old development results predate these changes.

The §14 focused development checks have passed 114 and 83 selected tests. The ordinary certificates a/b/f and the explicitly forced c/d/e now have passing checks, including both premise arrival orders. The full bind-menu probe included 224 retained-candidate entries, with probabilities 0.95503–1.08844 times uniform; it reports 0/8 outer committed openings and 8/8 episodes without imposing an empirical rate bar. The forced helper may select a legal, distinct binding alternative at the actual fork; the full suffix executes and its costs and keep decision remain live.

The first new answer-epoch development attempt, `development-epoch-answer-14-01`, failed before the first batch completed: an all-free region was mistakenly represented as an empty meaning. Its six-second result and exact source archive remain intact. Retrieval now carries a separate free-variable mask. Certificate (g) still needs ordinary unforced examples; the next epoch records these alongside answer-only cost differences, total episode costs, chooser movement, and declarative episode shares in both halves. No protocol is declared, no source is frozen for measurement, and no official attempt has been started.

The all-free region correction passed `development-region14-02` (18 tests). `development-epoch-answer-14-02` is the new unseeded development epoch at the preceding dry settings: budget 32, capacity 32,768, graded order, with the frozen observer and live paired training. Its source archive and timings are local to that development attempt. The prior receipts were rechecked after these repairs: 2,393 and 1,967 files, zero mismatches (`prior-integrity-section14.json`).

`development-epoch-answer-14-02` was stopped for a diagnosed scoring repair after 449.47 seconds, with partial progress and the reason saved in `interruption.json`. Empty-search minting filled a role but omitted `_bound_roles` until commit, so the trial answer scorer still charged it as unfilled. This is now recorded before scoring. The partial run is not a completed development certificate; it is retained in full. Some ordinary exhausted closings also retained an unfilled relation after minting their referents; those are not the single-new-name case of certificate g.

The corrected §14 check completed **13/13** in 61.82 s (`development-certificates14-05.log`). The pending-arrival fixture uses zero attention budget so the reference persists until the next sentence; it forces real surface attachment for whitespace units as well as the labeled grammar/binding choices. The empty-search mechanism is checked separately at budget 32 on ordinary c/e sentences with an explicitly forced query/conclude preference. This is conditional mechanism coverage, not an unforced learning result. `development-epoch-answer-14-03` is the new unforced answer epoch, with all earlier development outcomes retained.

`development-epoch-answer-14-03` was stopped after 713.71 seconds for an item-5 violation discovered in review: generic `ThoughtFeatures.semantic_metadata` encoded formation provenance for the chooser. The encoder now excludes `_formation_records` and `_formation_reason`; a direct feature-invariance test covers choice, probability and reference changes. The interrupted development outcome, source and helper archives remain intact. Its two observed non-tied question answer comparisons are partial evidence only. No full epoch certificate or declaration follows from them.

The provenance repair passed all **26** §14 and repair-2 checks in 73.76 s (`development-certificates14-06.log`). The strengthened pending-reference certificate also passed both orders in 23.91 s, checking the arriving `three` binding and nonvacuous backward reference in the reversed order. `development-epoch-answer-14-04` runs the repaired source unforced at the same dry settings. It preserves initial entropy (without setting a seed), chooser weights, full source and helper archives, and per-query outcome diagnostics.

`development-epoch-answer-14-04` failed after 599.88 seconds and 477 completed sentences: `a clause over a relation cannot fuse`. The writer rejected a fused clause retaining a relational reference. Its source comparison was unchanged. The partial run has one non-tied question answer comparison and nonzero chooser movement in its saved checkpoint; neither establishes the required completed epoch. The saved entropy and saved corpus presentation are being replayed for diagnosis (`development-replay-answer14-04-02`). An earlier diagnostic replay that restored global RNG but regenerated the corpus was stopped and retained; the corpus owns an independent unseeded generator. No declared measurement was started.

The diagnostic reproduced all 83 completed batches with matching occupancy and the same failure on batch 84. `verb` had bound a relational operand; subsequent transparent `tense` operations carried the relative flag, but clause reconstruction overwrote it with the point status of the VP head reference. The closing now preserves the journal flag. The strengthened regression fails on the archived failing source and passes on the repair; the wider check passed **69/69** in 73.89 s. `development-epoch-answer-14-05` reuses the saved unseeded entropy and presentation for a full repaired development epoch, with no forced grammar and no replacement of the failed outcome. See `clause14-diagnosis.json`.

The repaired unforced development epoch completed **1,449 sentences / 271 batches in 1,618.22 seconds**, occupancy 4,510, with unchanged source. One of four question comparisons had unequal answer costs (two had unequal total episode costs), and chooser parameters moved (MLP first weight norm Δ 0.13749; thought projection Δ 0.11306). Only 4/117 questions opened episodes; none bound correctly. The fixed declaration prerequisite **did not pass**: declarative episodes were 116/607 = 19.11% in the first half and 119/608 = 19.57% in the second. No protocol or measurement freeze follows. The source, completed outcome and diagnostic sentence-kind breakdown are retained. Timing per sentence: plain 1.1263 s, question with episode 1.8953 s, question without episode 0.9915 s, answer line 1.0906 s. All 53 unforced one-query exhausted declarative examples minted correctly; no unforced empty-search question example was observed.

The late increase is concentrated in counting/worked-successor sentences: 34/213 → 93/388 with episodes; variable-value premises decline 32/219 → 16/132 and dependent premises 12/73 → 5/44. This is diagnostic breakdown, not a replacement certificate. The original half-by-half criterion remains failed. `section14-review.json` contains the current review summary. Measurement launchers are prepared but refuse to start without a declared protocol and passed development certificate; neither exists. The prior receipts again verify with zero mismatches (2,393 and 1,967 files). No commit, push or bump was made.


## Section 14.8 validation

The first complete sweep is retained as `validation-sweep-01/`, with its exact
freeze in `validation-source-01/`. Seven stale assertions in four fixture files
failed: renaming erased source evidence; bound zero-evidence relations and
nested rows were still expected to become questions; and a purported complete
query description lacked native references. `validation-fixture-corrections.patch`
records the changes. The focused check passes 9/9. The learner and measurement
helpers did not change. The corrected tests are re-frozen before the required
green sweep. No declared math attempt has started, been retried, or replaced.


## Frozen verifier and required reporting

`binding-answers-frozen.diff` compares `BindingAnswers.py` with the original
frozen hash `7ac86e5f5980db35d9e92b0a4b89c35d60bf951cad14be54900f96d6f10169c7`.
The bodies of `references()` and `matches()` are byte-for-byte identical;
`binding-answers-verifier.json` records that check, and
`binding-answers-matches.py.txt` shows the verifier. The authorized loss-side
change scores a filled referent independently of the evidence pair or another
free role. Its exact-identity traversal is AST-identical to the frozen verifier's
traversal; the unfilled penalty and code-distance formula are retained.

The new reporting wrapper leaves the frozen corpus, presentation driver,
observer, and verifier untouched. It reads the already executed greedy compose
trial before commit to attribute an open reference to `what`, and records the
kept episode's native work meter. `greedy-what.json` reports each start's first
question and first-epoch greedy openings. `episodes-by-epoch-kind.json` gives
counts and mean work per episode for plain sentences, questions, and answer
lines, per epoch. `episode-batches.jsonl` retains the underlying observations,
including any partial failed batch. Opening share is report-only. Per-sentence
wall-time allocation divides the shared batch time equally and adds the
separately timed episode to its owning row, as in development.

The unforced recorder smoke and an explicitly forced positive recorder probe
are preserved; the latter checks that two open `what` references are recognized
and their answer lines open no episode. Neither is a declared learning attempt.


## Standing thirty and the §20 path comparison

All thirty original attempts completed once, with zero thought calls and zero
opened episodes. Sum passed 10/10, and both XOR bars passed 10/10. MM passed
8/10: run 1's best MSE was .22415423393249512 and run 9's was
.2413320243358612 against the unchanged strict .20 bar.

`mm-bisection/` compares each failing run's saved entropy state with the exact
6.2 mechanism landing (`e43638a`) in an isolated source archive. Both complete
200-step diagnostic trajectories match the original measured trajectory.
Landing and candidate have identical initial and final parameter hashes,
construction RNG, RNG states before/after every forward, IR masks, narrowing
actions, predictions, and losses. There is no changed numerical or RNG path
on either failing trajectory; this is stronger than the RNG-only exception in
operators plan §20. The raw 8/10 count remains a miss count, not a substituted
10/10. These four diagnostic executions are separate from the declared thirty;
no original attempt was retried or replaced. `standing-thirty-verification.json`
records every original result and the basis for proceeding.
