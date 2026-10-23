# Step 2b storage: review repairs

2026-10-22. Storage-only §36, continued under §§39–41, with the
[previous receipt](../README.md) and [Claude's HOLD](../landing/claude-review.md)
retained. No source has been published from this round. Work stays on
main. Identity threading, MTM values, tags, READING, derivation removal and
close-time mint removal remain outside this source.

## Repairs and scope

§59 addresses the three review findings. Closing questions and answer
fields retain the live clause/parse mode; stored rows gain no replacement
mode field. A live `ask` supplies query intent when following an LTM
description, matching the executor. Thought-memory records retain their
live-mode check. Four new cases cover writing and closing a zero-evidence
yes/no question, absolute/relative and with/without a supplied parsed view.
The unchanged direct/nested expectation regression covers held questions.

Occurrence end states again use `(-1, -1, -1)` in their legacy numeric
reference field. Typed slot/constituent references remain in the meaning;
Part returns retain their two native arguments. The empty native predicate
cache and unreachable lookup are removed.

The module set adds all direct clause writers and reading-commit callers,
plus the reviewed thought/expectation files. [Expanded scope](expanded-scope.json)
and [module coverage](module-coverage.json) identify each file. This is a
scoped gate set, not step 2's full sweep. §60 records the exact count ports: relative-insertion and lies `0 -> 2`
(opposite polarities, one unchanged Part edge); selected-relation `0 -> 1`
(two cases); concluded `not` `1 -> 2`; taxonomy conclusion `0 -> 1`; and
replacement-contract conclusions `0 -> 1` (three cases). Evidence,
ablations, Part counts and content stay exact.

The four predicate-argument fixtures [replay identically](predicate-arguments.json)
with both writers: their retained derivation roots have no bank seat. The
port checks each exact source-to-root link and its original full operand
content in the source occurrence. No reader or minting policy changes.
The joint-objective fixture asserts absence of the retired prediction
switch and reads its bounded predictor context through `_read_context`
before testing the unchanged `not p.requires_grad` predicate. All loss,
gradient and parameter-update checks stay; no detach is added.

The narrow-idea test now excludes every `ltm` operand reference instead of
comparing a row with itself. The [retained comparison](narrow-idea.json)
passes with both writers on the same captured diagnostic initialization;
this is diagnostic evidence, not a seeded gate. The assertion ledger also
records changed loop scopes, including the previously disclosed removal
of the nonexistent `where` column from the packed-row comparison.

## Every assertion, fixture and retirement

[Exact changed checks](assertion-ports.md), [machine-readable assertion
ledger](assertion-ledger.json), [every changed function](fixture-ledger.json),
[whole test patch](test-ports.patch) and [copied observer changes](copied-observer-ports.patch)
are relative to `bb0045d7a`, the unchanged Claude §38 commit. New regressions
and fixture helpers include their complete bodies. The only retired tests
remain the three P3 adjacency/nudge-only cases. P5 retires none. P2, tagged
P4, P6 and P7 are deferred. Every changed count is listed exactly.

The three §46 Part-read tests already failed independently of 2b after
§10.4 retired reader-side equality binding; published main carried those
stale tests. Their selected-refinement content remains gated by arrival
regressions. No wrong antecedent is relabeled and no learning claim is made.

## §41 notes resolved

1. **Forgetting:** §42 chooses the explicitly permitted conservative stage
   barrier: advance the selected low-value child's owner, then preserve
   pressure with `forget_pending` instead of substituting a costlier
   deletion. The flag persists below high water and across restore, while
   the configured interval still applies. One boundary is one pass; no
   occurrence advances twice or newly released child disappears in that
   pass. Gist vectors are unchanged. Two-pass and checkpoint regressions
   cover this choice; Claude accepted this justification in the HOLD.
2. **Fact admission:** `accept_fact` rejects stored `question` and `estimate`
   kinds. The question regression checks rejection before/after restore.
   A row stored as `observation` with evidence carries no saved mode; the
   former additional interrogative-mode refusal is retired. Incoming live
   TruthSet questions are still refused.
3. **Mode consumers:** `LTM0.MODES`, its column, raw read, semantic
   validation, migration and writer/admission readers are gone. Retired
   checkpoint keys are discarded. `reasoning.py` drops its stored-mode
   conjunct. `ThoughtFeatures.MODES` still encodes live meaning modes;
   stored meanings now supply the ordinary default. Closing questions,
   end-state answer seeds and held-description expectation use their live
   contexts as described above. Nothing reconstructs a saved grammar mode.

## Evidence and publication

The first frozen attempt `bfa9c94ee` has 674 passes, eight skips and one
module-group failure in the native end-state fixture. Its draw/replay and
common-handle port are retained. The second source `3c5cc2a1d` has 675
passes and eight skips, followed by Claude's HOLD. Development runs 01–12,
including all failures, remain under the parent folder. None of their
passing subsets is promoted to this repaired source.

`run_gates.py` records a new frozen hash and verifies all audited
production/configuration/test hashes before and after each group. The
eight regression files, eight-file native set and expanded module set run
on that source. There is no fresh series or full sweep. Claude's review
follows the complete gates; publication is then BasicModel main's push and
the WikiOracle gitlink bump. The §§34–35 review and Claude's next identity
spec follow publication. Item 3.0 has not been reached.

## Pre-freeze audit

The audit covers 63 production, configuration, fixture and test files
relative to the baseline: 70 changed-check groups, 79 modified existing
functions, five new regression functions (nine cases), three added fixture
helpers and exactly three retirements. Development-10 gives 346 passes,
one existing stage3 prerequisite skip and 12 failures; development-11 gives 11 passes
and the context-storage failure; development-12 passes that final case.
All are diagnostic only. Large captures are packed losslessly with byte
hashes in the [parent manifest](../packed-captures.json).

## Frozen source

`9c3899f2f6ee3c8b6878a52220479f150a5ac039` is the source for all three
expanded groups. Their commands, initial states and reports are recorded
here as they complete. No production, configuration or test edits follow
this freeze. Claude reviews only after the complete source-linked gates.

## Additional probe: source withdrawn before publication

The [held-content probe](held-idea.json) found an ordinary stored idea
raising at the nested thought-signature reader. §62 documents the guard
and two regression cases for the next source. This source's results remain
retained and will not be promoted, even if its selected gates pass.
