# Item 6.2 repair pass — October 7, 2026

Status: **accepted by Alec as the item 6.2 mechanism landing**, following
[Claude's review in thinking spec §11](../../specs/2026-10-07-thinking.md#11-review-of-the-repair-pass-claude-2026-10-07-accepted-as-the-mechanism-landing-the-math-gate-is-the-learning-step).
Commit, push and WikiOracle bump authorized on October 7. The
[acceptance record](acceptance.json) preserves the measurement limits.

The configured MM_query_reasoning run demonstrates completion, **not learned
chaining**. The 6.5 learning gates remain pending the million-sentence
checkpoint and are not claimed. The recorded 56/57 result below is unchanged;
the nested fixture seed is removed for the next measurement. MM_math_chain
is the separate learning step, which stops for Claude's review before commit.

This starts from the [held 6.2 candidate](../2026-10-07-item6-2/README.md),
source `d6d00f6640af051a4c84599ef5a5a235cb25faa9691b873b855699621cf80b62`,
following [Claude's review, spec §9](../../specs/2026-10-07-thinking.md).
The first receipt remains byte-for-byte intact (1632 files checked).

The [remaining review issues](review-notes.md) distinguish a fixture failure
from the measured training outcome. The thinking gate is **56/57**, because a
nested helper's seed call was rejected before the per-row integration could
construct its model. The configured run completed **300/300 epochs**, but its
single thought-credit comparison was an exact tie. It demonstrates completion,
not learned chaining. The repairs and all other selected checks passed.

## Repairs

Fill now carries a supplied meaning's constituent graph into the goal's
local table and rebases its transferred addresses, including binding by a
stored occurrence. Each nested child retains its own local table; shared
children remain shared. Fill validates the goal, supplied graph and result,
including local references in semantic metadata. Malformed or dangling
addresses raise at fill, before indexing or writing. Tensor gradients survive.

The exact saved construction fails before the change and succeeds after it,
writing the child and conclusion: [before](construction-before.json),
[after](construction-after.json). Twelve ownership cases cover empty and
nonempty goals, nested graphs, shared children, occurrence binding, malformed
addresses, nested metadata and gradients.

Three 11c tests now use the resolved 104-wide identity event, current
universe/neutral-property routing and native percept IDs in the live top-K
consumption window. Two tests were retired because their word-row reading
projection and `_primed_reading_step` contracts were removed. Their exact old
texts and reasons are in [the retirement record](test-retirements.json);
[all port descriptions](test-ports.json) and complete old test files are saved.
The retained 11c helper no longer sets seeds. All seventeen surviving nodes
remain in the explicit gate.

The MM routing fix loads the configured grammar before constructing its
predictors. Their routing width now matches this model's rule vocabulary,
including after another grammar was loaded. The optimizer smoke keeps its
training call; the configuration check verifies the widths. The thinking
launcher reads `result.json` from the returned report path.

## Measurements on one frozen source

| Check | Result |
| --- | --- |
| Default complete sweep | 5568/5568; 5283 passed, 284 existing skips, 1 non-strict XPASS |
| Explicit thinking gate | 56 passed, 1 failed; 57 selected |
| Original 25 certificates | 25/25 passed |
| Twelve ownership certificates | 12/12 passed |
| Ported 11c nodes | 17/17 passed |
| MM configuration / optimizer smoke | 2/2 passed |
| Per-row answer integration | Seed guard failed before model construction |
| XOR class / reconstruction / both, shared ten trainings | 10/10 / 10/10 / 10/10 |
| Sum control / MM_xor, ten each | 10/10 / 10/10 |
| Thought episodes in the standing thirty | 0 |
| Unforced configured MM_query_reasoning | completed; 300/300 epochs |

The configured run has one recorded credit observation: an exact cost tie,
with no nonzero raw gradient into the shared operation scorer. Its scorer's initial-to-final parameter change has L2 norm
`1.4528905087504884`. [The full outcome](mm-query-configured/outcome.json) reports
operation chains, cost pairs, credit sources, gradient norms and per-epoch
parameter movement; initial and final scorer tensors are saved beside it.


The movement is measured during the actual run, without reseeding, replaying
training or forcing a reading. Other owner losses also update the shared
scorer, so total movement alone does not establish learned chaining. The
focused answer and expectation certificates test its credit direction.

The four standing gate/config files match 6.5 exactly. All thirty trainings
were attempted once; no failed outcome was replaced. The sum control is read
before XOR starts. Resource limits and the independent MM/thinking/standing
schedule are in [the predeclared protocol](protocol.json).

## Review evidence

- [Machine-readable review summary](review-summary.json), [standing measurements](measurements/summary.json), [thinking result](thinking-gate/result.json) and [full sweep](final-sweep/result.json).
- [Only the repair's source changes](repair-source.patch); complete source and measurement-helper archives under `measured-source/`.
- [Frozen hashes](measured-source/freeze.json), [initial first-receipt hashes](first-receipt-hashes.json) and [thinking selection](thinking-gate-plan.json).
- `affected-01` records the development fixture's two-row/four-row mismatch; `affected-02` passes its corrected native read. `affected-03` passes all twelve ownership checks.
- `development-sweep-1` preserves the first green source before the final occurrence-binding case. The final sweep matches the delivered source. No configured training or measured gate had run before that change.
- [Premeasurement launcher correction](launcher-development.json); the measured thinking launcher has no post-report formatting failure.

The 6.5 learning gates — held-out anaphora, verb reuse, prediction control,
shuffled order, renamed vocabulary and determiner control, seeds 0/1/2 —
remain pending the million-sentence checkpoint and are not claimed.

## Final verification and concurrent spec revision

The [final verification](final-verification.json) matches the measured source,
all 191 frozen measurement helpers and all 1,632 first-receipt files. The
original 25-certificate source and four standing files remain unchanged.
`git diff --check` reports one trailing blank line in the retired-test file;
it is left in the frozen measured source and recorded for a later revision.

The [earlier review-document archive](measured-source/review-documents-before-final-verification.zip)
already contains an earlier §10. Concurrent changes revised that math-chain
gate and the equality contract; the [observed revision](spec-observed-at-final-verification.txt)
and its [difference](spec-review-to-final-verification.patch) are saved.
This receipt covers the recorded repair protocol. **MM_math_chain and the
amended equality-rewrite contract have not been implemented or measured in
this repair.** The live spec may continue to change independently of these
archived observations.
