# Step 2a: retire the LTM input-location column — 2026-10-22

Base: published BasicModel `804d8adcb` (WikiOracle `dbd9b52`).
Frozen source: **`d4f316411a1b51965f2448390743c46ee191b6ac`**.
Authority: [step-2 spec §16](../../plans/2026-10-21-step2-one-ltm.md#16-claudes-review-of-step-2a-decided-alec-2026-10-22)
and Alec's handoff. The incoming spec edits are retained in
`incoming-spec-16.patch` and included with this round.

This round removes `where` from the LTM0 buffers, closing write, row
export, reset and compaction. `SituationFrame` no longer carries it;
the relation-row view no longer supplies a synthetic zero location.
Old LTM checkpoints discard the retired column at load. `.when`, percept
location, clause working fields, metadata authentication, evidence, modes,
references and table ownership retain their existing contracts.

The authorized exact-value assertion is removed from
`test_clause_storage.py`. Spec §17 records Alec's separately authorized port of the
same file's shape assertion: remove `store.where.shape ==` and preserve
`store.when.shape == (8, 4)` (conversation reply, 2026-10-22). The
independent-components fixture omits its unused `SituationFrame(where=...)`
argument; every assertion in that file stays verbatim. No fixture is
resized, and no existing failed test is repaired under the new-test exception.

New regression coverage in `test_ltm0_location.py` checks the row and
frame exports, a current checkpoint and one carrying the retired column,
the retained evidence and
content, exact model time through restore and compaction, and reset.
Both checkpoint cases use the same native fixture; no seed is pinned.

On the frozen source, run once (all complete):

- The changed storage file, the two new location cases and the existing
  independent-components time-isolation case. These directly exercise
  the removed interfaces and the authorized test port.
- The eight regression files retained from the rename receipt.
- The retained eight-file set, with initial-state, reference and keep
  captures for its pending-premise cases and the bootstrap case.

The bounded runner stops on a failure. All outcomes and any failed runs
remain attributed to their source; no earlier pass is promoted. No fresh
twelve-draw series and no full sweep. Claude's review precedes publication
to both mains and the WikiOracle bump. 2b follows only after this landing,
with its three representation questions raised as required by §16. The two
R10 arrival-plan observations remain carried to 2c. No learning claim.

## Results on the frozen source

| Gate | Result | Report |
| --- | --- | --- |
| Direct storage and time checks | 25 passed | [storage](storage/report.html) |
| Eight regression files | 85 passed | [regressions](regressions/report.html) |
| Eight-file set | 113 passed, seven existing prerequisite skips | [native](native/report.html) |

Total: **223 passed, seven skipped, zero failures**. The skipped cases
remain the copular artifact prerequisite (one), re-identification in MTM
(four) and item-0 pending-premise learning (two). No skip was added or
changed in this round. [The summary](gate-summary.json) records the exact
source, selectors, outcomes and report digests; the raw bounded-runner
reports and captures are retained beside it.

The pending-premise first draws passed with **0 bound / 8 open forward**
and **0 bound / 16 open reverse**, zero attention budget and no forced
closure. Their initial model/RNG, generator, arrival, resolver and keep
records are retained in the `native-state`, `native-reference-evidence`,
`native-keep` and `native-reports` directories. The bootstrap initial state
is retained in `native-bootstrap-state`. Both new restore cases passed,
including strict loading of the retired column, with the evidence, content
and exact `.when` checks unchanged. There was no failed run on this source.

Production and test sources still match the frozen hash.
[Claude's read-only review](claude-review.md) says **publish, no blockers**.
The request, full tool stream, returned report and model/run metadata are
retained here. The requested parent-repository consumer search is recorded
in [parent-consumer-audit.json](parent-consumer-audit.json); it finds no
matching Python use outside the submodules. No extra gates were run.

The review's checkpoint wording correction is applied above: the second
case is a current-format checkpoint carrying the retired column, not an
old-schema fixture. `SentenceEndState.where` remains a percept/working
field in bounded recall history; its disposition is carried to the 5.5
context-wholes spec, with no change in this round. This landing publishes
the reviewed source and receipts to both mains with the WikiOracle bump.
Step 2a's rename and sole cull are complete; 2b and 2c remain outstanding.
Step 2b has not been implemented and item 3.0 has not been reached.
