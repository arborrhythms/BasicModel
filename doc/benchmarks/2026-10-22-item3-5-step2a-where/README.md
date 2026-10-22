# Step 2a: retire the LTM input-location column — 2026-10-22

Base: published BasicModel `804d8adcb` (WikiOracle `dbd9b52`).
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
frame exports, a new and a legacy checkpoint, the retained evidence and
content, exact model time through restore and compaction, and reset.
Both checkpoint cases use the same native fixture; no seed is pinned.

On the frozen source, run once:

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
