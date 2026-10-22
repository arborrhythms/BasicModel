# Review step 2a's frozen `where` cull

User-authorized review by Claude. Read only: do not run tests, edit files,
publish or make Alec's decisions. Review before publication is required.

The published rename is BasicModel `804d8adcb` / WikiOracle `dbd9b52`.
This round's source is `d4f316411a1b51965f2448390743c46ee191b6ac`.
Only three production files change: `bin/Layers.py`, `bin/ClauseRow.py`
and `bin/ReferenceContext.py`. Subsequent changes are receipts/docs only.

Read the step-2 spec, `doc/plans/2026-10-21-step2-one-ltm.md`, especially
§§16–17, and this receipt's `README.md`, `source-audit.json`,
`source-diff.patch`, `frozen-source.json`, `gate-runs.json` and
`gate-summary.json`. Inspect the changed source/tests and raw reports as
needed. The runner checks that bin/test/data still match the frozen hash.

Alec's decision: `where` belongs with percepts and comes off the LTM0
row, row export and SituationFrame; `.when` stays. He authorized removing
the exact-value check and separately authorized the shape-check port
retained in `proposed-shape-port.patch` and `shape-port-authorization.json`.
The latter leaves `assert store.when.shape == (8, 4)` intact. The only
other existing-test edit removes an unused `where=` constructor argument
from the time-isolation fixture; every assertion there stays verbatim.

The implementation removes the buffer and its close write, export,
compaction/reset entries, the frame field and the synthetic relation-row
location. Strict checkpoint load discards the old column. It leaves the
percept/Clause working field, `.when`, metadata authentication, polarity,
modes, refs, ownership and all remaining culls unchanged.

Gates are run once on this hash: the direct storage file, two new
checkpoint/compaction cases and the existing time-isolation case; the
eight regression files; and the retained eight-file set. Results are in
`gate-summary.json`. Full initial-state/reference/keep captures are
retained. No fresh twelve-draw series, no full sweep, no evidence promoted
from earlier hashes. All prior failures remain retained in their receipts.

Please verify the exact scope, authorizations, restore behavior, retained
time behavior, absence of missed live `where` consumers and the claimed
gate coverage. Give a clear publish/hold verdict with any blockers and
precise file locations. This review is for the 2a cull only. The three 2b
representation questions will be written and reviewed separately before
2b implementation. The two R10 arrival notes stay carried to 2c. There is
no learning claim and item 3.0 has not been reached.
