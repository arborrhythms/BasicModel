# Step 2b storage landing

2026-10-22. This is the final-source receipt for the storage-only round
defined by step-2 §36 and carried under §§39–41. It continues the
[complete scope, port and failure record](../README.md). All work is on
main; the baseline is `bb0045d7a` (Claude's §38 edit committed unchanged).

The first frozen source `bfa9c94ee` completed 683 scoped cases: 674 passed,
eight existing prerequisite skips, and one failed native end-state case.
Its [summary](../gate-summary.json), reports, initial states, and the
[captured-state diagnosis](../relative-end-diagnostic/run.json) remain
retained. The failing draw selected a relative Part handle; the fixture
looked it up only in LTM0. The exact §57 port uses the common location
reader, compares a Part's read for a Part handle, and still compares raw
LTM0 slots for an absolute occurrence. It keeps zero tolerances and every
cleanup/no-program assertion. Production is identical to `bfa9c94ee`.
The [captured replay after the port](../relative-end-port/run.json) is
diagnosis only, not a seeded gate or promoted evidence.

## Every port and retirement

The current [assertion ledger](assertion-ports.md) lists each changed check
against the baseline, including §57. The [fixture ledger](fixture-ledger.json)
contains every changed function, parameter and setup, including unchanged
assertion text whose loop scope or input changed. See [the whole test
patch](test-ports.patch) and [active observer copy ports](copied-observer-ports.patch)
for the complete context. New storage regressions are included in the
ledger with their bodies; no test evidence is substituted for an assertion.

P1, P3, P5 and the unused-refs absence/snapshot ports are recorded in the
parent receipt. P5's two fresh Part corruption subcases retain the approved
messages and retire no test. The only retired tests are the three
adjacency/nudge-only cases listed there and in this ledger. P2, tagged P4,
P6 and P7 remain deferred. Every changed completed-S count is explicit in
the ledger, including the six §40 operand cases and §55's composite case.

## §41 dispositions

- **Forgetting:** the receipt adopts the conservative stage barrier that
  §41 explicitly offered, with §42's rationale: advance the selected
  low-value child's owner, then carry pressure to a later boundary instead
  of substituting a costlier deletion. `forget_pending` persists below
  high water and across restore; the interval still applies. One boundary
  is one pass, no occurrence advances twice, no released child disappears
  in that pass, and the retained gist vector does not change. The two-pass
  and checkpoint regressions cover this alternative.
- **Fact admission:** stored `question` and `estimate` kinds remain
  refused. The new question regression checks the question's evidence and
  rejection before and after checkpoint load, plus ordinary fact admission.
- **MODES:** no active `LTM0.MODES` consumers remain. The old raw read,
  semantic validation, migration, writer and admission consumers are gone;
  the embedded-question test checks kind/evidence. Retired checkpoint keys
  are discarded. `ThoughtFeatures.MODES` remains for the live parse, and
  `reasoning.py` no longer applies its old stored-mode conjunct.

## Frozen evidence

`run_gates.py` pins the final hash in `frozen-source.json` and verifies the
audited production/configuration/test hashes before and after each group.
The regression files, eight native files (including the full bootstrap
file), and [direct files for every changed module](module-coverage.json)
run once each on that hash. No result from the first frozen attempt or a
development run is promoted. This is a scoped run, not step 2's full sweep
or a fresh-draw series. Existing prerequisite skips stay classified.

After all gates, Claude reviews this final hash and the full port package.
Publication requires that review, then a push of BasicModel main and a
WikiOracle gitlink bump. Identity threading, value storage, tags, READING,
derivation removal and close-time mint removal are outside this round.
No learning claim is made. The requested code review of §§34–35 follows
publication; Claude specifies that next line before implementation.

Final gate counts, review and publication are recorded below when complete.
