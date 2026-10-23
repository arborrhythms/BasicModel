# Step 2b: storage changes before identity threading

**Current receipt:** [word-form/README.md](word-form/README.md).
Earlier freezes remain retained: `bfa9c94ee` failed its relative-handle
fixture; `3c5cc2a1d` received Claude's HOLD; `9c3899f2f` passed its scoped
set but the additional held-idea probe found the guard fixed in §62.
`1f53af50c` then failed the rung-zero audit; §64 repairs word-form
ownership. None of those results is promoted to the new source.

2026-10-22. Work is on `main`, starting at `bb0045d7a2b7a7e5be584066ca1ee283fa899d75`
(Claude's §38 edit, committed unchanged). Published 2a is
`0b030e3ebdeca87f9ebf80739237437c3c75ea8f`. Step-2 §§36 and 39–41 govern;
§39 supersedes the earlier chat stop. The approved §40 port is applied.
The [earlier partial-work record](../storage/README.md) is retained unchanged.

This receipt is prepared for the source freeze. Frozen-source evidence and
Claude's verdict will be recorded below; development results are not gate evidence.

## Scope

- Drop serialized leaf postings, adjacency and its placement nudge, the
  `symbolCentroid` configuration, stored grammatical mode and unused LTM0
  `refs`. Keep `leaf_complete` as a non-persistent buffer; rebuild after
  the complete owner's Part table and authenticated sidecars load.
- Keep trust-derived evidence in its original orientation on LTM0. Only
  a negative meaning's trust-derived LTM1 edge pair reverses. Explicit
  pairs and the meaning's polarity retain their contracts.
- Keep an LTM0 occurrence for each completed S, including a relative
  sentence beside its Part edge. Relative closes still return their Part
  handle to existing readers. Derivation and close-time minting stay.
- Stage forgetting without rewriting a retained content vector. Freeze
  incoming dependencies for a boundary pass; a released child is eligible
  at a subsequent pass. Transitive live structure remains protected.
- Repair retired-column consumers, deep participant traversal, and
  provisioning's leaked recall context as documented in spec §§43–56.

No tags, READING layout, identity threading, MTM value redesign, derivation
removal or recurrence policy lands here. No learning result is claimed.

## The three §41 notes

1. **Forgetting:** use §41's explicitly offered conservative alternative.
   After advancing the selected low-value child's structural owner, stop
   that pass, preserving the retention ordering rather than deleting a
   more valuable substitute. Wording may advance on several rows; no row
   advances twice at the boundary. `forget_pending` persists unmet
   low-water pressure across boundaries and checkpoint load, including
   below high water. The interval still applies. Spec §42 gives the
   rationale; the two-pass coarsening test and new restore regression
   check the behavior. The native turnover test's assertion port is §49.
2. **Questions:** `accept_fact` still rejects the `question` and `estimate`
   record kinds. The new storage regression checks rejection, empty
   question evidence and mode-column absence before and after restore;
   an ordinary occurrence can still be accepted.
3. **MODES:** the retired LTM0 enum's consumers were its raw read, sidecar
   validation, migration, writer and fact admission, plus P1's stored-mode
   assertion. They now use the existing kind/polarity fields. Active code
   has no `LTM0.MODES` consumer. `ThoughtFeatures.MODES` describes live
   parse features and stays. Checkpoint compatibility only discards the
   old `grammatical_mode` key; no replacement column exists. `reasoning.py`
   no longer tests a fact's stored grammatical mode.

## Complete port record

The [exact assertion list](assertion-ports.md) contains each before/after
check, including parameterized evidence tuples, snapshot checks, staged
forgetting and the copied measurement probe. The [fixture ledger](fixture-ledger.json)
lists every changed function with its full diff, including changes that
leave the assertion text intact (notably §40's operand loop, P5 owner
selection and the production-path presentation fixtures). The whole test
delta is [test-ports.patch](test-ports.patch). The archived observer files
stay unchanged; their active copies' exact ports are in
[copied-observer-ports.patch](copied-observer-ports.patch).

P1 ports the embedded question check to its record kind and evidence. P3
removes the obsolete snapshot switch and retires exactly these three
adjacency-only tests:

- `test_operators_final.py::test_centroid_moves_without_losing_identity_and_caps_containment`
- `test_operators_final.py::test_adjacent_wholes_rewitness_compact_and_checkpoint`
- `test_review14_contracts.py::test_absent_adjacent_wholes_leave_the_lower_form`

P5 replaces one posting-corruption subcase with the two approved fresh
Part-checkpoint corruptions (invalid positive handle; wrong pair shape).
It retires no test. The fixture selects the actual exported concept owner;
both approved exception messages remain. Malformed tags wait for removal.
P4 here is only absence of unused `refs` and snapshots of the actual Part
arguments, by existing handle when new edges may append. P2, tagged P4,
P6 and P7 remain deferred.

The exact completed-S counts are also in the assertion ledger: attribution
0→2; implication 0→3; relative observation 0→1; embedded Part 0→2;
bare-NP implication `[0,1]`→`[0,1,2]`; equality 0→1; taxonomy 0→1;
relative Part 0→1; nested observation 0→2; referenced phrase 1→2;
repeated relative observation/estimate 2→3; truth-text ideas `[0]`→`[0,1]`.
The operand certificate changes `len(unknown)` to
`len(unknown)+1+int(nested)` (its six values are in §40); its three-slot
case changes 2→3. The composite-operand closing changes 0→2. Remaining exact count ports are listed in the linked
ledger, not inferred from totals.

Separately, captured comparisons with the pre-2b writer diagnose three
already-stale Part counts: 10→6, 14→15, and `len(words)+2`→`len(words)+1`
(§45, [both inventories](preexisting-counts.json)). Three Part-read cases
are ported to the already-decided retirement of the reader-side binder
(§46): equality evidence alone leaves a name pending without a resolver
choice. Actual selected arrival and MTM content checks stay live.

## Failures and evidence retention

Each development directory has its command, source patch hash, console
log, bounded-test report and initial captures. These are diagnostic work,
not promoted to frozen-source evidence.

- `development-01`: the first storage run; retired-reference consumers,
  count/fixture ports, deep traversal, retired read binding and time-audit
  inventory. All 22 failed cases remain in its report and captures.
- `development-02`: collection failed on an incorrect file selector; no
  test ran. The selector is corrected in subsequent runs.
- `development-03`: six failures, including the capture helper's deep
  recursion, native stage expectation, P5 owner selection, recall batch
  leak and obsolete archived observer inputs.
- `development-04`: two remaining failures: the real provisioning recall
  leak and the copied observer's hard-coded width.
- `development-05`: scoped follow-up passes after §§53–54, with one
  existing prerequisite skip.
- `development-06`: remaining native-file development check passes, with
  three existing prerequisite skips. This is not a twelve-draw series.

- `development-07`: four module-fixture failures; §55 records the count,
  index, native-concept and initial journal dispositions.
- `development-08`: the other three ports pass; the journal probe still
  fails, refuting its initial diagnosis. §56 corrects the setup and
  restores its original assertion.
- `development-09`: all nine journal cases pass with the corrected setup.

The first failed historical-writer comparison and the successful corrected
comparison are both retained, as are the recall diagnostic's state and
batch log. Large tensor captures use byte-preserving gzip, with original
and compressed hashes in [packed-captures.json](packed-captures.json).
`gzip -dk path.pt.gz` restores the exact original bytes; no tensor is
reserialized and no failed evidence is discarded.

## Freeze and gates

The [source audit](source-audit.json) pins changed production, configuration
and test files; [source-diff.patch](source-diff.patch) records their tracked
delta. New sources are included in the frozen commit. `run_gates.py`
checks their hashes and a clean source tree before and after each group.
It runs the regression files, the eight-file set (the full eighth file,
including the bootstrap snapshot), and the changed modules' direct tests,
without duplicate selectors. Final results belong only to that source.
The full sweep remains at step 2's end; there is no new fresh series here.

After the frozen gates, Claude reviews this source and receipt before
publication of BasicModel main and the WikiOracle gitlink bump. Identity
threading needs Claude's new spec; the next action is a dated code review
of §§34–35, not implementation. Item 3.0 has not been reached.
