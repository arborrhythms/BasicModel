# R10 fixture correction — 2026-10-21

Authority: [reconstruction spec §17](../../plans/2026-10-21-reconstruction-reads-the-bind.md#17-claudes-review-of-the-r10-stop-decided-alec-2026-10-21)
and Alec's handoff. The retained proposed patch is applied exactly: the
next-sentence frame reads a `SituationAddress` carrying the close's written
symbols and activations. Every assertion is verbatim, with none added.
Production remains the R10 hunk from `58722725d`.

The earlier [failed run](../2026-10-21-item3-5-r10/README.md) stays failed
and retained at its own hash: 110 passed, four prerequisite skips, one
new-fixture failure. Its diagnostic and Claude review identify the missing
frame metadata; they are not promoted to passing evidence.

Freeze this correction, then run the same five scoped files once on that
hash: bind thought, reference close, arrivals, participants and reference
reconstruction. The new gate must reach its final assertions, including
same-value equality (old line 147, now line 150), zero reference cost and
read isolation. No fresh series or sweep. A new Claude review precedes
publication to both mains and the WikiOracle bump. No learning claim.

Spec §17's two arrival-plan notes are carried explicitly into step 2c.
No arrival-plan behavior changes in this round.

## Frozen-source result

Frozen source: `23a41540f8cdc3f02e3a877e23474a1e05072617`. The one
requested five-file run passes **111 cases**, with **four existing
FutureWork prerequisite skips** and **no failures**. The old failed run
remains retained at `58722725d`; no result is promoted from it.

| File | Passed | Existing prerequisite skips |
| --- | ---: | ---: |
| Bind thought | 44 | 0 |
| Reference close | 22 | 4 |
| Arrivals | 16 | 0 |
| Participants | 16 | 0 |
| Reference term | 13 | 0 |

The [gate ledger](gate-summary.json) records the new gate passing through
its final assertion. Original lines 145–151 are now 148–154: the next
mention selects the introduction, has nonzero features, reads exactly the
same record value (line 150), costs zero, and leaves rows, bank and MTM
unchanged. Both captured feature vectors are `[1, 1, 0, 0, 0, 0, 0, 0]`;
both menus give candidate logit 1 versus dummy 0 and candidate probability
0.7310586. Both reference costs are zero. All 39 assertion/check nodes
in the file are unchanged, with none added. Production and configuration
are unchanged from `58722725d`.

Full initial-state and resolver captures, commands, logs and outcomes are
retained. No fresh series or sweep was run. Claude's read-only review of
this frozen hash precedes publication.

## Review and landing

[Claude's completed review](claude-review.md), model `claude-opus-5-5`,
verdict: **publish `23a41540f`**. It verifies the exact approved patch,
unchanged assertions, production frame path, final record-value and
isolation checks, and the scoped evidence. No blocking finding remains.
The review request, full stream, stderr and execution metadata are retained.

The plan's diagnostic `cost` of 1 is not the reference reconstruction term;
the registered term is zero in both cases, as the gate asserts. The two
arrival-plan observations remain for 2c. The correction used Alec's explicit
approval, not the new standing exception. All earlier failed runs remain
retained, and no gate or sweep is repeated at publication.

Publish this source and these receipts to BasicModel main, then bump and
publish WikiOracle main. Step 2 starts afterwards with the class-path check
and separate rename commit. No learning claim.
