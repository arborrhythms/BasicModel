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
