# R10: the named token exists in the preview — 2026-10-21

Authority: Alec's decision in [reconstruction spec §15](../../plans/2026-10-21-reconstruction-reads-the-bind.md#15-claudes-review-of-line-1-and-of-the-delegated-review--2026-10-21)
and the subsequent handoff. Line 1 was published first at BasicModel
`a514a71e0` (source `3a9db635e`, receipts `63c600712`) and WikiOracle
`b49960a`, without additional gates.

The own-object preview record now retains its site's content. Its anonymous
flag, pending name, kind and kept-close write rules are unchanged. MTM is
still untouched by preview and prepare. The production change is confined
to that record's construction in `ReferenceClose.preview`.

The new regression requires one name a nonempty, order-one reading through
`reference_orders`, as spec §13 allows. Two participant positions mention
that name in the same sentence. Its second mention must bind greedily to the first's
alias, with positive coverage features and zero registered reference cost.
The kept close writes one anonymous token with both occurrences. The next
sentence must read the same content, with positive features and zero cost.
The name remains pending until an arrival refines it. No existing test or
assertion is changed.

After freezing, run the bind-thought, reference-close, arrival, participant
and reference-term files once on that hash, retaining their captures and
any failure. These include the new gate and existing arrival refinements.
No fresh pending-premise series or sweep: x in that certificate has no
description. Then a read-only Claude review precedes publication. No
learning claim is made. Results and the source hash follow below.

## Frozen result and stop

Source: `58722725d50ab5d009c98f99c27e93c39458dac0`. The one requested
run completed **110 passed, four existing prerequisite skips, one failed**.
All existing assertions pass unchanged. The new gate fails at line 145,
its next-sentence selection, after all same-sentence, zero-cost, isolation,
anonymity, pending-name and kept-close checks pass.

| File | Passed | Prerequisite skipped | Failed |
| --- | ---: | ---: | ---: |
| Bind thought | 44 | 0 | 0 |
| Reference close | 22 | 4 | 0 |
| Arrivals | 16 | 0 | 0 |
| Participants | 16 | 0 | 0 |
| Reference term | 12 | 0 | 1 |

The [gate ledger](gate-summary.json), full logs, initial states and resolver
menus are retained. The failure's next menu is empty; it chooses its own
object, not a wrong antecedent. Its first-sentence second mention chooses
`('new', 0)` greedily: features `[1, 1, 0, 0, 0, 0, 0, 0]`, candidate logit
1 against dummy 0, candidate probability 0.7310586, reference cost zero.

The [frame diagnosis](frame-probe.json) restores the captured RNG and
executes the unchanged gate to the same assertion failure. My fixture
builds the next frame with `SituationFrame.from_row` alone, losing the two
occurrence activations from the kept close. Production passes these through
`SituationAddress` (`Models.py`'s `bind_observation_occurrence`, then
`SituationAddress.read`). Reading the same close with its actual metadata
selects the introduction, with the same positive coverage features and
zero reference cost. Rows, bank and MTM are unchanged by these reads.
This diagnostic is not a passing gate and does not change the failed result.

Recommended fixture correction: use `SituationAddress` with the close's
actual `symbols` and `activations` when constructing the next frame. Every
assertion and all production code stay unchanged. The standing handoff
says any failure under forced order stops for Claude and Alec, so that
correction has **not** been applied. The dated reconstruction-spec section
records the stop. Claude's read-only review confirms the production change and fixture
diagnosis, and recommends holding this failed hash pending Alec's
authorization of the fixture correction. R10 is not published; step 2 has not
started. No fresh series, sweep or learning claim.

## Claude review

[Claude's completed review](claude-review.md), model `claude-opus-5-5`,
finds the production hunk sound and independently confirms the missing
frame activations. Verdict: **do not publish `58722725d`**. Correct the
fixture only after Alec's disposition of the forced-order stop, freeze,
rerun the five files once and obtain review before publication. The
[exact proposed patch](proposed-fixture-correction.patch) is retained and
not applied; every assertion stays verbatim. The review's optional extra
assertion is not part of the proposal.

The reviewer notes that the diagnostic does not verify the later record
value equality assertion; the unchanged gate must reach and pass it after
the fixture correction. Two existing arrival-plan observations are
recorded in the review, with no R10 repair requested: the same-close joined
site's no-op refinement before uses are attached, and the next read's
preview-only arrival. Arrival items remain outside the reference term.

The full review request, stream, stderr and execution metadata are retained.
No production, test or configuration changed after the freeze. Alec has
been asked to approve the reviewed fixture correction; work remains at
R10, before step 2 and before item 3.0.
