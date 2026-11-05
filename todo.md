# TODO

> **Keeping this file useful (Alec, 2026-09-19).** This file lists what is
> left; it is not a log. A landed item is one line with its commit hash under
> "Done"; its evidence and narrative live in the document the line links to
> ([Testing](doc/Testing.md) for receipts, the topic doc for the design). An
> item leaves the list when its exit criterion is met, not when work on it
> starts. When a spec or plan leaves residue, that residue is a line here or
> it is lost. Reconcile this file at every landing, before the commit.

## Countdown to the full training session

Items count **down**: the next item to do has the highest number and **item 0
is the full training session**. (They are bullets, not a numbered list, because
Markdown renumbers a numbered list upward.) The order is the intended order; an item may
be taken early when it does not depend on a higher-numbered one. Codex builds;
Claude writes the specs and reviews each landing (Alec, 2026-09-21). "Done"
lines below keep the numbers their items had when they landed.

**Current (2026-10-29):** Items 3.5, 3.0, 2.9 (BasicModel `17e9454763`)
and 2.8 (`a8a09a6acb`) are LANDED. Item 2.7 is closed code-complete at
review freeze `472f5000850e` ([record](doc/plans/2026-10-27-item2-7-corpus.md)
§47); its carried work and November landing round remain below. Item 2.5
round A starts from the commit of that reviewed tree
([thinking spec §8](doc/plans/2026-10-27-item2-5-thinking.md)), including
D8's graded answer cost. **Update 2026-10-30:** round A LANDED at
`3718d9d0dc` ([spec §21](doc/plans/2026-10-27-item2-5-thinking.md); D9 masks `intuit` until round B);
round B and the one-backward round run as one Codex text (spec §23) under
spec §22: `intuit` a top-down Sigma over Pi with its own weights trained by
output, `ActivationPredictor`'s prediction the bottom-up spread replacing
location diffusion, D7 to future work, the cautious §16 package. Hand-off for a new conversation:
[2026-10-29](doc/plans/2026-10-29-handoff-2-5.md). No `RUN_SLOW` before
November.
**Update 2026-11-03:** item 2.5's completion round is in review
([thinking spec §41](doc/plans/2026-10-27-item2-5-thinking.md)). Item 2.0
below, from the 2026-11-03 decisions
([spec](doc/plans/2026-11-03-gradient-projection.md)), follows it.
**Update 2026-11-03 (evening):** every item above 2.5 (9, 8, 5.5, 4, 3.5, 2.9,
2.8, 2.7) collapsed to a Done line after an audit of its remaining work; the
blocks as they stood are in
[DevelopmentHistory](doc/DevelopmentHistory.md#todo-items-above-25-as-recorded-until-2026-11-03),
and the remaining work is in items 2.0, 2, 1.5, 1 and 0, each marked
"Carried ... 2026-11-03". The §41 return is reviewed in
[thinking spec §42](doc/plans/2026-10-27-item2-5-thinking.md).
**Update 2026-11-04:** the §43 return is reviewed in
[thinking spec §44](doc/plans/2026-10-27-item2-5-thinking.md). Alec decided
§42's (a) to (d): the decomposition teacher retires in 2.5 with the chooser
at the fixed residual argmin; the uniform fraction stays .5; a whole is read
at its presence; and, with (c), one rule for evidence, "evidence belongs to
its source" ([Architecture](doc/Architecture.md); the alternatives in
[FutureWork](doc/FutureWork.md#belief-across-the-epoch-boundary-perseverance-and-testimony-alec-2026-11-04-revisit-if-necessary)).
The Codex text is §44.1: items 65 to 81, then the freeze.
**Update 2026-11-04 (evening):** the §45 return is reviewed in
[thinking spec §46](doc/plans/2026-10-27-item2-5-thinking.md). The repairs are
built, with a regression in the presence read's inverse, three gaps in the
evidence rule and a byte axis sized for a sentence rather than a word.
XOR_grammar fit 5 failed because its read-back decodes the last training root
against a dictionary that moved in the final update, where the walk policy was
never trained. The Codex text is §46.1: items 82 to 91, then §44.1's 71 to 73.
**Update 2026-11-04 (night):** the §47 return is reviewed in
[thinking spec §48](doc/plans/2026-10-27-item2-5-thinking.md). The repairs are
built. Fit 5's held names keep their first-sight meanings while the dictionary
moves. Alec decided that MTM holds references to STM and LTM and computes
identity from them: names now (item 96), the full form in item 2.0
([memory spec §9](doc/plans/2026-11-03-item2-0-behavioral-chains.md)). A
headless composition's identity-0 alias was found in passing (item 94). The
Codex text is §48.1: items 92 to 97, then 71 to 73. For item 2.7: the word
surface capacity of 64 raises on 144 corpus percept runs (letter runs up to
126 bytes, high-byte runs up to 453); the corpus round constructs 512 or splits
such runs.
**Update 2026-11-05:** the §49 return is reviewed in
[thinking spec §50](doc/plans/2026-10-27-item2-5-thinking.md). Items 92 to 96 are
built. The default suite, which had not run since 3 November, fails 56 cases from
four rounds plus one memory regression; each is triaged with its repair. Alec
decided:
- `make test_all` leaves 2.5;
- six workers are allowed, recorded, if ten cross the memory ceiling;
- compile-guard raises are repaired at their producer;
- an identity refers to its known individual and to a kept thought's idea;
- a description reads its own noun phrase;
- MTM associates only on maximal overlap, now
  ([memory spec §9](doc/plans/2026-11-03-item2-0-behavioral-chains.md)).

The Codex text is §50.1: items 98 to 108, then 71 to 73, closing 2.5. Claude
rebased `doc/DevelopmentHistory.md`'s 146 links (`4502b1c566`).

**Update 2026-11-05 (evening):** the §51 return is reviewed in
[thinking spec §52](doc/plans/2026-10-27-item2-5-thinking.md). Items 98 to 106 are
built. Claude's suite on `e0fc4a92719e` completes with 24 failures, each ported or
repaired in the next text. Item 107, a structural chain over a word not yet learned
read as that word's definition, closes 2.5 with the recorded single-writer contract
for definitions. Alec placed trust in a new item 2.3: a concept's trust is its
magnitude, and operators need their own
([notes](doc/plans/2026-11-05-item2-3-trust.md)). By the countdown, 2.3 comes after
2.5 and before 2.0. The Codex text is §52.1: items 109 to 115, then 71 and 73,
closing 2.5.

**Order and branches (Alec, 2026-11-03):** work happens on main, with no
sandbox branches; the countdown order stands, 2.5, 2.0, 2, then 1.5, 1 and 0;
when item 2.5 completes, everything is on main, committed and pushed, then
each sub-repository is synced and the WikiOracle pointers are bumped and
pushed. `nanochat` carries no changes of ours (Alec, 2026-11-03): its fork
is upstream of 2026-02-19 plus Alec's two commits (the subpath `API_URL` fix
and the training-artifacts ignore), the recorded pointer `7be4dc3`; the local
checkout carries upstream to 2026-07-03 with the ignore re-applied and lacks
the `API_URL` fix, so its sync is a reset of the local checkout to the
fork's commit `7be4dc3` (Alec, 2026-11-03), with nothing pushed and the
pointer unchanged.

**History (2026-10-24):** Item 3.5 is LANDED. Iteration 1 published at BasicModel
`f306fc28aa`, WikiOracle `319a793`; iteration 2 landed 2026-10-24 at
BasicModel `75345f891e` (published main
`4f136f76faabafc24397765f4ffe90d01d3dc4bc`, WikiOracle
`69f82f5f77a3980020841fe859a09afb7733e7be`) after Claude's LAND review
([iteration spec §62](doc/plans/2026-10-23-verb-and-modal.md#62-claudes-land-review-of-75345f891e--2026-10-24)):
the second repair's five scoped groups pass **588 cases with 41 skips and
zero failures**; the one end-of-3.5 sweep (`0bd65c5e2c`, 35 failures) is
repaired, nothing retired. [Receipt](doc/benchmarks/2026-10-24-item3-5-iteration2/sweep/repair-02/README.md).
History note (2026-10-25): history after `f306fc28aa` was rewritten before the push (`537523ae04`); `75345f891e`, `a34325c85d` and `f095569a1f` are respectively `2067dbc25a`, `3561f39aa0` and `31c3e4a59e` on origin.
Learning remainders (line 3's description 1/4, the modal corpus 0/4, the
six placements 0/6) are item 0's with their budgets; layout/removal, STM
chunks, overlay retirement and the certainty record are item 3.0's.
**Item 3.0 steps 1–3 are committed (Alec’s hand-off); step 1 is approved** ([Claude's review, layout §13](doc/plans/2026-10-23-layout-and-removal.md#13-claudes-review-of-step-1-source-34848e154f--2026-10-25), 2026-10-25):
layout/removal source SHA256 `34848e154f0fa9912223d4117136d7eb63657cefaeca94a76c6574559a73819e`,
**911 passed, seven existing prerequisite skips, zero failures**.
[Receipt](doc/benchmarks/2026-10-25-item3-0-layout/README.md) records the initial
freeze's diagnostic failure, exact repair, ports and explicit retirements.
**Step 2, placement, is approved and committed**
([Claude's review, placement §21](doc/plans/2026-10-25-item3-0-step2-placement.md#21-claudes-review-of-step-2-source-1e94238b57--2026-10-25)):
source SHA256 `1e94238b57c99ba2b1003de2467e6f6acf543b68dc425263bed708509677d55e`,
**1,003 passes, seven standing prerequisite skips under the documented protocol**.
The [local receipt](doc/benchmarks/2026-10-25-item3-0-placement/README.md)
(README SHA256 `a2342de738bd35354ed5ad26c2ffd996c024226e8887e2643b83fb3c31ec8dc2`)
retains two original standing-fixture failures, their explicit measurement ports,
the transient-cache observer port and the resource stop. The pre-existing
unforced compound-case shape defect is retained in that receipt. Step 2 replaces
the interim context update. **Step 3's repair is approved**
([Claude's review, STM-chunks §27](doc/plans/2026-10-25-item3-0-step3-stm-chunks.md#27-claudes-review-of-the-repair-source-4b7fa2aaf7--2026-10-25-approved-with-two-items-before-the-commit)),
source `4b7fa2aaf7`: 1,113 passes, seven standing prerequisite skips,
32/32 complete premise readings. Its two closing items are recorded in
[§§28–29](doc/plans/2026-10-25-item3-0-step3-stm-chunks.md#28-closing-replay-the-companion-residual-is-learning-side--codex-2026-10-25):
placement's greedy coverage is also 1.0, so the **6/8 companion flag is
item 0's learning work**; estimates now carry zero time without consuming
an arrival tick, while current time remains a predictor input.
Closing source `cd49a1e84534d6c92f074276e2cfc57fccd1226a876bad4dce5905c6d6b3107d`
passed **1,113 cases with seven standing prerequisite skips and zero test failures**, plus one smoke ([closing receipt](doc/benchmarks/2026-10-25-item3-0-stm-chunks/closeout/README.md)).
The rejected source, failed percept certificate, first repair freeze and
[approved repair's results](doc/benchmarks/2026-10-25-item3-0-stm-chunks/repair-resumed/final/README.md)
remain separate. Step 4 source `44ea10180c` was rejected in §17: silence
entered the sentence keep comparison. Alec's decided repair excludes silence
from A while retaining its output training. Repair source
`acac0b3448f5a265f62a64bab6d0368563df0ab1ecb15d76d4990428ddc9d1f1`
finishes **1,490 passed, one failed, eight standing skips**, plus the smoke
([spec §§18–19](doc/plans/2026-10-25-item3-0-step4-retire-mode.md#18-asked-rows-alone-contribute-to-the-keep-comparison--codex-2026-10-25),
[repair receipt](doc/benchmarks/2026-10-25-item3-0-retire-mode/repair-04/README.md)).
Step 4 is approved as a mechanism round in
[§20](doc/plans/2026-10-25-item3-0-step4-retire-mode.md#20-claudes-review-of-the-repair-source-acac0b3448--2026-10-26-approved-as-mechanism-one-port-for-alec).
The approved two-stream port closes gate 6: all premise readings are nonempty
and both first premises are full; its file and both decomposition arms pass
6/6, with companions 6/8 ([§21 and receipt](doc/benchmarks/2026-10-25-item3-0-retire-mode/repair-05/README.md)).
**Step 5, orders learned**, is approved in [orders §19](doc/plans/2026-10-26-item3-0-step5-orders-learned.md#19-claudes-review-of-step-5-source-c4a1d92e3f--2026-10-26-approved-one-fix-before-the-sweep) on final
source `c4a1d92e3fc27decc669322a507b362dd615556e89041331fe1259116584ab95`:
1,647 passed, ten existing prerequisite skips, zero failures, plus the native
smoke; companions remain 6/8. The failed first freeze is retained separately.
[Spec §18](doc/plans/2026-10-26-item3-0-step5-orders-learned.md#18-final-scoped-measurement-and-review-hand-off--codex-2026-10-26),
[receipt](doc/benchmarks/2026-10-26-item3-0-orders/README.md).
**Item 3.0 LANDED — 2026-10-26**, approved in [orders spec §69](doc/plans/2026-10-26-item3-0-step5-orders-learned.md#69-claudes-review-of-the-serializer-round-and-the-item-30-sweep-test-freeze-ee7ca4e920bc-model-6fe273d2f12a7--2026-10-26-approved-for-landing-three-items-carried). The landing names step 5 `c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The learning records stay at item 0 at their listed budgets; the four runtime failure groups move to item 1 as named below; the unforced 1-to-2 raise certificate for “Felix is a cat. Ciao is a cat.” moves to item 3's corpus round. The [serializer receipt](doc/benchmarks/2026-10-26-item3-0-serializer/README.md) and [sweep receipt](doc/benchmarks/2026-10-26-item3-0-sweep/README.md), including repair-01, repair-03 and repair-04-slow-budget, remain local and gitignored. The complete default sweep retains 6,386 passes, 324 skips and one permitted XPASS; final model repair retains its smoke, 1,812 scoped passes and ten standing skips. The ten budget-policy checks and reduced stability replay pass. Companions remain 6/8; no complete passing slow sweep is claimed. No new test round accompanies landing.
[Handoff](doc/plans/2026-10-24-handoff-3-0.md).

**Historical step 1b review ledger (superseded by Current).**

Step 1b's native identity candidate `027ce224d` and STM1 candidate
`dde893527` are on
`codex/item3-5-first-pass`; its
[original receipt](doc/benchmarks/2026-10-20-item3-5-identity/README.md)
records the failed sweep and scoped repairs. **The migration decision is
resolved:** Alec retired stage3, np_definite and np_description. They are
unregistered and archived beside their producing receipts, never to be
read; their certificates are prerequisite-skipped until their producing
lessons are next needed with fixed budgets/seeds. Rejection of retired
coordinates is permanent. The
[STM1 follow-up](doc/benchmarks/2026-10-20-item3-5-stm1/README.md) records
the retirement and removal of the blanket cycle restriction.
**Step 1b landing remains:** the §6.10 close/read repair and saved
reverse-order replays pass. Both decomposition lengths pass in both arms
in [attempt 124](doc/benchmarks/2026-10-20-item3-5-stm1/decomposition-124.md),
all eight companion questions correct. The one full sweep completed 5,965 cases with 219 failures; the
[sweep repair record](doc/benchmarks/2026-10-20-item3-5-stm1/landing.md) tracks
218 passing scoped cases or declared ports and the capacity repair now
recorded in [review 3](doc/benchmarks/2026-10-20-item3-5-stm1/review3/README.md).
The native-119 replay passes with the original eight seats and budget.
Saved-state paired-reconstruction evidence remains separate. The
[pool-capacity question](doc/plans/2026-10-19-bind-thought-operation.md#10-landing-follow-up-a-full-concept-row-pool-codex-2026-10-20)
is decided (plan §10.1: ordinary referent stays an unminted LTM1
occurrence; mint only relation arguments at the close or by recurrence;
unseatable mint fails loudly; no capacity masking). Claude's third review
(plan §12) adds one blocking finding: the pending-premise certificate
(§14 e, both orders) fails at fresh initializations. Exact captures now
reproduce it; named-head, lexical-key and composed-marker repairs and the
working-read certificate port have passing regressions. The
[live-exploration contract question](doc/benchmarks/2026-10-20-item3-5-stm1/review3/reverse-scope-question.md)
has the D1 probe above; D2 is decided but queued, and the
six-fresh-per-order gate remains open. Complete A without weakening
the assertion or selecting passing seeds, then the
[§10.2 round](doc/plans/2026-10-19-bind-thought-operation.md#102-decided-alec-2026-10-20-later-the-determiner-prepares-the-token-for-identity-in-stm)
(Alec, 2026-10-20: the determiner's `lower` prepares the token to be
bound by identity in STM; no concept row and no edge for the token, the
lowering is inline; fixture back to 8 rows; the capacity certificate at the original inventory), then publish both
mains with the October 20 docs and a clean BasicModel tree.
*Fourth review (Claude, 2026-10-20 night, plan §13):* B accepted as
scoped (`mm-05` at eight seats; the 8192 fixture withdrawn); A's three
mechanism repairs and the line-422 working-read port accepted; A stays
open on the keep probe for the forward line-391 case (step-0 precedent:
repair the judge, not the certificate), Alec's scope decision for the
reverse line-398 assertion (recommended: no row keeps a forward
reference naming x), then six fresh initializations per order.
*Fifth review (Claude, plan §14):* the D1 probe is accepted; the keep is
decided by one byte of reconstruction with A=0 by design, so it is the
contract case; recommended to Alec: D2 now, the §10.2 round next (both
failing readings are the complement mint that §10.2 change 1 removes),
the twelve fresh runs on that source, and a contract split (mechanism
assertion in step 1b, the strict reading kept verbatim at item 0) only
if failures remain. **Alec decided: §10.2 first, contract later.**
*Sixth review (Claude, 2026-10-21, plan §16):* D2, the eight §10.2
changes and the carrier follow-up accepted; D1 dissolved (both replays
tie and keep greedy; twelve fresh pass with the unchanged assertion).
The 43 scoped failures are representation ports (authorized under the
kept-semantics rule) plus one mechanism gap: a bound use is not grouped
with its antecedent; recommended: the mention's row keeps the selected
occurrence in `role_refs[0]` (object permanence by reference) and the
read follows it. **Alec decided (2026-10-21): ports authorized; the
grouping lives in MTM, a medium-term identity pool on ConceptualSpace,
independent of LTM (plan §10.3)**; then final evidence by hash on the
frozen source. *Seventh review (Claude, plan §17):* MTM accepted as
§10.3 with both merge certificates passing unchanged; one repair before
the freeze (a re-closed occurrence must detach from its previous token)
and the fourth occurrence-view port authorized; then the final gates by
hash and publication. *Eighth review (Claude, plan §18):* the frozen
source `eac2fe87b` is accepted (finding A repaired, port B applied, MTM
not implicated); reverse fresh draws 2 and 5 fail by the R-only keep
(A=0 on unlabelled premises, the ambiguity rule leaves x open under a
kept conflicting equality); recommended to Alec: force the keep in the
forced certificate and keep the live-keep form verbatim at item 0.
**Alec decided instead (plan §10.4): `bind` is the only binder at the
arrival too; the reader's equality rule and ambiguity clause go; a
named site with no candidate opens an anonymous MTM token that the
arrival's re-run bind refines; forced like any operation in the
certificate; assertions, keep law and budget untouched.** The
[current gate/carry ledger](doc/benchmarks/2026-10-20-item3-5-stm1/status.md)
records completed scoped work: transient full-width STM1 and close
dissolution; shared six-column Part store; cube-coordinate maps and
inverse checks; Part-derived exact bootstrap and saved/random paired
reconstruction; historical saved-state §14 checks; predictor retirement;
G1–G3 and reference-close. These are not a completed landing.
Alec's §6.6 answers withdraw the exact-leaf inversion and width gates;
identity computed in STM is never written back into LTM. All-LTM2
integration is implemented in this step, including the expression reader. The per-dimension withheld mark belongs in concept content
beside the lanes at the row conversion.
Production keeps the prime ON and uncapped,
P/S/R/D=0 and φ=.5.

*Ninth review (Claude, plan §19):* the §10.4 core and its follow-up
repairs accepted; three repairs before the next freeze: bind is priced
by the failure to understand like every chooser operation (Alec: the
closing reference credit leaves A); refinement merges the pending token
into the chosen antecedent's token; the bootstrap zero-R failure is a
regression of the carrier hunk in `67c5b203f` (replay-bisected), to be
repaired. Alec, later: every noun adds an object to MTM automatically;
binding links it to an existing object or leaves it its own (a zero-logit
dummy), learned like any choice; the pending-premise certificate's
closure forcing is removed (no wrong binding is asserted; the strict
closure assertions move verbatim to item 0). Then every gate on the new hash.
*Tenth review (Claude, plan §20):* frozen `575cab06c` verified (carrier,
regressions with the six authorized ports pending, bootstrap). Alec: a
joined site is never re-bound; incompatibility makes the mention a new
identity; re-identification is FutureWork; thought binds follow decision
A; "reconstruction reads the bind" is the recommended next round after
step 1b. Repairs: nested-role answer view, kind adoption on merge.
*Eleventh review (plan §21):* §20 implemented and accepted; the series
stopped on a wrong binding (x to the space: markers were bind candidates).
Alec: bind is offered only on words the final parse identifies as nouns,
and only nouns are candidates (no ID → NP rule; identity stays an option);
the seven retained checks get the authorized dispositions (two thought
ports, one count port, four partition assertions to a re-identification
certificate under FutureWork). Then a new freeze and every gate.

**Naming sweep at the step 1b landing (Alec, 2026-10-21: "rename when
you think best"):** LTM1 → LTM0 (the occurrence store: order 0, no
identity), LTM2 → LTM1 (relations), the stale LTM3 retired; LTM1 is a
stratum of the LTM0 store (Alec), LTM2 = relations over relations;
renamed in place in the live contract docs, a naming line in the other
docs' amendment block, code comments by Codex in the same landing.

**Necessary deferral, accepted after evaluation:** the copular-identity
certificate needs its producing lesson. Item 3.5 owns stage 4e → NP
continuation → copular identity → artifact registration and unchanged c
certificate. No producing artifact exists; the October 18 decision withdrew
the proposed one-equality-hop reader patch in favor of this phase gate.
Graded versus shuffled (§9.6), derivation-optimality, long-run collapse and
the original failed native learning run remain at Item 0. Independent 5.5
learning gates remain separate. No extra training or regrade closes them. The
optional cold-start boundary learning certificate is also incomplete:
[native-117](doc/benchmarks/2026-10-20-item3-5-stm1/landing.md) hit its 901-second
runner bound during the original six epochs. Carry it to Item 0's learning
evidence with that budget unchanged; it is not a step-1b mechanism pass.

**Active mechanism:** Item **3.0**. Step 1 is approved
([layout §13](doc/plans/2026-10-23-layout-and-removal.md#13-claudes-review-of-step-1-source-34848e154f--2026-10-25));
step 2, placement, is approved in §21 and committed
([receipt](doc/benchmarks/2026-10-25-item3-0-placement/README.md),
[spec §21](doc/plans/2026-10-25-item3-0-step2-placement.md));
[step 3's repair is approved in §27](doc/plans/2026-10-25-item3-0-step3-stm-chunks.md#27-claudes-review-of-the-repair-source-4b7fa2aaf7--2026-10-25-approved-with-two-items-before-the-commit).
The closing replay moves the 6/8 companion residual to item 0; the decided
estimate now has zero time and consumes no tick. Closing source
`cd49a1e845` passes 1,113 cases with seven standing skips; both regressions, the unported clock test and the percept certificate pass ([receipt](doc/benchmarks/2026-10-25-item3-0-stm-chunks/closeout/README.md)).
Step 4 is approved in §20, and its approved two-stream port closes gate 6
with 6/6 scoped passes and companions 6/8
([receipt](doc/benchmarks/2026-10-25-item3-0-retire-mode/repair-05/README.md)).
**Step 5, orders learned**, is approved in orders §19 on `c4a1d92e3f`:
1,647 passed, ten existing skips, zero failures, plus the smoke; companions 6/8
([spec §18](doc/plans/2026-10-26-item3-0-step5-orders-learned.md#18-final-scoped-measurement-and-review-hand-off--codex-2026-10-26),
[receipt](doc/benchmarks/2026-10-26-item3-0-orders/README.md)).
**Item 3.0 LANDED — 2026-10-26**, approved in [orders spec §69](doc/plans/2026-10-26-item3-0-step5-orders-learned.md#69-claudes-review-of-the-serializer-round-and-the-item-30-sweep-test-freeze-ee7ca4e920bc-model-6fe273d2f12a7--2026-10-26-approved-for-landing-three-items-carried). The landing names step 5 `c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The learning records stay at item 0 at their listed budgets; the four runtime failure groups move to item 1 as named below; the unforced 1-to-2 raise certificate for “Felix is a cat. Ciao is a cat.” moves to item 3's corpus round. The [serializer receipt](doc/benchmarks/2026-10-26-item3-0-serializer/README.md) and [sweep receipt](doc/benchmarks/2026-10-26-item3-0-sweep/README.md), including repair-01, repair-03 and repair-04-slow-budget, remain local and gitignored.
3.5 landed 2026-10-24 at BasicModel `75345f891e`. *3.5's record:* it ran on main after
Item 4 publication, in the order [Alec's first-pass plan](doc/plans/2026-10-19-item4-start.md#next-on-codexs-plate-after-item-4-lands-item-35-first-pass-alec-2026-10-19)
gives: bind as thought; LTM1/LTM3 and the row amendment; stage 4e and NP
continuation before identity; verb group element; tense/aspect and second
difference; then the remaining 5.5 mechanisms. Each has its own receipt.
The [bind step-1 landing](doc/benchmarks/2026-10-19-item3-5-bind/landing.md)
has completed Claude's review, one full sweep (5,907/5,907 completed;
48 failures repaired with scoped checks), and final native validation:
72 passed, one copular prerequisite skip; both decomposition lengths pass
in both gain arms with all eight questions correct. Publication to both
mains preceded step 1b's G0 and identity migration. On the earlier
identity candidate, G1–G3, the 18 reference cases and both decomposition
lengths in both arms passed on the native scorer; that candidate's full
sweep completed all 5,926 cases with 15 failures,
followed by scoped repairs of 11 failed cases. The three artifact failures
remain in that historical result; their artifacts are now retired, not
regraded as successful loads. The later STM1/Part round repairs the native
cube, final §14 read failures and expression-type defects. Its scoped
gates pass; the one landing sweep is complete and its capacity failure
remains open before publication. The historical
[disposition](doc/benchmarks/2026-10-20-item3-5-identity/landing-disposition.md)
records each case. Its
[row audit](doc/benchmarks/2026-10-20-item3-5-identity/row-audit.md) explicitly
carries the physical LTM1/LTM3 split, context wholes, spatial extent and
situated occurrence lowering into step 2. Steps 2–6 have not started.
There is no identity-learning claim: the native reference-close lesson remains
0/146 bindings, 0/20 row pairs and zero A columns.
*Status (Claude, 2026-10-20):* step 0 round 2 is accepted and published on
both mains (`bfc834d79`/`d6eb0cf3b`, WikiOracle `cb4d718`); step 1, `bind`
as a thought operation, is reviewed on `codex/item3-5-first-pass` at
`f4f11aa5b` — conforms to the bind plan §1–§5, gates G1–G3 and both
decomposition certificates pass; Alec decided it lands after its sweep and
**step 1b** migrates the scorer to [bind plan §6–§6.2](doc/plans/2026-10-19-bind-thought-operation.md)
(identity only in the major structures: the meet as the content term,
containment for the form term, no nouns dictionary, no count tables, no
ICA event chart, STM as an index) before LTM1/LTM3.
*Review of the step 1b tree (Claude, 2026-10-20, later):* step 1 is
published (`204e1be55` / WikiOracle `a89c958`, with the answer-kind judge);
step 1b is an unpublished candidate carrying the identity migration, the
retired artifacts, STM1 in progress, the `PartStore` (LTM2 and the
taxonomy as one owner), the cube chart, and model-time `.when`. Every
decision of 2026-10-19/20 is in the
[decisions ledger](doc/plans/2026-10-19-bind-thought-operation.md#7-decisions-ledger-for-the-current-round-2026-10-20-what-is-carried-and-where)
with its status and the commit it lands in; Codex's §6.7 read question is
answered in [§6.8](doc/plans/2026-10-19-bind-thought-operation.md#68-claudes-answer-to-67-the-read-rule-needs-nothing-new-2026-10-20)
(symbol types and ordered pairs suffice; no metadata on Part). Open on the
branch: the live-cube counterexample, the §14 initialization read, the
expression read and the bootstrap read, the paired-reconstruction `.1`,
reverse-order premise resolution, `IntraSentenceLayer`.

The word-level evaluator built in 6.8 waits for the trained checkpoint;
acceptance does not authorize fresh-model bulk scoring. Every item above 2.5
collapsed to a Done line on 2026-11-03 after an audit of its remaining work;
what remained of them is in items 2.0, 2, 1.5, 1 and 0 below, with the
empirical gates of items 9 and 8 at item 0.

Work follows
[2026-09-15-next-sentence-as-the-production-objective.md](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md)
§10 and its completion gates. Current contracts, limits and receipts are in
[SelectedMeaning](doc/SelectedMeaning.md),
[KernelRetirement](doc/DevelopmentHistory.md#frame-kernel-test-migration),
[ExpectationRetention](doc/ExpectationRetention.md) and
[Testing](doc/Testing.md). The September 17 handoff and preserved candidates
are in [the checkpoint](doc/checkpoints/2026-09-17-production-spec/README.md).

**Publish rule (Alec, October 11):** repair → one regression per defect →
scoped measurement → candidate commit on a branch → **Claude's code review**
→ resolve review → one full sweep at landing → BasicModel push → WikiOracle
bump/push, with the co-author trailer. Candidate commits are authorized before
review; their commit hashes are the archive. No tarballs, manifests or
integrity checks. Receipts contain gate numbers and one table per question;
diagnostics stay in files.
Natural word →
operator associations use the existing compose/generate grammars. Mechanism
probes do not satisfy learning gates, and a null result is recorded as null.
**No test pins a random seed in order to pass** (Alec, 2026-09-21): a fixed
seed may make a *measurement* reproducible, never an *assertion* true. A test
that fails at some seed has found a defect; fix the defect, or let it fail.
Do not remove unused reasoning methods without Alec's review.

**The grammatical operations and their inverses are still in development
(Alec, 2026-09-26).** The compose, thought and generate catalogs are a
fragment, their routing is learned with sparse credit, and their tied
inverses are exercised mostly by reconstruction. No item's gate, and no
conference number, should be read as a claim about the performance of the
grammar as a set yet; a poor result on wording, parse depth or free
generation is expected at this stage and is recorded, not tuned away.

- **2.5. Thinking: derivation over chunks with trust, and `intuit`.**
   *Spec written (Claude, 2026-10-27):
   [doc/plans/2026-10-27-item2-5-thinking.md](doc/plans/2026-10-27-item2-5-thinking.md),
   from Alec's eight answers ([notes §10](doc/plans/2026-10-26-thought-design-notes.md))
   and the [`intuit` spec](doc/plans/2026-10-21-intuit.md) §5, whose
   leaf-index scan and `Mind.SERIAL` dependency it amends. D1–D4 decided
   (spec §7); the 2026-10-28 readings on training, fixed LTM, order zero
   and the pick are spec §§9–11. Two rounds: A (the serial path; Codex
   text in spec §8, starting from item 2.7's code-complete commit at
   review freeze `472f5000850e`) and B (`intuit`, which
   at round B also replaces item 2.9's location diffusion as the
   spreading mechanism, consolidating conceptual expectation and priming
   into one operator; spec §15, Alec 2026-10-28). Round A carries D8's
   graded binding cost plus text cost (2.7 record §47); 2.7's landing
   round remains in November. Decided direction (Alec, 2026-10-29), spec §16:
   one error signal and one backward: stop-gradient for attention and
   expectation, jitter of reconstruction's strength against output, and the
   chooser's log-probabilities in the same objective. Round A landed
   2026-10-29 at `3718d9d0dc` (spec §§17-21: `chunk_truth`, `.where`
   addressing, `infer`, D8, D9). Decided 2026-10-30 (spec §22): `intuit` is
   top-down with its own Sigma-over-Pi weights trained only by output; the
   bottom-up spread is `ActivationPredictor`'s prediction; no open-pass
   `intuit`; D7 to future work; round B and the one-backward round in one
   Codex text (spec §23).*
   The serial slot becomes the episode's focus chunk among the shared
   eight; `descend`/`return` follow tagged LTM1 references; operators
   `infer`, `refute`, `substitute` beside the catalogue; `intuit` proposes
   into chunks, the serial path confirms; trust computed along the path
   (minimum, maximum, lane exchange), learned only at the leaves; weak and
   strong forcing; the implicit route, pervasion over an object's parts and
   wholes in order-0 space, judged together with the explicit chain
   (notes §9). Rounds sized at spec time. Exit: the spec's gates green and
   one thought episode's records replaying a derivation with its pair.
- **2.3. Trust in operations** (Alec, 2026-11-05;
   [notes](doc/plans/2026-11-05-item2-3-trust.md)). A concept's trust is its
   magnitude. Operators need a trust value of their own, unless they exist as
   vectors, and a relation's trust composes its operands' and its operator's.
   Gates open in an order the data specifies: words first, then operations one
   at a time. A trained-model test tier (`RUN_TRAINED`) takes many grammar
   tests. Specified after 2.5 closes; nothing is built.
- **2.0. One gradient, one choice mechanism, memory as the state of
   affairs, behavioral chains and basic percepts** (Alec, 2026-11-03 and
   2026-11-04;
   [spec](doc/plans/2026-11-03-gradient-projection.md)). After item 2.5's
   completion round, before item 2, and before item 0: the gradient
   discussions Alec asked to settle "before step 0 large training", with
   the chains he placed "before step 0". Decided 2026-11-03 (spec §§1–8):
   `separateChooser` false by default, one update per sentence; output's
   gradient projected one-sidedly against reconstruction's on every shared
   unit (concept rows, definition edges, operator tensors), in place of
   jitter, which stays at 0 as an optional annealing; generate's proposal
   and credit made the same as compose's, and `walk_policy` retired; the
   decomposition teacher retired, its chooser at the residual argmin; the
   verb and modal teachers retired, verbs and modals trained by their
   participation, credit crossing sentences through MTM by the re-read,
   the verb-group assigner kept; every choice reward whole on the output
   side; the answer walk's residual rebooked to output; the derivation
   inverse given the forward's action context; the five terms evaluated on
   the model's internals kept, recorded in
   [GradientFlow](doc/GradientFlow.md#terms-on-internals). The order-zero
   loop (spec §8, "thinking is optional"): reconstruction from order-0
   concepts, understanding a thought process from order 0 to order 0,
   output the same decoder on the result; compose, think and generate are
   System 2 on System 1; MTM is the state of affairs, for thought and
   action only; `intuit()` within a step and expectation across steps are
   the order-0 operations. Reconstruction then ceases to involve the
   grammar; its build, reconstruction from order 0 with the two acceptance
   checks of spec §8, is the item's last part
   ([loop spec](doc/plans/2026-11-04-item2-0-order-zero-loop.md)):
   reconstruction is input (thinking)* output, the input reconstruction
   with no thinking step and the order-zero reconstruction after
   understanding, in which symbols take part. **Behavioral chains in MTM
   at order zero** (spec §9, [memory spec](doc/plans/2026-11-03-item2-0-behavioral-chains.md)):
   STM, MTM and LTM in one file, `bin/Memory.py`; STM a buffer of eight
   ideas and relations; MTM, over its time horizon, knits STM's and LTM's
   entries into per-order identities that maintain a state of affairs, by
   one link function with no branch on order, each collection meant as a
   connected path (a time window and likeness enforced, place and
   continuity scored): an extended event when it joins states, an extended
   noun when it joins mentions, the precursor of concepts and taxonomic
   relations; promotion writes the path (membership and successor rows); a
   recurring chain promotes to a chained event concept; a thing that
   persists through a chain is a noun candidate. **Basic percepts** (Alec, 2026-11-04,
   [spec](doc/plans/2026-11-04-item2-0-basic-percepts.md)): the last
   perceptual layer, a preallocated table sized in `data/model.xml`,
   whose percepts tie parts and wholes at their centroid, taking over that
   role from the first symbolic layer; order-0 concepts collect basic
   percepts as higher concepts collect symbols, order-0 symbols represent
   them, and symbolic space has no mereological read; the order-0 reverse
   runs through basic percepts to bytes; one WholeSpace replaces the
   per-stage copies left from the 2026-04-22 refactor. Metrics for the round (thinking spec item 60): held-out
   cost, acquisition, answers and each chooser's departure win rate, no
   chooser variance. Gates: the projection receipts per unit (cosine before
   projection, fraction of steps projected, fraction of output's gradient
   removed), the realized relative step per level (codes, verbs, modals),
   the tests named in spec §§3–5 ported with before/after, the gates of
   the loop, memory and basic-percepts specs, `make test`. Prerequisite
   (2026-11-04): the gradient spikes item 2.5's arms showed with
   `separateChooser` false, item 2.0's default (thinking spec §44, finding 6),
   attributed by 2.5's item 75 and repaired before item 2.0's measurements.
   One Codex text, written after the review of the §41 return.
   **Carried into item 2.0 on 2026-11-03 (audit of items 2.7, 8 and 3.5):**
   the supplied `yes` admission defect (2.7 record §47: 31 of 32 lesson-13
   absences are "yes" missing from the terminal bank): §38.1's three-stage
   report (admitted, shortlisted, exact route available) for every supplied
   form, with `yes` present in all applicable banks, and a certificate on
   the admission path (`Corpus.admit_training_forms` to
   `Generative.lexical_candidates`); Alec to confirm this home over item 0.
   If the sentence-82 MTM rejection (`MediumTermMemory._value`, 2.7 receipt
   `failure-index.json`, never captured, not recurred in eight later runs)
   recurs in the round's lesson-13 arms, capture the working value and pool
   state at `ReferenceClose.commit`'s deferred `mtm.bind` and repair at the
   producer; no clamp or fallback. The `XOR_grammar.xml` reconstruction gate
   (`test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`,
   four of four word multisets, unseeded; open since thinking spec §41.13,
   recommended for repair by the decomposition teacher's retirement in 2.5's
   item 65) is run before and after that retirement with the chooser at the
   residual argmin; its class-accuracy sibling passes; no seed or budget
   change. Output typing (carried from 3.5 line 7; decided, Alec, 2026-11-03):
   output types are per formal argument (`PartsOfSpeech.formal_arguments`,
   the `argout` faces beside the `argin` faces) and collapse to parts of
   speech as already described ([Language](doc/Language.md), "Distributional
   parts of speech"); the `argout` prototypes are kept per formal argument
   and read by the same occupation measure as the inputs.
   **Rounds (Alec, 2026-11-05: "after 2.5, 2.3 then 2.0, nothing else";
   "Just divide the work").** 2.0 stays one item, built in rounds, each with
   its own freeze; the loop is the last, as recorded. Already done in 2.5,
   struck from 2.0: the decomposition teacher's retirement (2.5 item 65), the
   `XOR_grammar.xml` fits around it (2.5 §52.1 item 71; 2.0 still runs the
   gate once after `walk_policy`'s retirement), and the spike attribution
   (2.5 items 74–75). Held-out cost waits for item 0's start; the five terms
   on internals stay recorded. The rounds, in order:
   1. *Behaviour-neutral moves:* STM, MTM, LTM0 and LTM1 into `bin/Memory.py`
      (whether `ShortTermMemory` moves too is decided in the text); thought
      ideas stamped and found by `.when`.
   2. *Repairs and the measured "before":* any event at or above 1e8 under
      `separateChooser` false that 2.5's item 71 reports, repaired at its
      producer; fail loud in the reverse and `runBatch`; the derivation
      inverse given the forward's action context; one WholeSpace; the
      supplied `yes` admission; the item 60 metrics and receipts recorded as
      the baseline. The sentence-82 rejection is a capture rule on every
      lesson-13 run.
   3. *One gradient:* `separateChooser` false with the one-sided projection
      and the sides (spec §§1–2).
   4. *One choice mechanism:* generate under compose's proposal and credit,
      `walk_policy` retired and the XOR gate run; the verb and modal teachers
      retired (participation, $\kappa$ in the model file); output typing.
   5. *Memory and basic percepts, one round* (Alec, 2026-11-05: "In 2.0, 5
      and 6 merged, but MTM only operative for basic level percepts (not
      parts and wholes)"). MTM's one link function, extended events and
      nouns, promotion of paths and chains, the full reference form, credit
      across sentences by the re-read, verbs over the state of affairs.
      Runs form concepts at every level (Alec, 2026-11-05): an event run's
      state is computed from its members as an identity's value is from its
      references, one order lower; events as mereological neighbours in
      world-space stay future work. MTM's link also runs over basic-level percepts, never
      over parts or wholes alone, and a recurring one promotes to a basic
      percept, which word admission creates today. The compose-time whole
      read and the per-word typed slots keep their logic and read basic
      percepts ("Why would this logic change?").
   6. *The order-zero loop*, input reconstruction then output through the
      same reverse.
   The momentum-carried projection and a learned top-down pair policy are
   built only if the receipts call for them.
- **2. Housekeeping.** Restore safe, faster bounded-test batching: the default
   256-case/16-file and 32-case/4-file workers exceeded the 8 GiB cap, and
   eight-case/one-file batches pass the same selection; do not raise caps or
   omit cases; reconsider [fixture reuse](doc/plans/2026-09-17-bounded-fixture-reuse.md)
   if useful ([evidence](doc/DevelopmentHistory.md#forward-owned-lexical-forms-september-19)).
   Delete the radix-backed non-word-major meronomy path (no legacy paths), and
   remove the stale `dispatch_per_row_reset` note in
   [the fold-ladder plan](doc/plans/2026-09-10-meronomy-fold-ladder.md#open-defects-found-on-the-way)
   (`taxonomy_parent_map` is now initialised).
   From the 11–11c refactor, verify legacy before deleting (no-legacy rule;
   reasoning methods need Alec's review): `intent_boosts`, WholeSpace `part_chain`,
   `_automatic_synthesize_higher_order` against the pool's discovery path,
   and the META `insert_meta` / `taxonomy_parent` bindings once item 7's
   concept-level taxonomy owns them (`overlap_where_tiling` and
   `where_tiling_for_pass` are already gone). Move the 2.9 MB
   `evaluation-source.tar.gz` out of the repository (the receipt keeps its
   hash) unless Alec wants it in-tree.
   *Added 2026-10-08 (the coverage audit):* delete the legacy typed STM
   `_buffer` that only grows, and the detached role view carried beside the
   native occurrence anchor (ICA spec §10); rename the native context-read
   `exploration_trial` flag; remove the one-release rename aliases that
   raise with the new name (`Queries.py` `_operation_form`'s retired map,
   `Language.py` `_check_operator_availability`) after the one-release
   compatibility period following 6.2's mechanism landing. Documents that still describe retired or superseded things,
   to correct in one pass: [Params](doc/Params.md) lists `thinkingBudget`
   as live and `selectedThoughtPolicyWeight` as a REINFORCE term (both
   rejected at load; `attentionBudget` is the one budget) and
   [QueryWork](doc/QueryWork.md) still meters `selectedThoughtBudget`;
   `complete.grammar`, `ladder.grammar` and the tied-reconstruction fixture
   still comment `chunk`/`lookup`/`quantize`/`arma`/`what`;
   [GradientFlow](doc/GradientFlow.md) still heads rounds 2d, 2e, 3a and
   the final update "review/uncommitted candidate … no commit or push"
   (all landed at `3de37eef5`, `cce3a4f7b`, `f4a68404e`), gives the serial
   leaf as "meaning poles exchanged for negative activation" (superseded by
   the two lanes, plan §42) and the round-3 scheme as the
   "narrowing-weighted ceiling" (superseded by the adjacent-word wholes,
   plan §40); [Architecture](doc/Architecture.md) still offers "each word's
   two symbols … either meaning-pole orientation" at pair search, "has-a …
   narrows (round 3)" in the scheme, and "dimensional lift remains with
   5.5" (FutureWork has it); the [catalogue](doc/specs/2026-09-29-operator-catalogue.md)
   §12 headings still say "(uncommitted)" and its table lists `bind` as
   open (decided, thought-loop plan §3–§4).
   *Added 2026-11-04 (Alec: "I would prefer powers of two for all
   constants"):* every constant set by hand becomes a power of two, and a
   width made by adding such constants stays a sum. One pass over
   `data/*.xml`, `data/model.xsd`, [Params](doc/Params.md) and the
   `TheXMLConfig` defaults, from the
   [audit](doc/plans/2026-11-04-powers-of-two-audit.md):
   - round the free sizes of its §3; a fixture whose tests depend on a
     value keeps it with a comment;
   - remove the 152 declared widths that `resolve_identity_layout`
     overwrites at load, so no configuration states a width the code does
     not run, and correct `data/BasicModel.xml`'s header;
   - delete the unread `timeout`; fix the side findings of its §6.
   The measured values of its §4 are item 0's decisions.
   **Carried into item 2 on 2026-11-03 (audit of the items above 2.5):**
   D5 (2.8 spec §5 and §7; 2.7 record §38.6): regenerate the two
   decomposition snapshots under `output/item3-5-identity-initial/` from
   seeds 4199722147 and 2428440506 under the 256-wide layout, keep both old
   hashes (`dbd7315f...`, `f1d141ab...`) beside the new ones with the
   generator command and the layout difference, then run both
   `test_math_chain_decomposition` arms at budgets 512 and 768 with the
   original assertions; no legacy loader, no seed replaced. Then the slow
   marks deferred at the 2.8 and 2.9 freezes (26 core, 4 tower-projection,
   3 native packing, both decomposition arms, and
   `test_review17_measurement_wiring_on_one_ordinary_batch`) run once under
   `RUN_SLOW`: in item 2.5's `make test_all` (thinking spec §40 item 64) if
   that freeze comes first, else here; `make test_all` green once on one
   hash after D5 (the four item-1 runtime groups excepted) and the default
   sweep green at the ten-worker 28 GiB ceiling, the `XOR_grammar.xml` gate
   decided first (item 2.0). The training harness's per-interval logger
   (carried from item 4): reconstruction loss, expectation discrepancy, LTM
   occupancy and forgetting passes, the categorical-discrimination index on
   the fixed four XOR and 68 FineWeb probes (CP, within and between
   distances, no thresholds) with the both-rate per bracket level, the
   luminosity of provisioned truths at each forgetting pass, the held-out
   two-truths §7 test-12 probe, one fixed reconstruction sample per
   interval, and order-0 inventory rows against `nVectors`, reusing
   captured readings and budgeting their collection cost; item 0's stop
   conditions read it; the 11–11c pool, raise, refinement and
   unknown-per-order counters are retired with learned orders. Extend
   `test/test_corpus_checkpoint.py` to run the real `runBatch` on a small
   config and assert the resumed next batch's losses and kept programs
   byte-identical, with explicit checks on LTM0 `surprise`, `access_count`
   and `forget_*`, the relevance weights, the comparison reader's weights
   and the form centroids; exit for the resume certificate: one `train.py`
   command on a small config with `checkpointEveryBatches` set, re-run to
   resume, recorded in a receipt. Give every configuration a meaning block,
   then retire `Language.py`'s unpaired form-kernel path and its
   `meaning_layout` guards (iteration spec §19 note 2; 58 guards, 20
   configurations without `meaningWidth`). Remove the deleted
   `test_forced_ordinary_bound_declaratives_open_no_episode` entry from
   `DEFAULT_CERTIFICATES` (`test/slow_tests.py` line 100). Confirm the 7.5
   shared-operator report assertion is gone under 6.9's one-owner rule and
   strike the FutureWork sentence that defers its verification. Reword the
   "objectless marker" comment at `ReferenceContext.py` about line 460 when
   parts of speech are learned. In the one-pass document correction, add:
   [FutureWork](doc/FutureWork.md) still keeps `timestamp` and leaves
   situation codes and before/after with 5.5 (retired at `552c626ae0`;
   context wholes and VP phases landed in 3.5), still speaks of a logged
   both-rate monitor that was never built, still says item 2.8 is not landed
   (`a8a09a6acb`), still puts the concept-face negative image's measurement
   "with todo item 9" (now item 0) and calls the 2c follow-up "awaiting
   review" (landed `fcaff2d19`); [SelectedMeaning](doc/SelectedMeaning.md)
   "comparisons remain item 8" and [Params](doc/Params.md) "item 8's
   structural rule" point to item 0 and the Done line; [Architecture](doc/Architecture.md)
   still presents `Priming.propagate`, `geometricSpread`, `_priming_boosts`
   and `_last_priming_diffusion` as built and calls round B "not built" in
   its earlier paragraphs while its as-built paragraph records their
   retirement; and the tied-reconstruction fixture's tense, morphology and
   aspect planning comment (`data/tied_reconstruction_output_benchmark.grammar`
   lines 149–151) goes with the `chunk` and `lookup` comments.
- **1.5. Lookups as one matmul, whole/part folds as scatters, the percept
   pass per sentence** (Alec, 2026-10-30;
   [plan](doc/plans/2026-10-30-item1-5-lookups-and-folds.md)). Before item 1, so the
   compiler work compiles what this builds under `fullgraph`, and before
   item 0. From the as-built cost reading of 2026-10-30 (plan §2): the
   derived concept forms become a tensor kept current incrementally
   (device CSR definitions, one segmented scatter per order, a validity
   mask); content-based lookups over absolute truth concepts in LTM and
   conceptual space become one matmul of the cue against the content
   matrix with top-k under the work meter (`cued_rows`, `relation_evidence`,
   taxonomy evidence, witnessed rows; LTM1's index maintained on `add`);
   the percept pass runs once per sentence over all words with gradients,
   since the percept depends on neither grammar state nor the expectation,
   with ragged parts in place of the 64-wide padding and the forecast
   scored for all words at once; the second trial forks at the departure
   site. Grammar operations themselves stay serial; tokenization and
   new-type admission stay on the host, cached per type. Gates (plan §4):
   `fullgraph` compile of every rewritten brick, byte-equal or
   certified-equal results against the current path, one before/after
   timing on lesson-13 and the math chain with the July baseline retaken,
   allocation reads with no resize. Decisions taken 2026-10-30 (plan §5):
   one Codex round for the four steps.
   **Carried into item 1.5 on 2026-11-03 (audit of items 2.8 and 2.7):**
   while rewriting the derived forms, the atom property fill (2.8 spec §11
   O1): `refresh_property_fills` computes a pair atom's fill from its two
   characters without the marker byte, and gives length and mint atoms a
   fill of ones (`bin/WordIdentity.py` still passes `bytes_of(atom[1])`);
   a certificate on `refresh_property_fills`, `containment_audit` unchanged.
   **Order (Alec, 2026-11-03):** item 1.5 follows items 2.0 and 2 and is built
   on main. The 2026-10-30 sandbox branch `item1-5-lookups-and-folds`
   (`9151b6f10b`, about 1,400 lines over `EntryRead`, `LTM1`, `Layers`,
   `MediumTermMemory` and `Spaces`) is a reference, ported onto main after
   2.0's move of the three stores into `bin/Memory.py`, not merged.
- **1. Compiler work.** *Added 2026-10-20:* the generated math suite under
   `RUN_SLOW` (item 3, moved from 3.5 on 2026-10-21) is the compiler test at scale — compiled and eager
   runs of the identical corpus, seed and budget agree in actions,
   probabilities, costs and parameter updates.
   **Carried from the item 3.0 landing (2026-10-26; [orders §69](doc/plans/2026-10-26-item3-0-step5-orders-learned.md#69-claudes-review-of-the-serializer-round-and-the-item-30-sweep-test-freeze-ee7ca4e920bc-model-6fe273d2f12a7--2026-10-26-approved-for-landing-three-items-carried)):** four runtime failure groups: `test/test_objective_conflicts_slow.py::test_native_production_objective_measurements` at the 28 GiB guard, with the remaining allocation unattributed after the Householder and candidate-context repairs; `test/test_testpoint.py::TestGPUDevicePlacement::test_model_on_gpu` at its 120 s deadline; `TestServerQueries::test_demo_queries`, `::test_response_format` and `::test_response_length_bounded` in `test/test_testpoint.py` at their 30 s request deadlines; and `test/test_math_thinking_training.py::test_successor_is_learned_as_a_verb`, `test/test_reconstruction_roundtrip.py::test_grammar_output_loss_not_silent_zero` and `::test_mm20m_grammar_free_derivation_roundtrip` at their 45-minute caps with half work. Retain all failed attempts in the [sweep ledger](doc/benchmarks/2026-10-26-item3-0-sweep/repair-04-slow-budget/slow-resolution.md); these are runtime failures, separate from item 0's learning records.
   *Added 2026-10-04:* the open read of 6.8-1 made the
   whole forward 2.6× slower (.141 s against .055 s on the XOR fixture;
   [profile](doc/benchmarks/2026-10-03-operators-attention/attention-profile-review/measurement.json));
   Alec: "the slowdown will have to be addressed." *2026-10-09:* the
   math-chain campaign ran at 1.5–2.5 s per sentence under two workers
   (episodes of 32 thought steps included), a week for thirty trainings;
   Alec deferred the thinking measurement "until we further optimize
   things" — this item's measurement now includes a thought episode's
   step and the per-sentence time at the math corpus. *2026-10-10:* profile
   the rising per-presentation runtime in the 6.1 curriculum (the final
   relation-answer stage took 1,001.95 seconds for 64 presentations plus
   its initial/final measurements). Its timing traces and active-autograd
   process sample are retained in the [6.1 receipt](doc/benchmarks/2026-10-09-item6-1/README.md);
   compare like-for-like work before assigning a cause. The decoder's two-word
   walk (6.8 plan §10.2) may pay part of it back. The throughput levers for the serial loop — batch
   across sentences, the known-word lookup concession, subsampled
   reconstruction, closing the host islands and moving
   the sparse stores to device CSR, single passes — are listed in
   [FutureWork](doc/FutureWork.md#throughput-levers-for-the-serial-loop-item-1-candidates)
   and belong here (Alec, 2026-09-25); the `serial` flag and the
   `subsymbolicOrder`/`symbolicOrder` knobs those levers named are rejected
   at load since 6.8, so the levers apply to the one interleaved loop.
   Forward chooser split/lift once per slot; backward's
   launch-bound kernel count per brick; B24 brick +25% against pre-ladder.
   The comparison reader of round 2e is a `functional_call` copy of the
   whole answer path (plan §23, observation 1); share its structure when
   the answer path grows.
   Exit: sentences/s and peak memory at the run's batch and brick size against
   the July baseline, re-taken on `99207a3` since the tower fold ladders are
   gone. New eager hot spots from 11–11c to measure and vectorise: the
   per-row loop in `cs_read_memberships`, the per-row composition in
   `_compose_order0`, the edge loops in `_reverse_field`, the per-reading
   descent in `refine_over_collected`, candidate growth in
   `_prepare_part_learning`, and the per-column loops in
   `promotion_observe`.
   The explicit slow reconstruction-cache probe fails its nonzero
   compose-gradient assertion on the current baseline; the cache claim remains
   unverified by that probe ([audit](doc/benchmarks/2026-09-25-item9-parity/README.md#validation)).
   *2026-10-30 (Alec):* the as-built cost reading and the lookups, folds and
   percept-pass work are item 1.5 ([plan](doc/plans/2026-10-30-item1-5-lookups-and-folds.md)),
   which runs before this item so the compiler work compiles what 1.5
   builds under `fullgraph`.
- **Item 0 binding evidence:** `test_live_pending_premise_closes_in_both_orders`
  retains the strict §14 arrival assertions verbatim, with `open_names=('x',)`
  and no named closure forcing. Its `pending_premise_learning` artifact is
  unregistered. Measure live closure at the trained checkpoint; correct-or-open
  is the step-1b mechanism gate. Arrival menus remain excluded from delayed
  mention training and answer supervision, carried here under bind plan §19.
  Under the grammar-only contract, x is a site only when its reading is
  order 1; this certificate gets no fixture order, so learning that a name
  reads at order 1 is part of what it must show (reconstruction spec §12).
   **Carried into item 1 on 2026-11-03 (audit of items 3.5 and 2.7):** the
   per-sentence timing includes the nested-admission path (word-form
   reindex, hash and trie rebuild after an admitted suffix; iteration spec
   §20.3) in one bounded lesson smoke, and the stationary-walk stop and the
   eager decoder's selected-inverse path (iteration spec §§12–13, §19 note
   5); exercise the three `torch._assert_async` sites (`ChunkJournal.py`,
   `Expectation.py`) once on MPS before the run (iteration spec §62 watch
   item). The generated math suite under `RUN_SLOW` was already here.
- **0. The full training session**: the long FineWeb run, only with items 11–1.5
   done and item 1 measured. Expectation on in `model.xml`; the
   `BasicModel.xml` flip follows the plan's §10 gates. Stop on rising
   expectation discrepancy, a reconstruction regression against the
   re-taken baseline, a forgetting pass deleting protected rows, a
   provisional-pool exhaustion warning, or order-0 inventory approaching
   `nVectors`. *Baselines:* the `99207a3` and reviewed-9b reconstruction
   figures predate round 3a, after which identified-word reconstruction is
   an exact zero audit; re-baseline reconstruction on the accepted
   `f4a68404e` source before the run (the Done lines of items 8 and 5 cite the old figures
   until then). *Condition (operators plan §13, confirmed by Alec):* the
   run trains by expectation and output along one path, with expectation's
   sources live and reconstruction retired to an audit, only where item 3's
   audit reads zero at the run's width; where it does not, the run keeps
   reconstruction as a loss for that part. Expectation across the
   architecture itself landed as item 4 (`fdeef42b8`); the invertible path,
   the width and the audit this condition reads were item 3's from
   2026-10-19 (spec §3.1) and are the capacity-run line carried below. At the trained checkpoint, resume the
   deferred **MM_math_chain learning measurement** under all four corrections:
   one-step problems first; worked steps scored as intermediate answers;
   an episode that binds a found candidate; and budget in steps, with each
   query's record charge bounded separately. The stopped campaign supplies
   no learning claim; preserve it and declare the corrected measurement
   before running ([closing receipt](doc/benchmarks/2026-10-09-item6-2-closing/README.md),
   [protocol corrections](doc/benchmarks/2026-10-08-math-chain-repair-2/protocol-corrections-14-13.json)). Bars:
   held-out binding 9 of 10, the chain bar, the expectation-only and
   zero-budget controls ([protocol](doc/benchmarks/2026-10-08-math-chain-repair-2/protocol.json));
   the four corrections are unimplemented and precede the measurement;
   re-declare it after item 2.0 retires the decomposition teacher and
   rebooks the choice reward to output.

   **Item 3.0 step-3 companion residual: 6/8, learning-side** (STM-chunks
   §27, replay §28). The length-1 question of document 1 in both arms keeps
   an explored `whole` over greedy `surface` under the existing R+A rule.
   On placement's untouched archive, the same greedy program already has
   coverage 1.0; its explored path also has 1.0 and loses on iterations.
   The chunk path's explored coverage 0.75 wins. No reader repair or forced
   keep is warranted by the decided replay criterion. Preserve seed
   **4199722147**, original snapshot and budget **512** for this residual;
   length 2 remains correct in both arms at seed **2428440506**, budget
   **768**. All 32 premise readings remain complete. Measure the chooser
   and reconstruction objective's learned choice at item 0, without
   substituting the standing existence proof for companion accuracy
   ([replay and closing receipt](doc/benchmarks/2026-10-25-item3-0-stm-chunks/closeout/README.md),
   [decision and diagnosis](doc/plans/2026-10-25-item3-0-step3-stm-chunks.md#28-closing-replay-the-companion-residual-is-learning-side--codex-2026-10-25)).

   **Slow-budget note (Alec, 2026-10-26):** the entries below retain their
   measured historical budgets. Optional RUN_SLOW repeated work now defaults
   to half; `RUN_SLOW_FRACTION=1` restores the full budgets for a like-for-like
   replay ([policy and separate results](doc/benchmarks/2026-10-26-item3-0-sweep/repair-04-slow-budget/README.md)).
   No earlier result is relabeled as a half-budget measurement. The companions'
   512/768 work budgets and one-shot contracts are unchanged; runtime stops
   remain separate from learning failures.

   **Item 3.0 landing — 2026-10-26:** [orders spec §69](doc/plans/2026-10-26-item3-0-step5-orders-learned.md#69-claudes-review-of-the-serializer-round-and-the-item-30-sweep-test-freeze-ee7ca4e920bc-model-6fe273d2f12a7--2026-10-26-approved-for-landing-three-items-carried) approves step 5 `c4a1d92e3f`, serializer `caece43fc0`, sweep repair `03fbbf2a7d9c`, final model repair `6fe273d2f12a7`, and test budgets `ee7ca4e920bc`. The learning records stay at item 0 at their listed budgets; the four runtime failure groups move to item 1 as named below; the unforced 1-to-2 raise certificate for “Felix is a cat. Ciao is a cat.” moves to item 3's corpus round. Learning claims use `RUN_SLOW_FRACTION=1`; a half-work result states its work. No RUN_SLOW before November at the earliest.

   **Item 3.0 sweep learning remainder:** blind `MM_20M_xor` reconstruction
   is 0.0 exact match after **160 epochs**, seed **3145616037**, in the
   initial slow sweep on `caece43fc0`. The repair slow run on `03fbbf2a7d9c`
   also records **0.0 exact match at 160 epochs**, seed **3510236551**.
   Location recovery is 0.0625 and 0.125 respectively; these are separate
   fresh initializations, not a learning improvement claim. The assertion
   stays unchanged; retain both original-budget records with the 6/8 companions for item 0
   ([both raw benchmark records](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/blind-learning-records.json),
   [initial failed-run record](doc/benchmarks/2026-10-26-item3-0-sweep/run-01/test_all/failures.md)).

   The separate scaffold exact-reconstruction gate also records **0.0 exact
   match at 160 epochs**, seed **1506395199**, with location recovery **1/3**
   on `03fbbf2a7d9c`. Its original 1.0 assertion remains, and its 160-epoch record stays at item 0
   ([raw record](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/reconstruction-learning-records.json));
   it is separate from the blind measurements above.

   The two ordinary generation-gradient variants also remain learning-side:
   one requested two-stream batch at learning rate .001, 48 trips per output
   walk, zero emitted concepts and zero conditioner gradient on the recorded
   initializations. The detailed replay chooses `not` / `null` and reaches
   no emitting continuation. Keep both nonzero-gradient assertions unchanged
   for item 0 ([walks and budgets](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/generation-learning-records.json),
   orders spec §30); no forced choice or additional training was supplied.

   Three native answer-gradient variants also have a zero conditioner gradient
   after one requested two-row batch at .001 and 48 trips. The recorded replay
   emits one unconditioned role, leaving the conditioned role pending; it keeps
   the strict assertions and original budgets for item 0
   ([trace](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/native-conditioner-learning-records.json),
   orders spec §§39–40). The fourth native variant passes after its form-credit
   and kept-field observer port on the original initialization; it is not
   another zero-total-gradient result.

   The full slow sweep's fresh decomposition initialization also retains a
   two-step proof failure: 0/2 final bindings at **571/768 work**, reproduced
   on both `03fbbf2a7d9c` and pre-repair `caece43fc0`; the one-step case is
   2/2 at **246/512**. Keep the strict proof and saved states for item 0
   ([comparison](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/decomposition-learning-records.json),
   orders spec §32). This is separate from the passing original-snapshot
   arms and their 6/8 companions. The existing expected arithmetic learning
   gates also remain below .9: direct arithmetic **.3125/.1290 at 40 epochs**,
   dependency chains **.5234/.1953 at 30 epochs** (train/test;
   [records](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/arithmetic-learning-records.json)).
   Serial supervised output reaches **.78125 best accuracy at 40 epochs**,
   learning rate .001, batch 8, after the completed-field observer port;
   the original 1.0 bar remains ([record](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/output-learning-record.json)).
   The cold-boundary corpus reaches its unchanged learning assertion after
   the equality-close repair: both boundary weights stay **-4.0 at six
   epochs**, batch 4, 48 rows, learning rate .001, boundary rate 40/update
   interval 16 ([record](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/boundary-learning-record.json)).

   The supplied neighboring-sentence answer gate also remains open: on the
   frozen sweep source `03fbbf2a7d9c`, the future-label arm's surface error is
   **.0860481 before and .0965504 after 30 updates**, batch 2, learning rate
   .001. Its original 10% improvement bar, answer masks and gradient checks
   remain unchanged ([record](doc/benchmarks/2026-10-26-item3-0-sweep/repair-01/what-learning-records.json)). The past-label
   arm records **.0799439251 → .0799437165** at the same budget against the
   unchanged 10% improvement bar; its masks and gradient checks also pass.

   The supplementary three-epoch XOR reconstruction draw (seed 1646304963)
   records 0 exact match against the unchanged .5 bar. A source comparison
   at the original sweep's recorded seed 2197030207 reproduces its .5 exact
   match and identical losses; neither record replaces the other or supplies
   a selected acceptance seed ([record](doc/benchmarks/2026-10-26-item3-0-sweep/repair-03/supplementary-roundtrip-record.json)).

   **6.1's learning gates** also remain at item 0's checkpoint: supplied
   answers and asked parts; free reading in the lesson's order on permuted
   fields, with order-lesson CE near zero; and the context stages in
   [stream-state §3](doc/plans/2026-10-08-stream-state.md#3-required-learning-stages-the-replacement-for-explicit-context-management).
   The support mechanism and the first two generated-curriculum stages are
   accepted; these learning claims are not.

   **Item 4's learning gates** remain open: §9.6 graded versus shuffled
   (seed 23's shuffled control has lower R); global derivation optimality;
   per-level residual/collapse measurements and exposure; learned repeated-cat
   recognition. The standing all-level XOR/math regression comparison is
   the preceding outstanding-gates round's §9.1 gate, not a deferred
   learning campaign. The original failed native run is not
   regraded or extended. The mechanism and controlled carrier certificates
   do not satisfy these learning gates ([receipt](doc/benchmarks/2026-10-19-item4-repair/README.md)).

   **Iteration 1 learning carry, 2026-10-24:** swap/verb source `65451694e`
   passes source order 8/8 but fails both original CE bars; the original
   lower and definite budgets each pass bracket/occurrence 1/1. Description
   stops at answers 1/4 with brackets 4/4. Carry description → adjective →
   definite-adjective → final evaluation → copular identity here, with
   the existing budgets, gates and prerequisites. Nothing after the stop
   ran; no copular prerequisite is lifted or artifact promoted. The older
   source `3b34b3f38` lower failure (occurrence 0/1, bracket 1/1) stays in
   its [receipt](doc/benchmarks/2026-10-23-item3-5-stage4e-np/README.md).
   On modal source `01effdcaf9`, both the free-leaf and affix corpora
   complete their fixed 24 training examples / 36 paired updates, then
   fail held-out answers 0/4 and required timeline histories 0/4. Carry
   certificate 15's learning here; no seed, budget, forcing or menu rescue.
   Each form's 21 retained sentence-batch keep records contain no
   alternative timeline and no fired teacher; no trial applied `adverb`
   to the marker. Start the learning remainder from operator choice.
   Before retrying, carry §19's teacher pairing by branch point rather
   than model-close tick; the corpus has no after-observation target for
   the verb teacher by design.
   The [iteration receipt](doc/benchmarks/2026-10-23-item3-5-verb-modal/README.md)
   retains every run and state; no learning claim is made.

   **Iteration 2 learning carry, 2026-10-24:** on repaired source
   `04c1132797`, all six occurrence §13.2 placements complete the original
   three epochs / 18 training documents / 162 paired updates and six
   held-out documents. Every placement scores answers 0/6 and exact
   reverses 0/6; selection counts are 6/6 only for the unmarked form,
   0/6 otherwise. Retained selected-output flags occur in three forms,
   but no event retains a case value; two forms show limited verb-teacher
   activity, and none shows a modal teacher or alternative timeline.
   Carry the learned phase selection, phase answers, surface recovery and
   word-class generalization here, with the original budgets and gates,
   no new draw or extra training. The mechanism tests do not satisfy
   these learning gates. All original failures, interrupted runs and
   same-draw complete replays remain in the
   [line-6 receipt](doc/benchmarks/2026-10-24-item3-5-iteration2/line6/README.md).

   **Item 6's learning gates** remain prerequisites at this checkpoint:
   stage 4e's ≥90% free-bracket and ≥95% NP-occurrence gates, zero duplicate
   writes, every lowered NP yielding one order-1 occurrence (description stamp
   reported, not required), and no unlowered-description occurrence; then
   balanced binding by group and document-close row pairs, with mint/bind
   fixed semantics and same-kind revision; VP temporal-difference prediction;
   direct-operand word classes; and deferred order-2 kind recovery. The stage 4e
   candidate and measurement exist, but its [gate fails](doc/benchmarks/2026-10-17-stage4e/).
   The VP teacher mechanism is published at `f306fc28aa`; order-2 kinds are
   not built. Do not start another native
   identity run before stage 4e passes. Retain stored-idea recovery beyond two
   words, order CE, relation trace 2/4, free readback 0/4, MM grammar recovery,
   compiled retention and c's marker-headed `five` as named gates, not
   completed learning. The paired zero-cost invariant is repaired by Step 0,
   with its original counterexample preserved in the receipt.

Everything that is decided in direction but not on this path is in
[FutureWork](doc/FutureWork.md).

   **Decisions collected at the start of item 0 (Alec, 2026-11-03):** the
   evaluation harnesses below, declared before the run: item 8's causal
   study (its 256-task held-out generator, the arms and the thresholds),
   6.5's six learning gates (no harness yet), and the held-out set behind
   the held-out cost metric of item 2.0's round; until then item 2.0
   reports acquisition, answers and each chooser's departure win rate.
   Added 2026-11-04, under Alec's powers-of-two rule
   ([audit §4](doc/plans/2026-11-04-powers-of-two-audit.md)), the measured
   constants:
   - `batchSize` 28, the memory operating point: 16 or 32;
   - `maxDocs` 11,207, chosen for 1,000,014 sentences: 8,192 or 16,384;
   - the concept width 1256 (`meaningWidth` 896);
   - `conceptLayers` 5: 4 or 8;
   - the thresholds at 3 (`conceptMintCount`, `raiseCount`,
     `mereologyRefinePatience`): 2 or 4.

   **Carried into item 0 on 2026-11-03, from the items above 2.5 (audit):**
   - *From 9.* At the trained checkpoint, once `checkpoint_readiness` reads
     at least 1,000,000 completed FineWeb training sentences
     (`training_state.fineweb_training_progress`; fused-AMP updates
     conservatively uncounted), register it under `fineweb` in
     `test/fixtures/phase-artifacts.json` (path, SHA-256, config, producing
     receipt; [registry](doc/benchmarks/2026-10-18-harness-gating/README.md));
     a missing exposure skips, never passes. Then run the five `fineweb`
     certificates: the 56 held-out wording, generation and rereading
     assertions on the loaded model, prediction's benefit from context,
     predictive thought at gain 1 against gain 0 (less work at no worse
     MSE/BCE), the depth-3 end state and truthset provisioning
     ([protocol](doc/benchmarks/2026-09-26-item9-followup/PROTOCOL.md)); the
     tiny-run nulls and the old context-free and assertion-thought scores
     are invalid controls.
   - *From 9 and 8, the causal gate, declared before the run.*
     Independently trained arms (learned thought, direct answer, equal
     actual compute, reconstruction only) from seeds 0, 1 and 2 for 10,000
     updates after qualification, on the 256 held-out tasks whose generator
     and hashes are frozen before training (the generator is not yet
     written; [protocol](doc/benchmarks/2026-09-26-item8/PROTOCOL.md));
     gate per seed on accuracy at matched actual work or work at matched
     accuracy; ordered prediction must beat the shuffled and context-free
     controls without worsening reconstruction or discrimination against
     reconstruction-only, with the reconstruction and discrimination control
     re-declared after item 2.0's order-0 reconstruction; report warmed
     sentences/s, operation counts and memory per arm (20 warm-up and 100
     timed updates). Item 8's routing-coverage comparison over declared
     nested catalogs that include one opaque candidate: the sentence opaque
     share must decline strictly in every seed; a catalog without an opaque
     candidate is a descriptive zero, not a decline.
   - *From 9, 6.5's learning gates, which have no measurement harness yet:*
     held-out anaphora against the rule-based binding of `206a0146` at
     matched exposure, verb reuse, prediction against the detached-target
     control, shuffled order, renamed vocabulary, and mint-after-"a" and
     bind-after-"the" against a shuffled-determiner control; seeds 0, 1 and
     2, never tuned ([spec](doc/specs/2026-09-26-independent-components.md),
     [gate list](doc/benchmarks/2026-10-07-item6-5/learning-readiness.json));
     declare and build the harness before the run;
     `test/test_sentence_references.py` is mechanism-only and does not
     satisfy them. Also the concept-face negative image's joint and
     useful-query learning (round 2 `3de37eef5`) at the checkpoint.
   - *From 5.5.* Occurrence spec §8's learning measurements with the
     six-placement retry: past and future of one event told apart in
     generation, the progressive not entailing completion, a marker
     recovered in reconstruction; recorded, not tuned (the
     irregular-as-stem-plus-past reading is withdrawn by §12).
   - *From 4.* Score the frozen NanoChat next-word manifest with
     `bin/eval_nanochat_grammar.py` at the trained checkpoint: top-1,
     reciprocal rank, intact minus shuffled; recorded, not tuned.
   - *From 2.7, before the run.* The capacity search and selected-prefix run
     of `test/test_corpus_large_slow.py` (B28/W256,
     `data/corpus-capacity-grid.json`, the D1 fitting target `nVectors` and
     `ltmCapacity` 262,144, relations and witnesses 65,536), recording the
     2.7 record §38.5 exits: loader time, admission and occupancy,
     high-water marks, sentences/s, `maxDocs`, `shardDir`, streams,
     `RowTime` range and ladder periods, the D ~ c k log N derivation with
     its assumptions, and the native R == 0 audit at the chosen width (the
     invertible-path condition above reads this line); the PS1024 to
     PS33792 points of 2.7 record §43 are development evidence, not a fit.
     The companion residual's "original snapshot" is item 2's D5
     regeneration from the recorded seed. Before the mass, count and
     kind-verb measurements: capture at sentence 132 of the final
     curriculum (`Queries._existing_symbol`, the four captures of the 2.7
     receipt `review-open-items.json`) and repair at the producer.
   - *From 2.7, the learning remainders.* The unforced 1-to-2 raise,
     "Felix is a cat. Ciao is a cat.", as lesson-13's relations criterion
     (160 sentences, 8 epochs, 80 supplied answers, `raiseCount` 3, nothing
     forced; latest 0 to 1 of 80 bound, cat at order 1, thinking spec
     §39.6), and the mass, count and kind-verb phases of the final
     curriculum (288 training and 16 held-out sentences, four epochs per
     phase; proper phase passed, mass unmet at 131 of 304, 2.7 record §46);
     order-2 kinds and kind-level verbs (3.5, 3, 2.7 §47) measured unforced
     on the lesson protocol, mechanism first if the checkpoint shows none;
     silence as the taught target on rows with no supplied answer (3.0 step
     4): the model learns to say nothing unless asked, measured at the
     checkpoint, no forced mask.
   - *From 3.5.* The §19 note-4 branch-point pairing is retired with the
     modal teacher (item 2.0 spec §5); the modal-corpus retry starts from
     operator choice under participation training.

### Done (newest first)

*2026-11-03: the eight items above 2.5 (9, 8, 5.5, 4, 3.5, 2.9, 2.8, 2.7) collapsed to the lines below after an audit of each one's remaining work; their blocks as they stood are in [DevelopmentHistory](doc/DevelopmentHistory.md#todo-items-above-25-as-recorded-until-2026-11-03), and the remaining work is in items 2.0, 2, 1.5, 1 and 0 under "Carried ... 2026-11-03".*

- **Item 2.7 — corpus at target size (formerly item 3):** closed code-complete at review freeze `472f5000850e`, commit `0aedc06414` (2026-10-29), reviewed in [spec §47](doc/plans/2026-10-27-item2-7-corpus.md#47-claudes-review-of-the-producer-repair-round-472f5000850e-and-the-closing-of-item-27--2026-10-29); corpus assembly and curriculum loader, stage-preserving packing and checkpoint continuation, the familiar tier, reading-mode identities, generation repair with certified completion, the definition fold, percept expectation with its in-loop update, admission and retrieval. Thirteen scoped groups 2,545 passed, 57 skipped; lesson-13 160/160, ideas pass, relations fail, answers 0/80 with text; final curriculum 131/304, proper phase passed, mass incomplete, count and kind-verb not reached ([receipt](doc/benchmarks/2026-10-29-item2-7-producer-repair-review/README.md), local). No landing, no acquisition claim. D7 went to [FutureWork](doc/FutureWork.md#the-residual-split-a-stopped-subtraction-or-the-correction-alec-2026-10-29) (§49); D8 landed in 2.5 round A `3718d9d0dc`; the non-finite kernel gradient and the order/parts disagreement were repaired in 2.5's completion round (thinking spec §§37.2, 38.1, 40). Remaining work moved to items 2.0 (the supplied `yes` admission, the MTM-value recurrence), 2 (D5's snapshot regeneration, the slow suite and sweep, one comment), 1.5 (the `WordIdentity` atom fill) and 0 (the capacity search and the run with the §38.5 exits and the width audit, the sentence-132 capture, the unforced 1-to-2 raise and the mass, count and kind-verb measurements, the silence gate).

- **Item 2.8 — order-0 concepts sorted over two form subspaces, one call:** landing `a8a09a6acb` (WikiOracle `a992602`), reviewed source `555e63e2f521` (spec §11), Claude LAND through Alec; identity join and property soft meet in one percept vector (`propertyDim` 256, `propertyOnes` 1), tower-1 content derived from byte membership, `_interval`, `centroid_forms` and `SymbolCentroid.place` retired with absence certificates, the four-report `containment_audit`; 20-case certificate and native lesson green, 2,053 scoped passes, 48 skips, 0 failures, with the two failed freezes `be4514dedea8` and `70f21c757906` preserved ([receipt](doc/benchmarks/2026-10-27-item2-8-subspaces-r3/README.md)). D5's snapshot regeneration and the two decomposition slow arms are item 2; the O1 atom fill is item 1.5.

- **Item 2.9 — priming as location-based diffusion:** landing `17e9454763` on frozen source `b031b11ee9f2`, Claude LAND through Alec with the R1/R2 document repairs ([spec §9](doc/plans/2026-10-26-priming-location-diffusion.md)); 1,873 gate passes, 47 existing skips, slow marks deferred, no `[V,V]` tensor ([receipt](doc/benchmarks/2026-10-27-item2-9-priming/README.md)). Superseded at item 2.5 round B (Alec, 2026-10-28/30; spec §10, [thinking spec §§22.3, 24](doc/plans/2026-10-27-item2-5-thinking.md)): `Priming.propagate`, the radius controller, `_priming_boosts` and the `geometric*` knobs retired after the paired lesson-13 measurement (both spreads 0/12, no learning claim); the `_priming_seen` ledger, the class centroids with the 15-degree contiguity read and `part_edges` stay ([retirement](doc/benchmarks/2026-10-30-item2-5-round-b/retirement.json)).

- **Item 3.5 closed — iteration 2 (lines 6–7) and the mechanism landing:** reviewed source `75345f891e` (on main as `2067dbc25a` after the 2026-10-25 history rewrite), published main `4f136f76fa`, WikiOracle `69f82f5f77`, Claude LAND through Alec ([iteration spec §62](doc/plans/2026-10-23-verb-and-modal.md#62-claudes-land-review-of-75345f891e--2026-10-24)): document covers and `.when` segments in batch-indexed stores, centroids on form with operations on meaning, VP phases as case selection, direct-operand occupations, the presentation boundary with persistent cover concepts, capacities 18/20/20 by derivation; 588 scoped passes and 41 existing skips on the second repair after the one end-of-3.5 sweep (6,504 cases, 35 repaired) ([receipt](doc/benchmarks/2026-10-24-item3-5-iteration2/README.md)). With iteration 1 and the step 1–2 lines below, 3.5 is a mechanism landing with no learning claim: the six phase placements (0/6), line 3's rerun and the modal corpus are item 0's; order-2 kinds, kind-level verbs and the math suite went to items 0 and 1; the root-position and event-history follow-ups are in FutureWork.

- **Item 5.5 — occurrence, tense, aspect, the preposition and surface form:** mechanism landing `552c626ae0` ([receipt](doc/benchmarks/2026-10-17-item5-5-mechanisms/README.md)): the clock column, `Clause.eternal` and the analyzer retired; order is the stream's history with store-order recency (`get_stm_chain`) and the stable address re-witnessed in place; `surface` as absorb and transpose with exact inverses; the preposition idempotent; nested sub-word wholes by the type criterion; the clock-, offset-, chain-, interleaving-, preposition- and surface-inverse certificates (`test/test_item55_mechanisms.py`). Relative `.when` and the address upsert landed earlier at `f4a68404e`; the remainder landed through item 3.5 — documents and streams as context wholes with a cover row, VP phases as case selection under one grammar across six marker placements, direct-operand classes (`f306fc28aa`, `75345f891e`/origin `2067dbc25a`; [receipt](doc/benchmarks/2026-10-24-item3-5-iteration2/line6/README.md)). No learning claim: spec §8's learning measurements and the six placements (0/6) are item 0's.

- **Item 4 — run harness and resume test:** the frozen next-word gate `bin/eval_nanochat_grammar.py` (document-held-out manifest, intact and shuffled-prefix controls, top-1 and reciprocal rank from the model's own word expectation bank; `test/test_nanochat_grammar_eval.py`), last repaired at `7285b382e6`; mid-epoch resume certified by `test/test_corpus_checkpoint.py` and `test_checkpoint_restores_optimizer_counters_and_rng` — corpus cursor, batch and step counters, RNG, optimizer state, STM slabs, MTM candidates and mints, priming ledgers, percept arrival state, LTM0 addresses and LTM1 relations restored and the same epoch continued on the remaining rows (`0aedc06414`); forgetting passes log deletions by origin and the value cut-off (`b1a19c5b8`). Carried: scoring the trained checkpoint (item 0); the per-interval logger set and the real-`runBatch` byte-identical one-command certificate (item 2). The number's earlier use, the expectation mechanism, has its own lines below.

- **Item 8 — structural preference and routing evidence:** landing `206a01468d` (WikiOracle `58cc110`) from handoff `074093ec52`; exact-score tie preference for the structural face inside the ordinary MLP (`GrammarPreference.py`), committed-program routing counts and the nested-catalog coverage comparison (`GrammarEvidence.py`), the arithmetic-isolation and renamed-vocabulary probes, and the review's corrected marginal and complete-position overlap; one source-matched 4,980-case sweep (4,653 passed, 326 skipped, one XFAIL) with reconstruction exactly matching reviewed 9b ([receipt](doc/benchmarks/2026-09-26-item8-review/README.md), [protocol](doc/benchmarks/2026-09-26-item8/PROTOCOL.md)). **No learned-utility claim:** the predeclared causal study, the declining-opaque-share comparison (no catalog yet holds an opaque candidate) and qualified-arm throughput are item 0; the `XOR_grammar.xml` reconstruction gate goes to item 2.0 with the decomposition teacher's retirement, its class-accuracy sibling passing; the MM-grammar gate is the seed-41001 regression Alec fixed on 2026-11-03 (thinking spec §41.7), with the twenty-start panel as the initialization-sensitivity record; item 6's carried learning findings are at item 0.

- **Item 9 — mature-checkpoint learning evidence:** `8bc710a` (exposure counter `training_state.fineweb_training_progress`, the 1,000,000-sentence readiness gate with fused-AMP updates uncounted, loaded-model wording, prediction and predictive-thought certificates) and `180288c32f` (phase registry: the `fineweb` entry in `test/fixtures/phase-artifacts.json`, five certificates in `PHASE_CERTIFICATES`, an absent artifact skips and never passes; the env-var runbook retired). Tiny-run nulls, incomplete reverse programs and the 5/56 wording failures are preserved as diagnostics; old context-free and assertion-thought scores are invalid controls ([original receipt](doc/benchmarks/2026-09-26-item9/README.md), [follow-up](doc/benchmarks/2026-09-26-item9-followup/README.md), [harness gating](doc/benchmarks/2026-10-18-harness-gating/README.md)). No qualifying checkpoint exists; the five certificates, the three-seed equal-update causal gate and 6.5's six learning gates (no harness yet) moved to item 0; 6.2's MM_math_chain measurement was already there.

- **Item 3.5 iteration 1, lines 3–5 and the meaning/form swap:** landing `f306fc28aa` (WikiOracle `319a793`), reviewed sources `65451694e` and `01effdcaf9`, Claude LAND through Alec; scoped gates pass, the fixed-budget lesson stop and both failed modal corpus forms are carried to item 0 without rescue. The required corpus finding and parent Ethics edit are published ([receipt](doc/benchmarks/2026-10-23-item3-5-verb-modal/README.md)).

- **Item 3.5 step 2c and step-2 sweep:** reviewed source `fcaff2d19`, frozen sweep repair `9f5500870`; Claude LAND through Alec, one retained full sweep and passing scoped repairs, no production change after review ([receipt](doc/benchmarks/2026-10-23-item3-5-step2c/sweep/README.md)).

- **Item 3.5 step 2 identity threading:** landing `148953f9b` (WikiOracle `175c101`), reviewed source `65ab7fbe9`, Claude LAND through Alec; 1,627 scoped passes and twelve existing skips ([receipt](doc/benchmarks/2026-10-23-identity-threading/README.md)).

- **Item 3.5 step 2b — storage-only round:** landing `39272e7a6` (WikiOracle `ade0399`), reviewed source `6a36b7cee`; 1,144 scoped passes, nine existing skips, Claude LAND, every failure and port retained ([receipt](doc/benchmarks/2026-10-22-item3-5-step2b/final/decoder-domain/README.md)). Identity, derivation removal and 2c remain open.

- **Item 3.5 R10 — named token content before the close:** landing `e243d7bdf` (WikiOracle `2ffda78`), reviewed source `23a41540f`; exact authorized fixture correction, 111 scoped passes and four existing prerequisite skips, Claude verdict publish ([receipt](doc/benchmarks/2026-10-21-item3-5-r10-final/README.md)).

- **Item 3.5 line 1 — grammar-only reference sites and reconstruction reads the bind:** landing `a514a71e0` (WikiOracle `b49960a`), reviewed source `3a9db635e`, receipts `63c600712`; Claude verdict land, all scoped gates pass, R10 follows separately ([receipt](doc/benchmarks/2026-10-21-item3-5-reference-reconstruction/README.md)).

- **Item 3.5 step 1b — native binding, MTM and participant menus:** reviewed source `cc3672cfb`; all requested gates pass, seven prerequisite skips and two nonblocking root-position follow-ups declared, no learning claim ([receipt](doc/benchmarks/2026-10-21-item3-5-review11-landing/README.md)).

- **Item 4 Step 0 — carried invariants and gates:** `bfc834d79`; all-level XOR/math comparison, paired reconstruction, uncapped prime/content-neutrality certificate, mention-ranking credit, one full sweep and scoped empty-menu repair ([receipt](doc/benchmarks/2026-10-19-item4-step0-round2/README.md)).

- **Item 4 — expectation mechanism:** `fdeef42b8` on `codex/item4-repair`; approved defaults, enabled parity measurement, whole §14 completion, one full sweep and scoped repairs ([receipt](doc/benchmarks/2026-10-19-item4-repair/README.md)). Failed learning and carried Item 0/5.5 gates remain open.

- **Item 5 — forgetting:** reviewed source `b1a19c5b8`, receipt `d15bb89c3`, with the approved dead-branch cleanup; LTM surprise retention, live boundary/checkpoint coverage and conservative concept reclamation. The sweep carries item 0's paired zero-cost counterexample; the link failure was environmental. [Review and receipt](doc/plans/2026-10-19-item5-start.md).

- **Item 6 — closing mechanism:** reviewed candidate `a3ec07fd0`/`b41cf3b5b`, consolidated on main in the [landing receipt](https://github.com/arborrhythms/WikiOracle/blob/main/doc/benchmarks/2026-10-16-item6-landing.md): shared bounded inverse and candidate type families; forward discard with the search exemption; positional operands and orders from stamps; paired stages 1–2; reference at the close with fixed mint/bind operators and revision. **No learning claim.** At that landing, VP vectors by temporal difference, order-2 kinds and stage 4e (constituents) were not built; §15's sequence carries alongside 5.5 and at item 0, with 4e first.


- Item 6 positional operands and paired singleton warm-up: `51f069c30`, pushed on main, WikiOracle `f4d3139` ([receipt](doc/benchmarks/2026-10-15-item6-orders-paired/README.md)). The singleton gate passes at 20/24; balanced identity remains unresolved (0/146 bindings, 0/20 row pairs), with no learning claim. Its reference-close follow-up is [measured separately](doc/benchmarks/2026-10-16-item6-reference-close/README.md) and accepted as the closing mechanism; learning remains open.
- Item 6 gate-off measurement and search-inverse exemption: `f1dbd12a2` (including `c647e1506`), pushed on main, WikiOracle `27a60da` ([receipt](doc/benchmarks/2026-10-14-item6-gate-off/README.md)); accepted in plan §11. No learning claim; the fresh corrected-objective warm-up and its singleton gate are measured separately.
- Item 6 part-2 M/N/O/I preparation and first eight-branch native experiment: `af634ed35` ([receipt](doc/benchmarks/2026-10-10-item6/identity-first/README.md)); reviewed in plan §7, no learning claim. R/S/T and all failed learning gates remain open.
- Item 6 R2 inverse-domain repair: `348fb86c7` + `90ef40a42`, pushed on main, WikiOracle `066fb5d` ([receipt](doc/benchmarks/2026-10-10-item6/identity-r2-inverse-domain/README.md)); accepted in plan §10. Reconstruction certificates and observer repaired; no learning claim, and part 2's lessons remain open.
- Item 6 corrected-objective warm-up: accepted `345734c38`…`d5d300aed` is pushed on main, with WikiOracle bump `7bf96de` ([receipt](doc/benchmarks/2026-10-15-item6-corrected-warmup/README.md)); accepted in plan §13 as a measurement, with no learning claim. The observation-only singleton gate was 8/24; paired training is measured separately in part 2.
- Item 6 R/S/T mechanism and matched measurement: `e38a19fbd`, WikiOracle `64a06d3` ([receipt](doc/benchmarks/2026-10-10-item6/identity-rst/README.md)); reviewed in plan §8.4, no learning claim. All failed learning gates remain open.
- Item 6 part-2 corpus and measurement preparation: `06f6e2b88` ([receipt](doc/benchmarks/2026-10-10-item6/identity-corpus/README.md)); M/N/O/I and the first experiment are accepted as preparation by plan §7, with no learning claim.

- **Item 6 H/I follow-up** (`9cc5f6e327`): unavailable reconstruction coverage is zero; I’s empty-inverse and remaining orientation causes are bisected, with learned order still open ([receipt](doc/benchmarks/2026-10-10-item6/followup/README.md)).

- **Item 6 part 1** (`324b84d67f`): shared stored/live inverse, dropped-support keep charge, cache recovery, captured-input isolation, family-order retirement and residual/greedy components; part 2 and measured failures stay open ([receipt](doc/benchmarks/2026-10-10-item6/part1/README.md)).

- **Item 6.1** (`4d9d9ca8a`): support masks, one read per placement, eight-space STM with witnesses, derivation reconstruction and the order lesson; curriculum harness with identity/permanence and word identification passing; learning gates at item 0 ([acceptance and receipt](doc/benchmarks/2026-10-09-item6-1/README.md)).
- **Item 6.2 closed** (initial mechanism `e43638a`; Alec accepted the closing October 9, thinking §14.13): mechanism complete, decomposition demonstrated; learning deferred to item 0 under the four protocol corrections—one-step problems first, worked steps as intermediate answers, binding a found candidate, and budget in steps with each query’s record charge bounded separately ([acceptance and receipt](doc/benchmarks/2026-10-09-item6-2-closing/README.md)).

- **Item 6.5 mechanism** (Alec accepted October 7; one landing from `f4a68404e`): definedness, native identity/change columns and global bind/mint; learning gates remain pending the million-sentence checkpoint ([acceptance](doc/benchmarks/2026-10-07-item6-5/acceptance.json)).
- `f4a68404e` **Operators update, final-b** (Alec accepted October 7): addressed rows, bipolar meanings, centroids, priming, catalogue and two-lane repair; all standing counts 10/10, R/E zero, green sweep ([acceptance](doc/benchmarks/2026-10-07-operators-final-b/acceptance.json)).
- `cce3a4f7b` Operators round 3a: identity by construction ([receipt](doc/benchmarks/2026-10-07-operators-round3a/README.md)).
- `3de37eef5` Operators round 2: accepted credit repair ([receipt](doc/benchmarks/2026-10-06-operators-round2e/README.md)).
- `73cd7b71b` Operators round 1 accepted and pushed, with WikiOracle bumped at `c9670b5` ([receipt](doc/benchmarks/2026-10-05-operators-update/README.md)).
- `42daf96f` Item 6.8 accepted: one attention, two spaces and one index; tag `6.8-landing`; class **7/10**, reconstruction **9/10**, joint **6/10**, MM_xor **10/10**, sum **10/10**, sweep **4,917 passed / 0 failed**, zero ownership conflicts ([receipt](doc/benchmarks/2026-10-03-operators-attention/README.md)); carried work: [FutureWork](doc/FutureWork.md#carried-from-the-68-1315-rounds-2026-10-04) and the [operators update plan](doc/plans/2026-10-05-operators-update.md).
- `bfe6a0d7` Item 6.9: the baseline of the composition mechanism (6.8 §12.1), one reconstruction/output decoder and one writer per weight; accepted class MSE .11475 and reconstruction 0/4 remain historical, with zero ownership conflicts ([closing receipt](doc/benchmarks/2026-10-03-item6-9-closing/README.md), [round history](doc/benchmarks/2026-10-03-item6-9-closing/todo-history.md)).
- `9810fc7` Item 7: two truths, indexed definitions, ended clause state and shared row-free predicates; accepted under review §25 after all three fixture ports and required checks ([landing receipt](doc/benchmarks/2026-09-30-item7-review-round5/landing/README.md)).
- `6906727` Item 7.5: one-operation exploit/explore derivations trained at each sentence closing, reduction pressure/deadlines and closing-gradient reporting; accepted with the unchanged depth-three campaign red ([receipt](doc/benchmarks/2026-09-27-item7-5-landing/README.md)).
- `8bc710a` Item 9b: shared fields, parallel-first context, association-first interpretation, fixed capacities and corrected occurrence/time objectives; Claude accepted the source-matched 4,948-case sweep ([receipt](doc/benchmarks/2026-09-26-item9b-occurrence-fix/README.md)).
- `8bc710a` Item 9 evaluation machinery: verified FineWeb exposure and maturity-gated wording/prediction/predictive-thought checks; empirical acceptance remains open above ([receipt](doc/benchmarks/2026-09-26-item9-followup/README.md)).

- `0e70001` Item 9 parity landing: sentence-owned reconstruction candidates, final saved roots, first-sight bank checks without scalar host reads, and the historical-answer warning ([receipt](doc/benchmarks/2026-09-25-item9-bank-sync/README.md)); expectation-learning gates remain open.

- `99207a3` Item 10 decided: dropped; normalized means deleted, native XOR and forward-evidence reconstruction pass unseeded, with mode exclusions asserted ([receipt](doc/benchmarks/2026-09-25-item10-forward/README.md), [evaluation](doc/benchmarks/2026-09-24-item10/README.md)).
- `99207a3` Item 11c residue: accepted three-update refine-before-raise patience, reset by strict improvement and cleared by pure/unknown reads ([receipt](doc/benchmarks/2026-09-25-item10-forward/README.md)).
- `d5f2e1a` Item 11c: native perception, located XOR, per-pole witnessing, nonzero focus and grammatical reference across orders ([receipt](doc/benchmarks/2026-09-24-item11c/README.md)).
- `15c9bde` Item 11b extent correction: containment at the subject extent, observed raw spans and the restored positive-only `MM_sparse_concept` smoke ([receipt](doc/benchmarks/2026-09-24-item11b-extent/README.md)).
- `aa235d2` Item 11b accepted-review corrections: recurrent parts, containment over canonical ids, and order-0 fields bound per turn with persistent concept identity ([receipt](doc/benchmarks/2026-09-24-item11b-corrections/README.md)).
- `db73581` Item 11b: native membership definitions, located fused parts and pervading wholes, with alternatives preserved through refinement; normalization residue closed ([review receipt](doc/benchmarks/2026-09-24-item11b-review/README.md), [design](doc/Architecture.md#item-11b-membership-read-and-extent-truth-corners-september-23)).

- `4b097dc` Item 11a learns WholeSpace properties over byte primitives, retains grounded extent evidence and passes six unseeded native XOR runs ([receipt](doc/DevelopmentHistory.md#item-11a-primitive-properties-and-grounded-extents-2026-09-23)).
- `e917d9c` Item 11 review corrections carry paired evidence through scoped dual folds, symbols, thought and checkpoints; calibrated background rejection and the full receipt pass, learned XOR remains null for item 11a ([receipt](doc/DevelopmentHistory.md#item-11-review-corrections-paired-evidence-september-23)).
- `eb45ae4` Item 11 lands the union default, optional conjunction pass and context-backed provisional rows; unseeded XOR learning remains null ([receipt](doc/DevelopmentHistory.md#item-11-concept-parts-and-provisional-rows-september-22)).
- `d4dc385` Item 10 (the first of that number) records the reconstruction baseline and packed/single parity gap, fixes MentalModel compaction overflow, and removes selected passing seeds (taken early from item 2; [receipt](doc/DevelopmentHistory.md#item-10-reconstruction-baseline-and-seed-audit-september-21)).
- `f8aa23c` Item 11 completes generation catalogue ownership, scoped output dispatch, checkpoint/Adam migration and supervised gradient boundaries ([receipt](doc/DevelopmentHistory.md#item-11-generation-ownership-september-21)).
- `afdcdfa` Item 12 completes the expectation review corrections: role-normalized retention surprise, indexed pairs, reading purity, bounded cache recovery and gradient norm ratios ([receipt](doc/DevelopmentHistory.md#item-12-review-corrections-september-21)).
- `7d7dc4f` Item 2 implements negative-image expectation and residual credit; measured joint/useful-query learning remains open under item 9 ([design](doc/ExpectationRetention.md), [receipt](doc/DevelopmentHistory.md#negative-image-expectation-september-21)).
- `6bf211a` Item 1d restores rotation-owned concept codes and completes the 1b/1c review corrections ([GradientFlow](doc/GradientFlow.md), [receipt](doc/DevelopmentHistory.md#item-1d-review-corrections-september-21)).
- `101dc22` Item 1c complete in mechanism: checked subsystem effects, indexed cued retrieval and retained frames; compound recovery remains unproven ([AccessibleMind](doc/AccessibleMind.md), [receipt](doc/DevelopmentHistory.md#accessible-mind-effects-september-20)).
- `7c2fa5a` Item 1b complete: objective-local state gradients and measured shared operator/codebook credit ([GradientFlow](doc/GradientFlow.md), [receipt](doc/DevelopmentHistory.md#gradient-factorization-september-20)).
- `6a6bb21` Item 1 complete: grammar-owned wording on real parsed text, held-out complete wordings/nouns, converse/sense controls and full-vocabulary generation/recomposition ([evidence and receipt](doc/DevelopmentHistory.md#working-grammar-wording-gate-september-20)).
- `d1d8b5b` Ordered compose-MLP inputs and production generate wiring (item 1 increment; the wording gate remains open, [SelectedMeaning](doc/SelectedMeaning.md), [Testing](doc/DevelopmentHistory.md#grammar-owned-wording-architecture-september-20)).
- `02db7ef` Review corrections: removed the standalone codec and legacy policy; completed chooser context, bounded memory and nested-controller mechanisms ([SelectedMeaning](doc/SelectedMeaning.md), [Testing](doc/Testing.md)).
- `288b56b` Initial controller replacement; review reopened item 1's wording gate ([current design and limits](doc/SelectedMeaning.md), [original receipt and review corrections](doc/DevelopmentHistory.md#selected-meaning-and-one-controller-september-20), [test dispositions](doc/DevelopmentHistory.md#frame-kernel-test-migration)).
- `60b1497` Forward owns anchored lexical forms by WORD row; capture reads retained provenance ([Testing](doc/DevelopmentHistory.md#forward-owned-lexical-forms-september-19); item 1 increment).
- `a70c77d` Direct `concept`-unary selected meaning (`what(quantize(x))`)
  preserves its live leaf into the normal controller; description operands
  still require owned occurrences. Ordinary pytest failures now complete their
  selected receipt while resource/protocol failures remain bounded/fail-fast
  ([Testing](doc/DevelopmentHistory.md#complete-diagnostic-receipts-september-19)).
- `943a64f` Lexical converse forms: `whole` vs `part` provenance frozen at
  program capture ([Testing](doc/DevelopmentHistory.md#forward-owned-lexical-forms-september-19)
  records the later forward-ownership correction).
- `fb5611c` Normal thought resolution grammar-owned; `resolveAnswer()` no
  longer calls `answer_query()` from a raw prompt (item 1 increment).
- `ea92851` **Item 0 complete:** model-declared `<thought>` allow-list between
  `<compose>` and `<generate>`; `<Queries>`, the `is`-aliases and
  `query="false"` retired. [Spec](doc/specs/2026-09-18-thought-operations-in-compose.md),
  [plan](doc/plans/2026-09-18-thought-operator-unification.md),
  [receipts](doc/DevelopmentHistory.md#explicit-thought-catalogue-september-19).
- `573fe0c` Bounded tests fast: no background QoS, `cpu−4` one-thread pool,
  28 GiB aggregate cap; 4,649 cases in 197 s (was 4,165 s serial).
- `6a34bbf` Sentence expectations retained as `estimate` occurrences without
  LTM leakage ([ExpectationRetention](doc/ExpectationRetention.md); item 2 foundation).
- `c3918a2` `9dc5c58` Typed `ThoughtResult` and nested evidence persist through
  history checkpoints (sidecar v3; [ThoughtHistory](doc/ThoughtHistory.md)).
- `e8b2f43` Catalogue refinement preserves mode, polarity, bindings, scope.
- `b705a48` `69233d1` Selected truth and `arma` results become the row's answer
  seed; `arma` emits `[3, D]`, never a fact.
- `04b34a3` Normal What runs on the selected thought trace.
- `4518db0` Structural and thought operators unified (predecessor of `ea92851`).
- `7923845` Shared selected-query work meter ([QueryWork](doc/QueryWork.md)).
- `3ab3c4b` Nested grammatical occurrences retained ([NestedRetention](doc/NestedRetention.md)).
- `3a5703e` Checked query execution phases ([QueryPhases](doc/QueryPhases.md)).
- `985b594` Replayable ordinary thought history ([ThoughtHistory](doc/ThoughtHistory.md)).
- `3ffb465` Bounded runner foundations, 512 MiB VQ tile, MPS routing receipt.
- Expectation review (plan §11): implemented and validated on September 16 ([record](doc/plans/2026-09-15-next-sentence-as-the-production-objective.md#116-implementation-and-validation-september-16)).
- Xcode licence accepted; trust term sign `|trust|` (Alec, September 18).
