# Decoder, operators and 6.8 — §11 results for Claude's review

This is the one receipt for the uncommitted candidate from published HEAD
`802abb1acc95e1bddc8cb237b13230a336681c49`, in the existing working tree.
The controlling review is [6.8 plan §§11.5–11.6](../../plans/2026-09-27-item-6-8-one-attention.md).
**Nothing is committed or pushed. Work stops for Claude's review.**

One measurement on the frozen source, with thirty trainings and no retries:
**XOR_grammar class 2/10, reconstruction 6/10,
joint 2/10; MM_xor 10/10; sum control 10/10.**
The sum control is 10/10, so the class count can be interpreted as a composition gate.
Below-comparison counts: class_pass 2/10 versus 9/10.
No measured run was omitted, repeated or repaired after seeing its result.

The accepted 6.9 baseline stays **MSE .1147481948, reconstruction 0/4,
zero ownership conflicts**. The previous accepted counts are §22's class
**9/10**, reconstruction **5/10**, sum control **10/10**. MM_xor was red
through 6.9 §17; the §11 comparison requested for it is the preceding §10
measurement's **10/10**, whose reader was then under review. This is 6.8 work.
There is no conference freeze.

The [original round](README-before-review10.md) and the
[complete §10 receipt](README-before-review11.md) remain preserved.
§10 measured class 7/10, reconstruction 0/10, MM 10/10, sum 0/10.
Claude's §11 review rejected the nonlinear numeric reader and replaced the
decoder policy-credit proposal with eligibility. The original single-run
class comparison remains **not a regression finding**.

## What changed

1. `_forward_head` no longer calls `PrimedSymbolReader`, for any configuration.
   Numeric answers retain the detached root/end slots and the existing affine
   priming-weighted code sum. Both actual `XOR_grammar` and `MM_xor` numeric
   routes are checked with retrieval replaced by a failing sentinel.
2. The learned scorer and consume gate remain in generation and thought.
   `reverseOutput` reads into its owned operand; reconstruction's source
   carrier is preserved. Thought uses only its row's already retrieved STM/LTM
   records and the existing shared meter; semantic content is consumed while
   metadata stays unchanged. Keys stay detached. Generate trials use copied
   allowances and the concluded generation charges one read.
3. Decoder eligibility now uses the existing pair-search residual against the
   best single-code least-squares explanation (the signed activation used in
   `readback_scores`). Both children must be nonzero valid shortlist codes,
   neither may repeat the parent, and the pair must explain it at least as
   well as one code. Equal fits prefer the two readable parts, even if a
   larger chunk code is also present. A supported pair masks STOP and unary
   rewrites. Otherwise a unique nearest one-code explanation emits; tied
   identities are not singular. No support or no stack room leaves work pending.
   **This residual comparison and progress check are the implementation of
   §11.6's readability rule, submitted for review. No new cutoff is used.**
   The policy still learns between eligible operations, as the real-op
   gradient probe demonstrates. No advantage loss was introduced.

Shipped `answerSynthesis=true` configurations using generation retrieval:
`BasicModel.xml`, `BasicModel_answers_tied_benchmark.xml`, `MM_add.xml`,
`MM_add_verb.xml`, `MM_math.xml`, `MM_ladder.xml`, `MM_ladder_idiom.xml`,
`MM_grammar_wording.xml`. Other configurations can explicitly call
`reverseOutput`; `MM_global` and `MM_qa` numeric heads also bypass retrieval.
The retrieval probes for those fixtures now use the generation consumer.

§10's sampled departures remain in every walk: one uniformly chosen eligible
round, action sampled from remaining eligible policy probabilities, prefix
replayed and suffix greedy, both paths costed at the same parameters, explore
kept only for strictly lower owner cost, ties to greedy. The raw decoder
STOP/undo margin and actual gradient/update audit remain. A positive raw STOP
margin does not imply that STOP is eligible under §11.6.

Frozen evaluation still admits no definitions or reservations. The small
three-item `MM_grammar_wording` evaluator mechanism check and the
[acceptance probe](nanochat_acceptance_probe.py) are retained. **Fresh BasicModel
scoring remains stopped; the trained NanoChat gate waits for item 4's checkpoint.**

## Thirty trainings on one frozen source

[Plan](review11-measurements/plan.json), [completion](review11-measurements/complete.json),
[raw results](review11-measurements/summary.json),
[source archive](review11-source/manifest.json). The sum controls were scheduled
first. Each XOR run trained once for the original 400 epochs and supplied
both original assertions; the observer verified one model identity for both
consumers. The tenth XOR run also supplied the ownership and walk audits.
MM uses its unchanged convergence test (best loss < .20, up to 200 epochs).
Sum uses the original sum-only grammar substitution and the unchanged control:
absolute checkerboard contrast ≤ 1e-4 and failure of the class bar.

Class requires all four labels and MSE < .05. Reconstruction requires all
four word multisets and no unavailable inverse; the historical selector
containing `50_pct` is unchanged. The §20.5 reporting bands stay: at 0 means
MSE < .05; at ¼ means within .02 of .25; the remaining errors fall between or
above ¼. Bands do not replace the gates.

| Run | XOR MSE | Band | Labels /4 | Reconstructed /4 | Class | Reconstruction | Joint |
|---|---:|---|---:|---:|---|---|---|
| 1 | 0.250024897 | at 1/4 | 2 | 4 | fail | pass | fail |
| 2 | 0.000021170 | at 0 | 4 | 4 | pass | pass | pass |
| 3 | 0.232321466 | at 1/4 | 4 | 4 | fail | pass | fail |
| 4 | 0.250000087 | at 1/4 | 2 | 4 | fail | pass | fail |
| 5 | 0.199993321 | between | 4 | 3 | fail | fail | fail |
| 6 | 0.250001577 | at 1/4 | 3 | 3 | fail | fail | fail |
| 7 | 0.000189622 | at 0 | 4 | 4 | pass | pass | pass |
| 8 | 0.201953973 | between | 4 | 3 | fail | fail | fail |
| 9 | 0.125663548 | between | 4 | 4 | fail | pass | fail |
| 10 | 0.250078655 | at 1/4 | 2 | 3 | fail | fail | fail |

XOR bands: {"at 0": 2, "at 1/4": 5, "between": 3, "above 1/4": 0}. Joint count: **2/10**.

| Run | MM best loss | MM result | Sum MSE | Sum band | Sum contrast | Control |
|---|---:|---|---:|---|---:|---|
| 1 | 0.193371147 | pass | 0.250006080 | at 1/4 | 0 | pass |
| 2 | 0.177499026 | pass | 0.250000030 | at 1/4 | 0 | pass |
| 3 | 0.157256484 | pass | 0.250000298 | at 1/4 | 2.98023224e-08 | pass |
| 4 | 0.199056327 | pass | 0.250000179 | at 1/4 | 2.98023224e-08 | pass |
| 5 | 0.195457578 | pass | 0.250000119 | at 1/4 | 2.98023224e-08 | pass |
| 6 | 0.190840751 | pass | 0.250040084 | at 1/4 | 2.98023224e-08 | pass |
| 7 | 0.187706605 | pass | 0.250011086 | at 1/4 | 0 | pass |
| 8 | 0.194826871 | pass | 0.250000000 | at 1/4 | 0 | pass |
| 9 | 0.197452113 | pass | 0.250000417 | at 1/4 | 0 | pass |
| 10 | 0.178401202 | pass | 0.250007957 | at 1/4 | 2.98023224e-08 | pass |

## Tenth-run audit

[Complete audit](review11-measurements/xor-10/ownership/),
[summary](review11-measurements/audit-summary.json),
[margin plot](review11-measurements/decoder-margin.png).
Ownership conflicts: **0**.
The audit observes 1200 actual optimizer steps;
800 reach decoder graphs and 400
do not. It saves 1600 batched first-step logit records.
First-step row/path eligibility: **{"compound": 6400}**.
All live rows are included in the margin summary, including masked actions.

| Binary index | Raw margin mean, epoch 1 → 400 | Final range | Nonzero gradient differences | Mean fixed-parent margin change |
|---|---|---|---:|---:|
| 0 | 2.002780 → 2.016802 | 1.979733–2.057335 | 91/6400 | 0 |
| 1 | 1.994078 → 1.980657 | 1.950680–1.999889 | 91/6400 | 0 |

The gradient difference is dL/dSTOP minus dL/dundo. Fixed-parent deltas come from the actual optimizer update; discarded paths may have zero gradients.

STOP is masked in all 6,400 audited first-step row/path observations and its gradient is exactly zero. Undo-gradient differences are at most 1.96e-9 in magnitude, and every recorded fixed-parent margin change is exactly zero. The epoch means vary with the parent representations; this is not evidence of a policy update.

| Walk | Comparisons | Explore kept | Strict-rule violations | Kept-path stability |
|---|---:|---:|---:|---:|
| attention.input | 1600 | 0 | 0 | 1586/1596 = 0.993734 |
| generate.decoder | 3200 | 0 | 0 | 1085/3196 = 0.339487 |
| compose | 1600 | 0 | 0 | 1591/1596 = 0.996867 |

The full audit saves row-wise paths and marginal distributions. No additional
training, attribution run, full sweep or completed full-model native run was
used to explain the counts. The optional full-model compile probe described
below was interrupted before this campaign, and is not a passing result.

## Verification and complete ports

The [review-start archive](review11-before/manifest.json) predates repairs.
[Contract audit](review11-contracts-final.json) retains complete old/new bodies of
this round's test ports and proves their existing assertions and seed calls
unchanged. [Complete published-HEAD ports](review11-final/test-ports.json)
cover **163** changed or new test files, with **zero changed seed calls**.
The protected class/reconstruction/MM assertions, test guards, Makefile,
pytest.ini and NanoChat manifest are byte-identical to HEAD. No XML or capacity
changed in §11. The original nine word-whole row changes remain disclosed in
the initial round's archived receipt.

Old STOP/undo-always-eligible assertions are preserved in explicitly isolated
walk-control fixtures with a declared eligibility oracle. They test replay,
sampling, numerical credit and trace isolation. The new real-operation probes
test production eligibility, including compiled parity, a readable chunk's
two children, a singular symbol, empty support, capacity exhaustion and
gradient credit between two admissible operations. Retrieval tests preserve
their assertions and move the consumer from the numeric head to generation.

| Saved probe | Outcome and disposition |
|---|---|
| [before-fixes](probes/review11-before-fixes/run.log) | Four decoder failures; one probe import error, saved before repair |
| [numeric-before-fix](probes/review11-numeric-before-fix/run.log) | Corrected probe reproduces nonlinear numeric retrieval |
| [first-repair](probes/review11-first-repair/run.log) | 40 pass; 11 old eligibility-fixture failures, saved before ports |
| [reader-port-before](probes/review11-reader-port-before/run.log) | 9 pass; 3 old numeric-consumer assertions fail after routing repair |
| [focused-ports](probes/review11-focused-ports/run.log) | 43 pass; one new gradient-fixture loss is orthogonal to the action difference; optional full-model compilation interrupted after about four minutes |
| [focused](probes/review11-focused/run.log) | 127 pass; 3 generation-fixture carrier lookups fail; optional full-model compilation deselected |
| [reader-ported](probes/review11-reader-ported/run.log) | All 12 retrieval checks pass after the fixture carrier fallback; no production repair |
| [observer](probes/review11-observer/run.log) | One real-batch margin/ownership wiring check passes |
| [echoic-before-port](probes/review11-echoic-before-port/run.log) | 7 pass; two legacy nonzero chooser-gradient assertions fail when eligibility leaves one action; the real hard traces pass |
| [final-fixtures](probes/review11-final-fixtures/run.log) | 9 echoic checks and 3 strengthened output fixtures pass, including compiled cases |

Together the final focused, reader and echoic checks cover **139 distinct passing
checks**. [Source bridge](review11-source/probe-source-bridge.json) proves
their production source matches measurement; only the final test-fixture
carrier lookup differs in the earlier focused run. Compiled decoder probes
passed; the interrupted optional full-model compilation is not counted.

After the campaign, two test fixtures were ported without changing any model
or gate code. The echoic fixture first checks its real masked hard trace, then
uses the declared choice oracle for its historical nonzero-gradient assertion;
the output fixtures supply explicit shortlist support so compilation and state
isolation are checked on emitted values. The
[final source bridge](review11-final-bridge.json) names both test-only differences
and proves the production, data and protected gate source still match the thirty
trainings. No measured model was repaired or retrained.

Every measured process keeps its exit, log, final arrays and assertions.
Source hashes are checked throughout. Guards remain **8 GiB / 1,800 seconds
per worker**, with at most three one-thread workers reserving **24 GiB** under
the existing **28 GiB** ceiling and CPU headroom. No seeds, bars, assertions,
optimizer settings or budgets were changed. One working tree; old prunable
Git registrations were not altered. No commits or pushes.
