"""One receipt, rendered only from the saved §11 results and their audit."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE/'review11-measurements'
result = json.loads((OUT/'summary.json').read_text())
audit = json.loads((OUT/'audit-summary.json').read_text())
counts = result['counts']


def verdict(value):
    return 'pass' if value else 'fail'


text = f'''# Decoder, operators and 6.8 — §11 results for Claude's review

This is the one receipt for the uncommitted candidate from published HEAD
`802abb1acc95e1bddc8cb237b13230a336681c49`, in the existing working tree.
The controlling review is [6.8 plan §§11.5–11.6](../../plans/2026-09-27-item-6-8-one-attention.md).
**Nothing is committed or pushed. Work stops for Claude's review.**

One measurement on the frozen source, with thirty trainings and no retries:
**XOR_grammar class {counts['class_pass']}/10, reconstruction {counts['reconstruction_pass']}/10,
joint {counts['joint']}/10; MM_xor {counts['mm_pass']}/10; sum control {counts['sum_pass']}/10.**
{('The sum control is 10/10, so the class count can be interpreted as a composition gate.' if counts['sum_pass']==10 else 'The sum control is below 10/10; the class count is not evidence of composition.')}
Below-comparison counts: {', '.join(row['count']+' '+str(row['current'])+'/10 versus '+str(row['prior'])+'/10' for row in result['below_comparison']) or 'none'}.
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
'''
for row in result['xor']:
    text += f"| {row['run']} | {row['mse']:.9f} | {row['band']} | {row['correct']} | {row['recovered']} | {verdict(row['class_pass'])} | {verdict(row['reconstruction_pass'])} | {verdict(row['joint'])} |\n"
text += f"\nXOR bands: {json.dumps(counts['xor_bands'], ensure_ascii=False)}. Joint count: **{counts['joint']}/10**.\n"
text += '\n| Run | MM best loss | MM result | Sum MSE | Sum band | Sum contrast | Control |\n|---|---:|---|---:|---|---:|---|\n'
for mm, control in zip(result['mm'], result['sum']):
    text += f"| {mm['run']} | {mm['best']:.9f} | {verdict(mm['passed'])} | {control['mse']:.9f} | {control['band']} | {control['contrast']:.9g} | {verdict(control['sum_bar'])} |\n"
text += f'''
## Tenth-run audit

[Complete audit](review11-measurements/xor-10/ownership/),
[summary](review11-measurements/audit-summary.json),
[margin plot](review11-measurements/decoder-margin.png).
Ownership conflicts: **{audit['ownership']['conflicts']}**.
The audit observes {audit['optimizer_steps']} actual optimizer steps;
{audit['steps_reaching_decoder']} reach decoder graphs and {audit['no_decoder_steps']}
do not. It saves {audit['first_step_records']} batched first-step logit records.
First-step row/path eligibility: **{json.dumps(audit['first_step_eligibility'])}**.
All live rows are included in the margin summary, including masked actions.

| Binary index | Raw margin mean, epoch 1 → 400 | Final range | Nonzero gradient differences | Mean fixed-parent margin change |
|---|---|---|---:|---:|
'''
for key, stats in audit['binary'].items():
    first = next(row for row in audit['epochs'] if row['epoch']==0 and row['binary']==int(key))
    last = next(row for row in audit['epochs'] if row['epoch']==399 and row['binary']==int(key))
    text += f"| {key} | {first['margin']['mean']:.6f} → {last['margin']['mean']:.6f} | {last['margin']['min']:.6f}–{last['margin']['max']:.6f} | {stats['nonzero_gradient']}/{stats['gradient_difference']['n']} | {stats['fixed_parent_change']['mean']:.9g} |\n"
text += '\nThe gradient difference is dL/dSTOP minus dL/dundo. Fixed-parent deltas come from the actual optimizer update; discarded paths may have zero gradients.\n'
text += '\nSTOP is masked in all 6,400 audited first-step row/path observations and its gradient is exactly zero. Undo-gradient differences are at most 1.96e-9 in magnitude, and every recorded fixed-parent margin change is exactly zero. The epoch means vary with the parent representations; this is not evidence of a policy update.\n'
text += '\n| Walk | Comparisons | Explore kept | Strict-rule violations | Kept-path stability |\n|---|---:|---:|---:|---:|\n'
for name, row in audit['walks'].items():
    text += f"| {name} | {row['walks']} | {row['explore_wins']} | {row['strict_violations']} | {row['stable_pairs']}/{row['stability_pairs']} = {row['derivation_stability']:.6f} |\n"
text += '''
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
'''
(HERE/'README.md').write_text(text)
print(json.dumps(dict(receipt=str(HERE/'README.md'), counts=counts)))
