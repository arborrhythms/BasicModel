# Item 6.9 work receipt

Work starts from published BasicModel HEAD
`d679df2b5a2665d72a99ca4b6dfd47c1ba048e99`. Item 7 is accepted and landed.
This is an uncommitted candidate for Claude's review, following
[the item 6.9 plan](../../plans/2026-09-29-item-6-9-xor-grammar.md), §4 in order.
Implementation has stopped after step 5 under the mandatory ten-run bar.
The class bar was met in **0/10 fresh unseeded runs**, with MSE from
.2044415629 to .5855241919. The current reconstruction gate also failed
10/10 runs. Steps 6–8 remain unimplemented, including the sum-only negative
control. The final source-matched full sweep completed all 5,161 cases:
4,832 passed, six failed, 322 skipped, and one was expected to fail.
The six failures remain unresolved for review.

## Measurement contract

Each full XOR table has 49 named attempts, including both slow XOR_exact
CLI assertions, both XOR_grammar gates, the MM_20M_xor harness-budget
round trip, and fifteen independent exact round trips. HEAD is an exported
copy of the published commit. Candidate and HEAD source hashes are checked
throughout each campaign. Source edits wait for the campaign to finish.
All attempts, including statistical failures, are retained.

No seed is pinned. The existing model configurations, 400-epoch XOR_grammar
budget, 900-update MM_grammar measurement, optimizer settings, and 8 GiB
worker guard stand. Measurement dispatch also retains the 24 GiB aggregate
guard. All ten class trials after step 5 must have four correct answers and
MSE below .05; otherwise implementation stops before steps 6–8.

Each linked table names every proof. Both XOR_grammar gates fail at every
checkpoint; all statistical outcomes are retained.

| Checkpoint | HEAD passed / failed | Candidate passed / failed | HEAD exact | Candidate exact |
|---|---|---|---|---|
| [Before changes](before/comparison.md) | 45 / 4 | 45 / 4 | 13/15 | 13/15 |
| [Step 1](step1/comparison.md) | 44 / 5 | 44 / 5 | 13/15 | 12/15 |
| [Step 2](step2/comparison.md) | 42 / 7 | 42 / 7 | 11/15 | 10/15 |
| [Step 3](step3/comparison.md) | 44 / 5 | 45 / 4 | 12/15 | 13/15 |
| [Step 4](step4/comparison.md) | 45 / 4 | 46 / 3 | 13/15 | 14/15 |
| [Step 5](step5/comparison.md) | 44 / 5 | 43 / 6 | 12/15 | 11/15 |

## Starting evidence

- [Published source and preserved historical records](starting-record.json).
- [Full named XOR table before changes](before/comparison.md): both fresh
  source tables have both XOR_grammar gates red and exact round trips 13/15.
- [Ten full MM_grammar runs before changes](before/mm-table.md): median
  ending MSE .0647774097 on HEAD and .0008435599 on the identical candidate.
  These are fresh unseeded measurements, not paired initializations.
- [Accepted historical XOR table](../2026-09-30-item7-review-round5/final-xor-comparison.md)
  retains the red XOR_grammar gates and 12/15 candidate exact round trips.
- [Accepted historical MM_grammar table](../2026-09-30-item7-review-round5/final-mm-grammar-table.md)
  and [item 7.5 receipt](../2026-09-27-item7-5-landing/README.md), including
  the red depth-three campaign, remain unchanged.

## Repairs and probes

Step 1 reads the four answers stored by the immediately-after-training
evaluation, requiring four correct classes and MSE below .05. It adds no
forward pass or clock tick. The report's embedding placeholder is unchanged.
[Failing probe](step1-red/run/result.json),
[passing probe](step1-green/run/result.json), and
[complete old/new bodies](step1-port/bodies.json) retain the test port.

The baseline observer exposed another distinction: the report caches one
decoded string, while the reconstruction gate renders four rows. The
[saved observer failure](observer-red/run/result.json) precedes the
[observer port](observer-port/bodies.json), which now records both fields.
Rendering the already recovered state adds no model forward or training.
The original baseline observations are preserved.

The [full table after step 1](step1/comparison.md) has 44 passes and five
failures on each tree. HEAD has 13/15 exact round trips, both XOR_grammar
gates red, and a fresh MM_grammar convergence failure (best MSE .2233935
over 900 updates). Candidate has 12/15 exact round trips and both
XOR_grammar gates red. No runtime changed in step 1.

Step 2 reads the concluded one- or three-slot state into the existing
numeric map's parameter geometry and cuts its gradient. Named concept
outputs keep their identity carrier. The
[saved failure](step2-red/run/result.json),
[passing mechanism probe](step2-green/run/result.json),
[26 passing affected checks](step2-affected/run/result.json),
[complete old/new runtime bodies](step2-repair/bodies.json), and
[new regression test bodies](step2-new-tests.json) record this change.

The [full table after step 2](step2/comparison.md) has 42 passes and seven
failures on each tree. Exact round trips were 11/15 on HEAD and 10/15 on
the candidate; both XOR_grammar gates remain red. The exact fixture uses
parallel reading, so the changed serial answer branch is not entered by
that fixture. These independent unseeded outcomes are retained without
statistical retries.

Step 3 stages each admitted word's object row at the eager reading boundary
and supplies its full dictionary code times activation to the grammar in
mixing and aligned bindings. Separators, word extents, capacity, and the
perception inverse retain their existing behavior. The
[saved binding failures](step3-red/run/result.json),
[passing binding and gradient-cut probes](step3-green/run/result.json),
[15 passing affected checks](step3-affected/run/result.json),
[complete old/new runtime bodies](step3-repair/bodies.json), and
[new regression test bodies](step3-new-tests.json) record this change.

The [full table after step 3](step3/comparison.md) has 44 passes and five
failures on HEAD, and 45 passes and four failures on the candidate. Exact
round trips were 12/15 and 13/15 respectively. Both XOR_grammar gates remain
red; the candidate class run has one correct answer and MSE .2662971479.

Step 4 makes the default concept-code `not` the full reflection `d -> -d`.
Explicit `representation='poles'` preserves the previous pole exchange and
the remaining occurrence coordinates. The
[saved negation failures](step4-red/run/result.json),
[operator repair bodies](step4-repair/bodies.json),
[saved old-test failures](step4-green-and-port-red/run/result.json),
[complete old/new test ports](step4-ports/bodies.json), and
[new regression test bodies](step4-new-tests.json) distinguish the two
representations without changing the fixture's rule set.
All [100 affected checks](step4-affected/run/result.json) pass, including
the native unlabelled-training check for the chooser's gradient.

The [full table after step 4](step4/comparison.md) has 45 passes and four
failures on HEAD, and 46 passes and three failures on the candidate. Exact
round trips were 13/15 and 14/15 respectively. Both XOR_grammar gates remain
red; the candidate class run has one correct answer and MSE .2749032638.

Step 5 costs both sentence trials before either optimizer step, then trains
each graph and keeps the strictly cheaper path in each row; ties keep the
greedy path. Existing saved-value hooks retain the original parameter
values. Each backward restores its own perception pullback and scratch
bindings, with no full model snapshot or prefix-sharing change. The
[fresh failures](step5-red/run/result.json) include a parameter change from
2.0 to 1.6 between identical trials, different costs in all four rows when
the real greedy derivation is repeated, and different parameter versions
in the aligned perception probe. The
[post-repair controls](step5-green-and-port-red/run/result.json) pass all
three new probes and the existing winner/commit checks. That run also
retains the old timing assertion's failure before its
[complete old/new port](step5-port/bodies.json): parameters are now equal
within each pair and advance before the next sentence. All row-winner and
next-perception assertions remain. The
[runtime bodies](step5-repair/bodies.json) and
[new regression tests](step5-new-tests.json) retain the exact changes.

The [broader affected run](step5-affected/run/result.json) then passed 32
checks and failed the eager and compiled packed-sentence checks: both cost
previews shared the staged prediction's autograd graph, which the first
backward freed. The [anomaly trace](step5-shared-red/run/worker-000.log)
locates its source in `predict_next_end_state`; the
[small predictor failure](step5-prediction-red/run/result.json) isolates
the same defect. Each preview now replays the predictor from the same
detached prior inputs, keeping the prior record, context and numerical
estimate intact while giving its loss a separate small graph. No whole
compose graph is retained to work around the error. The
[complete old/new body](step5-prediction-repair/bodies.json),
[passing predictor and eager packed probes](step5-prediction-green/run/result.json),
and [new regression body](step5-prediction-new-test.json) record the repair.
All [35 final affected checks](step5-affected-final/run/result.json) pass,
including compiled and eager packed sentences, prediction gradients,
independent perception pullbacks, identical-trial ties, winner commits,
and the numeric answer's gradient cut.

The [full table after step 5](step5/comparison.md) has **44 passed / 5
failed** on HEAD and **43 passed / 6 failed** on the candidate. Both
XOR_grammar gates are red. Exact round trips are **12/15** on fresh HEAD
and **11/15** on the candidate; the candidate's four exact-match failures
are retained, with no retries. Its starting table was 13/15, and the
preserved accepted historical candidate table was 12/15. This receipt
does not claim the no-regression exit bar has been met.

The [repeated ten-run MM_grammar table](step5/mm-table.md) retains all
900-update runs. Median ending training MSE is .1139670797 on HEAD and
.0686401706 on the candidate, compared with .0647774097 and .0008435599
in the two identical-source starting tables. The large unseeded variation
is visible in every table; these are not paired initializations.

The [ten-run table](grammar-ten/table.md) retains all four saved answers
and all four actual gate reconstructions for every class and reconstruction
trial. The [full-precision record](grammar-ten/summary.json) gives **0/10**
class-bar successes and the mandatory stop decision. One class run gets
all four labels right, but its MSE .2056156116 still fails the decided .05
bar. The reconstruction producer remains the existing perception reverse;
no claim of grammar reconstruction is made because step 6 was not reached.
There were no resource stops in these twenty trials. They overlapped the
tail of the step-5 table and native measurement, with the grammar guard
sampling all three process families through
[the combined companion view](companion_progress.py); peak combined memory
was 19.60 GiB. The original worker and aggregate limits were unchanged.

## Item 7.5 measurements

The fresh unseeded native timing run uses the unchanged current
`data/MM_ladder.xml`, two warm-up and five measured training batches,
and the original guard. It completed in 1224.15 seconds at 5.74 GiB peak
worker memory. Reconstruction before/during/after training was
.1230032034 / .1093158558 / .1057219785; warmed throughput was 1.2422741741
sentences/s. [Raw measurements](native-before/measurement.json),
[per-batch timings](native-before/batch-timing.json),
[resource receipt](native-before/driver.process.json).

The additional measurement immediately before step 5 includes steps 2–4
and still uses the original, unequal pair. It completed in 920.73 seconds
at 5.29 GiB peak worker memory. Reconstruction before/during/after training
was .1263466813 / .1104050681 / .1082045920; warmed throughput was
.7078896010 sentences/s. All five measured batches are retained, including
any later graph capture. [Measurements](native-prepair/measurement.json),
[timings](native-prepair/batch-timing.json), and
[resource receipt](native-prepair/driver.process.json) provide the direct
baseline for the pair correction. These measurements use fresh unseeded
initializations and do not establish a paired causal comparison.

The [final native measurement](native-after/measurement.json) completes in
1457.98 seconds at 5.349 GiB peak worker memory. Reconstruction
before/during/after training is .1149501931 / .1138415950 / .1189210640;
warmed throughput is .491403460 sentences/s. Against the immediately
preceding pair, the five measured warm batches take **1.4410×** as long
and whole-run peak worker memory is **1.0111×**. The
[full timing comparison](native-comparison.md) and
[per-batch record](native-after/batch-timing.json) retain late graph capture
and all measured batches. These observational ratios include different
unseeded initializations and concurrent workloads, not an isolated causal
allocation or speed measurement. The saved-value hooks avoid a full model
snapshot; prefix sharing remains future work.

The historical packed/single fixture cannot currently load on published
HEAD: `doc/benchmarks/2026-09-26-item7-5/parity.xml` still contains the
retired `WholeSpace.propertyBasis` element. The
[guarded failed attempt](parity-before-packed/driver.process.json) and
[loader traceback](parity-before-packed/driver.log) are retained. No
configuration was changed to work around this failure. The historical
parity result remains preserved; this attempt is not a new parity result.
The [final candidate attempt](parity-after-packed/driver.process.json)
fails at the same loader check, before model creation or a checkpoint.

The [explicit final depth-three check](depth3-after/run/result.json)
skips because no `BASICMODEL_FINEWEB_CHECKPOINT` containing at least one
million completed training sentences is supplied. Its depth-three assertion
and the historical red `[1, 1, 1, 1]` campaign remain intact. This prerequisite
skip is not a new passing depth-three measurement.

## Review status

Implementation stops after step 5 for Claude's review, without a commit.
The [full-sweep summary](full-sweep/summary.json) records **4,832 passed,
six failed, 322 skipped, and one expected failure** in 5,985.72 seconds
(99.76 minutes). The [coverage receipt](full-sweep/coverage.json) confirms
all 5,161 selected cases completed exactly once, with no missing,
duplicated or unreported cases. There were no resource stops or compile
cache retries. Peak worker memory was **5.002 GiB** and aggregate memory
**17.909 GiB**, under the unchanged 8/24 GiB guards.

The sweep adds 12 regression cases and 27 documentation-link cases over
the previous receipt, and removes none. All 39 additions pass. That older
full sweep predates the accepted item-7 fixture ports already present in
published HEAD; its six red-to-green outcomes are not repairs by this
candidate. The six failures below were passes in that earlier receipt.

The sweep has exposed an unchanged item-7 assertion in
`test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows`:
it expects `_word_symbol_concept_ids()` to be `None`. Step 3 now stages the
word's object references for unaligned reading too, and the returned IDs
are present. All preceding relation/definition assertions in that case
passed. The [saved failure](full-sweep/run/worker-059.log) remains red;
the test is not ported after the mandatory implementation stop.

The sweep also found a candidate compiled-path regression in
`test_compiled_understanding_captures_explicit_sentence_products`
([saved failure](full-sweep/run/worker-071.log)). The real full-graph entry
returns sentence products explicitly, but `_forward_per_stage` still calls
the new numeric head without supplying that explicit state. The head reads
`_stm_single_S` and raises because it is unavailable during this trace.
This failure remains unresolved; the passing eager and packed-brick checks
do not establish that the full-graph answer path works. No runtime repair
or assertion change is made after the mandatory stop.
The [full-graph query-mask check](full-sweep/run/worker-112.log) fails at
the same numeric-head guard before it can publish the explicit sentence
state. Both full-graph failures are retained independently.

Three further assertions remain red and require review of identity/evidence
semantics: [definition integrity](full-sweep/run/worker-075.log) finds
provisioned row ID `3` among the definition symbol IDs; the two
[truth-ingestion checks](full-sweep/run/worker-084.log) observe positive
evidence `1` where they require zero. Their assertions and fixtures are
unchanged. This receipt does not classify them as harmless ports or relax
their identity/evidence contracts.

The [717-file final source snapshot](final-source.json) and
[supporting inputs](final-inputs.json) match the completed full sweep;
the source also matches the final XOR/MM, twenty-trial grammar, and native
measurement manifests. The [final audit summary](summary.json) verifies
the original HEAD, unchanged configurations and four protected historical
records, and an empty staging area. The
[timing diagnosis](diagnostics/README.md) records the costly cold graph
capture and repeated independent model training. No commit, push, seed
selection, threshold relaxation, or configuration expansion has been made.
