# Item 6.9 — ownership continuation, 2026-10-02

**Closing dispatch complete; not accepted.** This is the single receipt
for plan §§15–17. MM_xor's unchanged `test_convergence` remains an unresolved
proof exempted by §17: its earlier consistent green result used promoted
chunks across the word boundary, one
percept per sentence. That was a lookup shortcut. True word-level MM_xor is
an acceptance test of item 6.8. Its saved probes below remain intact; no marker,
seed or bar was added or changed.

Work stayed in one working tree. Nothing was committed. HEAD remains
`d679df2b5a2665d72a99ca4b6dfd47c1ba048e99`. The starting candidate is saved in
[ownership-start.json](ownership-start.json) and [ownership-start.zip](ownership-start.zip).
The closing candidate is [source-final.json](source-final.json) and
[source-final.zip](source-final.zip). The continuation's production/configuration
changes are in [implementation.patch](implementation.patch), with a
[file inventory](implementation-files.json). These compare against the
continuation's starting candidate, rather than disguising prior uncommitted
work as new changes. No HEAD training or environment rebuild was run.
[Environment versions](environment-freeze.txt) are retained and the
[post-measurement freeze](environment-verification.json) is unchanged.

## Closing protocol and results

[closing_campaign.py](closing_campaign.py) runs the production native ownership
measurement first, then the named candidate measurements, conditional
attribution, moved slow cases, and one source-matched full sweep. The
[campaign plan](campaign-plan.json) fixes the source before the first run.
The ordinary worker ceiling stays 8 GiB. Only the production native stage-1
case uses the previously decided 24 GiB ceiling. Gate/table jobs use up to
three workers sharing 24 GiB; the full sweep starts on the reference schedule
of ten workers under this machine's ordinary aggregate reservation, reducing
concurrency only after an aggregate-memory stop. The native job and sweep do
not overlap. Guard-stopped or interrupted active cases are recorded, not
silently retried; continuation dispatches only cases never started.

There are ten fresh unseeded runs each of the class gate, reconstruction gate
and sum-only control. The predeclared first class and reconstruction runs
supply the corresponding XOR-table rows. Other named table cases run once,
including one MM_20M exact round trip. MM_grammar has ten final receipt runs.
Attribution runs ten each of R, R+E, R+A and R+E+A only if class is below 8/10
or reconstruction is no higher than 3/10. The saved receipt-local patches
change objective writers for attribution; they never introduce a production
switch. Sum passes only if its absolute checkerboard contrast is at most
1e-4 and it misses the class bar. The model's answers need not be exactly .5.

**Measurement caveat:** the observer made the predeclared first class run
import `util` with `MODEL_COMPILE=none`. The other pytest gates imported it
with `eager` before their helper changed the environment. No source, XML,
optimizer, seed or bar changed, but the capture policy differed. That first
outcome is retained, identified separately, and not retried. Equivalence of
these particular trainings was not measured. See
[measurement-caveats.json](measurement-caveats.json).

The unchanged reconstruction gate tests a free grammar inverse over known
object vocabulary, without original leaf witnesses. The trained reconstruction
cost uses the shared trial record, including witnesses and primed candidates.
A low tied cost alone therefore does not establish a passing free read-back.
Both results are retained; the gate is not weakened to match the training cost.

<!-- CLOSING_RESULTS_START -->
**Closing dispatch complete; results below require review.**

Native production stage-1: passed, 14.90 minutes guarded wall time, 21.400 GiB peak under the 24 GiB ceiling. This does **not** fit the ordinary 8 GiB ceiling. [Process](native-driver/process.json); [objective report](native-stage1/BasicModel_answers_tied_benchmark-ownership/summary.md).

| Campaign | Pass | Observed | Required runs |
|---|---|---|---|
| class | 1 | 10 | 10 |
| reconstruction | 1 | 10 | 10 |
| sum | 10 | 10 | 10 |


[Every answer, MSE, read-back and contrast](gate-results.md), with [raw values](gate-results.json). The first class result also supplies the XOR audit and has the capture-policy caveat above; it is not replaced.

XOR_grammar ownership: **0 conflicts** across 1200 recorded backwards; 18/73 parameter tensors reached. Inactive tensors remain listed. [Norms, cosines, term weights/magnitudes, selection and endpoint costs](xor-ownership/summary.md).

| Rank audit phase | Own-word occurrences | With activated competitor | Activated outranks own |
|---|---|---|---|
| train | 6400 | 0 | 0 |
| evaluation | 8 | 0 | 0 |


Native benchmark ownership: **0 conflicts** across 3 recorded backwards; 44/151 parameter tensors reached. Inactive tensors remain listed. [Norms, cosines, term weights/magnitudes, selection and endpoint costs](native-stage1/BasicModel_answers_tied_benchmark-ownership/summary.md).

| Rank audit phase | Own-word occurrences | With activated competitor | Activated outranks own |
|---|---|---|---|
| train | 168 | 0 | 0 |
| evaluation | 48 | 0 | 0 |


Native expectation was inactive: the sole training batch has no preceding sentence context and `intraLossWeight=0`. Its null/zero expectation measurements do not demonstrate a live expectation update. No activated surface competitor appeared in this run, so the ranking result has a zero competing denominator.

Named XOR table, completed groups: {'passed': 31, 'failed': 3}. The predeclared first class/reconstruction runs are reused.

| Group | Selector | Outcomes | Process result | Seconds |
|---|---|---|---|---|
| 0 | test/test_grounded_xor.py | {'passed': 6} | exit | 35.6156 |
| 1 | test/test_concept_output.py | {'passed': 10} | exit | 65.8897 |
| 2 | test/test_mm_xor.py | {'passed': 7} | exit | 328.803 |
| 3 | test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp | {'failed': 1} | exit | 42.5774 |
| 4 | test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct | {'passed': 1} | exit | 42.5777 |
| 5 | test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy | {'failed': 1} | exit | 108.99 |
| 6 | test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct | {'failed': 1} | exit | 194.924 |
| 8 | test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor] | {'passed': 1} | exit | 14.4756 |
| 9 | test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise] | {'passed': 1} | exit | 14.4666 |
| 10 | test/test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live | {'passed': 1} | exit | 14.4807 |
| 11 | test/test_reconstruction_roundtrip.py::test_xor_recon_grads_flow | {'passed': 1} | exit | 13.9449 |
| 12 | test/test_reconstruction_roundtrip.py::test_xor_percepts_tile_words | {'passed': 1} | exit | 15.6031 |
| 13 | test/test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget | {'passed': 1} | exit | 16.6119 |
| 14 | test/test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip | {'passed': 1} | exit | 106.643 |


MM_grammar: 10/10 complete; median ending training MSE **1.496212e-09**, with median evaluation MSE after 900 updates **2.372473e-09**. The comparable §14 ending training median was 4.35656e-11 (accepted item 7: .1066). This is the fixed 900-epoch direct-forward/raw-MSE measurement. It calls backward directly rather than the objective-owner dispatcher, so it does not validate joint-cost ownership training. It is also separate from the table test that can stop at its .20 bar.

| Run | Final training MSE | MSE after 900 updates | Final predictions |
|---|---|---|---|
| mm-01 | 2.21867e-12 | 5.2407e-12 | [1.043081283569336e-06, 1.000002384185791, 1.0000019073486328, 3.248453140258789e-06] |
| mm-02 | 3.25073e-13 | 3.2836e-12 | [4.76837158203125e-07, 1.0000019073486328, 1.0000011920928955, 2.8014183044433594e-06] |
| mm-03 | 0.000437343 | 0.000346227 | [-0.014306306838989258, 0.990951418876648, 0.9741520881652832, -0.020742416381835938] |
| mm-04 | 8.88596e-11 | 9.02232e-11 | [8.195638656616211e-06, 1.0000054836273193, 1.0000128746032715, 9.894371032714844e-06] |
| mm-05 | 1.1243e-11 | 1.35782e-10 | [1.6033649444580078e-05, 1.0000053644180298, 1.000015377998352, 4.559755325317383e-06] |
| mm-06 | 2.25819e-13 | 1.15663e-12 | [1.5497207641601562e-06, 1.000001311302185, 1.0000005960464478, 3.8743019104003906e-07] |
| mm-07 | 2.90357e-09 | 4.60916e-09 | [9.012222290039062e-05, 1.0000591278076172, 1.0000696182250977, 4.4405460357666016e-05] |
| mm-08 | 0.0190684 | 0.0189956 | [0.13814640045166016, 0.8644740581512451, 0.8628396987915039, 0.14042043685913086] |
| mm-09 | 9.83279e-05 | 5.20892e-06 | [0.002132892608642578, 1.001065731048584, 1.003232717514038, 0.002167999744415283] |
| mm-10 | 0.0280561 | 0.0279753 | [0.1648629903793335, 0.830040454864502, 0.8313652276992798, 0.16552233695983887] |


Attribution decision: {'required': True, 'counts': {'5': {'passed': 1, 'observed': 10}, '6': {'passed': 1, 'observed': 10}}}. [Decision record](attribution-decision.json).

| Attribution arm | Completed | Class passes | Reconstruction passes |
|---|---|---|---|
| R | 10 | 0 | 2 |
| RE | 10 | 0 | 4 |
| RA | 10 | 1 | 1 |
| REA | 10 | 0 | 3 |


[All attribution answers/read-backs/contrasts](attribution-results.json). All arms are fresh unseeded runs; the source remains unchanged. The probe filters objective-owned backwards while retaining the same forward architecture and optimizer cadence. Disabled predictors/readers remain present without their objective updates; A-off also removes the answer from trial comparison. Diagnostic logs may still contain a disabled objective’s cost.

Extra slow/moved cases: 170/170 attempted, 11.45 minutes; {'passed': 110, 'failed': 48, 'process_failed': 12}. [Record](extra-cases/complete.json).

Full sweep: **4976/4976 cases attempted**, **19.37 minutes**, {'passed': 4662, 'failed': 23, 'skipped': 290, 'xpassed': 1}; 0 process stops. Reference: **5,219 cases / 122 minutes**. Initial workers 10; final workers 10; aggregate ceiling 28.0 GiB. [Full receipt](full-sweep/receipt.json). Source matched: True; complete coverage: True.

These are [unique case outcomes](full-sweep/case-counts.json). `TestOrthogonalFlags::test_flags_match_expected` emits three successful subtest reports in addition to its case report; the raw report total therefore has three extra passes. It was executed once. The non-strict top-k XPASS is separate from failures.

[Failure evidence and comparison with prior failures](failure-triage.json); [case timing before/after](timing-results.json). A previously failed case is not automatically classified as an unchanged cause.

[Final completeness audit](final-audit.json): all ported/added test bodies have a completed outcome or a recorded stopped attempt; no duplicate attempts within either the sweep or the extra-case campaign. Source, checkpoints and the fetched MNIST data remain matched. A stopped attempt does not verify its assertions.

Weekly coverage warning: Latest weekly slow-test run has failures; see tmp/slow-tests/latest.json.
<!-- CLOSING_RESULTS_END -->

MM_xor happened to pass this receipt's single unseeded attempt: MSE .185681
at call 163 against the existing .20 bar. No repair was made or repeated
confirmation attempted; this single outcome does not change its §17 status.
The unchanged MM_grammar table proof also passed (MSE .194022). Their full
observations are in `measurements/gate-02/observations.jsonl`.

The named `XOR_exact` CLI output proof is a new failure, outside the accepted
MM_xor and intermittent MM_grammar exceptions. All 64 epochs report output
cost .3500 and reconstruction .0000; all four final answers are zero, giving
MSE .5 against the unchanged .05 bar. This configuration selects named native
concept evidence directly and has no trainable numeric reader. Its concept
coefficients belong to reconstruction under the new partition, so the answer
cannot update them. The companion direct-gradient proof still passes because
it explicitly optimizes those coefficients with answer MSE outside `runBatch`.
This is an ownership regression, not a waived convergence fluctuation. See the
[saved failure investigation](failure-investigations.json). The final candidate
is retained for review; no source or assertion was changed during measurement.

The conditional attribution completed all forty runs without a process stop.
R, R+E, R+A and R+E+A yielded read-back passes of 2/10, 4/10, 1/10 and
3/10 respectively; class passes were 0/10, 0/10, 1/10 and 0/10. These are
fresh unpaired initializations, not controlled paired differences. The result
does not identify a causal benefit or harm from expectation. The enforced
one-owner partition removes direct competing updates to one parameter, while
reconstruction still changes the inputs learned by the readers and the trial
choice still depends on reconstruction and answer costs. The new split did not
recover the §14 gate counts in this campaign.

The round stops for review with remaining work. The full sweep's 23 failures
include two matching historical `getW` failures, unported row-local priming and
understanding fixtures, old backward-observer signatures, three imports of the
merged stateful fixture, and tests still requiring retired expectation or
operator-training behavior. The surface-lesson failure does not label which
owner failed to change; its individual-owner cause is not established. The
[saved investigations](failure-investigations.json) distinguish those limitations
from the named `XOR_exact` ownership regression and the slow cases' unresolved
behavior failures. No post-freeze repair or assertion change obscures the result.

The extra campaign attempted all 170 selected cases: 110 passed, 48 failed,
and 12 stopped without completing their assertions. Four workers exceeded
8 GiB; the bounded runner aborted eight active peers when those workers stopped.
Those eight peers did not themselves exceed the ceiling. Continuation ran only
never-started cases. The full sweep then completed in one segment with ten
workers, peaking at 2.364 GiB per worker and 8.351 GiB aggregate. Its ordinary
8 GiB worker guard was unchanged. The production native run remains the sole
24 GiB exception. These timings do not convert failed or stopped checks into
validated performance improvements.

## Ownership and the shared understanding

[ObjectiveOwnership.py](../../../bin/ObjectiveOwnership.py) restricts each
backward to its parameter owner. Reconstruction owns perception, trainable
object codes, compose operators/tied inverses and compose choice. Expectation
owns predictors, with detached sources, targets, routing evidence and preceding
context. The supplied answer owns its numeric or generate readers and stops
at the understanding. Compose/generate lessons train their own chooser only;
no lesson or generation update reaches an operator. Penalties follow their
parameter owner. Step 5a's uncut answer and the gradient-projection machinery
are removed. The dictionary rotation for configurations with
`conceptualContextLearningRate` remains the existing non-autograd update.

Both trials are costed at the same parameter versions before either takes
its own optimizer step and reconstruction perception pullback. Explore wins
only with reconstruction no higher and reconstruction-plus-answer strictly
lower. Expectation is trained separately and is absent from the comparison.
A tie retains greedy. In the audited XOR run's last training pair, greedy
has R .011013 and A .358657; explore has R .016002 and A .240408. Explore
improves the total but raises R, so the rule retains greedy. This is a measured
tradeoff within the decided choice rule, not an ownership violation or a
causal explanation of all ten runs. Late-created predictors/readers are adopted into the
optimizer before their first applicable backward. An output-space alias of the
percept vocabulary does not make that vocabulary an answer-owned parameter.

Each trial creates one immutable
[SentenceUnderstanding](../../../bin/SentenceUnderstanding.py). The tied inverse
and answer receive the same object; a fixture verifies object identity.
Its fields are:

| Fields | Meaning / answer representation |
|---|---|
| `root`, `end_slots`, `end_depth` | Concluded root and the occupied slots; the numeric head reads detached slots |
| `roots`, `depths`, `sentence` | Packed sentence context and the active sentence index |
| `rule_ids`, `arities`, `rule_valid`, `operand_positions` | Recorded derivation and occurrence addressing; the affine reader includes a position-weighted rule histogram and unary/binary counts |
| `operation_values`, `witness_offsets`, `journal_columns`, `journal_valid` | Selected operation values and losses retained by tied inverses; position-weighted linear moments of 3D and 2D values |
| `word_values`, `word_rows`, `word_valid` | Per-word retained references; a position-weighted D-dimensional moment |
| `primed` | `rows`, `codes`, `weights`, `own`, `bytes`, `byte_valid`; the answer receives the priming-weighted D-dimensional code sum |

The additional numeric reader has `7D + number_of_rules + 3` inputs and a
zero-initialized affine weight. Moment weights are `(index+1)/width`. All its
features are detached. Using linear summaries preserves the additive sum
control; it does not add a new nonlinear bank scorer to XOR's numeric head.
Generate gets the same record as named detached context. This is the chosen
fixed-width encoding; GlobalAttention was recommended, not required. Existing
attention configurations retain their own answer-owned consume/scorer path.

Priming is the seen surface, `[B,V]`, with decay, `primingSpread` diffusion and
the sentence's seen bump. Serial reading now diffuses too. Edges come from the
store's device-side sparse indices, without the 4,096-edge cutoff; neutral
sources send no flow. Each row snapshots its sentence's own symbols plus the
`reconstructionBasisLimit` most primed others. Both trials share that snapshot.
Only surface-bearing rows are byte candidates; every valid row supplies answer
context. [Priming probes](repairs/priming-before.log) failed before the repair;
the unchanged new assertions then passed in [priming-after.log](repairs/priming-after.log).

The fixed-parameter [sum-record probe](probe_sum_record.json) found at most
1.19e-7 root contrast and 2.38e-7 record-feature contrast across three forwards
without an optimizer step. That is a representation diagnostic, not a substitute
for the ten trained sum controls. The full owner/term definitions and relative
baselines are in [GradientFlow](../../GradientFlow.md) and
[Training](../../Training.md). [FutureWork](../../FutureWork.md#separate-gradient-ownership)
and [todo 6.9](../../../todo.md) record the implementation and the MM_xor exception.

## Repairs, causes, and fixture ports

Every repair's saved process/log remains in the [repair run inventory](repair-runs.json).
The focused results establish the listed local repair, not final source-wide acceptance.

| Saved failure / diagnostic | Cause and change | Targeted result |
|---|---|---|
| `ownership-before-valid`, `ownership-contracts-before`, `priming-before` | Shared gradients, old projection/5a contracts and a shared/capped priming surface; replace with owners, record and row priming | Ownership and priming assertions pass |
| `expectation-owner-cause`, `vocabulary-alias-owner-before` | Lazy predictors absent from optimizer; percept dictionary reached through an output module alias | Adopt new parameters before backward; assign the dictionary to R |
| `input-and-missing-leaf-before`, `byte-address-promoted-before` | A word-slot count was used as a byte-address capacity; a mixing row could have no `word_texts` | Input byte geometry follows input length without raising any configured limit; stage missing text from its observed bytes. 17 related cases pass |
| `reconstruction-while-parameterized-before` | Dynamic capture lifted a Python integer into the reconstruction while-loop operands | Fix static extents at the compiled reconstruction boundary; the three real compiled admission cases pass |
| `unresolved-behaviours-before`, `probe_remaining_causes.json` | Category observations used byte percept ids, whose metadata were absent, in place of admitted object identities | Use the reading's grammar object rows and owner concept ids; 14 category cases pass |
| `probe_remaining_causes.json`, `reverse-record-ports-after`, `record-replay-assert-before` | A new reading can promote percepts/advance priming; additionally the public inverse overwrote the correct scored trial after its journal was discarded | Compare the retained record and common priming state; publish the already completed inverse. Six packed/compiled parity cases pass |
| `word-journal-ports`, `recent-journal-contract-before` | Synthetic fixtures used pre-record sentence state and a scalar intra loss; the complete sentence journal is now discarded | Stage the current tensor peer, explicit end slots and row-valued registry loss; preserve graph count, backward and state assertions |
| `bounded-expectation-ports`, `compiled-expectation-small-after` | Expectation no longer appears in the comparison cost; a synthetic closing owner lacked `_concept_owner` | Read E from the registry and supply the fixture accessor. Both compiled/eager boundary cases pass; the factored-role case passes after the fixture repair |
| `qa-ingestion-cause`, `qa-consumer-before-02` | TruthSet needs serial closing; the old parallel carrier swap could not affect the serial answer slots | MM_qa uses serial grammar, and its existing attention consumer reads the serial detached head input. Ingestion and consumer gradient probes pass |
| `recent-answer-probe-before` | Old answer test observed the former uncut cost and assumed an update exists during a no-answer phase | Observe registered answer cost and snapshot parameters when that cost is recorded; preserve both phase assertions |
| `runner-peer-before` | A pool stop omitted a peer's completed and active cases from its record | Account all peers before abort; 261 runner/related tests pass, preventing repeated attempts in the closing sweep |

The reference-side invariant was investigated before porting: its observer
also intercepted training's newly enabled tied inverse, where the recorded
lossy witness legitimately names an operand side. Its subject is the free
read-back, which searches candidates without that supplied side. The observer
now scopes the unchanged no-reference-side assertion to that free read-back;
it still checks that altering the concluded root changes the inverse parent.
Separately, order/compile comparisons must use the same retained record and
priming state, rather than a newly promoted reading or a discarded journal.
Complete old/new bodies, including those assertions, are available for review.
The public-replay failure was real: scored/replayed row costs were .0002683 and
.0002737, while the overwritten public costs were .84923 and 1.72259. No field
of the retained record had changed. [The diagnostic](probe_record_replay.json)
shows that difference directly.

There are **181 ported, 92 retired and 14 added test bodies** in
[test-dispositions.json](test-dispositions.json). Each retirement has a reason.
[test-port-bodies.json](test-port-bodies.json) contains complete old and new
bodies, and [test-ports.zip](test-ports.zip) contains complete old/new files,
including helper changes. Assertions of retained behavior were not relaxed.
The top-k overlap xfail is non-strict with its prior pass/fail history retained.
The final focused group passed 264 tests in 27.50 seconds; collection found
4,973 cases. Final sweep outcomes, rather than this focused count, govern the
receipt's acceptance status.

## Configurations, test cost, and remaining limitations

[configuration-dispositions.json](configuration-dispositions.json) lists each
removed configuration, the pre-deletion reference search, decision and
replacement. The 21 specified historical configurations are deleted. Training
and CLI defaults now use BasicModel; MM_20M_fineweb's seven fast cases are
ported. The six stateful LTM cases set `stateless=false` on the kept fixture.
Simple/tomatoes and their targets/loader are removed. D3 `ideaDecode`,
`wordStore` and overlap tiling are retired with their code, options and tests;
`radialStmReduce` remains. The eight specified configuration families move to
MM_ladder, MM_20M_xor or XOR_exact. Feature-specific local fixtures are
retained, but the closing runs expose incomplete ports: an old conceptual
geometry expectation, an expectation-on assertion on MM_ladder, and a duplicate
answerSynthesis setting. These are recorded as port errors, not relabeled as
old flakiness. No compatibility method was added for retired reading modes.
The four attention configurations remain; a weekly QA case still uses a
retired accessor and therefore does not reach its behavior assertion.

MM_add_verb drops ideaDecode and uses the decided smaller geometry. Its first
production batch (12 rows) trained in **1.080 seconds**, **4.897 seconds** guarded
wall time, at **938,329,744 bytes** peak; see
[the process record](repairs/add-verb-first-batch.json). Its guard stays 8 GiB.

The [timing-port list](test-time-ports.json) records old measurements and each
change. Capture stays enabled when compilation is the subject. Other behavior
fixtures execute the same numerical loop eagerly. Expensive compilation and
over-memory subjects move to the weekly tier. The compiled expectation fixture
keeps both sentence pairs and every assertion, with 16 bytes/three STM slots
instead of 128 bytes/eight slots. The original capture was sampled and interrupted
after 826 seconds; the smaller compiled/eager pair completed within a 506-second
file run. The reference case timings are lower bounds where a run was stopped.
The final sweep and extra-case runs supply the after timings without
another profiling campaign. Case timings are pytest call intervals (plus
failed setup/teardown where applicable). Successful shared fixture setup is
not charged to an individual case; the timing record also retains each
worker's full wall time and peak memory, so moving work into setup cannot be
mistaken for eliminating it. An ordinary-tier skip does not replace that
case's measured weekly-tier outcome.
Before extra-case dispatch, a [selection audit](slow-port-selection-before.json)
found four omitted slow selectors among the ports. The pending list was
[completed](slow-port-selection-after.json), including the packed-backward
contract port. No case was rerun. This changes only receipt
selection metadata, not the frozen source, measurement harness, or guards.

Git LFS was installed and the real MNIST training CSV fetched. The loader now
names an unfetched pointer. The real 32-row subset, two-row training batch,
passes with ergodic off/on, with finite output loss, changed parameters and
finite weights. The three checks took **2.88 seconds** pytest time and
**2.862 GiB** peak. This required a **test-local** matching 784-slot geometry:
`ergodic-only.xml` historically declares 784 percept slots and 20 conceptual
slots, so the lossless binding skips the mismatched shape and the 20-slot head
cannot reshape it. `mnist.xml` also has historical 1/1 I/O dimensions. Those
three production XMLs remain unchanged; the new smoke test is not evidence
that their original geometry trains. Its numeric perceptual reconstruction
also warns that mask/target inputs are missing and contributes zero. The saved
failures and warning are retained in `repairs/mnist-*.log`; this is an explicit
limitation for review, not a claimed MNIST reconstruction result.

The moved sparse/category-ablation/legacy priming bridge tests include older
weekly failures. Their assertions remain. A missing bridge writer, absent
order-zero field evidence and an inactive sparse SBOW path cannot be repaired
by relabeling the fixture. Final failures will be listed separately from the
new ownership/record failures. No accepted XOR proof other than the declared
MM_xor/MM_grammar exceptions is silently waived.

The final [documentation link check](documentation-links-final.json) passes for
all 257 Markdown files, and `git diff --check` is clean. The [final verification](verification-final.json)
also confirms the unchanged measurement harness, environment and HEAD. The timing inventory
contains 68 cases: 62 have an acceptable completed outcome, one failed, and five
have only a stopped attempt. Each stopped entry retains partial worker time and
peak memory, rather than presenting it as a completed case time. The full sweep
has no duplicate case attempts; its extra successful subtest reports are
accounted separately. All 181 ported and 14 added test bodies have an observed
outcome or a recorded stopped attempt across the closing scopes.

## Historical §16.1 diagnostic (superseded stop, preserved evidence)

**Blocked at §16.1.** The MM_xor failure is reproduced and its input and
answer geometry are recorded. A causal repair that restores its convergence
has not been established in this diagnostic pass. Following the explicit
stop condition in [plan §16.1](../../plans/2026-09-29-item-6-9-xor-grammar.md#161-mm_xors-xor-proof-regressed-with-the-reading-migration-blocking),
§15, the configuration cleanup, and the closing measurements have not begun.

No production source, configuration, test, seed, bar, or guard was changed
in this round. Nothing was committed. The existing uncommitted candidate is
preserved: all 706 files in [before.json](before.json) still match, checked
in [verification.json](verification.json). HEAD remains
`d679df2b5a2665d72a99ca4b6dfd47c1ba048e99`. The starting candidate is also in
[before.zip](before.zip). No HEAD or §13 model was run; §13 source was read
from the existing archive solely to inspect the migrated reading's code.

## 1. Saved failing probe

[probe_mm_xor.py](probe_mm_xor.py) runs one fresh, unseeded candidate with
the convergence proof's CPU/eager forward, Adam learning rate .01,
200-epoch budget, four training examples and MSE < .20 bar. It observes
the actual training forwards; observations add no training steps. The
proof optimizes answer MSE alone, so reconstruction and expectation are
not competing objectives in this failure.

| Outcome | Value |
|---|---:|
| Best MSE in 200 epochs | 0.223053232 — **fail** |
| Final MSE | 0.250249147 |
| Final answers, in dataset order | .503862917, .501617610, .503850698, .502563119 |
| Model/probe time | 8.166 s |
| Guarded process wall time | 11.649 s |
| Peak sampled process-tree memory | 505,546,072 bytes |
| Unchanged worker ceiling | 8 GiB |

Dataset order is hello world → 0, hello there → 1, loving world → 1,
loving there → 0. Complete measurements are in
[mm-xor-before.json](mm-xor-before.json), with
[log](mm-xor-before.log) and [process record](mm-xor-before-process.json).
This is the saved failure before any proposed repair; no repair was made.

## 2. What reaches the answer

MM_xor is **parallel** (`serial=False`, `symbolicOrder=0`). Its configured
`useGrammar='all'` does not cause the serial sentence chooser to run:
`_chart_compose_at_C` returns immediately on this path. Runtime inspection
also finds no `merge` module in any of the three body stages. Each stage
has a butterfly `ConceptualCombine` with its tanh enabled. Consequently
there are no selected not/intersection/union sentence derivations to
report. This differs from the serial XOR_grammar gate.

The percept carrier is `[4,8,14]`: **six content numbers**, four where
numbers and four when numbers per slot. Before promotion, words are runs
of byte percepts; the eight-slot budget truncates much of the second word.
After promotion each word has one six-number percept, with a separator
percept between the words and five padding slots. The remaining eight
coordinates are location/time bands, not additional lexical content.

Promotion occurs while processing rows of the same batch. Epoch 1 mixes
byte and promoted-word representations; epoch 2 has the stable word rows
14 (hello), 15 (world), 16 (loving), 17 (there), with separator row 5.
All per-word content values, identities, spans and bands are retained in
the JSON snapshots at epochs 0, 1, 2 and 199.

| Epoch | Three nonzero singular values of centred head input | Best affine-fit MSE | Affine solution's weight norm |
|---|---|---:|---:|
| 0 | .085352, .026246, .010770 | 3.3e-31 | 60.581 |
| 1, mixed promotion state | 1.039552, .469847, .069560 | below 1e-27 | 10.357 |
| 2, words promoted | .020156, .013899, .009191 | below 1e-27 | 106.509 |
| 199 | .063888, .018238, .007241 | 1.3e-30 | 123.897 |

These are singular values of the flattened four conceptual carriers
presented to the head, not serial roots. The fourth singular value is
roundoff. Affine fitting uses double-precision least squares on the input
with a bias column. The reported norm is the weight part of that solution;
it is not a fitted training result or deriv69.py's margin measure.

At the final epoch, both hello and loving have **six zero content
coordinates** after the percept read transform. World and there remain
distinct. The full understandings are nevertheless distinguishable,
including their bands: there is an exact affine fit in principle. This
rules out a complete rank loss in this run, but does not establish why
the existing optimizer fails to find an effective reading in its budget.

## 3. Fixed-parameter promotion diagnostic

[probe_promotion_geometry.py](probe_promotion_geometry.py) makes three
forwards on one further unseeded candidate with **zero optimizer steps**.
This isolates promotion from weight learning. Backward is used only to
observe reach; it never applies an update. It hooks the actual affine
readout input and records every non-None parameter gradient, the word
masters, their transformed reads and their gradients.

| Pass | Three nonzero centred singular values | Exact affine solution's weight norm | Actual answer weight norm |
|---|---|---:|---:|
| 0, bytes | .077569, .016595, .010291 | 95.372 | .026318 |
| 1, mixed promotion state | 1.024105, .447869, .053834 | 13.353 | .026318 |
| 2, promoted words | .022531, .017830, .013343 | 55.296 | .026318 |

Promotion changes the geometry even without learning, but this second
initialization does **not** show that promotion always worsens the XOR
affine fit: pass 2 needs a smaller solution than pass 0. The unusually
easy mixed pass is transient. It must not be counted as convergence.

The answer MSE reaches the percept codebook, the numeric head, and only
the **last** conceptual binding. The first two conceptual bindings have
no answer gradient on this path. Code inspection explains that observation:
each parallel binding receives the original percept stream; subsequent
WholeSpace carrier forwards return a neutral property field. Thus the
last binding does not consume the preceding conceptual binding as a
trainable recurrent input here. This behavior is also present in §13's
source; it is a limitation exposed by the reading change, not a newly
proved migration defect.

The clamped percept read is a straight-through estimator: zero content
does **not** imply a dead clamp derivative. Live code gradients were
measured. No claim of a severed percept gradient is supported.

Full data: [promotion geometry](probe_promotion_geometry.json),
[log](promotion-geometry.log), [process record](promotion-geometry-process.json).
This diagnostic took 3.615 s wall time and peaked at 477,988,136 bytes
under the same 8 GiB guard. It is not a second convergence campaign.

## 4. Findings and unresolved cause

Static comparison establishes a relevant change: the old radix branch
allowed promotion of a full lexer chunk, including a whole input phrase;
meronomy bounds promotion to individual words. A separate phrase row can
carry combination-specific information that separate word rows do not.
The small word codes, their collapse in the failed training, the transient
within-batch promotions, and the lack of effective recurrence are concrete
leads. None has yet been isolated as a sufficient cause with a successful
repair within meronomy.

Changing code scale or initialization, enlarging the budget, altering the
bar, or restoring phrase promotion would not be a demonstrated repair
under this request. None was done. The next investigation should isolate
which part of the parallel meronomy representation prevents the existing
nonlinear binding from learning; the saved probes provide the failing
case and the geometry to compare. There is no 5/5 confirmation, no claim
that the blocker is resolved, and no §15 implementation to review yet.

## 5. Remaining ordered work

After the §16.1 blocker is repaired and confirmed: §15 ownership and shared
understanding record, §15.6 configuration/test cleanup together with
§16.2–16.6, then the single closing measurement set and documentation
updates requested in §15.7. None of those downstream measurements was
spent on this unrepaired candidate.
