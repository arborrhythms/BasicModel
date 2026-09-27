# Item 7.5 sentence-seal review corrections

The per-sentence review corrections are implemented and **stopped for Claude's
review, uncommitted**. The source-matched full sweep completes **4,968 cases:
4,636 passed, 321 skipped, ten failed and one expected failure**. The ten failures
cover two gradient assertions, four obsolete evaluation-call fixtures, three
missing-observation/prediction checks, and the preserved relative-depth gate.
Both explicit XOR CLI gates remain red; MM passes this run, with its historical
failure retained. No assertion, seed or threshold was tuned to these results.

The [amended spec](../../specs/2026-09-26-one-operation-per-round.md) requires
two optimizer steps at each sentence seal, row-local winner selection, and
commit before perceiving the next sentence. Perception is cached once for the
two hard compose paths. Batch-end answer loss is outside their comparison.

This receipt supersedes the [whole-batch draft](../2026-09-27-item7-5-review/README.md).
Its earlier measurements and failures remain preserved there. The
[validation summary](validation-summary.json),
[changes since the first reviewed submission](changes-since-review.patch), and
[exact executable/configuration/test source](review-source.tar.gz) accompany
this receipt. Bounded tests use at most three workers, 8 GiB and
1,800 seconds per worker. No million-sentence model training is requested or run.

## Transaction and gradient ownership

Each sentence index across active rows has an exploit backward/update, then
an explore backward/update using the same cached perception and the updated
parameters. The objective compares reconstruction and prediction for that
sentence and row; batch-end teacher answers remain separate. Strictly lower
explore cost wins that row, with exploit winning ties. Only the selected
program admits lexical identities and records memory/expectation observations,
before perception of the next sentence. Evaluation runs deterministic exploit
only. At zero temperature exploration replays the exploit prefix, masks its
choice at one uniformly chosen used round, and then takes argmax.

Scratch state consists of functional STM and trace tensors; restoring it swaps
bindings. There is no model-wide snapshot or second full batch forward. Each
compose backward releases its graph immediately. Fresh leaves for the cached
perception pass their gradients back into the original perception graph.
Saved parameter/buffer values are shared per view and version so the second
update reads the values from the single perception forward. Immutable
intermediates retain a mutation check rather than another copy. The analytic
[gradient probe](pullback-gradient.log) checks both updates against the chain
rule; the bounded two-epoch word-store check exercises graph release.

The packed causal test uses ragged word boundaries and opposite row winners.
It checks one live perception per word, update/update/commit order, the winning
program and its occurrence chain as the next sentence's context, and both eager
and compiled numerical word bricks. Public time and the training-step count
still advance once per batch. The repeat-evaluation test fixes the logical
input clock: the model's `When` metadata is an input and would otherwise differ
between calls. The clock-advancement test remains separate, and the diagnostic
in `evaluation-clock-diagnostic/` retains the evidence for this distinction.
Numerical configurations without serial sentence seals retain their batch-end
objective.

## Development evidence

All attempts are retained, including the memory-limit exits. The first cached
perception implementation retained both full compose graphs and exceeded the
unchanged 8 GiB cap. Deduplicating saved values and skipping inactive eager
rounds were insufficient by themselves. Separating each compose graph from the
retained perception graph lets the unchanged two-epoch word-store check finish
at a measured 6.00 GiB peak. No cap was raised and no learning target was tuned.

The affected run also exposed old output-policy hooks that assumed all backward
calls happened at batch end. They now assert that sentence backward has no
output-policy credit and keep the original answer-gradient checks at batch end.
The unchanged raw-forward replay assertions caught a separate incomplete-forest
publication defect: a negative two-slot depth was collapsed to one slot.
Publication now preserves the forest while its negative depth still blocks
memory. Filtering incomplete packed observations now clones the mask rather
than mutating the input sentence layout. The original replay/recall assertions
and explicit forest/layout regressions cover these corrections.

The selection before restoring the sentence reconstruction objective completed **104 cases: 88 passed, 15 skipped,
one failed**. The failure is
`test_normal_supervised_output_respects_gradient_contract[True]`: the question
conditioner receives zero gradient from the scored first emitted word. Its
nonzero-gradient assertion is retained. The [observational probe](generation-gradient-boundary.log)
reproduces it on the same source: six/eight words emitted, neither row truncated,
conditioner gradient norm **0** from the first emission versus **8.610651** from
all emissions. This is consistent with conditioning the newest slot while the
left-to-right walk scores the older constituent first in a multi-slot answer.
It is an additional red outcome for review, not a passing or tuned gate.

The seven explicit checks on that earlier source completed: **five passed, two failed**. Fullgraph
runtime-length capture, packed operand rows, packed reconstruction, two-epoch
graph release and MM pass. Both unchanged XOR_grammar CLI checks fail at the
fixed six-row WholeSpace capacity during first-epoch reset/autobind. The
historical MM failure (.21757) remains in the preceding receipts; this passing
raw-forward run does not prove mature learned utility.

The first timing reissue is preserved in
`measurements-before-reconstruction-cost/`, with its exact source in
`source-before-reconstruction-cost.tar.gz`. It is superseded: the historical
serial configuration requested legacy event reporting, which inadvertently
left both sentence reconstruction costs zero. Its throughput does not measure
the required two sentence objectives. Word-major sentence seals now train tied
reconstruction independently of that reporting choice; the legacy event metric
remains reporting only. No loss threshold or learning seed was adjusted.

`reconstruction-cost-probe/` exposed compiler placeholder sizes in the saved
value cache. Fake tensors now stay owned by the compiler instead of entering
the runtime storage/version cache. `affected-before-target-detach/` was then
interrupted after reproducing a shared-graph backward error (45/84 cases
completed). Reconstruction targets were detached inside higher-order loop
bodies, leaving a zero backward edge from the loop to the original perception.
Detaching targets before loop capture preserves the intended constant-target
semantics and the explicit perception pullback. These attempts are retained;
neither is the final sweep.

The target-detach probe completes all 11 cases: eight passed, two failed on
new occurrence-ID assertions in the causal fixture, and the existing output
conditioner assertion remained red. Both native reconstruction/backward cases
reached the separate answer phase, and the legacy-report objective regression
passed using real higher-order loops. The causal fixture now enables LTM
consolidation and deliberately completed paths, then checks the actual winning
payload and occurrence identity in each row before the next perception.

A reconstruction configured for the old batch graph now compiles as a separate
numerical brick at the eager sentence seal. Explicit eager placement remains
available. This follows the new optimizer boundary; it does not alter the
reconstruction equations or measurement inputs.

## Corrected-source validation

The 669-file executable/configuration/test source has fingerprint
`6e84d24b2e13424b5e2b171c64dcba7d64cfec8ff62439127eba584529c8637d`.
The final affected selection completes all **47 cases: 46 passed, one failed**.
Every sentence-seal, temperature, tie, winner, ownership and reconstruction
mechanism case passes. The failing case remains
`test_normal_supervised_output_respects_gradient_contract[True]`, with a zero
question-conditioner gradient. Its assertion is unchanged.

The source-matched explicit selection completes all **seven cases: five passed,
two failed**. Fullgraph runtime-length capture, operand provenance, packed
reconstruction, two-epoch graph release and MM pass. The graph-release worker
peaks at **5.94 GiB**, below its unchanged 8 GiB cap. Both XOR CLI checks fail
at the six-row WholeSpace inventory during reset/autobind. Protected gate
sources, thresholds and fixtures are unchanged. MM's historical **.21757**
failure remains part of the evidence; a short passing run is not mature utility.

All three measurements completed without concurrent workers. The final full
sweep completes on this same source. Its scheduler isolates relative-STM and
native output cases in fresh workers to keep graph caches within the existing
caps. Every selected case completes exactly once; no cache retry or continuation
is needed. The [source manifest](source-manifest.json) and
[protected gate hashes](unchanged-contracts.json) bind these results to the code.

## Reissued measurements

The original seed-42 protocol is unchanged: batch size two, four validation
batches before training, two training warm-ups followed by five measured
training batches, then four validation batches. Each process runs alone on
CPU, one numerical thread, under 8 GiB / 1,200 seconds. Reporting uses the
original event reconstruction metric; both sentence objectives train the tied
byte traversal. These are short measurements, not a quality gate.

| Serial reconstruction | Before training | During training | After training |
| --- | ---: | ---: | ---: |
| Reviewed 9b | .1005906649 | .0948241442 | .0928765051 |
| First 7.5 submission | .1040502079 | .1041770041 | .1039280705 |
| Corrected sentence seals | .1065397672 | .1001331091 | .1001378968 |

Warmed throughput is **.9189261373 sentences/s**, including the original epoch
tails. The corresponding five measured batch calls average **2.167488242 s**:

| Component | Mean seconds per batch |
| --- | ---: |
| Exploit compose | 1.106395225 |
| Explore compose | .230206750 |
| Exploit backward, including perception pullback | .269424950 |
| Explore backward, including perception pullback | .269619625 |
| Batch-end backward | .000318300 |
| Sentence reconstruction/prediction scoring | .111816675 |
| Scratch snapshot | .000000742 |
| Scratch restore | .000276658 |
| Other, including perception, optimizer steps and commits | .179429318 |

The fixed two-batch warm-up does not guarantee every later graph shape has
already been captured: the first measured batch takes **5.65698 s**, and is
included unchanged. The first training batch takes **555.77970 s**, mostly
initial graph capture during sentence scoring; this cold cost is not hidden
in the warmed throughput. The complete serial process takes **745.42 s** and
peaks at **5.83 GiB**. Both trial costs are positive for every training row;
explore wins **5 of 14** row comparisons (5 of the 10 measured rows). The raw
per-batch costs, choices and complete timing split are in
[batch-timing.json](measurements/batch-timing.json).

Packed and single mean sentence byte costs are **exactly equal at
.6496902331709862**. All per-sentence records match, as do initial parameter
and dictionary fingerprints. Both layouts mark every row as truncated; restored
parity does not establish complete reconstruction or learned utility. The
packed/single processes complete in **290.30 / 485.74 seconds**, under the same
8 GiB / 1,200-second limits. The older first-submission costs
**.7877992094 / .6835970134**, and reviewed 9b's equal **.6839025617**, remain
comparison evidence, not tuning targets.

## Full sweep findings

The single full sweep completes **4,968 cases** with `RUN_SLOW=0`: **4,636
passed, 321 skipped, ten failed and one expected failure**, exit 1. All cases
complete exactly once in **5,459.09 seconds (90.98 minutes)**. It uses three
workers / 24 GiB aggregate, 8 GiB / 1,800 seconds per worker, and the existing
10,800-second suite deadline. Peak worker / aggregate memory is **6.19 / 13.42
GiB**; no time or memory guard fires. No million-sentence campaign is run.

| Unwaived failure group | Cases |
| --- | ---: |
| Shared-operator reconstruction-gradient reporting | 1 |
| Output question-conditioner gradient | 1 |
| Evaluation fixtures expecting whole-batch exploration | 4 |
| Native observation/prediction retention | 3 |
| Preserved relative-depth gate | 1 |

The output-conditioner failure is the same assertion recorded in the final
affected selection above. It remains a full-sweep failure as well.

The unchanged relative-depth campaign,
`test_thinking_kernel.py::TestDepth3RelativeEndState::test_first_trained_read_reaches_depth3_end_state`,
fails with **[1, 1, 1, 1]**: none of its four recorded end states has depth three.
Its original assertion remains intact. Earlier receipts' sixteen depth-one
observations remain historical outcomes; this sweep records its own actual list.

An additional unchanged assertion fails in
`test_gradient_factorization.py::test_normal_batch_logs_named_shared_operator_gradients`:
no reported shared-operator `reconstruction_norm` is nonzero. The grammar-chooser
learning integration passes on the same source. The diagnostic failure remains
an unwaived review finding; it is not relabelled as an expected failure or
removed from the sweep.

Four cases in `test_item9b_followup.py` fail: both variants of
`test_interleave_reads_native_context_before_any_serial_word`,
`test_interleave_epoch_reads_a_short_final_group_once`, and
`test_interleave_checkpoint_resumes_unread_serial_prefix`. Their call-list
assertions still require a whole-batch explore call during evaluation. The
observed calls contain the native context pass where needed and one serial
exploit pass, as required by the amended spec. These are obsolete fixture
expectations discovered by the frozen sweep. Their assertions and red outcomes
remain visible; they have not been waived or silently rewritten after measurement.

Three native cases in `test_expectation_defaults.py` expose missing observation
or prediction state: `test_native_runtime_reports_pairs_without_accumulating_or_updating`
records **one observation instead of two**, and
`test_native_future_and_other_row_changes_do_not_change_first_estimate`
has no comparison for the first row. With prediction disabled,
`test_expectation_off_keeps_every_packed_observation_in_ltm` also records
**one LTM observation instead of two**. These are additional unresolved behavior
findings. The assertions and their recorded failures remain unchanged; the
passing completed-path causal mechanism test does not establish that arbitrary
native paths always reach a usable sentence observation.

The complete [supervised result](full/result.json.gz) and
[raw requests, worker logs and reports](full/raw-receipt.tar.gz) are preserved.
After the supervisor saved its complete result, the one-off `run_full.py`
wrapper raised an unpacking error because it treated the returned dictionary
as a pair. The [wrapper exit record](full-wrapper-exit.json) retains that error;
the wrapper is retained as executed. It occurred after all 4,968 cases finished
and did not cancel or repeat a case. Independent coverage and source checks
verify the durable result before packaging.

The raw [serial measurement](measurements/serial-baseline.json),
[packed measurement](measurements/packed.json),
[single measurement](measurements/single.json), and
[comparison](reconstruction-comparison.json) retain the untuned protocol and
outcomes. Every bounded development attempt has its own archived receipt and source
delta in the validation summary. Final documentation links are checked
separately on this same executable source. This is the requested review stop;
learned utility remains unproven, and no commit or push has been made.
