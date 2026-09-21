# Expectation at the seal

Composition reads no expectation. The removed comprehension prior has no runtime
flag or compatibility path: neither serial comprehension nor parallel binding
accepts a predicted seed. `generate_sentence` understands any supplied seed text
normally, then sends the positive predicted idea to the existing `<generate>`
walk and full owned spelling inverse. Generated output is not an observation.

The seal retains the pure observation `o` and its prior estimate `e`. Per role,
the derived negative image is `n = -g * (1-m) * k * e`, and the conceived idea is
`c = o + n`. `k` is sigmoid predicted presence; `m` is the active question's
explicit grammatical open-role mask, not every unoccupied syntactic position.
`expectationGain` is a fixed model setting in `[0,1]`, default 1. Zero gives
beginner's mind while the predictor still runs and learns. Disabling
`sentenceExpectation` disables prediction and comparison as well. A cold row has
no image at any gain. Reading scope, priorities and order-zero fields never read
this estimate.

The existing chooser receives `c` per role, presence, the object mask and a
comparison-available bit in addition to its ordinary semantic and attended-memory
context. All these features are detached. A declared `not.thought` can produce a
serial inference through the shared `not` operator. An empty observed role's
negative image does not itself execute negation or write an observation.

## One pair, one observation

`TernaryTruthStore` is the single durable owner. A warm external boundary writes
an `estimate` row before its `observation` (or interrogative `question`) row and
links their stable occurrences in both directions. Only the observation enters
the chronological predictor context. Provisioned truths and internal thoughts
are not additional external observations. Estimate rows are never facts or their
own corroboration, and ordinary LTM evidence readers exclude them.

The estimate retains **all three vectors**, including roles with low predicted
presence. Its carrier mask identifies stored coordinates; three separate logits
carry learned occupancy. Thresholding occupancy must not erase a vector needed
for the residual. The estimate's checked bindings and scope inherit only the
preceding retained source occurrence; they do not copy the arriving target or
claim learned generation of novel metadata. Provenance records preceding source
occurrences, the external stream/document, and the intended observation. The
observation retains its own bindings, scope and original `o`.

`expectation_pair(gain=..., object_mask=...)` derives `r`, `n` and `c` from those
two rows. There is no third durable delta or conceived row. `c - n` recovers `o`
up to floating-point roundoff; the retained observation remains exact. Capacity
favors the observation when only one row fits.

Surprise uses **every** role: `r = o - e`, including a zero target for an empty
role. The differentiable prediction objective is mean squared residual plus
mean presence BCE. The detached row scalar is `s / (1+s)`, where `s` is
the all-role mean squared residual alone; presence is a separate training term. It is independent of gain and object mask. Unknown
surprise is `-1`, including older checkpoints, rather than a fabricated zero.
Compaction and reset preserve this column's row alignment. The forgetting pass
that uses it remains a separate item.

The existing semantic sidecar fingerprints links with scope and source text,
checks both directions on restore, and participates in origin-compaction
reachability. A tensor-only restore leaves rows with missing required sidecar
metadata unavailable. No second memory owner is introduced.

## Residual credit on the existing controller

`expectationPolicyWeight` (default 0) enables optional anticipatory episodes;
`expectationQueryBudget` (default 64) bounds actual work. The model's `<thought>`
catalogue must provide `arma`, and `ltmConsolidation` must be enabled to own
the prior occurrence. A positive policy weight rejects a missing owner. These are ordinary `run_selected_thought` episodes,
using the same chooser, checked executors, interaction history and optimizer.
They run before the incoming batch. Later packed slots have an empty optional
query phase and still receive chronological prediction at their seals. No query
runs in the packed drain after unseen input has changed staging.

Anticipation reads the row's prior observation chain and cued frames. It excludes
current STM, the parallel knowing field and the last input program from its
chooser view. Hard retrieved frames can fill older positions in the same bounded
predictor context; the latest external observation stays last. Query effects
cannot reinstate knowing on anticipation's behalf. Selected `arma` stages on the
existing pending-prediction owner. No observed input is changed by a query.

A completed residual supplies return `-(all-role MSE + presence BCE) - 0.01 *
actual work`. This score-function estimator owns an EMA baseline, separate from
supplied-answer credit and from ordinary prediction MSE. It stores detached
candidate features, selected indices and behavior log probabilities. At credit
time it replays the same chooser at its current parameters, with a detached
importance ratio capped at 4. This is bounded, potentially biased off-policy
credit when parameters have changed; it never backpropagates an old policy
graph. A delayed predictor is likewise replayed from its prior inputs if its
parameter version changed, while its original forecast remains the evidence.
This replay also covers the optional contrastive predictor objective. Reading
the object mask of a nested `what` follows its owned question occurrence,
charging serial scans to the same controller meter and respecting exhaustion.
No objective rewards a small conceived norm. Gain, object selection and reading
attention receive no gradient from these detached features or scalar returns.
The existing reading-attention handoff also detaches its scope before composition;
its own next-word supervision remains separate from residual credit.

Source encodings in the current prediction step retain the existing
[objective-local gradient contract](GradientFlow.md); observed targets, thought
effects and durable history are detached. Mechanism checks do not demonstrate
useful querying or a benefit from subtraction. Learning measurements and the
source-matched receipt belong in [Testing](Testing.md).

## Migration and regression dispositions

`sentencePrimingScale` and `architecture.prediction` are rejected as retired
settings. The old comprehension projection (`InterSentenceLayer.cast`/`prime`),
serial `_c_prior`, parallel seed slab, per-source residual fire flag and its
arm/fire APIs are gone. Old `cast.*` checkpoint tensors are discarded at the
owning discourse loader; there is no replacement input projection. Source-row
batch sizing now has an explicit host integer; sentence completion and SVO
reset retain their own existing owners.

The previous full-context chooser receives zero columns for the new conceived
features before its two action-kind columns. Existing logits are preserved;
only the resized first weight starts without old optimizer moments. The older
incomplete-context schema keeps its existing explicit reset migration. The
residual return baseline is transient and a newly constructed model starts afresh;
no pending trajectory or old autograd graph is restored.

The removed tests asserted the retired behavior: four prior-broadcast cases in
`test_c_prior_slotwise.py`, seven copied seed arithmetic cases in
`test_inter_sentence_prediction_shape.py`, one slow intersentence-seed case in
`test_conceptual_recurrence.py`, one cast/prime case in `test_discourse_space.py`,
two residual API cases in `test_stm_residual_no_sync.py`, and two once-only
residual cases in `test_subspace_context.py`. They are replaced by pure-forward,
absent-interface, and native nonzero composition checks. Fire-flag row/batch
cases now exercise the surviving sentence-completion lifecycle; no per-row
reset or microbatch isolation requirement was dropped. Output/What fixtures
remove the retired dispatch setting while retaining their prediction checks.
The explicit routing-calibration test initializes its active feature columns
independently of the ignored context width. Its training allowance and learned
execution assertions are unchanged. Budget-drain coverage includes comparisons
whose mask reads spend the final unit; history records that work before cutoff.

`test_negative_expectation.py` covers the signed identity, empty-role residual,
zero gain, presence, scope and binding provenance, nested question masks,
prior-view isolation, delayed credit, native unlabelled training alongside tied
reconstruction, checkpoint migration, and positive generation seeding. The
existing retention and sentence-stream suites retain packing, document, row,
capacity and checkpoint coverage. Learning results are reported separately in
[Testing](Testing.md).
