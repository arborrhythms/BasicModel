# What, spacetime learning, and concept formation: unified specification

> **Status:** active implementation specification and maintainer handoff,
> originally 2026-07-27 and revised 2026-09-08. The `what()` question/data/LTM
> foundation is implemented, but the definition of done is not. This revision
> integrates the unfinished work with separate input reconstruction and
> rational output-synthesis paths.
>
> **Immediate scope:** retain the implemented `what()` addressing and LTM
> parity substrate; make `Data.what()` the real answer-loss target; reserve
> `reverse()` for input reconstruction; add `reverseOutput()` from rational symbolic
> decoding through conceptual and perceptual synthesis to `OutputSpace`; and
> make reconstruction-primary learning protect one collective understanding.

## Status record (2026-09-09)

Done (on `main`): Step 0 question/`Data.what()`/LTM-parity/thinking
substrate (PR #8); the absolute dataset `where` given to the model
(8975365); Step 2's collective-field repair and Step 6's
reconstruction-priority projection + single optimizer step (a8a5142); Step 5's
`Data.what()`-authoritative answer target with row masking and the
`input_reconstruction` / `answer_construction` cost names (`primary_costs()`,
`test/test_what_training.py`).

Landed 2026-09-09 (parameter-gated where noted): Step 1 `Understanding`
(`bin/Understanding.py`); Step 2 `Model.reverseReconstruct()`; Step 3 `synthesize()`
entry points, `OutputSpace.from_percepts()` and `Model.reverseOutput()`; Step 4
`what()` through `reverseOutput()` -- Steps 3-4 under `<answerSynthesis>` (default
false, byte-identical); Step 6 shared `Understanding` + sampled branch-point
gradient diagnostics (`<branchDiagnosticsEvery>`); Step 8 root answer through
`reverseOutput()`.

Also landed 2026-09-09: relation-driven answer resolution (past = recall
from the discourse ring, future = prediction, cold memory flagged
unresolved); temporal text answers realized to an input-space surface and
scored against the embedded target sentence (Step 5 complete for every
question family); the tied-operator autograd-vs-finite-difference test.

Also landed 2026-09-09 (spec sweep): Step 7 causal temporal use via the
learned `question_conditioner` on the answer path (ablation test); named
perceptual bindings (`<synthesisBindings>`); the exact-identity guard;
`what_report()` per-family reporting; the reasoning hook for prompted
questions; the `<whatCurriculum>` and the canonical `BasicModel.xml` cutover
to `reverseOutput()` (Step 9); the joint-training band test (12.14).

Review fixes 2026-09-09 (Codex): answer-path modules materialize before
the checkpoint audit (reload round-trips); per-row question resolution
(`row_sources`); `past -k` recall reads a chronological per-row history
(the ARMA ring's fill layout is newest-at-low-end). Throughput (12.15): canonical
config on MPS, 12-step matched bench -- a8a5142 baseline 12.27, cutover
12.20 sentences/s (~0.5% cost; band 15%).

Closed 2026-09-09 (the last three items):

- Carrier-pure Space inverses (Step 1 residue): every `Space.reverse()`
  writes into the carrier it is given when that carrier is a fresh
  answer-path carrier (`SubSpace.carrier_pure`, allocated with its own
  per-batch bases by `carrier_like`); the live `Space.subspace`, the bases,
  reverse by-products, merge carriers and grammar cursors are untouched.
  `test_reverse_output_is_carrier_pure_without_the_guard` proves it with the
  synthesis guard disabled. The established reconstruction path is
  byte-identical (the seam only diverts for pure carriers).
- 12.16: the model owns its data authority (`model.data`), loss registry
  (`model.errors`, `record_loss`), loss composition (`_primary_loss`) and
  batch opening (`_open_batch`); Teacher is an optional provenance /
  interactive-context adapter and may be detached
  (`test_model_trains_with_teacher_detached`).
- Grammar-chooser question context: the STM bounded-reduce pass -- the one
  whose choices the bounded local (policy) objective credits -- never
  received `what_ctx`; grammar layers now default it from the installed
  context and `Model.what()` installs it on every grammar layer. With
  `<forwardGrammarWeight>` on, the chooser's What bias receives gradient
  through the policy objective, which `what_report()["policy"]` reports
  distinctly from the continuous answer credit (section 11).

Remaining by design: the chooser's hard choice is credited by its policy
objective, not by differentiating the answer loss through an argmax.

The companion [What, spacetime, and thinking design](../WhatSpacetimeDesign.md)
defines the governing architecture. This document turns that design into an
implementation sequence, test requirements, and acceptance gates. Runtime
behavior remains documented in [Componentization](../Componentization.md),
[Language](../Language.md), [STM](../STM.md), and [Training](../Training.md).

## 1. Current boundary

The current model already has the execution framework needed for evaluation
and learning:

1. `TheData` selects and prepares a batch.
2. `Model.run()` / `runEpoch()` / `runBatch()` execute the model.
3. The existing loss evaluates the response.
4. The optimizer optionally updates the model.
5. STM and LTM record the model's sequential state.

The merged implementation provides `WhatQuestion`, `Data.what()`,
`Model.what()`, document-safe relative lookup, paired LTM slots, parity,
closure pressure, and target-free chooser context. These are an implemented
foundation, not completion of this specification.

In particular, the current `Model.what()` wraps the ordinary projected output
head, and `runBatch()` records the desired `Data.what()` result without using
it as the actual answer-loss target. Temporal causal-use, grammar-driven
thinking, final-root scoring, separate reporting, and the matched B24
benchmark remain unfinished. The direct SymbolicSpace-to-OutputSpace head is
not the target answer architecture.

The 2026-09-08 local follow-up fixes stable LDU exact-zero diagonals and
trained butterfly padding, adds strict inverse/gradient tests, and wires
reconstruction-priority projection into `runBatch()` behind
`reconstructionPriority` (enabled in BasicModel). The subsequent cutover
replaces the RMS/same-row/mean conceptual bridge with the plain Sigma readout
described below. It does **not** implement the new generated-output path or
prove collective concept-only inversion. BasicModel's
current `detachedReverse=true`, reconstruction weight 1, and context-rotated
non-grad concept dictionary also remain; the new guard creates no missing
encoder gradient and costs no extra backward work on unlabeled FineWeb.

The first-level readout experiment adds a reusable
`ConceptsFromPercepts(Layer)` interface, a context-gated implementation, and
an ordinary Sigma implementation with optional connection-wise L1. Their
unit tests and controlled two-/six-reference comparison are implemented
(section 5.6). Plain Sigma is now wired into eager, K=2, and tensor-loop CSSub
execution, with stable typed references, sparse per-concept coefficients,
retained native P/W fold stacks, and checkpoint migration. BasicModel now
enables weak incoming-connection L1 (`conceptReadoutL1=0.01`) through a
row-local proximal Adam update; gates remain disabled. The broader overlapping-concept field, independent
quantization/collective decoding, and explicit `Understanding`/reconstruction
API remain part of Step 2; retained native state is not proof of those gates.

Present-input reconstruction supplies a restricted data case, but it is not
identical to answer construction. Every presentation may reconstruct its
input. `What(present)` separately asks the model to construct a response whose
desired content happens to be the current datum. The two results can match
externally while following separate downward paths and receiving separate
costs.

The repository contains a `Teacher` implementation and teacher-oriented
lesson types from the earlier design. They are migration code, not the target
architecture. No new behavior should depend on a Teacher, Student,
`SourceOracle`, `SpacetimeView`, `WhatFrame`, or separate query-planner class.

The accepted clean B24 benchmark is 41.989 complete sentences/s. It remains
the reconstruction baseline against which the cost of this iteration is
measured. It is not evidence that past/future queries or iterative thinking
already work.

The serial `symbolicOrder` path also remains incomplete as a concept learner.
It can stage, promote, decode, and order a selected concept row, but does not
yet establish higher-order nouns or reusable diachronic verb meanings. Those
capabilities remain in the preserved roadmap in sections 13-16.

## 2. Core contracts

The target exposes the following methods on the existing classes:

```text
Data.what(question)  -> desired What or unavailable
Model.forward(input) -> Understanding
Model.reverseReconstruct(understanding) -> reconstructed input
Model.reverseOutput(understanding, question) -> produced What
Model.what(question, input) -> forward(input) followed by output(...)
```

These methods delegate to infrastructure already owned by their class:

- `Data.what()` resolves the question against `TheData` during learning or
  evaluation.
- `Model.forward()` performs bottom-up perceptual, conceptual, and serial
  symbolic understanding without seeing the desired answer.
- `Model.reverseReconstruct()` uses the input-associated inverse path and ends at
  `InputSpace`.
- `Model.reverseOutput()` rationally decodes the question, constructs an answer
  symbol, and synthesizes it through conceptual and perceptual space before
  the final `OutputSpace` adapter.
- `Model.what()` delegates to the same `forward()` and `reverseOutput()` operations;
  it is not a direct task head or second generator.
- `Model.run()` remains responsible for presentation, execution, loss, and
  optional optimization.

The two `what()` methods answer the same question with different information.
The `Data` result is the desired response. The `Model` result is the model's
best response. The `Data` result must not be inserted into the model context
before the response is produced.

The logical run path is:

```text
question, input     = Data presentation
understanding       = Model.forward(input)
reconstructed_input = Model.reverseReconstruct(understanding)
actual              = Model.reverseOutput(understanding, question)

if learning or evaluation:
    desired = Data.what(question)
    reconstruction_loss = compare(reconstructed_input, input)
    answer_loss         = compare(actual.what, desired.what)
    loss                = weighted_sum(reconstruction_loss, answer_loss, ...)

if learning:
    update Model through the existing training path

if inference:
    attach actual as the presentation output when the caller requests it
```

There is no additional scoring service. Existing loss ownership remains in
`Model.run()`, but its bookkeeping must expose reconstruction and answer
construction as separate terms and optimize their combined graph.

## 3. Questions and data coordinates

### 3.1 The question contains its coordinates

The where and when of the requested answer are always contained in the
question. Do not pass a second privileged address beside the question. The
model sees the question, including its temporal meaning, as context for the
answer.

The coordinate representation may initially be structured and later become
fully grammatical. Regardless of encoding:

- it participates in producing the answer;
- it is not itself part of the desired answer; and
- the loss compares only the produced and desired `what`.

### 3.2 Absolute `where`

The question's coordinate is `where`: the zero-based presentation index in
`TheData`. For text data, it is the sentence index. The dataset exists all at
once, so this is a position in the dataset (an absolute `.where`), not a time;
the model's `.when` clock is never part of a question. `Data.what()` must
resolve at least:

```text
at index n, What(present)     -> data[n]
at index n, What(past, -k)   -> data[n-k]
at index n, What(future, +k) -> data[n+k]
```

The first implementation may begin with `k = 1`, but its types and tests must
not equate “future” with a permanently hard-coded next-token or next-sentence
head.

Document boundaries remain hard boundaries. A relative temporal lookup must
return unavailable rather than crossing into another document or split.

### 3.3 The model sees the absolute `where`

`where` and the resolved `target_where` are part of the target-free question
context. The model maps them into its own `.where` coordinate system
(normalized by the split extent plus a `.where` ladder over the presentation
index) and a learned, zero-initialized projection consumes the result; see
`WhatSpacetimeDesign.md` §2. `Data` must not overwrite or supervise the
model's internal coordinates directly.

A finer rung addressing part of the datum presented in one trial (for text, a
span within the sentence) is deferred and is not a separate coordinate.

### 3.4 Presentations reserve input and output

Each presentation index has an input side and an output side:

```text
DataPresentation:
    where: zero-based presentation index (an absolute .where)
    input: question/presentation
    output: desired or generated response, possibly absent
```

In learning and evaluation, `Data` has both the input and desired output. In
inference, it initially has only the input and the model supplies the output.
Attaching an inferred output must not renumber later presentations.

An implementation may use parallel arrays, a record, or an adapter over the
current split tensors. The logical invariant is stable indexing with room for
both sides, not a prescribed storage class.

## 4. Supported question families

Question families are variations within the same run path, not new operating
controllers.

| Question | Desired response during learning/evaluation |
|---|---|
| `What(present)` | Current datum; it may equal the separate reconstruction target |
| `What(past, -k)` | Earlier datum at the resolved index |
| `What(future, +k)` | Later datum at the resolved index |
| Supervised question | Explicit supplied answer, which may differ from the input |
| Inference question | No `Data` output; use the `Model` response |

The input may itself contain missing past, present, or future tokens. Training
their accurate completion develops the world model. The input may instead be
a question such as “what is your name?”, whose correct output is not a
reconstruction. The question's semantics determine which output is desired.

Existing one-step predictors fit this contract by asking `What(future, +1)`.
The shared interface must also admit previous-state recall, present
reconstruction, wider temporal offsets, and non-temporal supervised answers.

## 5. Analysis, reconstruction, and output synthesis

> **Naming (2026-09-09):** the two downward paths are `Model.reverseReconstruct()`
> and `Model.reverseOutput()` — each is a dual of `forward()` along the reverse
> path, one seeded by the analysed input and ending at `InputSpace`, the other
> seeded by a resolved answer symbol and ending at `OutputSpace`. The Space-level
> algebraic inverse keeps the name `reverse()`; top-down realization of a
> generated state is `synthesize()`.

### 5.1 One bottom-up understanding

`Model.forward()` remains the one bottom-up analysis pass:

```text
InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace
```

Subsymbolic processing activates mereonymic parts and wholes and processes
them into conceptual space. Serial symbolic processing then builds the input
understanding. There is no additional surface-trace buffer. The live
perceptual context is the activation of those parts and wholes.
The conceptual field retains simultaneously active overlapping concepts, such
as car and tire. Symbolic salience selects a level for a description; it must
not erase the other activations needed for collective reconstruction.

The forward result must expose or internally retain an explicit per-call
`Understanding` with at least:

```text
Understanding:
    perceptual_context
    conceptual_state
    symbolic_state
    reconstruction_carriers
```

It contains no desired answer. It separates the immutable logical products of
one call from mutable singleton `Space.subspace` state so reconstruction and
output synthesis can consume it without overwriting one another.

### 5.2 Input reconstruction

`Model.reverseReconstruct(understanding)` replaces the ambiguous generative meaning
of model-level `reverse()`. It reconstructs the presented input through the
top-down inverse of its analysis:

```text
understanding.symbolic/conceptual state
    -> inverse symbolic/conceptual path
    -> inverse perceptual path
    -> InputSpace
```

When an operation is configured as invertible, this path uses the inverse
matrix of its forward transform. It may consume the matching forward-local
binding carriers in `understanding.reconstruction_carriers`. Those carriers
belong exclusively to reconstruction and may not be used as an answer. They
may retain routing, indices, masks, and binding decisions needed by the
inverse, but not a pre-analysis surface tensor or other payload that bypasses
the processed activation of mereonymic parts and wholes.

At the `Space` level, `reverse()` retains its algebraic inverse meaning. At
the `Model` level, `reverseReconstruct()` orchestrates the reverse calls and owns the
input-reconstruction result.

### 5.3 Rational output construction

`Model.reverseOutput(understanding, question)` replaces direct projection from the
input symbol to `OutputSpace`. It first performs serial symbolic decoding. An
interrogative symbol represents an operation with an unresolved position, not
a response surface to replay:

```text
input question symbol
    -> grammatical evaluation, lookup, comparison, binding, or thinking
    -> answer symbol
    -> ConceptualSpace.synthesize(...)
    -> PerceptualSpace.synthesize(..., context=perceptual_context)
    -> OutputSpace
```

The symbolic step may immediately yield a truth value or other simple answer.
When it cannot, it opens an LTM input slot and iterates until rational thought
produces a scoreable root answer. Only the resulting answer symbol descends
through conceptual and perceptual synthesis.

The target delegation is:

```text
derivation = SymbolicSpace.resolve(
    understanding.symbolic_state, question, grammar_context, LTM)
answer_concepts = ConceptualSpace.synthesize(
    derivation.answer_symbol, derivation.bindings,
    context=understanding.conceptual_state)
answer_percepts = PerceptualSpace.synthesize(
    answer_concepts,
    context=understanding.perceptual_context,
    selections=derivation.synthesis_references)
actual = OutputSpace.forward(answer_percepts)
```

`resolve()` returns either a resolved answer derivation or an instruction that
extends the LTM thinking stack. A resolved derivation carries the answer
symbol, grammar trace, output sentence location, constructed prefix, and named
synthesis references. It contains no desired `Data` answer.

`synthesize()` is top-down realization from a generated state. It shares the
inverse-direction chain as its backbone but applies each Space's DEDICATED,
identity-initialized invertible `synthesis_layer` (decision 2026-09-09: the
answer dual has its own weights, trained by `answer_construction` and never
by reconstruction). It accepts the answer carrier rather than a cached
reconstruction carrier. Do not create a second model or an unrelated
language generator.

Output synthesis may use the live perceptual field as context. Activated
parts and wholes can prime concrete names, objects, words, and grammatical
forms as the answer symbol is decoded into concepts and then percepts. Context
is not an unrestricted residual input: contributions must be selected through
named grammatical attention or binding operations and retained in the trace.
This preserves rational symbolic mediation and prevents a percept-to-output
copy shortcut.

Imagination may fill unspecified perceptual detail from parallel, overlapping
concepts and learned associations. A symbolic intention does not have to
serialize all that detail. Keep the selected conceptual references,
constraints, and any sampling state replayable; do not mistake constructive
completion for an exact inverse of the observed input.

`OutputSpace` is the final modality adapter over constructed answer percepts.
It can emit text, a truth value, a scalar, or an action. A narrow linear task
adapter may remain for compatibility, but it is not the implementation of
general `Model.what()` and does not receive the unresolved input symbol
directly.

### 5.4 Shared parameters, separate state

Reconstruction and output synthesis share the learned perceptual/conceptual
coordinate system and may share transform parameters. They do not share
mutable decode carriers:

- reconstruction receives the input-associated state and may use exact
  forward caches;
- output receives a newly constructed answer symbol and read-only contextual
  activations;
- output cannot overwrite state required by reconstruction;
- reconstruction cannot replay answer-generation state; and
- either path can be evaluated first without changing the other's result.

This is analogous to human repetition and response. Both use the concepts and
perceptual detail activated by an utterance. Repetition attempts to reinstate
the understood input; answering first resolves a new symbolic intention and
then realizes it through the same conceptual/perceptual substrate.

### 5.5 Collective invertibility and multi-reference concept formation

The contract is that **all active concepts together reconstitute all
percepts**. Neither one P0 nor one selected concept/symbol must decode the
entire field. Perceptual and conceptual transforms should be primarily
invertible on their valid analysis image before quantization. Quantize each
concept independently when forming symbolic references. Finite-codebook
snapping may lose detail; require an inverse in principle as codebook
resolution grows, not a per-symbol residual or a preserved surface trace.

The live bridge now uses addressed part/whole evidence and a learned Sigma
readout. `native_fold_activation` and the fold-source mean have been removed.
For each existing word-concept candidate at each location:

```text
e_i = clamp(1 - ||native_WHAT - code_i||^2 / ||code_i||^2, -1, 1)
a_c = tanh(b_c + sum_i W_ci * e_i)       # observed, admitted references only
event_WHAT = a_c * concept_code_c        # one indexed decode, not one per fold
```

The distance uses the completed P or W native field according to the
reference's role. Exact matches give +1, zero stimulation gives 0, and wrong
patterns can give negative evidence; zero-energy prototypes give 0. Unlike
RMS, equal-energy patterns with different directions remain distinguishable.
Known absence of a staged whole property supplies -1; an unobserved/padded
location is masked out. WHERE/WHEN are copied from the actual part location,
or the whole location when no part exists, never averaged. Prior SS content
retains its own identity/validity instead of entering the current word's
activation. This does not introduce a learned context selector.

`IndexedSigmaConceptsFromPercepts` is the row-local storage form of the
section 5.6 Sigma law. Each concept owns eight weights and a bias, plus up to
eight stable `(P-or-W, code)` references admitted at the eager vocabulary
boundary. Admission appends existing concept relationships, initially
interleaving P and W; either role can use spare slots. It never inserts
fillers, silently reassigns learned slots, or scans the concept inventory.
This deterministic admission policy is not yet learned reference selection.
Coefficients are gathered once per sentence; sparse gradients and compact
`RowLocalAdam` state update only touched concept rows. Stages sharing a concept
dictionary also share this readout and reference manifest.

CSSub retains the complete processed P0..P3 and W0..W3 stacks separately at
their native widths, plus reference IDs/roles/evidence/masks, beside the
concept event. The tensor-loop carry preserves this state independently of
symbolic selection. `unbind()` returns the exact terminal P/W states, not
tower averages. Compatibility binders for already-conceptual-width sources
also use plain Sigma and retain their full source stack. Batch/sentence
teardown releases the transient field; it is not durable raw text, a
per-symbol residual, or an answer-generation shortcut.

This is the minimal live readout cutover, **not** completion of collective
inversion. The current word-candidate/STM schedule remains; a richer set of
overlapping concepts, independent concept quantization, and reconstruction
from concepts without native caches still need implementation and validation.
First-level concept formation knits several references into each concept at
a coherent `.where`, not one concept per fold. Eight is a tractability budget
per candidate, not a bound on all parts/wholes or a required contributor count.

Checkpoint tensor state stores coefficient rows; conceptual `vocab_extras`
stores reference identities. Older checkpoints lacking the entire new modules
initialize them without changing existing weights; partially present modules
still fail strict loading. Capacity growth preserves learned row prefixes.
Eager/compiled readout and reconstruction smoke tests are covered; a matched
B24 quality/throughput comparison remains outstanding.

P0 currently uses `1 - product(1 - part_i)`, a parameter-free membership union
before the learned Sigma ladder, not a Pi fold. Keep it as an aggregate
feature; it does not by itself satisfy the collective reconstruction test.

Local inverse invariants, now regression-tested:

- Stable LDU effective diagonals choose a nonzero positive branch at exactly
  zero, including zero gates and cancellation by noise. `sign(0)` cannot be
  used as the supposedly nonzero sign factor.
- Generic butterfly pairs touching padded coordinates stay identity in both
  directions after training. Otherwise information can move into a padded
  coordinate which the public forward result discards.
- Ergodic factors are sampled before forward and after reverse, with the
  paired inverse using the forward sample. No noise-policy change is needed.
- Square transforms have two-sided inverses; expansions are left-invertible
  on their image. A dimension-reducing right inverse is not evidence that
  input details survived.

The full hard codebook lookup already determines the forward value. Its STE
is a separate backward estimator: nearest selection is locally constant in
the encoder input, so removing the estimator cuts encoder credit. Retain it
where that learning path is required; exact indexed row reads use ordinary
gather gradients. The separate percept-store UNORM read STE passes corrective
gradients through a clamp for out-of-cube float masters. It neither approximates
lookup nor guarantees invertibility. Tests demonstrate the distinction; no
blanket STE removal or quantization-residual requirement is part of this cut.

### 5.6 Reusable `conceptsFromPercepts()` readout and Sigma/L1 comparison

`bin/Layers.py` now provides a common `ConceptsFromPercepts(Layer)` contract
with interchangeable `GatedConceptsFromPercepts` and
`SigmaConceptsFromPercepts` implementations. Both `forward()` and
`conceptsFromPercepts()` consume `percepts[..., I]`, an optional boolean
evidence mask, and (for the gated variant) optional local context. The result
is `[..., C]` independent signed concept activations. Input coordinates must
have stable `(space, code)` identities, including the part/whole role; output
coordinates identify concepts. This is a bounded candidate-bank primitive,
not a dense vocabulary-wide selector or a replacement for native WHAT
content. The caller supplies identity mappings and spatial admission.

The learnable gated formula is:

```text
g_ic = sigmoid(gate_logits_ic + sum_d context_d * context_weights_dic)
a_c  = tanh(b_c + sum_i evidence_i * g_ic * W_ic)
```

No input RMS, division by contributor count, or competition across concepts
is introduced. Unknown/masked evidence is excluded, not interpreted as
observed absence (-1); a completely unobserved candidate bank returns zero
even with nonzero bias. Context modulates participation without a direct
context-to-output term. It must come from available local understanding,
not the desired answer. An omitted context is neutral zero context.

Familiar serial recognition can use a WholeSpace word-property reference
and a particular PartSpace word-code reference. Additional serial operands
can create new Sigma unions of parts and Pi intersections of wholes; that
construction stays with those existing operators, not this scalar readout.
For membership composition, excluded union operands are neutral zero and
excluded intersection operands are neutral one. Parallel processing can use
the richer admitted reference set and multiple overlapping concepts. Neither
variant forces a two-reference solution, and neither erases the native
conceptual content needed for collective perceptual decoding.

The simpler variant is `tanh(b + Sigma(evidence))` with the existing Sigma
linear core (`nonlinear=False`) and explicit affine bias. This deliberately
does not apply the legacy Sigma input `atanh` chart. Its optional lasso is:

```text
R(W) = l1_lambda * sum_ic abs(W_ic) / C
```

Sparsity is on incoming **connections per concept**, not on percepts
globally, biases, codebook rows, or LDU factors. The same percept can retain a
nonzero connection to another concept. Zero lambda is the unregularized
control. Sigmoid gates and L1 do not enforce a hard maximum of eight;
candidate admission remains a separate responsibility. Exact zero weights
in a dense matrix also do not automatically save computation.

The dense experimental primitive offers two training choices, never both on
the same update:

- Add `regularization_loss()` to the smooth objective and use its gradient.
- Backpropagate the smooth objective alone, take a plain SGD step, then call
  `proximal_step_(learning_rate)`. The threshold is
  `learning_rate * l1_lambda / C`, producing exact zeros without penalizing
  bias. This is the standard [L1 proximal operator](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf),
  not the existing activation-shrink `SparsityRegLayer`. The scalar step is
  not an Adam/momentum-aware proximal update; zero edges may reactivate.

Neither readout claims its scalar output is an inverse carrier, and
`reverse()` explicitly rejects that interpretation.

#### Live weak-L1 integration

BasicModel selects `ConceptualSpace/conceptReadoutL1=0.01`; absent or zero
configuration preserves the unregularized behavior of other models. This
is a separate knob from the existing activation `l1Lambda`. The live indexed
Sigma uses the following penalty over the **distinct observed concepts** in
one batch, with `M` marking their admitted input connections:

```text
R = lambda / C_observed * sum_ci M_ci * abs(W_ci)
```

Repeated words, padding, dictionary capacity, and shared stage aliases do not
multiply this cost. Unknown references, unused slots, biases, and codebook
vectors are not penalized. The readout stays Sigma/tanh without gates; zero
connections retain their reference identity and may reactivate through later
smooth gradients. There is no hard pruning or forced two-reference definition.

The sparse row-local optimizer keeps its existing smooth-gradient Adam
moments, then applies the exact L1 proximal map in Adam's diagonal metric
inside the **same optimizer step**. For the admitted coefficients:

```text
D = sqrt(v_hat) + epsilon
y = W - learning_rate * m_hat / D
threshold = learning_rate * (lambda / C_observed) / D
W_new = sign(y) * max(abs(y) - threshold, 0)
```

This is a specialization of the metric proximal update in
[Proximal Adam, section 3](https://arxiv.org/html/1910.10094v2#S3), not the
dense experiment's uniform SGD threshold. Bias/unused columns retain their
ordinary smooth update. Only selected/gradient-touched rows and compact
moments are materialized, never a vocabulary-wide dense gradient. An observed
row with a zero smooth gradient can still receive sparsity pressure; an
entirely disconnected parameter (`grad=None`) stays untouched.
The shipped detached-reverse configuration can still produce that disconnected
case. This integration does not reconnect the encoder or train it by shrinkage
alone; the existing reconstruction-credit boundary remains separate work.
Regression tests cover an actual coupled `runBatch()` readout update and the
detached no-update case, along with an exact-two-reference synthetic fit.

`runBatch()` reports `concept_readout_l1` separately as a detached regularizer
cost. It does not backpropagate L1 as well as applying the prox. Reconstruction
and output costs keep their original definitions, and output-conflict
projection/norm capping happens before this update. L1 is the deliberately
accepted sparse-basis tradeoff: it can increase reconstruction error. Neither
Adam nor this regularizer guarantees monotonic reconstruction improvement.

The per-batch observation policy is consumed once and cleared on `zero_grad`
and checkpoint loading, including after an AMP-skipped step. Evaluation never
stages it. No new parameter or optimizer-state schema is needed: coefficient
rows, reference identities, and Adam moments retain their existing checkpoint
paths, while lambda comes from the current XML configuration.

#### Controlled comparison (2026-09-08)

Reproduce with:

```sh
BASICMODEL_DEVICE=cpu .venv/bin/python bin/bench_concepts_from_percepts.py
```

The probe has eight stable coordinates (four P and four W), eight concept
outputs, and known nonsingular target mixes with two or six references per
concept. A fixed target inverse decodes predicted activations. The only
smooth training objective is input reconstruction; activation fidelity is
evaluated separately. The decoder is fixed so it cannot counteract lasso by
rescaling its weights. No codebooks, learned inverse, output-construction
loss, native percept field, or language semantics are being tested.

Settings: three seeds (1, 2, 3), 256 training / 2,048 held-out samples, 1,500
full-batch plain-SGD steps, learning rate 0.3, one CPU thread, matched initial
effective weights; macOS ARM64, PyTorch `2.14.0.dev20260722`. The table
averages seeds; effective support counts
`abs(W_eff) > 0.02`, while exact support counts every nonzero coefficient.

| Target refs/concept | Readout | L1 | Reconstruction MSE | Exact nonzeros/concept | Effective inputs/concept |
|---|---|---:|---:|---:|---:|
| 2 | Gated | 0 | 3.21e-5 | 8.00 | 2.00 |
| 2 | Sigma | 0 | 1.32e-11 | 8.00 | 2.00 |
| 2 | Sigma | 0.01 | 3.95e-4 | 2.00 | 2.00 |
| 2 | Sigma | 0.1 | 3.29e-2 | 1.08 | 1.04 |
| 6 | Gated | 0 | 8.71e-4 | 8.00 | 7.75 |
| 6 | Sigma | 0 | 3.93e-7 | 8.00 | 6.00 |
| 6 | Sigma | 0.01 | 1.94e-3 | 7.33 | 6.08 |
| 6 | Sigma | 0.1 | 3.37e-2 | 1.00 | 1.00 |

The gated control uses 136 parameters; Sigma uses 72. The matched fixed
training budget favors plain Sigma on these **context-free** targets. Weak
L1 produces exactly the desired pair on the sparse task, with shrinkage
distortion. Strong L1 deletes useful references on both tasks, especially
the richer one. This supports keeping a simple Sigma control and choosing
sparsity against a reconstruction tolerance, not minimizing input count
alone. The follow-up selects Sigma **with weak L1** for sparse conceptual
definitions, accepting its reconstruction tradeoff. The table above remains
the controlled SGD comparison, not measured live proximal-Adam performance.
Contextual selection,
collective concept-only reconstruction, quantization, and B24 throughput
still need matched end-to-end tests.

## 6. LTM interaction slots

### 6.1 Store the model's input and response

Refine each LTM interaction slot to hold independently optional conceptual
representations of input and output:

```text
LTMSlot:
    input:  optional one-, two-, or three-slot input representation
    output: optional actual answer conceptual/symbolic representation
```

The output is the response actually produced by `Model`, not the desired
`Data` target. Supervised targets train the response through loss; they are
not copied into LTM as if the model had produced them.

The output may differ from the input. This is required for past/future
queries and ordinary question answering.

### 6.2 Slot parity supplies the thinking stack

The two halves of an LTM slot are independently optional:

- `(input, —)` opens an unanswered question;
- `(input, output)` records a complete interaction and does not change stack
  depth; and
- `(—, output)` closes the most recent unanswered question.

A slot with neither side is invalid. The chronological LTM sequence itself
encodes the stack; do not add a separate recursive frame hierarchy. Scanning
the sequence applies input-only slots as pushes and output-only slots as
LIFO pops. **Parity** means there are no unmatched input-only slots.

The opening input-only slot remains unchanged. A later output-only slot closes
it; implementations must not depend on filling the old slot in place.

## 7. Iterative thinking

The [mathematical thinking plan and tests](../plans/2026-09-09-mathematical-thinking.md)
define a bounded implementation milestone and behavioral acceptance tests for
this loop, using dependency arithmetic and progressive constraint resolution.

### 7.1 When thinking starts

All questions are queries into conceptual space. If the truth or illumination
of the relevant spaces is insufficient, `Model` may defer an answer and pose
a subquestion. That choice produces an input-only LTM slot and begins an
internal dialogue.

An internal subquestion uses the model's conceptual state. It cannot supply
the sentence index used internally by `Data`, because that coordinate is not
available to the model's self-query.

### 7.2 Evaluation loop

Thinking is iterative program evaluation, not Python recursion:

```text
evaluate the presented question through Model.what()
append the resulting LTM slot

while LTM is not at parity:
    evaluate the active conceptual question with the enlarged LTM context
    append the resulting LTM slot
```

An iteration may add another input-only subquestion, a complete
subquestion/answer slot, or an output-only answer to the most recent open
question. Results of completed subquestions are visible in LTM on the next
iteration and therefore transform the context in which the parent question
is answered.

For the supervised pair `Q: what is your name?`, `A: Alec`, one valid trace
is:

```text
(what is your name?, —)   # open the root question
(who is asking?, OpenAI)  # balanced subquestion adds useful context
(—, Alec)                 # close the root question
```

Only after the LTM returns to parity is `Alec` compared with the supervised
target.

### 7.3 Closure pressure and hard limit

Thinking must conclude. Add a closure-pressure value to the context already
used by the grammar chooser. It must increase monotonically on every
iteration that ends without parity. Rising pressure favors answering the most
recent open question and disfavors opening another question.

The exact schedule is configurable and should be selected by ablation. It
must have a finite limit. At that limit:

- the model emits its best-effort answer for the most recent open question;
- low confidence is allowed;
- `unknown`, `unresolved`, or a failure status is not a valid substitute for
  an answer; and
- if multiple questions remain open, forced best-effort output-only slots
  continue from the top until parity is restored.

This rule guarantees a scoreable root response and prevents indefinite
internal dialogue.

## 8. Grammar and the mind boundary

The existing grammar chooser remains the control mechanism. Its established
context already includes symbolic activation, STM, LTM, percepts, and
grammatical operations. Do not replace it with a new address-specific chooser
or an imprecise duplicate context object.

Add only the state required by this iteration:

- temporal meaning carried by the active question;
- the input/output masks and representations of relevant LTM slots;
- current parity or open-stack depth; and
- closure pressure;
- the resolved or still-open answer symbol;
- the current output grammatical derivation and sentence location;
- the output already constructed; and
- named perceptual/conceptual references available to synthesis.

The chooser must be able to select among answering the current input, leaving
it open while posing a subquestion, answering a subquestion in the same slot,
and emitting an output-only answer for the most recent open input. After an
answer symbol exists, the same chooser controls which grammatical derivation
realizes it and where that derivation contributes to the output sentence.

The existing symbolic-mind boundary remains normative:

- perception and the lexicon may present subsymbolic content;
- deliberate transformations of that content use named grammatical
  operations with explicit operands and results;
- LTM lookup, comparison, binding, and answer construction enter through
  grammatical operations rather than an unrecorded neural side channel; and
- grammar choice, word choice, and memory use retain a replayable trace.

The implementation mechanism may use the current grammar layers and trace
structures. This iteration does not require a general-purpose query planner,
new recursive grammar, or second generation model.

## 9. Training and loss

### 9.1 One training graph

All question families use the existing `Model.run()` training orchestrator.
For one presentation it constructs this graph:

```text
understanding = Model.forward(input)

input_prediction = Model.reverseReconstruct(understanding)
actual_answer = Model.reverseOutput(understanding, question)

desired_answer = Data.what(question)  # loss authority only

L_input  = compare(input_prediction, input_reconstruction_target)
L_answer = compare(actual_answer.what, desired_answer.what)
L_total  = weighted_sum(L_input, L_answer, existing auxiliary losses) # reporting

optimizer.zero_grad()
differentiate_branches_and_apply_reconstruction_priority(...) # section 9.4
optimizer.step()
```

Autograd differentiation is unrelated to the model's representational
`reverse()` methods. Differentiate the branch objectives separately at one
unchanged parameter version, apply the asymmetric projection below, and
perform exactly one optimizer step. The scalar reporting total no longer
implies a plain gradient sum on protected parameters. Do not detach the shared
understanding between branches or update parameters between traversals.

The desired answer may be selected before or independently of model execution,
but it remains unavailable to `Model.forward()`, `Model.reverseOutput()`, grammar,
STM, LTM, perceptual context, and generated output. It enters only target
preparation and loss calculation. A target-preparation adapter is allowed; a
second full model pass over the clean answer to manufacture privileged
concepts is not.

### 9.2 Separate primary costs

The two primary costs are independently present and independently weighted:

```text
L_primary = lambda_input * L_input_reconstruction
          + lambda_answer * L_answer_construction
```

`L_input_reconstruction` compares the reconstructed percept/input with the
presented input, or with the clean form when the presentation is deliberately
degraded. It trains the bottom-up representation and its input-associated
inverse. It does not compare against a past, future, or supervised answer.

`L_answer_construction` compares only the response constructed by
`Model.reverseOutput()` with `Data.what(question)`. Question coordinates are model
context, not output targets. For present, past, future, and supervised
questions, the resolved `Data.what()` value replaces the loader's incidental
output tensor as the authoritative answer target. When `Data.what()` is
unavailable during inference, `L_answer_construction` is absent rather than
zero-labeled.

The two weights are not required to sum to one. Each loss is normalized over
its own valid elements before weighting so a change in sentence length,
answer length, masking rate, or frequency of supervised examples does not
silently change the tradeoff. Existing grammar, memory, commitment,
boundedness, truth, and chooser losses remain separately named auxiliary
terms.

This preserves the scale advantage of unsupervised data. Every unlabeled
presentation can supply input reconstruction, and ordered text can also
construct present, previous, next, or degraded-input answer targets without a
human label. Explicit question/answer data adds answer-construction targets
that need not resemble the input. These are sampling modes for the same two
losses, not separate models or optimizer phases.

### 9.3 Where the gradients meet

Let `theta_shared` be parameters in the bottom-up perceptual/conceptual/
symbolic understanding used by both branches. Before priority projection,
the ordinary gradient is:

```text
dL_total/dtheta_shared
    = lambda_input  * dL_input/dtheta_shared
    + lambda_answer * dL_answer/dtheta_shared
    + auxiliary gradients
```

At the shared activation `h = forward(input)`, the unmodified gradient is:

```text
dL_total/dh
    = lambda_input  * dL_input/dh
    + lambda_answer * dL_answer/dh
    + auxiliary gradients
```

The reconstruction operation is therefore independent in product, target,
and per-call state, but not independent in learning pressure. Conversely,
autograd `backward()` is not a second model path: it differentiates the one
graph after both branch costs have been formed. The applied perceptual/
conceptual gradient obeys section 9.4 instead of simple addition. `forward()` should be
a target-free computation returning the understanding; `Model.run()` owns the
costs.

The reconstruction gradient reaches the shared understanding through the
inverse perceptual and conceptual path. The answer gradient reaches it through
`OutputSpace`, answer-percept synthesis, answer-concept synthesis, rational
symbolic decoding or thinking, and the input symbolic understanding from which
that process began. This answer gradient is required: the internal
conceptualization must become useful for correct judgment and response, not
only for replaying its input.

Shared invertible parameters remain structurally invertible. If a forward
operation uses `W` and reconstruction/synthesis uses its mathematical inverse,
both losses may update `W`; the next forward and inverse still use the same
updated parameterization. Output-specific grammar choices, attention, and
synthesis gates may have separate parameters without creating a separate
conceptual space.

For a tied parameter used in both directions, the answer derivative contains
two contributions:

```text
dL_answer/dW
    = (dL_answer/dh) * (dh/dW)        # analysis/understanding use
    + (dL_answer/d(W^-1)) * (d(W^-1)/dW) # synthesis use
```

This is not double-counting the answer cost; it is the chain rule for two uses
of the same parameter. Retain both terms before priority projection. The inverse derivative can
become large near singularity, so report conditioning or the existing
invertibility health metric and test the gradient on a well-conditioned
operator. An exact unmasked identity round trip has zero reconstruction error
and contributes no training signal. Masking or degradation is also
insufficient if the masked carrier still traverses only a cancelling
`W^-1 W` pair. Contextual completion, compression, a bottleneck, or another
non-cancelling stage must infer information that is absent from every carrier.
Assert an actual gradient at the shared understanding rather than assuming
one from a nonzero reconstruction error.

Perceptual context is differentiable only through explicit, replayable
grammatical operands such as named attention, lookup, or binding. This allows
answer loss to teach which active parts and wholes matter, while prohibiting
an unrecorded residual from input percepts directly to `OutputSpace`. The
actual model response, not the desired answer, is the output representation
stored in LTM.

### 9.4 Balancing and diagnosing pressure

Reconstruction is primary. For each optimizer-owned parameter tensor in the
shared forward perceptual and conceptual spaces (including WholeSpace),
compute `r = grad(weighted_reconstruction)` and `o = grad(weighted_output)`:

```text
q = o
if dot(r, o) < 0 and ||r|| > 0:
    q = o - dot(r, o) / dot(r, r) * r
q = q * min(1, rho * ||r|| / ||q||)   # zero q remains zero
if r is missing or zero: q = 0
applied_gradient = r + q + auxiliary_gradients
0 <= rho < 1                         # default rho = 0.5
```

Only the opposing projection is removed. Positively aligned output credit is
retained, subject to the magnitude cap. Reconstruction is never projected
against output. Include both analysis and inverse-synthesis uses of a tied
parameter in each objective's derivative. Use actual loss weights and any
live truth multiplier, not merely logged raw losses or detached multipliers.
Independent symbolic/output heads keep their ordinary output gradient.

Implement at the existing optimizer boundary; preserve sparse COO support
over touched rows. Do not subtract a huge output gradient from an already
accumulated total gradient, since cancellation can destroy the smaller
reconstruction signal. `backward_reconstruction_priority` differentiates
output, clears its protected gradients, differentiates the remaining loss,
then adds the projected output contribution. One optimizer step follows all
traversals; AMP scales all branch objectives equally before its usual unscale.
With no output objective, use the established single-backward path.

This guarantees `dot(r,q) >= 0` and `||q|| <= rho*||r||` up to numerical
precision. It is a first-order gradient constraint, not guaranteed monotonic
reconstruction improvement after a finite Adam/momentum step. Other auxiliary
gradients are not automatically protected. Training and evaluation must report:

- unweighted and weighted input-reconstruction loss;
- unweighted and weighted answer-construction loss by question family;
- gradient norms from each primary loss at the shared conceptual and symbolic
  branch points;
- cosine similarity between those two gradient contributions;
- the opposing component removed and remaining norm capped on protected
  parameters;
- input reconstruction quality alongside answer quality; and
- the existing grammar, memory, and throughput measurements.

An exact identity inverse, detached reconstruction or a disconnected encoder
may give `r=0`: then output has no budget on the protected parameters, while
independent output heads can still learn. This is an explicit consequence of
the requested norm priority. Do not insert a fake reconstruction error or
quantization residual to avoid it. Current BasicModel detaches its reverse
student and rotates its shared concept dictionary outside autograd; completing
the joint field architecture requires an explicit decision at those credit
boundaries. Perceptual fidelity, conceptual usefulness, and answer quality
must all be measured; projection alone does not establish any of them.

### 9.5 Thinking and credit

For a supervised presentation that thinks, calculate the answer loss only
after parity. The trace must retain the sampled grammatical decisions that
opened, extended, and closed the dialogue so the existing chooser-learning
path can assign credit. Intermediate answers may receive applicable
self-supervised, grammar, or memory losses, but they are not compared with the
root's desired answer.

Within one bounded thinking episode, retain differentiable generated states
until the root loss is formed; the finite iteration limit is also the initial
memory bound. Detach durable LTM entries after the episode/optimizer boundary
so training does not backpropagate through unbounded history. Any later
truncated-credit boundary must be explicit, configurable, and reported rather
than arising from an incidental detach. The final answer loss trains
continuous operands and the selected synthesis route. Hard chooser decisions
require the existing straight-through, two-pass, or policy-loss estimator;
ordinary autograd cannot credit an argmax or an unchosen branch. Record which
mechanism supplied each chooser update.

Closure pressure is a chooser input, not permission to replace the answer
loss with a parity-only objective. Returning quickly with a wrong answer and
thinking indefinitely are both undesirable; report answer quality and
thinking length separately.

## 10. Implementation sequence

### Step 0: retain the implemented foundation

Keep the merged question representation, `Data.what()` lookup, paired LTM
slot/parity substrate, target-free chooser context, and compatibility
adapters. Treat their existing tests as regression tests. They do not make the
direct output head or incidental loader target part of the target design.

### Step 1: capture one explicit `Understanding`

Refactor the bottom-up `forward()` result into a per-call `Understanding` that
retains perceptual context, conceptual state, symbolic state, and the exact
input-associated carriers required for reconstruction. Remove reliance on one
mutable `Space.subspace` value being overwritten successively by analysis,
reconstruction, and generation. Preserve a temporary adapter for callers that
still expect the established forward tuple.

### Step 2: repair the collective field and make reconstruction explicit

The RMS/mean bridge removal, sparse Sigma integration, native-field tensor
carry, checkpoint migration, and local LDU/padding fixes are implemented
(section 5.5). Continue the field-level work: keep distinct overlapping
concepts through the conceptual carrier and quantize each separately for
symbolic selection. Demonstrate collective percept recovery from all concepts
without requiring one symbol to contain the field, retaining quantization
residuals, or mistaking cached native-state recovery for concept-only inversion.
The explicit snapshot API and that broader field/STM migration remain unfinished.

Implement `Model.reverseReconstruct(understanding)` over the current inverse path and
reserve `Space.reverse()` for input-associated algebraic inversion. When
`invertible` is enabled, keep the forward and inverse matrices tied; do not
clone a second reconstruction model. Return the reconstructed input and a
named, normalized `input_reconstruction` cost without producing an answer.

An exactly invertible, unmasked identity round trip has zero error and
therefore supplies no learning signal. A mask or degraded input is useful only
when a contextual prediction/compression operation must infer information
that is unavailable in every inverse carrier. Reconstruction training must
retain that real information challenge rather than claiming pressure from an
algebraic identity or a nonzero but parameter-independent error.

### Step 3: add answer synthesis through the spaces

Add generated-carrier entry points to `ConceptualSpace.synthesize()` and
`PerceptualSpace.synthesize()`. They may invoke the same inverse-direction
operators used by reconstruction, including tied invertible matrices, but may
not read reconstruction-only carriers. Add explicit grammatical attention or
binding inputs for perceptual context, then make `OutputSpace` consume the
constructed answer percepts as a modality adapter.

Keep a narrow old projection only as a migration oracle. The completed
`Model.reverseOutput()` path is:

```text
symbolic answer/thought -> conceptual synthesis -> perceptual synthesis
    -> OutputSpace
```

### Step 4: route `Model.what()` through thought and synthesis

Change `Model.what()` from a wrapper around the established projected head to
`forward()` followed by rational symbolic evaluation and `reverseOutput()`. Connect
the existing grammar chooser, LTM context, answer symbol, derivation, sentence
location, constructed prefix, and named synthesis references. Simple truth or
binding questions may terminate in one operation; unresolved questions enter
the existing iterative parity loop.

### Step 5: make `Data.what()` the answer-loss authority

Replace the loader's incidental `outputTensor` as the answer-construction
target for present, past, future, and supervised questions. Keep the desired
answer out of every model-visible context and use it only in loss calculation.
Expose the old `lossIn`/`lossOut` bookkeeping, if retained internally, under
the unambiguous names `input_reconstruction` and `answer_construction`.

### Step 6: join the losses without joining branch state

Build reconstruction and answer construction from the same `Understanding`,
differentiate their weighted costs separately, apply reconstruction-priority
projection/capping on shared perceptual/conceptual parameters, and perform one
optimizer step. The legacy training seam now supports this policy; wire the
new reconstruction/output objectives and their independent weights into it.
Do not detach the answer branch at the conceptual or symbolic state. Resolve
the current detached-reverse/context-dictionary credit boundaries explicitly.
Do not alternate parameter versions between branches. Add sampled
gradient-norm and gradient-cosine diagnostics at the conceptual and symbolic
branch points; diagnostic gradient reads must not perform another update.

For a tied operator used during both analysis and synthesis, preserve both
parts of the derivative: the indirect derivative through the analyzed
understanding and the direct derivative through the synthesis use of the
operator. Add conditioning checks for matrix inverses rather than silently
breaking the tie when their gradients are difficult.

### Step 7: make temporal context causal

Train present, past, and future questions. Ablate or shuffle temporal question
content while holding the input fixed. The model must perform worse when that
content is wrong; otherwise it has not learned the query.

### Step 8: finish LTM-driven thought

Store conceptual input and actual constructed output with independent presence
masks. Derive the open-question stack from input-only and output-only slots,
feed completed subquestions into later evaluations, and finish grammar-driven
root-answer scoring. Preserve ordered LTM behavior, capacity limits, document
resets, detachment from expired graphs, monotonic closure pressure, a finite
iteration limit, and forced best-effort LIFO closure.

### Step 9: remove the obsolete boundary and expand the curriculum

After the reconstruction and answer-synthesis acceptance tests pass, migrate
callers off Teacher-oriented lesson APIs and the direct symbolic output head.
Delete or reduce the old classes to temporary compatibility adapters that
delegate to `Data.what()` and `Model.run()`. Begin with clean present
reconstruction plus simple supervised answers, then mix one-step past/future
completion and inference. Increase temporal distance and thinking depth only
after causal-use and parity tests pass. Run the matched B24 benchmark after
each architectural cutover.

## 11. Required tests

### Interfaces and compatibility

- `Data.what(What(present))` selects the same target as current
  reconstruction.
- One bottom-up `forward()` result is shared by reconstruction and answer
  construction; neither branch performs a second privileged encoding.
- `Model.reverseReconstruct()` returns the input reconstruction, and `Model.what()`
  reaches `Model.reverseOutput()` rather than the legacy projected head.
- Reconstruction and output can run in either order with identical results
  and without overwriting one another's carriers.
- A compatibility adapter is byte-identical to the established clean path
  before the new answer path is enabled.
- `Model.run()` remains the only training/evaluation orchestrator.
- No target architecture requires a Teacher, Student, source-oracle, view, or
  recursive frame class.

### Reconstruction and answer paths

- With `invertible` enabled, reconstruction uses the mathematical inverse of
  the matching forward transform and its input-associated carrier.
- Learned non-power-of-two butterflies, exact-zero LDU diagonals, zero gates,
  and ergodic paired samples have strict finite round-trip tests.
- With quantization disabled, independently active conceptual fields recover
  their percepts together. Symbolic salience selection leaves non-selected
  activations available; no test requires a single P0 or symbol to recover all.
- Distinct equal-RMS native fields remain distinguishable before quantization;
  independent concept identities/activations are not replaced by a source mean.
- A word-property whole and matching word-code part can activate one concept
  without six fillers. Additional serial operands remain usable for new
  unions/intersections, and richer parallel support remains available to
  reconstruction. Readout experiments distinguish exact weight sparsity from
  effective support and measure reconstruction distortion, not sparsity alone.
- Finite snapping is assessed by reconstruction distortion and increasing
  codebook resolution, without a per-symbol residual channel.
- Reconstruction does not consume an answer carrier; output synthesis does
  not consume a reconstruction carrier.
- A generated answer starts at the resolved symbolic answer and traverses
  conceptual synthesis, perceptual synthesis, and `OutputSpace`.
- There is no direct unresolved-SymbolicSpace-to-OutputSpace or unrestricted
  PerceptualSpace-to-OutputSpace shortcut.
- Perceptual context reaches output only through named grammatical attention
  or binding recorded in the trace.
- Holding the answer symbol fixed while changing a selected perceptual binding
  can change concrete realization; changing unrelated live activation cannot.
- Both an immediate truth/binding answer and a multi-step thought answer use
  the same synthesis interfaces.

### Coordinates and targets

- Sentence indices are zero based and stable when an output is attached.
- Past/future offsets select the correct row and never cross a document or
  split boundary.
- Changing the question's temporal meaning changes the desired target while
  holding the presentation fixed.
- Coordinates appear in model context but not in the compared output.
- Shuffling or zeroing temporal context measurably harms address-sensitive
  tasks.

### Learning isolation

- The desired `Data` output cannot be reached from the `Model.what()` call
  graph before loss.
- A supervised or future target does not enter LTM or runtime caches as the
  model's own response.
- LTM stores the response actually produced, including an incorrect response.
- Inference may attach a generated output without relabeling it as a supplied
  training target.
- Target preparation does not run the answer through a second full model
  encoding or add it to perceptual, conceptual, symbolic, STM, or LTM state.

### Loss and gradients

- `input_reconstruction` and `answer_construction` are normalized and reported
  both before and after their independent weights are applied.
- With only input-reconstruction loss enabled, shared conceptual parameters
  receive its gradient on a contextual-completion example whose removed
  information is absent from every reconstruction carrier.
- The raw answer loss differentiates through synthesis and shared analysis;
  its applied contribution on protected parameters obeys reconstruction
  priority. With no reconstruction gradient there, only independent heads
  retain output credit.
- Conflicting, aligned, orthogonal, zero, sparse, and tied-parameter gradients
  obey `dot(r,q) >= 0` and `||q|| <= rho*||r||`; the applied shared gradient
  equals `r + q` before auxiliary terms. An enormous opposing output gradient
  does not erase the smaller reconstruction gradient by numeric cancellation.
- Independent decoder and auxiliary gradients are unchanged; AMP applies a
  common scale and sparse dictionary gradients never require a dense inventory.
- A tied invertible operator receives the direct synthesis derivative and the
  indirect derivative through `forward()`; an autograd-versus-finite-difference
  test covers a well-conditioned small matrix.
- Neither primary branch is detached at the shared understanding, and there is
  exactly one optimizer step for their combined loss.
- Changing one primary weight changes its contribution without implicitly
  renormalizing the other; missing inference targets omit the answer loss.
- Gradient norms and cosine similarity are available at conceptual and
  symbolic branch points without changing the resulting optimizer update.
- An exact unmasked inverse identity is not counted as evidence of useful
  reconstruction learning.
- Reconstruction carriers contain inverse-routing metadata but no recoverable
  copy of the pre-analysis surface payload; perturbing the shared conceptual
  state changes the reconstruction.

### LTM parity and thinking

- An input-only slot increases derived open depth by one.
- A complete input/output slot leaves open depth unchanged.
- An output-only slot closes the most recent open input.
- An empty slot is rejected, and an output-only slot with no open input is
  rejected.
- Nested open inputs close in LIFO order.
- Opening slots are not modified in place when later answers close them.
- A completed subquestion changes the LTM context used for the next
  evaluation.
- Thinking uses an iterative loop and cannot grow the Python call stack.
- Closure pressure is monotonic and cannot reset when another subquestion is
  opened.
- At the limit, best-effort output-only slots restore parity for every open
  question; unknown/failure cannot bypass this behavior.
- The final root response, not an intermediate subanswer, is scored against
  the presented supervised target.
- Differentiable state is retained only for the bounded active thought
  episode; durable LTM history is detached at its declared boundary.
- Continuous answer credit and hard-choice policy credit are reported
  distinctly, and no test assumes autograd differentiates an argmax.

### Grammar and performance

- Slot operations and temporal question use appear in the grammatical trace.
- No hidden semantic-to-output path bypasses the grammar chooser.
- Output derivation, sentence location, constructed prefix, and every
  perceptual/conceptual synthesis reference appear in the replayable trace.
- Present reconstruction, past recall, future completion, supervised answer
  quality, both primary losses, gradient interaction, mean thinking
  iterations, forced-closure rate, and sentences/s are reported separately.
- Existing stateless/stateful, device, checkpoint, and two-pass chooser tests
  remain green.

## 12. Performance and definition of done

Use the matched B24 benchmark and report median wall-clock throughput after
warm-up. The 41.989 complete sentences/s result is the clean reference.

The next iteration is complete when:

1. `Data.what()` is the authoritative answer target and both existing classes
   expose working `what()` delegation without target leakage.
2. `forward()` produces one explicit per-call `Understanding` containing no
   desired answer.
3. Input reconstruction exclusively uses the input-associated inverse branch,
   including tied inverse matrices when configured.
4. Answer construction exclusively uses rational symbolic decoding followed
   by conceptual and perceptual synthesis; the direct symbolic projection is
   no longer the general `Model.what()` implementation.
5. Reconstruction and output have separate state, targets, normalized costs,
   weights, and reporting, while sharing the intended parameters and live
   understanding.
6. Separate primary-gradient inspection and reconstruction-priority projection
   precede one optimizer step; gradient-path, non-opposition, and norm-cap
   tests pass. The collective pre-quantization conceptual handoff is invertible
   in principle, without source averaging or per-symbol residual preservation.
7. Perceptual context affects output only through named, replayable grammar
   choices, bindings, or attention.
8. `Model.run()` trains and evaluates present, past, future, and supervised
   questions through this one execution and optimization path.
9. Temporal question content is causally used.
10. LTM stores conceptual input and actual constructed model output in
    independently optional halves.
11. Input-only and output-only slots implement a LIFO stack without a separate
    recursive frame structure.
12. Thinking incorporates completed subquestions, constructs the final root
    answer through the same output path, and always restores parity with
    best-effort answers.
13. Target isolation, document-boundary, branch-order, and synthesis-path
    tests pass.
14. Input reconstruction and answer quality both remain inside their declared
    acceptance bands under joint training; any persistent gradient conflict is
    measured and resolved explicitly rather than hidden by detachment.
15. Clean B24 median throughput is within 15% of the matched baseline, or any
    larger regression is isolated and explicitly accepted.
16. Teacher-oriented code is no longer an architectural dependency.

## 13. Preserved roadmap: noun and verb concept formation

The following work remains downstream of the shared `what()` loop. It is not
claimed as implemented by the immediate iteration.

### 13.1 Synchronic and diachronic concept order

The symbolic-order recurrence must eventually learn concepts from percepts
and concepts of the preceding order in two typed forms:

```text
N^(k+1)_t = synchronic_lift(P_t, W_t, C^k_t, ...)
V^(k+1)_(t1->t2) = diachronic_lift(C^k_t1, C^k_t2)

V^1 = [part = C^0_before, whole = C^0_after, kind = temporal,
       from_event = t1, to_event = t2]
```

Same-time combinations form higher-order nouns. Cross-time combinations form
verbs. Before/after roles retain direction. These event coordinates do not
replace the model's subjective `.when`.

Verb learning should satisfy:

```text
verb(initial noun concept) ~= final noun concept
L_VP = E[distance(T_VP(C_NP,t), C_NP,t+1)]
```

Repeated examples across different tracked individuals must improve one
reusable temporal concept rather than memorize sentence pairs. Begin with
unary transformations and later extend to role-bound argument tuples.

As initial admission thresholds, require at least four eligible transitions
across two tracked noun identities, an EMA transition error at least five
percent below identity/no-verb copying, and improvement on held-out eligible
transitions. These thresholds are ablation starting points, not proof of
causality.

### 13.2 Meaning and grammatical application

A surface verb's MetaSymbol binds to learned diachronic concept rows. Grammar
and argument roles select the intended sense and pass its atom to the shared
executor. `VerbLayer` applies a meaning but does not create it. Noun selection
likewise uses learned concepts rather than treating a surface row as the
object's meaning.

A transformation may be forward-only and many-to-one. Reconstructing a past
state is an abductive `Model.what()` query using memory; it is not a required
algebraic inverse.

### 13.3 Row-local transition evidence

Retain completed two-slot sentences in structured NP-VP form until the next
eligible state of the same tracked noun:

```text
E_t = (NP_t, VP_t, discourse_id, identity_binding,
       subjective_when_t, objective_event_support)
E_next.NP = next compatible state of that tracked individual
eligible_transition = (E_t.NP, E_t.VP, E_next.NP)
```

This bounded row-local cache transports evidence; it is not the verb
representation. Preserve sparse NP/VP row IDs, activations, discourse
identity, time, and role bindings. Reset at document boundaries, detach stored
states, compare affected noun states rather than whole sentences, and keep
ambiguous matches provisional.

Concept learning remains inside symbolic order. Compiled code may calculate
candidates and losses, then queue structural changes for a safe eager
boundary. Deferring mutation must not move meaning creation into grammar.

## 14. Preserved roadmap: shared sparse verb executor

Do not allocate a private dense invertible matrix per verb. Use a shared bank
of low-rank transformation experts with sparse gates and a control code from
the learned diachronic concept:

```text
z_after = z_before
        + sum(j in top_k(g_V)) g_V[j] U_j phi(R_j z_before)
        + B b_V
```

Begin with 32 rank-32 experts and four active experts per VP occurrence.
Condition gates on the VP concept and argument roles. The residual path
preserves unaffected structure; the selected sparse subspace carries change.

If quality saturates, ablate expert count, rank, and two-step composition.
Vocabulary growth should primarily add rows and sparse relations, not private
dense operators. Add verb-transition weight `0.25` and sparse gate penalty
`1e-4` as initial values; ramp transition weight over the first ten percent of
the verb-learning phase and report it separately from future-`what` loss.

## 15. Preserved roadmap: dimensions and grounding

PartSpace, WholeSpace, and ConceptualSpace widths remain independent. A
concept is defined by references and activations, not coordinatewise identity:

```text
{PS percept references, WS percept references}
    -> ConceptualSpace identity row and atom
```

Constituent richness grows sparse edges, not a required conceptual width.
After temporal learning works, ablate ConceptualSpace width 512 then 256 while
choosing WholeSpace independently. Select widths using reconstruction,
grammar, temporal quality, memory, and throughput.

Grounding repair remains deferred. Later work may replay high-trust perceptual
exemplars, measure PS/WS-to-CS stability, and repair unsupported concepts
without collapsing independent spaces or rewriting historical identities.

## 16. Acceptance gates for the preserved roadmap

### Ontology and concepts

- `NOTHING` and `EVERYTHING` retain their literal roles.
- Open, closed, and inconsistent conceptual openings remain distinguishable.
- Optional fillers may enter or leave without changing a tracked identity.
- Actual parthood composes only over intersecting event support.
- Subjective attention, conceptual identity, and data/event location cannot be
  substituted for one another.
- Symbolic order can learn both same-time nouns and ordered temporal concepts.
- Reversing before/after produces a distinct temporal relation.

### Verbs and memory

- Repeated transitions reuse and improve one learned verb concept.
- A retained VP trains only on an eligible later state of the same identity.
- Admitted VPs improve held-out transition error over identity copying.
- Grammar selects role-conditioned learned meanings rather than raw lexical
  rows.
- No verb owns a private dense invertible matrix.
- Model-produced memory remains distinct from supplied training targets.

### Quality and performance

- Report present, past, future, supervised, and thinking quality separately.
- Report grammar validity, transition quality, verb-sense selection, memory,
  and sentences/s.
- Preserve the matched B24 baseline and 15% immediate-iteration regression
  gate; establish other batch-size baselines separately.
- Attribute LTM-slot, temporal-query, concept-admission, transition-cache, and
  verb-expert overhead separately.
