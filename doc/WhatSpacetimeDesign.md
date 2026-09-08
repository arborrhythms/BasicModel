# What, spacetime, and thinking

> **Status:** high-level target design, revised 2026-09-08. This revision
> integrates the implemented `what()` foundation with the unfinished answer-
> construction work. It separates faithful input reconstruction from rational
> output synthesis while retaining one `Data`, `Model`, and `Model.run()`
> framework. The companion
> [teaching-modes specification](specs/2026-07-27-teaching-modes-and-next-iteration.md)
> is aligned with this revision.

## 1. One question, different authority

The common question is **what?** The question may ask what is present, what
was present, what will be present, or for a response that is not a
reconstruction of the input at all. Its answer depends on which existing
object is answering:

- `Data.what(question)` retrieves the answer available from the dataset.
- `Model.what(question)` produces the model's answer from its conceptual
  representation and memory.
- `Model.run()` remains the framework that presents data, runs the model,
  evaluates the result, and optionally trains it.

These are logical interfaces that delegate to the infrastructure already
owned by `Data` and `Model`; they do not introduce a new execution framework.
In particular, there is no Teacher class and no separate Student class in the
target design. `TheData` is the existing `Data` instance, and all learned
behavior remains on the existing `Model`.

The current clean reconstruction task supplies the first data case for this
interface, but reconstruction and response construction are distinct model
products. Every presentation may train the model to reconstruct the perceived
input. `What(present)` additionally asks it to construct a response whose
desired content is the current datum. Those targets may coincide externally
while taking different paths through the model. Making the question explicit
generalizes the response side of the same training loop to:

- `What(past)`, which asks for data at an earlier time;
- `What(present)`, whose answer target may equal the input-reconstruction
  target while remaining a separately constructed response;
- `What(future)`, which asks for data at a later time; and
- supervised questions whose correct response is determined by the question
  rather than by copying any temporal input.

`Data.what()` and `Model.what()` therefore answer the same question with
different authority. `Data` can use its data coordinate to retrieve the
desired answer during learning or evaluation. `Model` must answer using its
own learned conceptual and memory state. The answer returned by `Data` is a
training target, not an additional model input.

Conceptually, the run path becomes:

```text
question and input   = Data presentation
understanding        = Model.forward(input)
input_reconstruction = Model.reconstruct(understanding)
model_response       = Model.output(understanding, question)

if learning or evaluation:
    desired_response = Data.what(question)
    compare(input_reconstruction, input)
    compare(model_response.what, desired_response.what)

if learning:
    update Model through the existing Model.run() training path

if inference:
    the Model response may become the presentation's Data output
```

`Model.what()` is the public question-answering delegation over `forward()`
plus `output()`; it is not a direct projection head. The methods identify
where existing work is delegated and do not require a second lesson
controller, scoring API, source oracle, spacetime view hierarchy, or second
model.

## 2. Questions, data coordinates, and time

The where and when of an answer are always part of the question. They are not
separate privileged arguments supplied beside the question, and the model is
not trained to reproduce them. They provide context for producing the `what`;
the cost function compares only the desired and produced `what`.

The question's coordinate is `where`: the zero-based presentation index in
`TheData` (for text, the sentence index). The dataset exists all at once, so a
presentation index is a *position* in the dataset, an absolute `.where`, not a
time. The model's `.when` is its own clock (`when_time`, advanced once per
processed batch) and is never carried by a question. `Data.what()` uses `where`
as an index into the dataset; a question can identify the desired position
directly or relative to the presented one (`past`/`future` are displacements
along the dataset):

```text
presentation 8 + What(present)     -> data at index 8
presentation 8 + What(past, -1)   -> data at index 7
presentation 8 + What(future, +1) -> data at index 9
```

The exact representation of those semantics may be grammatical, conceptual,
or both. The requirement is that they occur within the question context seen
by the model. There is no separate objective-address structure that silently
selects a different target.

The absolute position is given to the model. `WhatQuestion.context_values()`
carries `where` and `target_where` alongside the relation bits and offset, and
`Model._what_grammar_context` maps both positions into the model's own
`.where` coordinate system: normalized by the split extent and through a
4-dim `.where` ladder (`WhereEncoding`, period = number of presentations) whose
output feeds the zero-initialized, learned `what_projection` of the grammar
chooser. The model therefore learns the mapping between dataset positions and
its internal coordinates; `Data` never writes the model's `.where`/`.when`.
Positions name locations, never content, so this adds no target to the model
context.

An implicit lockstep between dataset order and the model clock is not a
substitute: batch lanes share one `when_time` tick, an epoch wrap re-presents
the same row much later, and exploration trials do not tick the clock, so
"one presentation back" in dataset coordinates is not "one tick back" in model
time.

A finer rung of the same `.where` (a location within the datum presented at
one trial, for text a span within the sentence) is deferred; it is not a
separate coordinate. A corpus byte address as a coarser rung, and a
document-boundary bit for `past` questions at a document start, are noted
follow-ons.

Each stable presentation index reserves both sides of an interaction:

```text
DataPresentation:
    where: zero-based presentation index (an absolute .where)
    input: question/presentation
    output: desired or generated response, possibly absent
```

In learning mode, a presentation has an input and a desired output. In
inference mode, it initially has only an input; `Model.what()` supplies the
output. Reserving both fields means that recording an inferred output does
not renumber later sentence coordinates. Generated outputs should retain
their model provenance so that writing an inference to `TheData` does not
silently turn it into supervised source data.

This also makes clear that temporal prediction is not restricted to
one-step-next prediction. A question may ask for any represented past,
present, or future time. Existing prediction models are the special case in
which the question increments `where` by one.

## 3. Queryable spacetime through `Data` and `Model`

Training across past, present, and future questions teaches a queryable
spacetime representation of `TheData`:

- input reconstruction teaches recovery of the presented datum regardless of
  which response the question requests;
- present answering associates a present question with the present datum;
- recall associates a past question with remembered data;
- prediction associates a future question with later data; and
- supervised pairs associate a question with an answer that may differ from
  every presented temporal datum.

The same `Model.run()` path performs evaluation and optional learning in each
case. The question selects the desired response. This matters because the
input alone does not imply that the correct response is its reconstruction.
For example, an input may ask for a missing past token, a future sentence, or
the answer to “what is your name?”. All are `what()` questions; the temporal
and grammatical content of the question determines what counts as accurate.

Automatic completion of missing past, present, and future content trains the
model's world representation. Supervised training uses an explicit desired
response. Inference without a desired response produces a model output that
may be returned to the caller or recorded as the output side of the
`TheData` presentation.

## 4. One understanding, two downward paths

Bottom-up processing produces one internal understanding:

```text
InputSpace -> PerceptualSpace -> ConceptualSpace -> SymbolicSpace
```

There is no separate surface trace. Perceptual detail persists only as the
activation of mereonymic parts and wholes. Conceptual and symbolic decoding
may attend to those activations, so a compact symbol can be augmented with
specific perceptual detail as it descends back through concepts and percepts.
The representation is collective: car, tire, and other overlapping concepts
can be active together. Choosing one mereological level for a symbolic
description does not erase the others or require the chosen symbol to carry
all their detail. The full active conceptual field, not each individual
concept or P0 summary, should reconstitute the full perceptual field.

After understanding, reconstruction and answer production diverge.

### 4.1 Input reconstruction

Input reconstruction starts from the input's understood symbolic/conceptual
state and follows the inverse of the analysis path back to `InputSpace`:

```text
input symbols/concepts
    -> inverse symbolic/conceptual operations
    -> inverse perceptual operations
    -> InputSpace
```

When a configured operation is invertible, reconstruction uses the inverse
matrix belonging to the forward operation. It may use the exact forward-local
carriers needed to invert routing and bindings, but those carriers may not
cache the pre-analysis surface payload. Their content-bearing state is limited
to the processed activation of mereonymic parts and wholes. Its purpose is
fidelity to the presented input. In code, `reverse()` is reserved for this
reconstruction meaning; it must not also stand ambiguously for answer
generation.

### 4.2 Rational output construction

For a question, decoding the understood input directly would reconstruct the
question. Rational thought begins when serial symbolic decoding treats the
interrogative structure as an operation with an unresolved position. It either
resolves that position immediately or extends the derivation through LTM:

```text
QUERY(name-of(addressee), ?)
    -> symbolic evaluation / memory / thinking
    -> BIND(name-of(addressee), Alec)
    -> answer symbol
```

The question and `Alec` can therefore occupy or reference the same conceptual
location without being the same decode. The understood question activates the
name relation and an open role; the answer is the value bound at that role.
Input reconstruction decodes the still-interrogative structure. Output
construction first resolves the binding and decodes the resulting value.

A simple true/false question may resolve in one symbolic operation. A harder
question may open subquestions and require several iterations before an answer
symbol exists. In either case, the answer symbol, not the input question
symbol, is then synthesized downward:

```text
answer symbol
    -> ConceptualSpace synthesis
    -> PerceptualSpace synthesis with perceptual context
    -> OutputSpace
```

At a high level, `Model.output()` delegates as follows:

```text
derivation = SymbolicSpace.resolve(
    symbolic_state, question, grammar_context, memory)
answer_concepts = ConceptualSpace.synthesize(
    derivation.answer_symbol, derivation.bindings,
    context=conceptual_state)
answer_percepts = PerceptualSpace.synthesize(
    answer_concepts,
    context=perceptual_context,
    selections=derivation.synthesis_references)
response = OutputSpace.forward(answer_percepts)
```

Until `derivation.answer_symbol` is resolved, `resolve()` may extend the LTM
thinking stack rather than call synthesis. Its replayable result also carries
the grammatical derivation, answer-sentence location, constructed prefix, and
named conceptual/perceptual bindings used by the next output step.

`output()` is the model-level orchestrator for this path. At the space level,
`forward()` means bottom-up analysis, `reverse()` means input-associated
inverse reconstruction, and `synthesize()` means top-down realization of a
generated state. A synthesis operation may share an invertible layer's
parameters and use its inverse direction, but it receives the generated answer
carrier and may not silently substitute a cached input-reconstruction carrier.

The active perceptual field is available to output synthesis as context. It
can prime a name, object, word, grammatical form, or other activated part or
whole, but it is not itself the output seed. Perceptual contributions must be
selected through named, replayable grammatical attention or binding
operations. An unrestricted residual route from input percepts to output would
be a copy shortcut around rational symbolic decoding and is forbidden.
Conceptual synthesis also has access to the parallel, overlapping conceptual
field. Imagination may complete unspecified perceptual detail from that field
and learned associations, with controlled sampling where appropriate. A
symbol supplies an intention or salient description; it need not encode a
complete scene as a serial stream. Trace the selected conceptual references,
constraints, and sampling state, not every generated percept as a symbol.

The two downward paths need separate per-call state. Output synthesis must not
overwrite the forward carriers needed by reconstruction, and reconstruction
must not replay an answer's generated carriers. They may share parameters and
read the same live activations without sharing mutable decode state.

`OutputSpace` is consequently a final modality adapter. It may turn constructed
answer percepts into text, a truth value, a scalar, or an action, but it does
not decide the answer by projecting directly from the input symbol.

### 4.3 Invertibility contract and current gap

Perceptual and conceptual analysis should be primarily invertible before
quantization, collectively over their active fields. Concept-to-symbol
quantization applies to each concept independently. Finite codebooks may lose
detail; the conceptual inverse is required in principle as codebook resolution
increases, not by retaining a residual beside every snapped symbol. No
per-symbol quantization-residual or copied-surface channel is required.

`P0 = 1 - product(1 - part_i)` is the current parameter-free membership union
over a word's parts, before the learned Sigma ladder. It is not Pi's learned
fold, nor an individually invertible encoding of all those parts. This is
acceptable only if the collective processed field supplies the remaining
information. Use `W0..W3` for whole-source folds alongside `P0..P3`.

The live bridge now replaces the RMS/same-row/mean sequence with addressed
native-code evidence and one learned affine Sigma/tanh activation per existing
word-concept candidate/location. It decodes that concept once. Each concept
has up to eight stable typed P/W references, not one duplicate identity per
fold. Full processed native fold stacks remain beside the concept event and
survive symbolic selection; the inverse binder returns terminal P/W states
without averaging. WHERE/WHEN and prior symbolic content are kept separate
from the evidence readout. This preserves distinctions the previous RMS
erased, but retained native state is not proof of collective concept-only
inversion. Overlapping concept-field formation and its decoder remain work.

Several identified parts and wholes should knit into each first-level
concept at a coherent `.where`; this is not one concept per source. Eight
is the bounded parallel reference budget, not a requirement to fill eight
slots. A familiar serial word may need only its whole-space word property
and a particular part-space word code. Extra serial operands still serve
new unions of parts and intersections of wholes. Parallel processing should
retain richer overlapping content for conceptually driven reconstruction.

The reusable `ConceptsFromPercepts(Layer)` readout has two
implementations: a weighted tanh with learned reference/context gates, and
a simpler affine Sigma followed by tanh with optional per-connection L1.
Both preserve code-specific evidence and independent concept activations;
missing evidence is not observed absence. L1 can remove one concept's
connection without suppressing a percept for other concepts. These are
activation readouts, not inverse carriers. The live model uses Sigma with
weak L1 (`conceptReadoutL1=0.01`) in an indexed row-local form: no gates or
learned selector. A diagonal-metric proximal Adam update can make admitted
connections exactly zero while leaving biases and unused slots unpenalized.
This explicitly trades some reconstruction accuracy for sparse definitions;
it does not guarantee independent concepts or a collectively invertible basis.
Reference admission uses existing concept relationships with stable slots;
the full field-level migration is not complete. The controlled comparison,
live cutover, and production acceptance boundary are in
[specification section 5.6](specs/2026-07-27-teaching-modes-and-next-iteration.md#56-reusable-conceptsfrompercepts-readout-and-sigmal1-comparison).
Its strong-L1 result demonstrates why reducing input count cannot take
precedence over preserving useful reconstruction detail.

The local numerical fixes preserve nonzero effective LDU diagonals even at
exact zero and hold butterfly pairs touching padding at identity after
training. Their reasons are documented beside the implementation. Existing
ergodic sampling before `forward()` and after `reverse()` is compatible with
a paired round trip; both directions must use the same intervening sample.
The contract is an inverse on the image of valid perceptual analysis, not a
claim that every imagined concept lies on that image.

Hard codebook lookup and a straight-through estimator (STE) are different
choices: lookup determines the forward value; the STE supplies encoder credit
through locally constant nearest-code selection. The full hard lookup already
occurs. Retain an estimator where removing it severs upstream reconstruction
or answer learning; direct indexed row reads use genuine gather gradients.
The separate percept-store bounded-read STE allows an out-of-cube master
coordinate to receive a corrective gradient through clamping. It is not an
approximate lookup or proof of invertibility. A replacement must demonstrate
both exact selected values and the intended learning path.

## 5. Joint learning pressure

Reconstruction and output construction have separate costs but intentionally
meet in the internal understanding:

```text
L_total = lambda_input * L_input_reconstruction
        + lambda_answer * L_answer_construction
        + L_grammar + L_memory + L_commitment + other existing terms
```

Both branches are built at one parameter version and evaluated before one
optimizer step. The scalar total remains useful for reporting; its ordinary
gradient sum is not the update policy on protected perceptual/conceptual
parameters. Separate autograd traversals identify each branch's contribution
before reconstruction-priority projection. The framework's gradient
`backward()` is distinct from the model's representational `reverse()`.

More precisely, let the bottom-up understanding be `h = F_theta(x)`, the
input reconstruction be `x_hat = R(h, input_carriers)`, and the constructed
answer be `y_hat = G(T(h, question, memory), perceptual_context(h))`. The
unmodified gradient arriving at the shared understanding during autograd is:

```text
dL_total/dh = lambda_input * dL_input/dh
            + lambda_answer * dL_answer/dh
            + auxiliary contributions
```

Thus the reconstruction branch is independent in purpose, target, returned
value, and decode state, but it is not optimization-independent from answer
construction. Both intentionally teach the same understanding. `forward()`
itself should return `h`, not own either target or hide either cost; the
existing `Model.run()` boundary constructs and names the two losses.

`L_input_reconstruction` flows through `InputSpace`, the inverse perceptual and
conceptual operations, and the shared understanding that seeded the reverse.
It trains the model to preserve and organize enough mereonymic detail to
reconstruct what it perceived.

`L_answer_construction` flows from `OutputSpace` through perceptual synthesis,
conceptual synthesis, the serial symbolic derivation (including thinking), and
then into the shared understanding produced by `forward()`. Useful
conceptualization is shaped by both input fidelity and correct response, with
reconstruction taking priority. Inspect the full answer derivative before
projection; do not detach the answer at the symbolic branch point.

Contextual gradient paths remain subject to the mind boundary. Output loss may
train the explicit grammatical choices, attention, bindings, and upstream
activations that supplied perceptual context. It may not reach the desired
answer before loss, copy the target into memory, or use an untraced direct
percept-to-output projection.

Parameters may be shared without sharing state. If `W` is structurally
invertible, both costs may update `W`; reconstruction continues to use the
mathematical inverse of the updated `W`. Answer-specific grammar, attention,
and synthesis gates may have their own parameters, while the perceptual and
conceptual coordinate system remains common.

When the same `W` participates in analysis and again as `W^-1` during
synthesis, answer loss produces both an indirect gradient through
`F_theta(x)` and a direct gradient through the inverse use. Autograd must
retain and sum both before priority projection; this is one path with a tied parameter, not two answer
losses. Because derivatives through an inverse grow unstable near a singular
matrix, the implementation must monitor conditioning and preserve its
invertible parameterization rather than stopping the synthesis gradient.

An exact, fully observed round trip `W^-1 W x = x` has zero reconstruction
error and supplies no useful learning pressure. Masking or degradation alone
does not change that fact if a masked carrier merely makes the same cancelling
round trip. Useful reconstruction pressure requires contextual completion,
compression, a bottleneck, or another non-cancelling operation between
analysis and synthesis, with removed information absent from every carrier.
The implementation must verify a gradient at the shared understanding rather
than infer one from reconstruction error. This prevents algebraic
invertibility from being mistaken for learned understanding.

For each protected perceptual/conceptual parameter tensor, let `r` and `o` be
the reconstruction and output gradients **after** their actual loss weights
and truth modulation. The accepted policy is asymmetric:

```text
if dot(r, o) < 0 and ||r|| > 0:
    q = o - dot(r, o) / dot(r, r) * r
else:
    q = o

q = q * min(1, rho * ||r|| / ||q||)  # zero q stays zero
if r is missing or zero: q = 0
shared_gradient = r + q + separately named auxiliary gradients
0 <= rho < 1                       # initial rho = 0.5
```

Remove only the conflicting projection, preserving useful aligned output
credit. This gives `dot(r, q) >= 0` and `||q|| <= rho * ||r||` within numerical
precision. Reconstruction itself is not projected or downscaled. Independent
symbolic/output heads receive ordinary output credit. Parameters tied into
both analysis and synthesis receive protection on their complete derivative.
Sparse codebooks must stay sparse over touched rows.

This guarantees non-opposing output credit at the gradient level. It does
not guarantee a decrease in reconstruction loss after a finite Adam/momentum
step, nor protect against arbitrary auxiliary objectives; monitor actual
reconstruction quality as well. Report branch norms, alignment, and the
removed/capped contribution when joint output training is active.

The zero-reference case is consequential: an exact inverse identity, detached
reconstruction, or a disconnected encoder gives output no budget on protected
parameters under this policy. Independent output heads can still learn. The
current BasicModel uses `detachedReverse=true` and a non-grad, context-rotated
concept dictionary; enabling projection does not create a reconstruction
gradient through either boundary. Completing the field/answer architecture
must resolve those ownership and credit paths explicitly. No residual channel
is introduced merely to manufacture a reconstruction error.

Serial grammar introduces a separate credit boundary. Continuous answer loss
can train the chosen operations, their operands, synthesis, and the shared
understanding. A hard grammatical choice has no ordinary derivative with
respect to the unchosen policy logits; the existing straight-through,
two-pass chooser, or policy-loss mechanism must assign that credit from the
replayable trace. The plan must not claim a differentiable path through a hard
choice that autograd does not provide.

This resembles human learning. Perception is organized both to recover what
was experienced and to support useful judgment and action. Repeating a heard
question and answering it use the same active perceptual and conceptual field,
but repetition descends from the input understanding while answering first
resolves a new symbolic intention.

## 6. LTM records inputs and responses

LTM records the model's conceptual representation of both sides of an
interaction:

```text
LTMSlot:
    input:  optional conceptual representation
    output: optional conceptual representation
```

The input representation is the existing one-, two-, or three-slot
representation of the input sentence in conceptual space. The output is the
model's conceptual representation of its own response. It is not necessarily
a reconstruction of the input: for `What(past)`, `What(future)`, and ordinary
question answering, it will normally differ.

During supervised learning, the desired `Data` output supplies the loss, but
the response stored in LTM is the response actually made by the model. This
keeps memory a record of the model's experience and reasoning rather than a
hidden route by which the desired answer is copied into its context.

The two fields are independently optional within each LTM slot. That property
lets the existing sequential LTM act as a thinking stack:

- `(input, —)` pushes an unanswered question;
- `(input, output)` is a complete stimulus/response slot and leaves the
  stack depth unchanged; and
- `(—, output)` answers and pops the most recent unanswered question.

A slot with neither input nor output has no meaning. A question without an
answer in one slot begins the stack of internal dialogue. A later answer
without a question closes the most recent open question. Stack state is thus
the imbalance represented by the sequence of LTM slots; it does not require a
new recursive frame tree, a separate stack object, or an in-place update to
the opening slot. **Parity** means that there are no unmatched input-only
slots.

## 7. Thinking is iterative `what()` evaluation

All model questions are queries into conceptual space. When the truth or
illumination of the relevant conceptual spaces is insufficient, the model
may think by asking a subquestion through its own `what()` interface.
Internal subquestions do not use the sentence index known by `TheData`; the
model cannot supply a data coordinate it does not possess. They are formed
from the model's current conceptual question and memory.

Thinking is iterative program evaluation, not recursive model evaluation.
Each iteration appends an LTM slot. The next iteration receives the original
question together with the enlarged LTM context, so a completed subquestion
can transform how the still-open question is evaluated.

For the supervised pair:

```text
Q: what is your name?
A: Alec
```

one possible LTM sequence is:

| Iteration | LTM slot | Derived open-question stack |
|---|---|---|
| Understand the root but defer its answer | `(what is your name?, —)` | `[what is your name?]` |
| Ask and answer a subquestion | `(who is asking?, OpenAI)` | `[what is your name?]` |
| Answer the pending root using the new context | `(—, Alec)` | `[]` |

The second slot is balanced by itself, but its result is now part of LTM and
therefore part of the context used to answer the root. The output-only third
slot balances the earlier input-only slot. Once parity is restored, the
answer requested by `TheData` is available and the ordinary supervised loss
can be evaluated.

Thinking must conclude. A closure pressure increases with each unbalanced
iteration. It is supplied to the existing grammar chooser as part of its
context: as pressure rises, opening another input-only slot becomes less
favored and producing an output for the most recent unanswered input becomes
more favored. A finite iteration limit remains a safety boundary. At that
limit, the model must emit its best-effort answer for the most recent
unanswered input; `unknown`, `unresolved`, or a failure status cannot be used
as an escape from answering. The answer may carry low confidence, but it is
still an output-only slot and therefore restores one level of parity. If more
than one input remains open, forced best-effort answers continue from the top
of the stack until parity is restored. The exact monotonic pressure schedule
remains a specification decision.

## 8. Context for the grammar chooser

The grammar chooser already operates over a richer context than a new
address-specific list would describe. Its context includes symbolic
activation, STM, LTM, percepts, and grammatical operations, among the other
state already supplied by the model.

This design does not replace or duplicate that context. It requires the
chooser to be able to distinguish only the new, relevant state:

- the temporal meaning contained in the active `what()` question;
- the input and output sides of LTM slots;
- whether unmatched input-only slots remain; and
- the current closure pressure while thinking.

The chooser may use that context to answer, to leave an input unanswered
while posing a subquestion, or to emit an output for the latest pending input.
Those are grammatical/model choices within the existing run path, not calls
to a separate query planner.

The chooser also controls output realization after the answer is resolved. Its
context includes the answer symbol, current derivation, output sentence
location, already constructed output, and the perceptual/conceptual references
available for synthesis. These values let it choose the next grammatical
derivation and where that derivation contributes to the answer surface.

## 9. Prediction, supervision, and completion

Prediction and verification remain part of the existing `Model.run()`
training and evaluation loop. The substantive change is that the question
determines the desired output:

```text
question asks What(present) -> desired output may equal the input
question asks What(past)    -> desired output comes from an earlier index
question asks What(future)  -> desired output comes from a later index
supervised question         -> desired output is the supplied answer
inference question          -> no Data output; Model supplies it
```

The model always receives the temporal or other intent as part of the
question. It is rewarded for accurately completing the missing `what`, not
for reproducing the coordinates. If the question cannot yet be answered
directly, the same supervised presentation may include the iterative thinking
process described above. Training occurs after thinking returns the LTM stack
to parity and produces the requested response.

This design keeps the existing batching, optimizer, grammar chooser, STM, and
LTM owners, but it does not preserve the direct SymbolicSpace-to-OutputSpace
head or the ambiguous use of `reverse()` for both reconstruction and
generation. Implementation work must complete question-relative target loss,
split reconstruction state from output-synthesis state, route `Model.what()`
through rational symbolic decoding, and train both branches jointly rather
than building a second Teacher/student architecture beside them.
