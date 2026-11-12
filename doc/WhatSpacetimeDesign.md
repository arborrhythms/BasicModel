# What, spacetime, and thinking

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; a stream is the batch row that wrote a row, and a document is the cover row that starts it in that stream (reused at every later presentation; a row belongs to the document of the latest cover before it), so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded (2026-10-24).
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23; recorded 2026-10-30):** an LTM1 row is two tagged references (each to an LTM0 occurrence, a symbol or another LTM1 row, its tag saying which) with its kind read from the referents, not three. The unused `LTM0` columns are removed and the row has this layout (Alec, 2026-11-01).


> **Status:** the question-answering design, first written 2026-09-08 and
> kept to the code. It separates faithful input reconstruction from rational
> output construction while retaining one `Data`, `Model`, and `Model.run()`
> framework.

## 1. One question, different authority

The common question is **what?** The question may ask what is present, what
was present, what will be present, or for a response that is not a
reconstruction of the input at all. Its answer depends on which existing
object is answering:

- `Data.what(question)` retrieves the answer available from the dataset.
- `Model.ask(question)` produces the model's answer from its conceptual
  representation and memory.
- `Model.run()` remains the framework that presents data, runs the model,
  evaluates the result, and optionally trains it.

These are logical interfaces that delegate to the infrastructure already
owned by `Data` and `Model`; they do not introduce a new execution framework.
There is no separate Student class: `TheData` is the existing `Data` instance,
and all learned behavior remains on the existing `Model`. The model owns its
data, its error registry (`Model.errors`) and its loss: errors with targets
are registered there, penalties and reported metrics through
`Model.record_loss`, and a batch opens without a Teacher (`_open_batch` lets an
attached Teacher adapter open its lesson only when one is present).
`bin/Teacher.py` remains an optional adapter for lessons and clean targets; it
never reads or writes the model's `.where`/`.when` bands.

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

`Data.what()` and `Model.ask()` therefore answer the same question with
different authority. `Data` can use its data coordinate to retrieve the
desired answer during learning or evaluation. `Model` must answer using its
own learned conceptual and memory state. The answer returned by `Data` is a
training target, not an additional model input.

Conceptually, the run path is:

```text
question and input   = Data presentation
understanding        = Model.forward(input)
input_reconstruction = Model.reverseReconstruct(understanding)
derivation           = Model.resolveAnswer(understanding, question)
model_response       = Model.reverseOutput(understanding, derivation)

if learning or evaluation:
    desired_response = Data.what(question)
    compare(input_reconstruction, input)
    compare(model_response.what, desired_response.what)

if learning:
    update Model through the existing Model.run() training path

if inference:
    the Model response may become the presentation's Data output
```

`Model.ask()` is the public question-answering delegation over `forward()`,
`resolveAnswer()` and `reverseOutput()` (`WholePaths.output_only`); it is not a
direct projection head. The methods identify
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
time. The model's `.when` is its own clock (`when_time`, `Model.present()`),
advanced once per present sentence, per deposited thought and per produced
output, and is never carried by a question. `Data.what()` uses `where`
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

The same target-free context vector reaches the answer through the question
conditioner, `Model._question_conditioner`: one zero-initialized, bias-free
linear map per answer width, kept in the state dict as `question_conditioners`
so that switching widths discards nothing. It maps the context onto the first
occupied role of the conceptual answer, the root of the generate walk, as a
cube translation that leaves the position and time band unchanged. Because it
starts at zero the answer path starts independent of the question, and the
answer cost trains it, so temporal question content is used rather than
merely present: `test_what_training.py::test_temporal_question_content_is_causally_used`
changes only the question's temporal content, with the input fixed, and
requires the answer to change.

An implicit lockstep between dataset order and the model clock is not a
substitute: batch lanes share one tick per sentence slot, an epoch wrap
re-presents the same row much later, thoughts tick the clock too, and the
paths of one sentence share its tick, so "one presentation back" in dataset
coordinates is not "one tick back" in model time.

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
inference mode, it initially has only an input; `Model.ask()` supplies the
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
the answer to “what is your name?”. All are *what?* questions; the temporal
and grammatical content of the question determines what counts as accurate.

Automatic completion of missing past, present, and future content trains the
model's world representation. Supervised training uses an explicit desired
response. Inference without a desired response produces a model output that
may be returned to the caller or recorded as the output side of the
`TheData` presentation.

## 4. One understanding, two downward paths

> **Naming (2026-09-09; amended 2026-11-10):** the two downward paths are
> `Model.reverseReconstruct()` and `Model.reverseOutput()`, one seeded by the
> analysed input and ending at `InputSpace`, the other seeded by a resolved
> answer and ending at `OutputSpace`. Both end in one order-zero decoder,
> `BasicModel.reverse(x, field=...)`: reconstruction reverses the input's own
> order-zero field, and output reverses the order-zero field that the generate
> walk and its occurrence writer produce from the thought's conclusion. The
> decoder is shared by construction, so every reconstruction step trains the
> decoder that output uses; what stays separate is the per-call state.
>
> **The answer path's own weights:** the generate chooser, the question
> conditioner (§2) and, for a tensor output, `OutputSpace`'s `percept_adapter`.
> The per-Space synthesis layers are retired; a checkpoint's `synthesis_layer`
> keys are dropped with a warning. On the parameters output shares with
> reconstruction, its gradient is projected against reconstruction's (§5).

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

There is no known correct parse, only the parse the forward identified:
reconstruction follows that identified compose derivation, and no input parse
is ever imitated as output supervision. Output builds its answer, with a
possibly entirely different surface, by its own derivation through the
`<generate>` rules, with no teacher forcing in training or evaluation. Output
is trained only by supplied answers (and, on data that supervises output, by
silence where none was asked); input reconstruction targets and compose
choices never substitute for answer supervision, and a batch with no
supervised output makes no output update.

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
fidelity to the presented input. As built there are two input reconstructions
through the one decoder (`SentenceField.reconstruct`): the input
reconstruction reverses the open attention pass's field with no grammar, and
the order-zero reconstruction runs the generate walk on the sentence's
untransformed end slots first
([Training](Training.md#within-sentence-reconstruction-objective)).
`reverseReconstruct()` names this meaning; the decoder it ends in, `reverse()`,
is the one output uses too, run on a different field with separate per-call
state.

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
question may open subquestions and take several thought steps before an
answer exists. In either case the answer, not the input question, is then
realized downward:

```text
answer (the thought's conclusion)
    -> the generate walk and its occurrence writer (an order-zero field)
    -> the one decoder, reverse()
    -> OutputSpace
```

At a high level, the answer is resolved, then realized:

```text
derivation = Model.resolveAnswer(understanding, question)
    # the row's current or recalled completed field (section 9);
    # the thought controller transforms its ideas
response   = Model.reverseOutput(understanding, derivation)
    # condition the root on the question once, then the generate walk,
    # its occurrence writer, the shared reverse chain and OutputSpace
```

The derivation is prepared once and can be realized again without executing
another query. Its replayable result carries the grammatical derivation, the
answer's meanings and the sources of its rows.

`reverseOutput()` is the model-level orchestrator for this path. It may share
the decoder's parameters, but it receives the generated answer carrier and may
not silently substitute a cached input-reconstruction carrier.

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
read the same live activations without sharing mutable decode state. The
separation is structural: `SubSpace.carrier_like()` builds a fresh carrier
(`carrier_pure = True`) that shares only parameter-bearing bases with the live
subspace; every reverse write on a pure carrier (event, activation, recovered
input, concepts, basis) goes to that carrier, the merge glue never reads
reconstruction carriers for it, and the answer path runs under
`_synthesis_guard`, which hides the reconstruction carriers and restores the
live state afterwards.

`OutputSpace` is consequently a final modality adapter. It may turn constructed
answer percepts into text, a truth value, a scalar, or an action, but it does
not decide the answer by projecting directly from the input symbol. A numeric
output (XOR) is the one exception in form: it reads the path's composed state
through a single affine head, a modality with no grammar.

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
the full field-level migration is not complete. Strong L1 deletes useful
references, especially where a concept needs many, so sparsity is chosen
against a reconstruction tolerance: reducing input count cannot take
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
before the projection that gives reconstruction priority: in each optimizer
unit the answer gradient loses only its component opposing reconstruction's
(`GradientProjection.backward`; with `separateChooser=true` the owners are
separated instead). The framework's gradient
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

`L_answer_construction` flows from `OutputSpace` through the decoder, the
generate walk, the serial symbolic derivation (including thinking), and then
into the shared understanding produced by `forward()`. Useful
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
`reverseReconstruct()` checks it at run time: an unmasked reconstruction whose
cost is exactly zero sets `_reconstruction_identity_flag` and warns once that
the channel carries no learning signal, so a perfect round trip is never
reported as reconstruction learning. In training, adaptive omission supplies
the non-cancelling operation: part of each sentence is hidden out of band and
reconstruction must restore it.

A desired answer text is encoded as a scoring target by the real lexical stem
(`Model._embed_answer_texts`), not by a separate encoder, and the encoding is
read-only: it runs under `_synthesis_guard` with `_online_learning_frozen` set
on every Space, so no word is promoted into the percept store, no live carrier
or stream changes, and the stem's stash is restored. The detached result has
the shape of a presented input event and is never a model input.

*A superseded passage beginning "For each protected perceptual/conceptual parameter tensor, l" moved to [DevelopmentHistory](DevelopmentHistory.md) (2026-10-30).*

Serial grammar introduces a separate credit boundary. Continuous answer loss
can train the chosen operations, their operands, synthesis, and the shared
understanding. A hard grammatical choice has no ordinary derivative with
respect to the unchosen policy logits; the whole-path comparison assigns that
credit from the replayable trace, as a score function on the complete path's
change in $R + A$ when a departure is kept over the greedy path
([Training](Training.md#what-questions-and-the-two-primary-costs)). The design
must not claim a differentiable path through a hard choice that autograd does
not provide.

This resembles human learning. Perception is organized both to recover what
was experienced and to support useful judgment and action. Repeating a heard
question and answering it use the same active perceptual and conceptual field,
but repetition descends from the input understanding while answering first
resolves a new symbolic intention.

## 6. The interaction record

The model keeps a chronological record of its question-and-answer
interactions, `WhatInteractionMemory`, separate from the occurrence stores
`LTM0` and `LTM1`. Each `LTMSlot` has two independently optional sides:

```text
LTMSlot:
    input:  optional conceptual representation of the question
    output: optional conceptual representation of the model's response
```

A slot with neither side is invalid. The output is the model's own response,
not necessarily a reconstruction of the input: for `What(past)`,
`What(future)` and ordinary question answering it will normally differ.
During supervised learning the desired `Data` output supplies the loss, but
the response recorded is the response the model actually made, so the record
stays a record of the model's experience rather than a hidden route by which
the desired answer reaches its context. Values are detached when appended. An
input-only slot opens a question, an output-only slot closes the newest open
one, and a complete slot leaves the open depth unchanged; **parity**, no
unmatched input-only slot, is checked at each append.

## 7. Thinking

All model questions are queries into conceptual space. Thinking does not
iterate `ask()` over the interaction record: `Model.think()` understands once
and resolves through the ordinary thought controller, the thought grammar and
its chooser, under one shared work budget ([Reasoning](Reasoning.md)), and the
ordinary thought history keeps its own records with its own transitions.
Internal questions do not use the sentence index known by `TheData`; the model
cannot supply a data coordinate it does not possess, so they are formed from
its current conceptual state and memory. Every completed field offers the
controller an episode, and not thinking is the chooser's free first choice.
An answer comes only from a concluded, accepted thought; with no conclusion
the output role is empty and the model is silent
([Training](Training.md#what-questions-and-the-two-primary-costs)).

*Moved to [DevelopmentHistory](DevelopmentHistory.md#8-context-for-the-grammar-chooser) (2026-10-30): "8. Context for the grammar chooser".*
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
for reproducing the coordinates. If the question cannot be answered directly,
the same supervised presentation includes thinking: each path of the sentence
thinks before its output is generated, and training follows the comparison of
complete paths.

Resolution is per batch row (`Model._resolve_answer`): each row resolves its
own question's relation and offset and records its source (identity, recall,
prediction, cold memory). A present, supervised or inference question
resolves to the row's current completed field. `What(past, -k)` recalls the
row's own k-th most recent completed field from a bounded per-row history
(`Model._recall_sentence_history`, at most 64 detached fields), appended when a
sentence is admitted and cleared by that row's hard end-of-document reset
(`dispatch_per_row_reset`). A row whose history is shorter than k is cold and
stays unresolved. `What(future, +k)` stays unresolved until a conceptual
predictor exists. An unresolved row leaves the other rows of the batch
resolved. Training scores only supplied answers; automatic temporal targets
are evaluation metrics.

This design keeps the existing batching, optimizer, grammar chooser, STM, and
LTM owners and builds both branches in one model, with no second
Teacher/student architecture beside it: question-relative target loss,
separate reconstruction and output state, `Model.ask()` through the thought
controller and the generate walk, and joint training of the two branches
under the projection of §5.
