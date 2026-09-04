# What, spacetime, and thinking

> **Status:** high-level target design, revised 2026-09-04. This design uses
> the existing `Data`, `Model`, and `Model.run()` framework. The
> [teaching-modes specification](specs/2026-07-27-teaching-modes-and-next-iteration.md)
> predates this revision and must be brought into alignment after this design
> is agreed.

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

The current reconstruction task is the first instance of this interface. It
has effectively asked `What(present)`: given the presented input, reproduce
what is happening now. Making that question explicit generalizes the same
training loop to:

- `What(past)`, which asks for data at an earlier time;
- `What(present)`, which includes the existing reconstruction task;
- `What(future)`, which asks for data at a later time; and
- supervised questions whose correct response is determined by the question
  rather than by copying any temporal input.

`Data.what()` and `Model.what()` therefore answer the same question with
different authority. `Data` can use its data coordinate to retrieve the
desired answer during learning or evaluation. `Model` must answer using its
own learned conceptual and memory state. The answer returned by `Data` is a
training target, not an additional model input.

Conceptually, the existing run path becomes:

```text
question and input = Data presentation
model_response     = Model.what(question)

if learning or evaluation:
    desired_response = Data.what(question)
    compare(model_response.what, desired_response.what)

if learning:
    update Model through the existing Model.run() training path

if inference:
    the Model response may become the presentation's Data output
```

The `what()` methods identify where the existing work is delegated. They do
not require a second lesson controller, scoring API, source oracle, spacetime
view hierarchy, or parallel generation API.

## 2. Questions, data coordinates, and time

The where and when of an answer are always part of the question. They are not
separate privileged arguments supplied beside the question, and the model is
not trained to reproduce them. They provide context for producing the `what`;
the cost function compares only the desired and produced `what`.

For the present design, `when` is the zero-based presentation index in
`TheData`. For text, this is the sentence index. It provides an ordered data
coordinate that `Data.what()` can use as an index into the dataset. A question
can identify the desired time directly or relative to the presented time:

```text
presentation 8 + What(present)     -> data at index 8
presentation 8 + What(past, -1)   -> data at index 7
presentation 8 + What(future, +1) -> data at index 9
```

The exact representation of those semantics may be grammatical, conceptual,
or both. The requirement is that they occur within the question context seen
by the model. There is no separate objective-address structure that silently
selects a different target.

`where` is omitted from the initial implementation. A later design may use it
to address a location within the datum presented at one trial, which for the
current text data is a sentence. The model may map data `where` and `when`
coordinates to its own internal `.where` and `.when`, but it is not required
to do so. Data coordinates and model coordinates need not share a
representation.

Each stable presentation index reserves both sides of an interaction:

```text
DataPresentation:
    when: zero-based presentation index
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
which the question increments `when` by one.

## 3. Queryable spacetime through `Data` and `Model`

Training across past, present, and future questions teaches a queryable
spacetime representation of `TheData`:

- reconstruction associates a present question with the present datum;
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

## 4. LTM records inputs and responses

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

## 5. Thinking is iterative `what()` evaluation

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

## 6. Context for the grammar chooser

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

## 7. Prediction, supervision, and completion

Prediction and verification remain part of the existing `Model.run()`
training and evaluation loop. The substantive change is that the question
determines the desired output:

```text
question asks What(present) -> desired output may reconstruct the input
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

This design deliberately leaves the existing batching, forward execution,
loss calculation, optimizer, grammar chooser, STM, and LTM machinery in
place. Implementation work should extend those components with `what()`
delegation, question-relative target selection, paired LTM input/output
representations, and iterative parity handling rather than building a second
Teacher/student architecture beside them.
