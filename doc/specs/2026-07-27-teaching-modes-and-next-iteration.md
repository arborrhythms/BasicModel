# What, spacetime learning, and concept formation: unified specification

> **Status:** implementation specification and maintainer handoff, originally
> 2026-07-27 and revised 2026-09-04. This revision replaces the proposed
> Teacher/student architecture with the existing `Data`, `Model`, and
> `Model.run()` framework.
>
> **Immediate scope:** add `what()` delegation to `Data` and `Model`; make
> present reconstruction an explicit `What(present)` task; extend the same
> mechanism to past, future, and supervised answers; store both input and
> model response in LTM; and support bounded iterative thinking through
> balanced LTM slots.

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

Present-input reconstruction is already a restricted form of the target
behavior. It effectively asks `What(present)` and trains the response to match
the current input. The next iteration must expose that question explicitly
rather than constructing a second controller around the same loop.

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
capabilities remain in the preserved roadmap in sections 12-15.

## 2. Core contracts

Add the following methods to the existing classes:

```text
Data.what(question)  -> desired What or unavailable
Model.what(question) -> produced What
```

Both methods delegate to infrastructure already owned by their class:

- `Data.what()` resolves the question against `TheData` during learning or
  evaluation.
- `Model.what()` evaluates the question through the existing perceptual,
  conceptual, grammatical, STM, and LTM paths.
- `Model.run()` remains responsible for presentation, execution, loss, and
  optional optimization.

The two `what()` methods answer the same question with different information.
The `Data` result is the desired response. The `Model` result is the model's
best response. The `Data` result must not be inserted into the model context
before the response is produced.

The logical run path is:

```text
question, input = Data presentation
actual          = Model.what(question)

if learning or evaluation:
    desired = Data.what(question)
    loss    = compare(actual.what, desired.what)

if learning:
    update Model through the existing training path

if inference:
    attach actual as the presentation output when the caller requests it
```

There is no additional scoring service. Existing loss bookkeeping may remain
where it is while the old class boundary is removed.

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

### 3.2 Initial `when`

For this iteration, `when` is the zero-based presentation index in
`TheData`. For text data, it is the sentence index and encodes presentation
order. `Data.what()` must resolve at least:

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

### 3.3 Deferred `where`

`where` is omitted from the immediate implementation. A later iteration may
use it to address part of the datum presented in one trial—for text, a
location within the sentence.

Data coordinates and the model's subjective `.where` / `.when` encodings are
not required to be identical. The model may learn a mapping between them.
`Data` must not overwrite or supervise the model's internal coordinates
directly.

### 3.4 Presentations reserve input and output

Each presentation index has an input side and an output side:

```text
DataPresentation:
    when:  zero-based presentation index
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
| `What(present)` | Current datum; this includes existing reconstruction |
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

## 5. LTM interaction slots

### 5.1 Store the model's input and response

Refine each LTM interaction slot to hold independently optional conceptual
representations of input and output:

```text
LTMSlot:
    input:  optional one-, two-, or three-slot input representation
    output: optional model-response representation
```

The output is the response actually produced by `Model`, not the desired
`Data` target. Supervised targets train the response through loss; they are
not copied into LTM as if the model had produced them.

The output may differ from the input. This is required for past/future
queries and ordinary question answering.

### 5.2 Slot parity supplies the thinking stack

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

## 6. Iterative thinking

### 6.1 When thinking starts

All questions are queries into conceptual space. If the truth or illumination
of the relevant spaces is insufficient, `Model` may defer an answer and pose
a subquestion. That choice produces an input-only LTM slot and begins an
internal dialogue.

An internal subquestion uses the model's conceptual state. It cannot supply
the sentence index used internally by `Data`, because that coordinate is not
available to the model's self-query.

### 6.2 Evaluation loop

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

### 6.3 Closure pressure and hard limit

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

## 7. Grammar and the mind boundary

The existing grammar chooser remains the control mechanism. Its established
context already includes symbolic activation, STM, LTM, percepts, and
grammatical operations. Do not replace it with a new address-specific chooser
or an imprecise duplicate context object.

Add only the state required by this iteration:

- temporal meaning carried by the active question;
- the input/output masks and representations of relevant LTM slots;
- current parity or open-stack depth; and
- closure pressure.

The chooser must be able to select among answering the current input, leaving
it open while posing a subquestion, answering a subquestion in the same slot,
and emitting an output-only answer for the most recent open input.

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

## 8. Training and loss

### 8.1 Shared training path

All question families use the existing training loop. The initial work should
adapt batch preparation and target selection, then reuse forward execution,
loss accumulation, backpropagation, optimizer stepping, and existing
two-pass chooser exploration.

The desired answer must be selected before or independently of model
execution but remain unavailable to the model until loss calculation. A
future target or supervised answer must not enter percepts, STM, LTM, grammar
context, runtime caches, or generated output before the response is fixed.

### 8.2 Loss target

The primary loss compares only `actual.what` with `desired.what`. Question
coordinates are context, not reconstruction targets. Keep established
reconstruction, grammatical, commitment, boundedness, and chooser losses
where they remain applicable, reporting components separately.

Past/future target selection must not introduce a second full model pass over
the clean target merely to manufacture its conceptual representation. Begin
with the existing comparable output representation and add alignment work
only when the need is demonstrated.

### 8.3 Thinking and credit

For a supervised presentation that thinks, calculate the answer loss only
after parity. The trace must retain the sampled grammatical decisions that
opened, extended, and closed the dialogue so the existing chooser-learning
path can assign credit.

Closure pressure is a chooser input, not permission to replace the answer
loss with a parity-only objective. Returning quickly with a wrong answer and
thinking indefinitely are both undesirable; report answer quality and
thinking length separately.

## 9. Implementation sequence

### Step 1: introduce the question representation

Add the smallest representation that can express present, past, future, a
relative offset, and a supervised non-temporal question. Keep `where`
optional and absent initially. Ensure the representation travels inside the
model-visible question.

### Step 2: implement `Data.what()`

Delegate exact present lookup to current `Data` batching. Add document-safe
past/future target selection by zero-based presentation index. Return
unavailable at split/document boundaries. Adapt explicit supervised outputs
without requiring their equality to the input.

### Step 3: implement `Model.what()`

Expose a method that delegates to the existing model execution and returns
the produced `what`. Begin with byte-identical `What(present)` reconstruction.
Do not add a second model, generator, or data capability to `Model`.

### Step 4: route through `Model.run()`

Make question-family selection part of existing batch preparation. Resolve
the `Data` target for evaluation/loss and the `Model` response through the
ordinary run path. Preserve current inference behavior while allowing its
output to be attached to the presentation.

### Step 5: make temporal context causal

Train present, past, and future questions. Ablate or shuffle the temporal
question content while holding the input fixed. The model must perform worse
when that content is wrong; otherwise it has not learned the query.

### Step 6: extend LTM slots

Store conceptual input and actual model output with independent presence
masks. Preserve ordered LTM behavior, capacity limits, document resets, and
detachment from expired autograd graphs.

### Step 7: implement iterative parity

Derive the open-question stack from input-only and output-only slots. Loop
through the same model evaluation without recursion. Feed completed
subquestions back through LTM context. Add monotonic closure pressure, a
finite iteration limit, and forced best-effort LIFO closure.

### Step 8: remove the obsolete boundary

After parity with existing reconstruction is demonstrated, migrate callers
off Teacher-oriented lesson APIs. Delete or reduce the old classes to
temporary compatibility adapters that delegate to `Data.what()` and
`Model.run()`. No compatibility adapter may become a required model-owned
capability.

### Step 9: expand the curriculum

Begin with clean present reconstruction, then mix one-step past/future
completion, supervised answers, and inference. Increase temporal distance and
thinking depth only after causal-use and parity tests pass.

## 10. Required tests

### Interfaces and compatibility

- `Data.what(What(present))` selects the same target as current
  reconstruction.
- `Model.what(What(present))` is byte-identical to the established clean path
  before new training is enabled.
- `Model.run()` remains the only training/evaluation orchestrator.
- No target architecture requires a Teacher, Student, source-oracle, view, or
  recursive frame class.

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

### Grammar and performance

- Slot operations and temporal question use appear in the grammatical trace.
- No hidden semantic-to-output path bypasses the grammar chooser.
- Present reconstruction, past recall, future completion, supervised answer
  quality, mean thinking iterations, forced-closure rate, and sentences/s are
  reported separately.
- Existing stateless/stateful, device, checkpoint, and two-pass chooser tests
  remain green.

## 11. Performance and definition of done

Use the matched B24 benchmark and report median wall-clock throughput after
warm-up. The 41.989 complete sentences/s result is the clean reference.

The next iteration is complete when:

1. Both existing classes expose working `what()` delegation.
2. Existing reconstruction is represented as `What(present)` without a
   material regression in output.
3. `Model.run()` trains/evaluates present, past, future, and supervised
   questions through one path.
4. Temporal question content is causally used.
5. LTM stores conceptual input and actual model output in independently
   optional halves.
6. Input-only and output-only slots implement a LIFO stack without a separate
   recursive frame structure.
7. Thinking incorporates completed subquestions and always restores parity
   with best-effort answers.
8. Target isolation and document-boundary tests pass.
9. Clean B24 median throughput is within 15% of the matched baseline, or any
   larger regression is isolated and explicitly accepted.
10. Teacher-oriented code is no longer an architectural dependency.

## 12. Preserved roadmap: noun and verb concept formation

The following work remains downstream of the shared `what()` loop. It is not
claimed as implemented by the immediate iteration.

### 12.1 Synchronic and diachronic concept order

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

### 12.2 Meaning and grammatical application

A surface verb's MetaSymbol binds to learned diachronic concept rows. Grammar
and argument roles select the intended sense and pass its atom to the shared
executor. `VerbLayer` applies a meaning but does not create it. Noun selection
likewise uses learned concepts rather than treating a surface row as the
object's meaning.

A transformation may be forward-only and many-to-one. Reconstructing a past
state is an abductive `Model.what()` query using memory; it is not a required
algebraic inverse.

### 12.3 Row-local transition evidence

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

## 13. Preserved roadmap: shared sparse verb executor

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

## 14. Preserved roadmap: dimensions and grounding

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

## 15. Acceptance gates for the preserved roadmap

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
