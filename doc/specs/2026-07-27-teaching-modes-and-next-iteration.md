# Teaching modes, the symbolic mind, and the next iteration

> **Status:** implementation specification and maintainer handoff,
> 2026-07-27; cognitive-boundary and curriculum directives added 2026-07-28;
> oracle/student API and future-state revisions added 2026-08-03.
>
> **Immediate scope:** teach the student the objective coordinate system,
> make reading an explicit query/response process, and replace the retired
> next-sentence predictor with `Teacher.Data(address)` as a private truth oracle
> and `model.what(address)` as the student's addressed estimate. The estimate
> must cover a future conceptual state as well as present reconstruction, while
> its reasoning remains grammatical. Establish a noun-first linguistic
> curriculum before adding state transitions and modifiers. Thinking is
> specified as the inference-time continuation of the same mechanism, but
> autonomous truth-store admission remains deferred.

## 1. Current boundary

Teacher v1 is a clean reconstruction and performance baseline, not active
reading.

Today:

1. The data cursor selects a passage.
2. `Teacher` resolves its objective source address and clean `what` through
   the method currently named `Teacher.What`.
3. The complete passage is presented to the student.
4. The existing perceptual and grammatical path reconstructs it.
5. Teacher owns the reconstruction loss and scoring boundary.

The current `Teacher.What` is an exact dataset lookup, not a learned student
prediction. The revised interface calls that privileged lookup
`Teacher.Data(address)` and reserves `model.what(address)` for the student's
best estimate from evidence available at the time of the query.

The objective query is exposed as CPU categorical tensors, but no trainable
student module consumes it. The student therefore does not yet:

- learn the objective coordinate system;
- expose a learned, address-conditioned `model.what(address)`;
- choose or request a `where`/`when`;
- retrieve the perception at a requested address;
- depend on the address when perception is degraded or blank;
- predict a future state before the future perception has been presented.

The accepted clean B24 benchmark is 41.989 complete sentences/s. That result
is an upper bound for the current computational core, not the throughput of
the eventual addressed learner. Most of the improvement over the 6.803
sentences/s baseline comes from removing the independent next-sentence
predictor and its prefix-wide peer computation, not from Teacher bookkeeping.

The next iteration should spend some capacity and some lessons learning
objective coordinates. A modest throughput regression is expected and
acceptable if ablations prove that those coordinates causally improve
degraded, recall, and prediction lessons.

The clean reconstruction benchmark must therefore remain labelled a
reconstruction baseline. It does not yet measure the predictive ability of
`model.what(address)`.

## 2. Non-negotiable coordinate separation

There are three different coordinate systems. They must not be collapsed.

### 2.1 Subjective attention

The existing in-model `.where` and `.when` bands locate presentations in the
student's subjective workspace. Teacher must never read, stamp, replace, or
supervise those values directly.

### 2.2 Objective query coordinates

These identify the event or passage the student is being asked to know:

```text
ObjectiveWhere =
    (context/domain identity,
     split and direct row,
     document identity,
     sentence identity,
     character span,
     optional external alias)

ObjectiveWhen =
    (observation/sequence position,
     optional represented event time,
     optional relative target offset)
```

For a static text, document/sentence/span is the primary objective location.
Sequence position or a relative offset distinguishes previous, current, next,
and later observations.

### 2.3 Source version and provenance

Corpus revision is validity metadata, not the event time the student is asked
to recall or predict:

```text
SourceVersion =
    (snapshot/revision identity,
     shard/content digest,
     optional publication or capture time)
```

Teacher v1 currently carries snapshot identity in `ObjectiveWhen`. The next
schema revision should split it into `SourceVersion` while adding target
sequence/event time to `ObjectiveWhen`. This is a migration of the external
Teacher contract only; it must not touch subjective `.when`.

DOIs, publication dates, URLs, and titles are optional aliases or provenance.
They do not uniquely identify a passage. FineWeb supplies none of them
reliably, so the lossless local address remains corpus snapshot plus ordered
document/sentence/span coordinates.

## 2A. Non-negotiable boundary of the student mind

The student may have access to the subsymbolic content carried by percepts,
words, and concepts, but it may transform that content only by applying an
explicit grammatical operation. The symbolic content of those operations is
the mind's control language.

Consequently:

- perception and the lexicon may present subsymbolic content to the mind;
- the mind may retain and bind references to that content;
- every deliberate subsymbolic transformation of it must be an application of
  a named grammatical operation with explicit operands and result;
- LTM lookup, comparison, binding, and other reasoning actions must enter the
  computation as grammatical actions rather than as an unrecorded neural
  side-channel;
- grammar choice, word choice, and memory use must leave a replayable symbolic
  trace;
- no general MLP, attention blend, or hidden context path may bypass the
  grammar by mapping word/concept content directly to control or output.

This restriction is the intended safety and human-interpretability boundary:
the subsymbolic meaning of a word may remain unanalyzed, while the thought
that acts on it is expressed by syntax.

The exact software mechanism that enforces this boundary remains open. Sealed
semantic payloads, typed operation capabilities, trace validation, and
restricted operator parameterizations are possible implementations, not
requirements selected by this specification.

## 3. Query, observation, and scoring contracts

Active reading requires a strict separation between truth lookup, permitted
observation, and student estimation, even if Teacher v1 implements the first
two in one Python object:

```text
Student.Query(context) -> ObjectiveAddress
SourceOracle.Observe(ObjectiveAddress, observation_policy) -> Perception
model.observe(ObjectiveAddress, Perception) -> updated student state
model.what(ObjectiveAddress) -> WhatEstimate
Teacher.Data(ObjectiveAddress) -> private TeacherDatum
Teacher.Score(ObjectiveAddress, WhatEstimate, TeacherDatum) -> loss
```

- **Student** selects an address or accepts an assigned address.
- **SourceOracle** returns only the perception permitted by the lesson mode.
- **`model.what(address)`** returns the student's best current estimate. Its
  only explicit argument is the objective address; current perception,
  conceptual state, STM, and LTM are student-owned evidence, not extra truth
  arguments supplied by Teacher.
- **`Teacher.Data(address)`** is the exact corpus/world oracle. It returns the
  clean datum and provenance for scoring, and is callable only by the teaching
  controller, evaluation harness, and tests. The model, its memory, its query
  machinery, and its grammar must have no reference to this callable.
- **Teacher** retains the `TeacherDatum` privately and scores the estimate.

`Teacher.Data` and `model.what` must not be aliases or wrappers around the same
lookup, and the learned student call graph must have no capability to invoke
`Teacher.Data`. An outer training facade may coordinate both APIs, but only in
separate controller and student scopes. The capitalization in this
specification deliberately distinguishes the privileged dataset API from the
learned model API. The minimum records are:

```text
TeacherDatum:
    resolved objective address
    source version and provenance
    clean perceptual/source value
    optional clean surface rendering

WhatEstimate:
    requested objective address
    predicted conceptual state in [-1, 1]
    grammatical derivation distribution or sampled derivation
    optional reconstructed perceptual/surface value
    root and constituent WhatFrame trace references
    references to the student-visible evidence and memory reads used
    confidence and provisional/verified status
```

The public `model.what(address)` signature remains small because the model
owns its current state. Internally it must construct a typed evidence record so
tests can identify which conceptual, perceptual, STM, and LTM values were
available. No field of `TeacherDatum` may appear in that record.

The source response must not expose the private clean target on degraded or
blank lessons. Teacher-only content must never enter LTM, discourse caches,
grammar inputs, address embeddings, or working memory.

Introduce an explicit lesson record:

```text
TeachingLesson:
    mode
    assigned_address | student_request
    resolved_address
    source_version
    observation_policy
    presented_perception
    private_teacher_datum
    private_detached_target_concept_if_available
    student_what_estimate
    student_what_frames_and_evidence_refs
    prediction_asserted_at
    address_targets
    grammar_and_query_trace
    loss_components
```

The resolved address used for scoring must equal the address that generated
the presented perception. A student request may be snapped to a legal source
coordinate, but both the request and resolved address must be retained so
query accuracy can be scored.

## 4. Teaching and operating modes

Every lesson uses the same perceptual, conceptual, grammatical, and `What`
machinery. Modes differ in who selects the address, how much perception the
source returns, and which losses are available.

| Mode | Address selection | Perception | Primary learned ability |
|---|---|---|---|
| `READ_ASSIGNED` | Teacher/cursor | Complete | Present-input reconstruction and address association |
| `LOCATE` | Teacher/cursor | Complete | Infer or contrastively match objective coordinates from `what` |
| `READ_REQUESTED` | Student, initially constrained | Complete response at request | Move a reading cursor and request the intended passage |
| `DENOISE` | Assigned or student request | Partially degraded | Use address, history, and grammar rather than surface copying |
| `RECALL` | Past address | Blank or minimally cued | Reconstruct a past `what` from LTM and coordinates |
| `PREDICT` | Next/later address | Blank or minimally cued | Reconstruct a future `what` from history and context |
| `THINK` | Student-selected | Blank, retrieved, or imagined | Produce a provisional conceptual `what` for another internal step |
| `VERIFY` | Previously predicted address | Reality now available | Compare a prediction/thought with an observed `what` |

### 4.1 `READ_ASSIGNED`

This is the current clean lesson with one addition: a trainable address
encoder consumes the objective query. Because complete perception makes the
address optional, reconstruction loss alone cannot prove that the encoder was
learned. Attach address matching and inverse-location objectives to the same
pass.

This mode is passive reading. It is a necessary bootstrap, not the final
sensorimotor reading behavior.

### 4.2 `LOCATE`

Given a complete or lightly degraded `what`, the student must distinguish the
correct objective address from in-batch and nearby hard negatives.

Do not regress raw 64-bit identity hashes. Treat corpus, context, snapshot,
split, document identity, and external aliases categorically. Train ordered
document/sentence/span positions through relative deltas, buckets, or
continuous positional features.

Useful negative addresses include:

- a sibling sentence in the same document;
- the same sentence position in another document;
- an adjacent sentence;
- the same document under the wrong source version;
- a passage from a sibling text in the same context whole.

`LOCATE` can share the normal reconstruction forward. Dedicated locate-only
lessons are useful during bootstrap but should not require another grammar
parse.

### 4.3 `READ_REQUESTED`

The student emits a query before receiving the passage. Start with a small
relative action space:

```text
stay
previous sentence
next sentence
previous document
next document
seek relative sentence offset
select active document/context
```

A relative cursor policy is more efficient and learnable than a million-way
absolute softmax. The source resolves the action against the active document
and returns the allowed perception. Teacher scores both the query decision and
the reconstructed `what`.

Initially constrain legal actions to the active document and nearby cursor
positions. Expand to associative LTM/document retrieval only after local
cursor movement is reliable.

### 4.4 `DENOISE`

Remove progressively more lexical/perceptual content while preserving:

- the active reading/domain mode;
- objective address conditioning;
- student-visible history and LTM;
- discourse state;
- grammar state that was produced by the student.

Never preserve a constituent label or grammar trace that trivially identifies
the private clean target. Shuffling the address across a batch must measurably
worsen reconstruction; otherwise the student is ignoring it.

### 4.5 `RECALL`

Request a past address with blank perception. The answer must come from
student-visible memory and learned coordinates, not Teacher's private target.
Teacher scores the result against historical source data.

Recall should distinguish:

- source version;
- observation/event target time;
- assertion or storage time;
- the student's subjective time of remembering.

Only the first three belong to the external record. The fourth remains in the
student's subjective `.when`.

### 4.6 `PREDICT`

Prediction is an estimate of the `what` state at a next or later objective
address. Surface next-sentence reconstruction is one textual supervision for
that state, not the definition of the faculty. Prediction must not restore the
retired independent next-sentence predictor:

```text
current_address = address at t
future_address = address at t + delta

model.observe(current_address, permitted current perception)
predicted_future = model.what(future_address)

# Only after predicted_future is complete and isolated from the target:
future_datum = Teacher.Data(future_address)
Teacher.Score(future_address, predicted_future, future_datum)
```

The call to `model.what(future_address)` receives blank target perception. It
may use the model's current conceptual state, current permitted perceptual
state, STM/discourse, explicit LTM reads, and the target address including its
relative time. It may not use a forward activation, reverse cache, grammar
trace, or target encoding computed from `future_datum`.

For state-level supervision, capture the future conceptual carrier when the
clean future datum is processed by its ordinary later analysis, or by the same
ordered packed batch. Detach that carrier before using it as a target for the
earlier estimate. Do not construct another grammar forest solely to obtain a
target. If a target carrier is not yet available, defer the state-alignment
term and score the decoded source/surface estimate; do not leak the future
through an auxiliary student-visible pass. Any target-only analysis must be
stateless and must not populate student STM, LTM, discourse, runtime caches, or
evidence records. Decoded source/surface reconstruction remains a second
target, preventing a collapsed conceptual carrier from satisfying the state
loss by itself.

The LTM view is also taken **as of the prediction**, not as of target scoring.
It may include a properly tagged earlier prediction about the future address,
but it may not include an observation learned from the future datum merely
because the offline loader has already prefetched it.

The target may be known to the offline Teacher, but it must remain inaccessible
to the student until scoring. Any prediction stored before verification is a
claim with target time and provenance, not truth. When reality at that address
is later observed, `VERIFY` compares the prior estimate with a fresh
`Teacher.Data` result; it does not retroactively make the estimate true.

### 4.7 `THINK`

Thinking is the inference-time continuation of addressed reconstruction when
no external perception is supplied and the student selects its own query.

A thought need not first become a surface sentence. The canonical internal
record is conceptual:

```text
ThoughtRecord:
    requested objective or internal address
    conceptual what
    root/constituent WhatFrame references
    optional grammatical derivation
    optional surface rendering
    parent thought/memory references
    confidence
    status = imagined | inferred | predicted | verified | contradicted
    asserted_at
    represented/target event support
```

The thought enters bounded working memory and may condition the next query.
It must not enter the historical truth store automatically. A later
`VERIFY` event may compare it with perception, update confidence, and admit a
properly provenance-tagged claim under a future truth-admission policy.

Thinking therefore follows:

```text
desire_to_know what(address) -> root WhatFrame
  -> bounded constituent-What stack
  -> syntactic perception/STM/LTM queries
  -> grammatical reductions over returned constituent Whats
  -> one root grammatical/conceptual What construction
  -> provisional ThoughtRecord
  -> next desire_to_know or stop
```

Use explicit step, wall-clock, novelty, and confidence budgets so inference
cannot recurse indefinitely. Multiple grammatical derivations and metaphor
may later branch this loop, but the next iteration should retain one sampled
derivation per thought.

## 5. Replacement for the next-sentence predictor

The benchmark shows that the independent predictor was the dominant long leg.
Do not restore it under another name. The global attention/reasoning path must
be connected to the model and become responsible for choosing grammar over
words. To the greatest extent possible, its reasoning must itself be
syntactic rather than an MLP selecting grammar as an external tool.

### 5.1 Required behavior

Perception, symbolic activation, STM, LTM, the objective address, and the
current teaching/operating mode may all inform the mind. Their use must obey
the boundary in section 2A:

- perception, STM, and LTM reads are explicit syntactic query actions;
- retrieved items remain identifiable records rather than disappearing into
  an untraceable blended context;
- grammatical operations determine how subsymbolic word/concept content is
  transformed;
- the derivation determines the grammatical role before lexical realization
  chooses a word;
- the reasoning and the resulting sentence share one inspectable derivation
  where possible.

Preserve:

- one forest construction;
- one stochastic derivation selection;
- one top-down reconstruction;
- nonzero probability for second-, third-, and tail-ranked derivations;
- no independent predictive parse.

Bounded retrieval may be cached and reused across a derivation so the same LTM
query is not repeated for every candidate rule.

### 5.2 Open architecture

This specification intentionally does not select the internal architecture of
the syntactic reasoner. Open possibilities include a grammar-native agenda or
forest policy, explicit additive rule evidence, or a hybrid construction.
Whether internal thought uses surface linguistic syntax or a richer but still
human-readable grammar is also open.

An MLP tool-user that consumes context and selects grammatical tools remains a
useful experimental comparison, and may work functionally. It is not the
preferred interpretation of the symbolic architecture because its reasoning
would remain opaque. The previously proposed fixed context-MLP dimensions,
rank-32 conditioning, and FiLM design are therefore withdrawn as requirements.

The implementation must first settle the smallest mechanism that connects
global attention/reasoning to grammar while satisfying section 2A, then
measure it against the MLP baseline.

### 5.3 Query as syntax

Relative cursor movement and associative retrieval should be expressible as
syntactic actions whose result determines the objective `where`/`when` that
the unified `What` path reconstructs. The exact query grammar, scoring rule,
and search procedure remain open.

Reasoning itself is an unresolved **desire to know `what(address)`**. Satisfying
one `what` may require constituent `what` tasks, so the reasoner owns a bounded
stack (or equivalent explicit agenda) of `WhatFrame` records:

```text
WhatFrame:
    task id and parent task id
    requested objective or internal address
    grammatical role expected by the parent
    permitted evidence sources
    evidence cutoff / as_of coordinate
    selected grammatical query operations
    constituent WhatEstimate results
    reduction operation and partial result
    remaining step/depth budget
    status = unresolved | waiting | resolved | failed
```

The logical execution is:

```text
push desire_to_know(target_address)
while a WhatFrame is unresolved and budgets remain:
    choose a legal grammatical query/reduction operation
    query_perception(...) | query_stm(...) | query_ltm(...)
    push any required constituent what(subaddress_or_internal_question)
    reduce returned constituent Whats with a named grammatical operation
pop the resolved WhatEstimate into its parent, or return the root estimate
```

Perception, STM, and LTM therefore connect to the syntactic reasoner through
three explicit families of grammatical query operations. A returned value is
a referenced `WhatEstimate` or evidence record, not an anonymous attention
blend. The trace must show the query, address/pattern, source, returned record
identity, parent frame, and grammatical reduction that consumed it.

The stack is a semantic requirement, not a requirement for slow Python
recursion. Independent constituent queries may be batched, memoized within one
reasoning episode, or represented by an iterative chart/agenda, provided the
parent/constituent dependency graph and evaluation order remain replayable.
Constituent results are conceptual by default: they do not each trigger a
surface rendering or an independent full grammar forest. They are nodes or
subderivations in the root reasoning episode, and only the root is rendered
unless an explicit grammatical operation requests a constituent rendering.
Repeated `(source, query, as_of)` requests should reuse the same bounded result.
Cycle detection plus depth, step, and retrieval budgets must fail closed rather
than silently reading arbitrary memory or recursing indefinitely.

In a teacher-led lesson, Teacher supplies an addressed root task and later
scores the root answer. In `THINK`, the student may originate the root desire.
Teacher never prescribes constituent questions, pushes frames, chooses memory
records, or places private targets on the stack; those are student reasoning
actions.

### 5.4 Decouple model-level analysis and synthesis

At the model boundary, `forward()` and `reverse()` must no longer mean that
one is the numerical inverse of the other or that reverse can run only from
the most recent forward trace. That coupling is adequate for reconstructing a
present input, but it cannot generically generate the state at an address
whose perception has not yet occurred.

The required logical interfaces are:

```text
model.forward(perception, current_address) -> ObservedState
    # bottom-up analysis; updates permitted student state

model.what(target_address) -> WhatEstimate
    # evidence-conditioned estimate; target perception may be blank

model.reverse(WhatEstimate.conceptual_state,
              WhatEstimate.derivation) -> reconstructed/generative What
    # top-down synthesis from the estimated state, not from a cached input
```

`analyze` and `generate` may be introduced as clearer names while retaining
`forward`/`reverse` compatibility adapters. The behavioral requirements are:

- `model.what(address)` and top-down generation work after per-forward caches
  are cleared; they require no immediately preceding forward of the target;
- the root of generation is the estimated target conceptual state, not the
  conceptual state encoded from the current or target source value;
- present reconstruction is the special case
  `generate(model.what(current_address)) ~= Teacher.Data(current_address)`,
  not a requirement that `reverse(forward(x)) == x` by construction;
- analysis and generation may share lexicons, codebooks, grammar definitions,
  and selected parameters, but may have direction-specific parameters and
  losses;
- the causal transition from current evidence to a future state is
  forward-only. No inverse of that transition is required or used;
- current perception may influence a future estimate only as an explicit
  evidence operand of `model.what`, never as a hidden reverse cache.

This is a model-level change. Local grammatical operators may retain
`forward`/`reverse` or `compose`/`generate` pairs where inversion is meaningful.
Their local invertibility must not impose an inverse relationship between an
observed present state and an imagined future state.

### 5.5 Initial future-state prediction operation

The initial implementation should introduce one named grammatical operation,
`predict_state`, whose result is a conceptual carrier bounded to `[-1, 1]`.
The existing nonlinear `SigmaLayer` is a reasonable first implementation of
the bounded transform. Because the current `SigmaLayer` is substrate rather
than a chart-dispatched grammar operation, it must be owned by a traced
`predict_state` wrapper; it must not become an unrecorded global neural path.

The minimum typed operands are:

```text
PredictionEvidence:
    target objective address and relative target time
    current conceptual state
    current permitted perceptual state plus availability mask
    bounded LTM records returned by constituent syntactic What queries
    optional STM/discourse records returned by constituent syntactic queries
    prior grammatical state and the WhatFrame dependency trace
```

Conceptual, perceptual, memory, and address spaces need not have equal
dimensions. Give each role an explicit adapter into the predictor's input
width, preserve role identity when packing the operands, and record the
adapter, operand references, availability masks, and prediction operation in
the derivation trace. A missing perception is represented by an availability
mask and configured blank carrier, not by silently treating an ordinary zero
state as missing.

A concrete minimum implementation is:

```text
z_address = address_adapter(target_address, relative_time)
z_concept = concept_adapter(current_conceptual_state)
z_percept = percept_adapter(current_perceptual_state, percept_available)
z_memory  = memory_adapter(grammar_fold(explicit_ltm_records))
z_stm     = stm_adapter(grammar_fold(explicit_stm_records))       # optional

evidence = fixed_role_pack(z_address, z_concept, z_percept,
                           z_memory, z_stm, availability_masks)
future_concept = predict_state.SigmaLayer(
    invertible=False, nonlinear=True).forward(evidence)
```

`fixed_role_pack` is part of the named prediction operation, not a general
attention side-channel. `grammar_fold` must retain the identities and trace of
the bounded records it combines. Memory and perception operands may enter this
pack only after the corresponding query/constituent frame has resolved. The
initial predictor is forward-only: do not call
`predict_state.SigmaLayer.reverse()` and do not force its input and output
widths to match. The ordinary top-down grammatical generator decodes
`future_concept`.

The model may later replace this single-state estimate with several sampled
or scored future hypotheses. For the first implementation, uncertainty over
words and constructions remains in the stochastic grammatical derivation;
the continuous conceptual carrier is one bounded state estimate. Teacher must
score both its future-state agreement and its grammatical/surface realization
so a deterministic state loss cannot be satisfied merely by averaging or
collapsing incompatible continuations.

## 6. Address representation and losses

### 6.1 Encoder

Use separate components:

- learned categorical embeddings for context, corpus, source version, split,
  document identity, and external aliases;
- ordered positional encodings for document order, sentence order, character
  span, observation index, and relative target offset;
- a mask for absent optional fields;
- a small fusion MLP producing one address atom.

The address atom used by `predict_state` must pass through the explicit
address-role adapter described in section 5.5. Every predictor adapter must
emit a finite bounded value before the values are packed for nonlinear
`SigmaLayer`; no adapter may assume perceptual and conceptual dimensions are
equal.

Raw stable hash magnitude has no semantic meaning. Hashes may key embedding
rows but must never be interpreted as ordered scalars.

### 6.2 Shared-pass objectives

Initial scales:

```text
surface What reconstruction                 1.00
future conceptual-state alignment            0.25  (PREDICT only)
address/What in-batch contrastive match      0.10
relative cursor or position objective        0.05
student query-action objective               0.05
grammar chooser auxiliary                    0.10
chooser entropy floor                        0.01
```

These objectives should normally share one perceptual/grammatical forward.
Do not spend a second parse merely to predict an address. Report every
component separately and ablate each one.

For blank prediction lessons, the same surface reconstruction term supplies
the former next-sentence learning signal. Reuse the detached conceptual state
from the future address's ordinary analysis in the ordered training sequence;
do not parse the target a second time. If ordered target-state reuse is not
available for a lesson, mask the state-alignment term rather than fabricating
a target from a privileged input path. Begin with the same normalized
bounded-carrier distance used elsewhere, weighted as above, and retain the
decoded surface and grammar losses to prevent representational collapse.
Report state and surface accuracy separately. There is no additional
independent next-sentence loss or parse.

## 7. Proposed curriculum

The Teacher controls two independent curriculum axes: linguistic complexity
and the availability of perception. Linguistic structure must be staged first;
the 10/70/20 degradation schedule then applies within the currently permitted
linguistic stage.

### Linguistic curriculum: nouns, transitions, and modifiers

The Teacher's objective spacetime continuum is approximated in the student's
subjective universe by nouns and verbs:

```text
noun at t      = an object or object-state
verb over t    = a transformation from prior noun-state(s)
                 to subsequent noun-state(s)
```

The required teaching order is:

1. **Nouns.** Establish objects and object-states before asking the student to
   explain change.
2. **Nouns and verbs.** Once noun representations are sufficiently stable,
   learn verbs from repeated transitions between earlier and later noun
   states.
3. **Adjectives.** Add linguistic refinement of noun states.
4. **Adverbs.** Add linguistic refinement of verbs and transitions.
5. **Further linguistic complexity.** Introduce the remaining constructions
   progressively rather than assuming the complete grammar from the start.

The Teacher should be able to restrict both lessons and available grammar
productions to the current stage. The exact stability criterion for promoting
between stages, the mining of aligned state transitions, and the detailed
production subsets remain open.

### Perception curriculum

Retain the previously chosen 10/70/20 progression and a ten-percent clean
anchor, but make the mode mixture explicit.

### Phase A: coordinate bootstrap, first 10%

```text
80% READ_ASSIGNED
20% LOCATE
```

Use complete perception. Train address matching and inverse location during
the same forward. Do not enable blank prediction yet.

### Phase B: contextual association, next 70%

Continuously increase degradation. A starting mixture is:

```text
10% READ_ASSIGNED clean anchors
35% DENOISE
20% LOCATE or constrained READ_REQUESTED
15% RECALL
20% PREDICT
```

Gate `RECALL` and `PREDICT` on successful leakage tests and measurable address
use. Early in this phase, keep their perception minimally cued; anneal toward
blank.

### Phase C: autonomous query pressure, final 20%

```text
10% READ_ASSIGNED clean anchors
20% DENOISE
20% RECALL
30% PREDICT
20% READ_REQUESTED / query-policy lessons
```

Use maximum configured degradation. `THINK` may run as an evaluation mode, but
do not optimize unverified self-generated thoughts or admit them to truth.

Mode selection must be deterministic under the training seed and recorded in
the checkpoint so a resumed run preserves the curriculum.

## 8. Concrete next-iteration implementation

The next code iteration should stop after addressed, contextualized reading
and the first controlled degraded lessons. It should not attempt autonomous
truth learning, full verb induction, and metaphor simultaneously.

### Step 1: revise the Teacher contract

In `Teacher.py`:

- add `TeachingMode`, `SourceVersion`, `ObjectiveAddress`, `StudentQuery`, and
  the expanded `TeachingLesson`;
- expose the exact privileged lookup as `Teacher.Data(address)`;
- keep the old `Teacher.What(address)` only as a deprecated controller-side
  compatibility alias, never as the student's prediction API;
- move snapshot validity out of objective target time;
- retain compatibility adapters for v1 checkpoints/configuration;
- add deterministic mode scheduling;
- keep private clean targets inaccessible to the student.

In `data.py`:

- expose `observe(address, policy)` separately from the exact lookup used by
  `Teacher.Data(address)`;
- preserve document boundaries and exact source provenance;
- add legal relative-cursor resolution;
- never let a request cross context/document boundaries silently.

The learned student computation must not own or receive a `Teacher`,
data-loader, target callback, or other capability that reaches `Teacher.Data`.
The existing top-level training facade may coordinate Teacher and student
during migration, but must pass only the permitted observation and public
address across the student boundary.

### Step 2: add `model.what(address)` and split model-level directions

Add `WhatEstimate` and `PredictionEvidence` records, then expose one learned
`model.what(ObjectiveAddress)` entry point. It must collect only model-owned
conceptual state, permitted perception, STM/discourse, explicit LTM query
results, and address information.

Split the current model-level execution into independently callable analysis
and generation stages. Compatibility methods may retain the names
`forward()` and `reverse()`, but generation must accept a conceptual estimate
as its root and work with target-forward caches cleared. Preserve local
invertible grammar operators; remove only the requirement that whole-model
generation numerically invert the most recent analysis.

Add a present-input compatibility test before changing training behavior:
complete `READ_ASSIGNED` reconstruction through
`forward -> model.what(current_address) -> generate` must match the existing
path within an agreed numerical tolerance and remain within the throughput
gate.

### Step 3: add the objective-address encoder

Create a focused module such as `Addressing.py` rather than expanding
`Models.py` further. It should contain:

- categorical identity tables;
- ordered position encodings;
- address fusion;
- in-batch contrastive matching;
- relative cursor/query heads.

Wire one address atom into the student boundary without using subjective event
bands. Begin with `READ_ASSIGNED` and `LOCATE` only.

### Step 4: prove causal address use

Before degradation:

- round-trip every exact address through `Teacher.Data`;
- predict/match held-out addresses above hard-negative chance;
- swap addresses within a batch and verify address-matching loss increases;
- verify subjective `.where/.when` tensors remain byte-identical.

Then introduce light degradation and require shuffled addresses to worsen
surface reconstruction. If reconstruction is unaffected, stop: the address
encoder is being ignored.

### Step 5: connect global attention and reasoning through grammar

Make bounded, student-visible perception/STM/LTM reads available to the
grammatical derivation. Memory queries and their returned records must appear
in the symbolic trace, and no dense semantic-to-control bypass may choose
grammar or words outside that trace.

Represent each root desire to know as a `WhatFrame`, with an explicit bounded
stack/agenda for constituent `what` tasks. Add named grammatical operations
for perception, STM, and LTM queries and for reducing returned constituent
estimates into their parent. Implement memoization by `(source, query, as_of)`,
cycle detection, and hard depth/retrieval/step limits. Keep the logical
dependency graph even if execution is vectorized or batched.

Do not expose Teacher's private `what`, future truth, or clean grammar labels.
Do not repeat an identical memory retrieval for every forest candidate. The
query grammar, rule-scoring mechanism, and internal reasoner representation
remain open.

### Step 6: implement the forward-only future-state operation

Add a traced `predict_state` grammatical operation containing a nonlinear,
non-invertible `SigmaLayer`. Add bounded role adapters for objective address,
current conceptual state, permitted perceptual state, explicit LTM results,
and optional STM/discourse state. Do not assume equal carrier dimensions, do
not call the predictor's `reverse()`, and do not let its output bypass the
ordinary grammatical generator.

Add configuration gates for the prediction operation and its separately
reported future-state loss, retaining old-checkpoint behavior when disabled.
Construct `model.what(future_address)` before making the future datum visible.
Obtain the detached target conceptual state when that address is processed by
the ordinary subsequent analysis, preferably in an ordered packed batch. If
it is unavailable, mask or defer state alignment; a second grammar parse is
not an acceptable fallback.

First train and test on deliberately predictable local transitions. Compare
against previous-state copying and corpus-prior baselines before enabling the
general corpus mixture.

### Step 7: enable constrained student requests

Implement `READ_REQUESTED` over the small relative action set. Teacher scores
the requested versus intended cursor action and reconstructs the returned
passage. Keep normal corpus order as the dominant policy initially so query
exploration cannot destroy discourse continuity.

### Step 8: add the first partial and blank modes

Enable:

1. light `DENOISE`;
2. past-address `RECALL`;
3. next-address `PREDICT`.

Each uses the same single grammar forest. `PREDICT` additionally uses the
detached future conceptual-state alignment described above, but no second
grammar parse. Only after all three pass leakage and throughput gates should
the full curriculum be enabled.

### Step 9: scaffold the linguistic curriculum

Add explicit curriculum-stage metadata and the ability to restrict lessons
and productions to the noun stage. Specify the transition examples required
by the later noun-and-verb stage, but leave full diachronic verb induction
until addressed reading and stable noun learning are demonstrated.

### Step 10: specify, but defer, autonomous thinking writes

Implement `ThoughtRecord` and a bounded read-only thinking loop only after
addressed prediction works. Store thought records in working memory or a
provisional claim log. Persistent truth admission, contradiction repair, and
self-reward remain separate follow-up work.

## 9. Required tests

### Coordinate isolation

- Teacher and address encoders never inspect or mutate subjective
  `.where/.when`.
- Source version, objective target time, and subjective time are independently
  changeable.
- Raw identity hash order does not affect embeddings or query semantics.

### Query correctness

- `Teacher.Data(address)` returns the exact passage/state and provenance for
  every valid source address.
- `model.what(address)` returns a `WhatEstimate` and never invokes
  `Teacher.Data`, the dataset, or a target callback.
- Relative cursor actions resolve deterministically.
- Invalid context, split, version, or span combinations fail closed.
- Packed-row prefetch preserves the address associated with each sentence.
- Document and domain resets prevent cross-document cursor leakage.
- A root `model.what` can push two or more constituent `WhatFrame` tasks,
  resolve them, and reduce their estimates into the root in deterministic
  dependency order under a fixed seed.
- Repeated identical constituent queries reuse one bounded result; cycles and
  exhausted budgets fail closed.

### Learning and leakage

- Correct addresses beat nearby hard negatives.
- Shuffling addresses worsens degraded reconstruction.
- Blank lessons cannot access clean input through LTM, caches, traces, or
  Teacher fields.
- Future truth is unavailable to the student before prediction scoring.
- Capturing/detaching a future target state performs no extra student-visible
  STM, LTM, discourse, query, grammar-cache, or runtime-state writes beyond
  that address's ordinary analysis.
- Removing history worsens `PREDICT` relative to the full contextual model.
- Removing the address worsens addressed `RECALL` and `PREDICT`.
- On controlled factor-specific transitions, independently masking or
  shuffling conceptual state, current perception, and retrieved LTM records
  each worsens the cases constructed to require that factor.
- Held-out future-state prediction beats previous-state copying and a
  corpus-prior baseline before general `PREDICT` lessons are enabled.

### Direction and future-state independence

- `model.what(future_address)` runs before the target is observed and after
  target-forward/reverse caches have been cleared.
- Top-down generation from its predicted conceptual state works without an
  immediately preceding forward of the target datum.
- Present reconstruction still works through the decoupled interfaces.
- The `predict_state` `SigmaLayer` is nonlinear and bounded, is configured
  `invertible=False`, and its `reverse()` is never called.
- Predictor input and output dimensions may differ without adding a
  pseudoinverse or hidden projection outside the traced operation.
- Changing a current evidence operand can change the future estimate only
  through the recorded `predict_state` operands; no stale cache changes it.
- Future conceptual-state loss and decoded source/surface loss are finite,
  separately reported, and both resist a constant-state solution.

### Grammar and performance

- The grammar forest is constructed once per lesson.
- One derivation is sampled and reconstructed; no predictor-triggered second
  parse occurs.
- Producing or scoring a future conceptual target does not construct a second
  grammar forest.
- Identical bounded memory reads are not recomputed per candidate.
- Second-, third-, and tail-ranked derivations retain nonzero support.
- Clean reconstruction remains finite through address and context ablations.

### Symbolic mind boundary

- Every deliberate transformation of subsymbolic word/concept content is
  attributable to a named grammatical operation.
- Perception, STM, and LTM queries and returned records appear in the
  replayable derivation trace.
- Every constituent `what` records its parent, requested address/question,
  evidence source, result identity, and consuming grammatical reduction.
- No direct dense path maps semantic payloads to grammar control or words
  outside the grammatical derivation.
- Replaying the recorded operations and memory reads reproduces the same
  constituent stack and grammatical construction.
- Noun-stage lessons cannot silently invoke productions reserved for verbs,
  adjectives, adverbs, or later complexity.

### Thinking safety

- A thought cannot enter the truth store by default.
- Predictions retain asserted-at and target-event coordinates.
- Verification can mark a thought supported or contradicted without rewriting
  historical source truth.
- Step and time budgets terminate recursive thought.

## 10. Performance and speedrun gates

Use the accepted B24 clean result as the immediate comparison:

```text
41.989 complete sentences/s
14.572 GB peak RSS
```

For the first address-conditioned implementation:

- clean B24 throughput should remain at least 35.69 sentences/s, a maximum
  15% regression;
- report address matching and cursor accuracy alongside reconstruction;
- report `model.what` latency and the added future-state predictor parameter
  count separately from analysis and grammatical generation;
- report B28 separately after confirming memory headroom;
- report lessons/s, complete target sentences/s, and words/s because locate
  or query lessons may not be equivalent to one reconstructed sentence;
- report clean, degraded, recall, and prediction modes separately and as a
  fixed mixture;
- demonstrate in the profiler that future-state supervision adds no second
  grammar forest and identify the cost of retaining or pairing the ordinary
  future target state.

A one-million-sentence run is an engineering endurance benchmark, not by
itself a capability speedrun. At an aggregate 100 complete sentences/s it
takes 2 hours, 46 minutes, and 40 seconds. Before using it, freeze:

- corpus snapshot and accepted-sentence filter;
- mode mixture and curriculum;
- initialization seed policy;
- batch/word capacity;
- quality checkpoints and final held-out gates.

The eventual BasicModel speedrun should be **time to a fixed capability
threshold**, including addressed reconstruction, grammar validity, recall,
and future reconstruction. Until those thresholds are calibrated, call the
fixed-count benchmark the “1M-sentence mode-mixed endurance run.”

## 11. Definition of done for the next iteration

The next iteration is complete when:

1. `Teacher.Data(address)` is an exact private oracle and
   `model.what(address)` is a learned estimate with no route to that oracle.
2. A student address encoder consumes objective coordinates without touching
   subjective bands.
3. The student can match a passage to its objective address against hard
   negatives.
4. A constrained student query advances the reading cursor and retrieves the
   requested passage.
5. Lightly degraded reconstruction causally benefits from the correct address.
6. Model-level analysis and generation run independently: generation from a
   predicted conceptual state needs no target forward or inverse cache, while
   present reconstruction remains functional.
7. A traced, forward-only nonlinear `SigmaLayer` prediction operation uses
   typed objective-address, conceptual, perceptual, and explicit memory
   evidence to produce a bounded future conceptual state.
8. Held-out future-state and decoded-`What` prediction beat previous-state
   copying and corpus-prior baselines without an independent predictor or
   second grammar parse.
9. Global attention/reasoning evidence reaches grammar through traceable
   perception/STM/LTM query actions and bounded constituent `what` frames,
   without a dense semantic-to-control bypass.
10. Grammar, the parent/constituent reasoning stack, memory use, prediction
    operands, and word realization produce a replayable symbolic trace.
11. The Teacher can enforce a noun-only curriculum stage and represent the
    later noun-and-verb, adjective, adverb, and further-complexity stages.
12. Provisional thoughts and predictions remain outside historical truth;
    correctness tests pass and clean throughput remains above the 15%
    regression gate.

## 12. Open architectural questions

The directives above deliberately leave these decisions unresolved:

- how the code prevents non-grammatical access to subsymbolic payloads;
- the query grammar and the representation of retrieved records;
- how the syntactic reasoner scores and searches legal actions;
- how independent constituent `what` frames are scheduled and batched without
  losing their explicit dependency graph;
- whether internal syntax is surface language or a richer readable grammar;
- how to stabilize a learned future conceptual target while its analysis
  representation is itself changing;
- when a single bounded state estimate should become multiple scored future
  hypotheses;
- how noun stability is measured before verbs are introduced;
- how repeated state transitions are aligned and promoted into verbs.

Resolve these with the smallest measured implementation that preserves the
symbolic trace. Do not treat the speculative mechanisms in this document as a
settled design.

## 13. Deferred work

The following remain intentionally outside the next iteration:

- persistent admission of student-generated truth;
- reward from unverified self-generated thoughts;
- unrestricted associative query over all LTM;
- multiple simultaneous parses or future-state hypotheses for metaphor and
  multimodal prediction;
- full diachronic verb induction and sparse verb experts (the curriculum
  scaffold and noun-first gate are not deferred);
- creation/destruction and identity-split ontology;
- grounding/re-anchoring repair;
- replacing explicit objective addresses with purely in-sentence cues.

Those features should build on a measured addressed `What` loop rather than
being introduced before the student can demonstrably read, locate, recall,
and predict through one shared mechanism.
