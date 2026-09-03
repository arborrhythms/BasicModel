# Teacher, addressed What, and concept learning: unified specification

> **Status:** implementation specification and maintainer handoff,
> 2026-07-27; cognitive-boundary and curriculum directives added 2026-07-28;
> oracle/student API and future-state revisions added 2026-08-03;
> consolidated with the earlier Teacher reconstruction and temporal-concept
> plan on 2026-09-03. This is the single normative specification for both the
> addressed-reading iteration and the explicitly deferred concept roadmap.
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

The earlier plan has been removed after incorporating its ontology, concept
learning, verb representation, loss, and acceptance requirements here. The
revised oracle/student APIs, time/version separation, B24 performance gate,
and addressed-reading-before-verb-induction sequence govern throughout.
Preserved long-term requirements are in sections 14-17; they are not claims
about implemented capabilities or prerequisites for the first addressed
reading milestone.

The companion [What and queryable spacetime design](../WhatSpacetimeDesign.md)
explains interface ownership, the constituent-What stack, and the chooser's
derivation/location context. It is a high-level design view of this spec, not
a second implementation plan. Runtime behavior remains documented separately
in [Componentization](../Componentization.md) and [Language](../Language.md).

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

The serial `symbolicOrder` path also remains incomplete as a concept learner:
it stages a word/object concept row, repeats `compute_symbolic_reference` and
`promote_symbol_reference`, decodes the selected row, and advances its order.
It does not yet combine different rows into higher-order nouns, compare
adjacent-time noun states, or induce reusable diachronic verb meanings.
`VerbLayer` applying an operand is not evidence that such a meaning was
learned. The later work in section 14 must change that recurrence explicitly.

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

## 2B. Conceptual and event foundations

Keep perceptual/event mereology, conceptual identity/taxonomy, subjective
attention, and grammatical operations distinct. PartSpace and WholeSpace
percepts generate a spatiotemporally localized event partial order; its
transitive closure, not just its directly stored edges, is the mereology.
ConceptualSpace assigns persistent indices to hypothesized individuals and
kinds. Grammar selects and applies learned concepts; a lexical embedding is
not a substitute for a noun's identity or a verb's learned meaning.

The learned bridge from presentation to event support is:

```text
presents(subjective_where, subjective_when,
         concept_or_object, world_event_support)
```

World-event validity is separate from subjective attention. Transitive
consequences of relations supported on `A` and `B` hold on `A ∩ B`, not on
their union. No "tightest whole" truncates actual event parthood.

An order-0 concept is an indexed opening between a PartSpace construction
from `NOTHING` and a WholeSpace restriction from `EVERYTHING`:

```text
0 = NOTHING
1 = EVERYTHING
C_i = (identity_index_i, opening(P_i, W_i))

P_i < W_i     open: admits optional realizations or fillers
P_i = W_i     closed: determined at this description
P_i ≰ W_i     inconsistent: construction crossed its restriction
```

The opening is structural room in the event order, not a probability or a
three-valued parthood judgment. Type-level `fit()` describes possible
parthood; a localized compatible realization establishes actual parthood.
Neither a possible fit nor an open interval asserts necessary membership.
Zero retains its literal meaning, not a reinterpretation as uncertainty.

A car may retain its index while passengers, cargo, replacement components,
and containing wholes vary. During a passenger's presence in a house, the
passenger, their parts, the house, and its containing wholes participate in
the same transitive event hierarchy on that validity support. Stable identity
is a hypothesis over a trajectory of event realizations. If continuity or
integrity fails, close the denotation's validity interval; never silently
reuse a historical index. Creation, destruction, splitting, and merging are
later ontology work.

A book or text is also an indexed conceptual particle, possibly sharing
conceptual wholes with sibling texts beneath `EVERYTHING`. A title is
metadata, a stable document ID is discourse identity, and a book's physical
container is an event-local relation. The active text/domain concept scopes
reading without being encoded into subjective `.where` or `.when`.
Re-reading or moving a book need not change its identity.

Corpus loading must preserve corpus/document identity, sentence order, exact
character spans, and document boundaries. The split row is a lookup index
alongside independently checkable provenance. Packed rows must never carry
discourse state across a document/domain boundary. Objective coordinates
follow section 2, including separate source version and target time; the
older snapshot-as-`ObjectiveWhen` layout is compatibility data only.

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

The aggregate `TeachingLesson` is controller-private. The student receives a
separate public task/observation view containing only the permitted address,
mode, curriculum constraints, masks, and perception. Calling a field
`private_teacher_datum` does not make it private if the student can reach the
same object. Do not pass the whole lesson, Teacher, data loader, or privileged
spacetime source into the learned computation.

A queryable spacetime view provides the semantic operation
`view.what(address) -> evidence | unavailable`. Each view is bound to its
source namespace, access policy, source revision, and evidence cutoff. The
Teacher's privileged "what is there?" operation remains named `Data` in the
public controller API; `Teacher.What(where, when)` is only its legacy adapter.
Student perception/STM/LTM views cannot call that oracle or widen their own
access. An absent or forbidden observation is not permission to retrieve it
from another view. Internal thought addresses are explicitly tagged and
cannot resolve as corpus addresses.

The source address of a sentence and the grammatical position of an operand
within that sentence are separate types. Neither is the student's subjective
attention coordinate. Section 5.6 defines their use by the chooser, and the
companion design gives the high-level interfaces.

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

### 5.6 Chooser context, derivation, and sentence location

The chooser consumes a typed, student-visible `ChooserContext`, not Teacher's
lesson object or an opaque concatenation of unrestricted semantic state:

```text
ChooserContext:
    root request, active WhatFrame, parent role, partial construction
    mode and linguistic-stage production constraints
    active context/document, resolved target address, current reading cursor
    legal relative cursor actions and known sentence/span boundaries
    permitted perception references and availability masks
    current conceptual references and bound object/argument identities
    explicit STM/LTM query results, provenance, confidence, and as_of cutoff
    legal grammar rules, operand slots/spans, and partial derivation trace
    remaining budgets and seeded exploration state

Choice:
    grammatical query, reduction, or realization operation
    operand/result references and their grammatical slots/spans
    optional StudentQuery (relative cursor action or exact permitted address)
    rule/derivation log probability and supporting evidence references
```

This context exposes references and typed constraints. Any comparison or
transformation of subsymbolic payloads used to evaluate a choice is itself a
named, traced grammatical operation. A general semantic-to-control MLP or
unrecorded attention blend is not the implementation of this contract. The
exact lawful scoring mechanism remains an architectural choice to measure.

There are two decisions, which may share one reasoning episode:

1. **Source location:** a traced cursor/query operation selects a requested
   document/sentence/span. The source resolver validates the request against
   legal boundaries and reports the resolved address. Assigned lessons fix
   this root target; the chooser cannot move it to improve its score. In
   requested reading, selection happens before the requested perception is
   returned, and both requested and resolved addresses are recorded.
2. **Grammatical location and derivation:** legal rule/operand/slot choices
   determine how the conceptual answer is constructed and where each
   constituent belongs. On blank generation these are generated slots, not
   spans or word counts copied from a hidden target. Role selection precedes
   lexical realization; a generated output span is not a source address.

A query such as `next sentence` can therefore be selected from the active
document, cursor, discourse, parent task, and available history. After the
permitted response arrives, its public evidence can inform the construction.
Hidden future text, clean constituent labels, and Teacher's preferred
derivation must not participate in the earlier location decision. Once
resolved, all root outputs remain associated with that same target address.

Maintain one shared root grammar forest, one sampled root derivation, and one
top-down reconstruction. Constituent frames extend the same reasoning
episode; they do not each allocate a new sentence forest or render a sentence.
Resolve and memoize bounded evidence queries for reuse, retaining nonzero
support for every admissible derivation. Retaining a forest does not require
materializing every complete derivation; a packed representation is allowed.

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

### 6.3 Stochastic choice and credit assignment

Retain the complete admissible forest and sample one derivation rather than
reconstructing alternatives. The initial exploration target from the earlier
plan is approximately `0.80 / 0.15 / 0.04 / 0.01` probability mass for the
best, second, third, and remaining derivations. With fewer alternatives,
transfer unused tail mass to the lowest-ranked available choice. These are
initialization/ablation targets, not fixed ranks that prevent learning.
Anneal toward exploitation while retaining at least one percent total tail
mass when alternatives exist; do not permanently prune admissible candidates.

Stochastic exploration requires a learning signal. For the initial discrete
chooser use one sampled task loss and a detached, pre-update EMA baseline:

```text
advantage = stop_gradient(L_task(sampled_derivation) - EMA_loss_baseline)
L_chooser = advantage * log p(sampled_derivation) - 0.01 * entropy(p)
```

`L_task` excludes `L_chooser`; the baseline for this sample must not depend
on the sampled action. `p` is the actual sampling distribution, including any
exploration mixture. The score-function term supplies discrete-choice credit;
ordinary differentiable task losses still train the executed operations.
Computing log probabilities and entropy over a packed forest must not execute
a second reconstruction. The `0.01` entropy coefficient above is the chooser
entropy entry in section 6.2, not a second independently added entropy loss.
The chooser auxiliary scale is an initial ablation value, not ontology.

Teacher also retains required codebook commitment and boundedness terms. An
initial commitment scale is `0.25`; grounding/re-anchoring remains `0.00` in
this iteration. Later verb-transition and sparse-expert penalties belong to
section 15, not an additional independent next-sentence predictor. Legacy
predictor and forced grammar-rerun weights remain zero on the canonical path.

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

Implement the addressed-reading iteration as gated milestones, not as one
indivisible change. Its full scope includes controlled recall and future-state
prediction; its first milestone is smaller. Autonomous truth learning, full
verb induction, and metaphor remain deferred.

| Milestone | Work | Exit gate |
|---|---|---|
| A: addressed clean reading | Steps 1-4 plus noun-stage metadata from step 9 | Private/public separation, present compatibility, address matching, clean B24 throughput |
| B: contextual and controlled blank reading | Steps 5-8 plus noun-only production enforcement | Traced stack and chooser, constrained requests, causal light denoising, recall and future-state baselines |
| C: bounded thinking evaluation | Step 10 after milestone B | Provisional records, bounded termination, no autonomous truth writes |

Section 11 is the completion gate for the addressed-reading iteration
(milestones A and B). Milestone C and sections 14-17 are later work, not hidden
requirements for declaring milestone A complete. Light degradation in step 4
is a diagnostic to prove address use; the full mode mixture remains gated.

### Step 1: revise the Teacher contract

In `Teacher.py`:

- add `TeachingMode`, `SourceVersion`, `ObjectiveAddress`, `StudentQuery`, and
  the expanded `TeachingLesson`;
- expose the exact privileged lookup as `Teacher.Data(address)`;
- keep the old `Teacher.What(where, when)` only as a deprecated controller-side
  compatibility alias, never as the student's prediction API;
- move snapshot validity out of objective target time;
- retain compatibility adapters for v1 checkpoints/configuration;
- add deterministic mode scheduling;
- keep private clean targets inaccessible to the student.

Fix the existing temporary-context lifecycle during this migration: restore
source/snapshot identity together with context on normal and exceptional exit,
and isolate nested runtime lessons and staged cursors. Preserve validity of
previous corpus addresses. Separate public task data from the current
`ReconstructionLesson.clean_input` / `clean_what` fields before blank lessons
can run; a model-owned `teacher` reference is not a capability boundary.

In `data.py`:

- expose `observe(address, policy)` separately from the exact lookup used by
  `Teacher.Data(address)`;
- preserve document boundaries and exact source provenance;
- add legal relative-cursor resolution;
- never let a request cross context/document boundaries silently.

Missing or inconsistent provenance must fail closed, not fabricate a document,
sentence, or span from a fallback row. Known inline/runtime sources must
construct their explicit provenance at ingestion. Compatibility adapters may
translate a complete v1 address but must not invent absent coordinates.

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
- Temporary and nested contexts restore snapshot/version, lesson, and cursor
  state on normal and exceptional exits; earlier source addresses remain valid.
- Absent provenance is rejected; supported inline/runtime input has explicitly
  constructed addresses, not silently invented lookup fallbacks.
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
- Holding a forest fixed, chooser training raises the probability of a
  lower-task-loss derivation without evaluating a second derivation per lesson.
- Assigned targets cannot be changed by a chooser; requested cursor decisions
  precede receipt of the requested perception and retain resolution provenance.
- Source addresses, grammatical slots/spans, and subjective attention remain
  distinct, including on blank lessons without target-derived output lengths.

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

## 14. Preserved roadmap: noun and verb concept formation

This and the following sections preserve the earlier plan's requirements.
They follow addressed reading and stable noun learning, rather than delaying
all degraded lessons until full verb induction is complete. The initial
`predict_state` operation is not itself a learned reusable verb inventory.

### 14.1 Synchronic and diachronic concept order

The symbolic-order recurrence must eventually learn concepts from percepts
and concepts of the preceding order, in two explicitly typed forms:

```text
N^(k+1)_t = synchronic_lift(P_t, W_t, C^k_t, ...)
V^(k+1)_(t1->t2) = diachronic_lift(C^k_t1, C^k_t2)

V^1 = [part = C^0_before, whole = C^0_after, kind = temporal,
       from_event = t1, to_event = t2]
```

Same-time combinations form higher-order nouns: composite objects and
configurations. Cross-time combinations form verbs. Ordered before/after
roles retain direction even when the sparse activation reader is untyped.
The event-support coordinates above are not assignments to subjective
`.when`; provenance records both event order and presentation time.

Verb learning should satisfy:

```text
verb(initial noun concept) ~= final noun concept
L_VP = E[distance(T_VP(C_NP,t), C_NP,t+1)]
```

Repeated `grow(seed_i) -> sprout_i` examples across different tracked
individuals must improve one reusable temporal concept, not memorize a
separate sentence-pair identity. Training evidence must identify the initial
object state, action word or anonymous action, resulting state, and continuity
of the affected individual. Begin with unary transformations; later extend
to role-bound argument tuples.

Not every later change is explained by the intervening VP. As initial
admission thresholds, require at least four eligible transitions across at
least two tracked noun identities, and retain a provisional VP only when its
EMA transition error is at least five percent below the identity/no-verb
baseline. Confirm improvement on held-out eligible transitions. These are
ablation thresholds, not guarantees of causal identification from text.

### 14.2 Meaning learning versus grammatical application

A surface verb's MetaSymbol binds to one or more learned diachronic concept
rows. The grammatical role, argument bindings, and context select the intended
sense; its atom is passed to the shared executor. `VerbLayer` applies a verb
but does not mint its meaning. Noun selection likewise uses learned concepts,
not the surface word row as the object's meaning.

The diachronic event record retains references to its verb and constituents
for grammatical reconstruction. The causal transformation may be forward-only
and many-to-one: grammatical decomposability does not imply causal
invertibility. Inferring a past object from a result is an abductive
`model.what` using explicit memory, not a required algebraic inverse.

### 14.3 Row-local transition evidence

Retain completed two-slot sentences in structured NP-VP form until the next
eligible state of the same tracked noun is observed:

```text
E_t = (NP_t, VP_t, discourse_id, identity_binding,
       subjective_when_t, objective_event_support)
E_next.NP = next compatible state of that tracked individual
eligible_transition = (E_t.NP, E_t.VP, E_next.NP)
```

This bounded row-local LTM cache is evidence transport, not the verb
representation or an already implemented v1 guarantee. Preserve sparse NP/VP
row IDs, signed activations, discourse identity, time/provenance, and required
role bindings; do not retain the entire concept inventory, an undifferentiated
sentence activation, or a previous autograd graph.

- Commit only detached student-produced concept states, never private Teacher
  targets; keep persistent/global truth admission separate.
- Update sequentially at each intermediate packed-sentence boundary and reset
  on document/domain boundaries.
- Do not train a transition for a first sentence, across different domains,
  or between unsupported object identities.
- Compare the affected noun states, not whole-sentence activation differences.
- Keep ambiguous or unmatched records provisional; adjacency alone is not
  evidence that two sentences describe the same individual or one verb.

Concept learning belongs inside the symbolic-order process. The compiled
loop may compute candidates, activations, and losses, then queue structural
admission/relation growth for the safe eager/reset boundary. Deferring those
mutations does not move meaning creation into grammar. Implement genuine
same-time and temporal combination rather than describing repeated promotion
of one selected row as concept induction.

## 15. Preserved roadmap: shared sparse verb executor

Do not allocate a private dense invertible matrix per English verb. Use a
shared bank of low-rank transformation experts with sparse gates and a
bias/control code supplied by the learned diachronic concept:

```text
z_after = z_before
        + sum(j in top_k(g_V)) g_V[j] U_j phi(R_j z_before)
        + B b_V
```

`U_j`, `R_j`, and `B` are shared. Begin with 32 rank-32 experts, four active
per VP occurrence. Condition gates on the VP concept and argument roles so
different senses can select different mixtures. The residual path preserves
unaffected structure; the selected sparse subspace carries the change.
Existing sparse spectral gain may be one expert, but is not sufficient by
itself for general state and relation transformations.

If quality saturates, first ablate expert count, rank, and two-step
composition. Vocabulary and verb count should primarily grow rows and sparse
relations, not private dense operators. A learned past reconstructor is
optional; forward transition reconstruction is normative. Growth, movement,
possession, integrity, and activation are changes in `what`, not meanings
encoded directly in subjective `.when`.

Add verb-transition weight `0.25` and sparse expert-gate penalty `1e-4` as
initial ablation values, reporting them separately. Ramp the transition
weight from zero to `0.25` over the first ten percent of the verb-learning
phase. Do not conflate this later reusable-verb objective with the earlier
`PREDICT` future-state alignment term, and do not silently double-count one
transition under both losses. Keep one grammatical reconstruction per lesson.

## 16. Preserved roadmap: dimensions and grounding

PartSpace, WholeSpace, and ConceptualSpace widths are independent. A concept
is defined by percept references and their activations, not by a
coordinatewise PS-to-CS identity handoff:

```text
{PS percept references, WS percept references}
    -> ConceptualSpace identity row and atom
```

Constituent richness grows sparse edges and activations, not a required
conceptual width. Equal-width 1024 configurations are experimental choices,
not ontological constraints. Preserve the indexed activation/dictionary-decode
interface; do not introduce a dense PS-to-CS projection merely because widths
differ. Explicit role adapters inside `predict_state` are a separate,
traceable interface and do not redefine concept identity.

After addressed reading and temporal learning work, ablate ConceptualSpace
width 512 then 256, initially retaining PartSpace width 1024 and choosing
WholeSpace width independently for its property inventory. Explore shared
operator rank 32-64 and approximately 4-8 active primitives. Select widths
using reconstruction, grammar, transition quality, memory, and throughput,
not parameter count alone.

Perceptual part structure provides a current extension of an object; its
conceptual opening and relations provide a learned intension. Their sparse
reference interface permits drift. Grounding/re-anchoring is deferred: later
replay high-trust perceptual exemplars, measure PS/WS-to-CS reference stability,
and repair unsupported concepts without collapsing the spaces' independent
dimensions. Preserve historical indices and event validity during repair.

## 17. Acceptance gates for the preserved roadmap

These gates apply when the corresponding deferred capability is implemented;
they do not relabel the clean v1 benchmark as proof of concept induction.

### Ontology and mereology

- `0` and `1` remain `NOTHING` and `EVERYTHING`; open, closed, and inconsistent
  conceptual openings are distinguishable.
- Optional fillers can enter and leave without changing a tracked index.
- Actual parthood composes transitively only over intersecting event support.
  Direct generating links and their closure give the same event order.
- Subjective attention, conceptual identity, and objective event location
  cannot be substituted for one another.

### Concepts, verbs, and memory

- Order-0 concepts derive from PS/WS references at independent native widths.
- Symbolic order 1 can learn both a same-time noun and an ordered temporal
  concept; reversing before/after produces a distinct temporal relation.
- Repeated seed-to-sprout examples reuse and improve one `grow` concept.
- A retained VP trains only upon an eligible later NP state of the same
  tracked identity; rows and documents cannot contaminate one another.
- An admitted VP improves expected held-out transition error over identity
  copying. Applying the selected verb improves final-noun reconstruction.
- Grammar selects a role-conditioned learned temporal meaning rather than a
  raw lexical row. Distinct VPs can use distinct sparse shared-expert mixtures.
- No verb owns a private dense invertible matrix. Constituents remain
  grammatically recoverable even when the causal transformation is many-to-one.
- Historical truth distinguishes assertion time from represented event support;
  private targets and unverified predictions never become student truth by
  default.

### Quality and performance

- Report clean/degraded/blank reconstruction, grammar validity, transition
  quality, verb-sense selection, and sentences/s separately.
- Use the matched B24 baseline and 15% clean-regression gate from section 10;
  establish and report B28 separately rather than comparing B28 against B24.
- Attribute concept-admission, transition-cache, and verb-expert overhead
  separately. Width ablations must report quality as well as speed and memory.
