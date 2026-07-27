# Teaching modes, contextual grammar, and the next iteration

> **Status:** implementation specification and maintainer handoff,
> 2026-07-27.
>
> **Immediate scope:** teach the student the objective coordinate system,
> make reading an explicit query/response process, and replace the retired
> next-sentence predictor with one context-conditioned `What` mechanism.
> Thinking is specified as the inference-time continuation of that mechanism,
> but autonomous truth-store admission remains deferred.

## 1. Current boundary

Teacher v1 is a clean reconstruction and performance baseline, not active
reading.

Today:

1. The data cursor selects a passage.
2. `Teacher` resolves its objective source address and clean `what`.
3. The complete passage is presented to the student.
4. The existing perceptual and grammatical path reconstructs it.
5. Teacher owns the reconstruction loss and scoring boundary.

The objective query is exposed as CPU categorical tensors, but no trainable
student module consumes it. The student therefore does not yet:

- learn the objective coordinate system;
- choose or request a `where`/`when`;
- retrieve the perception at a requested address;
- depend on the address when perception is degraded or blank.

The accepted clean B24 benchmark is 41.989 complete sentences/s. That result
is an upper bound for the current computational core, not the throughput of
the eventual addressed learner. Most of the improvement over the 6.803
sentences/s baseline comes from removing the independent next-sentence
predictor and its prefix-wide peer computation, not from Teacher bookkeeping.

The next iteration should spend some capacity and some lessons learning
objective coordinates. A modest throughput regression is expected and
acceptable if ablations prove that those coordinates causally improve
degraded, recall, and prediction lessons.

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

## 3. Query, observation, and scoring contracts

Active reading requires three logically distinct actors even if Teacher v1
implements two of them in one Python object:

```text
Student.Query(context) -> ObjectiveAddress
SourceOracle.Observe(ObjectiveAddress, observation_policy) -> Perception
Teacher.Score(ObjectiveAddress, Student.What, private clean What) -> loss
```

- **Student** selects an address or accepts an assigned address.
- **SourceOracle** returns only the perception permitted by the lesson mode.
- **Teacher** retains the clean target privately and scores the result.

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
    private_clean_what
    student_what
    address_targets
    grammar_trace_if_permitted
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

Prediction is blank-perception reconstruction at a next or later objective
address. It is not a separate faculty and must not restore the retired
next-sentence predictor:

```text
query = next objective address
perception = blank
context = history + LTM + current concepts + discourse
student.What(query, context) -> predicted what
Teacher.Score(predicted what, future source truth)
```

The target may be known to the offline Teacher, but it must remain inaccessible
to the student until scoring. Any prediction stored before verification is a
claim with target time and provenance, not truth.

### 4.7 `THINK`

Thinking is the inference-time continuation of addressed reconstruction when
no external perception is supplied and the student selects its own query.

A thought need not first become a surface sentence. The canonical internal
record is conceptual:

```text
ThoughtRecord:
    requested objective or internal address
    conceptual what
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
query policy
  -> addressed LTM/perception retrieval
  -> cached contextual state
  -> one grammatical/conceptual What construction
  -> provisional ThoughtRecord
  -> next query or stop
```

Use explicit step, wall-clock, novelty, and confidence budgets so inference
cannot recurse indefinitely. Multiple grammatical derivations and metaphor
may later branch this loop, but the next iteration should retain one sampled
derivation per thought.

## 5. Replacement for the next-sentence predictor

The benchmark shows that the independent predictor was the dominant long leg.
Do not restore it under another name. Replace its useful function with a
contextual state computed once per sentence or thought and reused by the
existing grammar chooser.

### 5.1 Context encoder

Compute:

```text
h_context = ContextEncoder(
    perception summary,
    current ConceptualSpace state,
    top-k LTM retrieval,
    discourse/history state,
    objective address embedding,
    source-version embedding,
    teaching/operating mode)
```

Recommended starting geometry:

- objective address embedding: 128 dimensions;
- cached context state: 256 dimensions;
- two-layer context MLP with 512 hidden units;
- top eight detached LTM records;
- one context computation per sentence/thought, not per candidate rule.

These are initial ablation points. Keep the context state narrower than the
ConceptualSpace carrier unless quality shows a need for parity.

### 5.2 Context-conditioned grammar

The grammar chooser should not become a token-generating LLM. It continues to
score explicit symbolic alternatives in the retained derivation forest:

```text
score(rule_or_leaf, chart_state, h_context)
    = base_score(rule_or_leaf, chart_state)
    + low_rank_bilinear(rule_or_leaf, h_context)
    + mode_bias
```

Start with rank-32 bilinear or FiLM-style conditioning. Cache the projected
context once and reuse it at every chart decision. Preserve:

- one forest construction;
- one stochastic derivation selection;
- one top-down reconstruction;
- nonzero probability for second-, third-, and tail-ranked derivations;
- no independent predictive parse.

Historical and semantic computation belongs primarily in `ContextEncoder` and
LTM retrieval. Grammar consumes that state to choose syntax, concepts, and
operators; it should not repeatedly rediscover the entire history at each
node.

### 5.3 Query policy

Add a small head over `h_context` that chooses relative cursor actions and,
later, associative retrieval targets. This head replaces the control function
of the old next-sentence predictor:

```text
QueryPolicy(h_context) -> next objective where/when action
```

The result determines what the unified `What` path tries to reconstruct. It
does not generate the sentence itself.

## 6. Address representation and losses

### 6.1 Encoder

Use separate components:

- learned categorical embeddings for context, corpus, source version, split,
  document identity, and external aliases;
- ordered positional encodings for document order, sentence order, character
  span, observation index, and relative target offset;
- a mask for absent optional fields;
- a small fusion MLP producing one address atom.

Raw stable hash magnitude has no semantic meaning. Hashes may key embedding
rows but must never be interpreted as ordered scalars.

### 6.2 Shared-pass objectives

Initial scales:

```text
surface What reconstruction                 1.00
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
the former next-sentence learning signal. There is no additional
next-sentence loss.

## 7. Proposed curriculum

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
- move snapshot validity out of objective target time;
- retain compatibility adapters for v1 checkpoints/configuration;
- add deterministic mode scheduling;
- keep private clean targets inaccessible to the student.

In `data.py`:

- expose `observe(address, policy)` separately from `truth(address)`;
- preserve document boundaries and exact source provenance;
- add legal relative-cursor resolution;
- never let a request cross context/document boundaries silently.

### Step 2: add the objective-address encoder

Create a focused module such as `Addressing.py` rather than expanding
`Models.py` further. It should contain:

- categorical identity tables;
- ordered position encodings;
- address fusion;
- in-batch contrastive matching;
- relative cursor/query heads.

Wire one address atom into the student boundary without using subjective event
bands. Begin with `READ_ASSIGNED` and `LOCATE` only.

### Step 3: prove causal address use

Before degradation:

- round-trip every exact address through `Teacher.What`;
- predict/match held-out addresses above hard-negative chance;
- swap addresses within a batch and verify address-matching loss increases;
- verify subjective `.where/.when` tensors remain byte-identical.

Then introduce light degradation and require shuffled addresses to worsen
surface reconstruction. If reconstruction is unaffected, stop: the address
encoder is being ignored.

### Step 4: add one cached contextual state

Add `ContextEncoder` once per sentence boundary. Retrieve only bounded,
detached student-visible history. Feed its cached projection to:

- the grammar chooser;
- the query-policy head;
- concept/lexical selection where an existing explicit seam exists.

Do not feed Teacher's private `what`, future truth, or clean grammar labels.
Do not invoke the context MLP once per forest candidate.

### Step 5: enable constrained student requests

Implement `READ_REQUESTED` over the small relative action set. Teacher scores
the requested versus intended cursor action and reconstructs the returned
passage. Keep normal corpus order as the dominant policy initially so query
exploration cannot destroy discourse continuity.

### Step 6: add the first partial and blank modes

Enable:

1. light `DENOISE`;
2. past-address `RECALL`;
3. next-address `PREDICT`.

Each uses the same single grammar forest and surface reconstruction loss.
Only after all three pass leakage and throughput gates should the full
curriculum be enabled.

### Step 7: specify, but defer, autonomous thinking writes

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

- Exact source addresses return exact passages.
- Relative cursor actions resolve deterministically.
- Invalid context, split, version, or span combinations fail closed.
- Packed-row prefetch preserves the address associated with each sentence.
- Document and domain resets prevent cross-document cursor leakage.

### Learning and leakage

- Correct addresses beat nearby hard negatives.
- Shuffling addresses worsens degraded reconstruction.
- Blank lessons cannot access clean input through LTM, caches, traces, or
  Teacher fields.
- Future truth is unavailable to the student before prediction scoring.
- Removing history worsens `PREDICT` relative to the full contextual model.
- Removing the address worsens addressed `RECALL` and `PREDICT`.

### Grammar and performance

- The grammar forest is constructed once per lesson.
- One derivation is sampled and reconstructed; no predictor-triggered second
  parse occurs.
- Context is computed once per sentence/thought.
- Second-, third-, and tail-ranked derivations retain nonzero support.
- Clean reconstruction remains finite through address and context ablations.

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
- report B28 separately after confirming memory headroom;
- report lessons/s, complete target sentences/s, and words/s because locate
  or query lessons may not be equivalent to one reconstructed sentence;
- report clean, degraded, recall, and prediction modes separately and as a
  fixed mixture.

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

1. A student address encoder consumes objective coordinates without touching
   subjective bands.
2. The student can match a passage to its objective address against hard
   negatives.
3. A constrained student query advances the reading cursor and retrieves the
   requested passage.
4. Lightly degraded reconstruction causally benefits from the correct address.
5. One cached historical/semantic context conditions the grammar chooser.
6. Next-sentence behavior is expressed as blank-perception reconstruction at
   the next address, with no independent predictor or second parse.
7. Provisional thoughts and predictions remain outside historical truth.
8. Correctness tests pass and clean throughput remains above the 15%
   regression gate.

## 12. Deferred work

The following remain intentionally outside the next iteration:

- persistent admission of student-generated truth;
- reward from unverified self-generated thoughts;
- unrestricted associative query over all LTM;
- multiple simultaneous parses for metaphor;
- full diachronic verb induction and sparse verb experts;
- creation/destruction and identity-split ontology;
- grounding/re-anchoring repair;
- replacing explicit objective addresses with purely in-sentence cues.

Those features should build on a measured addressed `What` loop rather than
being introduced before the student can demonstrably read, locate, recall,
and predict through one shared mechanism.
