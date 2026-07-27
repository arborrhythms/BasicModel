# Teacher reconstruction, event concepts, and learned verbs

> **Status:** partially implemented, 2026-07-27. The clean Teacher boundary,
> loss ownership, exact objective source address, legacy-predictor removal,
> and clean endurance benchmark have landed. The objective query is exposed
> but is not yet consumed by a trainable student encoder. This plan does not
> describe the serial concept loop as complete: in particular, the current
> `symbolicOrder` loop promotes and decodes an already-selected concept row
> but does not yet learn synchronic or diachronic concepts. The operational
> teaching modes and immediate implementation handoff are specified in
> `doc/specs/2026-07-27-teaching-modes-and-next-iteration.md`.
>
> **Directive:** replace the collection of independent prediction and learning
> losses with a Teacher-owned reconstruction objective. Concepts are learned
> from PartSpace and WholeSpace percepts during the symbolic-order process.
> The same process must learn verbs as reusable maps from an object's earlier
> concept to its later concept. Grammar selects and applies those learned
> concepts; grammar does not create their meanings.

## 1. Governing distinctions

The design depends on keeping four structures distinct:

1. **Subjective attention.** An event's `.where` and `.when` locate a
   perception, memory, or imagination in the individual's attentional
   workspace. They do not encode an object's world location or identity.
2. **Perceptual/event mereology.** PartSpace and WholeSpace percepts form a
   spatiotemporally localized event partial order. Its transitive closure is
   the actual mereology; direct links are only a compact generating graph.
3. **Conceptual identity and taxonomy.** ConceptualSpace gives a persistent
   index to a hypothesized individual or kind and learns taxonomic relations
   among concepts. The index tracks an individual across changing event
   realizations but does not guarantee that its denotation remains valid.
4. **Grammar.** Grammar chooses word references and applies operators to
   concepts. It is downstream of concept learning and must not substitute a
   lexical embedding for the learned noun or verb concept.

The bridge from subjective experience to the event mereology is a learned
reference relation:

```text
presents(subjective_where, subjective_when,
         concept_or_object, world_event_support)
```

World-event support and validity intervals are separate from subjective
`.where`/`.when`. Transitivity composes only over compatible world-event
support: if two relations are valid on supports `A` and `B`, their transitive
consequence is valid on `A ∩ B`, not their union.

The top and bottom of the mereological carrier retain their literal meanings:

```text
0 = NOTHING
1 = EVERYTHING
```

Intermediate structure is not a reinterpretation of zero as uncertainty.

## 2. Concepts as openings between the towers

An order-0 concept is an indexed opening between a construction from
PartSpace and a restriction from WholeSpace:

```text
C_i = (identity_index_i, opening(P_i, W_i))
```

where `P_i` is built upward from `NOTHING`, `W_i` is analyzed downward from
`EVERYTHING`, and a valid opening satisfies:

```text
P_i < W_i     open: admits optional realizations or fillers
P_i = W_i     closed: completely determined at this description
P_i ≰ W_i     inconsistent: construction crossed its whole restriction
```

The gap is structural room in the event order, not a probability or
three-valued parthood judgement. A candidate fits an opening when combining
its part and whole support leaves a valid order interval. Type-level PS/WS
codes therefore participate in a `fit()` order of **possible** parthood.
Necessity is not required merely because a type-level fit exists.

At a particular event, a compatible fit may be realized as actual parthood.
Actual event parthood is transitive. Possibility is represented by the
type-level `fit()` order; actuality comes from a localized realization.

Openings allow a car to retain its conceptual index while passengers, cargo,
replacement components, and surrounding wholes vary. A wheel may contribute
persistently to the car's lower construction, while a passenger temporarily
fills available structure. The passenger's presence in a house genuinely
places the passenger, their parts, the house, and the house's containing
wholes in one transitive event hierarchy for the relation's validity period.
No "tightest whole" limits that transitive closure.

Stable identity is consequently a hypothesis over a trajectory:

```text
concept index -> event realization at t1
              -> event realization at t2
              -> ...
```

If continuity or integrity later fails, close the denotation's validity
interval rather than reusing its index. Creation, destruction, splitting, and
merging are deferred from the first Teacher implementation, but historical
indices must never be silently reassigned.

## 3. Texts, domains, and reading

A text or book is an indexed conceptual particle and may share conceptual
wholes with sibling texts, ultimately fitting under `EVERYTHING`. Its current
physical containers and locations are localized event relations, not a
permanent identity path. A title or corpus name is metadata; a stable document
identifier is the discourse identity.

Reading supplies:

```text
reading mode
active text/domain concept
objective source address
subjective attention .where/.when
Perception
STM/LTM
```

The active text concept scopes what is being read. Subjective `.where/.when`
select the current presentation within that scope through `presents`; they do
not contain the text identifier. Re-reading, scrolling, and moving a physical
book may reuse subjective locations, so `.when` and the active domain remain
part of the query context.

FineWeb loading must preserve document identity, corpus identity, sentence
order, and document boundaries. Packed B28 rows may contain consecutive
sentences from one document but must never carry discourse state across a
document boundary.

The Teacher's objective address is separate from subjective attention:

```text
objective_where =
    (corpus_snapshot, split, document, sentence, character_span)
objective_when =
    (snapshot_or_revision_identity, optional_source_time)
```

The ordered split row is retained as a direct lookup index alongside the
document/sentence/span provenance. A DOI may replace or alias the document
coordinate when supplied, but a DOI and date alone do not identify a sentence,
and FineWeb's shard schema provides neither. The local shard SHA-256, corpus
release, ordered document index, sentence index, and character span therefore
form the initial lossless address.

These objective coordinates are categorical source-query data that the student
must learn to use. They are never written into, substituted for, or derived
from the model's existing subjective `.where` and `.when` bands.

## 4. Teacher-owned reconstruction

The Teacher's truth query is:

```text
Teacher.What(objective_where, objective_when) -> clean what
```

The student's public cognitive query receives that objective address as
external conditioning while retaining its own subjective attention state:

```text
Model.What(objective_where, objective_when,
           perception, ltm, discourse_context)
```

The returned `what` is an imagination in the broad sense: reconstruction of
the requested present, past, or future event. Prediction is not a separate
faculty. A future query is reconstruction with unavailable current perception
and a later objective target time. The student's subjective `.when` still
orders its own presentation and computation; Teacher does not set it.

Introduce a `Teacher` that owns lesson construction, target access, masking,
loss composition, and reporting. The student may read only its presented
Perception, STM/LTM, discourse state, and query signals. Teacher-only clean
content must never enter student-visible LTM, WholeSpace state, sentence
caches, or grammar inputs.

The canonical lesson is present-input reconstruction:

1. Contextualize clean text as the current reading event.
2. Resolve and expose the separate objective source address for that event.
3. Run the student's bottom-up perceptual and top-down grammatical processes.
4. Ask the student to reconstruct the clean `what` for that objective address.
5. Let the Teacher compare the reconstruction with the clean text and assign
   the complete loss.

The first implementation supplies exact objective coordinates as padded
integer tensors on a boundary separate from the event carrier. Integer
magnitudes are not semantic; a student-side address encoder must treat corpus,
snapshot, split, and external identifiers categorically and positions as
ordered coordinates. Clean reconstruction is benchmarked before that encoder
is allowed to affect the numerical path. After objective-coordinate
conditioning is stable, the curriculum may progressively replace explicit
addresses with relative or in-sentence reconstruction cues. Teacher retains
the resolved address privately so it always scores the same requested event.

Use a mixed degradation curriculum:

```text
first 10%:   complete present input
middle 70%:  continuously increase degradation
final 20%:   maximum configured degradation
all phases:  reserve 10% clean anchor lessons
```

Degradation masks content, lexical identity, and leaked constituent labels,
while retaining the active discourse concept, reading mode, objective query,
prior student memory, and grammar state. Fully blank perception therefore asks
the student to reconstruct the present from history, address, and context.

Teacher loss includes:

- clean `what` reconstruction;
- objective source-address consistency;
- grammar validity and, on clean lessons only, an available derivation trace;
- ConceptualSpace/codebook commitment and required boundedness terms;
- temporal-concept and verb-application reconstruction once enabled.

Degraded and blank lessons use one surface reconstruction score. They must not
force a second independent grammar evaluation. The grammar chooser should
retain the entire admissible derivation forest, rank it once, and stochastically
choose one derivation for the single top-down reconstruction. Use a
temperature-controlled categorical or straight-through Gumbel sample with an
initial probability mass of approximately:

```text
best derivation:          0.80
second-best derivation:   0.15
third-best derivation:    0.04
remaining forest:         0.01, distributed with nonzero support
```

If fewer than four derivations exist, transfer the unused tail mass to the
lowest ranked available alternative. Sampling one derivation per lesson is
sufficient stochastic exploration provided no admissible candidate is
permanently pruned before sampling and the Teacher's reconstruction reward
trains the chooser. Stochasticity alone supplies exploration, not credit
assignment. For v1, use a one-sample advantage estimator:

```text
advantage = stop_gradient(L_task(sampled_derivation) - EMA_loss_baseline)
L_chooser = advantage * log p(sampled_derivation) - 0.01 * entropy(p)
```

Minimizing this term raises the probability of choices whose Teacher loss is
below the baseline and lowers the probability of worse choices. Here `L_task`
is the Teacher's scored task loss before adding `L_chooser`. Computing
`p` and its entropy over the retained forest does not run the other
derivations. This is a noisy but unbiased estimate of expected discrete-choice
loss and avoids the cost of a second parse. Once clean reconstruction
stabilizes, anneal toward exploitation while retaining at least one percent
total tail probability. Legacy predictor losses and forced grammatical reruns
remain disabled in the canonical Teacher configuration.

Use the following initial loss scales; report every component separately:

```text
surface reconstruction:       1.00
grammar/chooser auxiliary:    0.10
codebook commitment:          0.25
verb transition:              0.25
sparse expert-gate penalty:   1e-4
grounding/re-anchoring:        0.00 in v1
```

Ramp the verb-transition weight from zero to `0.25` over the first ten percent
of the verb-learning phase so an untrained executor cannot destabilize clean
reading. Treat these values as initial ablation points, not ontology.

The truth store remains historical. Predictions may be stored with distinct
`asserted_at` and represented/target-event validity, but future Teacher truth
must not be written into student-visible memory before it is evaluated. The
first implementation uses detached, row-local LTM transition records only;
policy for committing student reconstructions or predictions to persistent
global LTM is deferred to `todo.md`.

### 4.1 Deferred grounding and drift

The part structure of percepts can serve as the current extension of a mental
object, while the opening and relations in ConceptualSpace serve as its
learned intension. Their indexed, sparse interface is deliberately loose:
extension and intension can drift apart. Grounding and re-anchoring are not
part of the first Teacher implementation. A later phase should replay
high-trust perceptual exemplars, measure PS/WS-to-CS reference stability, and
repair unsupported concepts without collapsing the independent dimensions of
the spaces.

## 5. Concept order: nouns and verbs

The symbolic-order process learns concepts on top of percepts and concepts
from the preceding order. It has two required forms.

### 5.1 Synchronic concepts: higher-order nouns

Concepts combined within one event time form higher-order nouns:

```text
N^(k+1)_t = synchronic_lift(P_t, W_t, C^k_t, ...)
```

These represent composite objects, configurations, and other same-time
concepts.

### 5.2 Diachronic concepts: verbs

Concepts combined across adjacent event times form verbs:

```text
V^(k+1)_(t1->t2) = diachronic_lift(C^k_t1, C^k_t2)
```

At order 1, the temporal relation may be recorded using the existing ordered
concept roles:

```text
V^1 = [part = C^0_before,
       whole = C^0_after,
       kind = temporal,
       from_when = t1,
       to_when = t2]
```

The ordered record preserves causal direction even if the sparse activation
reader remains untyped. Same-time and cross-time concepts must have explicit
kind metadata so grammar and diagnostics cannot confuse them.

The essential verb-learning objective is deliberately simple:

```text
verb(initial noun concept) ~= final noun concept
```

LTM retains two-slot sentences in structured NP-VP form until the next
eligible occurrence of the tracked noun is available:

```text
E_t = (NP_t, VP_t, discourse_id, identity_binding, subjective_when_t)
E_(t+1).NP = next contextual state of the same tracked noun
```

The VP in `E_t` is trained after the following noun state is observed. Across
eligible examples, it should explain the average transition between successive
NP positions:

```text
L_VP = E[distance(T_VP(C_NP,t), C_NP,t+1)]
```

Repeated examples such as:

```text
grow(seed_1) -> sprout_1
grow(seed_2) -> sprout_2
grow(seed_3) -> sprout_3
```

must converge on one reusable temporal concept rather than memorizing each
sentence pair. The corpus must provide enough evidence to identify the
initial concept, action word or anonymous action, resulting concept, and
continuity of the affected particle.

Individual transitions may contain changes that the VP does not explain.
Learning is therefore statistical: an admitted VP must improve the expected
transition over the identity/no-verb baseline, rather than reproduce every
successive NP exactly. Initially promote a provisional temporal concept after
at least four eligible transitions spanning at least two tracked noun
identities, and retain it only when its exponential-moving-average transition
error is at least five percent below that baseline.

Start with unary object transformations. Multi-argument verbs later generalize
the same rule to role-bound tuples of concepts.

### 5.3 Learning versus grammatical application

Verb meaning is learned during the `symbolicOrder` concept process. A surface
verb's MetaSymbol binds to one or more learned diachronic concept rows.
Grammatical derivation uses its role and context to choose the intended row,
then passes that row's atom to the shared verb executor.

`VerbLayer` applies a learned verb; it does not mint or define the verb.
Likewise, grammar selects noun concepts learned from same-time percepts rather
than treating the surface word row itself as the object meaning.

Grammatical decomposability and causal invertibility are separate. The
diachronic sentence/event concept retains references to the verb and its
constituents so grammar can reconstruct them. The causal transform itself may
be forward-only and many-to-one. Exact inversion is optional; reconstructing a
past object from only its result is an abductive `What` query using LTM, not a
required algebraic inverse.

## 6. Sentence state and symbolic-loop integration

Maintain a small, row-local LTM cache of completed two-slot sentences as
structured NP-VP records. Do not immediately reduce those records to a single
undifferentiated sentence activation. When a later compatible NP state
arrives, close the prior record as:

```text
(previous NP, intervening VP, next NP)
```

and let the temporal branch of the `symbolicOrder` loop update or propose the
VP concept. Thus verbs are learned in the same concept-learning recurrence as
nouns, but their evidence crosses a sentence boundary. Grammar later applies
the learned VP concept; grammatical application is not the learning event.

The cache is an implementation transport, not the verb representation. It
should retain sparse NP and VP concept row IDs, signed activations, subjective
sentence time, discourse identity, and necessary role/identity bindings. It
must not retain the full concept inventory or a previous autograd graph.

Rules:

- Commit only student-produced concept states, never Teacher-only targets.
- Detach the committed prior sentence state to bound B28 memory.
- Reset on document/domain boundaries.
- Update sequentially at intermediate packed-sentence boundaries.
- Do not form a transition for the first sentence, across different domains,
  or when successive NP positions do not support the same tracked identity.
- Compare the relevant affected noun concepts, not undifferentiated
  sentence-wide activation differences.
- Keep unmatched or ambiguous records provisional; do not force every adjacent
  sentence pair to train a verb.

Concept admission is structurally host-side today. The compiled symbolic loop
may compute transition candidates, activations, and losses, then queue
admission or relation growth for the existing safe eager/reset boundary.
Semantic learning still belongs to the symbolic-order process; safe deferred
mutation does not move meaning creation into grammar.

## 7. Efficient verb representation

Do not allocate an invertible matrix per English verb. Keep one shared
verb-execution layer containing a bank of shared low-rank transformation
experts, and let each learned diachronic concept select a sparse combination
of them. This is more expressive than a single shared matrix without scaling
dense parameters linearly with the number of VPs.

A suitable forward form in the ConceptualSpace chart is:

```text
z_after = z_before
        + sum(j in top_k(g_V)) g_V[j] U_j phi(R_j z_before)
        + B b_V
```

where every `U_j`, `R_j`, and `B` is shared, while the verb concept supplies
sparse gates `g_V` and bias/control code `b_V`. Start with 32 experts of rank
32 and activate the top four per VP occurrence. Condition the gates on the VP
concept and bound argument roles, so polysemy can select different mixtures.
The existing sparse spectral gain may remain one expert, but it is not
sufficient by itself for general state and relation changes. If transition
quality saturates, first ablate expert count, rank, and two-step composition;
do not immediately introduce one private dense matrix per verb.

The executor should preserve unaffected conceptual structure through its
residual path and change only a sparse subspace selected by the verb and
argument roles. A learned reverse/past reconstructor may be added later, but
forward reconstruction is the normative verb objective.

Repeated transition training will cause some ConceptualSpace directions to
represent common temporal changes such as growth, movement, possession,
integrity, or activation. These are changes in `what`; subjective `.when`
orders their presentations but does not itself encode the semantic change.

## 8. Independent dimensionality of the spaces

PartSpace, WholeSpace, and ConceptualSpace dimensions are independent.
Concepts are defined by sets of PS/WS references and their activations, not by
a coordinatewise identity handoff:

```text
{PS percept references, WS percept references}
    -> ConceptualSpace identity row and atom
```

The number and richness of constituent percepts affect sparse edges and
activations, not the required ConceptualSpace feature width. Vocabulary and
verb count increase rows and sparse relations, not dimensions.

The current equal-width 1024 configuration is an experimental/configuration
choice, not an ontological constraint. Correct documentation that describes
PS->CS as an identity handoff.

Initial dimensional experiments after the Teacher and temporal-learning paths
work:

```text
PartSpace native width:       retain 1024 initially
WholeSpace native width:      choose independently for its property inventory
ConceptualSpace width:        test 512, then 256
shared verb operator rank:    32-64
active verb primitives:       approximately 4-8
```

Use the existing indexed activation/dictionary-decode seam; do not introduce a
dense PS->CS projection merely because native percept and concept dimensions
differ.

## 9. Current implementation gap

The aligned serial hot path currently does the following:

1. Eagerly stages one word/object concept row before the compiled word loop.
2. Repeats `compute_symbolic_reference` and `promote_symbol_reference` for
   `symbolicOrder` passes.
3. Decodes the same selected row and increments its order.
4. Resolves the staged object reference before Language chooses an operation.
5. Lets `VerbLayer` apply the supplied lexical/concept operand.

It does **not** currently:

- combine different concept rows during the serial symbolic passes;
- learn same-time higher-order noun rows in that recurrence;
- compare adjacent-time noun concepts;
- mint or refine diachronic verb concepts;
- resolve surface verb words to role-conditioned temporal meanings.

Implementation must therefore add a genuine concept-learning branch to the
serial symbolic-order loop rather than describing row promotion as concept
induction.

## 10. Implementation sequence

1. **Document and type the ontology.**
   - Update the architecture, spaces, mereology, and language documentation
     with subjective addressing, event mereology, openings, `fit()`, and the
     synchronic/diachronic distinction.
   - Add explicit concept-kind and event-validity metadata without changing
     numerical behavior.

2. **Land the Teacher seam on clean reconstruction.**
   - Centralize current reconstruction losses in `Teacher`.
   - Have the Teacher supply lossless objective source coordinates on a
     separate query seam, retaining private resolved coordinates for scoring.
   - Do not read, stamp, or replace the model's subjective `.where/.when`.
   - Preserve exact clean-input behavior and disable legacy predictor/forced
     grammar rerun weights.
   - Sample one derivation from the full retained forest, including occasional
     second-, third-, and lower-ranked alternatives.
   - Report reconstruction quality and sentences per second at B28.

3. **Make `symbolicOrder` form concepts.**
   - Add same-time concept combination and ordered temporal promotion.
   - Queue structural codebook mutations at safe eager barriers.
   - Preserve one-pass grammar execution.

4. **Learn and select verbs.**
   - Retain two-slot LTM entries in NP-VP form per discourse row.
   - On the next compatible NP, train the preceding VP on
     `T_VP(NP_previous) ~= NP_next`.
   - Bind surface verb references to learned temporal concepts and make
     grammatical role selection choose among senses.
   - Use one shared sparse bank of low-rank experts, initially 32 rank-32
     experts with four active per VP.

5. **Enable degradation.**
   - Apply the 10/70/20 curriculum with 10% clean anchors.
   - Progress through partial and blank perception only after clean
     reconstruction and verb learning are causal and measurable.
   - After explicit objective-address use is stable, test replacing it with
     relative and in-sentence reconstruction cues.

6. **Ablate conceptual width.**
   - Test 512 and 256 independently of native PS/WS widths.
   - Select the smallest width that preserves reconstruction, grammar, and
     learned transition quality.

## 11. Tests and acceptance criteria

### Ontology and mereology

- `0` remains `NOTHING` and `1` remains `EVERYTHING`.
- Valid openings, closed openings, and crossed/inconsistent openings are
  distinguishable.
- Optional fillers may enter and leave without changing the tracked concept
  index.
- Actual event parthood is transitively closed only on intersecting validity
  support.
- Direct-link storage and its transitive closure produce the same event order.
- Subjective `.where/.when` cannot be mistaken for world-event location.

### Teacher and memory

- Complete-input Teacher reconstruction matches the existing clean path.
- Partial and blank inputs never recover clean targets through leakage.
- The Teacher-provided objective address resolves the same requested input
  used for scoring and never mutates subjective `.where/.when`.
- B28 discourse rows remain isolated and reset at document boundaries.
- The truth store distinguishes assertion time from represented event support.
- Legacy predictor and forced grammar-rerun losses remain zero.
- One stochastic derivation is reconstructed per lesson; second-, third-, and
  tail-ranked admissible candidates are observed over repeated lessons.
- Holding the forest fixed, the chooser increases the probability of a
  lower-Teacher-loss derivation without evaluating a second derivation in the
  same lesson.

### Concepts and verbs

- Order-0 concepts are derived from PS/WS percept references with independent
  native and conceptual dimensions.
- Symbolic order 1 can learn both a same-time noun concept and an ordered
  before/after verb concept.
- Reversing before/after times produces a different temporal relation.
- Repeated seed-to-sprout examples reuse and improve one `grow` concept.
- The VP retained with a prior NP is trained only when the next compatible NP
  for the tracked identity is observed.
- A promoted VP improves average transition error over the identity/no-verb
  baseline on held-out eligible transitions.
- A grammatical derivation selects the learned temporal concept rather than
  using the raw surface word embedding as verb meaning.
- Applying the selected verb improves reconstruction of the final noun.
- No verb codebook entry owns a private dense invertible matrix.
- Multiple VPs can select distinct sparse mixtures from the shared expert bank.
- The causal transform may be non-invertible while grammatical constituents
  remain reconstructable from the diachronic event record.

### Performance

- Report clean, degraded, and blank reconstruction accuracy; grammar validity;
  temporal-transition reconstruction; verb-reference selection; and sentences
  per second.
- The clean B28 Teacher path may regress by no more than 15% from the accepted
  optimized baseline before degradation and verb-learning overhead are
  reported separately.
- Conceptual-width ablations report both quality and throughput; no width is
  selected solely from parameter count.
