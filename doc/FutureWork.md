# Future work

Cross-cutting items that are decided in direction but not yet specified or
built, with the decision that opened them and the document that owns the
detail once it exists. Section-local future-work lists elsewhere
([Language.md](Language.md#future-work-nouns-from-partspace-adjectives-from-wholespace),
[SymbolFirewall.md](SymbolFirewall.md#future-work-out-of-scope-for-this-pass))
stay where they are; this file indexes the items that span documents.
Items marked **urgent** block ordinary operation before long.

## 1. Operation profile: optimal versus human

**Decision (Alec, 2026-09-16).** Parameterise optimal versus more-human
operation so that both can be explored in one model. One `<operation>`
element in `model.xml` (`optimal` | `human`, default `optimal`) selects the
human-like behaviour for every item that
[Philosophy.md](Philosophy.md#discrepancies) marks **profile**:

| Item | `optimal` | `human` |
|---|---|---|
| Default belief | Cartesian: comprehension registers at trust `0`; provenance asserts | Spinozan: comprehension asserts at provenance trust; unbelieving is a later revision |
| Verbatim retention | generation from stored structure | generation from stored structure |
| Forgetting | capacity wall, recency only | forgetting model (§2) |

The profile is a switch over documented behaviours, never a hidden
heuristic; each behaviour has its own test under both settings. Params.md
gains the element when the first profile-dependent behaviour lands.

## 2. Forgetting and consolidation (urgent)

Specified in [the forgetting spec](specs/2026-09-16-forgetting.md)
(2026-09-16). Summary: `<ltmCapacity>` is the limit; a pass runs when the
store is nearly full, never mid-sentence and not more often than a minimum
interval, and deletes the lowest-valued unprotected rows down to a target
occupancy. Value combines trust (high trust preferred), utility (a row that
is easily deduced from other rows is not worth keeping: low expectation
surprise for ideas, one-step derivability for relations) and contribution
to luminosity (dissonant rows are forgotten; full coverage gain is optional
because it is expensive). Relations over deleted rows cascade; references
are remapped on compaction. The human profile adds an age term. The earlier
direction here, that isolated rows should be forgotten, is withdrawn: an
isolated row can score high on all three terms.

## 3. Lossy reconstruction trace; inversion as learning

**Decision revised (Alec, 2026-09-28).** A completed row stores its one-slot
or three-slot end state, never a derivation, under
[two truths §11.1](specs/2026-09-16-two-truths-ideas-and-relations.md#111-a-row-holds-structure-never-a-derivation-decided).
The operation record serves reconstruction and exploration only while the
sentence is open, then is discarded. There is no durable trace to decay
under either profile. Reading back is generation from the stored structure;
item 6 measures its current weak recovery without assuming exact wording.

The earlier proposal to drop derivation detail gradually is superseded.
Experiments that make reconstruction harder while a sentence is open would
need a separate learning specification. Source-text retention and subordinate
row coarsening remain in [the forgetting spec](specs/2026-09-16-forgetting.md#4a-detail-before-rows-alec-2026-09-20).

## 4. Context across documents

The hard reset at a document boundary clears transient context. People
carry context across texts and conversations. To specify: what survives a
document boundary under the `human` profile (the expectation observation
view, priming heat, the What interaction slots), how a new document's
first sentences are scored against a carried context without penalising a
genuine topic change, and whether the boundary becomes a learned soft
signal rather than a cursor fact. The current per-row document key and
clause-scoped reset in the integrated plan §8.3 and §11 are the base.

<a id="5-n-ary-meta-and-interpretation-time-discrimination"></a>

## 5. Definition rows and interpretation-time discrimination

**Amended (Alec, 2026-09-29).** The earlier n-ary META proposal is replaced
by `word DEF object` rows in
[two truths §17](specs/2026-09-16-two-truths-ideas-and-relations.md#17-definitions-word-def-object-decided-alec-2026-09-29).
Several definitions express synonymy and polysemy. One derived index returns
the objects of a word and the words of an object; learning their codes does
not change these identity links. `interpret` writes the definition and
replaces the word by its object in the existing inventory row. The grammar
selects among defined objects from context at interpretation time.

The remaining learning question is how the chooser ranks those objects in
context and credits a wrong selection through expectation and reconstruction.
Sentences that state a definition, and `Equals` versus `Def`, belong to the
grammatical operators update. Definitions' participation in forgetting is
ordinary row participation; any bias by truth kind belongs to item 5.

## 6. English arithmetic Q/A curriculum

**Proposal (Alec, 2026-09-20).** Generate English questions and answers from
bounded mathematics to supply output-training data and reproducible held-out
evaluation, for example “What is two plus two?” → “Four.” This belongs with
generation and supervised output (NEXT item 3 in [todo](../todo.md)); later
reasoning-utility comparisons remain item 4. This records a curriculum direction,
not an implemented English dataset or a new learning result.

Start with a small vocabulary of number NPs and a learned successor VP:
`successor(two) = three`. Teach the finite counting facts, then compose
addition from succession and multiplication from addition. Common sums and
products may be memorized through ordinary learned associations and memory;
evaluate that separately from deriving unfamiliar combinations. These are
grammatical meanings selected by the existing MLPs. Arithmetic on concept
coordinates or symbol addresses does not implement numerical addition.

Beyond the initial vocabulary, build number NPs compositionally using decimal
place value and carry, with addition and multiplication as compound VPs.
Teach English number names alongside this structure, including irregular
forms such as “eleven” and “twelve”; English spelling is not itself the
place-value algorithm. Do not allocate an independently memorized atomic NP
for every larger result. Variable binding and algebraic identities need their
own later lessons; decimal notation alone does not establish those abilities.

The existing [math generator](../bin/exact.py) can produce exact labels on the
data/evaluation side. Its current [loader](../bin/data.py) supplies numeric
one-hot answer labels, so English answer text and the normal supplied-answer
training path still need to be wired. Present the question through ordinary
compose, let thought establish the answer, and realize it through generate
and the normal vocabulary. During end-to-end Q/A, the desired answer and
oracle calculation are scoring targets, never an answer seed or a runtime
calculator. A lesson that supplies an already encoded, on-manifold idea to
train realization is useful too, but is identified separately from a test of
answering the question. Follow the concluded-idea gradient boundary and shared
operator contract in [the integrated plan §8.4](plans/2026-09-15-next-sentence-as-the-production-objective.md).

Generate separate training and evaluation sets with fixed seeds and partitions
over mathematical problems before applying wording templates. Keep equivalent
commuted problems together when evaluating unseen arithmetic, and keep
memorized-fact and paraphrase-only results identifiable. Reserve new operand
combinations, complete question wordings, larger composed numbers, carry
patterns and derivation depths for distinct evaluations. All base successor
facts may be taught; repeating them in evaluation measures acquisition, not
generalization. Score answer meaning as well as English realization through
the full vocabulary. Retain renamed-vocabulary controls for arithmetic meaning,
and the matched-compute comparisons before claiming useful learned reasoning.

Existing [successor-VP probes](../test/test_verb_successor.py) establish local
operator behavior under their own setup; they do not establish this end-to-end
English Q/A curriculum. The older [mathematical-thinking specification](specs/2026-09-09-mathematical-thinking.md)
retains the original motivation, but its retired controller and interpreter
are not part of this proposal.

## 7. Episodic memory: a few active indices beside each row

**Decision (Alec, 2026-09-20).** LTM holds serial form only: a row is one
idea per role, and several things at once can be stored only by chaining
NP / VP into a compound sentence, which approximates episodic memory and
keeps only what was *said*
([accessible-mind spec §2.7.1](specs/2026-09-20-accessible-mind-subsystems.md)).
The extension that makes episodic memory proper is small: **room, next to
each three-slot row, for a few indices of the most highly active percepts or
concepts** at the moment the row was written. Those are the things that were
present but never composed into the sentence.

What is stored is indices with their activations, not vectors — a sparse
field over the existing tables, so it is re-read through the current
dictionary and costs a few numbers per row beside three idea vectors. With
the row's where / when / source binding it is an episodic trace: a stored
state, bound to its occasion.

**Sharpened (Alec, 2026-09-28): what the single stored vector structurally
cannot hold.** "The fact that we are storing a single point in conceptual
space in LTM is the equivalent of a story, or semantic memory. An episodic
memory would be able to store the activations of all symbols (but that would
require a massive bandwidth, so we might approximate that with a top-K over
symbolic indices)." So the top-K is not an approximation of the *vector*; it
recovers a coordinate the vector cannot carry. Each concept's evidence is a
pair, which splits exactly into `d = c⁺ − c⁻`, the signed evidence, and
`m = min(c⁺, c⁻)`, the dissonance
([accessible mind §2.0](specs/2026-09-20-accessible-mind-subsystems.md)). A
stored resultant carries `d` and is blind to `m` — *both* and *neither* land
on the same point, and the origin is uncertainty. Semantic memory is the
`d` of what was composed; episodic memory is the surviving `m`, and the
activations that were never composed at all. That is the bandwidth the top-K
is buying, and it says what the selection rule must preserve.

To specify: how many (a handful); which tables (perceptual rows from the two
towers, conceptual rows of order 0 and above, or both); the selection rule
(top activation at the closing, excluding the codes the sentence itself
contributed; the row itself stores no word list or activation snapshot); whether retrieval
**reinstates** them into parallel knowing, bounded and decayed, so that
remembering differs from knowing; their use as retrieval cues and as the
evidence for discriminating a word's senses at interpretation time (§5);
their remapping when a table is compacted; and their place in forgetting —
they are detail, so they go first
([forgetting spec §4a](specs/2026-09-16-forgetting.md#4a-detail-before-rows-alec-2026-09-20)).

The psychological grounding is in the accessible-mind spec §7: the episodic
trace as an index to the pattern of activity rather than its content (Teyler
& DiScenna 1986), reinstated at retrieval (Danker & Anderson 2010); the fast
system as the binder of what co-occurred (O'Reilly & Rudy 2001); and
multiple-trace memory, where stored instances keep the senses a single
reading loses (Hintzman 1986; Jamieson et al. 2018).

## 8. Recall as estimate plus residual

**Direction (Alec, 2026-09-20).** Expectation is a negative image: what is
conceived of a sentence is the composed idea less what was predicted
([accessible-mind spec §2.6](specs/2026-09-20-accessible-mind-subsystems.md#26-expectation)).
The store already keeps the two terms as a linked `estimate` / `observation`
pair with the residual derived ([ExpectationRetention](ExpectationRetention.md)),
and the observation row keeps the full idea. That is the `optimal` layout and
item 2 builds on it unchanged.

The `human` profile goes one step further, as a stage of
[forgetting §4a](specs/2026-09-16-forgetting.md#4a-detail-before-rows-alec-2026-09-20):
the expected part of a row is detail, and detail goes first. A row coarsened
to its **tag** keeps only the residual and the pointer to its context, and is
recalled by adding the *current* world-model's estimate back. Recall then
drifts toward the schema as the world-model changes, which is the human
pattern: typical actions falsely recognised, atypical ones kept (Graesser,
Gordon & Sawyer 1979; Bartlett 1932; Brewer & Treyens 1981).

To specify: when a row is coarsened to its tag (after its estimate row is
gone, or before); what the pointer must retain so the estimate can be formed
again (the context rows, which may themselves have been forgotten); how a
tag row is ranked and unfolded, since a residual is not an idea on the
manifold and generativity is measured on ideas; and how a tag row's
provenance says that part of what it recalls is reconstruction.

## 9. Exclusion at the readout

**Direction (Alec, 2026-09-20).** "The object appears to the mind as the
negation of the non-object", and that negation is non-affirming: it removes
the non-object and puts nothing in its place
([accessible-mind spec §2.6.7](specs/2026-09-20-accessible-mind-subsystems.md#267-attention-the-exclusion-of-the-non-object)).
So attention needs no prediction in order to focus, and the architecture
already has it in that form: the reading scope, and the intent channel of the
priority surface at the concept pyramid's per-order top-K (floor of zero,
never a veto), behind `<relevance>`.

What is not built is the interaction at the level of rows. Item 2 applies
the estimate "to everything but the object of observation" at the level of
roles, where it cannot touch composition. At the level of rows it would make
an expected object quicker to recognise *within* a sentence: expected
non-object rows of order ≥ 1 lose priority, so the object competes against
less.

To specify: how the estimate, an idea, becomes a priority over rows
(`quantize`, restricted to rows of order ≥ 1 by the ramsification table);
what sets the object of observation at this level (conceptual activation as
the origin of reading attention,
[Architecture](Architecture.md#symbolic-weights-reconstruction-parse-time-attention-2026-06-30));
and the price, which is the reason this waits — selection at the readout can
change what is composed, so the purity invariant of spec §2.6.3 no longer
holds by construction. The residual must still be learned from what was
there, not from what was selected (statistical learning needs attention:
Turk-Browne, Jungé & Scholl 2005), which may mean composing twice. Decide
after item 2's measurements (spec test 32) show whether role-level sparing is
enough.

## 10. Fold-ladder tuning questions

The meronomy fold ladder's open questions Q1–Q3 — admission knobs, the
coverage schedule, and the category-utility floor
([plan](plans/2026-09-10-meronomy-fold-ladder.md)) — are tuning questions with
no evidence to settle them until the full training session produces some.
Moved from the todo on 2026-09-21.

## 11. Teacher-to-LTM persistence

Deferred until the NP-VP transition cache exists. Moved from the todo on
2026-09-21.

- When the deferred NP-VP transition cache is implemented, retain only
  detached, row-local student-produced records. Do not describe that cache as
  already implemented in Teacher v1 or commit reconstructions/predictions
  directly to the persistent/global truth store.
- Define an admission policy for student-generated memories with provenance,
  confidence, `asserted_at`, represented/target-event support, revision, and
  contradiction handling.
- Prevent Teacher-only clean targets and future information from entering
  student-visible LTM before evaluation.
- Add leakage, replay, correction, and document-boundary tests before enabling
  persistent writes.

## 12. Objective-address conditioning

A roadmap that follows the clean Teacher throughput gate; the constraints
belong to the Teacher specification it cites. Moved from the todo on
2026-09-21.

- Follow the [unified Teacher specification](specs/2026-07-27-teaching-modes-and-next-iteration.md)
  and its gated milestones. The [What and spacetime design](WhatSpacetimeDesign.md)
  describes the interface ownership and chooser/stack context.
- Keep corpus/snapshot/document/sentence/span coordinates on the Teacher query
  seam; never reuse or overwrite the model's subjective `.where`/`.when`.
- Split target observation/event time from source snapshot validity; retain the
  latter as provenance rather than overloading objective `when`.
- After the clean Teacher throughput gate, add a student-side encoder that
  embeds corpus/snapshot/split IDs categorically and document/sentence/span
  positions as ordered coordinates. Raw hash magnitude must have no meaning.
- Prove that an addressed clean input round-trips through the privileged
  `Teacher.Data(address)`, keeping `Teacher.What(where, when)` only as a
  controller-side compatibility adapter. Measure whether the separate
  `model.what(address)` uses addresses on partial and blank lessons without
  accessing private clean content.
- Treat DOI and source date as optional aliases/provenance. FineWeb has neither;
  retain shard SHA-256 and corpus release metadata instead of fabricating them.


## Four corners as prompts

Neither prompts attention; both prompts division of the subject into parts
on which the predicate has a pure reading. Division is pi in the order-0
field. A higher-order symbol cannot split; descend to its cases and their
field first. The 11c review residue implements the refinement/raise route:
contiguous support stays in field refinement, while discontiguous support
may be symbolized above order 1 only after local refinement stalls. The
strict-improvement/patience criterion is recorded for review in
[Architecture](Architecture.md) and the [receipt](benchmarks/2026-09-24-item10/README.md).
Choosing new attention targets from a neither reading remains future policy
work. See [two truths §1.1](specs/2026-09-16-two-truths-ideas-and-relations.md#11-both-is-a-compositional-fact-decided-alec-2026-09-23).

### Shared-mode erosion measurement (item 9b)

The three-seed job is retired from the test suite. Its original
script is [archived](benchmarks/2026-09-25-item9b/erosion/three_seed_job.py)
with the source and findings; use the archived runtime to reproduce it.
It ran N serial sentences, then parallel processing, then a label read-back.
The current schedule reverses the first two steps and removes the third.
**All of the old interleave rows and comparisons involving them are void for
the current schedule.** No replacement three-seed job is required for landing.

The first measurement
compares serial, parallel and interleave:2 from the same initialization for
each of three seeds, with symbols offline and with their owned reverse-pi
read-back. It uses four XOR probes and 68 word probes from 20 FineWeb launch
documents, after one pass over four XOR inputs and one complete sentence per
document. FineWeb categories are native orthographic properties, not semantic
categories. The [receipt](benchmarks/2026-09-25-item9b/README.md) preserves the
protocol, inputs, full measurements and every failed prediction.

On all three seeds the FineWeb native readings have lower categorical
separation and higher within-category discrimination after parallel training
than after serial training. The wider prediction does not hold: parallel
training admits no more structural alternatives; label read-back lowers rather
than raises categorical separation in every FineWeb condition; interleaving
does not keep separation near the serial value while retaining its predicted
within-category gain. Native XOR separation is zero in all conditions. This
short measurement does not establish the proposed erosion-and-restoration
mechanism, and the zero XOR result is separate from the native XOR learning
tests. No seed, threshold or training duration was adjusted to make an ordering
pass. More training and semantic probes are future experiments, not evidence
already supplied by this receipt. Item 4 will log the inexpensive distance
reduction from [CategoricalDiscrimination](../bin/CategoricalDiscrimination.py)
on the [fixed four XOR and 68 FineWeb probes](../data/categorical_discrimination_probes.json).
It accepts captured readings and returns CP, within-category and between-category
distances plus pair counts. It does no training or extra label read-back;
unknown/zero readings remain in the metric. The harness must budget the cost of
collecting probe readings when choosing its logging interval. This is a
descriptive measurement with no required ordering or threshold.

The motivation remains the qualified human evidence summarized in
[Philosophy](Philosophy.md#attention-as-one-bracket-both-as-the-fields-report-and-the-sharing-of-the-two-modes-2026-09-25)
and [plan §2](plans/2026-09-25-item-9b-mode-sharing-and-interpret.md#2-reasons-the-psychological-evidence):
shared semantic access and label feedback, category effects of verbal
interference, acquired equivalence and verbal overshadowing, and meditation
studies of discrimination and category flexibility. Those analogies motivate
the directional hypotheses; the present code-level results do not validate
the psychological account.

## Throughput levers for the serial loop (item 1 candidates)

**Recorded 2026-09-25 (Claude, on Alec's instruction) for item 1, the
optimization item, once its todo entry is next edited.** The compiled July
baseline ran at 6.803 sentences/s at B24 ([receipt](benchmarks/2026-07-27-pre-teacher-baseline.md));
the September eager, batch-1, expectation-on measurement ran at 0.26. The
per-word arithmetic is of the same order as a depth-4 NanoChat's per-token
arithmetic (about 40M parameters touched either way); the gap is scheduling.
Compare by bytes seen and wall time only
([pilot](NanoChatGrammarPilot.md#question-and-falsifiable-first-milestone)).

1. **Batch across sentences.** The word loop is a recurrence, so positions
   cannot be parallelized, but sentences can; recurrent training scales
   nearly linearly with batch until compute-bound. The pilot config still
   defaults to batch 4.
2. **Known-word lookup under the serial flag (Alec).** Known words are
   looked up in PartSpace, not translated. If the lookup needs a cache, or a
   hint about when to stop synthesizing over bytes and return the word's
   code, add it as an optimization concession under the serial-mode flag,
   beside the concessions already there. The byte ladder then runs in full
   only for unknown words, which is where `interpret` mints new objects
   ([item 9b](plans/2026-09-25-item-9b-mode-sharing-and-interpret.md)).
3. **Subsample reconstruction.** The tied reverse walk with its candidate
   search costs about as much as the forward; run it on every k-th batch or
   a subset of rows through the existing placement knob, with the
   reconstruction measurements re-baselined for the chosen k.
4. **Close the host islands.** Vectorize and compile the eager per-row and
   per-edge loops named under item 1 (`cs_read_memberships`,
   `_compose_order0`, `_reverse_field`, `refine_over_collected`,
   `_prepare_part_learning`, `promotion_observe`), and move the host-side
   COO dictionaries of the sparse stores to device CSR tensors so the whole
   word step compiles as one graph. The last such pass, the 2026-07-06
   vectorization of `stage_analysis_spans` / `property_spans`, bought a
   third of an epoch byte-identically (noted in the
   [fold-ladder plan](plans/2026-09-10-meronomy-fold-ladder.md)).
   Item 9b adds explicit candidates here: the host dictionary walk that
   gathers a ragged field, event attribution, testimony admission/retirement,
   and context scheduling. Its archived CPU erosion harness used the same native tensor
   recurrence bodies with an eager loop dispatcher to avoid recompilation as
   sentence banks change. Those measurements make no throughput claim.
5. **One pass each** for `subsymbolicOrder` and `symbolicOrder` in the
   first session.

Compounded, these put the serial loop in the tens of sentences per second,
NanoChat's order on the same machine; parity is not claimed.

## `.when` is redundant across the elements of one input (noted 2026-09-26)

`.when` is the incrementing sinusoid of the model's subjective step
counter. Every element of one input — its positions, percept events and
symbol occurrences — carries the same value, and the value differs only
across LTM rows, where it addresses the chain. Within a field the band is
therefore pure redundancy (Alec, 2026-09-26): it costs band width on every
element and, if scored per element, distorts reconstruction objectives, as
the item 9b bisection showed. Candidate simplification: carry `.when` once
per field beside the field's bracket, stamp it on elements only when they
leave the field (a ended row, a captured program, a symbol occurrence
that thought produces), and keep the exact clock side-band as now. This is
future work: the current grammar still transports the temporal band on every
element, with that shared band excluded from per-word reconstruction. Any
carrier change must preserve the grammar's temporal operations. The
`.where` band keeps the start only; a part's end follows from its byte
length, a whole's does not — a known limitation of start-only stamping.

## Global coordinate transport through learned operators

The two-rung ladder exactly decodes clean registry addresses. Its tolerance
to noise after learned composition and inversion remains a separate question.
The [item 9b experiment](plans/2026-09-25-item-9b-mode-sharing-and-interpret.md#4f-the-band-transported-through-the-grammar-reduced-periods-then-rungs-proposed-2026-09-26)
refuted reduced periods as a remedy for the September 26 loss jump: the extra
cost came from scoring a shared timestamp on every word. More rungs or rotary
transport therefore need their own measured positional-error problem and
evaluation; the occurrence/time correction does not implement them.

## The dynamic stop: glossing above words, descending at novelty (item 6.8-2)

Item 6.8 makes attention one mechanism — a bracket over the input, narrowed
until each bracket's encoding is pure — with the stop pinned at words for
the first session. The dynamic stop removes the pin: a known multi-word
unit that reads purely at its wider bracket is glossed as one symbol, and
narrowing descends below a word only where the reading is *neither*
(unknown content) or *both* (heterogeneous). This is speed reading that
slows at novelty. It is decided against item 1's per-word throughput
baseline: it pays for itself only if glossing known units saves more than
the open read costs, which on FineWeb should hold once the recurring units
are admitted, and it depends on the field's *both* being reliable, which
the logged categorical-discrimination index and both-rate monitor.

## Modality as the third index (noted 2026-09-28)

Words narrow the domain of discourse along three indices — which thing,
which stretch of time, which alternative — and individuate along each by
determiner, tense and modal
([accessible-mind spec §2.0.1](specs/2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention)).
The first two have machinery: `lower` and the recency buffer for things,
the symbolic layer's `.when` and the row's address for times. The third has
none. A modal ranges over alternative courses of events: "might" picks
some, "must" takes all, and a conditional narrows which alternatives are in
play. The reading of 2026-09-28 was that a modal bundle is a sigma over
alternative processes, so that modality is order and not a fifth dimension
of conceptual space. **Alec withdrew it on 2026-09-29:** "To square, now:
modality is not order, it is dimension; I think these are different."
Concepts stay opaque: the dimension is a learned subspace, not a band of a
concept's code.

*Added (Alec, 2026-09-29).* With the verb phrase as a projection beside
the noun phrase, modality is "a subsequent projection (so from a 3d NP to a
4D NP + VP to a 5D NP+VP+MP), where the MP is a modal phrase"
([operator catalogue §4.4](specs/2026-09-29-operator-catalogue.md#44-the-noun-phrase-and-the-verb-phrase-two-implementations-and-the-candidate)).
The 5D counts phrases, each in a learned subspace of its own, so a
concept's code stays opaque. Open with it: which modal sentences are a
modal phrase within the point and which an operator row over another row,
as two truths has "it is certain that P".

To specify when it is taken up: what supplies the alternatives (the explore
derivation and the predictor's estimate are the two sources of unrealised
continuations the model already has); how a modal row differs from an
asserted one without borrowing trust, which is univalent and is not
evidence; and whether "every", "always" and "must" share one mechanism,
since each takes all along its index and writes a relation instead of
lowering.

Smaller items from the same discussion, none of them scheduled:

* **The verb's band is hard-coded.** `τ = 0.1` and the clamp `±8` fix what a
  verb can mean: an effect below the threshold is exactly the identity, and
  one above the clamp is unreachable. By the live-wiring rule they are
  `model.xml` parameters.
* **Verb sparsity as a measurement.** On the complement of `w`'s support the
  noun passes through exactly, so `‖w_v‖₀` bounds what can be recovered
  without the derivation (item 6). Log its distribution over the learned
  verbs, and whether the touched dimensions cluster, as a single-domain
  constraint on verbs would predict (item 4).
* **Asymmetry of modification.** `lower` and `lift` add their operands
  before one shared map, so they are symmetric by construction and "gun
  oil" cannot be told from "oil gun", though one is a kind of oil and the
  other a kind of gun. Decided by Alec, 2026-09-29: "Compounds are
  subtyping, which explains their order effects." So the asymmetry wanted
  is sub-typing's, and the adjective is symmetric with its noun. The Ground is the head's own shape, its cases at
  every order, so no new parameters are needed, and the result keeps the
  head's order (Alec, 2026-09-28); what is missing is the wiring by which a
  modifier selects among the head's cases before they are folded back
  (accessible mind §2.0.1). `lexical_gate`, which would give a
  word's code a slice of the operator, has no callers.
* **What order measures.** The determiner lowers order and modification
  leaves it alone (Alec, 2026-09-28: "order only drops when the determiner
  is applied"). Order is "determined by sigma or pi fold over a class" and
  is not part of speech, with which it "only sometimes" corresponds: two
  count nouns may differ in order. A set of discrete concepts is one order
  above its members; a part of one concept's extension keeps its order
  ("the set vs part difference"). Open: whether parts made by combining
  what is not symbolic are always contiguous, which Alec put as "perhaps";
  and what happens to a part, "blue cats", when it is named and becomes a
  discrete concept that can be a member of a set.
* **Election or accumulation.** Decided by Alec, 2026-09-29: the noun
  phrase combines by election, and idempotently, "a red red bird is no
  more red than a red bird", "so perhaps a form of intersection". The
  adverb is the other kind: "a very, very, fast runner is faster than a
  fast runner"
  ([operator catalogue §4](specs/2026-09-29-operator-catalogue.md#4-noun-adjective-verb-and-adverb)).
* **Generic against universal.** "Cats sleep" tolerates exceptions and
  "every cat sleeps" does not; both write a relation row, and nothing yet
  distinguishes their force. *Since 2026-09-30 both lower, "cats" by an
  implicit "all" and "every" like "all" in the other number
  ([5.5 spec §6](specs/2026-09-30-occurrence-tense-aspect.md#6-surface-form-and-markers));
  where their force differs is still open.*
* **Privatives.** "Destroy", "kill" and "cancel" remove a property, which a
  positive gain cannot do. Like "fake" and "former", they belong to `non`
  or to a shift of the domain, not to displacement.

## Operator names (future work, Alec, 2026-09-29)

"The operator rename can also be future work." What each operator is to do
is decided in [the operator catalogue](specs/2026-09-29-operator-catalogue.md)
and stands; the names wait here.

| today | the new name | decided about it |
|---|---|---|
| `chunk` | `synthesize`, with `analyze` its reverse | it "pairs better with analyze" (catalogue §3.1) |
| `what`, `lookup` | `query` | one operator, which returns conceptual content (§3.6) |
| `quantize` | `symbolize`, with `conceptualize` its reverse | lower-level operators, perhaps of thought only (§3.4) |
| `arma` | `expect`, or none | open (§3.5) |
| `verb`, `adverb`, `preposition`, `tense`, `aspect`, `morphology` | names by what is computed; candidates in §3.9 | the code reads declared properties of a rule and never its name (decided; that part is not deferred) |

A rename is made in one pass, in grammar files, code, tests and documents,
with the old name kept nowhere.

## Lift and lower as dimensions; surface form and tense (future work, Alec, 2026-09-29)

"Lift/lower and surface/tense can all move to future work, or at least
after item 6."

* **`lift` and `lower`.** They were intended to raise and lower the order;
  "perhaps they are better suited to dimensional lifting/lowering". On
  that reading `lift` puts a phrase beside the description in dimensions of
  its own, NP → NP + VP → NP + VP + MP, and `lower` is its reverse face,
  the projection that takes the phrase back out, while the order is raised
  by the fold and lowered by its inverse, the determiner
  ([operator catalogue §5.4](specs/2026-09-29-operator-catalogue.md#54-lift-a-phrase-beside-a-phrase)).
  Today the two compute one symmetric sum, and neither undoes the other.
* **`surface`, with tense, morphology and aspect.** `surface` is "a surface
  transformation of the words, like tense and aspect", so that one lexeme
  and its forms, and a language's freedoms of word order, are one structure
  and not separate words and syntaxes (catalogue §7). Tense, morphology and
  aspect "need more work before they are included in any grammar". None of
  the four changes a value today (catalogue §9). *Taken up (Alec,
  2026-09-30)* as todo item 5.5, with the preposition and the meaning of
  `.where` and `.when`
  ([5.5 spec](specs/2026-09-30-occurrence-tense-aspect.md)); `lift` and
  `lower` stay here.

## Credit to the chooser from a supplied answer (noted 2026-09-29)

*Historical decision (Alec, 2026-10-01), superseded by October 2 ownership below:*
step 5a trained the whole path for a sentence with a supplied answer,
including the chooser and the object codes
([6.9 plan §4 step 5a](plans/2026-09-29-item-6-9-xor-grammar.md#4-the-plan)).
Credit by choice alone was measured and does not suffice: 0 of 10 runs met
XOR_grammar's bar, also with four explore trials; the whole path met it in
9 of 10 ([§3.13](plans/2026-09-29-item-6-9-xor-grammar.md#313-after-step-5-what-holds-xor-back-measured-2026-09-30-and-10-01)).
The note as written on 2026-09-29 follows.

In item 6.9's configuration the answer's error trains only the map that
reads the concluded idea. The state is cut at the concluded idea (decided
2026-09-20), and the chooser is trained by what the sentence's two trials
cost, which is reconstruction and expectation and not the answer
([6.9 plan §3.4](plans/2026-09-29-item-6-9-xor-grammar.md#34-what-the-answers-error-trains)).
Measured in isolation, XOR is learned without it. A sentence that comes
with an answer could count the answer's error in the comparison of its two
trials: credit by choice, which the cut allows, since no gradient passes
through the idea. Whether the grammar's choices should answer to a teacher
at all, or only to reconstruction and expectation, is Alec's to decide when
it is taken up.

<a id="separate-gradient-ownership"></a>

## Separate gradient ownership for reconstruction, expectation and output (taken up in item 6.9, Alec, 2026-10-02)

**Taken up by [6.9 plan §15](plans/2026-09-29-item-6-9-xor-grammar.md#15-one-writer-for-each-weight-alec-2026-10-02).**
Alec approved the structural split: reconstruction owns perception, parameterized
object codes, compose operators/tied inverses and the compose chooser;
expectation owns only predictors; the supplied answer owns only its readers.
Both consumers share one understanding record. Expectation detaches sources
and targets, and the answer detaches that understanding. Backward is restricted
to the owners. Lessons train choosers without moving operators. The current [§22 decision](plans/2026-09-29-item-6-9-xor-grammar.md#223-witnesses-and-the-optimizer-b-decided)
selects only by strictly lower reconstruction, ties to greedy, and trains
the reader on the kept trial alone in that historical decision; operators
round 2c uses one mean-loss reader step for compose departures and the
kept root for narrowing departures. Reconstruction uses momentum descent;
its witnessed inverse is retired. Product and mean replace conjunction
and disjunction, while min/max remain separate catalogue operators.
[GradientFlow](GradientFlow.md) records the implementation and its audit;
[the receipt](benchmarks/2026-10-02-item6-9-ownership/README.md) records measurements.
The closing ownership candidate reached each primary gate in 1/10 runs.
Its zero recorded ownership conflicts do not establish improved learning;
the ownership round was not accepted (§18). The [§20 receipt](benchmarks/2026-10-02-item6-9-reconstruction/README.md)
measures reconstruction as the sole writer of unconstrained codes, the
witnessed and free read-back terms, and answer ownership of XOR_exact's
reader coefficients. Its primary class gate passes **0/10** and reconstruction
**2/10**, against §14's 8/10 and 3/10; both audits still record zero ownership
conflicts. Conditional attribution also leaves reconstruction incomplete with
expectation and answer writers disabled. Its small, unpaired arms do not
identify a causal improvement; none of the 40 attribution runs meets both
existing bars. The sweep and moved cases expose remaining ports and failures,
so this is not an acceptance claim. The
[§20.3 catalog](plans/2026-09-29-item-6-9-xor-grammar.md#203-catalog-set-aside-now-to-return-once-reconstruction-and-xor-hold)
records what is set aside: unit-sphere codes, distributional pressure,
activation geometry and the answer's reach into the understanding. VQ EMA
is not planned to return, and the contextual rotation is retired.
The [§21.6 field proposal](plans/2026-09-29-item-6-9-xor-grammar.md#216-witnesses-operators-and-the-field-alec-2026-10-03)
and the §20.3 alternatives remain catalogued. The
[§§21–22 receipt](benchmarks/2026-10-03-item6-9-free-readback/README.md)
adds code/anchor displacement and modal-derivation stability to the ownership
audit. Its class gate passes 9/10 and reconstruction 5/10, against §14's
8/10 and 3/10; the sum-only control passes 10/10. The required attribution
condition is not triggered. In the audited XOR run the codes and chooser
anchors have zero displacement at float32 precision, so the answer's learning
does not demonstrate learning in those weights. Saved tensors also show an
inner negation whose intermediate operand is absent from the raw primed bank:
free pair search chooses a repeated word instead. These are measured limits
for review, not permission to restore any catalogued mechanism or alter a
gradient scale. The audit does not infer stability from a small gradient.
The closing sweep and moved cases remain red, including unintended effects
of the shared inverse on answer generation and of the mean kernel on truth
penalties. The saved moved-case inverse also exposes target probabilities
below the existing byte clamp. These issues and the incomplete rule-only
journal scope remain in the receipt; the improved gate counts do not close
6.9 or authorize reintroducing the catalogued mechanisms.

The [§25 closing receipt](benchmarks/2026-10-03-item6-9-closing/README.md)
supersedes those historical status statements and closes 6.9 as a measured
baseline. Reconstruction now owns the single generate decoder and the antipode
term; output owns only its readers/conditioner. The echoic snapshot stays after
the seen write: the §24.4 pre-write proposal is withdrawn. The earlier §22
9/10 class and 5/10 reconstruction counts remain prior measurements, not the
new candidate's result. The closing receipt records its one shared training.

Forward dependence remains: reconstruction can change the evidence the answer
or expectation has learned to read. Future work can measure that drift and
compare it with learning speed and read-back. Disjoint writers remove direct
competition over an optimizer parameter; they do not prove that every block's
objective decreases together. No new norm-matching rule is implied.

### Earlier optimization proposal (deferred)

The earlier joint-update proposal is retained below as a secondary direction,
not an implementation decision. Alec deferred it here in favor of investigating
structural separation. It changes no 6.9 code, configuration, optimizer,
threshold or measurement and supplies no implemented guarantee.

The earlier §14 candidate demonstrated that accurate answers and accurate
read-back could coexist: three of its ten class-gate runs met both bars.
That historical result says nothing about their simultaneous expectation cost
or the later ownership candidate. A low reconstruction pass count alone cannot
diagnose interference.

#### Establish where interference occurs

Use reconstruction, expectation and supplied-output errors from the existing
Error registry, with the same uninformed baselines and declared weights.
At each actual sentence update, record each gradient's dot product with the
actual parameter displacement, plus each objective's cost before and after on
the same targets and recorded derivation. Report shared parameter groups and
private reading-map parameters separately. Keep selection's effect separate
from the optimizer's effect. This would be a new diagnostic after review,
not a reinterpretation of the existing stage-1 gradient measurements.

The earlier, now retired correction addressed only output versus reconstruction.
Expectation then retained its own gradient. Adam, the proximal operation and the subsequent
parameter projections all contribute to the final displacement. Both trial
references are computed before the first update, so the second update also
needs an audit at its actual starting parameters. A favorable raw-gradient
cosine is therefore insufficient evidence of a favorable completed update.

An algebraic example illustrates the optimizer issue, without claiming it
occurred in these runs. Let `gR = (1, 1)` and `gO = (-2, 2)`: their dot product
is zero. The combined gradient is `(-1, 3)`, whose negative lowers reconstruction
locally. Applying a positive coordinate scaling `diag(10, 1)` produces the
displacement `(10, -3)` instead, and `gR · d = 7 > 0`: reconstruction rises
locally despite the earlier orthogonality.

#### Proposed update rule to test

Choose the shared update jointly from all three objectives, seeking a direction
that helps each active objective. Preserve each private module's own update.
For a parameter displacement `d` and objective gradient `g_i`, the local
no-increase condition is `g_i · d <= 0`. A concrete candidate is to find the
displacement closest to the optimizer's proposed displacement subject to those
conditions on the shared parameters. This is a joint calculation; independently
correcting pairs can undo an earlier correction.

The chooser uses a straight-through gradient. A local constraint on that
surrogate does not guarantee what a different hard derivation will do. Record
fixed-derivation costs and subsequent policy-selected costs separately.

Evaluate the completed candidate displacement, including the optimizer's
stateful scaling and the model's required parameter projections, against the
same three costs. Local linear constraints do not guarantee finite-step cost
improvement. A rejected trial update must also restore optimizer state and
state/version side effects; rejecting only the parameter values is incomplete.

If there is no useful common improvement, report the conflict explicitly and
retain reconstruction's declared precedence. Do not quietly choose another
objective's priority by gradient magnitude. Strictly forbidding any cost
increase can stall at a compromise even when useful progress elsewhere remains;
the fallback trade-off therefore needs an explicit design decision. This note
does not introduce tolerances, a new step count, a new learning rate, or a new
training budget. Both sentence trials must still be compared at identical
parameters before either is trained.

Keep the existing relative errors; do not normalize every gradient to unit
length. A nearly solved objective should not acquire a large update merely
because its residual is small. Targetless penalties keep their own strengths
and category, and remain visible in the audit of the completed update.

#### Basis and limits

Joint gradient choices are an established approach to multi-objective training;
[Sener and Koltun (2018)](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
study that formulation. [CAGrad (2021)](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
balances aggregate progress with the worst local task improvement and discusses
the risk of stopping at an arbitrary compromise. Neither citation supplies a
guarantee for this model's Adam/proximal/projection sequence. The completed-step
check and reconstruction priority above are proposed integration requirements.

The first decision should be informed by the completed-step diagnostic. It can
separate objective interference from discrete derivation changes, inadequate
inverse learning, or read-back quality remaining poor despite decreasing cost.

## A snapshot where a sentence's two trials branch (future work, Alec, 2026-09-30)

Item 6.9 costs a sentence's two derivations under the same parameters
before the optimizer steps on either ([6.9 plan §4 step 5](plans/2026-09-29-item-6-9-xor-grammar.md#4-the-plan)).
The explore derivation repeats the exploit derivation up to the round
where it is made to differ, so its prefix is computed twice. Alec, of the
equal comparison: "let's opt for making this efficient; maybe even a
snapshot at where they branch, if that helps significantly (the snapshot
overhead may be high, in which case don't). Let's add this optimization as
future work." When it is taken up: measure the time and memory of the
prefix recomputed against a snapshot of the reading's state at the
branching round, and keep the snapshot only if it saves significantly.

## Stages of learning as gates (proposed, 2026-09-28)

Proposed by Alec, not decided: "We have to wait until a model is trained
before we can express truth in that model, before LTM is meaningful in a
consistent way. This suggests stages of learning, and perhaps they can act
like gates. Object permanence has to be learned before objects. Words have
to be learned as concepts before we can interpret them as references."

Three dependencies were named on 2026-09-28:

| what waits | for what |
|---|---|
| objects | object permanence |
| words interpreted as references | words learned as concepts |
| truth expressed in the model, and a meaningful LTM | a trained model |

On 2026-09-29 Alec gave an order: "words must be learned before symbols are
minted for the objects that they represent. And that must happen before
identity/object permanence." That is words, then the symbols of their
objects, then identity, and it puts object permanence *after* the objects'
symbols, where the first row above puts it before objects. Asked which
holds, he settled it the same day: "I was just stating what I take to be a
practical necessity: Words must be stable. Then they can represent
something. Then they can represent the same thing (this is object
permanence)." So the order is three steps, and object permanence is the
third:

| step | what | what it needs |
|---|---|---|
| 1 | words are stable | a form that recurs is kept |
| 2 | they represent something | a stable word, which `interpret` replaces by its object |
| 3 | they represent the same thing: object permanence | an object to be the same as |

The first row of the table above, objects waiting for object permanence,
is superseded by it.

It came up because a truth given to an untrained model is stored with its
trust and with no evidence of identification, so the truth view is empty
([two truths §13 G](specs/2026-09-16-two-truths-ideas-and-relations.md#13-review-round-2-claude-2026-09-28-on-the-candidate-after-12)).
That is correct, and a gate would say so in advance instead of leaving an
inert row.

**The progression** *(Alec, 2026-09-29)*: "we need progressive learning:
learn words, learn the objects they represent, then learn small sentences.
Then learn more complex sentences, just like children learn."

**Corpora that exist** *(checked 2026-09-29; licences not checked)*. None is
staged in that way. Each covers a part:

| corpus | what it is | stage it could serve | ordered |
|---|---|---|---|
| Wordbank | which words children know at which ages, from parent reports | words | by age |
| AO-CHILDES | about 5M words of American English speech to children | small sentences, growing | by the child's age |
| TinyStories | 2.7M synthetic stories on about 1,500 words, at the level of a child of three or four | small sentences | no |
| BabyLM | 10M and 100M words: speech to children, dialogue, children's books, subtitles, Simple English Wikipedia | small to complex sentences | no |
| Leaner-Pretrain | 71M words with vocabulary and structure simplified | simple sentences | no |

The stages are an order on what the model may mint, not a pairing supplied
by a corpus: "I don't mean pairing words to objects in the way that you
suggest: merely that words must be learned before symbols are minted for
the objects that they represent" (Alec, 2026-09-29). So one text can serve
every stage, the gates deciding what is learned from it. BabyLM is the
candidate he named: plain text, one file for each of its domains, so that
it can be read in an order of the domains, speech to children first.
Ordering the data by difficulty was tried widely for ordinary language
models in the BabyLM challenge and was largely unsuccessful; that tested
the order of examples, not gates on what the model may do, so it does not
settle this proposal, and it does say that order alone should not be
expected to help (Huebner et al. 2021; Eldan & Li 2023; Warstadt et al.
2023; Frank et al. 2017; Yang et al. 2025).

**If another modality is ever added** *(Alec, 2026-09-29)*: "at some point,
LLMs learn objects through multi-modal training, so Fei-Fei Li's dataset
would be relevant". Two come from her group, and both are keyed to WordNet,
so their labels arrive already arranged as a taxonomy of nouns. ImageNet
fills WordNet's noun sets with images. Visual Genome annotates about 108,000
images with their objects, attributes and the relations between pairs of
objects, and with descriptions of regions, which is nearer to a sentence
about a scene (Deng et al. 2009; Krishna et al. 2017). Neither is wanted
while the model reads text alone.

To specify when it is taken up: what is measured to open each gate, and
whether it can close again; what a gated capability does while it waits,
refuse or defer or store inertly; how these gates relate to the ones that
exist, item 9's exposure count, the recurrence gate of 9b and the patience
of 11c; and whether the same gates order the curriculum. The order in
which the text would be read, and the correspondence of the gates with
Piaget's stages, are in
[gradual training](#gradual-training-an-age-ordered-corpus-read-in-stages-proposed-2026-09-29).

## Gradual training: an age-ordered corpus, read in stages (proposed, 2026-09-29)

Proposed by Claude on 2026-09-29 in answer to Alec's "we need progressive
learning: learn words, learn the objects they represent, then learn small
sentences. Then learn more complex sentences, just like children learn",
and written down at his request the same day. Nothing here is decided or
built. It has two halves, which are separate: the order in which the text is
read, and the stages that say what the model may do with it
([stages of learning as gates](#stages-of-learning-as-gates-proposed-2026-09-28)).

**The order of the text.** One corpus is ordered by age. AO-CHILDES is the
speech addressed to children in the American English transcripts of CHILDES,
for children from birth to six years, with the children's own utterances
removed, in the order of the age of the child spoken to: 2,000,352
sentences, 27,723 different words and 4,960,141 words in all (Huebner &
Willits 2021; Huebner et al. 2021). Read in bands of age, youngest first,
it is the nearest thing there is to "just like children learn". The
assembly proposed:

| step | what is learned | from |
|---|---|---|
| 1 | words | Wordbank's early vocabulary, in the order in which children acquire it |
| 2 | the objects the words represent | no corpus: `interpret` over the words of step 1 ([two truths §17](specs/2026-09-16-two-truths-ideas-and-relations.md#17-definitions-word-def-object-decided-alec-2026-09-29)) |
| 3 | small sentences | AO-CHILDES, the youngest bands first |
| 4 | complex sentences | AO-CHILDES at the older bands, then the written parts of BabyLM, then FineWeb |

The repository's own lesson sets, such as `data/grammar_wording.json`, are
lessons and not a corpus, and could seed steps 1 and 3.

**The stages.** Alec's name for them is Piagetian. Piaget's four stages, and
what in this model answers to each:

| Piaget's stage | age | what the child comes to have | what answers to it here |
|---|---|---|---|
| sensorimotor | birth to 2 | object permanence | percepts, their parts and wholes; a unit that recurs is kept; identity carried by prediction ([item 6.5](specs/2026-09-26-independent-components.md)) |
| preoperational | 2 to 7 (the symbolic function from 2 to 4) | symbols and language | a word is a concept, and `interpret` replaces it by its object |
| concrete operational | 7 to 11 | conservation, reversibility, class inclusion | operators that have inverses; a set one order above its members, and the part rows between them |
| formal operational | 12 and after | reasoning from hypotheses | relations between truths (implies, operator rows), modality as [the third index](#modality-as-the-third-index-noted-2026-09-28), the thought grammar |

Three cautions belong with the table.

* *It is an order for gates, not a claim about development.* What is
  borrowed is that each stage needs the one before it. The ages are
  Piaget's and are disputed: infants of three and a half to four and a half
  months already look longer at an event in which a hidden object has
  ceased to exist (Baillargeon 1987), far earlier than the eight to twelve
  months he gave. A gate opens on a measurement and never on a band of age.
* *The two halves can come apart.* The order of the text is what the
  BabyLM challenge tried and found largely unsuccessful for ordinary
  language models; the stages are gates on what the model may mint, which
  that result does not test. So the order of the text is the weaker half,
  and the comparison to run is the same text with and without the gates,
  and with its bands in order and shuffled.
* *The order of the gates is Alec's, and it is not Piaget's.* "Words must
  be stable. Then they can represent something. Then they can represent
  the same thing (this is object permanence)" (2026-09-29, "a practical
  necessity"). Piaget has object permanence before the symbolic function,
  because the child's first objects are things seen and handled. A reader
  of text has no objects but those its words stand for, so for it the
  permanence of an object can only follow the word that represents it.
  The table above is a correspondence of kinds of achievement, and the
  order in which the model reaches them is the three steps of
  [the section above](#stages-of-learning-as-gates-proposed-2026-09-28).

To specify when it is taken up: the bands of age and how many sentences
each holds; the measurement that opens each gate; whether a gate, once
open, is open for good; and the two comparisons above, with no seed chosen
for either.

References: Piaget (1954), *The Construction of Reality in the Child*;
Inhelder & Piaget (1958), *The Growth of Logical Thinking from Childhood to
Adolescence*; Inhelder & Piaget (1964), *The Early Growth of Logic in the
Child: Classification and Seriation*; Baillargeon (1987), "Object
permanence in 3½- and 4½-month-old infants", *Developmental Psychology*
23(5), 655–664; Huebner & Willits (2021) and Huebner, Sulem, Fisher & Roth
(2021) for AO-CHILDES. The stages' ages and contents were checked against
secondary sources on 2026-09-29; page numbers were not.

## Ergodic exploration everywhere (future work, Alec, 2026-10-02)

Alec: "Keep Ergodic, it's better in principle than random weight init; if it
works, we'd prefer everything is Ergodic (but that can be future work)."

**What it is today** ([Ergodic](Ergodic.md)). `ErgodicLayer` gives a layer
the effective weight `bias·W + var·ε`: the learned weight, trusted by
`bias`, plus sampled noise scaled by `var`, with `bias + var ≈ 1`. The
optimizer does not train the two scalars. After each backward,
`paramUpdate()` sets them from the observed gradient energy `s`:
`var = s/(s+κ)`, at most .95. So exploration is high where the gradient is
still large and falls as learning settles.

An ergodic layer starts from structure, not from chance:

- `LinearLayer(ergodic=True)` starts at the identity.
- `InvertibleLinearLayer` starts with identity LDU factors. Its noise enters
  the factors, so its inverse stays exact.

**Where it runs today.**

- The `<ergodic>` flag turns it on for the layers built on `ErgodicLayer` in
  `Layers.py`: `LinearLayer`, `InvertibleLinearLayer`, `LDUReadout` and four
  others.
- Two MNIST configurations turn it on, `ergodic.xml` and `ergodic-only.xml`,
  and `test_basicmodel.py` tests its layers.
- No language configuration turns it on.

**What "everything is Ergodic" would mean.** Every learned map starts from
the identity, or from the structured start its kind admits, and explores by
gradient-energy noise rather than starting from random weights. These start
from randomness today:

- the concept dictionary's rows (random signed rows on the hypersphere);
- the chooser and scorer networks;
- the expectation predictors;
- the answer's readers;
- the operators' projections from code to operation.

**To settle when taken up.**

- **Dictionaries.** A codebook with more rows than dimensions cannot start
  at the identity, so exploration there means noise on rows from a structured
  (for example orthogonal) start. Item 6.9 found that random signed codes
  make XOR readable almost for free
  ([6.9 plan §3.11 and §3.13](plans/2026-09-29-item-6-9-xor-grammar.md#311-whether-the-bar-can-be-reached-at-all)).
  A structured start must not lose that.
- **Ownership.** The energy sensor should read the gradient of the weight's
  own objective. With one writer for each weight
  ([6.9 plan §15](plans/2026-09-29-item-6-9-xor-grammar.md#15-one-writer-for-each-weight-alec-2026-10-02)),
  it does so by construction.
- **Run to run.** The noise is sampled on every forward. Runs are compared
  as unseeded measurements, with no seed pinning.

**Evidence first.**

1. The kept MNIST test, run with `ergodic` on and off. This is the A/B that
   `simple.xml` against `ergodic-only.xml` used to give.
2. Then the XOR proofs with ergodic layers.
3. Then a BasicModel stage-1 comparison.

"If it works" means equal or better learning at equal updates, unseeded and
repeated.

## Multi-resolution `.where` tiling (future work, Alec, 2026-10-02)

Alec, of the overlapping tiling retired in item 6.9's configuration cleanup:
"in principle, this is the kind of multi-resolution .where tiling that makes
JPEG efficient. Let's at least add it as future work, with enough detail
that it could be [revived] without difficulty."

**The idea.** JPEG spends its bits where an image has detail: coarse
structure is coded once, and refinement is added only where it is needed.
The `.where` analogue holds the input at several granularities at once
(sentence, word, separator run, typed run). Each region settles at the
coarsest whole that accounts for it, and is refined only where parts and
wholes disagree. A fixed budget of wholes then covers the whole input
coarsely and spends the rest where detail is needed.

**What was built.** It was added in `c113ff9d` (2026-07-15, "an automatic
granularity algorithm") and is intact at `d679df2b`, the last commit before
the 6.9 cleanup. Restore it from there.

- **The setting.** `<overlapWhereTiling>`, default false, requires
  `<mereologyRaise>`. The configuration `data/MM_overlap_tiling.xml` is
  MM_mereology with this one flag.
- **`WholeSpace.stage_overlapping_spans`** (`Spaces.py`) builds the
  lattice: four kinds of whole per row, deduplicated.
  1. a typed run, the maximal run of one character type (wholes are types);
  2. a word bounded by separators;
  3. a separator run;
  4. the enclosing sentence.

  Words and separators are placed first, so the fixed WholeSpace event
  budget always holds a complete tiling of the surface. Typed refinements and
  the sentence parent follow. The metadata lattice is never truncated.
- **`WhereTilingLayer`** (`Layers.py`) finds where PartSpace's parts and
  WholeSpace's wholes agree, over overlapping candidates, with fixed shapes,
  in parallel. For every local family it computes:
  - equality;
  - immediate containment;
  - coverage (one gap-free cover for each container);
  - a route: null, settled, sigma (fold the parts up), pi, or raise.

  A part may equal a whole and also be an immediate part of a larger whole,
  so a settled word stays available as a constituent of its phrase or
  sentence.
- **`build_schedule(part_spans, whole_spans, passes)`** runs a fixed number
  of refinement passes, refining the frontier of unsettled parts at each
  pass. It returns the observations of each pass, the accepted wholes and the
  overflow counts.
- **`WholeSpace.stage_where_tiling` and `where_tiling_for_pass`** hand the
  schedule to each subsymbolic pass. The hooks are in
  `Models._lex_embed_stem`.
- **`bin/eval_where_tiling.py`** is a corpus evaluator that does not depend
  on the lexer. It reads JSONL `text` with gold UTF-8 byte `spans`. PartSpace
  starts from byte atoms, WholeSpace proposes the overlapping wholes, and the
  layer must settle the gold surface objects within its passes.
- **`test/test_where_tiling.py`** holds its tests.

**How it fits item 6.8.** Item 6.8 narrows one bracket over time: the open
read, then `divide`, `descend` or `gloss`. Multi-resolution tiling holds the
same hierarchy at once. Revived inside 6.8, the lattice would become the
open read's multi-scale field:

- each bracket's children are already tiled, so `divide` and `descend`
  choose among existing tiles instead of computing cuts;
- `gloss` settles a region at its coarsest pure whole;
- 6.8's word whole, a maximal run of letters (6.8 plan §3a), is the second
  kind refined by the first.

**What would show that it works.** At a fixed budget of wholes, compared
with the flat tiling:

- reconstruction fidelity;
- the number of bracket operations per sentence;
- the evaluator's accuracy on settled objects, on a corpus with gold spans.

## Set aside during item 6.9, to return once reconstruction and XOR hold (Alec, 2026-10-02)

Alec: "let's create a catalog of what we want to reintroduce after getting
things working." The catalog is
[6.9 plan §20.3](plans/2026-09-29-item-6-9-xor-grammar.md#203-catalog-set-aside-now-to-return-once-reconstruction-and-xor-hold):
unit-sphere codes (magnitude as certainty under inner product);
distributional pressure on the codes (co-activation, one mechanism); the
primed bank as context and candidates; a spatial representation of the
priming field; and two things recorded as not returning (the VQ EMA refresh)
or as fallback only (the answer's reach into the understanding). Each entry
names the condition for its return.

The catalog is extended by [§24.5](plans/2026-09-29-item-6-9-xor-grammar.md#245-what-decoding-needs-and-where-each-part-comes-from-alec-and-claude-2026-10-03)
and [§25](plans/2026-09-29-item-6-9-xor-grammar.md#25-decoding-from-conceptual-space-and-closing-69-alec-2026-10-03).
The antipode's pull-apart half is implemented inside reconstruction; co-activation
attraction remains deferred. The operators update is to supply identities from
conceptual space and the remaining catalogue. Item 6.8 owns the open read and
MM_xor's word-level XOR proof. Surface markers own operand-order recovery.
The standing XOR rule is no regression against the [closing record](benchmarks/2026-10-03-item6-9-closing/README.md),
including its explicit red gates; this is not a claim that every gate is green.
Next: the [operators update](plans/2026-10-05-operators-update.md), then 6.5 (6.8 accepted; no conference freeze; Alec, 2026-10-05).

## Carried from the 6.8 §13–§15 rounds (2026-10-04)

Decided the same day and recorded in the 6.8 plan
([§13.4](plans/2026-09-27-item-6-8-one-attention.md#134-perceptual-space-is-the-basis-of-zero-order-conceptual-space-alec-2026-10-04),
[§14](plans/2026-09-27-item-6-8-one-attention.md#14-review-of-the-13-measurement-claude-2026-10-04),
[§15](plans/2026-09-27-item-6-8-one-attention.md#15-review-of-the-14-measurement-claude-2026-10-04)),
[Architecture](Architecture.md#two-spaces-one-index-the-symbol-in-perceptual-space-the-concept-in-conceptual-space-decided-alec-2026-10-04)
and [Philosophy](Philosophy.md#the-sign-in-two-spaces-genera-are-not-located-2026-10-04).
Not in the 6.8 rounds; each names its home.

- **Kleene connectives over codes** (operators update): `∧ = min`, `∨ = max`,
  `not = −d`, replacing the product and probabilistic-sum kernels chosen in
  6.9 §22 for invertibility; the inverses are codebook searches either way.
  Catalogue [§3.8](specs/2026-09-29-operator-catalogue.md#38-what-was-decided-earlier-in-the-pass).
- **The fold composing forms above words** (operators update): the proposed
  located word fold is superseded by round 3a's boundary pairs, exact length
  thermometer and collision mints. A word remains a max over its parts;
  anagrams need no general positional fold. Higher-order forms still use
  the binding kernels. Meanings and their connectives remain later work.
- **The complement's bootstrap** (operators update, the distributional item):
  the conceptual wholes' locations (sets, situation and document codes,
  properties as concepts), trained by co-activation, from which a new word's
  context mean can start; without them order-0 meanings stay at zero.
- **The `not` items** (operators update): `NonLayer` computes `1 − x`;
  `ConjunctionLayer` reads `max(c⁺, c⁻)`; the binding kernel pools signs.
- **The negative image on the concept face** (item 2): the §2.6.1 mechanism
  is implemented in the uncommitted round-2 candidate: gain, presence and
  role attention gate the concept complement; storage retains the observed
  value and thought reads the conceived value. Form is protected. Learning
  utility still waits for a populated complement and the later measurements
  ([receipt](benchmarks/2026-10-06-operators-round2/README.md)).
- **Sparse percept presences** (perception, when it is trained): dense
  prototypes' joins saturate toward everything (mean cosine between word
  forms ≈ .98 at any scale, XOR term of a unit root .03–.05); presences on
  ~30% of coordinates give ≈ .83 and .25–.35. The gates' receipts report
  mean cos(L) so the state is known; nothing is set.
- **Footprints checked at load** (operators update): catalogue §1 rule 10,
  each operator's reads and writes over form, meaning and poles, checked
  against rule 2's declared writes.
- **The inverse's chooser** (operators update, first item; Alec, 2026-10-05:
  "The inverse should have a chooser also"): a learned score over candidate
  decompositions (operation to undo, pair or word), features the relative
  residual under each inverse, the candidates' activation and priming, the
  end state; trained by cross-entropy toward the true decomposition (the
  input's words, the compose trial's operation), teacher forcing with free
  inference; replaces the argmin residual and the fixed `activation × cosine
  × priming`; separate parameters from the forward chooser (shared weights
  would let the inverse's cross-entropy reinforce the forward's choices
  without cost). The walk policy (undo / unary / STOP) trained the same way.
  (6.8 plan §16.4.) The earlier "one scorer for compose and generate" is
  withdrawn in its shared-weights form.
- **A dense chooser signal, if wanted** (later): the straight-through
  mixture's first-order comparison at every round, entered as a control
  variate for the score-function estimator (REBAR, Tucker et al. 2017; RELAX,
  Grathwohl et al. 2018), which keeps the estimator unbiased; a plain sum of
  the two is biased again (6.8 plan §16.3).
- **The attention chooser's estimator** (operators update; Alec, 2026-10-05):
  the input-attention walk scores its bracket actions with the compose
  chooser's parameters; its straight-through credit was found reaching the
  shared chooser through the perception pullback (6.8 plan §21.1, §22) and was
  detached at the sentence handoff. Round 1 supplied the score-function
  estimator; round 2 makes its sentence mechanism live by handing
  poles to the sentence and sharing the sentence's departure and `R+E+A`
  comparison with compose. Its owner remains reconstruction. Measurement
  of actual nonzero advantages is in the round-2 receipt. Its unchanged
  parallel MM caller still bypasses the sentence owner step; that coverage
  requirement is unresolved.
- **The decomposition chooser's exact-fit precedence and feature
  standardization** (operators update; 6.8 plan §23): after cross-entropy on
  four sentences the chooser's context weights outweighed the fit for one
  root ("hello there → hello hello", §22 run 10); an exactly recomposing pair
  is taken before context has a say, and the activation features (projection
  coefficients, unbounded) are standardized.

**Status, 2026-10-06 (operators update [plan §9.5, §12.3, §14](plans/2026-10-05-operators-update.md)).**
Landed in round 1 (accepted 2026-10-06): the `not` items (withdrawal and pole
exchange, bilattice conjunction), footprints at load, the inverse's chooser
with exact-fit precedence and standardized features, the attention
estimator (score-function, found silent by construction — plan §5). Round 2
is an uncommitted candidate held for review after its gate missed: class
4/10, reconstruction 6/10, MM_xor convergence 9/10, sum 10/10; sweep green
and ownership zeros retained. The initial observer counted only the sentence payload
([round-2 receipt](benchmarks/2026-10-06-operators-round2/README.md)); the
round-2b replay establishes a live MM trajectory, so its miss counts:
the trial cost as the owner-step total, one departure over both walks with
the narrowed poles handed off, the negative image at the closing on the
concept face (spec §2.6.1, confidence-gated, so it needs no meanings to
exist). Round 3: the fold composing forms is replaced by boundary-marked
pairs plus length as the parts (anagrams separate without position), the
content width, the has-a wholes and the narrowing-weighted centroid. Round
4: the complement's bootstrap (membership as properties, co-activation
through shared wholes), the Kleene connectives over meanings, expectation
and targets by inversion. Round 5: the attention filter. The dense chooser
signal is not wanted: the forward stays a tree (plan §12.2); sparse
presences are subsumed by round 3's sparse superposition.


Operators round 2b corrects the keep to reconstruction alone and trains the
answer reader on both detached trials; credit still uses `R+E+A`. It preserves
activation magnitude at the pole handoff and audits all consumer paths. The
closing image remains landed as a mechanism, with zero complement in the gate
fixtures. The [round-2b receipt](benchmarks/2026-10-06-operators-round2b/README.md)
records the live MM comparison and the residual signed interpretation before
disjunction. Centroid placement remains round 3; its final containment cap
will use decreasing part count and the new before/after audit. Higher-order
order preservation is conditional on monotone folds, not added here.

The round-2b call-profile check distinguishes the configurations: `MM_xor`
bypasses all four pole consumers; `MM_grammar` reaches both reference-slab paths.
All three paired MM_xor trajectories still differ from the landing. That
difference is recorded without attributing it to a consumer that did not run.


Operators round 2c installs magnitude interpretation on forms and signed
interpretation on meanings. It restores one reader update per sentence:
both roots averaged only for compose departures, otherwise the kept root.
The concept-only image remains a mechanism with zero width in the grammar
gates. Narrowing negation has zero advantage there until meanings exist;
its shared SCG term remains available on configurations whose costs can see it.
The [round-2c receipt](benchmarks/2026-10-06-operators-round2c/README.md)
records class 5/10, reconstruction 10/10, sum 10/10 at the quarter floor,
and live MM_xor 9/10, with a green sweep and zero ownership, displacement
and sentence-path perception gradients. It misses the standing class gate;
§20 subsequently excludes its bisection-proven RNG-only MM path. The round-2c
candidate is not accepted. All final grammar policies
use conjunction; all narrowing credits tie; the one-step reader contract is
confirmed by its optimizer counters. MM_xor's path is now isolated: retired percept sampling
changes the generator state used by the later input-reconstruction mask.
Centroid placement, higher-order order enforcement and meaning connectives
remain in their subsequent rounds.

Operators round 2d changes the sentence departure to a uniform walk draw,
then a uniform eligible-round draw within that walk. Its `K·R_walk·W`
correction preserves the estimator and the existing owner. The reader,
keep, costs, image and form/pole separation are unchanged. Plan §20 applies
the RNG-only amendment to MM_xor after repeating its first-forward bisection.
The [round-2d receipt](benchmarks/2026-10-06-operators-round2d/README.md)
retains the thirty unseeded gate runs, paired replays and prior receipts.
No centroid, meaning-connective or other later-round work is added.

The round-2d mechanism is verified, but class **2/10** misses the standing
7/10 gate. Reconstruction is **10/10**, sum **10/10 at ¼**, and raw MM_xor
**10/10**; the repeated bisection confirms its RNG-only exclusion. All final
roots are conjunction and all labels are correct. The all-disjunction starts
switch permanently by epochs 22, 65 and 106, while three failed runs already
use conjunction throughout. The eight class MSE misses remain in the receipt
with their cost comparisons. Walk proposals are balanced, narrowing advantages
tie, reader counters confirm one update, the sweep is green, and the ownership,
gradient and displacement audits are zero. Round 2d is not accepted under
plan §21; later-round work stays carried.

Operators round 2e separates the presented reader from the comparison
reader. The former trains once on the reconstruction-kept root only; the
latter retains 2d's paired compose-root training and supplies only the
advantage's answer cost. Both remain answer-owned and read detached roots.
The [round-2e receipt](benchmarks/2026-10-06-operators-round2e/README.md)
records their MSE on the same kept roots, reader norms, operator flips and
the late cost comparisons. The keep, walk-first departure and SCG correction
remain unchanged. Earlier receipts, Claude's plan and Philosophy are preserved.
The candidate is held uncommitted for Claude's review.

Measured round 2e: class **9/10**, reconstruction
**10/10**, sum **10/10 at ¼**, and raw MM_xor
**10/10** under the repeated RNG-only bisection. The measured
standing gate is met; acceptance remains pending Claude's review.
Final evaluation has 40/40 conjunction roots and 40/40 correct labels.
Every run uses greedy conjunction throughout from epoch 140 at the latest.
The receipt separates 1 miss categorized as late flip,
0 reader plateaus and 0 other misses, with both
reader trajectories and late trial costs. All twenty sum/XOR runs have
400 updates per reader, zero presented weight on the rejected root, and
final active reader Adam counters of 400. The complete source sweep is
green; ownership conflicts, sentence-path perception gradients and
code displacement are zero. All thirty unseeded gate trainings are
retained without retries or replacements. No commit or push.
