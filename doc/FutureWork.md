# Future work

Directions that are decided but not yet specified or built, each with the
decision that opened it and the reasons that constrain it. The current design
is in the other documents; what is scheduled, and in what order, is in
[todo](../todo.md). Where an entry is scheduled its plan is linked: the
[last-architecture-round specification](plans/2026-11-11-item1-8-last-architecture-round.md)
(specified and not yet built) and the
[test development plan](plans/2026-11-09-item0-5-test-development.md). The
main training run is the full training session, where the learning
measurements named below are taken. Section-local future-work lists elsewhere
([nouns from PartSpace, adjectives from WholeSpace](DevelopmentHistory.md#future-work-nouns-from-partspace-adjectives-from-wholespace),
[SymbolFirewall](SymbolFirewall.md#future-work-out-of-scope-for-this-pass))
stay where they are.

## Learned identity components: the gates still owed

The mechanism is built: gradient sparse coding for the change columns, with
individuals as native concept rows
([Gradient flow](GradientFlow.md#trained-total-and-diagnostic-objectives)).
Learned identity, verb reuse and natural determiner cues remain unproven until
the trained checkpoint of the main training run, on all declared seeds. The
gates that would show them: held-out anaphora against the earlier rule-based
binding at matched exposure; verb reuse, one change column for one event kind
across held-out streams; prediction error against a detached-target control;
shuffled-order and renamed-vocabulary controls; and the determiner cue, `bind`
preferring a new referent after an indefinite determiner and an earlier one
after a definite, against a shuffled-determiner control. The harness for them
is the test development pass's.

Two limits stand whatever the gates show: the components say which individuals
are present and which change occurred, not who did what to whom, which stays
with the closing's role slots (and co-variation cannot separate two things in
one frame); and they read only the conceptual content band, never the `.where`
and `.when` bands. The detached role view carried beside the native occurrence
anchor is transitional and is retired in the final housekeeping.

<a id="placement-from-the-registered-priming-surface-one-metric-for-sbow-diffusion-and-bind-alec-2026-10-20-proposed-with-the-objective-to-decide"></a>

## Placement from the registered priming surface: the second-order context (Alec, 2026-10-20)

Alec: "can the priming surface be used to do the SBOW after registering
the context on that surface? I think there is a shared metric here that we
are not taking advantage of." The objective was decided as proposed on
2026-10-22 and is built: at each kept close a word's meaning code takes one
step that raises its meet ratio with the placed codes the seen surface
activates and lowers it against sampled negatives, in the one conceptual space
of [Orders
§10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21)
(`MereologicalCodes.place_rows`), so the overlap that drives `bind`'s content
term and placement is one quantity.

What remains is the second-order context. Skip-gram with negative sampling
factorizes the shifted co-occurrence PMI (Levy & Goldberg 2014); random-walk
embeddings factorize the *diffused* co-occurrence, the sum of powers of the
normalized adjacency (Qiu et al. 2018, NetMF). A registered surface that
diffuses is that object, and reading it would give placement the words that
never co-occur but share neighbours. As built the seen surface does not
diffuse and placement never reads the predicted spread, so its context is
first-order co-occurrence weighted by recency. Two constraints hold for any
return of the diffused context:

- Placement reads what was observed, never a prediction computed from the
  codes it places, and in conceptual space's own metric, never the percept
  cube's: forms and meanings do not share a metric
  ([SymbolFirewall](SymbolFirewall.md)).
- Collapse is the risk: a code that is everything overlaps everything. The
  negatives are the brake, and the measurement is the collapse diagnostic: the
  mean meet ratio against random codes must not rise while the ratio against
  context rises.

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
| Forgetting | value and bounded history age under pressure (§2) | same pressure-driven pass (§2) |

The profile is a switch over documented behaviours, never a hidden
heuristic; each behaviour has its own test under both settings. Params.md
gains the element when the first profile-dependent behaviour lands. The
element is not built, in `model.xml` or in code.

The default belief does not wait for it. Alec, 2026-11-11: "Trust of percepts
is 1, seeing is believing. After word-concepts are turned into
object-concepts, though, trust is uncertain (0) to start with." That is the
`optimal` column and what is built (content embedded in a supply enters with
both lanes at zero); the last architecture round keeps it
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
section 3.7).

## 2. Forgetting and consolidation

The forgetting pass is built: retention by value under pressure, wording lost
before rows, compaction of deleted entries and conservative concept
reclamation ([STM §10](STM.md#10-ltm-as-the-chain-of-stm-end-states),
"Forgetting"). What remains:

- *Sizing for the long run*, and the learning measurements at a mature
  checkpoint (the main training run). Sizing must leave room for a whole
  document between passes, because admission raises at capacity rather than
  write part of a sentence's graph or forget inside a document.
- *Consolidation.* Nothing merges near-duplicate entries, and concept
  reclamation takes only exact redundancies. Learned consolidation and extra
  weights by truth kind or use are not decided, and no learned-consolidation
  claim follows from the mechanism tests.
- *Cost.* The pass compares every row with every row; the
  [cheaper pass](#a-cheaper-forgetting-pass-running-luminosity-on-ltm-alec-2026-10-24)
  is the last architecture round's.

## 3. Lossy reconstruction trace; inversion as learning

**Decided (Alec, 2026-09-28).** A completed row stores its end state, never a
derivation. The operation record serves reconstruction and exploration only
while the sentence is open, then is discarded, so there is no durable trace to
decay under either profile, and reading back is generation from the stored
structure. The earlier proposal to drop derivation detail gradually is
superseded.

What remains is detail before rows, in the order wording, then structure,
then gist (Sachs 1967; Reyna & Brainerd 1995). The first stage is built in the
forgetting pass: under pressure an entry loses its retained source text, which
it does not need, since it regenerates from its stored structure. Coarsening a
row short of deleting it is not built; §8 has the `human` profile's form of
it. Experiments that make reconstruction harder while a sentence is open would
need a separate learning specification.

## 4. Context across documents

The hard reset at a document boundary clears transient context. People
carry context across texts and conversations. To specify: what survives a
document boundary under the `human` profile (the expectation observation
view, priming heat, the What interaction slots), how a new document's
first sentences are scored against a carried context without penalising a
genuine topic change, and whether the boundary becomes a learned soft
signal rather than a cursor fact. The document cover of each stream and the
clause-scoped reset are the base.

**Direction (Alec, 2026-10-08).** Context is content to be learned, not state
to be managed; the LTM is shared, with identifiers that separate rows. What
survives a boundary under the `human` profile is then what the learning
stages teach, not a specification here. The stages are the five lessons of
stage 7, "Context as content", of
[the generated supervised curriculum](Training.md#the-generated-supervised-curriculum-in-order-alec-2026-10-09):
batch differentiation, return after interruption, shared truth across
documents (situated against absolute rows, decided by credit and never by a
flag), priming as a prior, and expectation from the right history. Their
learning gates are read at the checkpoint of the main training run.

## 5. Definition rows and interpretation-time discrimination

A word's definitions are `word DEF object` rows, several of them expressing
synonymy and polysemy, and the grammar selects among a word's objects from
context at interpretation time
([STM §10](STM.md#10-ltm-as-the-chain-of-stm-end-states), "Definitions";
decided 2026-09-29, in place of an n-ary META row). The remaining learning
question is how the chooser ranks those objects in context and credits a wrong
selection through expectation and reconstruction. The math corpus does not
exercise it: numerals are opaque number words and it writes no definition
rows. A forgotten definition leaves its word new when it is next read; any
bias in forgetting by truth kind is undecided (§2).

## 6. English arithmetic Q/A curriculum

**Proposal (Alec, 2026-09-20).** Generate English questions and answers from
bounded mathematics to supply output-training data and reproducible held-out
evaluation, for example "What is two plus two?" answered "Four." The first part
exists: the graded math rows (`data/math/`; number words zero to
twenty, counting facts, worked successor steps, and questions whose answer is
a chain), in which mathematics enters only as language
([Training](Training.md#the-generated-math-suite-what-it-exercises-and-what-it-cannot-2026-10-20)).
Whether the model learns the chain is measured in the main training run.
Addition that reaches unseen pairs is thinking: the recursion
$a + S(b) = S(a + b)$ is a policy the chooser learns from worked steps, not a
stored rule, and two controls isolate what thinking adds, the answer line
withheld (expectation credit alone) and a zero attention budget (the verb
phrase alone).

What stays here is the larger curriculum, once the chain is learned:
multiplication, multi-digit numbers, subtraction and word problems.

- Compose addition from succession and multiplication from addition. Common
  sums and products may be memorized through ordinary learned associations
  and memory; evaluate that separately from deriving unfamiliar combinations.
  These are grammatical meanings selected by the existing MLPs.
  Arithmetic on concept coordinates or symbol addresses does not implement
  numerical addition.
- Beyond the initial vocabulary, build number NPs compositionally using
  decimal place value and carry, with addition and multiplication as compound
  VPs. Teach English number names alongside this structure, including
  irregular forms such as "eleven" and "twelve"; English spelling is not
  itself the place-value algorithm. Do not allocate an independently
  memorized atomic NP for every larger result. Variable binding and algebraic
  identities need their own later lessons; decimal notation alone does not
  establish those abilities.
- The desired answer and the oracle calculation ([exact.py](../bin/exact.py))
  are scoring targets, never an answer seed or a runtime calculator, and no
  arithmetic controller or interpreter runs in the model. What thought hands
  to generate is an idea composed over existing concepts, never a free answer
  vector, so the output loss never trains the inverse on ideas that compose
  cannot produce. A lesson that supplies an already encoded, on-manifold idea
  to train realization is useful too, but is identified separately from a
  test of answering the question.
- Generate separate training and evaluation sets with fixed seeds and
  partitions over mathematical problems before applying wording templates.
  Keep equivalent commuted problems together when evaluating unseen
  arithmetic, and keep memorized-fact and paraphrase-only results
  identifiable. Reserve new operand combinations, complete question wordings,
  larger composed numbers, carry patterns and derivation depths for distinct
  evaluations. All base successor facts may be taught; repeating them in
  evaluation measures acquisition, not generalization. Score answer meaning
  as well as English realization through the full vocabulary. Retain
  renamed-vocabulary controls for arithmetic meaning, and the matched-compute
  comparisons before claiming useful learned reasoning.

Exact arithmetic is the testbed because correctness, progress and the value
of each intermediate result are decidable, so whether thinking makes progress
can be measured rather than asserted. The
[successor-VP probes](../test/test_verb_successor.py) establish local operator
behavior under their own setup; they do not establish this curriculum.

## 7. Episodic memory: a few active indices beside each row

**Built (2026-11-11).** An episodic entry is the order-zero concepts of a
moment, kept beside the sentence's `LTM0` row: up to `episodicConcepts`
concepts of the open attention pass, as references in reading order with one
signed value each, stored detached with the row's stamp, and reviewed by a
recall by time, which publishes them as the querying stream's field
([Architecture](Architecture.md#episodic-and-semantic-memory-in-the-two-systems-2026-11-11);
[STM §10](STM.md#10-ltm-as-the-chain-of-stm-end-states)). Alec, 2026-11-11:
"it will clarify the architecture to store concepts instead of percepts. That
will allow us to do animal cognition (system 1 reconstruction) without
calling compose and generate."

The reason for an entry is the one given when it was first proposed (Alec,
2026-09-20 and 2026-09-28). LTM holds serial form only: a row is one idea, and
several things at once can be stored only by chaining, which serializes what
was simultaneous, strains generativity as the chain grows and keeps nothing
that was active but not composed. Composition commits to one reading, so a
word's other senses and the candidates that nearly won are lost unless they
are kept beside the row. A stored idea is a story, semantic memory; storing
the activations of all symbols would need a massive bandwidth, so a few of the
strongest stand for them.

What is open:

- **The cue by content.** A read from a concept to the entries in
  which it was active, which needs no grammar. The last architecture round
  specifies it as the query on concepts: the cue is a field, a row's match is
  the coverage of the cue in the row's references, and an index from concept
  to rows is kept
  ([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
  section 3.8).
- **Narrowing by place.** Whether a recall also takes a `.where`
  bracket. An entry keeps no place (an earlier proposal, 2026-10-09, kept a
  bank of located fields, each candidate carrying its `.where` and `.when`
  extent), and the last architecture round decides that nothing in memory is
  addressed by time or place, so in that round the recall by time gives way to
  the query on concepts.
- **Review.** A review displaces the querying stream's present field
  until the next open attention pass, and a reinstated field keeps the stamp
  it was stored with; this is built on Claude's reading, which Alec may
  override. Open: a bound or a decay on a reinstated field, so that
  remembering differs from knowing, and whether learning or expectation should
  read a field that is lived again but not observed again (an entry is
  detached and trains nothing).
- **Transfer.** The operation that takes two entries and returns a
  duration, and who reads it.
  [Kaufmann and Viera (2026), "Temporal cognition in apes"](https://doi.org/10.1080/00048402.2026.2649411)
  count informational transfer among the marks of a temporal representation,
  with systematic and flexible use: a duration with a count gives a rate, a
  duration with a velocity gives a distance. They propose no memory
  architecture, so this is a consistency check and not a derivation. `.when`
  is read today as an age by forgetting, as a distance by the recall by time,
  as an order by `relation_evidence` for `'implies'`, and as the present by
  `ActivationPredictor`. No computation takes the duration between two
  remembered events as a quantity for another computation; episodic entries
  supply its operands, the same concept at two stamps
  ([`.when` as a duration](#when-as-a-duration-store-only-the-deltas-alec-2026-11-10)).
- **The earlier design.** The indices first proposed were the most
  active percepts or concepts at the moment a row was written. An entry now
  holds the strongest order-zero concepts with their signed evidence
  $d = c^+ - c^-$. The dissonance $m = \min(c^+, c^-)$, which a stored
  resultant cannot carry (both and neither land on the same point), is not
  kept: a reading whose two poles are equal leaves no reference. Whether it
  should be is open.
- **Forgetting.** An entry follows its row through compaction, forgetting and
  the epoch reset. Open: whether an entry is detail that goes before its row
  (§3).
- **Senses.** An entry's use as evidence for discriminating a word's senses
  at interpretation time (§5) is not built.
- **System 1 alone.** An entry is taken only where the open attention pass
  runs and is written only at a sentence's close, so a configuration that
  composes no sentences keeps none. Whether a mind with System 1 alone should
  keep and re-enter episodes is open.

## 8. Recall as estimate plus residual

**Direction (Alec, 2026-09-20).** What is conceived of a sentence is read
against what was predicted of it. The store keeps the two terms as a linked
`estimate` / `observation` pair with the residual derived, and the
observation row keeps the full idea
([ExpectationRetention](ExpectationRetention.md)); retention reads the same
pair, so a fully predicted sentence has low surprise and is forgotten first.
That is the `optimal` layout.

The `human` profile goes one step further, as a stage of forgetting's detail
before rows (§3):
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
the non-object and puts nothing in its place.
So attention needs no prediction in order to focus, and the architecture
already has it in that form: the reading scope, and the intent channel of the
priority surface at the concept pyramid's per-order top-K (floor of zero,
never a veto), read as the pyramid's ranking score (`_assemble_relevance_priority`)
on the parallel path.

What is not built is the interaction at the level of rows. Expectation's row
gain applies when thought reads the closing and spares the open grammatical
objects, where it cannot touch composition
([ExpectationRetention](ExpectationRetention.md)). At the level of rows an
exclusion would make an expected object quicker to recognise *within* a
sentence: expected non-object rows of order $\ge 1$ lose priority, so the
object competes against less. The exclusion acts in selection, not in content.
A readout that depends only on relative scores (top-K, arg-max, softmax) is
unchanged by a shift common to all candidates, so lowering every non-object by
$\delta$ selects what raising the object by $\delta$ selects,
$\mathrm{softmax}(z - \delta\,\mathbf{1}_{\neg X}) = \mathrm{softmax}(z + \delta\,\mathbf{1}_{X})$:
facilitation is reduced competition, not a boosted signal. Written into
activations as a signed value, the exclusion would be an affirming negation
and would fill the mind with conceived absences of everything unattended.

To specify: how the estimate, an idea, becomes a priority over rows (its
nearest rows, restricted to rows of order $\ge 1$ by the ramsification table);
what sets the object of observation at this level (conceptual activation as
the origin of reading attention,
[Architecture](Architecture.md#symbolic-weights-reconstruction-parse-time-attention-2026-06-30));
and the price, which is the reason this waits: selection at the readout can
change what is composed, so an exclusion there is not neutral to content by
construction. The residual must still be learned from what was there, not
from what was selected (statistical learning needs attention: Turk-Browne,
Junge & Scholl 2005), which may mean composing twice. Decide after
measurements with the expectation gains on show whether sparing at the
closing is enough.

## 10. Fold-ladder tuning questions

The form of the meronomy fold ladder's settings is settled
([Spaces](Spaces.md#fold-ladder); [Params](Params.md)): the admission radius
(`admissionRadius`) and the division threshold (`lbgThreshold`) are one
geometry setting, the coverage schedule shares the thought loop's closing
pressure, and `boundaryTypes` chooses the start of the learned boundaries.
Their values remain tuning, to be read from the full training session.

## 11. Teacher-to-LTM persistence

Deferred until the NP-VP transition cache exists.

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

A roadmap that follows the clean Teacher throughput gate (the clean B24 median
throughput within 15% of the matched baseline). Every item keeps the
coordinate contract of the
[What and spacetime design](WhatSpacetimeDesign.md#2-questions-data-coordinates-and-time),
which also describes the interface ownership and the chooser and stack
context: the coordinates of a requested answer are part of the question and
condition the answer, never a privileged address passed beside it, and `Data`
never writes the model's `.where` or `.when`.

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
- Further rungs of a question's coordinates: a span within the sentence as a
  finer rung of the same `.where`, a corpus byte address as a coarser one, and
  a document-boundary bit for `past` questions asked at a document's start.

## 13. LTM enhancement: episodic content and summary statistics beside the semantic row (Alec, 2026-10-19)

**Direction (Alec, from the conference).** A stored row should hold some
episodic content, not only the semantic vector, and some summary statistics
such as surprise and use count. The question behind it: to which of our own
actions do we have conscious access afterward, since a metacognitive memory
is what lets later control improve on earlier performance.

**What a row carries and what it does not.** It carries `surprise` at write
time, the access count, the two evidence lanes, the writing stream, its
`.when` stamp and the episodic entry of §7. It does not carry a count of
*reads* (retrievals by the thought loop or priming, as opposed to
re-witnessing), an effort trace of the reading (compose steps, departures
taken, keep-cost margin), or an outcome mark set later (an answer that
failed, a later contradiction).

**Human evidence (a summary; the citations were not re-verified).** People
retain the contents that passed through working memory, in order, and
evaluative summaries of their own processing: confidence, feeling of knowing,
error awareness, effort and fluency, and a trace that an item was
self-generated. They do not retain the process: memory for one's own
fixations is near chance, verbatim form is lost beyond the clause and
regenerated from meaning at recall, syntactic choices are not reportable, and
explanations of choices are reconstructed. So the human-like record is the
stream history plus a few summaries per row, not the attention path and not
the kept program; the surface is regenerated from the idea by the generate
grammar.

**Proposed shape.** Beside each row: (a) the episodic part, §7's entry;
(b) summaries: surprise and the access count (both exist), a read count,
effort (steps or departures, keep-cost margin) and an outcome mark; all a few
numbers, detail first in forgetting. Consumers: forgetting's value function
(surprise already feeds utility), the chooser's control (where to re-read,
when to explore), and interpretation-time discrimination (§5). Not recorded:
the `.where` trajectory of a reading and the kept program. The summaries are
unscheduled.

## Four corners as prompts

The four corners are the values of a predicate over a thing, derived when
read from the concept's evidence pair and never stored
(`ConceptEvidence.corners`): true, false, both (its parts differ under the
predicate, a fact about composition and not a contradiction) and neither
(unknown). Neither prompts attention; both prompts division of the subject
into parts on which the predicate has a pure reading. Division is pi in the
order-0 field. A higher-order symbol cannot split; descend to its cases and
their field first. The refinement and raise route is built: contiguous support
stays in field refinement, while discontiguous support may be symbolized above
order 1 only after local refinement stalls
([Architecture](Architecture.md)). Two policies remain.

- *Attention targets from a neither reading.* Choosing new attention targets
  from a neither reading is future policy work.
- *When to refine the concept.* Alec: "heterogeneity is a cue to narrow
  attention, or to learn more", at two timescales. Within a reading, *both*
  over a bracket cues division until the parts are pure; that is the walk's
  law today (`narrowing_mask`: both permits divide, pure permits gloss,
  neither permits descend). Across occurrences, a *both* that **persists at
  the narrowest extent** cues refinement of the concept into parts where the
  predicate is uniform (at order 0 a distinguishing mint or a property split,
  above it a sub-concept), and *neither* cues witnessing. The policy owed is
  this second cue: when to stop dividing the reading and refine the concept
  instead, and what counts as persistence (a candidate counter, undecided:
  the magnitude of definedness, $m = n/(n+k)$). Candidate attention's support
  mask reads the same pair per candidate, so the policy is a question of
  training that mask.

### Shared-mode erosion measurement

The hypothesis comes from the qualified human evidence summarized in
[Philosophy](Philosophy.md#attention-as-one-bracket-both-as-the-fields-report-and-the-sharing-of-the-two-modes-2026-09-25):
the serial and the parallel mode share one conceptual structure, and each
erodes the other somewhat.

The first measurement compared serial, parallel and interleave:2 from the same
initialization for each of three seeds, with symbols offline and with their
owned reverse-pi read-back, on four XOR probes and 68 word probes from 20
FineWeb launch documents, after one pass over four XOR inputs and one complete
sentence per document. FineWeb categories are native orthographic properties,
not semantic categories.

On all three seeds the FineWeb native readings have lower categorical
separation and higher within-category discrimination after parallel training
than after serial training. The wider prediction does not hold: parallel
training admits no more structural alternatives; label read-back lowers rather
than raises categorical separation in every FineWeb condition; interleaving
does not keep separation near the serial value while retaining its predicted
within-category gain. Native XOR separation is zero in all conditions. This
short measurement does not establish the proposed erosion-and-restoration
mechanism, and the code-level results do not validate the psychological
account. No seed, threshold or training duration was adjusted to make an
ordering pass. It ran N serial sentences, then parallel processing, then a
label read-back; the schedule has since reversed the first two steps and
removed the third, so its interleave rows do not describe the current
schedule, and the three-seed job is retired from the test suite.

What remains: more training and semantic probes are future experiments, and
the training harness's per-interval logger is to log the
categorical-perception index from
[CategoricalDiscrimination](../bin/CategoricalDiscrimination.py) on the
[fixed four XOR and 68 FineWeb probes](../data/categorical_discrimination_probes.json).
It accepts captured readings and returns
$CP = \bar D_{between} - \bar D_{within}$, the mean L2 distance between the
readings of probe pairs that straddle a category boundary minus the mean over
pairs inside one category, each unordered pair counted once, with both means
and the pair counts. It does no training or extra label read-back, and
unknown/zero readings remain in the metric. The harness must budget the cost
of collecting probe readings when choosing its logging interval. This is a
descriptive measurement with no required ordering or threshold.

## Throughput levers for the serial loop

The compiled July baseline ran at 6.803 sentences/s at B24;
the September eager, batch-1, expectation-on measurement ran at 0.26. The
per-word arithmetic is of the same order as a depth-4 NanoChat's per-token
arithmetic (about 40M parameters touched either way); the gap is scheduling.
Compare by bytes seen and wall time only
([pilot](DevelopmentHistory.md#question-and-falsifiable-first-milestone)).

1. **Batch across sentences.** The word loop is a recurrence, so positions
   cannot be parallelized, but sentences can; recurrent training scales
   nearly linearly with batch until compute-bound. The pilot config still
   defaults to batch 4.
2. **Known-word lookup (Alec).** Known words are looked up in PartSpace, not
   translated: the word is the recurring unit the fold ladder admitted, and
   `interpret` begins from that word row, so any shortcut belongs to the
   lookup. If the lookup needs a cache, or a hint about when to stop
   synthesizing over bytes and return the word's code, add it as an
   optimization concession. The byte ladder then runs in full only for
   unknown words, which is where `interpret` mints new objects.
3. **Subsample reconstruction.** The tied reverse walk with its candidate
   search costs about as much as the forward; run it on every k-th batch or
   a subset of rows through the existing placement knob, with the
   reconstruction measurements re-baselined for the chosen k.
4. **Close the host islands.** Vectorize and compile the eager per-row and
   per-edge loops (`cs_read_memberships`,
   `_compose_order0`, `_reverse_field`, `refine_over_collected`,
   `_prepare_part_learning`), and move the host-side
   COO dictionaries of the sparse stores to device CSR tensors so the whole
   word step compiles as one graph. The last such pass, the 2026-07-06
   vectorization of `stage_analysis_spans` / `property_spans`, bought a
   third of an epoch byte-identically. Further candidates: the host
   dictionary walk that gathers a ragged field, event attribution, testimony
   admission and retirement, and context scheduling.
5. **Vectorize the effective-evidence refresh.** `LTM1.refresh_effective`
   recomputes the effective pair of every live row, in `relation_order`, in
   Python whenever a radius, a stored pair, an operation measurement or a
   gate changes. That is adequate at present capacities; it can be vectorized
   by order level, all rows of one dependency level at once.

Compounded, these put the serial loop in the tens of sentences per second,
NanoChat's order on the same machine; parity is not claimed.

**Lookups, folds and the percept pass (decided 2026-10-30, not built).**
Scheduled before the compiler work, so that the word step compiles under
`fullgraph`. It takes up levers 2 and 4 above where its steps name them; the
other levers remain candidates for the optimization pass.

1. The sparse stores (`layer.features`, the membership matrices) get
   preallocated device CSR views of fixed edge capacity that raise when
   exhausted and never resize. `MereologicalCodes.derive` becomes one
   segmented `scatter_reduce` per order (`amax`/`amin` over the subspace
   bounds, the same $q^{w}$ and $1-(1-q)^{w}$ folds as `fold_presence`) over
   the rows whose validity bit is clear; an optimizer step clears only the
   bits of rows whose edge values changed.
2. A content lookup is one product of a cue against a content matrix, as the
   `LTM0` cued read (`EntryRead.cued_rows`) already is.
   `MediumTermMemory.match_candidate` and `FamiliarTier.witnessed_match`
   become one cosine of the support centroid against the masked codebook
   rows of the same order, top-1 over `conceptMatchCos`, with support-set
   equality as a mask and only the literal key matched exactly. `LTM1`'s
   index is maintained on `add`, not rebuilt.
3. The percept pass runs once per sentence over $[B, W, P]$ with gradients,
   with ragged parts (a CSR of part ids per word) in place of the 64-wide
   padding. A known word reads its identity row's `amax` instead of
   recomposing its atoms, and the arrival state (a causal prefix mean and a
   shift) and the forecast are computed for all words at once, with the
   forecast's inputs and targets detached.
4. An explore path resumes from `SentenceFork`'s recorded state at its
   departure word instead of rerunning the shared prefix with masking.
5. Every rewritten brick traces under `fullgraph=True` with no host read.

Results must equal the current path: the lookups return the same ranked
candidates and scores at the same work meter. Grammar operations stay
serial; tokenization and new-type admission stay on the host once per batch,
cached per word type.

Step 4 is the snapshot Alec asked for on 2026-09-30, when the paths of a
sentence came to be costed under the same parameters before one update:
"let's opt for making this efficient; maybe even a snapshot at where they
branch, if that helps significantly (the snapshot overhead may be high, in
which case don't)." Each explore path repeats the exploit path up to its one
departure, so its prefix is computed again. Measure the time and memory of the
prefix recomputed against a snapshot of the reading's state at the branching
round, and keep the snapshot only if it saves significantly.

## `.when` is redundant across the elements of one input (noted 2026-09-26)

Every element of a sentence carries a `.when` band that follows from the
sentence's one tick. Carrying one shared bracket per sentence could avoid
repeating it on every element. It must preserve located binding and the
captured program. Not built.

## Global coordinate transport through learned operators

The `.where`/`.when` band is transported through the grammar's operators and
their inverses, which is what makes it a positional encoding in the
transformer sense. The two-rung ladder exactly decodes clean registry
addresses. Its tolerance to noise after learned composition and inversion
remains a separate question. Reduced periods were refuted as a remedy for the
loss jump of 2026-09-26: the extra cost came from scoring a shared timestamp
on every word, which the occurrence and time correction removed
([Spaces](Spaces.md)). More rungs or rotary transport therefore need their own
measured positional-error problem and evaluation.

If transport noise proves to matter, the ladder generalizes to $k$ rungs with
a fixed ratio of 32 between neighbours and byte resolution at the finest;
decoding resolves branches from the coarsest rung down, so every cell keeps
the same angular margin over the transport noise. The band grows to $2k$
dimensions: three rungs cover 262,144 addresses and four cover 8,388,608, and
the same construction applies to `.when`. If more rungs degrade transported
accuracy, the remedy is rotary transport: each operator applies position as
an exact rotation of content, learned maps act only on content, relative
offsets come from products, and exact decoding reads only untransported
bands.

## The dynamic stop: glossing above words, descending at novelty

Attention is one mechanism, a bracket over the input, narrowed until each
bracket's encoding is pure, with the stop pinned at words. The dynamic stop
removes the pin: a known multi-word unit that reads purely at its wider
bracket is glossed as one symbol, and narrowing descends below a word only
where the reading is *neither* (unknown content) or *both* (heterogeneous).
This is speed reading that slows at novelty. It is decided against the
optimization pass's per-word throughput baseline: it pays for itself only if
glossing known units saves more than the open read costs, which on FineWeb
should hold once the recurring units are admitted. It depends on the field's
*both* being reliable; the categorical-discrimination index and the both-rate
per bracket level that would show it are for the training harness's
per-interval logger and are not logged yet.

## Modality as the third index (noted 2026-09-28)

Words narrow the domain of discourse along three indices (which thing, which
stretch of time, which alternative) and individuate along each by determiner,
tense and modal ([Language](Language.md), "Parts of speech are roles"). The
third index has a first mechanism: the modal is the second difference, a
diagonal per modal in one shared basis, learned from equal-time events on an
identity's alternative timelines, and a modal clause withholds the lowering
(`ModalLearning`). Alec, 2026-09-29: "To square, now: modality is not order,
it is dimension; I think these are different." Concepts stay opaque: the
dimension is a learned subspace, not a band of a concept's code. Open with it:

- which modal sentences are a modal phrase within the point and which an
  operator row over another row, the way "it is certain that P" is carried:
  an operator row over the row of P, so that modality never sets trust;
- how a modal row differs from an asserted one without borrowing trust, which
  is univalent and is not evidence;
- whether "every", "always" and "must" share one mechanism, since each takes
  all along its index and writes a relation instead of lowering;
- the modal phrase as a dimension of its own beside the noun phrase and the
  verb phrase
  ([below](#lift-and-lower-as-dimensions-future-work-alec-2026-09-29)).

Smaller items from the same discussion, none of them scheduled:

* **The verb's band is hard-coded.** $\tau = 0.1$ and the clamp $\pm 8$ fix
  what a verb can mean: an effect below the threshold is exactly the
  identity, and one above the clamp is unreachable. By the live-wiring rule
  they are `model.xml` parameters.
* **Verb sparsity as a measurement.** On the complement of $w$'s support the
  noun passes through exactly, so $\lVert w_v \rVert_0$ bounds what can be
  recovered without the derivation. Log its distribution over the learned
  verbs, and whether the touched dimensions cluster, as a single-domain
  constraint on verbs would predict.
* **Modifier plus noun.** A compound selects among its head's cases
  ([Language](Language.md)); modifier plus noun as a selection over the
  concepts in the count noun's extension is the last architecture round's
  ([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
  section 3.5). `LiftLayer.lexical_gate`, which would give a word's code a
  slice of the operator, has no callers.
* **What order measures.** Order is "determined by sigma or pi fold over a
  class" (Alec, 2026-09-28) and is not part of speech: a set of discrete
  concepts is one order above its members, and a part of one concept's
  extension keeps its order ([Orders](Orders.md)). Open: whether parts made by combining what is not symbolic are
  always contiguous, which Alec put as "perhaps"; and what happens to a part,
  "blue cats", when it is named and becomes a discrete concept that can be a
  member of a set.
* **Generic against universal.** "Cats sleep" tolerates exceptions and
  "every cat sleeps" does not. In a generic both write the same part row,
  the subject's region as part of the predicate's, and their difference in
  number is the surface's; where their force differs is still open. A cat that does not sleep adds
  evidence against beside the evidence for, the corner *both*, which prompts
  refinement: one exception does not make the relation false.
* **Privatives.** "Destroy", "kill" and "cancel" remove a property, which a
  positive gain cannot do. Like "fake" and "former", they belong to `non`
  or to a shift of the domain, not to displacement.

## Operator names (future work, Alec, 2026-09-29)

"The operator rename can also be future work." Most of the renames have since
been made (`synthesize`/`analyze`, `ask`, `query`, `isTrue`/`exist` as pairs,
the one `attentionBudget`), and the code reads a rule's declared properties,
never its name ([Language](Language.md#grammar)). What remains:

| today | the new name | decided about it |
|---|---|---|
| none | `symbolize`, with `conceptualize` its reverse | lower-level operators, perhaps of thought only; no present operator is renamed to them, they are future operators |
| `verb`, `adverb`, `preposition` | names by what is computed | not chosen |

A rename is made in one pass, in grammar files, code, tests and documents,
with the old name kept nowhere. The retired names still raise with their new
names; removing those one-release aliases is housekeeping.

## Lift and lower as dimensions (future work, Alec, 2026-09-29)

`lift` and `lower` were intended to raise and lower the order; Alec: "perhaps
they are better suited to dimensional lifting/lowering". On that reading
`lift` puts a phrase beside the description in dimensions of its own, noun
phrase, then noun phrase and verb phrase, then those with a modal phrase, and
`lower` is its reverse face, the projection that takes the phrase back out,
while the order is raised by the fold and lowered by its inverse, the
determiner. The count of dimensions is a count of phrases, each in a learned
subspace of its own, so a concept's code stays opaque.

As built, `lower` is the determiner and `lift` is the binary sigma whose
result is a sentence, inheriting its subject's order
([Language](Language.md#grammarlayer-forward--reverse-inventory)). The
dimensional lift is unbuilt. What it needs is one thing used twice: learned
subspaces that keep each phrase to its own dimensions going in and take the
phrases apart coming out, so that the row's one point gives each phrase back
by projection and needs no witness; the reverse is exact when the subspaces
are orthogonal and together span the space. Side by side has a price: the
phrases share one width, each has only its share of the dimensions, and the
reverse is exact only so far as each keeps to its share. The modal phrase is
the same operator applied a second time, so no operator is added for it.
Raising the order is not `lift`'s: a set is made from its members by the
sigma fold over them.

## Objectives that share parameters: drift and interference

The structural split of 2026-10-02 (reconstruction owns perception, the
object codes, the compose operators with their tied inverses and the compose
chooser; expectation owns only predictors; the supplied answer owns only its
readers) survives as the `separateChooser=true` option. The default is one
projected update, in which the answer reaches shared parameters less its
component that opposes reconstruction
([Gradient flow](GradientFlow.md#default-projected-gradients)).
Three things remain.

- *Forward dependence.* With disjoint writers, reconstruction can still
  change the evidence the answer or expectation has learned to read. Disjoint
  writers remove direct competition over an optimizer parameter; they do not
  prove that every block's objective decreases together. Measure that drift
  and compare it with learning speed and read-back. No new norm-matching rule
  is implied.
- *Where interference occurs.* A completed-step diagnostic, not built: at
  each actual sentence update, record each objective's gradient dot product
  with the actual parameter displacement, plus each objective's cost before
  and after on the same targets and recorded derivation, with shared
  parameter groups and private reading-map parameters reported separately and
  selection's effect kept apart from the optimizer's. A favorable
  raw-gradient cosine is insufficient evidence of a favorable completed
  update, because the optimizer's per-coordinate scaling, the proximal
  operation and the parameter projections all contribute to the displacement.
  With $g_R = (1, 1)$ and $g_O = (-2, 2)$ the dot product is zero and the
  combined gradient is $(-1, 3)$, whose negative lowers reconstruction
  locally; under the positive scaling $\mathrm{diag}(10, 1)$ the displacement
  is $(10, -3)$ and $g_R \cdot d = 7 > 0$, so reconstruction rises locally
  despite the orthogonality. A low reconstruction pass count alone cannot
  diagnose interference, and the diagnostic can separate it from discrete
  derivation changes, inadequate inverse learning, or read-back that stays
  poor while cost falls.
- *A joint update, not adopted (2026-11-03).* The alternative to the
  unit-by-unit projection chooses the shared update jointly from all
  objectives: the displacement closest to the optimizer's proposal subject to
  $g_i \cdot d \le 0$ for each active objective, with each private module
  keeping its own update
  ([Sener and Koltun 2018](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html);
  [CAGrad 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html),
  which discusses the risk of stopping at an arbitrary compromise). Its
  conditions, if the diagnostic reopens the question: the calculation is
  joint, since correcting pairs independently can undo an earlier correction;
  local linear constraints do not guarantee a finite-step improvement, so the
  completed displacement, with the optimizer's stateful scaling and the
  model's projections, is evaluated against the same costs, and a rejected
  trial update restores optimizer state and side effects as well as parameter
  values; where no common improvement exists the conflict is reported and
  reconstruction keeps its declared precedence, never a priority chosen by
  gradient magnitude; and gradients are not normalized to unit length, so a
  nearly solved objective does not acquire a large update because its
  residual is small. Neither citation supplies a guarantee for this model's
  optimizer, proximal and projection sequence. The narrower conditions on the
  built projection are
  [below](#conditions-on-the-projection-and-the-pair-policy-2026-11-09).

## Expectation by inversion and the reconstruction audit (decided 2026-10-06, unscheduled)

The invertible-path program, decided on 2026-10-06 and unbuilt. It
is separate from expectation as built, which is local: each level keeps its
own predictor over detached sources and an error bounded at that level
([ExpectationRetention](ExpectationRetention.md)), because unfolding one
expectation down the derivation would couple the levels, one wrong estimate
at the top becoming a wrong expected operand at every round below it. Alec:
expectation "propagates across the model ... only once the path is
invertible; before that its sources stay detached". Three parts:

1. **Expectation by inversion.** The row-level prediction, inverted through
   the committed operations, gives an expected operand at every round; the
   actual against it is the layer's surprise; the maps (verb, adverb, lift)
   take exact targets by inversion: target propagation, no gradient through
   the choice, no straight-through, no mixed forward. The target is the
   better of a sentence's complete derivations, its row inverted through the
   committed operations, so learning runs on surprise along the whole chain
   and not only at the output. A difference between two derivations is
   a layer's signal only where that layer's parameters could have produced
   the better state: the chooser at the departure round, by its
   score-function credit, the readers, and the maps downstream of the
   departure for their share; a difference caused by a different operator
   is not a map's error. The word-level expectation follows from the
   row-level predictor by inversion rather than by a predictor of its own.
   Expectation's sources then go live: the rule detaching them exists
   against collapse, which an invertible path cannot do.
2. **Reconstruction as an audit.** With injective forms and exact unbinding
   through the bank, reconstruction is a structural audit that must read
   zero, not an objective; the decomposition chooser becomes a lookup; the
   trainable surfaces are the chooser, the maps, the predictors and the
   readers, trained by expectation and output along one path. Where "mostly
   invertible" means approximately, the reconstruction term stays as a loss
   for that part. Today identified-word reconstruction is already the
   $R \equiv 0$ audit at gate width; the rest of the path keeps the loss.
   Already invertible: the invertible projections
   (`InvertibleLinearLayer`), the butterfly cascades, `LDUReadout` and the
   operators' tied inverses (the verb and adverb maps given the modifier),
   with the content-addressed bank as clean-up memory. Not invertible:
   binding, which commutes and zeroes the coordinates an operand is silent
   on, so operands return only through the bank search, exact when forms are
   injective and the width suffices, their order from the roles; and the
   hard choice of the chooser. With the forms and binding exact the collapse
   channel closes structurally, since an invertible path cannot discard
   information, and the balance question reduces to expectation against
   output on the shared maps.
3. **The width.** Exact clean-up of $k$ bound items against $N$ bank
   entries needs a width that grows as $D \approx c\,k \log N$, with clean-up
   through the index, never brute force. For an independent Rademacher bank,
   with a bundle the sum of $k$ selected $\pm 1$ codes and a normalized dot
   read, a half-unit margin error at one entry has probability at most
   $e^{-D/(8k)}$, so $P(\text{any error}) \le N e^{-D/(8k)}$ and
   $D \ge 8k\ln(N/\delta)$ suffices. The built identity is not that ensemble:
   forms use boundary-pair and length codes, max/min membership and an exact
   index with collision refinement. The bound therefore neither justifies
   changing the configured width nor certifies $R \equiv 0$. The
   finite-vocabulary identity audit is exact at the configured widths; the
   composed-path $R \equiv 0$ audit remains open, and reconstruction keeps
   training every part that lacks an exact inverse.

The milestone: the audit at zero at production width $\Rightarrow$ the main
training run trains by expectation and output through one path; where it does
not read zero, the run keeps reconstruction as a loss for that part. A
candidate gate, when it is specified: the expectation discrepancy and the
class bars at the standing XOR fixtures with the inverted targets on, against
the detached-source baseline, with the collapse diagnostic (the *both* mass on
the leaf) reported. The split of expectation and top-down reconstruction by
residual, which does not wait for an invertible path, is
[the residual split](#the-residual-split-a-stopped-subtraction-or-the-correction-alec-2026-10-29).

## Compose follow-ups (2026-09-27)

Of what was left over when compose came to take one operation per round, the
training temperature of the compose draws is built
(`architecture.composeTemperature`, with evaluation deterministic), and no
test asserts the shared-operator report any more. Two things remain: a
**sentence parsimony / work term**, accepted in direction for a later
specification, and the rename of the native context-read `exploration_trial`
flag, which is housekeeping.

## One bank: the collision rule for property rows, and a merge rule (decided 2026-10-06, unscheduled)

Alec's one bank over "is X" and "contains X"
([Architecture](Architecture.md#the-scheme-confirmed-2026-10-06-expectation-and-surprise-through-the-architecture),
"One bank, two relations") grows only when
necessary to separate two objects: *contains X* comes from the segmentation
(a word's parts, a sentence's words) and *is X* from the predicates (a
thing's own row, its classes), and the bank adds a new row for a new
identity, the cheapest distinguishing feature (an unshared pair, a count,
the length) when a new thing collides with an old one, and otherwise
nothing. The collision rule is built for word identities (each word's own
positional triple, bits drawn until separated). Not built:

* **Property rows still split on dispersion.** `maybe_split_property_row`
  (`Spaces.py`) splits a WholeSpace property row when assignment variance
  over at least `lbgMinCount` pulls exceeds `lbgThreshold`. The decided rule
  replaces it: a thing's code must differ from every other thing's; when a
  new thing coincides with an old one, mint the cheapest distinguishing
  feature, and otherwise nothing.
* **A merge rule.**
  Collision-minting only separates, and its inventory depends on the order
  in which words are read (inventories differ run to run). Abstraction
  comes from the sigma fold one order up, from co-activation, and from
  pruning by value the distinctions that are never used. The last is not
  built for the concept inventory (order-0 definitions, alternatives,
  feature groups, mints): concept reclamation takes only exact redundancies
  (§2).
* **Revision conditions formation (Alec, 2026-10-08).** A mint made on
  an exhausted search is provisional: identities may be broken or
  associated after the fact, and "that alteration should condition the
  probability with which we formed the identity in the first place", a
  delayed credit to the recorded bind, mint or open choice through its
  provenance. There is no local metric for how many things there are; the
  identity components over enough occurrences are what can decide it.
  Breaking and associating are LTM operations over rows and references, the
  merge rule above and its converse, and both leave the formation record in
  place as provenance. That record keeps the choice and the probability the
  chooser gave it, as an inference row keeps its premises, so the later
  split or merge reaches it as a delayed cost or reward, the way a
  conditional's trust is learned from the fate of its conclusions.

## Stages of learning as gates (proposed, 2026-09-28)

Proposed by Alec, not decided as a whole: a model has to be trained before
truth can be expressed in it and before LTM is meaningful in a consistent
way, which suggests stages of learning that act like gates. The order he
settled on 2026-09-29, as "a practical necessity": "Words must be stable.
Then they can represent something. Then they can represent the same thing
(this is object permanence)."

| step | what | what it needs |
|---|---|---|
| 1 | words are stable | a form that recurs is kept |
| 2 | they represent something | a stable word, which `interpret` replaces by its object |
| 3 | they represent the same thing: object permanence | an object to be the same as |

The stages are an order on what the model may mint, not a pairing of words
with objects supplied by a corpus, so one text can serve every stage, the
gates deciding what is learned from it. The need shows in a truth given to an
untrained model: it is stored with its source's trust and evidence $(0, 0)$,
the corner *neither*, so the truth view is empty. That is correct, and a gate
would say so in advance instead of leaving an inert row.

Two coarse authority gates are built, on `interpret` and on the clause, which
cap trust until measured ready
([Architecture](Architecture.md#authority-gates-2026-11-08));
the last architecture round turns them into one measured gate for each
operator, with the order of introduction inferred from kept trees
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
section 3.7). What remains beyond that:

- *A minimum for the interpret gate.* The gate's fraction is read over
  however few words have been admitted, so one stable word out of one opens
  it. If the gate is to wait for a lexicon rather than a word, a `<minimum>`
  admitted-word count beside `fraction` is one datum and a few lines in
  `TrustGates`; it belongs with the lesson manifest, where the gates are
  first exercised.
- *To specify:* what is measured to open each finer gate, and whether it can
  close again; what a gated capability does while it waits, refuse or defer
  or store inertly; how these gates relate to the ones that exist (the
  exposure count, the recurrence gate and the refinement patience); and
  whether the same gates order the curriculum. Further curriculum criteria,
  category transfer and chooser policy experiments are lesson work.

The order in which the text would be read, and the correspondence of the
gates with Piaget's stages, are in
[gradual training](#gradual-training-an-age-ordered-corpus-read-in-stages-proposed-2026-09-29).

## Gradual training: an age-ordered corpus, read in stages (proposed, 2026-09-29)

Proposed by Claude on 2026-09-29 in answer to Alec's "we need progressive
learning: learn words, learn the objects they represent, then learn small
sentences. Then learn more complex sentences, just like children learn". It
has two halves, which are separate: the order in which the text is read, and
the stages that say what the model may do with it
([stages of learning as gates](#stages-of-learning-as-gates-proposed-2026-09-28)).

**The order of the text** is assembled: the developmental corpus reads
Wordbank's early vocabulary in the order in which children acquire it, then
AO-CHILDES in bands of the child's age, youngest first, then written text
([Training](Training.md)). AO-CHILDES is the speech addressed to children in
the American English transcripts of CHILDES, for children from birth to six
years, with the children's own utterances removed: 2,000,352 sentences,
27,723 different words and 4,960,141 words in all (Huebner & Willits 2021;
Huebner et al. 2021). Corpora not used, which could serve later stages
(checked 2026-09-29; licences not checked): TinyStories (2.7M synthetic
stories on about 1,500 words, at the level of a child of three or four),
BabyLM (10M and 100M words: speech to children, dialogue, children's books,
subtitles, Simple English Wikipedia) and Leaner-Pretrain (71M words with
vocabulary and structure simplified); none of the three is ordered (Eldan &
Li 2023; Warstadt et al. 2023; Frank et al. 2017; Yang et al. 2025).

**The stages.** Alec's name for them is Piagetian. Piaget's four stages, and
what in this model answers to each:

| Piaget's stage | age | what the child comes to have | what answers to it here |
|---|---|---|---|
| sensorimotor | birth to 2 | object permanence | percepts, their parts and wholes; a unit that recurs is kept; an individual is a minted concept row, and `bind` joins each later mention to it |
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
* *The two halves can come apart.* Ordering the data by difficulty was tried
  widely for ordinary language models in the BabyLM challenge and was largely
  unsuccessful; that tested the order of examples, not gates on what the
  model may mint, and it does say that order alone should not be expected to
  help. So the order of the text is the weaker
  half, and the comparison to run is the same text with and without the
  gates, and with its bands in order and shuffled.
* *The order of the gates is Alec's, and it is not Piaget's.* Piaget has
  object permanence before the symbolic function, because the child's first
  objects are things seen and handled. A reader of text has no objects but
  those its words stand for, so for it the permanence of an object can only
  follow the word that represents it. The table above is a correspondence of
  kinds of achievement, and the order in which the model reaches them is the
  three steps of
  [the section above](#stages-of-learning-as-gates-proposed-2026-09-28).

To specify when it is taken up: the measurement that opens each gate;
whether a gate, once open, is open for good; and the two comparisons above,
with no seed chosen for either.

References: Piaget (1954), *The Construction of Reality in the Child*;
Inhelder & Piaget (1958), *The Growth of Logical Thinking from Childhood to
Adolescence*; Inhelder & Piaget (1964), *The Early Growth of Logic in the
Child: Classification and Seriation*; Baillargeon (1987), "Object
permanence in 3 1/2- and 4 1/2-month-old infants", *Developmental Psychology*
23(5), 655-664; Huebner & Willits (2021) and Huebner, Sulem, Fisher & Roth
(2021) for AO-CHILDES. The stages' ages and contents were checked against
secondary sources on 2026-09-29; page numbers were not.

## Ergodic exploration everywhere (future work, Alec, 2026-10-02)

Alec: "Keep Ergodic, it's better in principle than random weight init; if it
works, we'd prefer everything is Ergodic (but that can be future work)."

An ergodic layer's effective weight is its learned weight plus sampled noise,
the noise's share set after each backward from the observed gradient energy,
so exploration is high where the gradient is still large and falls as
learning settles; and it starts from structure (the identity, or identity LDU
factors, whose inverse stays exact under the noise), not from chance
([Ergodic](Ergodic.md)). Today the `<ergodic>` flag turns it on for the
layers built on `ErgodicLayer`; two MNIST configurations use it and no
language configuration does.

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
  (for example orthogonal) start. Random signed codes were found to make XOR
  readable almost for free.
  A structured start must not lose that. The reason: the answer is an
  affine read of the understanding, which gives XOR exactly only if the
  composition is not additive. With random signed object codes a nonlinear
  composition (minimum, maximum, `intersection` or `union`) puts the XOR
  pattern in the span of the four understandings and a constant for nearly
  every draw, while a sum never does, so a structured start must keep the
  codes signed and in general position.
- **Ownership.** The energy sensor should read the gradient of the weight's
  own objective. With one writer for each weight, each objective's backward
  restricted to the weights it owns (`separateChooser=true`), it does so by
  construction; under the default projected update a shared weight receives
  reconstruction's gradient and the answer's non-opposing part, and which
  of them the sensor reads is to settle.
- **Run to run.** The noise is sampled on every forward. Runs are compared
  as unseeded measurements, with no seed pinning.

**Evidence first.**

1. The kept MNIST test, run with `ergodic` on and off.
2. Then the XOR proofs with ergodic layers.
3. Then a BasicModel stage-1 comparison.

"If it works" means equal or better learning at equal updates, unseeded and
repeated.

## Multi-resolution `.where` tiling (future work, Alec, 2026-10-02)

Alec, of the overlapping tiling retired in the configuration cleanup:
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
that cleanup. Restore it from there.

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

**How it fits the one attention.** Attention narrows one bracket over time:
the open read, then `divide`, `descend` or `gloss`. Multi-resolution tiling
holds the same hierarchy at once. Revived inside it, the lattice would become
the open read's multi-scale field:

- each bracket's children are already tiled, so `divide` and `descend`
  choose among existing tiles instead of computing cuts;
- `gloss` settles a region at its coarsest pure whole;
- the word whole, a maximal run of letters, is the second
  kind refined by the first.

**What would show that it works.** At a fixed budget of wholes, compared
with the flat tiling:

- reconstruction fidelity;
- the number of bracket operations per sentence;
- the evaluator's accuracy on settled objects, on a corpus with gold spans.

## Set aside, to return once reconstruction and XOR hold (Alec, 2026-10-02)

Alec: "let's create a catalog of what we want to reintroduce after getting
things working." Of the catalog, distributional pressure on the meaning codes
has returned as placement, the primed bank as the inverse's candidates, and
the answer's reach into the understanding as the default projected update;
the unit-sphere constraint on the codes, the VQ EMA refresh and the antipode
term are not returning. Still catalogued:

- *A spatial representation of the priming field*: location trained
  long-term from activation. The activation a concept induces over the
  store's edges is its row of the successor representation
  $(I - \gamma D)^{-1}$, and codes whose similarities reproduce those
  signatures embed the diffusion, so the shortlist by activation and the
  nearest codes by location agree: co-activated concepts are drawn together
  and never co-activated ones pushed apart, since collapse is attraction
  without repulsion. Placement takes up the first-order part; the diffused,
  second-order context remains
  ([above](#placement-from-the-registered-priming-surface-one-metric-for-sbow-diffusion-and-bind-alec-2026-10-20-proposed-with-the-objective-to-decide)).
- *Activation geometry*, catalogued by name only.
- *The field proposal*: store a set of activations over known concepts, with
  the field's AND and OR over them, in place of a vector. A row already holds
  references to its words' concept rows, so decoding a sentence from its row
  is exact without inversion; the proposal's home is the open read's
  presences over concepts. The last architecture round specifies a related
  step, the field of known concepts that a point stimulates
  ([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
  section 3.3).
- *An order marker*: the symmetric connectives leave no word order in the
  understanding, so the reconstruction gate admits a transposition; order
  would come from a non-commutative binding.

## Thought residue (2026-10-09; unscheduled)

Three findings of the thought-loop audit remain outside the completed
thinking mechanism and need a later specification: the set an open-role
`part` question finds is not written into order-zero or higher-order knowing,
where a later operation could read it; the thought faces have no specified
inverse, so a thought cannot be read back through generate as a sentence can;
and `query`, which returns the best match, writes nothing to knowing, a write
the accessible mind leaves open ([AccessibleMind](AccessibleMind.md)).

## Left unbuilt by the one attention (2026-10-04)

What the building of the one attention left unbuilt when it closed
(2026-10-07). Three larger items have sections of their own:
[expectation by inversion with the reconstruction audit](#expectation-by-inversion-and-the-reconstruction-audit-decided-2026-10-06-unscheduled),
[the collision rule for property rows and the merge rule](#one-bank-the-collision-rule-for-property-rows-and-a-merge-rule-decided-2026-10-06-unscheduled)
and
[the dynamic stop](#the-dynamic-stop-glossing-above-words-descending-at-novelty).
The rest:

- **Word pairs at the sentence rung.** A word's form is identified by its
  boundary-marked letter pairs and its length. The same construction one rung
  up, adjacent **word** pairs plus the sentence's word count as the
  sentence's parts, would make "dog bites man" differ from "man bites dog" by
  form alone, as "bana" and "banana" differ by letter pairs and length.
  Overlapping parts identify a sequence without storing position: anagrams
  share their letters but not their pairs, length is implicit in the count of
  boundary-marked pairs, the fold stays one parallel join, and the rare
  collisions are the cases identity already handles (the open-bigram account
  of visual word recognition, Grainger and Whitney 2004). Today operand order
  at the sentence rung is recovered by the roles and surface markers, not by
  the form. Whether the rung needs it is a measurement.
- **The concept-face image's learning utility.** The mechanism by which
  expectation gates the concept complement is built; whether joint and
  useful-query learning improve with it on is measured at the checkpoint of
  the main training run.
- **Sparse percept presences** (perception, when it is trained): dense
  prototypes' joins saturate toward everything (mean cosine between word
  forms about .98 at any scale, XOR term of a unit root .03 to .05);
  presences on about 30% of coordinates give about .83 and .25 to .35. The
  gates report mean cos(L) so the state is known; nothing is set.
- **The attention walk on the parallel path.** The input-attention walk
  shares the sentence's departure and comparison with compose; its unchanged
  parallel MM caller still bypasses the sentence owner step, and that
  coverage requirement is unresolved.
- **A dense chooser signal** is not wanted while the forward stays a tree.
  If it ever is, the straight-through mixture's first-order comparison at every
  round enters as a control variate for the score-function estimator (REBAR,
  Tucker et al. 2017; RELAX, Grathwohl et al. 2018), which keeps the
  estimator unbiased; a plain sum of the two is biased again.

## `intuit`: the completeness fallback and the measurement that sizes it (Alec, 2026-10-21)

The direction (Alec, 2026-10-21): the top-down path from the conceptual
space to the answer is a reasoner that uses tools, or not, to produce the
next conceptual state. The thought grammar is that reasoner's structured
output: each step selects one operator of the catalogue plus pointers into
STM, MTM and LTM, credited by the keep. To preserve completeness, one of the
choices can be a learned map from meaning to meaning: given LTM and STM as
context it should be as capable as a modern transformer, and giving it the
catalogue's tools is what lets it stay small.

It is cheaper than a dense model in three ways. Knowledge leaves the weights:
facts are rows and edges, retrieved at no parameter cost. The grammar is
given, not learned: a selector over a fixed minimal catalogue has parameters
proportional to the state width times the number of operators plus the
pointer attention, with nothing per word pair; the vocabulary lives in the
bank. Depth is bought with work, not layers: a loop with a small policy
replaces stacked layers, and each step is charged by the work meter. The cost
is that discrete choices are credited through the keep rather than by
backpropagation through a dense map.

`intuit` is built as that choice: one layer of order-zero prediction, a Sigma
over Pi with negation at the literals over remembered rules, with its own
weights trained only by its use in output
([Reasoning](Reasoning.md), "Intuition, understanding and thought"). Two
things remain.

- *It may answer and reconstruct.* Alec, 2026-11-11: "An intuition is just as
  we discussed; it can do both output and reconstruction". Today
  `ThoughtAnswer.accepted` rejects evidence of kind `intuition`; the last
  architecture round changes that
  ([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
  section 3.6).
- *The measurement that decides how much of it is needed (the main training
  run).* On the graded math-English corpus, parameter count at matched
  accuracy: the structured chooser with the catalogue, against the same
  chooser with an `intuit` operator of a given size, against a transformer
  alone with and without STM and LTM as context. How often `intuit` is chosen
  at matched accuracy is the diagnostic: a large or frequently used soft
  operator means an operator or a piece of state is missing from the
  catalogue, which is the finding to act on. Transparency follows the same
  line: what `intuit` knows lives in weights; what it writes is a row that the
  hard tools can verify.

## Re-identification in MTM (future work, Alec, 2026-10-21)

When a later mention's evidence is incompatible with an earlier binding ("x
is the box", then "x is the ball"), x becomes a new MTM identity, which,
Alec said, "doesn't disturb earlier binding, but also misses an
identification. Re-identifying can be mentioned as future work." A joined
occurrence is therefore never moved. Re-identification would
recognize later that two identities are one, or that a joined site was
misidentified, and revise. A revision is also credit: the choice that
formed the identity (bind, mint or open, with the probability the chooser
gave it, kept in its formation record) receives a delayed cost or reward
through that provenance, as a conditional's trust follows the fate of its
conclusions
([one bank](#one-bank-the-collision-rule-for-property-rows-and-a-merge-rule-decided-2026-10-06-unscheduled),
"Revision conditions formation"). Starting point: each occurrence's chosen
antecedent is kept, tokens are derived by following the links, a revision
moves only the site and what chose it, and direct dependents are reconsidered
once; the cosine of `MediumTermMemory.match_candidate` and MTM's access
history are the common read to build from. Precedents: Haug's revisable
anaphora, Recanati's mental-file linking and splitting, Lee et al.'s
higher-order coreference, Doyle's truth maintenance, Tauer et al.'s
incremental entity resolution.

Its certificate exists and is skipped until the mechanism does:
[`test_reidentification_revises_the_earlier_ambiguous_binding`](../test/test_reference_close.py),
registered under the phase `mtm_reidentification`, with four partition
assertions in which an earlier ambiguous mention moves to a new group.

**By name beyond the horizon (Alec, 2026-10-28).** "We need a persistent
identity store by name, but we also need to reject that identity sometimes":
the variable of a math problem keeps its identity across the problem and must
be rejected when a new problem reuses the name. Today
`MediumTermMemory.names` holds the name within `mtmHorizon`, and `bind`'s
incompatibility rule rejects an antecedent for a new mention. Whether a
name's identity should persist beyond the horizon, and what rejects it, is
unspecified.

## Root participant positions (review follow-up, 2026-10-21)

One case is open from the review of native binding. At a root,
`verb(noun, punctuation)` starts outside a noun position, so its non-head
punctuation can become a participant; the declared head alone does not
distinguish a root attachment from a predicate-and-object verb. (The
companion case, a standalone composite object without a determiner, is
built: it is read through its declared head.) No grammar rule, word class or
marker mask was adopted for the root case. It waits for Alec's design
decision, with explicit root-position semantics and coverage of the shape.
Claude's recommendation: decide the root case by the non-head's recorded
occupations, as the ordinary menus already do (a non-head whose occupations
are marker positions is an attachment, never a participant).

## Intersection or union as the basis of identity (Alec, 2026-10-21)

Alec asked whether successive binds intersect an identity, so that it is
defined progressively, and asked that its intersection or union basis be
flagged as future work. He keeps the question open on purpose (2026-10-22),
for what it says about kinds and for the Buddhist reading
([Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21);
[Philosophy](Philosophy.md#identity-by-intersection-and-by-union-the-materialist-and-the-uncollapsed-models-alec-2026-10-20)).

As built, an identity's value is computed from its member references each
time it is read and is never stored; the promotion fold uses the union of its
mentions; and a new mention is scored by the meet, the share of its
description that the identity covers. A union admits partial descriptions:
"the black cat" is covered when any earlier occurrence said black. An
intersection would keep only what every occurrence shares, the invariant
that [Orders](Orders.md) §2 and §10 make the noun's characterization.
Philosophy and Orders §9 hold both models: the intersection as a cache kept
while each member's meet with it stays high, the set otherwise. Open: which
basis the read uses for binding and which for characterizing the individual,
or whether both are kept, as for explicit and implicit definitions.

Two neighbouring points are settled. A higher-order concept's content is not
this question (Alec, 2026-10-24): the centroid of the member symbols' forms,
weighted by their evidence in both lanes, is kept on the symbol paired with
the concept, while a concept's meaning is arranged by context, by placement.
And the identity that a pointing word carries across time empties as the
things it was diversify (Alec, 2026-10-31; [Orders §11](Orders.md#11-pointers-identity-and-relations-across-known-concepts-alec-2026-10-31)). Claude's
reading of the second, which Alec has not confirmed: an identity across time
is read by what persists across its occurrences, its invariant, while a
symbol's content and a class's content stay the centroid.

## Perceptual cover: episodic LTM (Alec, 2026-10-24)

A document's start is marked in its stream by a cover row that holds a
minted per-document concept ([Architecture](Architecture.md),
`DocumentCover`). Alec: the cover "can be conceptual or perceptual, but
perceptual LTM requires episodic memory that we don't have yet, so let's make
that future work." A perceptual cover is the title page as a percept, saved
in LTM so that a re-read of the document is recognized by its cover across
batches and sessions. Until then the cover concept is kept per source
document and its cover row is written again at each presentation, so only a
re-read of the same source is recognized. Episodic entries now exist and hold
concepts, not percepts (§7): matching a re-read against a kept entry is the
cue by content of §7, and the last architecture round keeps no perceptual
memory for now.

## A cheaper forgetting pass: running luminosity on LTM (Alec, 2026-10-24)

Alec: make the forgetting algorithm much cheaper, "probably by building a
running luminosity measure on LTM." The last architecture round builds it
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
section 3.11).

**What the pass costs today.** `Forgetting.retention_values` recomputes
every term at each boundary pass over the whole store: luminosity by a
pairwise cosine of all fused points in 128-row blocks (`points @ points.T`,
with the same-order, same-scope and opposite-pole masks), quadratic in the
store; coverage, when `<forgetCoverage>` is on, by a leave-one-out
catuskoti coverage over all idea rows, quadratic again; and age by
`history_age`, the model time since each row's stamp. Trust (the larger
evidence pole) and utility (the write-time surprise) are read from columns.
At a million rows the pairwise term alone is on the order of $10^{12}$
multiply-adds per pass.

**The proposal.** Keep luminosity as a column, maintained incrementally
rather than recomputed: when a row is written or re-witnessed, compare it
only against the rows its content already indexes to (the content
postings by code or the priming surface's neighbourhood), in the same scope
and order, and update the dissonance flag of the pair's weaker member; when a
row is deleted, release the rows it alone made dissonant. Coverage, if kept,
becomes a running sum of per-coordinate pole maxima with a per-row
contribution, so leave-one-out is a subtraction, not a recount. The pass then
reads its columns and sorts, linear in the store, and the retention order is
unchanged because the same quantities are maintained.

## Concepts as regions, certainty from evidence, no sphere (Alec, 2026-10-24)

Alec: "The field of concepts would be richer if it represented regions
rather than projections, so I worry about using concept length as a
certainty measure and projecting on the sphere rather than the unit ball."

**The framing he confirmed the same day.** An order-0 concept is a point on
the hypersurface with a scalar certainty per lane: the radius is not lost, it
is the certainty, so the projection loses nothing, and distance stays a
matmul instead of a pairwise Euclidean. A higher-order representation is a
region containing those points. Certainty and the positive and negative
evidence are orthogonal. Centroids remain the symbol side's (percepts and
symbols), and a concept's meaning is placed by context. The certainty half is
built: a concept's trust is the length of its for lane
([Architecture](Architecture.md), "Concept and relation trust"), and the
readers of length as certainty (`Queries.extent`, the `gain` executor,
`Interpret.activate_code`) stand under this framing.

**What remains.**

- *The region of a higher-order concept.* A spherical cap (a center and an
  angular radius) keeps every test a matmul: point-in-cap is a dot product
  against a threshold, cap-in-cap an angle sum, and dissonance among points
  the existing `points @ points.T`. Boxes give coordinatewise containment but
  need min-plus products that no BLAS kernel computes; they stay the percept
  cube's mereology. The change to the layout would be one extent scalar per
  lane on rows of order one and above, not a doubled width; evidence stays in
  the columns. No extent is stored today.
- *Readers that still match on the sphere.* `Forgetting.retention_values`
  normalizes all fused points and tests "same region" by cosine above
  `<forgetSimilarity>`; `DocumentCover.read` and the expectation keys match by
  cosine or normalized dot; the identity candidate search of
  `IndependentComponents` normalizes its directions and basis. These can move
  to the meet (`IdentityRows.meet_ratios`, a region overlap, the sum of minima
  over mass) without a layout change.
- *A box over a scalar.* Alec, 2026-11-11: "can the model learn the word for
  a number (where that word is a concept with part and whole bounds that are
  hyperrectangles)?" The last architecture round proposes it for the scalar
  it places beside the words: a word's concept over the scalar is a box whose
  bounds are the least and the greatest of the values heard with the word, and
  a value belongs to it by the two coverages of `IdentityRows.meet_ratios`
  ([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
  section 3.10). On one dimension that is an interval; the rule is the same
  on several.

## Runnable source-only archive (Alec, 2026-10-24)

**Requested.** Produce an executable-style bundle of BasicModel's sources
in the release archive, so running the model does not require the full
development checkout. This is future packaging work, not a bundle already
built or a change to the model.

Build from an explicit list of required implementation sources, with a
launcher, dependency versions, essential configuration/grammar/schema
files and minimal run instructions. Record the source commit and included
file hashes. Keep tests, test fixtures and experiment drivers, evidence,
logs, checkpoints, training corpora, virtual environments and Git
history out of the bundle. Any required external model or dataset is a
declared input, not silently copied from the developer's machine.

Choose the executable packaging and supported runtime when this item is
specified; state which Python and native dependencies must be installed.
Store the resulting bundle as an archive/release artifact outside the Git
source tree. Gate it by unpacking into an empty directory outside the
checkout, installing only the declared dependencies, and running the public
entry point on a minimal input without access to the test tree, local
evidence or cached development state.

## Image modality: amplitude, measure and Kosko's cube (Alec, 2026-10-25)

Alec: "That's a flaw in my model; it's missing a magnitude in subsymbolic
computation, since we have only parts and wholes. But letters are a
discrete training set." The image modality is future work, and Kosko's
work is part of it. The scalar case comes first, in the last architecture
round: one continuous value beside the words, a percept whose code is the
value, on which part and whole are a less-than discrimination on the number
line
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
sections 3.9 and 3.10). Signals of several dimensions and images stay here;
for images, `.where` would become two-dimensional, and wholes and parts
hyperrectangles in the unit hypercube (Alec, 2026-11-11). A first path
for images, as numbers of many dimensions with no place, is under
[Numeric input as a modality](#numeric-input-as-a-modality-xor-as-a-system-1-model-and-one-model-class-alec-2026-11-11-to-specify).

**Where magnitude is lost today.** (1) The join is idempotent: a word's
form is the coordinatewise max of its parts ([Spaces, fold ladder](Spaces.md#fold-ladder)),
so repeated constituents vanish and count lives only in occurrence rows.
(2) Percept parthood is the clipped cosine
([Mereology](Mereology.md#parthood-as-clipped-cosine-projection)),
symmetric and blind to magnitude; the exact dominance `Ops.partOf` and
`IdentityRows.meet_ratios` (the sum of minima over mass) are the readers that
respect it. (3) A concept's length is its certainty
([above](#concepts-as-regions-certainty-from-evidence-no-sphere-alec-2026-10-24)),
so quantity cannot live there. For bytes none of this costs anything; a
letter is present or absent.

**How intensity is carried is not settled.** The first proposal (2026-10-25)
was amplitude as the degree of presence of a part or whole, such as r, g and
b values, with letters at 0 or 1: the cube is already presence in $[0, 1]$ on
a UNORM store ([Spaces](Spaces.md#percept-presence)), an occurrence would
carry the amplitude while its row holds the support, and percept parthood
would become Kosko's subsethood. It was set aside for images the same day in
favour of intensity as a code
([next entry](#intensity-as-codes-not-amplitude-alec-2026-10-25-decided)).
The last architecture round's scalar is a third reading: the value itself is
the percept's code, and part-of is dominance. Which stands when images come is
Alec's to say.

**Kosko's cube.** The percept cube is Kosko's: fuzzy sets as points of the
unit hypercube, with the sigma-count $M(A) = \sum_i a_i$ as measure,
$M(A \triangle B) = \sum_i |a_i - b_i|$ as the fuzzy Hamming ($\ell_1$)
distance, subsethood $S(A, B) = M(A \wedge B) / M(A)$, and fuzzy entropy
$E(A) = M(A \wedge A^{c}) / M(A \vee A^{c})$ as the distance from the nearest
corner (Kosko 1986, 1990, 1992). The complement $1 - x$ is the cube's own
([Spaces](Spaces.md#percept-complement)). One quantity gives the metric, an
asymmetric part relation and a crispness measure.

**Images (future).**

- *The basis comes before the mereology.* Meet and join mean overlap only
  on coordinates of disjoint support (location by spectral band, a localized
  wavelet or quadtree basis); dense mixed codes saturate.
- *Space needs connection.* `.where` becomes a region with area, the fold
  ladder a quadtree
  ([multi-resolution tiling](#multi-resolution-where-tiling-future-work-alec-2026-10-02)),
  and the five relations extend to RCC-8 (external connection, tangential
  part), so a connected object is told from a scattered sum.
- *Concepts.* Hue concepts are scale-invariant cones, so the order-0 point
  with its certainty stands; quantity concepts (bright, many) read a
  log-amplitude property in WholeSpace, never a concept's length.
- *Number follows.* Count is the measure on discrete parts, an approximate
  number sense is a noisy $\log M$ (Weber), and successor is one more unit
  part, which grounds the opaque numerals' successor chain
  ([Training, math suite](Training.md#the-generated-math-suite-what-it-exercises-and-what-it-cannot-2026-10-20)).
- *Continuity survives the snap.* The codebook snap keeps its residual, so
  quantization gives categorical perception without erasing continuity
  ([§8](#8-recall-as-estimate-plus-residual)); exact-key matching gives way
  to a log-scale tolerance for graded percepts.
- *Measures on extents are set aside for now* by the decision that intensity
  is a code: lights mix additively and filters multiplicatively, so extents
  would carry measures (radiance, area, count) under a sum with the overlap
  correction $M(A \vee B) = M(A) + M(B) - M(A \wedge B)$ while intents keep
  $[0, 1]$ and the product, in a vector lattice (Riesz space) that holds both
  the order and addition; and a whole's value would be its total, whose
  direction is its quality and whose norm its quantity, not its centroid.
- *Data.* Alec, 2026-09-29: "at some point, LLMs learn objects through
  multi-modal training, so Fei-Fei Li's dataset would be relevant". Two come
  from her group, and both are keyed to WordNet, so their labels arrive
  already arranged as a taxonomy of nouns. ImageNet fills WordNet's noun sets
  with images. Visual Genome annotates about 108,000 images with their
  objects, attributes and the relations between pairs of objects, and with
  descriptions of regions, which is nearer to a sentence about a scene (Deng
  et al. 2009; Krishna et al. 2017).

**References.** Kosko (1986), *Fuzzy entropy and conditioning*, Information
Sciences 40. Kosko (1990), *Fuzziness vs. probability*, International
Journal of General Systems 17. Kosko (1992), *Neural Networks and Fuzzy
Systems*, Prentice Hall. Luxemburg & Zaanen (1971), *Riesz Spaces I*,
North-Holland. Randell, Cui & Cohn (1992), *A spatial logic based on
regions and connection*, KR '92. Gardenfors (2000), *Conceptual Spaces*,
MIT Press. Wyszecki & Stiles (1982), *Color Science*, Wiley. Dehaene
(2003), *The neural basis of the Weber-Fechner law: a logarithmic mental
number line*, Trends in Cognitive Sciences 7.

## Intensity as codes, not amplitude (Alec, 2026-10-25; decided)

**Decided for images, for now.** Intensity is a code: 255 RGB levels are 255
atoms (one percept row per level), or a more intelligent discrete coding such
as octave filterbanks (one atom per band). A pixel's intensity is then
*which* atoms fire, a crisp percept like a letter; no amplitude coordinate,
no graded occurrence, no rate code.

**Why.** The value of order in the codes is transitivity: "we know for any
two percepts if one is a part of the other" (Alec), and it is what gives
relative meaning to zeroth-order percepts. That requirement stands, and it is
built as one partial order over two form subspaces, identity and property,
read by the one call `Ops.partOf`
([Spaces](Spaces.md), "The Cube as a Lattice; Two Form Subspaces"). Three
ways of carrying amplitude were considered and set aside:

1. Amplitude as a per-coordinate degree on the occurrence while the type's
   row holds the crisp support. Parthood then has to be read from supports,
   a heavy computation on a part composed of many sub-parts, and a dense
   code's join saturates for wholes of many parts, so support inclusion goes
   vacuous.
2. Amplitude as firing rate on the priming surface (seen bumps weighted by
   intensity, the surface's leaky integration as the rate code), codes
   staying binary.
3. Kosko's subsethood $M(A \wedge B)/M(A)$ as parthood between graded
   occurrences, which answers "how much of A's mass lies in B", not "which is
   a part of which": a dim letter is "part" of a bright word and a bright
   letter is "not quite" part of a dim one.

## Synchronize explicit and implicit meanings (Alec, 2026-10-27)

Alec: "Classes which get raised will need initial learning
to cover their members. Explicit and implicit classes should be treated
the same; please add a future work to synchronize explicit and implicit
meanings."

Every raised class needs the same initial learning to cover its members,
whether raised explicitly or implicitly. Naming a class and recording its
members stay distinct: treating the two raises alike does not make every
object named by the same word a member of one class.
The future work is to synchronize the meanings learned through those two
routes. Explicit and implicit definitions stay available for judgement,
as decided on 2026-10-21: agreement and disagreement retain independent
for and against evidence. This entry does not choose an automatic class
merge or identity rule. The synchronization mechanism and its checks
remain to be specified.

## The residual split: a stopped subtraction or the correction (Alec, 2026-10-29)

Expectation and top-down reconstruction split by residual, each learner with
one teacher: the local predictor learns only from its own local error and no
top-down gradient reaches it, and the inverse of the completed understanding
is scored against the observation less the detached local expectation, the
forecast frozen when the item arrived (one recomputed at the close would
offset a target it never predicted). Residuals are signed, per lane at the
conceptual layer and never netted, and maps with an exact inverse stay
audits.

Two forms of the rule differ in what the inverse learns. In the correction
form, `correction = inverse(u)` is compared with
`stop_gradient(observed - expected)`, so the inverse's optimum is $x - e$, a
different function from $x$: the inverse also reads the detached expectation,
outputs only the correction, and so learns only what the expectation missed.
In the form built at the percept layer, the detached expectation is
subtracted from both sides, $(\hat{x} - e) - (x - e) = \hat{x} - x$, so the
error and its gradient are as before, whatever is stopped
([ExpectationRetention](ExpectationRetention.md), "Percept arrivals"). Asked
to confirm the correction form, Alec answered that subtracting the
expectation is meaningful because the expectation term's gradient is stopped.
The stop keeps expectation's parameters out of the top-down gradient, the
ownership half of the split, and that is kept.

To settle: which form each layer builds, and whether the correction form
reduces the capacity the inverse needs, measured before that advantage is
claimed (an explicit learned residual inverse against the native inverse's
translated leaf chart). Until then the percept layer keeps the built form,
and the conceptual split is not built: its local predictor would be
`ActivationPredictor`, and its inverse the completed understanding's readback
into the order-zero concept field the sentence produced, per lane.

## Time words for the model's own past (Alec, 2026-11-12)

A question such as "what did I eat yesterday" is a region of event space
applied to memory as a mask, and the time a memory entry was written is a
coordinate of its event, read as an age; both belong to the last
architecture round
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
section 3.12). What stays open is the teaching: a time word is a box over
that age, as a number word is a box over a felt quantity, but no lesson yet
teaches one, and model time counts sentences and thoughts, so a word for a
day needs day boundaries to arrive as events.

## Signs and indexicals once there is a field (Alec, 2026-10-31 and 2026-11-01)

A word that points ("this", "that") gets no class and no word list. An
interpretation may be a symbol, the word's conventional object, or a `sign`,
Peirce's index: `sign` opens a pending MTM token for the thing pointed at and
gives it the null-concept, one reserved, frozen concept with no content,
exempt from the operations that rely on a definition; a kept definition names
the token and never defines the pointing word. A pointing word interpreted as
a symbol is raised at `raiseCount` distinct relata, and its identity across
time empties as the things it was diversify
([Orders §11](Orders.md#11-pointers-identity-and-relations-across-known-concepts-alec-2026-10-31); [Architecture](Architecture.md), "Sign and the
null-concept"). An explicit `point()` operation, a symmetric raise, a counter
of broken transitivity and a built deictic class were considered and
withdrawn. What remains:

- *The forcing.* Alec, 2026-11-01: "the better way to implement this is for
  interpret() to be optional, and if it happens, then the interpretation may
  be a sign or a symbol. Again, interpretation can be initially forced since
  words are so often meaningful beyond themselves." In reading mode
  `interpret` stays mandatory (`InterpretLayer.mandatory`), so the third
  reading, the icon, in which a percept stands for itself and the leaf is not
  interpreted at all, is built but reached only inside a certificate. Lifting
  the forcing for text, by a parameter or by the field's arrival, is a
  decision for then; outside reading mode the chooser may interpret, sign or
  do neither.
- *Indexicals under multimodal input.* With perceptual input beyond text (the
  [image modality](#image-modality-amplitude-measure-and-koskos-cube-alec-2026-10-25)),
  an indexical mention's candidates are the field's current percepts, the
  order-zero occurrences at the mention's `.where` and `.when`, weighted by
  location as MTM's candidates are today; a `sign` token's content is the
  pointed-at percept, and an uninterpreted token's content is the percept
  itself. A sound field would make the icon literal: today the model's only
  sounds are onomatopoeia, sounds rendered in letters, and a bark heard would
  be a percept read as itself, with no word between. Until then an indexical
  is an MTM reference and does what a pronoun does.
- *Names and pronouns by binding history.* No retained fact separates a name
  from a pronoun at initialization: both are objectless words in a
  participant position. A learnable cue is a word's binding history across
  documents: a name stays with one referent, a pronoun moves among many.
  Learning the distinction from that history is not built.
- *Inspection.* The mark of a pointing word is a transitivity that breaks:
  "this is Felix" and "this is Ciao" give Felix = Ciao, which their witnessed
  distinctness contradicts. Such concepts, joined to several distinct
  concepts at their order, are found by inspecting the `LTM1` rows
  periodically; by decision no counter is built.
- *Relations across known concepts.* Alec: equals() "will eventually store
  known concepts across a relation rather than as a single idea", and how
  those ideas are combined and associated can be revisited from LTM. Only the
  order reading of Orders §11 is applied today.

## Terms evaluated on the model's own internals (Alec, 2026-11-03)

Alec ruled that "the teacher should not be reaching into the internals of the
model". Under that rule the walk, decomposition, verb and modal teachers are
retired, with the field, iteration, forward-discard, content and reference
reconstruction terms; a learned top-down policy, if one is wanted, is
credited only by output. The terms still evaluated on the model's own states
supply no target to a choice and are kept for now by his decision: the two
walks' residuals, the reconstruction walk's departure credit, the definition
fold, and memory's link coverage and verb and modal state errors. Their
definitions, owners and sides are in
[Gradient flow](GradientFlow.md#terms-on-internals).

What remains is to revisit each kept term under the same principle. Replace
it by a target outside the model (the input, a supplied answer) or by the
term's functional participation, as the verbs and modals now learn;
otherwise, record why it stays. Participation means that an action is
credited by what reads its result: a verb's `lift` output becomes its
subject's new state in the identity seat, so whatever reads that state with
gradient tells the verb what the state should have become, and a modal clause
writes only the subject's alternative-timeline seat, so the modifier learns
what readers of that seat need beyond what readers of the actual seat need.
Not scheduled.

## Behavioral chains: MTM at order zero (Alec, 2026-11-03; amended 2026-11-12)

Alec: "We might also have MTM operate on order zero concepts as a way of
creating behavioral chains? Animals have that, and we might want to store
those in MTM before promotion to chained event concepts." They are to be in
place before the main training run: as MTM identifies across nouns, it can
identify "the same" event by its gestalt properties, which may even help in
the creation of nouns, since a thing that persists through a chain of events
is a candidate noun.

The linking is built. MTM has one job, linking occurrences across time
(identity for the mentions of one thing, process for an identity's events,
causality for a course of events' states), by one function at the kept close,
and one promotion rule: `conceptMintCount` recurrences promote a collection to
a concept one order above its members, with successor rows along its path
([Architecture](Architecture.md), `MemoryLink`). Two structures follow from
the amendment of 2026-11-12, which replaces the earlier constraint that MTM
opens no order-zero identity: the same order-zero reading met again is an
order-zero identity in MTM, the candidate that
`FamiliarTier.observe_candidate` opens for promotion (MTM is ramsified, so an
order-zero identity is a necessary consequence); and states joined in time
order form a sequence, kept until recurrence promotes the chain to a chained
event concept. What remains is recognizing the present event by its
order-zero identity, which is the last architecture round's
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
decision 21).

## Belief across the epoch boundary: perseverance and testimony (Alec, 2026-11-04; revisit if necessary)

Decided for now: evidence belongs to its source and lasts as long as that
source is held. Within an epoch the history is held, and forgetting removes
whole memories and never zeroes a belief. At an epoch boundary the history is
released and everything it produced goes, while given knowledge keeps its
supply's evidence. Testimony carries its teller's authority on the outer
relation only: content embedded in a supply enters unverified, with both
lanes at zero ([Architecture](Architecture.md), "Evidence belongs to its
source"). The boundary is not a psychological event but the re-presentation
of one history to the weights. Two alternatives stay open, each closer to how
people behave:

- **Perseverance at the boundary.** A row kept for given knowledge would keep
  the history's belief, in `LTM0` and `LTM1` alike. People keep beliefs after
  their grounds are discredited (Ross, Lepper and Hubbard 1975), and after
  forgetting where they learned them (source amnesia). The cost: the previous
  epoch's belief becomes a floor for the next. `LTM1` merges evidence by the
  per-lane maximum, so re-reading the history cannot lower it, and the epoch
  no longer re-presents one history.
- **Accepting testimony.** Content embedded in a supply would enter with some
  belief, instead of unverified with both lanes at zero. People tend to accept
  what they comprehend before evaluating it (Gilbert 1991). The cost: "Bob
  said X" would raise belief in X by mention alone.

Revisit if a trained model's answers show given knowledge cut off from the
beliefs it depends on, or if a psychological measurement calls for it.

## Events as mereological neighbours (Alec, 2026-11-05)

Asked how a state is computed from a run of event members, Alec: "Just as
some nouns share an identity, some events are mereological neighbors (in
world-space, not .where, although we can infer one from the other for words,
if not what they represent). We can defer this to future work." Nouns that
share an identity are one identity in MTM; events that are neighbours share a
boundary in world-space. For words the two addresses can be inferred from
each other, but not for what the words represent. As built, an event run's
state is computed from its members as an identity's value is computed from
its references, one order lower, and the run stays a sequence. The
world-space neighbour account is not built.

## Left open by trust in operations and the one gradient (2026-11-08 to 2026-11-11)

### Separate word form order from leaf referent order

Word concepts should generalize token instances at form order 1, with argument
classes at order 2. The reading leaf separately carries its referent's order.
This changes grammar order rules, including bind eligibility and lift
lowering, and is deferred; today native words stay at 0 and arguments at 1.

### Grammar XOR optimization that does not settle

A panel of 2026-11-08 observed grammar-XOR non-convergence in about a third of
initializations. Failing trajectories oscillated between roughly 0.05 and 0.25
loss through all 400 epochs, with transient lower minima rather than sustained
convergence. The mechanism remains open for the lessons work. The panel does
not identify the argument-fit carrier as the producer; silencing it lost all
three runs.

### Trained-tier artifacts from tiny models, and two class measurements (2026-11-08)

Alec's observation: highly trained models at very small sizes, with very few
words, may be producible quickly and would serve as the registered artifacts
that make trained-tier tests runnable; the eight-word `XOR_grammar.xml` model
is one, and the `xor_grammar` and `mm_xor_grammar` phases already name their
producers in `test/fixtures/phase-artifacts.json`. They belong with the lesson
artifacts of the test development pass. The lessons work also reads two
measurements before a lesson is said to have taught a part of speech: that a
word's class follows its participations rather than its meaning resemblance,
and the collapse table, the pairwise cosines between the operator arguments'
concepts (`PartsOfSpeech.collapse`), read beside the argument radii under the
class-contiguity rule.

### Unused property attribution

`PrimitiveProperties.reverse` and `reverse_poles` (`bin/PerceptProperties.py`)
keep their `1e-12` normalization floors but have no callers since basic
percepts replaced that read; they go with the next change to that file.

### Conditions on the projection and the pair policy (2026-11-09)

Projecting the momentum-carried step, rather than the instantaneous gradient,
is built only if measurements show reconstruction rising where output was
projected. A learned pair policy is built only if answers show errors of pair
orientation or near-ties among approximate inverses. It would join the
comparison of its walk under the ordinary proposal and score-function credit,
keep the joint residual in its score, read no feature from the derivation's
operand order, and use a fixed score scale, never a division by the parent's
energy. The pair shortlist ranks candidates by form similarity times priming
before either meaning lane is read, so meaning arbitrates only among the K
form-nearest words on each side. Alec: meaning matters, particularly with
large lexicons; the shortlist's dependence on form is revisited with that
policy. The lesson owners `compose_lesson` and `generate_lesson` are to be
named for the three grammars, compose, think and generate, when that code next
changes.

### Open questions of the whole-path design (2026-11-11)

1. The question conditioner and the generation read change the state between
   the thought and the output walk, and no departure explores them; whether
   they belong to the thought, or go, is open.
2. A path explores one departure, so an improvement that needs two departures
   together is never found; paths with more than one departure are not built.
3. On a statement an accepted conclusion becomes the answer and pays for being
   said; whether a thought may keep a conclusion without saying it is open.
4. Omission hides a word's whole form or a percept slot; withholding a concept
   from a seen form (the unfamiliar word), omitting a constituent, and
   corruption in place of omission are not built, the last deliberately.

The lookups-and-folds port still has to adopt the basic-percept feature
layout.

## `.when` is model time in every store (Alec, 2026-11-10)

Alec: ".when should not be per-document, it should be model time, even for
semantic memory." It is built: every stamp is on one model-time ladder, one
tick per sentence and one per thought step; learned readers take an age or a
difference, never the absolute stamp; document ordinals remain addresses
only; and both clocks are put back at every epoch boundary, so a history
presented in the same order gets the same stamps in every epoch
([Architecture](Architecture.md), "One where, one when, many whats";
`RowTime`). Three points are left.

- *The horizon.* With the clock bounded by one epoch, `RowTime.HORIZON` could
  be the declared epoch length.
- *Recurrence needs order.* The stamps recur only when the order of
  presentation recurs, and some generated corpora shuffle premises each
  epoch.
- *Use.* The last architecture round keeps the clock and addresses nothing
  by it
  ([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
  section 3.8). Whether `.when` should become a duration is the next entry.

## `.when` as a duration: store only the deltas? (Alec, 2026-11-10)

**To consider.** Alec: "should .when become a duration (and serve as a
means of compression) by only storing memory deltas?"

Two readings, Claude's, neither decided:

1. *Time as a difference.* An entry's `.when` stores the time since the
   previous entry of its stream, and absolute time is the running sum. A
   difference does not depend on when the model reads, and it is the quantity
   that transfer takes
   ([§7](#7-episodic-memory-a-few-active-indices-beside-each-row),
   "Transfer").
2. *Write on change.* Memory writes an entry only when its content
   changes, and the entry's `.when` is the interval over which that
   content held, a run-length code in time. A temporal whole is then a
   maximal run of constant content, as a word whole is a maximal run of
   letters.

*In the code.* `RowTime` stores a point. An earlier endpoint bracket carried
a duration, which was retired from the event tail because nothing decoded it;
`DocumentWhenEncoding.encode(start, end=None)` still takes an end and its
`decode_span` returns `start, start + 1`, so a width other than one can be
written and is never read back.

*To settle if taken up:* whether a delta runs per stream or per store; what
a recall and forgetting (an age) read when only deltas are stored (a running
sum per read, or an absolute stamp kept at intervals); whether
write-on-change applies to episodic entries alone or to idea rows too; and
whether a re-witness extends an interval instead of keeping the first
stamp. [Kaufmann and Viera (2026)](https://doi.org/10.1080/00048402.2026.2649411)
count a duration used as a quantity, informational transfer, among the
marks of a temporal representation (§7, "Transfer").

*Decided earlier (Alec, 2026-09-30) and not built: a row keeps only its
start.* A state holds from its row's start until a later row of the same
content with evidence against begins in the same stream; with none it holds
on. Memory stays append-only: an ending is a new row, never a value written
back into an old one, so a feeling that lasts for days is two rows, its onset
and its end. This is the event calculus's law of inertia (Kowalski and Sergot
1986), the usual answer to the frame problem, and it replaced an eternal
flag, a refresh rule and a stored duration. A query "does P hold at t" finds
P's latest onset at or before t and checks that no complement begins in
between; no such reader exists yet. Any delta or run-length scheme for
`.when` should be weighed against this rule.

## A fragment or a sentence understood in one attentional window, without grammar (Alec, 2026-11-11; to investigate)

Alec: "Would the architecture eventually be capable of processing sentence
fragments or short sentences grammatically? "The cat" might be one
attentional window, producing the same conceptual activation as the words
combined grammatically?" He asked for an investigation of the possibility,
and in particular of whether System 1 might learn to understand whole
sentences without grammatical operations.

**What exists.**

- *A fragment is grammatical.* The close needs no clause: where no relation
  was composed it writes an idea of one slot, so "The cat" composes to an
  idea and is stored as one.
- *The window.* Every sentence opens with an attention pass over the whole
  input as one bracket, with no grammar and no narrowing
  (`FamiliarTier.open_sentence`). It feeds the spread and opens or updates
  the familiar tier's candidates; it mints nothing. The dynamic stop
  [above](#the-dynamic-stop-glossing-above-words-descending-at-novelty)
  would gloss a known unit at its wider bracket.
- *A phrase becomes one concept.* A phrase that recurs without
  contradiction is minted as a concept defined by its parts
  (`conceptMintCount`), and the open pass can then match it whole.
- *System 1 learns from System 2.* Expectation is trained on the detached
  order-zero contents left after System 2 has acted, so it learns in System
  1's space from System 2's activation of it; its prediction is pooled over
  occurrences, so it is rendered only after the grammar orders it.
- *One structure for both readings.* Serial concepts exist in the parallel
  reading through the same structure.

**What is not known.** Nothing requires the two routes to agree. There is a
parity test of the shared inventory across the two readings, on words. No
certificate is known that compares, for a phrase or a sentence, the
order-zero contents read in one window with the contents the grammar gives.

**The investigation.**

1. *The same activation for a phrase.* For a minted phrase, compare the
   order-zero field read in one window (the open pass with expectation and
   no compose) with the field the grammar inverse writes from the composed
   idea (`OutputOccurrences.generated`). A parity certificate at the level
   of the phrase, with the distance between the two fields reported as the
   phrase recurs.
2. *Whole sentences without grammatical operations.* Whether System 1
   alone (perception, the order-zero concepts, expectation and `intuit`)
   comes to produce for a short sentence the order-zero contents that
   compose and generate produce. Measured with the reading's compose walk
   withheld at test, on a model trained with it: the distance between the
   two fields; the reconstruction under omission, as the share of its
   input the model can do without (the two omission rates); and the answer
   cost of questions, whose answer still comes from the thought.
3. *When attention descends.* Under whole paths the narrowing walk is in
   the departure draw, so reading a bracket whole or descending into it is
   credited on $R + A$. Whether the model stops above the word as units
   become familiar is the dynamic stop's measurement.

**An existence argument, and what it does not give (Alec, 2026-11-11).**
"I guess if system2 teaches system1 (intuit), and "understanding" means the
total symbolic activation as a result of processing a sentence, then there
is a map through s1 intuit() which is equivalent to the s2 grammatical
processing; the related proof for single-layer MLPs can be used as a
proof."

The map exists, by two results:

- *The one he names.* A network with one hidden layer approximates any
  continuous function on a compact set to any accuracy (Cybenko, 1989;
  Hornik, Stinchcombe and White, 1989; Leshno, Lin, Pinkus and Schocken,
  1993, for any activation that is not a polynomial). System 2's map has
  hard choices, so it is continuous only in pieces; the same density holds
  in $L^p$ (Hornik, 1991), which leaves a small set of inputs near the
  choice boundaries.
- *The one that fits `intuit`.* `intuit` is one pass of meets and joins over
  literals: `Intuition._evaluate` takes the minimum of the known evidence
  with a term's literals and the maximum over what fires, with a `pi` and a
  `sigma` weight for each term. That is a disjunctive normal form with
  graded truth, and every Boolean function of the symbols has one exactly.
  It gives which symbols are active. On graded evidence a minimum is not
  System 2's product, so the numbers need not be the same.

What neither result gives:

- *Size.* The number of terms can grow as $2^{n-1}$ in the number $n$ of
  symbols a sentence combines; parity, of which XOR is the case $n = 2$,
  needs every one of them. `intuitionCapacity` is 4,096 terms. A deep
  composition can need an exponentially wide shallow map (Telgarsky, 2016;
  Eldan and Shamir, 2016).
- *Learning.* That the map exists does not say System 2's activations teach
  it.
- *New combinations.* The grammar extends to a combination it has not seen
  by construction; a flat map is equivalent only where it was taught.
- *Inputs.* The map exists only for what is a function of what System 1
  reads. Where the understanding rests on memory that System 1 does not
  read, there is none.

So the investigation is of size, learning and coverage, not of existence.
It predicts where the two readings part first: with the number of symbols
combined (parity-like combinations before others), with the novelty of a
combination, and with dependence on memory.

**What it waits for.** A corpus on which phrases recur (the corpora of the
test development pass and the main training run) and the shared-mode erosion
question ([above](#shared-mode-erosion-measurement)), since a reading
without grammar is the reading Alec expected to erode serial boundaries.
The stages it measures exist: each path reads, reconstructs through
generate, thinks and answers, and the paths are compared once on $R + A$
([Gradient flow](GradientFlow.md#whole-path-credit-and-concept-episodes-2026-11-11)),
with the omission rates adapted so that the reconstruction of everything
holds at `omissionTarget` ([Training](Training.md), "Adaptive omission").

No mechanism is proposed. This entry asks for the measurement first.

## Numeric input as a modality, XOR as a System 1 model, and one model class (Alec, 2026-11-11; to specify)

Alec asked for a numeric input as future work, with XOR converted to use it:
"Probably xor should be converted to a system 1 model. That means just one
concepts-to-concepts step at order 0 (Which is roughly equivalent to its
previous sigma pi, I think?) And in fact, we probably want to have only a
BasicModel class".

The first two are specified and not built. The last architecture round makes
a number a percept whose code is its value, reads the field out through one
linear layer, and moves the XOR proofs to two numeric inputs on a file with
field operators only, where XOR is $x_1 + x_2 - 2\,(x_1 \wedge x_2)$ and the
one concept needed beyond the two inputs is their intersection
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
decision 9 and section 3.9). Two findings behind that form:

- *"Order 0" means no grammar, not order-zero concepts alone.* In the grammar
  XOR gate the linear head reads each path's composed sentence state, the
  root and end slots, which are of higher order. The order-zero field there
  holds only the two words' concepts, $x_1 = h + w$, $x_2 = h + t$,
  $x_3 = l + w$ and $x_4 = l + t$ for the four sentences, and every affine
  read $f$ gives $f(x_1) - f(x_2) - f(x_3) + f(x_4) = 0$ where the classes
  $0, 1, 1, 0$ need $-2$.
- *One step holds conjunctions and a disjunction.* XOR is
  $a \bar{b} \lor \bar{a} b$; neither one sigma nor one pi alone gives it.
  `XOR_exact.xml` and the grounded XOR test already learn it with no grammar,
  by conjunctions over order-zero memberships and one learned sigma row
  whose positive pole is the answer.

Outside the round:

- *One model class.* `bin/Models.py` has `BaseModel` and
  `BasicModel(BaseModel)`; `ModelFactory` builds `BasicModel` only, and nine
  test files construct a bare `BaseModel()` as a helper. Folding `BaseModel`
  into `BasicModel`, with the nine helper constructions ported, is
  housekeeping. (`MentalModel`, the `MM_*` family, the XOR files and `mnist`
  are configurations of `BasicModel`, not classes.)
- *The whole-slab body* of numeric configurations that are not XOR proofs
  (`mnist.xml`, `model.xml`'s default) is not changed by the round.

**Images by the same path (Alec, 2026-11-13).** "Please ensure a path for
the number modality to process images as well as XOR (the only different
being the percept dims; MNIST-size images would be a good test, but no
need for multimodal at this point, that work is left to someone else)."

- *One reader.* A row of numbers is flattened to its dimensions: 2 for
  XOR, 784 for a 28 by 28 image. The numeric reader already takes the
  product of the input shape as its dimensions.
- *Level sets.* Each dimension's number is categorized against a part and
  a whole ([Spaces](Spaces.md#percept-level-sets)). The numeric reader
  today gives each dimension's concept the value as evidence, a
  magnitude; the scalar beside the words is already categorized. Moving
  the numeric reader to level sets is to build.
- *One step of System 1* lies between the concepts and the linear output.
  For XOR the one concept needed beyond the inputs is promoted when both
  inputs recur together. A whole image's set of lit pixels does not
  recur, so for images the step has to be a learned map between concepts
  (*Claude's reading*; [Reasoning](Reasoning.md), "Implicit and explicit:
  what each system holds").
- *The test.* A 28 by 28 input of two levels, which differs from XOR in
  its dimensions alone: it runs forward and backward and its
  reconstruction returns every number. Learning on such images is first a
  measurement, beside the same file with the System 1 step removed. Grey
  levels follow. Two modalities at once are not part of it.

To settle: whether a dimension is the percept's place, as a letter's
position is; where the levels of a grey image come from; and which System
1 step it is
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
section 22.9).

## The certainty of a word's mapping to its object, from a second modality (Alec, 2026-11-11; future work)

Alec, asked what trust `interpret` should give a new definition:
"Conceptual trust is just the certainty of the concept (its length), so
interpret should somehow reflect the certainty of the mapping from the
signifier to its signified. We might not be able to do that without direct
evidence of the signified coming from a different modality". He named the
math training corpus as the place it might be done, "one" being a symbol for
the continuous value 1.0, and several modalities as definite future work.

**What exists.** A concept's trust is its length. `InterpretLayer.define`,
the one writer of a word's definition row, writes the pair $(1, 0)$; the
row's effective evidence is that pair under the least of the word's length,
the object's length and the gate's cap (`LTM1.refresh_effective`). So the
mapping's certainty is stood in for by the certainty of its two ends: nothing
measures the mapping itself. The last architecture round gives a toy of the
measurement for one scalar beside the words, where each moment that holds both
the word and the number is one view of the definition
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
section 3.10). What follows is beyond that scalar.

**To specify.** A moment that holds two modalities at once, a word and the
value it names; the evidence that the object named by the word is the
concept the other modality gave, counted as views of the definition row as
a concept's views are counted now; and the definition's trust as that
agreement. The math corpus is the first case: "one" beside 1.0.

**What it waits for.** The numeric input of the last architecture round, and
a decision on how two modalities share a moment.

## Operations on continuous values: sum and product on the feeling (Alec, 2026-11-11; to settle)

Alec, having placed a scalar beside the words in the last architecture round,
where feelings are continuous and concepts discrete: "We might also want some
actual mathematical operations in this space: less than, multiplication,
addition, min, max that can operate on the values directly rather than
operating on the codes as proxies offerred by the codebook."

Settled for now, the same day: part, max and min, so that no overflow rule
has to be invented; they give a fuzzy ordering with a t-norm and its conorm.
The last architecture round has those three and builds no sum or product on
values: on a percept whose code is the value, less-than is the part-of test
(`Ops.partOf`), and the least and the greatest of two readings are the field's
`intersection` and `union`
([specification](plans/2026-11-11-item1-8-last-architecture-round.md),
section 3.9).

**What would be new.** A product and a sum of values. Today `ProductLayer`
and `SumLayer` act on two concepts' codes (a coordinatewise product, marked
exploratory; a mean), and `conjunction` multiplies activations.

**To settle.** A sum of two values leaves the unit interval: bounded at 1,
the mean, or $a + b - ab$. And the rule of 2026-10-07 that the math cannot be
direct: once a numeral is grounded in the scalar, an addition on values
answers "one plus one" with no chain of successors. Claude's reading of
how both hold: that rule is about words; these are operations on the
feeling, as animals and infants compare and roughly add magnitudes without
symbols (cited from memory, not checked: Dehaene 1997; Cantlon and Brannon
2007); and they stay out of what a word's arithmetic can call. In floating
point the feeling's sum is exact unless something makes it approximate.

**Settled (Alec, 2026-11-13).** "arithmetic do not operate on the codes:
that would break codebook lookup. So even with number, we categorize it
perceptually: that puts us in a codebook, and percept activations have a
kind there but no magnitude. So for now, the arithmetic operations are
conceptual-space only." The rule of 2026-10-07 stands. The four of
arithmetic, add, subtract, multiply and divide, act on concepts, and "can
work in tanh-bounded space or something (0..inf)": in the chart, mapped
back to the unit range ([Language](Language.md), "The operators"). What
is left to say is what a difference does below zero and a quotient at a
zero divisor.

## Removing the tanh and atanh of sigma and pi (Alec, 2026-11-13; to investigate)

Alec: "it seems that the tanh and atanh were added as excessive precautions
against unit overflow: we can just use clipping and some remedy for
saturation."

- **What they are now.** `SigmaLayer` is `tanh(atanh(x) @ W)`, and `PiLayer`
  with its tanh is `tanh(W @ atanh(x) + b/2)`. Stacked, the tanh that leaves
  one and the atanh that enters the next cancel, so the pair adds no
  nonlinearity, and the gradient of atanh has to be capped near the walls.
- **What else they do.** The pair is a bijection, so `reverse` is exact; and
  a sum made in those coordinates approaches one without being cut off.
- **To investigate.** Clipping with a gradient passed straight through in
  their place; what `reverse` returns at a clipped coordinate; and a pi that
  is a product over unsigned values, in which a zero stops the product,
  where the odds of `PiLayer` make an absent value the identity
  ([Reasoning](Reasoning.md#the-split-of-sigma-and-pi-alec-2026-11-13)).

## Learning the structure of interpretation (Alec, 2026-11-13; future work)

When a word is created, the structure it is interpreted through is created
with it: "meta-concept(symbol(word-concept), symbol(object-concept))".
"Meta-concepts, which we have introduced previously, are concepts that combine
the symbol for the word-concept and the symbol for the object-concept." The
meta-concepts are collected under one symbol, "which will basically act like a
dictionary, and act as a cue to the architecture that an object should be
interpreted", and `interpret` "will replace word-concept with (one of) the
object-concepts found under that meta-symbol (this will also handle
polysemy)". Meta-concepts are introduced on page 112 of *The Whole Part*,
one of the books attached to the author's ORCID record
([0000-0002-6570-7571](https://orcid.org/0000-0002-6570-7571)). The language
update of 2026-07-02 has the same pair, as an ordered pair of word-symbol and
object-symbol ([DevelopmentHistory](DevelopmentHistory.md)).

- **To learn it.** "All of that could be learned, but lets mark that as future
  work." The structure is built when the word is created; learning it in
  place of building it is not attempted yet.
- **To make it fast.** `interpret` is made accurate first, "even though it is
  not efficient, and then we can create a cache or bootstrap if necessary".

## Left open by the split of sigma and pi (Alec, 2026-11-13)

- **Splitting the definition store.** Its rows could be of two kinds, a symbol
  as a pi over concepts and a concept as a sigma over symbols, where one table
  holds both folds over the same rows today. "The definition store can still
  be split, just flag it in future work"
  ([Reasoning](Reasoning.md#the-split-of-sigma-and-pi-alec-2026-11-13)).
- **The trust of a word's mapping, set by success.** "the trust of a words
  mapping is really something the architecture should be setting based on
  success. lets just use one, and flag it." It is one now.
