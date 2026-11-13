# Item 1.8: the last architecture round

2026-11-11. The specification of todo item 1.8, the round after item 2.0's
whole-path round and
before the optimization items (1.5, 1) and the final housekeeping. Alec
decided its contents in one conversation on 2026-11-11 and asked for it to
be written (decision 19). This text consolidates a first writing that grew
by additions through that day; how it grew is in `todo.md`'s lines of that
date. Claude wrote it; Codex builds it. The names below were read on round
7 as landed, `a39cbfdef8`, and are to be checked again on the whole-path
round's landing, which changes some of them. There is no Codex text yet
(§7).

## 1. Decided (Alec, 2026-11-11)

1. **The round and its place.** "all should go into the last architecture
   round (which should happen before optimization and final
   housekeeping)". "It sounds decided as item 1.8". "Let's write the 1.8
   spec while waiting for codex. Let's not worry about who does 1.5 , 1,
   0.5; we just want to work in parallel, with you doing most of the spec
   work."
2. **The grammar file tells System 1 from System 2; three operators are
   field operators.** "System 1/2 is the grammar Vs no grammar distinction,
   which can be broken down even more finely; if we allow the interpret
   operator, we end up with symbols. Otherwise we end up with (e.g.) the
   intersection/union/negation grammar that defines animal cognition. So
   maybe "no grammar" models should just look like a grammar file with
   three operators instead of a hardcoded sigma-pi-negation over all
   three?" "Ok. So we can distinguish serial and parallel learning by the
   grammar file, and three of those operators are field operators, the
   rest being vector operators." "Conjunction/disjunction stay symbolic,
   we generalize union/intersection to multiple-concept operations."
3. **A point is understood by its projection onto the field.** "we can keep
   doing what we are doing with "unique" conceptual vectors (using
   operators on vectors to create points in conceptual space), but then
   the understanding of those will involve their projection onto the
   entire conceptual field. So in that way, parallel reduces to serial at
   the grammar layer, and then projects back. But animal cognition always
   involves [the] complete conceptual field and the three operators that
   can be applied to that whole field (the fuzzy tetralemma logic)." On
   what a serial operand is: ""One anchor and all concepts entailed by
   it", since part and whole are kind or mereological terms. But whole
   entails part and part does not entail whole, so they are similar
   mechanism (at least at order zero). These entailments are just the
   existing conceptual order." On the single vector: "As an argument
   against the single vector processing as opposed to the field; it's
   basically a grandmother neuron. But being able to use a single vector
   to stimulate its conceptual entailments; reasonable." On what a field
   cannot hold: "memory stores known things, not novel things." And:
   "Concepts in parallel can coexist, in serial they have to be a part of
   different STM slots."
4. **The loop is an examination phase, and an animal has it.** "So serial
   learning is enabled by interpret() and the loop that allows paying
   attention to a sentence multiple times ( in order to do serial
   processing). Should we allow the loop in animal cognition, allowing
   animals to pay attention to different aspects of the image even though
   they do not have the ability to symbolically represent those
   attentions? In which case, the loop is best seen as an examination
   phase instead of a symbolic loop." Claude answered yes (§2.2: the loop
   and its six actions exist and need no symbol); decision 1's "It sounds
   decided" covers it.
5. **An intuition may answer and reconstruct.** "An intuition is just as we
   discussed; it can do both output and reconstruction".
6. **Trust.** "Trust of percepts is 1, seeing is believing. After
   word-concepts are turned into object-concepts, though, trust is
   uncertain (0) to start with." What `interpret` gives a new
   object-concept: "Interpret trust is ... TBA." And then: "Conceptual
   trust is just the certainty of the concept (its length), so interpret
   should somehow reflect the certainty of the mapping from the signifier
   to its signified. We might not be able to do that without direct
   evidence of the signified coming from a different modality; we might be
   able, therefore, to do it for the math training corpus; "one" is a
   symbol for the continuous value 1.0 . But multiple modality is definite
   future work."
7. **Operators are enabled by their trust, under one file and a full
   grammar.** "We sort of short-circuited corpus-gated learning by
   operator-gated learning. Since that can be written into the grammar
   file, we can just add/enable operations one at a time (hopefully
   without altering the graph?)" And: ""Enabling operators" refers to the
   fact that a model can be produced by training with one model file and a
   full grammar, and operator trust is also stored, so that we don't
   introduce too much too fast (I think we had trust as the gate
   previously). Some operators like np vp are essential even to derive NPs
   from events, so will be necessary before the other operators; we can
   infer these from priority in a syntactic tree."
8. **Memory is addressed by content.** "the query() on concepts needs to be
   there also. So that means that .when does not seem necessary at this
   point: we store episodic concepts (probably by reference) and a single
   semantic concept (by content), and while order is important (and there
   is a .when baked into the semantic concept and possibly the episodic
   concepts), but there is no external .when or .where (it is not
   perceptual memory)." "Ok, keep .when as an external model time. But we
   shouldn't be using it; LTM is content addressable, and especially as we
   are storing concepts, they will have a .when muxed into them when they
   are pulled out of the conceptual codebook." "I think .when and .where
   were useful when LTM stored percepts, but if we are not storing
   percepts, we have no need of those fields. Do animals need perceptual
   memory, or is invertible concepts enough? I think for now it's enough."
9. **Two modalities, in and out, on the conceptual field.** "A continuous
   output (and continuous input) can be treated as alternate modalities."
   "Reconstruction is verbal from the conceptual field, or numeric from
   the conceptual field; these are now integrated." "Numeric input can
   just be a linear layer onto concepts, the opposite of numeric output,
   so let's do it. It will make xor much easier, also." "Linear layer
   might as well operate over all xor tests, as long as we have verbal
   tests that cover what they were doing symbolically." Earlier the same
   day: "xor should probably be converted to a system 1 model. That means
   just one concepts-to-concepts step at order 0", and "we probably want
   to have only a BasicModel class".
10. **Modifier plus noun is a selection.** "Modifier plus noun might be
   better as an actual selection operator over all concepts in the
   extension of the count noun. Agree?" Claude agrees (§3.5): it is how
   one of the operator's two paths already works.
11. **A cheaper forgetting pass.** "Good", to building it before the main
   run; the direction is his of 2026-10-24: "probably by building a
   running luminosity measure on LTM."
12. **The open points.** On the ten points of §5.1, after answering the
   fourth (decision 6): "1 promotion on recurrence", and "The rest of the
   defaults are fine."
13. **A scalar beside the words.** "Im actually leaning toward making
   numeric input/output a part of 1.8, so that interpret can have objects
   that it can perceive directly (e.g. interpret("one") = 1.0). it would be
   a toy example of multimodal for a 1-D (or N-D) numeric signal." Its
   scope: "the image processing can be future word, and the ND stuff also,
   but the scalar case can be the continuous "feeling" that corresponds to
   the discrete boundaries established by symbolic discrimination." What
   it is for: "But lets imagine tying this set of percepts to the existing
   text. If we provide both perceptual spaces to a single conceptual space
   for its integration, then we can ask questions like which is more or
   less, and have interpret() result in bindings to the other modality.
   That means that we can do things like teach number recognition just
   like large AIs do object recognition in photos."
14. **A number is a percept, and a percept has one lane.** "percepts could
   probably operate on the value directly, and parts/wholes just becomes a
   less-than discrimination on the number line." "so it would work for XR,
   but could generalize to images fairly easily." (Claude reads "XR" as
   XOR.) Of images: ".where would become 2D, and wholes and parts become
   hyperrectangles in the unit hypercube." Against a second pole that
   Claude had put on the percept: "No, percepts do not need two lanes,
   that is just for concepts."
15. **Part, max and min on values.** "so feelings are continuous, concepts
   are discrete, and we have a better model of nonconceptual knowing:
   continuity. We might also want some actual mathematical operations in
   this space: less than, multiplication, addition, min, max that can
   operate on the values directly rather than operating on the codes as
   proxies offerred by the codebook." Told that a sum and a product on
   values would be new and raise two questions: "Maybe part, max, and min,
   so that we don't have to invent overflow rules. That will at least give
   us a fuzzy ordering and norm/t-norm."
16. **A word for a number is a concept with bounds.** "does this in fact
   give us enough for a primitive example of object recognition?"; "or
   symbolization? can the model learn the word for a number (where that
   word is a concept with part and whole bounds that are
   hyperrectangles)?" Its limits: "although we can represent 1,000,000
   numerically, and we might not know what 1M amount of a feeling would
   feel like."; "so the mapping of feeling to words is not linear."
17. **The XOR proofs read 0 and 1.** "XOR is traditionally done in 2d, so 0
   and 1 would be fine for the input values."
18. **The scalar's defaults.** On six points of mechanism that Claude had
   put to him: "Your defaults are fine for this." §5.2 lists them, with
   the two that were revised afterwards.
19. **This writing.** "Please write the 1.8 spec , I will let you know when
   codex finishes the next 2.0 round".
20. **The scale: zero to infinity on the unit interval.** "We might as well
   allow numeric values in 0 to 1 range, but map that to "zero" to
   "infinity", so that we can represent all numbers, and let the scale sort
   itself out with some sort of spacing algorithm (so its an ordinal dim
   that becomes approximately real with increased ordinates)." The spacing,
   asked (2026-11-12): "Running rank (Recommended)", and what he means by
   it: "My understanding of "running rank" is a linear interpolation over
   recognized concepts, from 0-1." Told that over the numbers a language
   names often that spacing comes out close to logarithmic: "If that is the
   case, then we should do something like a logarithmic mapping from 0..INF
   to 0..1". So the spacing is a fixed logarithmic map (§3.10, §5.2 e).
21. **Recognition is a question of identity.** Asked whether `bind` should
   consider the concepts of the moment's field and count such a binding as
   perception: "You seem to be asking about identity. I had theorized that
   order-0 events could not be nouns, because generalizing over events
   could yield only a general event. But that serves in MTM as an order-0
   identity, and is used by order 0 as a promotion criteria." Then:
   "Confirming also that order-zero events are suitable for order-zero
   identity. I thing we already decided that MTM is ramsified, so this
   would be a necessary consequence". This amends the constraint recorded
   on 2026-11-03, "no order-0 identity is opened, since a chain at order 0
   is a sequence, not an identity" (behavioral-chains plan item 5).
   Claude's reconciliation, with his of 2026-11-04 ("MTM, when it links
   order-zero concepts (events), creates a single extended event
   (sequence)"): events linked by succession make a sequence; the same
   order-zero reading met again is an order-zero identity, the candidate
   that `observe_candidate` already opens for promotion. §5.2 d has the
   rule.
22. **The value is the code.** "Also, for the numeric modality, please
   confirm if we can use the values as the numerical codebook codes
   directly, rather than creating arbitrary codes, it would seem to be more
   efficient." Claude confirmed it (§3.9): it is also what makes part,
   max and min the code operations themselves.

## 2. Today (round 7 as landed, `a39cbfdef8`)

Read by three readers whom Claude briefed, and spot-checked by Claude where
a rule below rests on it. What a reader inferred and did not read is marked.

### 2.1 The same logic in several forms

Evidence is a pair of lanes, for and against, each in $[0, 1]$.

| Where | What it reduces over | Arithmetic | Learned |
|---|---|---|---|
| `SparseLayer.fold_presence` (`Layers.py`), the fold along stored definitions; called by `_compose_order0`, `_concept_rung_presence`, `cs_forward_content`, `cs_read_percept_field` | the edges of one concept's definition | pi: the least of $v^{w}$; sigma: the greatest of $1 - (1 - v)^{w}$; a dual pass gives the other lane | the edge exponents $w$ |
| `SparseLayer.attribute_presence`, its inverse; called by `cs_reverse_presence` | the same edges, downward | $y^{1/w}$; a sigma row chooses its heaviest case when no field is observed; a pi row attributes to every part | none here |
| `Attention.narrow_words`, actions 3 to 5 | the words one bracket covers | and: (least for, greatest against); or: the dual; not: the lanes exchanged | none |
| `Attention.field_reduce` | any axis | the same | none; its callers `FieldAndLayer.forward` and `FieldOrLayer.forward` are not on the loop's path |
| `IntuitionLayer.power`, `Intuition._evaluate` | a rule's literals, then its consequence | the same powers on least and greatest | `pi`, `sigma` |
| `Ops.intersection`, `Ops.union` (`IntersectionLayer`, `UnionLayer`) | two vectors | coordinatewise least (a zero coordinate passes the other operand) and greatest | none |
| `Ops._conjunction_kernel`, `_disjunction_kernel` (`ConjunctionLayer`, `DisjunctionLayer`) | two vectors | $ab$ on a product of unit directions; $a + b - ab$ on their probabilistic sum | none |

So the field already has one arithmetic, the weighted least and greatest on
two lanes. Only the two vector operators of the last row use the product
form.

`ConceptualSpace.add_concept_edge(row, col, weight, *, conjunctive=False,
negated=False)` writes a definition edge: "W_pi intersects order-0 field
concepts. Order-0 W_sigma unions alternatives in that field", and a negated
part addresses the other lane. The fold reads the conjunctive matrix
wherever `conceptualPi` is set. Nothing calls it with `conjunctive=True`:
the intersection of several concepts has a fold and no writer.

### 2.2 The loop of attention

`Attention.narrow_words` runs up to `attentionBudget` steps (32) over a
`BracketTable` (intervals of the input, with `done` and `spent`). At each
step the chooser picks one of six actions, the grammar file's field rules:

- `divide` (0) splits a bracket at the first place where its lanes
  disagree; `descend` (1) cuts at the first part; `gloss` (2) accepts a
  word that reads purely;
- `and` (3), `or` (4), `not` (5) reduce the lanes of the words a bracket
  covers (§2.1) and write the result for that bracket.

`narrowing_mask` gives the legal moves from the four corners: `divide`
where a bracket reads both, `descend` where it has parts and reads
neither. The chooser is `model._stm_reducer()`, the compose
`operation_layer`, scored through `score_unary` against `bracket_anchor`;
it is credited with the compose choices in `_compose_score_function_loss`.
`ModelAttention.handoff` publishes what was accepted and the lanes. Before
the loop, `FamiliarTier.open_sentence` reads every word's place at once
with no grammar: the glance.

### 2.3 What switches the sentence driver on

`_derive_use_grammar` returns `'none'` for a file with only field rules and
the unary pi and sigma folds, else `'all'`. `_has_word_objects` is false
where `OutputSpace.conceptIds` is set; with `'none'` it is the value of
`ltmConsolidation`; otherwise `conceptLayers > 1 or ltmConsolidation`. The
driver `_run_sentence_word_bricks` runs where `_sentence_ends`
(`word_brackets and operation_layer is not None`). Its phases:

1. clock, positions, `begin_sentence`;
2. perception word by word (`forecast_percept`, `perceive`, `symbolize`);
3. the glance (`_prime_sentence_symbols`, `open_sentence`);
4. attention's loop (`_attention_read`);
5. compose trials (`sentence_pair` at `a39cbfdef8`, replaced by
   `WholePaths` in the whole-path round: `handoff`, `SymbolField.read`,
   `compose_word`);
6. the path's cost (`_sentence_path_cost`: `SentenceField.reconstruct`);
7. the kept handoff and the published field;
8. `_commit_sentence` (`commit_read_words`, MTM promotion, the closing
   thought);
9. `ConceptSpread.complete`, `finish_sentence`.

Phases 1 to 4, 7 and 9 need only the field. Phase 5 needs the grammar's
operators, and `commit_read_words` in phase 8 needs `interpret`'s readings.
The whole-path round rewrites phases 5 to 8 as paths.

`XOR_exact.xml` never runs the driver (it sets `conceptIds`): it runs the
parallel body with the field, where `teach_concept_lessons` teaches
memberships, `observe_native_cases` and `promote_candidates` promote each
familiar group of percepts to a case row, and the label trains the sigma
edges from the output concept to the cases, added at weight zero. Its
"conjunctions" are percept wholes, not intersections of concepts.

### 2.4 A point and a field

- A point is one vector of the conceptual space: a word's value, an STM
  slot, a composed root. A field is evidence over known concepts:
  `ConceptualField` (`concept_ids`, `evidence` of shape concept by stream
  by reading by two lanes, `where`, `when`).
- Point to field happens in two places today. At the end of a generate
  walk `OutputOccurrences.generated` reads each emitted word against the
  bank (`readback_scores`, a cosine on the form band) and builds a field,
  closed over its definitions. And `IdentityRows.meet_ratios(query,
  candidate)` gives the two coverages $\sum \min(q, c) / \sum c$ and
  $\sum \min(q, c) / \sum q$; twelve modules use it for containment
  (`ConceptTrust.observe`, `CaseView.coverage`, `Intuition.write_knowing`
  and others).
- Field to point: a class's content is the evidence-weighted centroid of
  its members (`ConceptHierarchy.refresh`; Alec, 2026-10-24).

### 2.5 Modifier plus noun

`CompoundLayer`: "Select the right head's native sigma cases with the left
modifier". Where the noun's cases are staged, `CaseSelection.select_cases`
keeps `torch.minimum(requested, mod)`: the noun's cases met with the
modifier's observed field, and a repeated test is the same test
(`CaseView.select`: "Reapplying a test is an intersection, not another
gain"). Where the operands are meaning-layout vectors, `select_view` tests
by a learned kernel (`meaning_kernel`) in the meaning space.

### 2.6 Intuition

`ThoughtFaces.intuit` is a thought executor. Its terms are LTM1 rows that
`ThoughtLTMCapability.intuition_rules` finds through the expected knowing:
antecedent, consequent and the row's evidence as trust. `Intuition.execute`
records evidence of kind `intuition`; `AccessibleMind` sends it to
`Intuition.write_knowing`, which merges the consequence into the field.
A held firing is offered again as an `infer` candidate, and only then can
it be an answer: `ThoughtAnswer.accepted` rejects kind `intuition`. The
bank's `pi` and `sigma` belong to the output objective.

### 2.7 Trust

- *Percepts.* `cs_read_memberships` reads the percepts' lanes with no trust
  factor.
- *Concepts.* `ConceptTrust` keeps views and agreement for each concept and
  sets its radius to agreement over views times $1 - e^{-\text{views}/3}$
  (`trustViews`). A new word and a new object-concept start with no views,
  so radius 0.
- *Definitions.* `InterpretLayer.define`, the one writer of a word's
  definition row, writes the pair $(1, 0)$ unless told otherwise; the row's
  effective evidence is that pair times the least of the word's radius,
  the object's radius and the gate's cap.
- *Operations.* Each operation has a concept (`OperationSeats`), seated at
  radius 0; every kept use is one view of it (`PartsOfSpeech.observe_kept`).
  `LTM1.refresh_effective` scales each row's evidence by the least of its
  endpoints' trust, its operation's radius and `TrustGates.cap`.
- *Gates.* Two, in `model.xml`: `interpret` (opens at a stable-word
  fraction) and `clause` (opens at a threshold on `lift`'s definiteness,
  and exempts `interpret sign bind lift` by a written list). A closed gate
  caps evidence at `closedTrust` (.1); it never removes an operation from
  the chooser's menu. Nothing derives an order among operators: there is no
  precedence table (`surface_schema` and `rule_probability` have no live
  reader). `OperationSelectionLayer.set_stage_menu`, which would mask operations in
  the chooser, has no caller.

### 2.8 Memory, time and place

- An LTM0 row's vector holds a where band and a when band
  (`when_start`; `RowTime`: "Model time in the percept band, never a
  separate durable time column"). An LTM1 relation row has a separate
  column `percept` of eight coordinates, where then when.
- Time is *addressed* in two places: `EntryRead.recall_time`, reached by
  `ThoughtLTMCapability.query(when=...)`, which is also the only way to a
  review of an episodic entry; and the interval scopes of
  `EntryRead._scope_contains`.
- Time is *read as content or order* in many: forgetting's age
  (`history_age`), recency (`Occurrence.recency_weights`), the ages given
  to learned readers (`RowTime.age_band`, `relative_idea`), the order of
  events (`Queries.relation_evidence`), the order of relation candidates in
  `cued_rows` (by the `percept` column, newest first).
- The cued read `cued_rows` scores a cue against stored meanings role by
  role, with the when band zeroed on both sides. It does not look at an
  entry's references.

### 2.9 Numbers in and out

- *Out, three ways.* `OutputSpace.read_concepts` (under `conceptIds`): the
  for-lane of a named concept, with no weights. `OutputSpace.forward`'s
  linear map on the flattened symbolic vectors, with a regression bias.
  And `_forward_head` on the per-word path: a linear read of the root and
  end slots, plus `answer_record_reader`.
- *In.* `dataType` `numeric` builds a dense slab that the perceptual and
  conceptual bodies transform; there is no percept identity and no field
  of known concepts. No XOR configuration uses it.
- *The XOR proofs* (learning certificates; 146 files of the test tree name
  an XOR configuration, most as a fixture):

| Proof | Configuration, pipeline | Read-out | Bar |
|---|---|---|---|
| `test_mm_xor.py::TestMMXorConvergence::test_convergence` | `MM_xor.xml`, whole slab | linear on symbolic vectors | error below .20 in 200 epochs; a stopping gate of every round |
| `...::test_learns_xor_signal` | the same | the same | error below .26 in 600 epochs |
| `...::test_mm_grammar_learns_xor_signal` | `MM_grammar`, per word | `_forward_head` | error below .20 in 900 epochs; behind `RUN_TRAINED` |
| `test_grounded_xor.py::test_native_unseeded_xor` (six cases) | `MM_xor_fixture.xml`, parallel field | the field, read in the test | four cases promoted, error below $10^{-6}$, two positive edges, controls zero |
| `test_concept_output.py`, the curriculum (three) | `XOR_exact.xml`, parallel field | named concept | exact XOR, two positive weights, inputs decoded, controls zero |
| `test_explicit_dimensions.py::TestXorExactCliReconstruction` (two) | `XOR_exact.xml` from the command line | named concept | reconstruction at least half, error below .05 |
| `...::TestXorGrammarLearnsXor::test_xor_class_accuracy` | `XOR_grammar.xml`, per word | `_forward_head` | four correct, error below .05; behind `RUN_TRAINED` |
| `...::TestXorGrammarReconstruction` | the same run | words | four of four word multisets |

### 2.10 The forgetting pass

`Forgetting.retention_values` recomputes every term at each pass. Its
luminosity compares every row's vector with every other's in blocks of 128
(`points[start:end] @ points.T`); with `forgetCoverage` it also leaves each
row out in turn. `BasicModel_corpus.xml` sets `ltmCapacity` to 1,048,576.

### 2.11 A word and its object

- `InterpretLayer.forward` takes a word's object from the word's existing
  denotations: with exactly one it calls `define(word, obj)`; with several
  it leaves the choice to the grammar. `InterpretLayer.define(word, obj, *,
  description=None, trust=None, evidence=None)` is "The sole writer of
  word DEF object" and takes evidence.
- Objects are not perceived. They are minted by
  `FamiliarTier.promote_identity` from retained occurrences, so every
  object is itself made from words.
- An object is said by its word: `ConceptualSpace.word_surface_for_row`
  ("Spell a witnessed object or a form still waiting for its object")
  calls `InterpretLayer.lexical_form`, which returns a form only where
  exactly one word names the object (`definitions.words(obj)`).
- A percept's field has a pole axis (`PerceptField.values`: percept by
  batch by reading by pole). `Ops.partOf(S1, S2)` is "Elementwise S1 <= S2
  reduced over the last dim with all".

## 3. Rules

Where a rule is Claude's reading or default and not Alec's word, it says
so and is listed in §5. Alec accepted those of §5.1 (decision 12), so each
stands as decided; §5.2 lists three that he has not spoken to.

### 3.0 Terms, and the operation this round introduces (Alec, 2026-11-12)

Alec, 2026-11-12: "For kinds, a token implies a type. For mereology, a
whole implies a part. So the Idea is the intension of the
sentence-concept. The Field is the extension of the sentence-concept. The
sentence words can be transformed into a single vector in conceptual
space, which does not (necessarily) exist as a single concept in
conceptual space. That intension can be transformed into a set of implied
concepts by lowering its order over the field of known concepts."

In the code's names (Claude's mapping):

- *The sentence-concept* is what a sentence's words compose to. It need
  not be a known concept, so it has no reference; it is held as a vector.
- *The idea* is that vector, the sentence-concept's intension: the
  composed point, stored whole in `slots[row]`.
- *The field* is the sentence-concept's extension: the known concepts the
  idea implies, each with its evidence (`ConceptualField`). The entry of
  an `LTM0` row keeps its strongest members by reference (§3.16).
- *Lowering the idea over the known concepts* is `FieldProjection.stimulate`
  (§3.3). Its two steps are the two implications: a whole implies a part,
  which is the coverage of a concept's code in the point; and a token
  implies a type, which is the fold through the definitions. This is the
  operation the round introduces.
- A type reached by the second step is of a higher order than its token
  and still below the sentence-concept; "lowering" is said of the
  sentence-concept (Claude's reading). §3.3 calls that fold "upward",
  meaning from a token to its type.
- An `LTM0` row therefore holds both sides of one sentence-concept: its
  intension, the idea, and the strongest of its extension, the entry.
- *The formula selects the extension.* Alec, 2026-11-12: "We already have
  a mechanism for mapping words to a sentence concept bidirectionally. So
  if we walk the idea to the sentence (a WFF), the wff can select the
  extension precisely." This is §3.3's rule for a composed point: its
  field is its generate walk's, and the projection does not bypass the
  grammar's inverse. A known concept is lowered by its stored extension; an
  idea is lowered by walking it to its formula, whose leaves are known
  concepts and whose operators say what is selected among them. The match
  of a bare vector against every concept's code (`stimulate` on a point)
  is for a leaf and for a file with no vector operator.
  - At a sentence's close the formula is in hand, so the entry is taken
    from it with no walk. The walk is for an idea met without its
    sentence: one recalled from `LTM0`, or one a thought concluded.
  - As built, `OutputOccurrences.generated` stimulates each emitted leaf
    and sets the leaves' fields side by side; the formula's operators are
    not applied over those fields, except `compound` (§3.5). The
    determiner as the selector among a kind's members (§3.13) is the
    missing one. (Claude's reading of the code and of what the quotation
    asks.)

### 3.1 One grammar file, three kinds of operator

- *Examination*: `divide`, `descend`, `gloss`. They move attention.
- *Field*: `intersection`, `union`, `not`. They take and return evidence
  over the field (decision 2).
- *Vector*: every other operator, `interpret`, `bind`, `lift`,
  `conjunction` and `disjunction` among them ("Conjunction/disjunction
  stay symbolic"). They take and return points.

The file is the switch (decision 2). What a file declares decides what
runs:

- field operators only: the glance and the fold along definitions, on the
  parallel body, as `XOR_exact.xml` runs today;
- examination operators: the loop of §3.4;
- vector operators: compose and generate; `interpret`'s readings are
  committed where `interpret` is declared.

`_derive_use_grammar`'s two values and the conditions of
`_has_word_objects` on `conceptLayers`, `ltmConsolidation` and
`conceptIds` give way to that. No configuration changes its pipeline
silently: before the switch moves, every configuration under `data/` and
every test fixture whose pipeline would differ is listed with its port,
and `model.xml`'s default grammar keeps the parallel body (its examination
rules move to the files that examine). Claude's safeguard.

### 3.2 The field's logic has one definition

- *Arithmetic.* `intersection` and `union` on the field are
  `SparseLayer.fold_presence`'s weighted least and greatest on the two
  lanes, over whatever they are applied to: the edges of a definition, the
  words of a bracket, the concepts of a cue. Attention's actions 3 and 4
  are its unweighted case, and `IntuitionLayer.power` is the same
  arithmetic. `not` exchanges the lanes; `non` withholds both, as now.
  This is the fuzzy tetralemma logic of decision 3. The product forms stay
  inside `conjunction` and `disjunction`.
- *Names.* The file's field rules `and` and `or` are `intersection` and
  `union`, one operator each: `FieldAndLayer`, `FieldOrLayer` and
  `Attention.field_reduce` go. `Ops.intersection` and `Ops.union` remain
  as coordinate helpers for forms (Claude's reading).
- *Applied by the grammar.* Where the grammar applies a field operator to
  chosen operands, it reads their fields (§3.3) and writes a field; the
  point that stands for the result is that field's key.
- *The intersection of several concepts needs a writer.* The conjunctive
  definitions have a fold and no writer (§2.1). Alec chose promotion on
  recurrence (§5.1 item 1). A set of concept lanes found
  active together again (the existing candidate rule of
  `observe_candidate` and `promote_candidates` under `raiseCount`) is
  promoted to an order-zero row whose conjunctive edges are those lanes,
  written by `add_concept_edge(..., conjunctive=True, negated=...)`. That
  is what `observe_native_cases` does today for a familiar group of
  percepts. What proposes a set: the glance, in a file that does not
  examine; in a file that examines, only a bracket on which the loop chose
  `intersection`, so that a text is not promoted sentence by sentence.

### 3.3 A point and its field

- *`stimulate(point)`.* For each known concept, the evidence that the
  point entails it is the candidate coverage of the concept's code in the
  point, lane by lane ($\sum_d \min(p_d, c_d) / \sum_d c_d$,
  `IdentityRows.meet_ratios`), followed by the fold of §3.2 upward. It is
  decision 3's "use a single vector to stimulate its conceptual
  entailments", and "whole entails part": a concept is entailed to the
  degree its content is contained.
- *The field of a composed point.* A point made by vector operators is
  unfolded by its generate walk, as reconstruction has done since
  2026-11-10; its field is the walk's (`OutputOccurrences.generated`),
  with each emitted leaf stimulated. The projection does not bypass the
  grammar's inverse (§5.1 item 2).
- *The key of a field* is the evidence-weighted centroid, as built.
- *What reads a stimulated field in this round:* a field operator applied
  to chosen operands (§3.2), modifier plus noun (§3.5), the query's cue
  (§3.8) and the numeric output of a file that has vector operators
  (§3.9).
- A field holds known concepts only. A new combination is a new pattern
  over them; it is stored as references and a key, and becomes a concept
  when it recurs (decision 3, "memory stores known things").

### 3.4 Examination

- The loop of §2.2 runs in every file that declares an examination
  operator, with or without vector operators. Its chooser stands without
  the compose operators: in a file that has none, the chooser's menu is
  the field actions.
- In a file with no vector operator a path is the glance, the loop and the
  field it leaves. Its cost is $R$, `reconstruction_reverse` of that field
  (§3.9), plus $A$ where an output is supervised. An explore path departs
  once in the loop. Nothing is composed, generated or interpreted, and no
  action is recorded as a symbol.
- The close still writes the moment: an anchor row of kind `episode` with
  its episodic entry, the references in reading order, which is the memory
  of the examination; and MTM's order-zero chains as built in item 2.0.
- In a file with vector operators the loop is what it is today.

### 3.5 Modifier plus noun is a selection

`compound` selects, among the concepts in the noun's extension (its staged
sigma cases), those of which the modifier holds: the field's intersection
of the noun's cases with the modifier's field, as
`CaseSelection.select_cases` computes it, idempotent. The learned kernel
test of `select_view` remains only where the modifier has no field over
the noun's cases, and each use of it is counted (§5.1 item 7). `compound` stays a vector operator in the file: two chosen operands
in, one point out, a field operation inside.

### 3.6 An intuition may answer and reconstruct

- `ThoughtAnswer.accepted` accepts evidence of kind `intuition` where the
  meaning has no open slot. An accepted intuition is the operand of the
  output walk, or is read by the numeric output.
- What an intuition wrote into the field (`Intuition.write_knowing`) is
  part of the field that reconstruction reads, verbal or numeric. Nothing
  more is built for that.
- `pi` and `sigma` train from the kept path's output, under the
  whole-path round's rule for the reader.
- Its terms stay the rules found through the expected knowing. In a file
  with no vector operator there are no such rules, and System 1's step is
  the fold along definitions (§5.1 item 8).

### 3.7 Trust, and operators enabled by it

- *Percepts*: evidence read from percepts is taken as read, as built
  ("seeing is believing").
- *A new object-concept* starts at radius 0, as built: a concept's trust
  is its certainty, its length. A definition's trust should be the
  certainty of the mapping from the word to its object, and that needs
  evidence of the object from another modality (decision 6), which this
  round gives for the scalar alone (§3.10); beyond it, future work
  ([FutureWork](../FutureWork.md#the-certainty-of-a-words-mapping-to-its-object-from-a-second-modality-alec-2026-11-11-future-work)).
  Every other definition row stays $(1, 0)$ under
  the least of the word's length, the object's length and the cap, which
  stands in for the mapping's certainty by the certainty of its two ends.
- *One file, the full grammar.* No stage file and no mask:
  `OperationSelectionLayer.set_stage_menu` and `stage_menu` go, having no caller.
- *An operator's trust* is its concept's radius, as built: it rises with
  kept uses and scales the evidence of what the operator wrote.
- *The order of introduction is inferred, not listed.* The `clause` gate's
  written exemption (`interpret sign bind lift`) gives way to a rule over
  the syntactic tree. The rule (§5.1 item 3): the fixed
  operations and the operators that can form a clause's root in the file
  are exempt from the start; every other operator's evidence is capped at
  `closedTrust` until the operators that stand above it in kept trees are
  definite (`PartsOfSpeech.operation`'s definiteness at the gate's
  threshold). One measured, monotone gate for each operator, where there
  are two today.

### 3.8 Memory is addressed by content

- *The query on concepts.* `ThoughtLTMCapability.query` reaches episodic
  entries by their references. The cue is a field (§3.3); a row's match is
  the coverage of the cue in the row's references, by `meet_ratios` on the
  two lanes that the signed values give; a cue may combine concepts with
  the three field operators. An index from concept to rows is kept at
  `EpisodicMemory.write` and at forgetting. The best rows are reviewed by
  `EpisodicMemory.reinstate`, as a recall by time is today.
- *Nothing is addressed by time or place.* `EntryRead.recall_time`, the
  `when` and `window` branch of `query`, `best_match`'s routing of a cue
  that holds only a time, and the interval branch of `_scope_contains` go.
- *The relation rows' `percept` column goes.* `cued_rows` orders relation
  candidates by store order; `history_age` gives a relation the age of the
  sentence row it came from; `MemoryPaths` copies no bands. Every reader
  and writer of the column is listed with its replacement.
- *What stays:* the model clock (`when_time`, `RowTime.next_time`), the where
  and when bands inside an LTM0 row's vector, and every reader of time as
  content or order (§2.8). A reviewed entry's readings take their time
  from the row's own stamp, as now: the time is in the concept, not beside
  it.

### 3.9 Numbers in and out, on the field

- *In: a number is a percept* (decision 14). For `dataType` `numeric` on a
  file with field operators, each number of the input row is one basic
  percept of its own, and its presence is the value, in $[0, 1]$. It has
  one lane: "percepts do not need two lanes, that is just for concepts",
  which is Philosophy's "Perception carries no negation". No layer is
  learned at the input, and the dense slab that `numeric` builds today is
  not used on such a file.
- *The value is the code* (decision 22). A basic percept's code is built
  from its native atoms' codes (`BasicPercepts.code`: the join of the
  atoms' rows in the native basis), and for bytes those rows are drawn
  patterns. A numeric dimension instead has one atom whose basis row is a
  fixed axis of its own band in the percept cube, and a reading's code is
  the value on that axis: $f(n)$ (§3.10) at that coordinate and zero
  elsewhere. No row is allocated for each value, no pattern is drawn and
  no collision is separated. It is what makes decision 14 hold in the
  codes themselves: `Ops.partOf` ("Elementwise S1 <= S2") on two readings'
  codes is their less-than, and their join and meet, the field's `union`
  and `intersection`, are their greatest and least. A drawn code for each
  value would lose the order.
- *A value is never told by its direction.* Every reading of one dimension
  has the same direction, so a reader that identifies by direction, the
  "scale-free perceptual cosine" of `readback_scores` or a nearest-code
  snap, sees one identity, which is right: the identity is the dimension,
  and the value is the magnitude along it. Values are told apart by order
  and containment (part, min and max; `meet_ratios`), as the boxes of
  §3.10 read them, and by nothing else.
- *Its concept.* Each numeric percept has one order-zero concept, made at
  construction, whose single feature edge reads that percept, so the
  concept's for-lane is the value. A value of 0 is no presence and gives
  no evidence.
- *Operations on the value are part, max and min* (decision 15), and they
  are the field's own: part is the ordering, min is `intersection` and max
  is `union` of two readings (§3.2). That is a fuzzy ordering with a
  t-norm and its conorm, and nothing overflows. No sum and no product on
  values is built; `SumLayer` and `ProductLayer` stay operators on codes.
- *Out.* `OutputSpace` reads the field by one linear layer: a weight for
  each concept row of the inventory and each lane, fixed in shape at
  construction, zeroed for a row when the row is retired, and a bias. It
  replaces `read_concepts`, whose named concept is the one-hot case
  (`conceptIds` goes, §5.1 item 7), the vector-linear branch on field
  configurations, and `_forward_head`'s read of the sentence state with
  `answer_record_reader`.
- *What the output reads* is the field at the path's end: in a file with
  no vector operator, the field after the fold; in a file with vector
  operators, the field (§3.3) of what the thought concluded, or of the
  understanding where the file has no thought entries.
- *Reconstruction is one cost from the field* (decision 9).
  `reconstruction_reverse` runs `BasicModel.reverse` from the field down
  to the percepts: a text percept ends in its bytes, a numeric percept in
  its presence, which is the number.
- *The XOR proofs* of §2.9's table, and any other learning certificate on
  an XOR configuration that the inventory finds, move to numbers in and
  out on a file with field operators only. Their input has two dimensions
  with the values 0 and 1 (decision 17): two numeric percepts, where a 0
  is no presence and a 1 is felt as $f(1) = .409$; the linear output
  takes the scale, so nothing in the proof depends on it. No absence has to be inferred, because the output is
  linear with signed weights and XOR is
  $x_1 + x_2 - 2\,(x_1 \wedge x_2)$. The one concept needed beyond the two
  inputs is their intersection, both present, which §3.2's promotion
  writes when the two recur together. Names are kept and bars are kept or
  tightened. The old read-out's assertion of "two positive edges"
  (`XOR_exact.xml`, the grounded test) is ported with a record: the
  weights are now two positive and one negative. This ends the numeric
  head of whole-path §3.12.
- *The controls.* The affine control (`sum`) keeps its floor of
  $\tfrac14$. A second form is added: the same file without the promotion
  of the intersection stays at $\tfrac14$.
- *The verbal cover.* Each proof's row in the inventory names the verbal
  test that covers what it did symbolically, before it moves (§5.1 item
  5): `TestXorGrammarReconstruction` stays on the symbolic file, where
  `conjunction` and `disjunction` compose the four sentences and the
  inverse returns their words; the tests of what the two operators mean on
  meanings stay; and the four sentences answered with a word for the class
  is added as a measurement.

### 3.10 A scalar beside the words

Decision 13: one continuous value, the "feeling", may accompany a
sentence, and the words draw their boundaries on it. Two perceptual
spaces, the text's and the scalar's, give one conceptual space its
evidence; the conceptual space already reads two sources today, the part
tower and the whole tower (`cs_read_memberships` in `open_sentence`).
Where a rule below is Claude's default that Alec has not spoken to, it
says so (§5.2).

- *The moment.* A lesson sentence may carry one number beside its text.
  The text is read as now. The number is a percept as in §3.9; it and its
  concept join the glance's field as one more reading, with no place.
- *The object is perceived.* A word read in a moment where the scalar is
  present has a concept over the scalar as a candidate object, and
  `interpret` defines against it through `define(..., evidence=...)`, the
  one writer. Until now every object was made from words (§2.11).
- *A word's concept over the scalar is a box* (decision 16; Claude's
  default, §5.2 a). Each word heard with the scalar gets a concept of its
  own, minted on recurrence as objects are today
  (`FamiliarTier.promote_identity`). The concept keeps a part bound and a
  whole bound: the least and the greatest of the values the word was heard
  with, the field's `intersection` and `union` taken over the definition's
  views. A value belongs to it when the lower bound is part of the value
  and the value is part of the upper bound; by degrees these are the two
  coverages of `IdentityRows.meet_ratios`, 1 inside the box and falling as
  $v / \mathit{lo}$ below it and $\mathit{hi} / v$ above. Only part, min
  and max are used, and nothing is trained by gradient. On one dimension
  the box is an interval; on several it is a hyperrectangle, by the same
  rule.
- *The bounds are perceptual* (Alec, 2026-11-12): "Although feelings are
  bare numbers, they should still be bounded by perceptual parts and
  wholes, which are knit together at the level of basic percepts (the only
  kind collected by concepts)."
  - As built, they are not. `BasicPercepts.numeric_dimension` reserves one
    basic percept for the whole axis, "independent of values", with a
    placeholder part and no whole. A reading's code is its value on that
    axis and nothing else. Each word's bounds are two numbers, `low` and
    `high`, in a dictionary beside the percepts
    (`ScalarPercepts.state(owner)['boxes']`), and
    `ScalarPercepts.field_memberships` recognizes a value by a loop over
    that dictionary.
  - Alec, 2026-11-12, on Claude's reading of it: "Do in 1.8. Each percept
    is categorized against perceptual parts and wholes, whose codes are
    the values. So the order zero concept is a generalization over basic
    percepts. We should share as much code between the two modalities as
    possible."
  - So: a felt value is categorized against perceptual parts and wholes
    whose codes are values, a part it contains and a whole that contains
    it; its basic percept knits them
    (`BasicPercepts.reserve(parts, wholes)`, as for text); a word's
    concept over the scalar is an order-zero concept that collects basic
    percepts, as every concept does (`add_concept_feature(row, 'bp',
    ...)`), and generalizes over them, the part common to them and the
    whole that covers them being the least and the greatest of the box.
    The dictionary's `low` and `high` go, with `field_memberships` and
    the scalar branch of `stimulate`; a value is recognized by the read
    that serves text. The numeric rows of §3.9 are not changed in this
    round (Claude's default: the XOR proofs rest on a number being the
    evidence of its axis's concept).
  - What follows for §11.1: two words heard over the same interval collect
    the same basic percept and have one code, and the reference says which
    word is said (§12); two words over different intervals have different
    basic percepts and different codes, which a bare value on one shared
    axis does not give them.
- *One word, one concept.* `lexical_form` spells an object only where
  exactly one word names it (§2.11), so two words are two concepts over
  the scalar and never two lanes of one.
- *The definition's trust is counted* (decisions 6 and 18). Each moment
  that holds both the word and the number is one view of the definition;
  its agreement is whether the value fell inside the box as it stood, as
  `ConceptTrust.observe` counts a concept's views; the row's pair grows
  with views as a concept's length does. That is the certainty of the
  mapping from the word to its object, for the definitions that have
  direct evidence. Every other definition stays as §3.7 leaves it.
- *Saying, and restoring.* An object over the scalar is said by its one
  word. Under the whole-path round's omission the number may be omitted
  whole, as a word may be: an omitted number is restored through the
  word's concept, as a value inside its box (its lower bound, the part
  every member contains), and an omitted word is restored from the number
  through the box that holds it.
- *Recognition* (decision 21; Claude's reading, §5.2 d). A value is shown
  with a question, "what is this?", and the supervised answer is its word,
  as a photograph is given its label. "this" points at the present event,
  at order zero, as demonstratives do (Orders §10: "This is Felix, that is
  Felix"); the event is held in MTM as its order-zero identity, the
  candidate that `observe_candidate` opens for the reading. What is asked
  is the class of which that event is an occurrence: an identity of order
  one or more, such as the concept over the scalar that a word names,
  whose box holds the value. No order-zero concept becomes a noun.
  *Deferred (Alec, 2026-11-12: "defer it."):* the addendum of §13.1 makes
  the word's concept over the scalar an order-zero concept, which this
  bullet's requirement of order one or more contradicts, and Codex stopped
  on it. Recognition leaves this round. It is built in the next 1.8 round
  on `compose_extend` (§3.17, "Two directions"), as the case of a value
  present with no word for it, with no rule of its own about the order of
  the answer. Certificate 13 is claimed in this round without its
  recognition part.
- *Comparison.* Two readings and a question, "which is more or less",
  answered in words. The test is part on the two readings; the words of
  the question get their meaning from the answers, since no operator has a
  predefined surface (Alec, 2026-09-20).
- *Which lessons are required* (Claude's default, §5.2 b). Recognition of
  the values that were trained is required (certificate 13). Recognition
  of values that were not trained, comparison, and "zero" are measured and
  reported.
- *The scale* (decision 20). A number $n$ from zero to infinity is felt as
  a presence $v = f(n)$ in $[0, 1]$, with $f(0) = 0$, $f(\infty) = 1$ and
  $f$ increasing. Part, max and min depend only on order, so whatever $f$
  is, they give the same answers: less-than, the least and the greatest
  of two numbers are those of their feelings. That is why the scale can
  sort itself out without changing what the operations compute. A word's
  box is kept on the numbers heard with the word, and its membership is
  read through the current $f$, so the box stays right if $f$ moves.
- *The spacing is logarithmic* (decision 20, §5.2 e). With
  $g = \ln(1 + n)$, the feeling of a number is $f(n) = g / (1 + g)$: zero
  at 0, one only at infinity, and logarithmic in between, so that equal
  ratios give nearly equal steps among the numbers in common use. The
  form is Claude's, for "something like a logarithmic mapping": the
  logarithm alone has no upper end, and $g / (1 + g)$ gives it one. Some
  values: one .409, two .524, three .581, ten .706, a hundred .822, a
  thousand .874, a million .933, a billion .954. Single precision still
  tells $n$ from $2n$ at $10^{100}$, where $n / (1 + n)$ stops at about
  $10^{8}$. The map is fixed: it has no state, nothing to checkpoint, and
  nothing moves when a concept is recognized. It replaces the
  interpolation over recognized concepts that decision 20 first defined.

- *Zero is an absence, not a feeling.* A value of 0 is no presence, and
  nothing matches an empty percept (`meet_ratios` gives no evidence for
  empty content). A word for 0 cannot be a box over the feeling. It names
  an absence, and an absence is inferred against an expectation, never
  perceived (Philosophy, "Absence is inferred, never perceived"): it is
  evidence on a concept's against-lane, where a number was expected and
  none was felt.
- *Where the grounding stops* (decisions 16 and 20). A numeral can stand
  for a million. Under decision 20 a million has a place on the feeling,
  near its top, but the feeling tells it from two million only coarsely:
  the scalar grounds the order of all numbers and the size only of those
  it spaces widely. Every other numeral is known
  through other words, by the successor and by thinking, and the rule of
  2026-10-07 that the math cannot be direct stays as it is: no arithmetic
  acts on a grounded numeral's value.
- *The mapping from the feeling to a word is not linear* (decision 16),
  and nothing here assumes it is: each word's box lies where its examples
  put it, the boxes need not be evenly spaced, and a value between two
  boxes belongs wholly to neither.
- *The bounds of the toy.* Numbers from zero to infinity, felt through
  $f$. Its words name numbers that are felt: "one", "two" and "three"
  (Claude's default, §5.2 c, revised for decision 20, which puts infinity,
  not one, at the top of the feeling). One lesson file; no corpus.
- *Toward several dimensions and images, future work* (decisions 13 and
  14). With images `.where` has two dimensions and a whole or a part is a
  hyperrectangle in the unit hypercube. Claude's reading of how the scalar
  is the smallest case: a percept's code $c$ is the far corner of the box
  $[0, c]$, `Ops.partOf` is the containment of such boxes, and a `.where`
  bracket is already an interval. A value as a percept's presence is the
  "amplitude is presence" proposal that the decision of 2026-10-25 set
  aside for images ("Intensity is a code"); which stands when images come
  is Alec's to say.

### 3.11 The forgetting pass

As the entry of 2026-10-24 proposes
([FutureWork](../FutureWork.md#a-cheaper-forgetting-pass-running-luminosity-on-ltm-alec-2026-10-24)):
luminosity is a column, updated when a row is written or witnessed again
by comparing it only with the rows its content already indexes to, in the
same scope and order, and released when a row is deleted; coverage, where
`forgetCoverage` is on, is a running sum with each row's contribution, so
leaving one out is a subtraction; deducibility is cached at the write. The
pass then reads its columns and sorts. The retention order does not change.

### 3.12 Recall: an entry holds every order, and a question is matched order by order (Alec, 2026-11-12; amends §3.8)

Alec, 2026-11-12, on "what did I eat yesterday": it "should construct a
region of conceptual event space"; "Yesterday will be a region, which is
better specified in a higher-order space (all points in order-0 are
prototypes)"; "Cuing with the order zero field, as a whole, is not
desirable; but it should be content addressable". Then: "LTM0 is stored as
a custom high-order field like a verb modified noun, and a set of zero
order events. These are semantic and episodic memories." And, the note this
section is written from:

"Episodic/semantic are just an approximations of order ramsification in
Russel's sense. With that, a given LTM0 time slot may store concepts at
several orders, and should basically look like a data brick. Queries will
therefore look like inner product against content vectors that are also
represented per-order; there is no need to reduce orders, since in general
it would take an infinite number of lower order vectors to represent the
higher order region. Matches will therefore be per-order. Matches against
space and time may be the intersection of two results; combining those
vectors prematurely will look for conceptual memories that represent "food
I ate yesterday" as events, whereas the questioner most likely was looking
for an answer that yielded the meet (?) of two generalities."

What follows is Claude's writing of that in the code's names. On the
points marked below as Claude's reading or proposal, Alec, 2026-11-12:
"Your recommendations seem fine."

- *The entry is a brick.* A sentence's field already holds every order;
  each concept's order is `owner._row_order`, and the working field is
  laid out by order (`owner._field_caps`, `owner._field_order_slice`).
  `EpisodicMemory.select` keeps only the order-zero concepts, the strongest
  `episodicConcepts` of them. It keeps the strongest of every order
  instead: `episodic_concepts` and `episodic_values` gain an order axis,
  row by order by slot, and `LTM0` is time by order by content. What was
  called episodic is its low orders and what was called semantic its high
  ones. (Claude's reading of "data brick": an order's slice is a content
  vector over that order's concepts, stored as references and signed
  values as the entry is now. The other reading is a dense vector for each
  order in that order's code space.)
- *The idea stays.* The composed vector in `store.slots[row]` is the
  sentence's own point, a new combination and not a known concept (§3.3);
  `LTM0.cued_rows` reads it as built. (Claude's reading. Whether it is the
  brick's top slice is open.)
- *The question is a brick of the same shape.* Its content at an order is
  its own field at that order (`stimulate`, each concept at its
  `_row_order`). Nothing is carried from one order to another, in the
  entry or in the question.
- *A match for each order.* For row $t$ and order $k$, with $q_k$ the
  question's content and $e_{tk}$ the row's, lane with lane and never
  netted, $M_{tk} = \langle q_k, e_{tk} \rangle / \langle q_k, q_k \rangle$:
  one product for the whole window. An order at which the question holds
  nothing has no match and is left out; it is not a zero. (The division by
  the question's side alone is Claude's proposal: a cosine divides by both
  sides, and would lower a row for holding more than the question, which
  is where the answer is.)
- *Time is a result of its own, and so is place.* A time specification is
  matched against the rows' `time`, as a box (§3.10) over their ages. It
  is never matched against content: a stored sentence that spoke of
  "yesterday" holds that concept in its content, which is another thing
  than having been written yesterday.
- *Results are intersected; nothing is summed beforehand.* The rows a
  question leaves are the `intersection` (§3.2) of its results: the orders
  it speaks at, and its time. That is the meet: ordering generalities by
  the events they hold, the events under both are the greatest lower
  bound, and `IdentityRows.meet_ratios` uses the word the same way. Four
  entries and the question "what did I eat yesterday", holding "I" and
  "eat" at order zero and "food" at order one, with the time specification
  "yesterday":

  | entry | order 0 | order 1 | time | meet | one combined vector |
  |---|---|---|---|---|---|
  | I ate rice, one day ago | 1.00 | 1.00 | 1 | 1.00 | 0.75 |
  | I ate bread, nine days ago | 1.00 | 1.00 | 0 | 0.00 | 0.75 |
  | I walked, one day ago | 0.50 | 0.00 | 1 | 0.00 | 0.25 |
  | I said "yesterday", five days ago | 0.50 | 0.00 | 0 | 0.00 | 0.50 |

  In the last column the time has been folded into the content as the
  concept "yesterday" and one product taken: it cannot tell yesterday's
  rice from the bread of nine days ago, and it raises the sentence that
  only spoke of yesterday.
- *Relevance.* Alec: "that intersection determines a Relevance scalar that
  can be used to sort the memories for retrieval into STM". The meet is
  the row's relevance, one number. Rows are sorted by it, the newer first
  among equals as now, and the best `max_records` are brought back by
  `EpisodicMemory.reinstate`, which publishes a recalled entry's concepts,
  each at its order, as readings in the field the stream is examining.
  (That this field is what "into STM" means is Claude's reading;
  `EntryRead.recall_identities` is the other path, into MTM's chains with
  the head read by STM.)
- *What this replaces.* `EpisodicMemory.query` flattens every concept of
  the cue, of whatever order, into one request and takes one coverage
  (`meet_ratios`) for each row in a loop: one combined vector. The
  extension table of this section's earlier writing, which carried a
  higher-order term down to its members, is withdrawn.
- *An entry holds, at a higher order, what its sentence entailed when it
  was written.* A definition learned afterwards does not reach an older
  entry until the entry is written again. (Claude's note of a consequence;
  the upper slices of a closing sentence's field were read in the code's
  layout, not measured on a run.)
- *Only what is conceived is remembered.* Alec, 2026-11-12: "storing
  concept references allows memory only of what we have known, we have to
  conceive of it to remember it; sounds reasonable." `EpisodicMemory.select`
  keeps a reading only where the field has a concept for it, and
  `EpisodicMemory.reinstate` passes over a reference whose concept is no
  longer in the inventory (it is counted, `ids_skipped`). What is new in a
  sentence is kept as the idea's vector, which holds no references; it is
  found through the brick only by the concepts that were known when it was
  written.
- *Time is called time.* Alec: "Stamp is a bad name. Time seems
  reasonable." The time an `LTM0` entry was written is named `time` in code
  and documents, so that it is not taken for the `.when` of a percept;
  `.when` keeps naming a percept's time, and a chunk's in the open sentence
  (the chunk is Claude's reading). It still gives forgetting its age and
  two stored situations their order. No lesson of this round teaches a time
  word (Claude's default).
- *Still by content.* Nothing is addressed by time or place (§3.8). The row
  window (`ltmWindow`) bounds the read.
- *Relations stay, and nothing queries them.* Alec, 2026-11-12: "Don't
  remove the relation rows, they are exactly what structure thought and
  provide for conceptual taxonomy. But we don't build a query for them,
  they are atemporal relations that structure the data of ltm0 and don't
  really have anything to query over. They continue to provide input to
  relations over concepts (which can be seen as analogous to Permanent
  Term Memory, in a promotion order that includes STM, MTM, LTM, and PTM)
  which form the knowledge taxonomy."
  - The recall of this section ranges over `LTM0` rows only, and no read
    by content or by time is built over relation rows.
  - A relation row holds no time, as built since the `percept` column went
    (§3.8). Where an order in time matters it is read from the two `LTM0`
    rows the relation ties (`relation_evidence`, for `implies`), and
    forgetting gives a relation row the age of the sentence row it came
    from (`Forgetting.history_age`).
  - The relation rows and the taxonomy have one owner as built: the
    concept allocator's `parts` is the `LTM1` table (`Layers.py`, "LTM1 and
    the symbol taxonomy, one owner"). A concept's definition edges in the
    concept store are written from its part and whole references
    (`_populate_concept_weights`) and when a recurring candidate of MTM is
    promoted to a concept (`FamiliarTier`, `pool.promote`). That store of
    relations over concepts is what the quotation likens to a permanent
    memory; nothing is renamed for it.
  - The reads by reference that thought already makes stay as built:
    `ThoughtLTMCapability.relation_evidence` and `LTM1.rows_referencing`
    follow a row's two references, as uses of the one read of §3.14.
    `LTM0.cued_rows` no longer appends the relation rows to the candidates
    of its read of ideas (§3.14).

Certificates of this rule, beside §4's:

15. `episodic_concepts` has an order axis. An entry whose sentence held
    "rice" at order zero and "food" at order one matches a question that
    holds "food" at order one, at that order, and no reading is carried to
    or from order zero.
16. The match is taken lane with lane; an order at which the question holds
    nothing is left out of the intersection; the read is one product over
    the window, with no loop over rows.
17. No `LTM0` entry's time is named `.when` or a stamp in code or
    documents.
18. Of two entries with the same content at order zero, one holding the
    question's concept at order one and one not, the intersection leaves
    the first only.

Built in this round: the entry of every order, the match for each order,
the intersection, the relevance and the name `time`. The time result
itself (the box over the rows' ages) is not built yet: nothing teaches a
time word in this round, so it would have no caller (§6). The
intersection is built so that a further result over rows joins it
unchanged. Its certificates then: a box over an entry's age leaves exactly
the entries whose age it holds; a stored sentence whose content holds a
time word is not matched by a time specification on that account; of two
entries with the same content, one written inside a question's time and
one outside it, the intersection leaves the first only.

### 3.13 The determiner lowers and selects (Alec, 2026-11-12; decided in direction, details open)

The complete account is §3.17, which stands where this section differs.

Alec, 2026-11-12: "We might also have an outdated determiner rule. If the
field of a concept is its extension, the parts that it entails, and its
intension, the types that it entails, then the determiner (the, a, every,
etc) applied to that will be a discrete selection operation over that
field."

Decided, Alec, 2026-11-12, after §3.0's rule that the formula selects the
extension: "Great, so determiners do lowering/selection on their
arguments, to either one or multiple vectors." A determiner is the
operator of the formula that lowers its argument, a kind, to its
extension and selects there: one member, or several. What follows is the
state of the code and Claude's readings, which wait for Alec; it is built
in the next 1.8 round, with §3.16.

- *The result is the selected field.* Alec, 2026-11-12: "In a linguistic
  context operators select the best match, they compose and generate
  intentions. We we need those operators to generate an extension from an
  intension, which is easy for known concepts, and then we can perform a
  selection. So the result of the operation is the selected field, not
  the sentence idea or its constituent word ideas." So an operator has two
  uses. Composing and generating, it takes and gives intensions, as built.
  In its field use, it takes the extensions of its operands, which for a known
  concept is a lookup, and selects among them; what the walk of a sentence
  returns is then one selected field, where `OutputOccurrences.generated`
  now returns the leaves' fields side by side (§3.0). `compound` already
  has this second use (§3.5, "a field operation inside"). Claude's
  proposal for the others: narrowing one operand by another (a noun by a
  modifier, a subject by a predicate) is `intersection`; joining kinds is
  `union`; negation is `not` or `non`; the determiner is the discrete
  choice; and each operator's field use is declared with it in the grammar
  file, not in a table in code.
- *One or several.* An operator of the file takes two operands and gives
  one point today. Several vectors out is new; Claude's proposal is that
  they travel as a field over the selected members, which is what the
  field operators of §3.1 already take.
- *As built.* `LowerLayer` ("A determiner selects a member; its marker
  contributes no content") passes the noun's content through its kernel
  (`compose_blocks`, `meaning_head_role = 1`) and carries a
  `determiner_mode`, which the grammar file sets with `referent`: `mint`
  for `lower`, `bind` for `definite`, `kind` for `universal`. The
  selection itself is rule code elsewhere, over earlier mentions and never
  over the noun's field: `ThoughtLTMCapability.bind_reference` offers the
  situation's frames and the rows this episode already holds, and
  `determiner_mode` is consulted in `ClauseJournal`, `ClauseScope`,
  `ChunkJournal`, `ReferenceContext`, `ReferenceParticipants`,
  `Generative`, `Models` and `Language`.
- *The field in memory.* A concept's extension in memory is the result
  of §3.12 at its order, the rows that hold it; its intension is what
  those rows hold at the orders above. (Claude's reading.)
- *The four selections.* Alec, 2026-11-12: ""the" is selecting a known
  identity from MTM, "a" is selecting a single subconcept, "some" selects
  some subconcepts, "every" selects the entire extension." This replaces
  Claude's earlier reading, in which "the" read long-term memory and "a"
  wrote a new member. Two readings of Claude's that it opens, for Alec:
  what "a" selects is entered in MTM as an identity, since a later "the"
  finds it there; and which single subconcept "a" takes, and which
  several "some" takes, is the strongest in the sentence's own field
  after its other operators have narrowed it.
  Alec on the second: "("A cat" should select a generic "cat",
  presumably; perhaps it can be a centroid?)" So "a" gives one generic
  member, the centroid of the kind's subconcepts (`FieldProjection.key`
  is the evidence-weighted centroid of a field, as built), and not the
  strongest one. Which several "some" takes is still Claude's reading.
- *Not lowering, in general.* Alec, 2026-11-12, accepting the proposal for
  the other operators: "it really is an operation on the field of active
  concepts and not necessarily lowering the order of a concept." So an
  operator's second use is its *field use*: an operation on the field of
  active concepts, at whatever orders they are. Only the determiner goes
  from a kind to its members. Where this section says "lowering" of an
  operator other than the determiner, read "its field use".
- *Extend.* Alec, 2026-11-12: "compose, think, generate, extend", and,
  correcting Claude's reading that it would be a fourth section of the
  grammar file: "extend should operate on the existing conceptual
  activations, or a top-k select thereof. if we have the STM tree, we can
  just iterate over its existing compose(), calling extend() on each
  operator to process the set of conceptual activations."
  - *Extend* is the name of an operator's field use. It is a method of the
    operator, beside its `compose` and `generate`; no section and no rule
    of the grammar file is added for it, and nothing is parsed again.
  - It is called over the tree the sentence's `compose` already left in
    STM, once at each operator of that tree.
  - Its operand is the set of conceptual activations that exists, the
    field the sentence's words left, or the strongest $k$ of them. Each
    operator narrows or selects in that set; no extension is built up
    from the leaves.
  - What the calls leave is the selected field of this section's first
    bullet, and the row written at the close takes its references from
    it.
  - Claude's readings, for Alec: an operator tells its operands'
    activations apart by their places (`.where`); an idea met without its
    sentence gets its tree from the generate walk first (§3.0); and $k$
    is a parameter of the file.
- *Beside §3.5.* `compound` narrows within the noun's extension and keeps
  its order; the determiner then takes a member and lowers it. Both read
  the same extension.

### 3.14 Long-term memory is two tables; the rest is a cache or is named (Alec, 2026-11-12)

Alec, 2026-11-12: "If possible, reduce extra LTM fields; relation_evidence,
rows_referencing. LTM0.cued_rows will become expensive if they are
maintained over all entries. We mostly want an ltm0 that has per-order
entries so that we can do matrix multiply to query them, and an LTM1 that
stores relation pairs over those. Information beyond that can probably be
treated as a cache, and be marked as such. I don't think ltm1 needs to
store the order of the LTM0s that it relates, but please correct me if
that is necessary for relations."

- *The two tables.* An `LTM0` row is its per-order entry
  (`episodic_concepts`, `episodic_values`, §3.12), its `time`, its idea
  (`slots`), its evidence (`truth_plus`, `truth_minus`) and the address by
  which a relation names it (`address_keys`). An `LTM1` row is an ordered
  pair (`refs`, `ref_kinds`) and its evidence (`truth_plus`,
  `truth_minus`).
- *No order is stored in a relation.* Alec is right. A relation needs its
  pair to be ordered, left and right (part and whole, antecedent and
  consequent), and nothing else about order:
  - the order in time of two related `LTM0` rows is read from those rows'
    own `time` when a conditional needs it, as `relation_evidence` does
    now for `implies`;
  - the conceptual order of an end is the end's own (a concept's from the
    allocator, an `LTM0` row's from its entry).
  `LTM1` has two columns about order today, and both are about when the
  relation rows themselves were written. `store_order` has one reader,
  the ordering of relation candidates in `LTM0.cued_rows`; it goes with
  that reader. `relation_order` serves as a dependency order, so that a
  pair which names another pair is refreshed after it
  (`refresh_effective`); it can be rebuilt from the pairs and is a cache.
- *Reads that go through every row are replaced.* As built:
  - `LTM0.cued_rows` reads a record for each candidate, up to `ltmWindow`
    of them, in a loop, and copies three role vectors for each. It is
    applied only to the rows the product of §3.12 returned
    (`max_records`), never to the window.
  - `ThoughtLTMCapability.relation_evidence` walks every relation row in a
    loop on each call. It reads the rows that hold its given ends instead:
    `LTM1.find` when both are given, the rows holding the one given end
    otherwise.
  - `LTM1._index` rebuilds its four dictionaries (`_pairs`, `_parts`,
    `_wholes`, `_references`) over every relation row after any write,
    because `LTM1.add` writes `refs` in place and the index is keyed on
    that tensor's version. `add` and the forgetting pass keep them up to
    date instead.
  - `EpisodicMemory`'s index from concept to rows (`_episodic_index`) goes;
    the product replaces it.
  One read over `LTM1` remains, the rows that hold a given reference;
  `rows_referencing`, `relation_evidence`, `find` and the part and whole
  lists are uses of it.
- *What is marked a cache.* A cache can be cleared and rebuilt from the two
  tables with no read changing. Each store names its caches in one
  declared list, as `LTM1.columns` names its columns, and a cache is not
  saved in a checkpoint. From Claude's reading of the code, for Codex to
  confirm reader by reader:

  | store | field | what it is rebuilt from |
  |---|---|---|
  | `LTM1` | `_pairs`, `_parts`, `_wholes`, `_references` | `refs`, `ref_kinds` |
  | `LTM1` | `relation_order` | the pairs that name other pairs |
  | `LTM1` | `effective_plus`, `effective_minus` | the evidence and the trust gates (`refresh_effective`) |
  | `LTM0` | `_index_occurrences` | `address_keys` |
  | `LTM0` | `_cue_roles`, `_cue_role_mask`, `_cue_available`, `_observation_ids` | scratch, already not saved |
  | `LTM0` | `recurrence_codes`, `recurrence_weights` | the rows assigned to each reading (`MemoryConsolidation`) |

- *What is neither.* These are not the two tables and cannot be rebuilt
  from them, so they are not caches. They stay in this round and are
  listed in one place in the store, each with its one use; whether each
  remains is a later decision (§6):

  | store | field | its use |
  |---|---|---|
  | both | `access_count`, `surprise` | forgetting |
  | `LTM1` | `sentence_reference` | forgetting gives a relation the age of its sentence row |
  | `LTM1` | `operator_reference`, `operation_seat` | the trust of an operator (§3.7) |
  | `LTM0` | `_semantic_rows` | a row's roles, references, scope and bindings |
  | `LTM0` | `document_keys`, `sentence_content_keys`, `sentence_index`, `source_documents`, `source_authorities`, `semantic_fingerprint`, `metadata_required`, `_texts`, `_document_sources`, `_cover_symbols` | where a row came from, and its text |
  | `LTM0` | `stream` | which input stream a row belongs to |
  | `LTM0` | `recurrence_readings`, `recurrence_symbols`, `recurrence_documents`, `_expectation_rows` | recurrence and expectation |

Certificates of this rule, beside §4's:

19. Clearing every declared cache of `LTM0` and of `LTM1` and rebuilding
    leaves every read of a fixture store unchanged, and no declared cache
    is in a saved checkpoint.
20. On a store filled to `ltmWindow`, one recall reads records for
    `max_records` rows, not for the window; `relation_evidence` with both
    ends given reads one relation row; a write to `LTM1` followed by a
    read rebuilds no index. Each is asserted on a count, not on time.
21. `LTM1.store_order` and `_episodic_index` are in no source file.

### 3.15 The fields that are neither table nor cache: audit (Claude, 2026-11-12; for Alec's decisions)

Alec, 2026-11-12: "I would like to remove the ones that are unnecessary;
the model should have a small footprint, doubly so for the LTM memories."
Nothing below is decided. Sizes are measured on a store of width 370 with
28 streams: an `LTM0` row is about 1,900 bytes, of which the idea vector
is 1,480 and the per-order entry 240.

| group | field | bytes | real use found | if deleted |
|---|---|---|---|---|
| forgetting | `forget_cutoff` | scalar | written, never read | nothing |
| forgetting | `LTM0.access_count` | 8 a row | not read by forgetting; a threshold of uses in `ReferenceClose.exclusion_pairs`; a re-read counts as a new witness in `IndependentComponents.commit` | both change |
| forgetting | `LTM1.access_count` | 8 a relation | the luminosity term of `forget_parts` | relations ranked by trust, surprise and age only |
| forgetting | `surprise`, both stores | 4 | the utility term of the retention value | the term is constant |
| the sentence a relation came from | `LTM1.sentence_reference` | 8 a relation | a relation's only time: its age in `forget_parts`, ages passed to `LTM0` rows, event order of a conditional between relations, the spatial reading of a part between relations | every relation reads time zero |
| operator trust | `LTM1.operator_reference` | 8 a relation | the only stored copy of the verb of a subject, verb and object relation; also scales evidence by the verb's trust | the verb of such a fact is lost |
| operator trust | `LTM1.operation_seat` | 8 a relation | which grammar operation composed the row; scales evidence and applies the gates of §3.7 | the gates no longer act on evidence |
| operator trust | `gates`, `operation_catalog` | a few per operation | gate history; a checkpoint guard | not a footprint item |
| semantic record | `_semantic_rows` | a small dictionary a row | `bindings` and `scope` only, no role vectors; a row's identity, kind, polarity, order and open references are read from `bindings` | a row is a bare idea with no identity or kind |
| semantic record | `scope` inside it | | no production writer found (`set_scope` is called by tests only) | nothing found |
| semantic record | `semantic_fingerprint` | 32 a row | one reader: a checkpoint load checks the dictionary against it | a mismatched dictionary loads unnoticed |
| semantic record | `metadata_required` | 1 a row | detects a restore that lacks the dictionary | such a restore reads rows as bare ideas |
| provenance | `address_keys` | 8 a row | how a relation names an `LTM0` row | relations cannot name rows |
| provenance | `document_keys` | 32 a row | a document's authority (fact or observation, user or provisioned), cover and direct-kind classification, the count of documents that gates consolidation, exclusion of the row being re-read | those stop |
| provenance | `sentence_content_keys` | 32 a row | a self-check for a collision of the 64-bit address; otherwise tests only | the content part of that check |
| provenance | `sentence_index` | 8 a row | that check, and exclusion of the same position on a re-read | a re-read may refer to the row it replaces |
| provenance | `source_documents`, `source_authorities` | 33 a row allocated, used one per document | a tensor copy of `_document_sources` for a restore without the sidecar | the load path must read the sidecar |
| provenance | `_texts` | variable | text of supplied truths: contradiction messages, the match on resubmission, forgetting's wording stage | messages and the match lose the text |
| provenance | covers (`_cover_symbols`, `cover_rng_state`) | fixed | document segments: scoped reads, MTM aging, the document count | no segments |
| stream | `stream` | 1 a row for each stream | separates concurrent batch streams in history, covers, MTM aging and reference candidates | streams see each other's memory; answers about the past break |
| recurrence | `recurrence_documents`, `recurrence_used` | 9 a reading | rebuilt on every call | caches |
| recurrence | `recurrence_codes` | 4 times the width a reading | the match of a row to a reading; kept only for promoted readings with no rows left | to be checked: rebuildable from member rows and the promoted concept |
| recurrence | `recurrence_readings`, `recurrence_symbols`, `_recurrence_identities` | 8 a row; 8 a reading | which reading a row joined; which concept a reading became | consolidation of a reading seen in three documents stops |
| expectation | estimate rows and `_expectation_rows` | one whole `LTM0` row for each predicted sentence | every read excludes them; `expectation_pair` has no caller outside tests; the predictor trains on the live comparison | the retained record of what was expected; one row freed for each predicted sentence |

Found on the way, for Codex to confirm: `refresh_effective` applies the
seat's trust twice; its `named` computation cannot change a result;
`TrustGates.measurements` has a branch that does nothing;
`LTM1._reserved_handle` reads `role_refs` from the dictionary, which never
holds it; `EntryRead.cued_rows` takes a `stream` it never uses.

Where the footprint is: these groups are about a tenth of a row. The idea
vector is nearly four fifths of it; the references of the per-order entry
are stored as 64-bit numbers (160 bytes a row, 80 as 32-bit); and an
estimate row is a whole row.

### 3.16 Long-term memory reduced (Alec, 2026-11-12; decisions on §3.15; Claude's readings wait for him)

Alec, 2026-11-12, group by group:

1. "Forgetting statistics should just be reduced to a single "activity"
   value, which can be incremented when accessed, set by surprise, and
   periodically reduced unless its value is 1 (in which case it becomes a
   permanent memory)."
2. "The sentence a relation comes from should be removed; we know and do
   not necessarily remember where we first learned."
3. "Trust is a good column to keep for relations only." "Trust is one for
   the epistemic slots, since they are directly observed, and can be
   inferred for semantic vectors by their length (semantic vectors must
   store the entire vector in a different store, since the sentence does
   not exist as a single concept)."
4. "Semantic record can be deleted. Bindings and/or should be stored in
   the MTM, which can be recomputed ( ? )."
5. "A semantic vector comes from the associated episodic concepts, which
   may or may not be enough to reconstitute the sentence verbatim.
   Delete."
6. On the stream: "what if we regarded LTM streams as unique to batches,
   while weights are shared ... the downside of deleting this field would
   be the inability to keep streams separate ... the main deliverable is a
   mind; memories could then be learned as a single world-history pass."
7. "Recurrence and expectation can be deleted."

And: "Probably ltm1 and ltm0 have to be stored as separate tensors, but
both ought to be able to be compact representations which allow efficient
matmul retrieval and minimal storage."

What that leaves (Claude's writing of it):

- *`LTM0`, one row for a sentence.* Its per-order entry (references and
  signed values, §3.12), its `time` and its `activity`. No evidence pair:
  a slot of the entry was observed and its trust is one.
- *The ideas, a tensor of their own.* The composed vector of each row,
  whole, because a sentence is not one concept and cannot be a reference.
  Its trust is its length.
- *`LTM1`, one row for a relation.* The ordered pair and its kinds, its
  trust (the evidence pair) and its `activity`. No sentence reference and
  no time.
- *Activity* replaces `access_count`, `surprise`, the `forget_*` scalars
  and the retention value of §3.11 with its maintained columns
  (`RetentionColumns`). A write sets it from the sentence's surprise; an
  access raises it; a pass lowers every value that is not one and drops
  the lowest when the store is over its bound; a value of one is never
  lowered or dropped. (The step, the rate and the value written when the
  surprise is unknown are parameters; Claude's reading.)
- *Gone:* `LTM1.sentence_reference`; `_semantic_rows`, `metadata_required`
  and `semantic_fingerprint`; `document_keys`, `sentence_content_keys`,
  `sentence_index`, `source_documents`, `source_authorities`, `_texts`,
  `_document_sources`; `stream`; the recurrence table and
  `MemoryConsolidation`; estimate rows, `_expectation_rows` and
  `expectation_pair`; the rows' `truth_plus` and `truth_minus` in `LTM0`.

Amended the same day (Alec): "LTM0 is both sentences and ideas, you can
store them both in a row. We probably want to store the ideas first, they
are the semantic memory that can be used to re-present a field, I think
that makes them contiguous for matmul? The "sentence rows" are the
episodic memory, and since they are indices to several activated concepts,
we need not store their order (that is stored on the concepts)."

- *One row, both memories.* The idea and the entry are columns of the
  same `LTM0` row; "a tensor of their own" above means only that each
  column is its own array under one row index, as `slots` and
  `episodic_concepts` are now. That is what makes the ideas contiguous for
  a product: an array of rows by width holding ideas only. One array
  mixing ideas with references would be neither contiguous for the ideas
  nor of one number type.
- *The entry has no order axis.* It is a flat list of references to the
  concepts the sentence activated. A concept's order is the concept's own
  (`owner._row_order`), so the match for each order of §3.12 groups a
  row's references by their concepts' orders when it is taken; the array
  of every concept's order is a cache. This withdraws the order axis of
  §13 item 3a, which Codex has begun (`episodic_orders`).
- *For Alec, two points this opens* (Claude's proposals): with a slot's
  trust at one there is no value to store, so the lane, for or against,
  is the sign of the reference and `episodic_values` goes; and with one
  list for all orders, it is filled by the strongest reading of each order
  in turn, so that no order crowds another out of `episodicConcepts`
  slots.

What follows from it, each for Alec to confirm or correct:

a. *The verb of a relation.* `LTM1.operator_reference` is the only stored
   copy of the verb of a subject, verb and object relation. It stays as a
   third reference of the row, named for what it is. (Claude's
   recommendation.)
b. *The gates.* `LTM1.operation_seat` is what the per-operator gates of
   §3.7 act through. It stays unless the gates go. (Claude's
   recommendation: stays.)
c. *How a relation names a sentence row.* Today by an address that is a
   hash of the document, the position and the content, all of which go.
   Claude's proposal: by the row's `time`, which is one tick for a
   sentence. That needs one stream (f).
d. *Supplied truths lose their authority.* With `document_keys` and the
   text gone, a row no longer says whether it was supplied by the user,
   provisioned or read; `clear_origin`, `withdraw_origin`, the match on
   resubmission, the fact and observation kinds and the source text in
   contradiction messages go with it.
e. *The document covers go*, since a cover is recognized by its document
   and segments are built from the stream: no reads scoped to a document,
   no aging of MTM by document, and D1's operand-reference scoping has
   nothing to scope by.
f. *One stream writes memory.* A batch reads several documents at once
   and `stream` is what keeps their rows apart. Without it, memory is
   written by a pass of one stream; a batch of several either writes no
   long-term memory or mixes its documents at one time.
g. *Bindings.* What is pending (an open question's slots, a forward
   reference, a name awaiting its referent) is held in MTM while it is
   pending. What a row denotes is read from its entry. Which thought
   produced a row is not kept.
h. *Kinds.* With covers, estimates, authority and the dictionary gone, a
   row has no stored kind; a question still open is one that MTM holds.
i. *Reading a sentence again.* With no address, a sentence met again is
   recognized by its entry (a match of one, §3.12) and raises that row's
   activity; no second row is written. (Claude's proposal.)
j. *Promotion.* With the recurrence table gone, a concept is promoted
   from MTM's candidates only; in long-term memory what lasts is what
   reaches an activity of one.
k. *Certificate 12 and §3.11* (the forgetting pass on maintained columns)
   are replaced by the pass on activity.

Answered (Alec, 2026-11-12), point by point; this is built "in another
1.8 round":

a. *No operator on a relation.* Alec: "I thought we removed the relations
   operators, becuase it was always conditional, part/whole, or if/then
   depending on the order of its pair?" That is the decision of record: a
   relation is one pair and its kind is read from its two ends
   (`LTM1.readings`). The code still has a third kind, `'operator'`
   (`ClauseRow.predicate_relation`, `ClauseJournal`), written when a verb
   takes a completed relation as its argument, with the verb kept in
   `LTM1.operator_reference`. It goes; such a sentence is an idea row, and
   its verb is in the idea and among the entry's concepts. (That the
   sentence is then an idea row is Claude's reading.)
b. *The gate is the operator's amplitude.* Alec: "Operator gates should be
   the amplitude of the the operator, stored as a concept. I'm concerned
   that you didn't know this becuase we previously specified it." Decision
   7 already says it ("I think we had trust as the gate previously"), and
   the code holds an operator's trust as its concept's radius
   (`OperationSeats`, `ConceptTrust.positive_length`). §3.7 as Claude
   wrote it added a second gate on top, one boolean for each operator with
   `closedTrust` and a threshold (`TrustGates`). That second gate goes:
   the amplitude of the operator's concept is the gate. Claude's reading
   of what follows: the amplitude acts when the operator is applied, on
   the trust the row is written with, so no relation row stores which
   operation composed it and `LTM1.operation_seat` goes with
   `refresh_effective`'s scaling by seat.
c. *Rows are named by their time.* "Good".
d. *Authority of supplied truths goes.* "Fine, we can look in memory to
   see which book we opened prior to learning, or something like that."
e. *No cover rows.* "They won't stand out, but we can at least store the
   concept of the book (by presenting a cover or directly creating the
   concept and storing a reference to it; perhaps we can force the latter
   to happen without reaching into the model architecture)." Claude's
   reading: the book is an ordinary concept, made by the corpus presenting
   it as input; no row kind, generator or segment exists for it.
f. *One stream writes memory.* "Good".
g. *Bindings.* Not objected to: what is pending is MTM's; what a row
   denotes is its entry; which thought produced a row is not kept.
h. *A sentence read again is a new row.* "too much lookup. But it will be
   forgotten." No recognition of repeats.
i. *Promotion.* "Promotion can happen for unpromoted memories with high
   activity. When promoted, we can seal their activity to 1 to keep them
   and not consider them for subsequent promotion." A row whose activity
   passes a bound is promoted to a concept, its activity is set to one,
   and it is not considered again.
j. *Forgetting is the pass on activity.* "Yes, we make it simpler."

The two proposals of the amendment above, answered (Alec, 2026-11-12):
"The lane is not stored, but the trust value is signed. Strongest
activation is fine." So each reference of the entry keeps a signed trust
value, as `episodic_values` holds now, and its sign is the lane; no lane
is stored apart from it. The one list is filled by the strongest
activations, whatever their orders. (One signed value cannot hold a
concept that a sentence activated both for and against; as built, the
slot takes the difference. Claude's note.)

The text of this round for Codex is written when §14 has returned and
been reviewed, on the source that return leaves.

### 3.17 Extend, and the field of understanding (Alec, 2026-11-12; the complete account; built in the next 1.8 round)

This section gathers what §3.0, §3.13 and §3.16 record in the order it was
said. Where it differs from them, this section stands. Each statement is
Alec's unless it is marked as Claude's reading or as open.

#### Terms

- "For kinds, a token implies a type. For mereology, a whole implies a
  part."
- *The sentence-concept* is what a sentence's words compose to. "The
  sentence words can be transformed into a single vector in conceptual
  space, which does not (necessarily) exist as a single concept in
  conceptual space."
- *The idea* "is the intension of the sentence-concept": that vector.
- *The field* "is the extension of the sentence-concept": a set of known
  concepts, each with its trust on the two lanes.
- *The field of understanding* is the field a sentence is left with when
  every operator of its tree has been extended. "It is non-conceptual
  understanding in that not all order-zero concepts are necessarily
  symbolized (although in the current architecture all of them are)": a
  member of the field need not have a word.
- An operator has four uses: "compose, think, generate, extend". In the
  first three it takes and gives intensions: "In a linguistic context
  operators select the best match, they compose and generate intentions."
  In the fourth it works on a set of active concepts and gives a set: "the
  result of the operation is the selected field, not the sentence idea or
  its constituent word ideas."

#### The operation

"So *every* operator needs an extend(): we start with the set of concepts
in the extension of the noun phrase, and then extend() it until we have a
new set, which is the field of understanding."

1. *It is a method.* `extend` is a method of every operator, beside
   `compose` and `generate`. No section and no rule is added to the
   grammar file, and nothing is parsed again.
2. *It starts from what is active.* "extend should operate on the existing
   conceptual activations, or a top-k select thereof." The starting set is
   the extension of the noun phrase: the concepts its words activated,
   with their trust. The extension of a known concept is stored, "which is
   easy for known concepts"; nothing is built up by matching a vector
   against every concept's code. (That the strongest $k$ is a parameter of
   the configuration file is Claude's reading.)
3. *It follows the tree that compose left.* "if we have the STM tree, we
   can just iterate over its existing compose(), calling extend() on each
   operator to process the set of conceptual activations." One call for
   each operator of the tree, in the order compose applied them.
4. *Each call takes a set and gives a set.* An operator of two operands
   takes the sets of both. It tells its operands' activations apart by
   their places in the sentence (`.where`). (Claude's reading.)
5. *What the last call leaves is the field of understanding.*
6. *It is an operation on the field, not a lowering.* "it really is an
   operation on the field of active concepts and not necessarily lowering
   the order of a concept." Only a determiner goes from a kind to its
   members.
7. *An idea met without its sentence*, one recalled from `LTM0` or one a
   thought concluded, has no tree in STM. "if we walk the idea to the
   sentence (a WFF), the wff can select the extension precisely": its
   generate walk gives the tree, and the same calls follow. (That the walk
   comes first and extend second is Claude's reading.)

#### Two directions (raised by Alec, 2026-11-12; not settled)

Alec: "We might want to to define compose_extend() and generate_extend()
methods on all operators. We have already agreed that generate_extend is
necessary to stimulate the field, perhaps a compose_extend() could help
use describe what we are seeing when there are no words present."

Claude's reading, which waits for him. An operator would have four
methods, two directions on each side of a sentence-concept:

| | toward the whole | toward the parts |
|---|---|---|
| on intensions | `compose`: words to the idea | `generate`: the idea to words |
| on extensions | `compose_extend`: active concepts to the field of what is seen | `generate_extend`: an idea's field, along its walk |

- `generate_extend` is point 7 above: an idea met without its sentence is
  walked by `generate`, and each operator of the walk gives its operands'
  part of the field.
- `compose_extend` is points 2 to 5 when a sentence is read, on the tree
  `compose` leaves. With no words present, it works on what perception
  activated alone, and the tree it builds is what `generate` then says: a
  description of what is seen.
- Recognition (§3.10, "what is this?") is then that case, a value present
  with no word for it, and needs no rule of its own about the order of
  the answer.

#### What each operator's extend does

Every rule of the grammar file's `<compose>` section is listed. "Alec"
marks what he stated or accepted; "proposal" is Claude's and waits for
him; "open" has no answer yet.

| operator | composing, as built | extend | whose |
|---|---|---|---|
| `intersection` | a field operator already | the `intersection` of its operands' sets (§3.2) | Alec |
| `union` | a field operator already | their `union` | Alec |
| `not` | swaps the two lanes | swaps the two lanes of the set | Alec |
| `non` | withholds both lanes | withholds both lanes of the set | Alec |
| `compound`, `compound_unary` | selects the head noun's cases by its modifier | narrowing: `intersection` of the noun's set with the modifier's, as it does inside now (§3.5) | Alec |
| `adverb` | a modifier applied to a verb | narrowing: `intersection` | Alec, by the rule for narrowing |
| `verb` | subtypes a verb by its object | narrowing: `intersection` | Alec, by the rule for narrowing |
| `lower`, `lower_unary` ("a") | passes the noun through; mode `mint` | one generic member, the centroid of the kind's subconcepts (`FieldProjection.key`) | Alec ("presumably; perhaps it can be a centroid?") |
| `definite`, `definite_unary` ("the") | mode `bind` | "selecting a known identity from MTM" | Alec |
| `universal` ("every") | mode `kind` | "selects the entire extension" | Alec |
| ("some") | no rule in the file | "selects some subconcepts"; which, the strongest of the narrowed set, is a proposal | Alec; the rule is to be added |
| `generic` | a nominal type marker | the kind itself, nothing selected | proposal |
| `disjunction` | a probabilistic sum of activations | joining: `union` | Alec, by the rule for joining |
| `conjunction` | binds forms, or meets two evidence pairs | `intersection` where it meets evidence; where it joins two kinds ("cats and dogs"), `union` | open |
| `preposition`, `preposition_unary` | packages a phrase headed by its marker | its object's set, unchanged | open, with the kernel's head (§11.3 D9) |
| `surface`, `surface_unary`, `sign` | attach a marker or keep a word's carrier | the content operand's set, unchanged | proposal |
| `interpret` | takes a word to its object | the object's set in place of the word's | proposal |
| `bind` | reads a chosen entry of the field | that entry's set | proposal |
| `lift`, `transpose` | structural rewrites | the set unchanged | proposal |
| `ask` | marks a question | the set unchanged; the open operand adds nothing to select by | proposal |
| `part`, `whole`, `equal`, `implies` | state a relation between two operands | open: what a relation sentence leaves as its field | open |
| `gloss`, `divide`, `descend` | examination's actions on attention's places | open | open |
| `sum`, `product` | arithmetic on two concepts | not in this round (§6) | |

What "a" selects is entered in MTM as an identity, so that a later "the"
finds it (Claude's reading).

#### What the field of understanding is used for

- *The row.* An `LTM0` row holds both sides of one sentence-concept: its
  idea, the intension, and references to the strongest members of its
  field of understanding with their signed trust, the extension (§3.16).
  "A semantic vector comes from the associated episodic concepts, which
  may or may not be enough to reconstitute the sentence verbatim."
- *Recall.* A question's field of understanding is matched against the
  rows' references, order by order (§3.12).
- *Reference.* What "the" selected is in the field, so resolving a
  reference is part of extend and not a search made beside it.
- *Saying.* A member that has a word is said by it (§12); a member without
  one cannot be said, which is the sense of "non-conceptual".

#### Extend as the mechanism of recall (Alec, 2026-11-12: "I think it gives mechanism there"; Claude's writing of it, confirmed by him: "Yes, that is it"; amends §3.12)

- *The question's side.* §3.12 took a question's content from a match of
  its vector against every concept's code. It is the question's field of
  understanding instead, left by `compose_extend` over the question's own
  tree. This is the "formula or region description" of §3.12's first
  quotation, evaluated.
- *The rows as the set.* A term's result over the rows is still one
  product, at that concept's order (§3.12). How results combine is no
  longer a fixed intersection of every order: each operator of the
  question's tree combines its operands' results over the rows as its
  extend says, narrowing by `intersection`, joining by `union`, `not` and
  `non` on the lanes, a determiner selecting one row, some or all. The
  operand that `ask` leaves open selects nothing. A time word, once one is
  taught, is one more narrowing, by its box over the rows' `time`.
- *Reading the answer.* A recalled row's idea is walked by `generate`, and
  `generate_extend` gives each operand's part of its field; the part at
  the question's open operand is the answer. This replaces the match by
  role of `LTM0.cued_rows` as the way an answer is read.
- *Two steps, because the row is lossy.* A row keeps references to the
  strongest members of its field only, so the products find candidates,
  and `generate_extend` on the few that are recalled gives their whole
  fields.

#### Extend as System 1's teacher (raised by Alec, 2026-11-12; not settled)

Alec: "Since system2 teaches system1 ... it suggests that we may be able
to teach the intuition (i.e. the order-zero sigma/pi/negation mapping) in
the process of explicit language processing."

- *As built, nothing teaches it.* The record of 2026-11-11 (FutureWork,
  "An existence argument, and what it does not give") shows that a map
  through `intuit` equal to the grammar's processing exists, and says that
  existence "does not say System 2's activations teach it". Intuition's
  `pi` and `sigma` train from the kept path's output (§3.6); its terms are
  relation rows; System 1's step in a file with no vector operator is the
  fold along definitions (`FieldProjection.close`).
- *Corrected the same day.* Alec: "we decided, I think, that sigma pi
  updates should be the mechanism of the spreading activation, rather than
  just priming ... s2 relies on the s1 priming and backprop will take the
  gradient through both." The decision he recalls is of 2026-10-28: the
  open pass of `intuit` over the learned definitions is the spreading
  mechanism, because the spreading it replaced never learned from answers.
  What is built differs in two ways, and "nothing teaches it" above is too
  strong:
  - the spread is `ConceptSpread.open_prediction`, an `ActivationPredictor`
    and not the fold of `pi` and `sigma` along definitions. The predictor
    is a small network (`Expectation.ActivationPredictor`: a linear layer,
    a tanh and a linear layer over the context, then a product with each
    concept's key and a sigmoid); it has no least or greatest and reads no
    definition. `intuit` is separate from it: a thought executor whose
    terms are relation rows, evaluated by least and greatest with a `pi`
    and a `sigma` weight (`Intuition._evaluate`). The two meet at one
    point: the predictor's expected knowing orders `intuit`'s rules
    (`ConceptSpread.rule_order`). A third piece is also of `pi` and
    `sigma`, the fold along the concepts' definitions
    (`FieldProjection.close`);
  - every reader of it (attention, the compose chooser, retrieval,
    `intuit`'s rule order) reads it detached, so no gradient of System 2's
    cost passes through it. It is taught by a target: it trains on its own
    error against the concept state the sentence's processing leaves
    (`ConceptSpread.loss`, `expectation.concepts`).
  So System 2 teaches System 1 today as a target and not through
  backpropagation. When the built form came to differ from the decision
  was not traced.
- *What teaches each, as built* (read in the code, 2026-11-12; Alec: "we
  had discussed integrating them"):
  - *The predictor.* At a sentence's opening it predicts the order-zero
    state from the state observed before. Its target is the order-zero
    field of the kept path once its thought has ended
    (`ConceptSpread.complete`, "The next observed field is a teacher,
    never a differentiable target"). Its cost is the squared error
    `expectation.concepts`, its parameters are in the expectation group,
    and nothing else trains on that cost.
  - *`intuit`.* Its terms are relation rows, written when sentences that
    state relations close, with the row's evidence as trust. Its weights
    are one `pi` for a literal and one `sigma` for a term
    (`IntuitionLayer`), in the output group: a held intuition is
    evaluated again with its weights live when it is written into the
    field (`Intuition.write_knowing`), and the output cost reaches them
    by gradient. That a gradient arrives on a run was not measured.
  - *The fold along definitions.* Its edges are made when a candidate of
    MTM is promoted and when a member is admitted to a class
    (`cover_members`). On a verbal file its weights are trained by
    `DefinitionFold`, whose teacher is the evidence of the relation rows,
    detached ("only the sigma read is live"); on the numeric XOR files
    they train from the output cost.
  So the two have different teachers: the predictor a target from what
  processing leaves, `intuit` the answer's cost by gradient.
  By order (Alec's question, 2026-11-12): the predictor and `intuit` both
  map order zero to order zero (`ConceptSpread.open_prediction` reads
  and predicts order-zero rows only; `Intuition` is "the order-zero
  thought calculus" and expands higher orders to order zero first). The
  fold along definitions is the one that goes between orders, from a
  token to its type; it has no rule about order of its own, so a
  definition between two concepts of one order is folded too.
- *What extend supplies* (Claude's reading). Every sentence read gives a
  pair: the concepts its words activated, and the field of understanding
  its operators left. That pair is an example of the map to be taught.
  And the steps between them are already in System 1's own terms: an
  extend that narrows is an `intersection`, a `pi`; one that joins is a
  `union`, a `sigma`; `not` is the swap of lanes. So each call of extend
  is a term of the kind the fold applies.
- *A way to teach by it* (Claude's proposal). The set an extend leaves is
  observed as a candidate in MTM with the operands it came from, as
  `FamiliarTier.observe_intersection` observes one now; when it recurs it
  is promoted to a concept whose definition is that term over its
  operands (`add_concept_edge`, conjunctive or not, negated or not). The
  fold then gives that set in one step, without the tree.
- *The same, by a target* (Claude's proposal, with the one above). The
  spread becomes the fold of `pi` and `sigma` along the definitions, as
  decided on 2026-10-28, and its target becomes the field of
  understanding that extend left: the error between the fold of what the
  words activated and that field trains the definitions' weights. System
  2's choices are hard, so a target reaches the weights where a gradient
  through those choices would not.
- *What the record already warns of.* A flat map is right only where it
  was taught, and the grammar extends to a new combination where it is
  not; combinations like parity need as many terms as they have cases;
  and what rests on memory System 1 does not read cannot be taught.

#### What exists, and what does not

- The field of a sentence at its close is read from its words
  (`cs_read_memberships`, `SentenceField.open_field`), and a bare vector
  is matched against every concept's code (`FieldProjection.stimulate`).
  Neither follows the tree.
- `OutputOccurrences.generated` walks an idea to its leaves and sets the
  leaves' fields side by side; no operator is applied to them.
- `compound` selects inside its compose (`CaseSelection.select_cases`).
  `intersection`, `union`, `not` and `non` act on fields
  (`FieldLogic.reduce`).
- No operator has an `extend`. The determiner passes its noun through and
  sets a mode (`LowerLayer.determiner_mode`); the choice is made elsewhere,
  among earlier mentions.

#### Certificates, when it is built

22. Every operator of the grammar file has an `extend`, and a file whose
    operator lacks one fails at load.
23. On a sentence with a modifier, a noun and a verb, the calls are made
    once for each operator of the tree in compose's order, and the field
    they leave is the intersection the three narrowings give; no vector is
    matched against the concept codes on the way.
24. With "cat" a kind of three subconcepts: "a cat" leaves their centroid,
    "every cat" all three, and "the cat" the identity MTM holds, or
    nothing when MTM holds none.
25. An idea recalled from `LTM0` is walked to its tree and extended to the
    field its sentence left when it was written.

### 3.18 Two networks of System 1: the literature, and the round (approved by Alec, 2026-11-12; Part A as he then changed it)

Alec, 2026-11-12: "The definition fold and the spreading activation are
complementary, and should be seen as a single mechanism. The definition
fold takes low-order conceptual activation to high-order conceptual
activation by entailment. The spreading activation currently only works on
order-zero ... So both are spreading network activation across all
concepts. we can probably model this as several partitioned sigma-pi
layers (ramsified, again) for inter-layer and intra-layer connections.
So, both would be taught when symbolic processing was happening if
expectation (which we are assuming is the single priming mechanism) is
trained by the difference between the lossy input and the actual input (or
perhaps the reconstruction): missing words get inferred from context. That
leaves intuit() as a mechanism that does roughly the same thing; from a
conceptual state, we produce a new conceptual state. But lets leave this
as a separate mechanism for insights that trigger knowing as a result of
observation directly, where that knowing may be produced by symbolic
interpretation and alteration of the conceptual field. So we have separate
system1 inference networks, one that learns the input based on missing
information, and one that learns from conceptualizing or thinking about
the input."

#### The literature (Claude; every reference re-checked against its source on 2026-11-12)

*The first network: expectation, taught by what is missing.*

- *Spreading activation and priming.* A pair of associated words is
  recognized faster than an unassociated pair (Meyer and Schvaneveldt,
  1971; the two strings were shown together, and the pairs were
  associates). Collins and Loftus (1975), extending Quillian, explain
  priming as activation spreading from a concept's node along the links
  of a network in a decreasing gradient. Posner and Snyder (1975) set out
  two components, an automatic activation that is fast and takes no
  capacity, and a conscious attention that is slower, limited and brings
  inhibition; Neely (1977) separated them by the interval between prime
  and target, finding the facilitation from a category name to its
  members at every interval and the attentional effects only at long
  ones.
- *Without association, and among abstract words.* A meta-analysis finds
  priming between words related in meaning and not associated: small,
  about a quarter of a standard deviation and half the associative effect
  (Lucas, 2000). Hutchison (2003) agrees for some relations (function,
  synonyms, antonyms) and disputes that category coordinates prime
  automatically without association. Associated abstract words prime one
  another (Bleasdale, 1987); for abstract words priming has been found
  with association and not with likeness of meaning alone (Ferre and
  others, 2015). That abstract concepts are organized by association and
  concrete ones by similarity rests on one patient (Crutch and
  Warrington, 2005) and has had mixed replication. A category name speeds
  its members (Rosch, 1975; Neely, 1977), in part by expectancy (Neely,
  Keefe and Ross, 1989). No study was found of one category-level word
  priming another. So the route with good support that involves a
  higher-order word runs from a category down to its members.
- *Levels.* The interactive activation model (McClelland and Rumelhart,
  1981) has a feature, a letter and a word level, with excitation and
  inhibition between adjacent levels and, within a level, inhibition
  only. It is a precedent for propagation between levels with competition
  inside one, not for one unit raising its neighbour in a level. A
  sigma-pi unit takes a weighted sum of products of other units' outputs
  (Rumelhart, Hinton and McClelland, 1986); Feldman and Ballard (1982)
  have a conjunctive connection of a nearby kind, the greatest over sites
  of a thresholded sum at a site.
- *Taught by the missing part.* When a cough replaces a speech sound,
  nineteen of twenty listeners report that nothing is missing (Warren,
  1970). The cloze procedure deletes words for readers to restore, as a
  measure of readability (Taylor, 1953). The brain's N400 response to a
  sentence's last word is larger the less that word is expected by its
  cloze probability (Kutas and Hillyard, 1984, who read it as priming). A
  network trained only to predict the next word, on a small artificial
  language, groups its words by grammatical class and by meaning (Elman,
  1990). In predictive coding the connections downward carry predictions
  and those upward the remaining error (Rao and Ballard, 1999). Recalling
  a stored event from a small part of it begins with Marr (1971); a fast
  system that stores single events beside a slow one that finds the
  structure across many is McClelland, McNaughton and O'Reilly (1995).
  Reconstructing an input from a corrupted copy (Vincent and others,
  2008) and predicting hidden words from their context, "inspired by the
  Cloze task (Taylor, 1953)" (Devlin and others, 2019), are the same rule
  in machine learning.
- *Where it is weak.* In nine laboratories the effect of a word's
  predictability at the noun was reproduced and the effect at the article
  before it was not; the authors conclude against routine pre-activation
  of a word's sound, not against prediction in general (Nieuwland and
  others, 2018). Prediction is graded. And the literature's spread adds
  activations with inhibition beside them, where ours takes least and
  greatest on two lanes.

*The second network: intuition taught by thought.*

- *Deliberate work becomes direct.* Search that is controlled at first
  becomes automatic detection with consistent practice (Schneider and
  Shiffrin, 1977; the course of that learning is in the second part,
  Shiffrin and Schneider, 1977, where hits rose from about half to about
  nine in ten over 1,500 trials). Knowledge that starts as facts
  interpreted by general procedures is compiled with practice into
  procedures that apply directly (Anderson, 1982). Performance is
  automatic when it rests on retrieving a stored past solution in one
  step, in place of an algorithm (Logan, 1988). In Soar, when a subgoal is
  solved, chunking stores what it used and what resulted as a rule that
  later applies in place of the subgoal's work, in like situations and
  not only the same one (Laird, Rosenbloom and Newell, 1986). People
  reuse earlier inferences to answer later ones faster (Gershman and
  Goodman, 2014), and a network trained on the results of a tree search,
  which then guides the next search, is described by its authors in the
  terms of the two systems (Anthony, Tian and Barber, 2017).
- *Intuition as recognition.* "Intuition is nothing more and nothing less
  than recognition" (Simon, 1992, p. 155). Skilled intuition develops only
  where the environment is regular and there has been opportunity to learn
  it, with prolonged practice and fast, unambiguous feedback (Kahneman and
  Klein, 2009); here that is the condition that the thought teaching it
  was right.
- *Where it is weak.* What is compiled is right only where it was
  practised, which is the limit §3.17 records.

*That there are two.* The fast side of dual-process theory is not one
system. Stanovich (2004) names it the autonomous set of systems, which
includes implicit and associative learning, conditioning, and rules
practised until automatic; Evans and Stanovich (2013) call "System 1" a
misnomer that should be plural, and define the fast kind by autonomy
alone. Learning from regularities needs no deliberation: people who only
memorize strings from an artificial grammar come to judge new strings
better than chance (Reber, 1967; that they cannot state the rules was not
confirmed in the 1967 text), and infants of eight months segment a stream
of syllables after two minutes from the statistics between neighbours
(Saffran, Aslin and Newport, 1996). The criticism of the theory is that
its features are continuous and have not been shown to fall into two
clusters (Keren and Schul, 2009; Melnikoff and Bargh, 2018); it is aimed
at a partition into exactly two, not at this difference.

*Judgment.* Valid, with two reservations. Both networks have direct
precedent, and so does keeping them apart. The reservations: the
difference between the two is a difference of teacher, which the
literature supports, not a proven difference of mechanism, since both
are a map from a conceptual state to a conceptual state; and the claim
that the first is taught "when symbolic processing was happening" is
ours, since the literature's version needs no symbols.

*How the references were checked.* Each was opened at its publisher's
record or the paper itself on 2026-11-12, and each statement above was
tested against the abstract or the text. Every citation was right. Five
of Claude's first characterizations were not and are corrected here: what
Hutchison (2003) concluded; what kind of study Crutch and Warrington
(2005) was; the connections within a level of the interactive activation
model; how much precedent there is for priming between higher-order
words; and which paper is the source for recall from a part. Read from an
abstract or a secondary source only: Marr (1971), Kutas and Hillyard
(1984), Neely (1977), Crutch and Warrington (2005), Reber (1967).

#### The round

Approved, Alec, 2026-11-12: "Approve as proposed, but please re-check the
references and reference them in the doc/*md where relevant." Its first
part was then changed in discussion the same day, and he asked for the
result: "I think the is the way to go, please add it as a round." What
follows is the round as it now stands. It comes after the round of §3.16
and §3.17, on which it depends.

*What changed, in Alec's words.*

- On the kind of network: "I have some preference to create both networks
  as sigma-pi networks for consistency, unless you think the existing
  priming network has excellent merit (I think it does use fewer
  parameters, but this is not a place where thrift is justified)."
- On likeness and definitions: "we could consider doing intra-order
  priming via cosine angle and inter-order priming via sigma pi, which
  only requires good spatial distribution of the concept vectors (which we
  might not be relying enough on if we represent the interconnections with
  weights)"; "not maintaining the concepts^squared matrix is particularly
  attractive. (at order zero)".
- On where the nonlinearity is: "My feeling is that order-0 expectation is
  a nonlinear operation over the conceptual field, which is not possible
  using the hypersphere. But a sparse sigmaPi with ten million concepts is
  still enormous parameterwise and computationally, and my general feeling
  is that type relations are in fact strict (felix is a cat) but neighbor
  relations are not (Felix is like Ciao). So perhaps expectation is
  nonlinear in virtue of vertical propagation."
- On one graded match: "Re entailment vs expectation, we can make it fuzzy
  and achieve both: entailment is not a binary".

The first plan of Part A, edges learned within each order of the
definition store, is withdrawn: at order zero they are a table of pairs.

*What it rests on* (Claude's checks; the toys are arithmetic on a few
hand-made concepts, not runs of the model).

- *The predictor it replaces has good merit, not excellent.* Its size does
  not grow with the inventory (`Expectation.ActivationPredictor` with its
  default hidden width of 64: $386\,w + 330$ parameters at code width $w$,
  counted from its constructor), and it reaches a concept it has never met
  in a context by the likeness of codes. But it forms one query for a lane
  of a stream and scores every concept against it, so what it raises is a
  direction among the codes; nothing it learns can be read edge by edge,
  and nothing it learns is a definition that the fold, `intuit`, extend or
  thought can use.
- *Likeness alone cannot give an expectation that depends on a
  combination.* Four sentences: A and B give X, C and D give X, A and D
  give Y, C and B give Y. With likeness pooled by a sum, X must beat Y in
  the first two and lose in the last two; adding the first two conditions
  and adding the last two makes one quantity both greater and less than
  zero. With the places learned freely and two hundred restarts, likeness
  alone reaches three of the four cases and never four, pooled by a sum or
  by the greatest, with nonnegative vectors or any.
- *Going up and down gives it.* With each sentence kept as a container of
  its three concepts: up from what is present to each container by the
  share of it that is present, down to its missing members. The right word
  has two thirds and every other one third, in all four cases.
- *Where the nonlinearity lies.* With a member taking the greatest of what
  its containers hand down, the lesson is right at any sharpness of the
  match. With a sum in its place the four cases tie when the match is in
  proportion, and are right once the match is raised to a power above one.
- *One graded match keeps the XOR proofs expressible only if it can be
  sharpened.* XOR of two inputs from the inputs and one conjunction of
  them is exact with the least, as built; impossible with the share
  present in proportion (worst error one half); exact with the share
  present raised to the power two or four.
- *The places are already taught by co-occurrence.* At every kept close
  `MereologicalCodes.observe_close` moves each registered row toward the
  rows it occurred with and away from sampled others, "skip-gram's
  objective ... read from the model's own registered co-occurrence, in
  conceptual space's metric" (Architecture, "Placement"); a promoted class
  takes its members' centroid (`derive_class_placements`). A spread over
  the places was built once ("Priming as location-based diffusion",
  decided 2026-10-26, `Priming.propagate`) and replaced by the predictor on
  2026-10-28.
- *Cost.* Edges within order zero are extra parameters. A product of
  likeness over every concept of order zero adds none, since the codes
  exist, but costs the code width times their number for every sentence.
  Going up and down touches only what holds the active concepts and what
  those hold.

*Part A, expectation by going up and down.*

1. *Containers.* What holds a concept, and can be gone up to: a concept
   whose definition has it as a member (the edges of the definition store);
   a candidate of MTM whose support has it; and a row of `LTM0` whose entry
   refers to it. (Confirmed by Alec, 2026-11-12.)
   - *Part-whole relations that are stored and not used* (Alec,
     2026-11-12: "I think we have unused part-whole relations in symbolic
     space"; traced in the code the same day).
     - *Never read for a decision.* The symbol space's own taxonomy
       (`Language.Taxonomy`, a hierarchy of parents and children over
       symbols with a priming step that walks it): `Taxonomy.add` has no
       caller, `attach_knowledge` is called only by tests, and its switch
       `symbolicPriming` is off by default. The difference between a kind
       and a part: `LTM1.readings` computes it and `ClauseRow` folds both
       into `part`. The rows that chain one occurrence to the next
       (`MemoryPaths.write`), whose one reader `ChainRead.path` has no
       caller. And the writers `relate`, `reify_concept`,
       `synthesize_higher_order`, `singleton_concept` and
       `assert_concept_relation`, which only tests call.
     - *Read only by thought.* The part and whole relations that
       sentences state, where they are not an admission to a class:
       `relation_evidence`, the faces of thought and `Intuition.pervasion`
       read them, and nothing in the open pass does. They do not reach
       the definition store.
     - *Used.* The membership of a member in a raised class: copied into
       the definition store as an edge (`cover_members`) and read to place
       the class (`Priming.part_edges`). The copy keeps the greatest value
       it has had, so later evidence against a membership does not lower
       it; only `DefinitionFold`'s cost draws it back.
     - *Claude's proposal, for Alec.* A stated part or whole relation is a
       relation of the strict kind this pass goes up and down through,
       and is admitted to the definition store as a class membership is,
       so that it is a container; this is the relation rows providing
       "input to relations over concepts" (§3.12). The symbol space's
       taxonomy, the chain rows and the uncalled writers go with the
       reduction of §3.16.
2. *Up, by one graded match.* The degree to which what is active entails a
   container is the share of the container that is present, lane with
   lane and never netted: the candidate coverage of
   `IdentityRows.meet_ratios`, the measure `FieldProjection.stimulate`
   uses for a code in a point. It is raised to the container's `pi`, a
   power that starts at one. At one the degree is in proportion; as the
   power rises it approaches the least. A degree of one is entailment and
   a degree below one is expectation: "entailment is not a binary".
3. *Down, to what is missing.* A container hands its degree to its members
   that are not present. A member takes the greatest of what is handed to
   it, lane with lane, raised to its `sigma`.
4. *Likeness ranks; it is never taken over an order.* Among the members a
   container hands down, each is weighted by the likeness of its place to
   the places of the container's present members, in the conceptual
   space's own measure (`meet_ratios` on the placement directions).
   "Felix is like Ciao" is read among cats. No product is taken over all
   the concepts of order zero.
   - *Checked at every order* (Alec, 2026-11-12: "Please verify that
     hypersphere proximity works at all orders, between concepts at that
     order"). Run on the built code with a small fixture: six words given
     places by hand in two groups, four classes of them, and two classes
     of those classes. Every concept of orders zero, one and two has a
     unit place; a class has the centroid of its members and a class of
     classes the centroid of its classes (`derive_class_placements`,
     lower orders first). Between two concepts of one order, like ones
     are near and unlike ones far at each order: cosine 0.996 against
     0.342 at order zero, 0.999 against 0.342 at order one, and 0.301
     between the two unlike concepts of order two; the space's own
     measure (`meet_ratios`) orders the pairs the same way. Three limits:
     a class has no place until every member has one; the flag that its
     members lie within fifteen degrees of the centroid
     (`placement_valid`) is read by nothing but a report, so a class of
     scattered members still gets a centroid that is like none of them;
     and the check sets the words' places by hand, so it shows the
     mechanism and not that training spreads the places well.
5. *One pass.* Up once and down once; nothing is repeated. For the rows of
   `LTM0` the step up is the product of §3.12 over the window, for
   candidates MTM's entries, for definitions the store's edges.
6. *It replaces the predictor.* `Expectation.ActivationPredictor` leaves
   the conceptual space (`concept_predictor`, the network of
   `ConceptSpread.open_prediction`). The pass writes the expected knowing,
   at every order. What reads it (attention's relevance, the compose
   chooser, retrieval, the order of `intuit`'s rules) reads it detached,
   as now. Both networks are then of sigma and pi.
7. *The guard stands.* "Codes never read a value computed from codes"
   (Alec, 2026-10-20). The pass's output is not written to the seen
   ledger, and no cost of the pass moves a place.
8. *What teaches it.*
   - The places, by co-occurrence at every kept close, as built.
   - The containers, by being made: a row when a sentence closes, a
     candidate when MTM observes one, a type when a candidate is promoted.
     An expectation is there after one meeting.
   - The powers `pi` and `sigma`, and the weights of definition edges, by
     omission. `Omission` hides words and percept slots and knows what it
     hid; the cost is the difference between the expectation from the
     lossy input and the field of the actual input, the latter a teacher
     and never a differentiable target, as `ConceptSpread.observe` treats
     its target now. It takes the place of `expectation.concepts`.

*Part B, intuition taught by thought* (as approved).

9. `intuit` stays as it is in kind: least and greatest over terms with a
   `pi` and a `sigma`.
10. Its teacher becomes what thinking left. The order-zero field of the
    kept path after its thought, which teaches the predictor today,
    teaches `intuit`: the difference between what `intuit` gives from the
    observed field and what the field became. The gradient of the output
    cost stays beside it.
11. Its terms come from relation rows as now, and from the steps of
    extend and of thought that recur, promoted as §3.17 proposes.
12. It is evaluated once when the sentence is observed, before any
    thought, so that what it has learned is known directly. (This is the
    open pass of the decision of 2026-10-28.)

*Certificates.*

26. On the lesson of four sentences above, read once each: with the last
    word of each omitted, the pass expects it above every other concept,
    in all four. Two controls do not reach four: likeness alone, and the
    pass with a sum in place of the greatest on the way down at `pi` one.
27. After one sentence is stored, its omitted word is expected from the
    rest of it, through its row.
28. A conjunction with one of two members missing has a degree of one
    half at `pi` one and a lower degree at a higher `pi`; wholly present,
    it has one at every `pi`.
29. Among a type's members handed down, the one whose place is most like
    the present member's is first.
30. The pass takes no product over the concepts of order zero, asserted
    on a count.
31. The XOR proofs hold (certificate 11). This is measured before
    anything else of the round is built, since step 2 changes the fold
    they use.
32. A conclusion that thought has reached several times is given by
    `intuit` at observation with thought withheld; on a combination not
    met before it is not, and thought still reaches it.
33. `ActivationPredictor` is in no conceptual space.

*Risks.*

- Step 2 changes the fold of a conjunctive definition, which the XOR
  proofs and every field with a promoted intersection use. By the
  arithmetic above the proofs stay expressible only when `pi` rises above
  one. Whether they still converge is measured first, and the round stops
  there if they do not.
- Part A changes what attention and the chooser read on every sentence,
  so a regression can appear anywhere.
- A type with very many members hands its degree to all of them. The
  ranking of step 4 and a bound on how many take it keep that from
  spreading thin.

*Confirmed by Alec, 2026-11-12 ("Yes, all three").*

a. That candidates of MTM and rows of `LTM0` are containers, beside
   types (step 1).
b. That the one graded match takes the place of the weighted least in the
   fold of a conjunctive definition (`FieldLogic.reduce`, `intersection`,
   as `FieldProjection.close` applies it), as "achieve both" implies. The
   field operators between two fields would not change.
c. Whether a type also raises a type like it, by the likeness of their
   places. It is left out here: the literature found for it is thin
   (above), and it can be added once measured.

*Not in the round.* Folding `intuit` into the pass, which Alec has set
aside; any gradient of System 2's cost through the pass; edges learned
within an order; and any product of likeness over the concepts of order
zero.

#### A further proposal: vertical relations over symbols, horizontal over all concepts (Alec, 2026-11-12; under discussion)

Alec: "I guess what I want to propose is that the percept taxonomy, which
is already mereologically structured, can continue to be structured
(although over symbols using SigmaPi). We might then be able to place all
concepts in a single conceptual associative space that is not ramsified.
That would give vertical relations over symbols and horizontal relations
over all concepts."

Claude's thoughts, for the discussion:

- *It is the division already reached, given to the two spaces that
  exist.* Strict relations, by which expectation goes up and down, are
  the symbol space's; likeness is the conceptual space's. The two share
  one index (a symbol for a concept, the invariant of 2026-09-14), so one
  row has a place in the one and its relations in the other.
- *What is there already.* All concepts have had one similarity space
  since 2026-10-22 ("Every concept of every order is placed in one
  similarity space"). The definition store is of sigma and pi over those
  rows, and continues the percepts' structure: a concept of order zero
  collects basic percepts, which knit parts and wholes. A concept's order
  is already derived from its part relations where none is recorded
  (`_part_order`). The symbol space has a taxonomy of its own that
  nothing fills (`Language.Taxonomy`; Part A above).
- *What it would change.* The structure of sigma and pi is named and
  owned as the symbols' taxonomy, one structure and not two. The
  conceptual space loses its partition by order (the pools of
  `_order_caps`, the slices of `_field_caps`); an order is then the height
  of a symbol in the taxonomy and no axis of the conceptual space.
  Likeness is taken among concepts whatever their heights. A part or
  whole relation that a sentence states is a vertical relation, which
  settles the proposal of Part A, step 1, in its favour.
- *What it fits.* A concept with no symbol has a place and no vertical
  relation: "not all order-zero concepts are necessarily symbolized"
  (§3.17). And the two families of the literature, a hierarchy of labeled
  links and a space of likeness, become the two spaces.
- *How far it reaches in the code.* The partition is in few places
  (`_order_caps`, `_field_caps` and `_field_order_slice`: 24 uses in four
  files, and the order passed to `_csw_concept_row`, 23 uses in eleven).
  Order as a property is in many (`order_of`, `reference_orders`: about a
  hundred uses in eighteen files) and would stay, read as a height.
- *Questions it opens.*
  1. *Answered.* Alec, 2026-11-12: "it also prevents a fairly meaningless
     centroid definition for higer-order percepts that are not in physical
     space. What I would like to get as a result is that the center of a
     table is its perceptual centroid (collected in physical space),
     whereas the center of the class of all tables is its class centroid
     (collected in prototype space, i.e. a center of mass that collects
     the kinds of things together over its order)." Both centroids exist:
     a particular's form is the weighted centroid of its basic percepts
     (`MereologicalCodes._percept_centroid`), and a class's place is the
     evidence-weighted centroid of its members' places, lower orders
     first (`derive_class_placements`).
     *Claude's first reading of it was wrong* (that a class keeps its
     centroid in the conceptual space and has no form). Alec: "No, genera
     are located in genera-space. Parts relate to wholes mereologically,
     tokens relate to types though learned (unobservable) SigmaPi
     connections. All of this is in perceptual and symbolic space.
     Conceptual space is organized as the hypersphere, and completely
     determined by context (proximity)."
     *The reading now, for Alec to confirm.* The centroid of a class over
     its members is the class's location in genera-space, on the
     perceptual and symbolic side: its form, the center of mass of its
     members' forms at their order. Its place in the conceptual space
     comes from context alone, as any concept's does. As built it is the
     other way about on both counts: a class's form is a copy of its base
     word's code (`ConceptIndex.index_part_row`), and its place is its
     members' centroid and is never moved by context
     (`placement_derived`, `derive_class_placements`).
  1a. *Whether activation that spreads up through percepts and symbols
     should reach the concepts.* Alec: "if we let that propagate to
     concepts (is this desireable?), it will have a structurally
     reinforcing effect over the (non-ramsified) conceptual distribution."
     As built it does not: placement's context is "Observed symbols only"
     (`placement_context`), and a member that is seen leaves its class at
     neutral. Claude's view: desirable, graded and bounded.
     - *What it does.* A type is present whenever a member is, so the
       step by context draws the type toward its members' contexts and
       the members toward the type and so toward one another. The
       centroid of a class then arises in the conceptual space by
       learning, and likeness comes to follow the taxonomy, which is what
       makes ranking members by likeness useful.
     - *The guard.* It follows learned relations from what was observed
       and reads no value computed from codes, so it does not break
       "Codes never read a value computed from codes" (Claude's reading
       of Alec's guard).
     - *Collapse.* A type that holds nearly everything is present in
       nearly every sentence and draws everything to one point, and the
       sampled negatives cannot brake it, since it is never absent. The
       degree handed up should be the graded one, and a type's pull
       should fall with how often it is present.
     - *Hardening.* A wrong membership would draw the places together and
       then look natural by likeness. Its pull should be weighted by the
       relation's trust.
     - *Measure.* With and without it: the collapse audit that placement
       has, and the certificate of expectation on omitted words.
  1b. *The three spaces, as Alec states them* (2026-11-12): "Percepts live
     in physical space, determined by .where. Symbols live in a higher
     order (mental) space, their locations are determined by the sigma/pi
     folds. both of these are in the same dimensional space of percepts.
     Concepts, across all orders and modalities, live in a contextual
     space and are treated alike. This is a big association space trained
     by contextual occurrence." And on the collapse feared in 1a: "it does
     so in proportion to the cardinality of its membership, so it will
     have minute effect. A type becomes present whenever one of its
     members is is not a binary statement."
     - *Collapse, withdrawn.* With presence handed up as the share of a
       type's membership that is present, a type of very many members is
       present to a minute degree when one of them is. The remedy Claude
       asked for is the graded match itself.
     - *What it means in the code* (Claude's reading). A symbol's
       location is the fold of its definition over its members'
       locations, in the percepts' dimensions; the correction of 1 said
       "center of mass" and should say this. Today an order-zero concept's
       form is the weighted centroid of its basic percepts
       (`_percept_centroid`) and a class's is a copy of its base word's
       code. Every concept, a class included, is placed by context
       (`place_rows`); `placement_derived` and `derive_class_placements`
       go.
     - *The field is the context.* Claude asked whether a type's
       presence in the conceptual space is the share of its membership.
       Alec: "tokens and types show up by the perceptual snd symbolic
       activations, which are run on presence and folds. Those determine
       a (vertical) perceptual field over all orders which can be used as
       the context in which the conceptual embedding takes place (so we
       are not just looking at the surface symbols like word-to-vec)."
       So a type's presence is what its fold gives, and the question of a
       share was the wrong one. The context of placement is the vertical
       field over every order, where today it is the words observed
       (`placement_context`). (Claude's reading: the field meant is the
       one from presence and the folds, and not `stimulate`'s match of
       codes, so that the guard on codes holds.)
     - *Collapse, and the sphere.* Alec: "Collapse could be balanced by
       the distribution of conceptual codes over the hypersphere, maybe?"
       Run with the built `placement_step` on a toy of forty concepts in
       four kinds, three members of one kind to a sentence, the likeness
       between members of one kind and of different kinds comes out:

       | context of placement | same kind | other kinds |
       |---|---|---|
       | the words observed, as built | 0.991 | 0.000 |
       | with each kind's own type | 0.994 | 0.013 |
       | with a broad type over all forty, wholly present | 0.984 | 0.109 |
       | the same, the broad type at its share | 0.989 | 0.029 |
       | the same, negatives drawn by how often a row is present | 0.990 | 0.063 |

       The sampled negatives of the built step are that balance: with a
       broad type wholly present in every sentence the kinds are drawn
       together a little and stay apart. Several broad types nested in
       one another would add their pulls; drawing the negatives by how
       often a row is present halves the effect here. It is a toy of
       forty concepts and one broad type, not a run of the model.
  2. Recall matches order by order (§3.12). With no orders in the
     conceptual space it would group a row's references by their
     symbols' heights, or match them all at once.
  3. Whether it is built with this round's Part A, which is the same
     pass, or as a step of its own. Claude's suggestion: the ownership
     and the stated relations with Part A; the end of the partition as a
     step of its own, measured apart.

#### Where that discussion came to rest (Alec, 2026-11-12; Claude's summary waits for his confirmation)

Alec, tying it back to what occasioned it: "The expectation() and intuit()
operations that operate from concepts to concepts are shaped by
expectation over the input itself and the system2 content, respectively."
And, at the end of the argument: "SO, long story short, symbols should not
take the centroid of the percepts that they collect; they are epistemic
points. To return to how that influences priming; we have a vertical
structure that collects percepts (sigmaPi pairwise graph) and a horizontal
structure that provides meaning (ramsified hyperspheres)."

On the way: "The higher-order percepts are currently centroids of lower
order percepts, which is meaningless. Therefore, to give meaning to the
location of the "table" concept, it must live in the center (centroid) of
all of the individual table prototypes. That is NOT a physical part/whole
space, it is a token/type space." "the perceptual parts and wholes unite
at the basic percept, and basic percepts are collected as parts only from
then on as order increases." "the dimensionality of symbols should be zero
(they are epistemic points which we do not feel, but which serve to
represent collections)." "The structure of percepts is the structure of
extensions, of specifically characterized entities. The structure of
concepts is the structure of intentions, of generally characterized
entities. the latter are ideas; the former are the field."

Claude's first summary had four things wrong, which Alec corrected the
same day: "Vertical: a sigma-pi graph that collects symbols (percepts are
not 0-d, so cant be collected). Also, a class is placed by context, but it
is the context after vertical token/type propagation (which will not pull
across orders, but after two rounds, the propagation up and down will
cause types to exert influence on related tokens. and this is system2
influencing system1)." "Re "Ramsified hyperspheres", its just one
hypersphere, but concepts are distributed according to ramsified
algorithms (types live in token space, wholes live in part space)." "the
symbolic activation determines placement within conceptual space, but
again, only relative to other concepts within the same order. Hopefully we
have well-specified symbolic activation at this point, and characterized
how it is used, and how (I think?) it should be understood as the field
(which was previously described as only order-zero concepts). we do not
get grammatical operations in the field, but we do get "types of types of
tokens" just like we had multiple layers of parts and wholes." And:
""meaning lies in the conceptual space" is too vague to put in the doc."

The summary as corrected, in the code's names:

- *Vertical.* A graph of sigma and pi that collects symbols. A symbol is a
  point of it, with no location and no dimensions. A percept has
  dimensions and is not collected; parts and wholes unite at the basic
  percept, and above it symbols are collected as parts only. The graph is
  the definition store.
- *The field is the symbolic activation.* What is present, and what the
  folds of the graph make of it going up, types and types of types, and
  coming down. It holds every order and no grammatical operation. It is
  what Part A computes, and it is used for: expectation, read by
  attention, the compose chooser, retrieval and the order of `intuit`'s
  rules; the context of placement; the entry a sentence leaves in `LTM0`
  (§3.16); what extend works on (§3.17); and what recall matches
  (§3.12). The field was described before as the order-zero concepts
  only (`SentenceField.descended`).
- *Horizontal.* One hypersphere holds every concept's place. A concept is
  placed by its context, and its context is the symbolic activation after
  the propagation between tokens and types. A place is moved only
  relative to concepts of its own order. A type therefore acts on its
  tokens through activation, up and then down to the related tokens, and
  not by a pull across orders; "this is system2 influencing system1".
- *What changes from what is built.* The context of placement is the
  symbolic activation, where it is the words observed
  (`placement_context`). `place_rows` moves a row relative to rows of its
  own order, where it takes every placed row in the context. A class is
  placed by context like any concept: `placement_derived` and the
  centroid of `derive_class_placements` go. And a symbol above the basic
  percept carries no perceptual form: the copy of a base word's code into
  a class's row (`ConceptIndex.index_part_row`) and a form taken as a
  centroid of percepts for a row above order zero (`_percept_centroid`)
  go.
- *One caution* (Claude's). Part A weights the members handed down by the
  likeness of their places (step 4). If that weighted activation were the
  context of placement, places would be moved by a value computed from
  places, against the guard "Codes never read a value computed from
  codes". The context of placement should be the activation the graph
  gives before that weighting.
- *What this supersedes of the proposal above.* That genera have a
  location in a space of their own, and that the conceptual space has no
  ramification at all: it is one space whose concepts are distributed
  order by order. The philosophy document's "Genera Are Not Located"
  stands.
- *Part B is unchanged.* `intuit` is taught by what thinking left.

## 4. Certificates and the one measurement

*Certificates*, in the fast suite unless a proof's own marker says
otherwise:

1. *The file is the switch.* For three small files (field operators only;
   field and examination; with vector operators) the phases that run are
   §3.1's and no other. `model.xml`'s default keeps the parallel body. The
   list of configurations whose pipeline changed, each with its port.
2. *One arithmetic.* `intersection` and `union` over a definition's edges,
   over a bracket's words and over a cue give `fold_presence`'s values on
   the same evidence at unit exponents; `not` exchanges the lanes. No
   `FieldAndLayer`, `FieldOrLayer` or `Attention.field_reduce`; a grammar
   file that names the field rules `and` or `or` fails loudly with the new
   names.
3. *The writer of intersections.* A set of lanes active together
   `raiseCount` times becomes one order-zero row whose conjunctive edges
   are exactly those lanes, the against-lanes addressed as negated parts,
   and the fold gives the row the least of them. In a file that examines,
   only a bracket on which the loop chose `intersection` proposes a set.
4. *`stimulate`.* A concept's code stimulates itself at 1. A conjunction's
   code stimulates each of its parts at 1, and a part's code stimulates
   the whole at its coverage. A member's class is present after the fold.
   A point that contains no known concept gives an empty field.
5. *Examination without vector operators.* The loop runs; a path makes no
   compose and no `BasicModel.generate` call; an explore path departs once
   in the loop; the close writes an anchor with the entry in reading
   order.
6. *Modifier plus noun.* The selection equals the least of the noun's
   cases and the modifier's field; applied twice it equals applied once;
   each use of the kernel test is counted.
7. *Intuition.* An intuition with no open slot is accepted and is the
   output's operand; the field after `write_knowing` reverses to what was
   written.
8. *Trust.* Evidence read from a full percept is 1 with no factor. A new
   object-concept's radius is 0. No `stage_menu` attribute or method. On a
   constructed tree the gates follow §3.7's order, open once and survive a
   checkpoint.
9. *Memory.* A query by concept returns the rows whose entries hold it,
   ranked by coverage, under intersection, union and negation of cue
   concepts. No `recall_time`, no `when` branch of `query`, no interval
   scope. No `percept` column on LTM1: a checkpoint that carries one loads
   with it dropped and named. On a constructed store the forgetting ages
   and the order of relation candidates are what they were.
10. *Numbers.* A numeric percept's presence is its value and it has one
   lane. A reading's code is $f(n)$ on its dimension's fixed axis and zero
   elsewhere; no codebook row is allocated for a value. `Ops.partOf`
   between the codes of two values is their less-than, and the `intersection` and `union` of two readings are the
   least and the greatest of the two values. No layer is learned at the
   input. The output layer is linear in the field's evidence, has one
   weight row for each concept row and zeroes a retired row's. No
   `read_concepts`, `conceptIds`, sentence-state branch of `_forward_head`
   or `answer_record_reader`. `BasicModel.reverse` returns a numeric
   percept's value from the field.
11. *The XOR proofs*, each under its kept name at its bar, on an input of
   two numbers with the values 0 and 1; the affine control and the control
   without the promoted intersection both at $\tfrac14$; each proof's
   verbal cover named and passing.
12. *Forgetting.* On a constructed store the maintained columns give the
   recomputed pass's retention order after writes, witnesses and deletes,
   and the pass makes no row-by-row product.
13. *The scalar beside the words* (§3.10). On its lesson, "one", "two"
   and "three" each beside its number, and one word never beside a number:
   each of the three has a concept of its own over the scalar, whose box
   is the least and the greatest of the numbers it was heard with, and a
   definition whose trust has grown with its views; the fourth word has no
   object in the scalar. A reading's code is $f(n)$ on the scalar's axis.
   With the number omitted, its reconstruction lies inside the word's box;
   with the word omitted, the words reconstructed from 1, 2 and 3 are
   "one", "two" and "three". Recognition: shown 1, 2 or 3 with the
   question and no word, "this" binds to the present event's order-zero
   identity and the model answers "one", "two" or "three", the binding
   accepted as perception. A moment with no number is unchanged on a
   seeded run.
14. *The ports*, each with its old and new text.

*One measurement*, on the frozen candidate:

- The XOR proofs once each, and ten unseeded trainings of the numeric XOR
  file reporting the four answers, the three weights and the promoted row
  of each run. A failed `test_convergence` stops the round, as in every
  round.
- The whole-path round's lesson arm once, with: walks, episodes and
  seconds for each sentence; the sentence at which each operator's gate
  opened; how often `compound` used the kernel test; and, for each
  sentence's understanding, how far the walk's field and the stimulated
  field agree (reported, not a gate).
- One forgetting pass on a synthetic store of $2^{20}$ rows, in seconds,
  before and after.
- The scalar's lessons (§3.10), reported and not gates: recognition of
  numbers that were not trained (.5 to 3.5 by halves, and 10), which shows
  each word's box and the numbers that belong to none; comparison, the share
  of pairs answered rightly; and "zero", heard where a number was expected
  and none was felt.
- One complete sweep.

*Two toys, run before specifying* (Claude, 2026-11-11; local receipt
`doc/benchmarks/2026-11-11-item1-8-numeric-xor-toy/`; proxies and not the
model's code).

The first tried a learned linear layer onto concepts' lanes, the form
first specified for the numeric input: 40 initializations, 3,000 epochs at
most; solved means four answers on the right side of $\tfrac12$ and error
below .05.

| Variant | Solved |
|---|---|
| linear in and out with no clipping (the affine control) | 0 of 40, error .25 |
| a clipped input layer alone, started at random | 17 of 40 |
| a clipped input layer alone, started at the lane map $(x, 1 - x)$ | 0 of 40, error .25 |
| lanes fixed at $(x, 1 - x)$; least and greatest with learned subsets | 39 of 40 |
| the same with product and probabilistic sum | 40 of 40 |
| a learned input layer started at random, with least and greatest | 25 of 40 |
| a learned input layer started at the lane map, with least and greatest | 37 of 40 |
| lanes fixed; one conjunction made by rule for each input seen; a learned union | 40 of 40, median 16 epochs |

What it left behind, now that no layer is learned at the input (decision
14): a learned clipped input layer is a hidden layer of its own and can
solve XOR without any field logic, so the proofs would then certify
nothing about the field; conjunctions made by rule settle in every run and
soonest, where conjunctions learned by gradient stall in 1 to 3 runs of
40; least and greatest do about as well as the product form.

The second checked the form now specified: inputs 0 and 1 with one lane,
the two inputs' concepts, their promoted intersection, a linear output.
The weights are $1, 1, -2$ exactly; without the intersection the error is
.25; a linear output started at random settles in 40 of 40 runs.

## 5. Answers and defaults

### 5.1 The ten points of the first writing, as answered

Each was open with Claude's default. Alec answered the fourth (decision
6), then: "1 promotion on recurrence", and "The rest of the defaults are
fine." (decision 12).

1. *Who writes the intersection of several concepts.* Promotion on
   recurrence (§3.2): a set of lanes that recurs together is promoted to a
   row, bounded by the candidate pool as percept groups are. Not edges at
   weight zero trained by the output.
2. *A composed point's field goes through the grammar's inverse*: the
   walk, with each leaf stimulated; reconstruction stays costed on that
   field (§3.3).
3. *The order of operators from the syntactic tree*: §3.7's rule.
   Root-forming operators and the fixed operations are trusted from the
   start; an operator's cap opens when the operators above it in kept
   trees are definite.
4. *The first trust of a definition written by `interpret`*: the certainty
   of the mapping, which needs a second modality (decision 6). This round
   counts it for definitions whose object is the scalar (§3.10) and
   changes no other.
5. *The verbal cover of the XOR proofs*: §3.9's list, with the four
   sentences answered by a word for the class reported and not required.
6. *The numeric input layer starts at the lane map.* Fallen away: no layer
   is learned at the input (decision 14).
7. *Removals*: `conceptIds` and `read_concepts`; `set_stage_menu`;
   `compound`'s kernel test kept only as a counted fallback.
8. *System 1's step where there are no rules* is the fold along
   definitions (§3.6).
9. *The standing gate moves with the proofs*:
   `TestMMXorConvergence::test_convergence` runs the numeric XOR file at
   the same bar, and still stops a round.
10. *Recall by time is removed*, not kept beside the query on concepts;
   time stays readable as content and order (§3.8).

### 5.2 The scalar: what was answered, and three defaults

*Answered.* The scalar case is in the round, images and signals of several
dimensions are not (decision 13). A number is a percept with one lane and
no learned input layer (decision 14). Part, max and min, and no sum or
product on values (decision 15). The XOR proofs read 0 and 1 on two
dimensions (decision 17). And by "Your defaults are fine for this."
(decision 18): the scalar is the last item built, with its certificate and
no other gate; a definition's trust is counted by views and agreement, for
these definitions only; one lesson file, the math corpus later; numerals
stay opaque for arithmetic; the value is the percept's presence; the
percept is the round's one numeric input.

*Revised after that answer, for the record.* Claude's default that "one"
and "zero" be two lanes of one concept did not survive a trace: an object
is spelled only where exactly one word names it (§2.11), so each word has
a concept of its own. Claude's suggestion of a second pole on the percept
carrying $1 - v$ was withdrawn on Alec's word (decision 14). Claude's
proposal of two felt levels for the XOR proofs was withdrawn on his word
(decision 17), with the worry behind it, which a check showed unfounded.

*Claude's defaults that Alec has not spoken to.* The rules use them.

a. *The box.* A word's concept over the scalar has as its bounds the least
   and the greatest of the values heard with the word, with membership by
   the two coverages, and nothing trained by gradient (§3.10). The
   tightest box around a word's examples is the standard learnable case,
   and a convex region is what a natural concept is taken to be (from
   memory, not checked: Blumer and others 1989; Gardenfors 2000). What it
   does not give is a guess for a value outside every box.
b. *Which lessons are required.* Recognition of the trained values is in
   certificate 13. Recognition of untrained values, comparison and "zero"
   are measured and reported.
c. *The toy's words.* "one", "two" and "three", each at its number felt
   through $f$; "zero" is an absence, inferred on a concept's
   against-lane. (Before decision 20: "half" at .5 and "one" at 1.0.)

*Answered.*

e. *The spacing $f$* (decision 20). Answered, then changed. First the
   running rank, which Alec defined as "a linear interpolation over
   recognized concepts, from 0-1"; then, since that comes out close to
   logarithmic, "something like a logarithmic mapping from 0..INF to
   0..1": $f(n) = g / (1 + g)$ with $g = \ln(1 + n)$ (§3.10). The two
   readings as they were first put: (1) A fixed curve, for
   instance $f(n) = n / (1 + n)$: one is .5, two is two thirds, three is
   three quarters. Simple and stable, but in single precision it stops
   telling numbers apart above about ten million. (2) The running rank of
   $n$ among the numbers the model has met, interpolated: with few
   numbers only their order is known, and the more there are the closer
   the scale comes to a continuous one, which is decision 20's "ordinal
   dim that becomes approximately real with increased ordinates". It is
   how sensory neurons are found to space a range, by the distribution of
   what they meet (Laughlin 1981); over the numbers of ordinary language,
   which grow rarer as they grow, it gives a compressed scale like the
   one people show (from memory, not checked: Dehaene and Mehler 1992;
   Prat-Carrabin and Woodford 2022). Its cost is that $f$ moves as numbers
   are met, so it is updated slowly, over a horizon as omission's rates
   are, and boxes are kept on the numbers, as §3.10 says. Claude's
   recommendation: (2).

*Traced, and answered.*

d. *What makes the perceived concept the thing that is said* (traced on the
   whole-path round's returned source `26dd5a0bda60` by a reader Claude
   briefed, and spot-checked). A concept present only in the moment's
   field cannot become the answer today, for three reasons.
   - *It is never a bind candidate.* The thought's `bind`
     (`ThoughtFaces.bind`, `Queries.bind_reference`,
     `ReferenceClose.bind_site`) takes its candidates from
     `context_records(frames)`: stored LTM frames, this episode's read
     frames and the MTM pool, each admitted only "if identity in
     alloc.individuals". Individuals are allocated at order one or higher,
     so no order-zero concept qualifies. `ReferenceClose.accessible` serves
     the closing's word sites, not the thought's `bind`. Nothing reads the
     moment's field into any menu; by then the published field is the
     generated one, and the open field survives only in
     `_sentence_open_fields[sid]`.
   - *A binding is not an answer.* `bind` stamps the meaning's producing
     operation `bind`, and `ThoughtAnswer.accepted` accepts only `infer`,
     `query` and `isTrue`, or evidence of kind `pervasion`, `retrieval` or
     `truth`.
   - *An empty question admits nothing.* `bind_site`'s score needs the
     candidate to cover the site's content wholly; whether an open "what"
     has content is a choice of the learned grammar (inferred, not traced).
   Saying is not in question: an accepted concept with one word is said
   by it (`OutputOccurrences.generated`, `word_surface_for_row`,
   `lexical_form`), and `BindingAnswers` scores it.

   *Alec's answer* (decision 21) moves the question from perception to
   identity: an order-zero event has an order-zero identity in MTM, the
   one promotion uses, and he confirmed that order-zero events are
   suitable for it. *The rule of §3.10's "Recognition"* (Claude's reading
   of his answer, which his confirmation settles): "this" binds to the
   present event's order-zero identity in MTM, and the question's open
   reference binds to the class whose member the present event is.
   `ReferenceClose.bind_site` adds to `context_records(frames)` a record
   for each individual (order one or more) whose evidence in the moment's
   open field (`_sentence_open_fields`) contains the present reading, that
   is, whose membership read gives it evidence there; the event's MTM
   order-zero candidate is the occurrence the binding is made for. Since
   such individuals are already admitted by `context_records`' filter, the
   missing link is that the moment's field is read into the menu, with the
   present event's MTM order-zero identity admitted beside the
   individuals. A binding so made carries evidence of kind `perception`,
   which `ThoughtAnswer.accepted` accepts as it accepts `retrieval`.

## 6. Not in the round

- *One model class.* Folding `BaseModel` into `BasicModel` and porting the
  nine helper constructions is housekeeping.
- *The whole-slab body for configurations that are not XOR proofs* (the
  `numeric` slab of `mnist.xml` and of `model.xml`'s default) is not
  changed.
- *Examination of a numeric row.* A row of numbers has no places to
  examine in this round; the glance reads it whole.
- *Several modalities in one moment, beyond §3.10's scalar*: signals of
  several dimensions, images, and a corpus that pairs words with values
  (decision 13).
- *A sum and a product on values* (decision 15).
- *The existing percepts' pole axis.* Only the numeric percept's one lane
  is specified here; `PerceptField.values` keeps its shape.
- *The dynamic stop, recall that drifts toward the typical, memory for
  one's own processing, duration as a quantity, and context across
  documents* stay where FutureWork has them.
- *Items 1.5, 1 and 0.5* are not assigned here (decision 1).
- *The fields of long-term memory that are neither table nor cache*
  (§3.14's second list: where a row came from and its text, the stream,
  recurrence and expectation, a row's semantic record). Whether each
  remains is decided one by one after this round.
- *The determiner as lowering and selection* (§3.13): decided in
  direction, built in the next 1.8 round.
- *The time result of recall* (§3.12), until a lesson teaches a time
  word.
- *Recognition of a value* (§3.10, "what is this?"): deferred by Alec to
  the next 1.8 round, on `compose_extend`.

## 7. Codex text (Claude, 2026-11-12; waits for item 2.0's completion; not to paste yet)

Alec, when it was first handed over: "Now? We have not completed and
pushed 2.0". So item 2.0 is completed first by its own text
(whole-path §11: land the accepted
round, push, repair its five defects). This text then opens by landing
those repairs, whose hash is filled in at their review, and builds item
1.8. Its item 4 moved to that text; the numbering is kept so that the
references between items stand.

~~~
Item 1.8: land the repairs that complete item 2.0, then build the last architecture round. The repairs are accepted in doc/plans/2026-11-11-one-pair-per-stage.md §13. Steps 1 to 3 land them and change no code, data or test; items 5 to 16 build item 1.8 as doc/plans/2026-11-11-item1-8-last-architecture-round.md states it. Read that specification's §1 to §6 first: §1 has Alec's decisions in his words, §2 the code as it stood at a39cbfdef8, §3 the rules, §4 the certificates and the one measurement, §5 what was answered, §6 what is not in the round. Section numbers below are that specification's unless named otherwise.

1. Check. The live tree's source hash by the receipts' method (test/bounded_tests.source_snapshot, SHA-256 of its compact sorted JSON) equals <the repairs' returned source, filled in at their review>. Since that return only handwritten documents changed. Run test/test_doc_links.py once; it must pass. No make test: the repairs' one sweep is the record.

2. Commit, on main, every file the repairs changed, added or deleted, with the handwritten documents changed since the whole-path landing. Receipts under doc/benchmarks/ stay local; force nothing in. Commit message in the form of 16a6a79b81: subject "Land accepted item 2.0 repairs"; body: land the returned source, with the acceptance in whole-path section 13; the five repairs of section 10.3 and their cases; validation. No model or assistant identifier anywhere in the message.

3. Record the landing: append one line to todo.md and one line to the whole-path specification after its §13, "Landed as <hash>" with the date; commit as "Record item 2.0 repairs landing"; then push main once. Item 2.0 is then complete. Report both hashes in your return. Then continue at once with item 5 from this landed main.

4. (Moved: the repairs are made by the text that completes item 2.0, whole-path §11.)

5. The grammar file is the switch (§3.1). Operators are of three kinds: examination (divide, descend, gloss), field (intersection, union, not) and vector (all others, conjunction and disjunction among them). What a file declares decides what runs: field operators only, the glance and the fold on the parallel body; examination operators, the loop of item 8; vector operators, compose and generate, with interpret's readings committed where interpret is declared. _derive_use_grammar's two values and _has_word_objects' conditions on conceptLayers, ltmConsolidation and conceptIds give way to that. Before the switch moves, list every configuration under data/ and every test fixture whose pipeline changes, each with its port; model.xml's default grammar keeps the parallel body.

6. One field logic (§3.2). intersection and union on the field are SparseLayer.fold_presence's weighted least and greatest on the two lanes, over whatever they are applied to (a definition's edges, a bracket's words, a cue's concepts); not exchanges the lanes; non withholds both. The file's field rules and and or are renamed intersection and union, one operator each, and a file that names them and or fails loudly with the new names; FieldAndLayer, FieldOrLayer and Attention.field_reduce go, and Attention.narrow_words' actions 3 to 5 use the one implementation. conjunction and disjunction stay vector operators with their product forms. The writer of intersections is promotion on recurrence (§5.1 item 1): a set of concept lanes active together raiseCount times, found by observe_candidate and promote_candidates, is promoted to one order-zero row whose conjunctive edges are those lanes, written by add_concept_edge(..., conjunctive=True, negated=...) with the against-lanes as negated parts; in a file that examines, only a bracket on which the loop chose intersection proposes a set.

7. A point and its field (§3.3). stimulate(point): for each known concept the candidate coverage of its code in the point, lane by lane (IdentityRows.meet_ratios), then the fold of item 6 upward. A composed point's field is its generate walk's (OutputOccurrences.generated) with each leaf stimulated, and reconstruction stays costed on it. A field's key is its evidence-weighted centroid, as built. The readers of a stimulated field in this round: a field operator applied to chosen operands, modifier plus noun, the query's cue, and the numeric output on a file with vector operators.

8. Examination (§3.4). The loop runs in every file that declares an examination operator. In a file with no vector operator its chooser's menu is the field actions; a path is the glance, the loop and the field it leaves, costed R (reconstruction_reverse of that field) plus A where an output is supervised; an explore path departs once in the loop; nothing is composed, generated or interpreted; the close writes the episode anchor with its entry, and MTM's order-zero chains as built. With vector operators the loop is as today.

9. Modifier plus noun (§3.5). compound selects among the noun's staged cases those the modifier holds: the least of the cases and the modifier's field, as CaseSelection.select_cases computes it, idempotent. select_view's kernel test remains only where the modifier has no field over the noun's cases, and each use of it is counted. compound stays a vector operator in the file.

10. Intuition (§3.6). ThoughtAnswer.accepted accepts evidence of kind intuition where the meaning has no open slot; an accepted intuition is the output walk's operand or is read by the numeric output. What Intuition.write_knowing wrote is part of the field that reconstruction reads. The bank's pi and sigma train from the kept path's output under the reader rule. In a file with no vector operator there are no rules, and the fold is System 1's step.

11. Trust and the order of operators (§3.7). Percept evidence is taken as read; a new object-concept starts at radius 0, as built. OperationSelectionLayer.set_stage_menu and stage_menu go. The clause gate's written exemption (interpret sign bind lift) gives way to a rule over the syntactic tree: the fixed operations and the operators that can form a clause's root in the file are exempt from the start; every other operator's evidence is capped at closedTrust until the operators that stand above it in kept trees are definite (PartsOfSpeech.operation's definiteness at the gate's threshold); one measured, monotone gate for each operator, checkpointed as the gates are now.

12. Memory addressed by content (§3.8). ThoughtLTMCapability.query reaches episodic entries by their references: the cue is a field, a row's match is the coverage of the cue in the row's references by meet_ratios on the lanes their signed values give, and a cue may combine concepts with the three field operators; an index from concept to rows is kept at EpisodicMemory.write and at forgetting; the best rows are reviewed by EpisodicMemory.reinstate. EntryRead.recall_time, query's when and window branch, best_match's routing of a time-only cue and _scope_contains' interval branch go. LTM1's percept column goes: cued_rows orders relation candidates by store order, history_age gives a relation the age of its sentence row, MemoryPaths copies no bands; list every reader and writer of the column with its replacement; a checkpoint that carries it loads with it dropped and named. What stays: the model clock, the bands inside an LTM0 row's vector, and every reader of time as content or order.

13. Numbers in and out (§3.9, §3.10; decisions 14, 15, 20 and 22). On a file with field operators and dataType numeric, each number of the input row is one basic percept with one lane, and its code is the value on a fixed axis of its own band in the percept cube, f(n) = g/(1+g) with g = ln(1+n), and zero elsewhere: no row per value, no drawn pattern, no learned input layer; the dense numeric slab is not used there. Ops.partOf on two readings' codes is their less-than, and union and intersection are their greatest and least; no sum or product acts on values. Values are never told apart by direction (cosine), only by order and containment. OutputSpace reads the field by one linear layer, a weight for each concept row and lane, fixed in shape at construction and zeroed for a retired row, plus a bias; it replaces read_concepts and conceptIds, the vector-linear branch on field configurations, and _forward_head's read of the sentence state with answer_record_reader. The output reads the field at the path's end (§3.9). reconstruction_reverse runs BasicModel.reverse from the field to the percepts: bytes for text, the value for a number.

14. The XOR proofs (§3.9, decision 17). Every XOR learning certificate (§2.9's table, and any other the inventory finds) moves to numbers on a file with field operators only: two numeric inputs with the values 0 and 1, the fold with its promoted intersection, the linear output. Names are kept and bars kept or tightened; the "two positive edges" assertions are ported with a record (now two positive weights and one negative). The controls: sum at 1/4, and the same file without the promotion of the intersection at 1/4. TestMMXorConvergence::test_convergence runs the numeric XOR file at its bar and still stops a round. Before a proof moves, name its verbal cover (§3.9): TestXorGrammarReconstruction stays on the symbolic file, the tests of what conjunction and disjunction mean on meanings stay, and the four sentences answered with a word for the class are added as a measurement.

15. The scalar beside the words (§3.10; decisions 13 to 16 and 18 to 22), built last, with certificate 13 and no other gate. A lesson sentence may carry one number; its percept and concept join the glance's field as one reading, with no place. A word read where the scalar is present gets a concept of its own over the scalar, minted on recurrence as objects are (FamiliarTier.promote_identity), whose box is the least and the greatest of the numbers heard with it, kept on numbers and read through f by the two coverages of meet_ratios; InterpretLayer.define writes its definition with evidence, and the definition's trust grows with its views as ConceptTrust counts a concept's. An omitted number is restored through the word's box (its lower bound); an omitted word from the number, through the box that holds it. Recognition (§5.2 d; decision 21): ReferenceClose.bind_site adds to context_records(frames) the present event's order-zero identity in MTM (the candidate observe_candidate opens) and each individual whose evidence in the moment's open field (_sentence_open_fields) contains the reading; "this" binds to the former and the question's open reference to the latter; the binding carries evidence of kind perception, which ThoughtAnswer.accepted accepts. Zero is an absence, inferred on a concept's against-lane, never a box. The lesson's words are one, two and three; no arithmetic acts on a value (the rule of 2026-10-07 stands). One lesson file; no corpus.

16. The forgetting pass (§3.11). Forgetting.retention_values reads maintained columns: luminosity, updated at a write or a witness by comparing the row only with the rows its content indexes to, in the same scope and order, and released at a delete; coverage, where forgetCoverage is on, a running sum with each row's contribution; deducibility cached at the write. The retention order does not change, and the pass makes no row-by-row product.

17. Certificates and the one measurement as §4: the fourteen certificates; the XOR proofs once each, and ten unseeded trainings of the numeric XOR file reporting the four answers, the three weights and the promoted row of each run; the whole-path round's lesson arm once, with walks, episodes and seconds for each sentence, the sentence at which each operator's gate opened, compound's kernel-test count and how far the walk's field and the stimulated field agree; one forgetting pass on a synthetic store of 2^20 rows, in seconds, before and after; the scalar's lessons (untrained numbers .5 to 3.5 by halves and 10, comparison, "zero"), reported and not gates; one complete sweep. With the landing: model.xsd and Params for every parameter added or removed, and Architecture, GradientFlow, Orders, STM and Philosophy amended by date where they describe what changed.

Testing: direct before-and-after cases and the targeted files while you build; the measurement of item 17 and make test once (env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected) when you offer the freeze. A failed test_convergence or shared-concept case stops the round; anything else is repaired at its producer under the recorded rules, or recorded. Where the specification is unclear or wrong, decide by its rules, record the decision and its reason in the return, and continue. Receipt under doc/benchmarks/<date>-item1-8/ with commands.json, frozen-source.json, before-source.json (the landed item 2.0 source), source.patch, ports.json (every port's before and after text) and offered-documents.json sealed before the sweep. No seed pinning, no resize of a fixture (the thought store's bounded growth stays as authorized), nothing retired without a record, no language named in code, fail loud. No commit or push after step 3.

Return as §8 of doc/plans/2026-11-11-item1-8-last-architecture-round.md, through Alec for Claude's review.
~~~


## 8. Implementation return (2026-11-12; through Alec for Claude's review)

### 8.1 Landing and offered source

The item 2.0 landing was already authorized, committed and pushed before this round: whole-path implementation `b35a4bd3167049b9ef3559104d478d89fd869e80`, landing record `d88acd3fab28951bb04baee7076b87cf4b5fcf90`, completed repairs `9f7923d67f8921ef5c14afdf2fe18c536802018b`. The latter is this round's unchanged HEAD. Todo's final item 2.0 amendment and whole-path §12.1 supersede §7's landing placeholder. No duplicate landing, commit or push was made. The before-source is `a22afbd67871f95acd69e1d0e268251b4c0f7a72c97dd82becd541d47cabc5e2`; its initial link check passed 1,014 cases.

The offered implementation source is `607ab7f8c5daf5a613ee8478aa6776b3ef96056607b482f7e351acafefe6bc37`. Local receipts are under `doc/benchmarks/2026-11-12-item1-8/`: `before-source.json`, `frozen-source.json`, `source.patch`, `commands.json`, `ports.json`, `offered-documents.json`, the configuration inventories and proof inventory. Receipts remain ignored. This is a candidate with recorded failures, **not a claim that all fourteen certificates are satisfied**. The completed measurement and sweep results below leave the offered implementation unchanged.

### 8.2 Implementation and decisions

- Grammar declarations select field, examination and vector execution. The default keeps its tier-local `pi`/`sigma` substrate body; these declarations are distinguished from symbolic choices. Examination-only paths close through a common field inverse and episodic anchors. This reconciles the default-body requirement with the phrase “every other operator.”
- `FieldLogic` owns the weighted lane arithmetic used by sparse definitions, attention and intuition. Recurrent intersections write negated edges for against-lanes. `FieldProjection` reads candidate coverage and closes definitions; generated leaves are stimulated after their inverse walk.
- Numeric fields have fixed perceptual axes and a fixed linear reader over inventory seats and lanes. Retired reader seats clear weights and optimizer state. `conceptIds`, `read_concepts` and the sentence-state answer reader are removed. `XOR_exact` keeps 64 epochs and uses existing output `lrScale=10` for f(1)=.409. Convex readout certificates use LBFGS without choosing a seed; the ten measured trainings use the native file's training loop.
- The former symbolic configurations survive as `MM_text.xml`, `MM_text_fixture.xml` and `XOR_symbols.xml` at their original capacities. The numeric ports necessarily change the input/event layout to two numbers. The full before/after text of every data/test port is recorded. Existing symbolic inverse and operator-meaning coverage remains; its failed bar is not relaxed.
- Per-operator monotone gates replace both written groups and every stage menu. Roots and fixed operations are exempt; kept ancestors supply the introduction order. The retired group/except-list tests are replaced with this contract and listed in the ports. Old gate checkpoints fail loudly; an old LTM1 `percept` column loads with a named drop warning.
- LTM1 readers/writers in Memory, ClauseRow, RowTime, EntryRead, SentenceStrata, MemoryPaths, Forgetting and corpus lesson reporting use sentence references and store order. Episodic recall indexes signed conceptual content. Model time and LTM0's own bands remain.
- Compound uses staged case coverage and counts semantic fallback operands. Its fixed tensor body still calls the kernel with zero placeholders for grounded rows to retain compilation compatibility; the count is of actual fallback operands, not physical batched calls. A scalar integration fixture uses the established eager-reading fixture after the compiled generator exposed a nested conditional tangent error; no production compiler setting is changed.
- Retention caches contrary similarities and semantic dependencies at writes. Coverage keeps coordinate leaders and reduced values; its read retains the former float32 mean/subtraction so sorting ties do not change. A hypothetical retained subset uses cached comparisons. The boundary pass makes no row-pair products. Bounded, unpromoted field recurrence statistics survive presentation reset; old occurrence histories do not, so repeated numeric presentations can reach promotion.
- One optional scalar is read beside text, without a place. Per-word bounds, counted definition evidence, identity promotion and perceived event/class binding are implemented. `reconstructModalities` exposes the captured scalar modality. The current direct certificates distinguish a native binding test from acquisition of the question's grammar and an emitted word. No scalar corpus or arithmetic operator is introduced. Zero creates no box.

### 8.3 Certificate disposition

| Certificate | Evidence and remaining limit |
| --- | --- |
| 1, file switch | Field, examination and vector phase tests; default parallel body; before/after configuration census. |
| 2, one arithmetic | Sparse, bracket/cue arithmetic and intuition tests; retired field spellings fail loudly. |
| 3, intersection writer | Numeric recurrence plus against-lane edge certificate; examination proposals require the chosen intersection. |
| 4, stimulate | Self=1, both parts=1, partial whole coverage, upward class closure and empty field cases pass. |
| 5, examination | Real train/eval paths pass with compose/generate forbidden and an owned inverse. The direct case checks an anchor exists; ordered episodic references also have memory tests. |
| 6, compound | Twelve case-selection/compiled-chooser cases passed; fallback counts are measured in the lesson arm. |
| 7, intuition | Accepted intuition and live knowing-to-reconstruction gradient cases pass; field readout uses the concluded field. |
| 8, trust | Constructed ancestor order, monotone openings, checkpoint/epoch persistence and rejected legacy configuration pass. |
| 9, memory | Signed-reference recall, compaction and named legacy-column migration pass; relation-only whole-path repairs pass. |
| 10, numbers | Native fixed axes, linear output updates, row retirement, reverse and owned-understanding cases pass. |
| 11, XOR/verbal cover | Numeric proofs pass their targeted bars. **The unchanged symbolic reconstruction case failed 0/4**: hello/world and there were not recovered distinctly. It remains failed, not retired. |
| 12, forgetting | Maintained values and order checked against recomputation after writes, witnesses, changed scope and deletion, with products forbidden in the pass. A float32 tie defect was repaired before freeze. |
| 13, scalar | Counted native boxes, direct omission readback and event/class perception binding pass. Full question-to-word recognition and omission through the complete model failed in the native lesson; the supervised-question attempt also failed before evaluation. Direct box lookup is not counted as that result. **Certificate 13 is not satisfied.** |
| 14, ports | Before/after data and test text is sealed in `ports.json`; verbal cover and changed contracts are named separately. |

Both stopping gates passed in the frozen targeted run and the complete standard sweep; no stopping failure occurred. Other failures are recorded under §7's rule. Architecture, GradientFlow, Orders, STM, Philosophy and Params have dated amendments; model.xsd removes `conceptIds` and declares the replacement gate parameters.

### 8.4 Frozen measurement and sweep

All measurements below use source `607ab7f8c5daf5a613ee8478aa6776b3ef96056607b482f7e351acafefe6bc37`, with no implementation changes after the offer. No seeds were selected. The receipt retains failed attempts as well as successful commands; a campaign wrapper returning zero does not override a failed inner command.

**Frozen certificates and stopping gates.** The two stopping cases passed (2 passed, 3.63 s). The certificate/proof selection passed 83 cases in 64.85 s, including the numeric ports and their controls. The symbolic verbal cover returned 11 passed and 1 failed in 445.31 s: `TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct` recovered 0/4. Its four reconstructed texts were `hello \x00\x00\x00\x00\x00`, the same again, and two entirely NUL-filled word pairs. The original bar and test name remain. This is a failed part of certificate 11.

**Ten native numeric trainings.** Each used `data/XOR_exact.xml`, its 64 epochs and native training loop. All ten returned four correctly classified answers with MSE 0.0004208456958. In input order (0,0), (0,1), (1,0), (1,1), every run returned `[-0.0187482014, 1.0231008530, 1.0231008530, 0.0162661895]`, reader weights `[2.5449194908, 2.5449194908, -3.7416417599]`, and promoted row 2. The fixed axes and reader initialization make these unseeded runs identical; they are not ten random readout starts. The individual records retain full precision, edge indexes and elapsed time.

| Run | Four answers | Three weights | Promoted row | Seconds |
| --- | --- | --- | --- | --- |
| 1 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 6.169 |
| 2 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 6.244 |
| 3 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 7.371 |
| 4 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 7.132 |
| 5 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 7.079 |
| 6 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 7.073 |
| 7 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 7.029 |
| 8 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 7.099 |
| 9 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 7.111 |
| 10 | −.0187482, 1.0231009, 1.0231009, .0162662 | 2.5449195, 2.5449195, −3.7416418 | 2 | 6.574 |

The separate supervised verbal class measurement retained the symbolic file's capacities and ran 400 epochs (502.49 s). For `hello world`, `hello there`, `loving world`, `loving there`, targets were `zero`, `one`, `one`, `zero`; answers were `hello hello world`, `hello hello there`, `loving hello world`, `loving hello there`: **0/4**, reported rather than a stopping gate (`verbal-class-supervised.json`). The first passive class observer passed a list where a prepared tensor was required and recorded that harness error; it supplied no class score.

**Whole-path lesson arm, one attempt.** The unchanged `explore-1.xml` lesson configuration completed 25 of 160 sentences before `ThoughtLTMCapability` rejected `query`. The producer is the query descriptor's method grant in `Queries.py`, which still permits `best_match` and `resolve_description`, while `ThoughtFaces.query` calls `query`. This is left visible on the frozen source. Completed sentences total 70.096 s, 97 generation calls, 50 thought episodes and 50 paths. Every sentence's counts and seconds are retained in `lesson/whole-path-sentences.jsonl`; gates and compound counts are in `lesson/item18-sentences.jsonl`.

The gates open at sentence 0 were the exempt roots/fixed operations `interpret`, `sign`, `part`, `whole`, `equal`, `implies`, `generic`, `lift`, `bind`. No other gate opened in the completed prefix. The cumulative compound fallback count at the last completed sentence was 25,071 operands. The walk/stimulated-field comparison is **unavailable for all 25 sentences**: the passive observer looked for the generated field after trial restoration, when that attribute was absent. This is an observer limitation, not evidence that the model had no generated field. The arm was not rerun to replace either result.

**Scalar lesson and certificate 13.** Twenty-four word-plus-reading presentations completed: six each of `one` with 1, `two` with 2, `three` with 3, and `unfelt` without a number. The learned boxes were [1,1], [2,2], [3,3], each with six views and six agreements; `unfelt` had no scalar box. Captured modality reads named the appropriate box only at 1, 2 and 3; .5, 1.5, 2.5, 3.5 and 10 belonged to none. The actual answer path returned `this \x00\x00\x00\x00` for all eight values: **0/3 trained recognitions**. A modality-box read is not an emitted recognition answer.

With the number omitted, the captured modality inverse restored 1, 2 and 3 from the respective word boxes. With text empty and just the number present, all three full-model probes raised `input reconstruction requires a completed parallel field`. The comparison attempt recorded **0/12 correct trials**: two wrong emitted answers, one missing-candidate `KeyError`, and nine pinned-trial capacity exceptions. The zero lesson then failed on pinned-trial capacity before a measurement could be made; **zero is unmeasured**. This first probe harness did not call `End` after every question, so accumulated retained trials limit what its capacity failures establish; no fixture was resized.

A separate native supervised-question certificate attempt used the same lesson and called `End` after each presentation/question. Seven supervised question updates completed, then a lesson presentation failed in `ClauseRow.write_clause` → `Memory.link_estimate_observation` with `expectation link must end at an external observation`, before any recognition evaluation. Its receipt is `scalar-supervised-certificate.json`. This does not establish trained recognition. **Certificate 13 is not satisfied**, despite the passing box, definition-trust and constructed perception-binding unit cases.

**Million-row forgetting.** The new pass completed on 1,048,576 rows in **72.425 s**, deleting 262,144 and retaining the requested 786,432. Cache preparation took **40.507 s** separately. The synthetic store has identical positive-only content, one stream, no text or semantic dependencies; it is a favorable indexed case. The before arm uses HEAD's old `Forgetting.py` planner against the same current LTM0 substrate, not a fully isolated old runtime; this distinction limits any comparison. The old planner did not finish within its 900 s guard (901.021 s including termination; setup 1.379 s). No completed before-pass time or speedup ratio is claimed.

The first before/after harness construction accidentally inherited a stream capacity of 1,048,576 and attempted a quadratic stream buffer; both were memory-killed before a pass. The corrected synthetic setup explicitly uses one stream. The first scalar observer used a nonexistent `AnswerConstruction.available` property and was interrupted before answers were collected; its corrected observer reads the emitted texts. These are harness corrections on the same frozen implementation, not hidden model retries. Original harnesses, corrected harnesses and their hashes remain in the receipt.

**Standard sweep, completed once.** The exact command `env -u RUN_TRAINED -u RUN_SLOW make test` completed all **8,096/8,096 cases** in **2,154.558 s** (35 min 54.6 s): **7,595 passed, 177 failed, 323 skipped, 1 expected failure**. Nothing was deselected. The native runner returned 1 for test failures; make returned 2. There were no incomplete cases and no compile-cache retries. Both stopping cases passed in this run as well. The existing standard limits were retained: 28 GiB aggregate test memory, three ordinary workers, 1,800 s per worker and 10,800 s for the suite.

The last symbolic reconstruction case also failed **0/4** in this sweep: `hello world` → `hello hello`, `hello there` → `hello hello`, `loving world` → `hello loving`, `loving there` → `hello loving`. This is separately preserved from the frozen targeted draw, whose emitted texts differ. The 177 failures span 89 files. Frequent observed symptoms include 32 cases rejecting the `query` capability, 21 mentioning absent `numeric_percepts` state, unported mock readouts without `field_readout`, retired time-query arguments and LTM1 column assumptions, compose paths with no legal exploration alternative, and changed sparse-fold/gradient expectations. These are diagnostic groupings, not a claim that all failures have been independently reduced to their producers.

The native result and logs remain at `output/tests/20261112-064330-566771/`. The round's receipt also contains `full-sweep-result.json`, `full-sweep-outcomes.json`, `full-sweep-failures.json` (every failed node and full report), `measurement-summary.json` and the per-run/per-sentence measurement records. `commands.json` contains exactly one `full-sweep` invocation. The original offer's documents and patch remain sealed; final prose is recorded separately.

**Final identity.** After the sweep, all 1,071 implementation inputs still match `frozen-source.json`, with compact-sorted-JSON source hash `607ab7f8c5daf5a613ee8478aa6776b3ef96056607b482f7e351acafefe6bc37`. `returned-source.json` records the matching snapshot. The port inventory contains 118 complete before/after texts, and the post-port configuration census contains 41 configurations with no parse errors. No source edits or second sweep followed the offer.


### 8.5 Disposition for review

**Returned through Alec for Claude's review; not ready to land.** The numeric architecture and focused mechanism checks are implemented, but this candidate does not satisfy every certificate. Certificate 11's symbolic reconstruction and certificate 13's full scalar recognition/omission fail. The whole-path lesson stops at a missing query-capability grant. Scalar supervised acquisition additionally encounters an expectation/observation ownership error. The standard sweep's failures are retained in its reports, not repaired after the offered freeze. Todo authorized the standard suite after item 1.8, so no baseline full sweep was run in this round; the inventory alone does not establish which failures predate it.

The measurement also has explicit gaps: only a prefix of the lesson ran; its passive field-agreement observer captured no comparable field; zero was not measured; and the old million-row forgetting pass timed out. Box lookup, constructed binding and a fast synthetic pass do not substitute for the missing full-model behavior. A subsequent repair round needs to address those producer failures and complete the deficient certificates/measurements under its own recorded source. This return does not retire their bars or request acceptance of the failures.

No implementation, data or test changed after the freeze, and no commit or push was made for item 1.8. The local receipt retains the original offered documents and patch, with the final return's prose recorded separately in `returned-documents.json`.

## 9. Review of the return (Claude, 2026-11-12)

### 9.1 What was checked

- *The returned source.* `test/bounded_tests.source_snapshot` recomputed over
  the main checkout gives `607ab7f8c5daf5a613ee8478aa6776b3ef96056607b482f7e351acafefe6bc37`
  over 1,071 inputs, as §8.1 and §8.4 state; nothing changed after the freeze.
- *The sweep* (`output/tests/20261112-064330-566771`): 8,096 cases, 7,595
  passed, 177 failed in 89 files, 323 skipped, 1 expected failure; both
  stopping gates passed.
- *Attribution.* Compared by node with the last complete sweep before this
  round (`output/tests/20261111-184104-806f03`, the whole-path candidate,
  126 failures): of the 177, **160 passed then**, 3 were not in that sweep,
  and 14 failed then too; 108 cases that failed then pass now. The 14:
  `test_grammar_object_leaves.py::test_grammar_resolves_each_word_to_its_full_object_code[MM_grammar.xml]`;
  `test_item55_mechanisms.py::test_clock_independence` and
  `::test_offset_independence`;
  `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers`;
  `test_reading_curriculum.py::test_supplied_text_reader_preserves_shared_work_spent_during_sentence_scoring`;
  `test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully`;
  `test_router_fires_per_word.py::test_serial_words_use_the_shared_operation_layer[both]`
  and `[per-word]`;
  `test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_runepoch_runtime_split_no_raise`,
  `::test_store_truths_idempotent_clear_then_record` and
  `::test_store_truths_records_truths`;
  `test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`;
  `test_stage_caches_retired.py::TestReverseRoundtripStillCompletes::test_terminal_stm_snapshot_available_after_forward`;
  `test_stm_recon_from_cleared_cache.py::test_cache_clears_and_chart_generate_rederives_from_stm`.
- *The failures by symptom:* 32 `thought capability does not admit
  'query'`; 17 `numeric_percepts` missing (13 on `BasicModel`, 4 on a test
  stub); 12 `compose has no legal alternative at the selected exploration
  round`; 12 mocks without `field_readout`; 6 removed `concept_ids` or
  `_concept_ids`; 4 `ThoughtLTMCapability.query()` given the retired `when`;
  3 `TrustGates.measurements()` without `occupations`; 3 numeric inputs that
  disagree with their declared dimensions; 2 reads of the removed LTM1
  `percept` column; the rest are single.
- *Certificate 11.*
  `test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`
  passed in the 2026-11-11 sweep and passes on `9f7923d67f`'s code (run
  2026-11-12, 545 s); it fails 0/4 on the returned source, twice. It is a
  regression of this round.
- *Three defects found while the documents were checked against the code*
  (§9.3 D7 to D9), read in the returned source; none was run.

### 9.2 Verdict

Not accepted yet. The stopping gates pass and the architecture stands: the
field logic, examination, the numeric modality, the gates and the cheaper
forgetting pass. But certificates 11 and 13 fail, the lesson arm stops on a
producer defect after 25 of 160 sentences, and 160 cases that passed before
the round fail after it. A round's own repairs complete it before the next
starts (Alec, 2026-11-12), so §10 completes item 1.8.

### 9.3 Defects

Introduced by this round:

- **D1, the `query` permission.** `ThoughtFaces.query` now calls
  `context.ltm.query(...)`, but the `query` descriptor's `method_grants`
  (`bin/Queries.py`, `ThoughtExecutorDescriptor('query', ...)`) still grant
  only `best_match` and `resolve_description`, and `_ThoughtCapabilityView`
  refuses any other name: 32 failures, and the lesson arm's stop. The same
  edit stopped passing the operand `references` with which `best_match`
  scoped its cover and timeline reads.
- **D2, `numeric_percepts`.** `BasicModel._forward_per_stage` reads
  `self.numeric_percepts`, which only `create_from_config` assigns; a model
  built another way never has it (13 failures), and test stubs lack it (4).
- **D3, a departure without an alternative.** `choose_operation` raises
  "compose has no legal alternative at the selected exploration round" in
  serial reading and in the shared-stack contract tests (12 failures). The
  cause was not looked for.
- **D4, certificate 11.** XOR_grammar's symbolic reconstruction returns
  `hello` and NUL bytes (0/4).
- **D5, certificate 13.** The answer path emits `this` and NUL bytes; a
  number beside empty text raises `input reconstruction requires a completed
  parallel field`; the supervised lesson fails in `ClauseRow.write_clause` ->
  `Memory.link_estimate_observation` ("expectation link must end at an
  external observation").
- **D6, the lesson observer** looks for the generated field after trial
  restoration, so no field was compared.

Present before this round and still in the returned code:

- **D7, `non` has two rules.** On a meaning block,
  `MeaningCodes.compose('non')` clears the for lane and keeps the against
  lane, which is the wording of the lanes decision (Alec, 2026-10-07: "`non`
  clears the for lane"). `FieldLogic.reduce('non')` and `NonLayer` in its
  pole form withhold both lanes (§3.2). The two differ only when the operand
  carries evidence against: `non` of a negated concept leaves the negation
  standing under the first and gives the unknown under the second. The
  definition of 2026-09-28 is that `non` "sets the expressed pole to zero, is
  non-affirming", acting "on the pole a concept expresses"; for a negated
  operand the expressed pole is the against lane. *Decided (Alec,
  2026-11-12):* "'non clears the for lane' is probably a typo; I believe
  that non- removes both positive and negative evidence, while not swaps
  the two types of evidence." One rule everywhere: `non` withholds both
  lanes.
- **D8, walk credit with `separateChooser=true`.** `objective_parameter_groups`
  maps the objective `chooser` to the reconstruction owners, while the
  generate policy is output-owned (the same method requires `generate <=
  output`). `output.walk_credit` is filed under `chooser` and
  `reconstruction.walk_credit` under `reconstruction` (`bin/WholePaths.py`),
  so in the separate option no walk departure's credit reaches the generate
  policy that drew it. The default is unaffected: `GradientProjection.sides`
  places the terms on R and A, and every parameter receives both gradients.
- **D9, the surface rule's head.** On a meaning block
  `GrammarLayer.compose_blocks` gives `LaneKernel` the operand
  `meaning_head_role` as its head, 0 unless a layer sets it, and the kernel
  returns its head, adjusted by the other operand (unchanged at
  initialization). `SurfaceLayer` does not set it, so marker plus content
  composes to the marker's meaning, although the rule "projects its right
  operand" (its docstring), declares `semantic_operand = 1`, and is written
  `head="argin2"` in the grammar; the closing (`ClauseJournal`) reads the
  content. `PrepositionLayer` has the same two declarations (kernel head 0,
  `semantic_operand = 1`); whether its kernel head should be the preposition
  or its object is not recorded anywhere.

Alec's decisions of 2026-11-12, put to him with this review:

- **D10.** Delete the unused attention objective: `AttentionObjective.field_cost`
  is called only by `SentenceAttention.costs`, which nothing in the model
  calls since the whole paths train attention by R + A.
- **D11.** `IdeaEvaluation` (a thought's result with no declared relation to
  the question is bound to it when one kernel and one context explain both,
  within $10^{-4}$): "It sounds like a poor and brittle evaluation, so I'm
  inclined to remove it."
- **D12.** An `LTM0` entry holds every order, a question is matched order
  by order, and its results are intersected (§3.12, which amends §3.8).
  Being refined with Alec; not in the text of §10 yet.
- **D13.** The time an `LTM0` entry was written is named `time`, not
  `.when` (§3.12).
- **D14.** The determiner lowers its argument and selects one member or
  several (§3.13): decided in direction by Alec, 2026-11-12; next round.
- **D15.** Relation rows stay and no query is built over them (§3.12;
  Alec, 2026-11-12).
- **D17.** The feeling's bounds are perceptual parts and wholes knit at a
  basic percept, and the two modalities share their code (§3.10, "The
  bounds are perceptual"; Alec, 2026-11-12; §13.1).
- **D16.** Long-term memory is two tables; reads that go through every row
  are replaced, caches are declared, and `LTM1.store_order` goes (§3.14;
  Alec, 2026-11-12).

### 9.4 Accepted as built, with notes

- The ten numeric XOR trainings are identical because the fixed axes and the
  reader's initialization are deterministic, and no seed was chosen: one run
  says what ten say, and the requirement is met.
- The million-row forgetting pass completed in 72.4 s; the old planner did
  not finish in 900 s, so no ratio is claimed.
- The compiled generator's nested-conditional tangent error, which a scalar
  fixture met and the eager-reading fixture avoids, belongs to item 1.
- The symbolic configurations survive as `MM_text.xml`, `MM_text_fixture.xml`
  and `XOR_symbols.xml` at their capacities, as §3.9 asks.

## 10. Codex text: complete item 1.8 (Claude, 2026-11-12; for Alec to paste)

~~~
Complete item 1.8 (doc/plans/2026-11-11-item1-8-last-architecture-round.md).
Your return (§8) is reviewed in §9: not accepted yet, and this round
completes it. Work in the main checkout on the returned source
(607ab7f8c5...). Commit and push nothing until Alec says so.

1. Attribution (§9.1): 160 of the sweep's 177 failures passed in
   output/tests/20261111-184104-806f03 and are this round's unless you show
   otherwise; 14 failed there too and are listed in §9.1; 3 were not in that
   sweep. Confirm the 3 and say for each of the 14 whether this round
   changed its cause.
2. Repair what this round introduced (§9.3 D1 to D6). Find each cause
   yourself and repair it where it arises; a mechanism added only to pass a
   gate is not a repair.
   a. D1: each thought face is granted exactly the LTM methods it calls.
      Decide whether the content query keeps the operand-reference scoping
      that best_match had, and record why.
   b. D2: numeric_percepts is read on models that were never given it.
   c. D3: "compose has no legal alternative at the selected exploration
      round".
   d. D4: certificate 11, which passes on 9f7923d67f's code.
   e. D5: certificate 13's three failures.
   f. D6: the observer reads the generated field before trial restoration.
3. Port the tests that call what this round removed on purpose (concept_ids,
   field_readout on mocks, time-only queries, the LTM1 percept column, the
   new argument of TrustGates.measurements, and the like). Keep every
   assertion whose contract still holds; a test whose contract the round
   retired is retired by name, with the 1.8 decision that retired it.
4. Repair §9.3 D7 to D9, each with a certificate that fails before the
   repair and passes after it.
   a. D7: non withholds both lanes everywhere (MeaningCodes.compose('non')
      joins FieldLogic.reduce('non') and NonLayer's pole form). Certificate:
      for operands expressing for, against and both, not and non give the
      same lanes through MeaningCodes, FieldLogic and both NonLayer
      representations.
   b. D8: with separateChooser=true, a departure's score-function credit
      trains the policy that drew it: both walk credits reach the generate
      policy, compose and thought credits reach their selection layers, and
      no other parameter. Certificates: in the default mode one sentence's
      parameter gradients are the same before and after the change; in the
      separate option a walk departure moves the generate policy and no
      reconstruction-owned parameter.
   c. D9: on a meaning block the surface rule's kernel takes as its head
      the operand it projects, the right one, in forward and in reverse.
      Certificate: at initialization, marker plus content composes to the
      content's meaning, and the reverse recovers the content from the
      parent and the marker. For the preposition change nothing: report
      which operand its kernel takes as head, what the closing reads, and
      your recommendation for Alec.
5. Alec's decisions (§9.3 D10 and D11). Delete AttentionObjective.field_cost,
   SentenceAttention.costs and the tests of them, keeping validate_floor and
   tolerate_heterogeneity, which live attention uses. Retire IdeaEvaluation:
   the module, the isomorphic branch of ThoughtStream and its test, so that
   thought binding relies on declared relations only.
6. Then rerun certificates 11 and 13 and the whole-path lesson arm (the
   explore-1.xml configuration of your return, 160 sentences, with the
   repaired observer), and one default sweep:
   env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected. A failure
   that predates the round is listed, not repaired, unless §9.3 names it.

Rules: §3 and §7 of the specification. No seed is chosen so that an
assertion passes; no fixture is resized; legacy paths are deleted, not
kept; no language is named in code. Continue without stopping and record
each decision you take for Alec; stop only for a design question, written
for Alec with your recommendation. A failed
test/test_mm_xor.py::TestMMXorConvergence::test_convergence or
test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class
stops the round. The documents are being restructured on the branch
testing, so keep document edits to what this round changes.

Return as §11: the attribution; each repair with its cause and its
certificates; the reruns' numbers written out (receipts stay local and are
not linked); the sweep's result; the failures that remain, with their
producers; and the returned source hash.
~~~

## 11. Completion checkpoint and design question (2026-11-12; through Alec for Claude)

**Not a completed or accepted return.** Work stops at the design question
below, as §10 permits. No commit or push was made. The main checkout's
HEAD is still `9f7923d67f8921ef5c14afdf2fe18c536802018b`; its starting
source was verified as
`607ab7f8c5daf5a613ee8478aa6776b3ef96056607b482f7e351acafefe6bc37`.
The source at this checkpoint, by the same compact sorted
`bounded_tests.source_snapshot` method, is
`ed4c6a3141e2569b3786dacd6e85cbd86b30c83e4b165ac327ea25bcf2a16892`
(1,070 inputs). Local receipts contain the source manifest, patch,
commands, attribution and full before/after text of 68 data/test ports.
They are neither committed nor linked here.

### 11.1 Design question for Alec: the identity of a generated scalar class

**Recommendation:** retain a bound object's reference as an operand of
the existing generation walk. When that operand is emitted as a terminal,
use its unique word definition through the interpretation inverse to
realize its form. Keep the stimulated semantic field for semantic
scoring. Do not infer the selected identity from cosine, invent another
numeric code, copy a desired answer, or introduce a second answer path.

The choice is whether generation may retain that bound reference, or must
continue to receive vectors alone. This changes the generation operand's
contract, so it is put to Alec before implementation.

The concrete evidence is independent of whether the question's grammar
has learned. In the retained diagnostic model, distinct scalar objects
185 (named `one`) and 197 (named `this`) have exactly equal 360-coordinate
codes: only coordinate 95 is nonzero, at 0.4093838930130005. Each has its
own unique word definition. A question word also heard beside a scalar
receives its own box under §3.10, so these are legal distinct identities,
not a collision to remove by changing the lesson. Supplying the correct
class's concluded value directly to `reverseOutput` still emitted four
NUL bytes. This probe isolates realization from grammar acquisition.

The identity survives in `AnswerDerivation.answer_meanings`, but
`_materialize_answer_idea` and `generate` pass its vector and occupancy.
`readback_scores` uses perceptual cosine; scalar-axis values therefore
share a direction. `OutputOccurrences.generated` chooses a lexical row
for the bracket length, then stimulates the emitted point;
`FieldReverse.reverse` reads that field's percept features. A scalar-only
class has no spelling features, and that inverse skips numeric axes when
producing bytes. Thus two issues meet: loss of a known bound identity and
the missing interpretation inverse from a named numeric object to its
word. A vector-only interface cannot distinguish two equal codes, even
with perfect training. §5.2's statement that saying was already handled
does not hold for this newly admitted object representation.

The extended diagnostic was interrupted after twelve supervised updates
to inspect this producer; it is not a certificate or an increased lesson
budget. Its interrupted trial required clearing diagnostic STM leases
before inspection; that interruption artifact is not attributed as a
production defect. No seed was selected for a passing outcome.

### 11.2 Attribution of the 177 failures

The comparison confirms **160 failures whose nodes passed before, 14
whose nodes also failed before, and 3 absent nodes**. The 160 remain
attributed to this round. The three absent nodes are:

| Returned node | Earlier counterpart and result |
| --- | --- |
| `test_compose_pair_driver.py::test_complete_paths_train_in_one_step_and_commit_once` | `test_pair_trains_twice_and_commits_once` failed: one update versus its two-update expectation. Whole-path combined training retired that expectation; the current one-update/one-commit assertion is preserved. |
| `test_joint_objectives.py::test_real_runbatch_steps_sentence_trials_then_batch_only_objectives[MM_text.xml-expected0]` | The `MM_xor.xml-expected0` parameter passed. The text fixture was renamed during the numeric XOR port. |
| `test_two_mode_dispatch.py::test_native_carrier_contract_replaces_traversal_switch[MM_text.xml-False]` | The `MM_xor.xml-False` parameter passed. Same fixture port. |

For each of the fourteen earlier failures, this is the causal comparison,
not an exemption based on the old failure count. Names below are under
`test/`.

| Earlier failing node | Did this round change the failing cause? |
| --- | --- |
| `test_grammar_object_leaves.py::test_grammar_resolves_each_word_to_its_full_object_code[MM_grammar.xml]` | Yes: missing sentence prediction became an illegal compose departure. Legal vector-grammar fixture port repairs the current cause. |
| `test_item55_mechanisms.py::test_clock_independence` | Yes: missing prediction became missing concept-index access. The occurrence/relation namespace repair and sentence-reference port address it. |
| `test_item55_mechanisms.py::test_offset_independence` | Yes: the same changed cause and repairs as the clock case. |
| `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` | The general cause remains a stale reader inventory. The round added `ScalarPercepts.join`, now inventoried. The older `Spaces._radix_part_events` versus `_radix_code_events` discrepancy is reproduced on HEAD and left unrepaired. The earlier sweep also named obsolete sentence-field readers. |
| `test_reading_curriculum.py::test_supplied_text_reader_preserves_shared_work_spent_during_sentence_scoring` | Yes: the earlier missing `lesson_cross_entropy` became strict thought chronology failure. Mixed-trial thought selection repaired its current producer; the curriculum file now passes. |
| `test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully` | Yes: the earlier missing prediction became an unstaged reconstruction bank (HEAD independently exposed the namespace error). The current shared query/namespace repairs make this case pass. |
| `test_router_fires_per_word.py::test_serial_words_use_the_shared_operation_layer[both]` | Yes: missing prediction became no serial sweep under the field grammar. The symbolic fixture now explicitly declares vector operations. |
| `test_router_fires_per_word.py::test_serial_words_use_the_shared_operation_layer[per-word]` | Yes: the same changed cause and fixture port. |
| `test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_runepoch_runtime_split_no_raise` | Yes: missing prediction became an illegal compose alternative. The vector fixture port addresses it. |
| `test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_idempotent_clear_then_record` | Yes: the same changed cause and port. |
| `test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_records_truths` | Yes: the same changed cause and port. |
| `test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row` | No: observed native feature row `{0}` versus expected `{4, 402}` is unchanged in the earlier sweep, HEAD and returned source. Left unrepaired. |
| `test_stage_caches_retired.py::TestReverseRoundtripStillCompletes::test_terminal_stm_snapshot_available_after_forward` | Yes: missing prediction became no serial records under the field-only fixture. Explicit vector grammar restores this test's intended pipeline. |
| `test_stm_recon_from_cleared_cache.py::test_cache_clears_and_chart_generate_rederives_from_stm` | Yes: missing prediction became an empty generate catalogue. The vector fixture port addresses it. |

The isolated HEAD comparison of these fourteen was ten passes and four
failures in 133.17 seconds. The last selection of surviving original
failure names on the evolving completion source was 153 passes and eight
failures (161 reports). Six of those eight subsequently passed after the
mixed-trial and simple-model repairs; the two unchanged failures are
listed above. Renamed ports and the long symbolic certificate were
checked separately. This selection is not a substitute for a full sweep.

### 11.3 Repairs, causes and decisions so far

- **D1:** each thought face's grant now names the methods its executor
  calls; the unused infer grant was removed. The fourteen-face grant
  certificate passes. Content queries retain operand-reference scoping:
  a cue may rank within the referenced cover/timeline region, but must
  not broaden it or turn unverified material into evidence. Query uses
  the new content recall and reinstatement API; ask retains its actual
  `best_match` call and grant.
- **D2:** `numeric_percepts` is initialized to `None` in the common model
  constructor, including models not made by `create_from_config`.
  Minimal test hosts declare their used capability. The capture branch
  tests aligned binding before accessing aligned grammar phases, so dense
  simple models do not acquire an unrelated required attribute. The
  four spacetime cases pass after this correction.
- **D3:** fixtures that still test serial vector execution used the
  newly field-only operator names, leaving no legal departure. They now
  declare the vector operations their assertions exercise. A separate
  native arity defect treated `bind` as a binary stack pop; lesson/replay
  arity now agrees with its unary stack contract and field-side operand.
  The deadline test keeps its original bounds and checks legal unary
  steps followed by the required binary reduction.
- **D4:** stimulation treated shared form as conceptual entailment and
  did not read the separate meaning lanes correctly. `FieldProjection`
  now computes candidate coverage independently on for/against meaning
  blocks, using form only when a candidate has no meaning block. The
  common-form/different-meaning regression fails before and passes after.
  One unchanged symbolic training passed its four-of-four grammar inverse
  assertion in 634.09 seconds; its separate parallel-field audit was only
  one of four. Neither is the still-required final rerun.
- **D5, partial:** scalar-only input now enters the native field close
  even with no word cache; active scalar rows have the correct percept
  span shape. Acquisition moves from staging to the kept close, so
  promotion cannot retire an identity still needed by that trial.
  Promoted identities retain their order; repeated views update the
  definition's evidence. An empty intuition no longer closes an open
  question. A perceived class supplies its own content, independently
  of the current event's content. A kept thought's concluded meaning is
  handed to the public answer path; a completed reference question emits
  its bound value rather than all the question's roles. Direct before/
  after cases cover the empty-intuition, class-content and bound-answer
  defects. Recognition remains unsatisfied for the producer in §11.1.
- **D5, expectation-link eligibility:** `Memory.kind_of` mistook an
  external mention of a cover identity for a new document-cover entry.
  It now requires the native cover occurrence's address as well as its
  identity. The real cover/mention/expectation-link regression failed
  before and passed after (2.56 seconds). This reproduces an introduced
  eligibility defect; the original stochastic failing row was not saved,
  so it is not claimed to prove that row's sole cause.
- **D6:** the observer captures the generated field inside reconstruction,
  before restoration, then retains the winning stream's comparison.
  A diagnostic arm completed all 160 sentences in 776.459 seconds.
  `non` opened at sentence 45, `ask` at 66; exempt operators were open at
  zero. There were 123,891 counted compound fallback operands by the
  final sentence. The final sentence's comparison had 370 coordinates,
  maximum absolute difference 0.24024587869644165 and mean difference
  0.013655873946845531. Per-sentence records are local. A final arm on the
  completed source remains required.
- **D7:** `MeaningCodes.non` now withholds both lanes. All six combinations
  of not/non and for/against/both agree through MeaningCodes, FieldLogic
  and both NonLayer representations. The against and both non cases
  failed before; all pass after.
- **D8:** separate ownership assigns both walk score credits to the
  generate policy, and compose/thought credits to their selection
  owners. Both walk update certificates failed before and pass after;
  reconstruction-owned parameters do not move. A real default-mode
  sentence's 22 nonempty parameter gradients are exactly equal before
  and after, with the initial random streams captured for comparison,
  not selected for success. Compose/thought owner certificates pass.
- **D9:** SurfaceLayer declares meaning head 1. Its initialized forward
  and reverse certificate failed before and passes after. Preposition is
  unchanged: its kernel takes operand 0, while semantic closing reads
  operand 1. Recommendation for Alec: give it head 1 too unless the
  intended meaning is explicitly the marker's rather than its object's.
- **D10/D11:** deleted `AttentionObjective.field_cost`,
  `SentenceAttention.costs`, their unused observed-field constructor input,
  `IdeaEvaluation` and the isomorphic binding branch. Live
  `validate_floor` and `tolerate_heterogeneity` remain. Binding proofs now
  require declared relations.

Further introduced producers found while porting were repaired where
they arise: MemberRead separates local occurrence indices from relation
locations; numeric perception constructs the active microbatch before
use; the weighted false power has a finite derivative without a clamp
artifact; percept definition folding takes minimum for conjunction's
for-lane and maximum for its against-lane; reconstruction reporting reads
the retained public field inverse. Mixed thought trials now retain the
largest selected identifier frontier and select the active prefix of
reserved STM tensors and lists. The latter two regressions each failed
before and pass after; the combined native chronology/whole-path set
passed 42 cases in 33.06 seconds.

### 11.4 Ports and retirements

Ports replace removed concept-ID readers and mock `field_readout` methods
with the field contract, supply trust occupations, replace LTM1 percept
stamps with sentence references/store order, and replace external
time-only queries with content cues. They preserve stream isolation,
work accounting, inverse fidelity, gradient ownership, cache lifetime,
truth/evidence and one-kept-close assertions. Given-answer tests now
provide both content and declared occupancy; an unresolved input is not
silently treated as a supplied answer. A trial-mask test selects its
compose departure explicitly so it continues to test two compose trials
under the broader whole-path departure menu, without selecting a seed.

Retired by name under Alec's D10 decision:
`test_attention_objective.py::test_outside_residual_is_the_full_observation`,
`::test_real_reconstruction_and_floor_count_each_valid_item_once`, and
`::test_undefined_inverse_coordinates_have_no_inside_charge_or_gradient`.
The field-cost assertion in
`test_native_percept_inverse_final.py::test_unsupported_inverse_keeps_its_learning_gradient`
is retired by D10; its unsupported product-inverse assertion remains as
`test_product_inverse_is_not_certified`.

Retired by name under D11:
`test_idea_evaluation.py::test_common_context_evaluates_distinct_inputs_without_mutating_kernel`
and `::test_equal_outputs_without_input_equality_do_not_supply_a_proof`.
The positive isomorphic-evaluation fixture in thought binding is replaced
by a declared-equality proof, and the negative test still rejects
isomorphism without a relation.

Retired by §3.9's removal of the concept-ID answer head:
`test_understanding_readout.py::test_named_concept_head_keeps_its_identity_carrier`.
Its replacement coverage tests the live final-field reader. Other renamed
cases are ports, not dropped assertions: the four time-only recall cases
and STM content-cue case follow §3.8; the XOR dataset case follows §3.9's
four numeric rows; evidence-fold and intuition cases follow §§3.2/3.6;
answer-gradient and ownership cases follow the final-field/shared-trunk
contract. Their complete old and new texts remain in the local ledger.

### 11.5 Verification limits and work remaining

Both stopping gates most recently passed: two cases in 3.79 seconds.
At the checkpoint, the completion/scalar/answer/attention/spacetime set
passed 63 cases with one standing skip in 47.62 seconds.
The document-link check passed 1,014 cases in 126.86 seconds.
The latest prepared-answer file passed seven cases with one standing skip
in 36.44 seconds. The preceding answer/attention/spacetime selection had
37 passes, one failure and one skip; its failure was the remaining
given-answer occupancy fixture, subsequently ported and covered by those
seven passes.

The six-view scalar box/direct-binding and scalar-only full-model cases
pass, but they do not establish the required full question-to-word
recognition certificate. An earlier eighteen-update supervised probe
completed and recognized zero of three values. An extended probe also
exposed an unresolved multi-role thought write without a semantic chain
or checked binding after 26 updates; the later instrumented diagnostic
did not reproduce it before interruption. Its producer remains to be
traced, not hidden by weakening the writer's guard.

**No final freeze, final certificate 11/13 rerun, final lesson rerun or
default sweep has been performed for this completion.** In particular,
the required `env -u RUN_TRAINED -u RUN_SLOW make test` has not been spent
on an incomplete source. It must run once, with nothing deselected,
after the design question and remaining D5 work are resolved. Remaining
known failures are the scalar realization/binding path above, the
unresolved multi-role thought writer, and the two reproduced older
inventory/native-feature failures in §11.2. No claim that all 160
introduced failures are cleared is made before that sweep.

The document acquired §3.12 and D12/D13 during this run. Their region-mask
query and LTM0 `time` changes are preserved in the document; they were
not silently implemented beyond the pasted D1–D11 assignment. Their
inclusion in this completion was also put to Alec.

## 12. On the question of §11.1 (Claude, 2026-11-12; decided by Alec)

Read against the code, §11.1 is right in what it reports.
`OutputOccurrences.generated` chooses a row of the bank for each emitted
word by `readback_scores` on the word's vector, uses that row for the
length of the word's place only, and makes the field by stimulating the
vector; `realize` then spells from the field's form features. An object
over the scalar has no form features (§3.10: its code is its place on the
scalar), so nothing is spelled. `Models._materialize_answer_idea` hands
generation the answer's vector and occupancy and not the object the
answer was found to be.

Nothing has been forgotten when this happens, and no store's retention is
in question: the object is known at the moment of answering and is not
handed on.

Decided, Alec, 2026-11-12: "generation may carry the answer object's
reference so the object is said by its one word."

The recommendation that was put to him: accept §11.1. §2.11 and §3.10 already
say an object is said by its one word (`InterpretLayer.lexical_form`,
through `word_surface_for_row`); the recommendation carries the object's
reference to the place where the word is emitted, so that this rule can
act. Comprehension already carries references beside vectors
(`ConceptualMeaning.role_refs`); generation would do the same. One
addition: where no reference is bound and a bare value is to be said, the
word is the one whose box holds the value (§3.10, "an omitted word is
restored from the number through the box that holds it"), chosen by the
stimulated field and never by cosine, and then said the same way.

## 13. Codex text: finish item 1.8 (Claude, 2026-11-12; for Alec to paste)

~~~
Finish item 1.8 (doc/plans/2026-11-11-item1-8-last-architecture-round.md)
from your checkpoint of §11 (source ed4c6a3141...). The work of §11.3 and
§11.4 stands as you describe it and is reviewed at the final return.
Commit and push nothing until Alec says so.

Alec's decisions since your checkpoint (2026-11-12):
- §11.1 is accepted: "generation may carry the answer object's reference
  so the object is said by its one word" (§12).
- §3.12 has been rewritten and no longer describes a region mask. It,
  §3.14 and D13 are in this completion (§9.3 D12, D13, D16).
- Not in the round: the determiner (§3.13), the time result of recall
  (§6), and the preposition's kernel head, which waits for Alec and stays
  unchanged.

1. Saying an object (§11.1, §12).
   a. A bound object's reference travels as an operand of the existing
      generate walk, from AnswerDerivation.answer_meanings through
      Models._materialize_answer_idea to the terminal that
      OutputOccurrences.generated emits. A terminal that carries a
      reference is said by the one word that names its object
      (InterpretLayer.lexical_form, through word_surface_for_row). The
      stimulated field still scores meaning. No identity is taken from a
      cosine, no code is invented, no desired answer is copied, and there
      is no second answer path.
   b. A terminal with no reference whose point lies on the scalar is said
      by the word whose box holds its value (§3.10), chosen by the
      stimulated field. Report what is emitted for a value no box holds.
   c. Certificates, failing before and passing after: two objects with
      equal codes and different words are each said by their own word when
      their reference is given; certificate 13 whole, its recognition and
      its omitted-word parts included.
2. Finish D5, and trace the multi-role thought write that had no semantic
   chain or checked binding (§11.5) to its producer. The writer's guard is
   not weakened.
3. Recall (§3.12, D12).
   a. episodic_concepts and episodic_values gain an order axis, row by
      order by slot: one slice for each order the concept allocator
      declares (owner._order_caps()), fixed at construction.
      EpisodicMemory.select keeps the strongest episodicConcepts readings
      of each order; a concept's order is owner._row_order, as
      EpisodicMemory.stage reads it. A checkpoint of the old shape loads
      its entries at order zero, with a warning, as the retired percept
      column does.
   b. The match. The question's content at an order is its own field at
      that order (stimulate). Nothing is carried from one order to
      another, in the question or in the entry. For each row t of the
      window and each order k at which the question holds anything,
      M[t,k] = <q_k, e_tk> / <q_k, q_k>, not above one, for lane with for
      lane and against with against, never netted. It is computed for the
      whole window at once, with no loop over rows; a gather of the
      question's content at the stored references is that product. An
      order at which the question holds nothing is left out, not counted
      as zero.
   c. The relevance of a row is the intersection (FieldLogic) of its
      matches. Rows of relevance zero are not returned. The rest are
      sorted by relevance, the newer first among equals, and the best
      max_records are brought back by EpisodicMemory.reinstate, each
      concept at its order. Operand-reference scoping stays as you decided
      it under D1. Build the intersection so that a further result over
      rows joins it unchanged; the time result is not built in this round.
   d. EpisodicMemory.query's loop over rows, its meet_ratios for each row
      and the index from concept to rows (_episodic_index, index_row,
      rebuild_index) go.
   e. Certificates 15, 16 and 18 of §3.12.
4. Long-term memory is two tables (§3.14, D16).
   a. LTM0.cued_rows reads records only for its explicit references, as
      built, and for the rows item 3 returned. It no longer takes the
      newest rows of the window, or every relation row, as candidates. Its
      match by role on the rows it reads stays as built. LTM1.store_order
      goes with its reader. List every test whose result changes because a
      row is no longer reached, with the row and the reason; the wider
      candidates are not restored to pass one.
   b. ThoughtLTMCapability.relation_evidence walks no table: with both
      ends given it is LTM1.find, with one end given the rows that hold
      it. rows_referencing, relation_evidence, find and the part and whole
      lists are uses of one read, the rows that hold a given reference.
   c. LTM1.add keeps _pairs, _parts, _wholes and _references up to date,
      so that a write followed by a read rebuilds nothing. A forgetting
      pass may rebuild them once.
   d. Each store declares its caches in one list, and a cache is not saved
      in a checkpoint; an old checkpoint that holds one loads, with a
      warning. Start from §3.14's first table and confirm each field
      reader by reader. A field that cannot be rebuilt from the two tables
      moves to the second list, with your reason. The fields of the second
      table stay, listed in one place in each store, each with its use.
   e. Certificates 19, 20 and 21 of §3.14. Report the stored size of one
      LTM0 row and of the store at the largest constructed capacity,
      before and after this item and item 3.
5. The name (D13). The time an LTM0 entry was written is named time in
   code and documents: entry_when, RowTime.stamp, read_stamp, write_stamp
   and any other name of it. The layout index when_start names the
   percept's band and stays, and `.when` keeps naming a percept's time and
   a chunk's. List each old and new name. Certificate 17.
6. Then, on the finished source: certificates 11 and 13, the whole-path
   lesson arm (the explore-1.xml configuration of your return, 160
   sentences), and one default sweep,
   env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected. A failure
   that predates the round is listed, not repaired.

Rules: §3, §7 and the rules of §10's text. Find each cause yourself and
repair it where it arises. No seed is chosen so that an assertion passes;
no fixture is resized; legacy paths are deleted, not kept; no language is
named in code. Continue without stopping and record each decision you take
for Alec; stop only for a design question, written for Alec with your
recommendation. A failed
test/test_mm_xor.py::TestMMXorConvergence::test_convergence or
test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class
stops the round. Keep document edits to what this round changes.

Return as §14: each item with its cause and its certificates, failing
before and passing after; the tests of 4a; the cache table as confirmed;
the renamed names; the stored sizes; the reruns' numbers written out
(receipts stay local and are not linked); the sweep's result; the failures
that remain, with their producers; and the returned source hash.
~~~

### 13.1 Addendum to the text of §13 (Claude, 2026-11-12; for Alec to paste)

~~~
Addendum to the text of §13, item 1 (Alec, 2026-11-12; §3.10, "The bounds
are perceptual"). Do it with item 1, before certificate 13 is claimed.

Alec: "Although feelings are bare numbers, they should still be bounded by
perceptual parts and wholes, which are knit together at the level of basic
percepts (the only kind collected by concepts)." And: "Each percept is
categorized against perceptual parts and wholes, whose codes are the
values. So the order zero concept is a generalization over basic percepts.
We should share as much code between the two modalities as possible."

1d. The feeling beside the words is perceived as text is.
   - A felt value is categorized against perceptual parts and perceptual
     wholes whose codes are values: a part it contains, a whole that
     contains it. Its basic percept knits them, by
     BasicPercepts.reserve(parts, wholes) as for text. The one atom for
     the axis that BasicPercepts.numeric_dimension reserves, with a
     placeholder part and no whole, is not the percept of a reading of
     §3.10.
   - A word's concept over the scalar is an order-zero concept that
     collects the basic percepts it was heard with
     (add_concept_feature(row, 'bp', ...)), and generalizes over them: the
     part common to them and the whole that covers them, which are §3.10's
     least and greatest. Its membership of a value is §3.10's, unchanged,
     and comes from the read that gives a text concept its membership.
   - No bound is kept beside the percepts. The low and high of
     ScalarPercepts.state(owner)['boxes'] go; ScalarPercepts.membership,
     field_memberships and the scalar branch of FieldProjection.stimulate
     go with them, and so does whatever else in that dictionary the
     definition's evidence or MTM already holds. Say where each of its
     fields went.
   - Share the text modality's code wherever the two do the same thing:
     categorization against parts and wholes, the basic percept, the
     concept's collection, the read, forgetting, the checkpoint. Report
     which functions both modalities now call and which remain for numbers
     alone, each with its reason. How a part and a whole whose code is a
     value come to exist, and whether a reading's identity is its
     tightest part and whole or all of them, are yours to decide and
     record for Alec.
   - The numeric rows of §3.9 are not changed: there a number is the
     evidence of its axis's concept, and the XOR proofs rest on it. Report
     whether they could take the same path; change nothing.
   - Item 1b then reads: a terminal with no reference whose point lies on
     the scalar is perceived as an input value is, and is said by the word
     whose concept that percept belongs to.
   - Certificates, failing before and passing after: a reading's basic
     percept has a part and a whole whose codes are values; two words
     heard over different intervals have different codes; two heard over
     the same interval have one code and are each said by their own word
     (item 1c); a value is recognized with field_memberships and the
     dictionary's bounds in no source file; certificate 13 whole.

Return it under item 1 of §14, with the shared and the separate functions
listed.
~~~

## 14. Codex completion return (2026-11-13; through Alec for Claude)

**Completion returned for review.** Alec's approved turnover-test port
(§14.8) and the checkpoint fixture's pipeline port (§14.9) pass. The final
default sweep completes all 8,141 selected cases, with two earlier failures
listed in §14.7; both stopping gates pass. Work is in the main checkout,
with no commit or push. The returned source is
`a8c9adced8e2a323038e01f768d36daeef4d6454e405f5ae78aaa8712b5a6290`.
This is the compact sorted `bounded_tests.source_snapshot` content hash,
not a Git commit. Git HEAD remains
`9f7923d67f8921ef5c14afdf2fe18c536802018b`. The incoming checkpoint was
`ed4c6a3141e2569b3786dacd6e85cbd86b30c83e4b165ac327ea25bcf2a16892`.
Receipts remain local; their numbers are written below. Sections 11.2 to
11.4 retain the earlier attribution and repairs. This section records
only the completion's changes and final reruns. The new §3.16 memory
reduction and §3.17 operator extensions are for the next round; §3.18's
two-network work follows them.

### 14.1 Saying and perceptual scalar bounds

Alec answered the order question: the present reading is a basic percept,
referenced as `('bp', id)` in MTM, and its named collection is an order-zero
concept, referenced as `('sym', id)`. Identity does not depend on provenance.
The subsequent amendment to §3.10 defers native question recognition to
`compose_extend` in the next round; certificate 13 in this return excludes
that part. Other order checks and §3.9's numeric rows retain their contracts.

**Saying.** References from `AnswerDerivation.answer_meanings` pass through
`Models._materialize_answer_idea` into the existing generate walk's operand
stack. A retained terminal carries its reference into
`OutputOccurrences.generated`; an inverse operation consumes the parent's
reference rather than assigning it to arbitrary children. The owned reference
identifies the object, and `word_surface_for_row` obtains its unique
`InterpretLayer.lexical_form`. The ordinary grammar inverse says that word.
The terminal's original point still supplies the stimulated field used for
semantic scoring. A zero semantic point with a valid reference also says its
word: multiplying the native word by semantic support had produced NUL bytes.
No cosine supplies identity, and no desired answer supplies the reference or
spelling.

The incoming source failed the three initial saying cases in 4.03 seconds:
equal-code objects with different names, a bare value inside an interval, and
a bare value outside all intervals. They passed after reference transport and
realization in 9.24 seconds; the related output-walk run passed 35 and skipped
two in 28.13 seconds. The zero-semantic-support case failed before in 5.10
seconds and passed in the following 17-test run, 7.50 seconds.

**Perception.** A distinct positive felt value admits an endpoint in the
basic percept owner's fixed-capacity value table. Its part and whole both
have that value as code; the basic percept is their tightest pair, reserved
by `BasicPercepts.reserve`, then promoted. This is the decision about endpoint
creation and identity: exact endpoints are admitted on first perception,
and identity uses the tightest pair, not every containing endpoint. Negative
references below the existing numeric-axis namespace identify these native
value endpoints. A value occupies its part tower's scalar coordinate and its
whole tower's scalar coordinate. Neither coordinate is a learned code or an
extra concept.

A named collection is minted on recurrence, with basic-percept features at
order zero. Its common part and covering whole are derived from those
features. Different intervals therefore have different codes; identical
intervals have equal codes and retain distinct object references and names.
The two boundary readings join the alternative witnessed percepts and then
intersect the boundaries through `FieldLogic`. On the scalar this is exactly
§3.10's two coverages. Bare output values use that shared conceptual read;
full membership permits a word. A value in no interval emits the **empty
string**. Partial membership is not a naming decision.

The three initial perceptual-bounds certificates failed before the addendum
in **2.87 seconds**: `test_scalar_object_collects_value_bounded_basic_percepts`,
`test_intervals_generalize_their_percepts_and_have_distinct_codes`, and
`test_scalar_has_no_private_bounds_or_membership_reader`. They passed in
the **33-pass run, 14.18 seconds**, and again in the final 103-test run.
Additional cases cover both endpoints, generalized
intervals, agreement of native and stimulated readings, a text percept using
the same collection reader, retirement, and restoration into a fresh model
through the integrated checkpoint. The removed dictionary and reader are
absent from production sources; `stimulate` contains no scalar branch.

| Retired scalar state | Current owner or derivation |
|---|---|
| `dimension` (axis atom, concept, row, coordinate) | The extra axis concept is removed. A reading refers to its value-bounded basic percept; coordinates follow the existing basic-percept tower layout. |
| `boxes[word].low`, `.high` | Not stored. The basic-percept feature collection supplies the common part and covering whole. |
| `boxes[word].object` | The existing word/object definition table, read through `InterpretLayer`. |
| `boxes[word].views`, `.agreement` | MTM counts and witnessed agreements before promotion; existing concept-trust counters after promotion. The definition's stored evidence holds the resulting certainty. |
| `boxes[word].candidate` | The named pending collection in MTM. Its defining word distinguishes it from another word heard with the same percept. |
| `scalar_state`, `membership`, `field_memberships` | Deleted. Old checkpoint sidecars warn and are discarded because they contain no witnessed percept collection; their definitions must be relearned. |

The collection's perceptual bounds are unscaled native geometry. Its counted
certainty belongs to its definition; it does not allocate a meaning-direction
radius that would change those bounds. At six agreeing views, the definition
has evidence 0.8646647334. Present readings stay transient references in MTM;
there is no LTM occurrence fabricated to make a basic percept addressable.

| Shared functions and owners | Use in both modalities |
|---|---|
| `BasicPercepts.reserve`, `promote`, `subspace_bounds`, `part_codes`, `whole_codes`, `lookup_rows` | Knit and read native part/whole witnesses in one basic-percept table. |
| `PerceptCollections.observe`, `MediumTermMemory.append_path` and `promote`, `add_concept_feature` | Count a named collection's observations and retain basic-percept membership in the concept's feature table. Ordinary conjunctive definitions retain their existing conjunction contract. |
| `BasicPercepts.read_collection`, `PerceptCollections.read_native`, `Spaces.cs_read_percept_field` | Generalize a collection's boundary readings, then use the common concept and definition fold. Text supplies native part/property readings; numbers supply value containment readings. |
| `BasicPercepts.read_points`, `PerceptCollections.read`, `FieldProjection.stimulate` | Project a point through those same basic-percept boundaries and collection fold. |
| `MemberRead.read`, `MemoryLink.member_order` | Read and classify a basic percept by its reference, including when no separate concept annotation accompanies it. |
| `InterpretLayer.define`, `lexical_form`, `word_surface_for_row` | Store counted definitions and say the uniquely naming native word. |
| `ConceptForgetting` and the existing basic-percept/allocator/MTM checkpoint paths | Retire and restore the same owners; no scalar bounds or lifecycle sidecar remains. |

Number-only functions are `NumericPercepts.feeling`/`number` (the fixed
spacing and inverse), `BasicPercepts.perceive_value` (admission of exact value
endpoints), and `ScalarPercepts.code`, `bare`, `perceive`, `reading`, `join`,
`commit`, `reconstruct`, `decode`, `concepts` and `observe` (the optional
unplaced input, numeric boundary adapter, paired observation, and modality
inverse). Text's `PartSpace.part_memberships`, native byte/property readers
and byte inverse remain text-specific because values have no byte locations.
§3.9 still uses `BasicPercepts.numeric_dimension` and value-as-evidence for an
axis concept. It could use value-bounded percepts only by changing that feature
convention and the XOR proofs; that change is not made.

Native acquisition exposed two integration causes beyond the initial cases.
MTM promotion tried to write a basic percept as an LTM event; percept
collections now write their membership through the feature table. Two words
hearing one value also collided in the pending collection key; that key now
includes the defining word. Finally the shared memory reader now accepts the
basic-percept address itself, rather than requiring a duplicate annotation.
Each has a failing-before case and a passing-after case. The subsequent
scalar/memory group passed 61 tests in 24.81 seconds.

### 14.2 D5 and the thought-write producers

The writer's semantic-chain/checked-binding guard is unchanged.

* Scalar acquisition formerly reused a word's discourse collection and
  overwrote its member order. The shared `PerceptCollections.observe`
  writer now owns a distinct pending collection, keyed by its defining
  word; members are basic-percept references at order minus one. It
  promotes those members through `add_concept_feature`, not through an
  LTM event. The original collection-separation regression failed before
  in 2.29 seconds; the after group passed 17 in 11.01 seconds. The later
  addendum certificates in §14.1 replace its scalar-specific machinery.
* `ReferenceClose._prepare` computed a member order but omitted it from
  `_reference_sites`. Thought could select a different-order identity,
  rejected later by `Memory.record_link`. The site now carries the order.
  The certificate failed before in 2.32 seconds; 95 related tests passed
  and four skipped in 35.53 seconds. `ThoughtClosing.closing_question`
  also dropped those sites when it made the trial question. It now
  carries `_reference_sites` with `_forward_references`. The trial case
  failed before (one failure, one pass, 2.72 seconds); both cases pass.
* An empty `query` or `ask` returned no meaning, and
  `ThoughtStream.integrate` filled the executable request as if that
  request were its conclusion. The request has an operator and an
  operand: it is neither a semantic chain nor a checked binding. Each
  face now returns `requested_meaning`; an empty read retains that
  unanswered operand, its open references and zero evidence. A cutoff
  before the face ran leaves the existing goal. No result is invented.
  The direct controller certificate fails on the old integration; the
  recursive certificate, `test_an_empty_read_inside_a_conclusion_writes_its_operand_recursively`,
  puts this actual integrated result in a retained parent's constituent
  and reaches the same recursive writer guard as §11.5. Both `query`
  and `ask` fail before (two failures, 2.03 seconds) and pass after
  (two passes, 2.36 seconds), with the guard unchanged.

**Limit of the historical trace.** The original §11.5 receipt saved its
traceback, but neither its operands nor its initial random state. The
recursive certificate establishes the empty-read producer and its repair;
it cannot identify the uncaptured original child with certainty. A new
unseeded run on the incoming source stopped after 13 supervised updates
at the intuition replay-pair check (227.035 seconds), before reaching that
writer. Its captured initial state was replayed on the current source
for comparison, not selected for passing. That replay exposed the missing
basic-reference read repaired in §14.1, then reached 26 updates and a
separate collection-order rejection in 104.353 seconds.

The latter producer is `ReferenceClose.commit`'s sign/no-op branch: it
opens an already retained occurrence at order zero. `MemoryLink.link`
has no selected target, and `MediumTermMemory.open` reuses the occurrence's
self-owned token, whose retained `member_order` is one. `record_link`
correctly rejects mixing those orders. The three involved functions
(`link`, `open`, `record_link`) are AST-identical to `9f7923d67f`;
a direct replay of the same self-owned-token/order-change inputs raises
the same guard on both HEAD and current source. This lifecycle behavior
predates this round and is reported, not changed.
The scalar-question diagnostic belongs to the recognition work now
deferred by §3.10. It supplies no recognition certificate. The earlier
18-update diagnostic's zero of three correct answers is likewise not
claimed as a passing result.

### 14.3 Recall: D12, certificates 15, 16 and 18

`episodic_concepts` and `episodic_values` now have fixed
row-by-order-by-slot shapes. Production construction supplies
`len(owner._order_caps())`; staging reads `owner._row_order`. Selection
retains the strongest readings independently at each order. The signed
slot representation retains both lanes in separate slots when necessary,
rather than netting them. Old two-dimensional checkpoint entries load at
order zero with a warning, including episode-state restoration.

`EpisodicMemory.matches` computes the whole window's products together.
It gathers the question's readings at the stored references, admits only
the same order and lane, divides by the question's squared norm for that
order, and caps at one. Repeated stored readings of the same reference and
lane contribute their strongest reading once. Empty question orders are
omitted. `relevance` intersects the present result columns through
`FieldLogic.reduce`, so a later result over rows can join the same axis.
Only positive results are returned, by relevance and then newest first.
Selected records are reinstated at their own orders. The former per-row
`meet_ratios` calculation and concept-to-row index are deleted.

D1's operand-reference scoping remains: explicit reference to a document
cover restricts the recall to that cover's members. A content query is
still a query about its operands; dropping that restriction would allow
an otherwise matching assertion from an unrelated document to answer it.
The scope-membership construction can iterate over addresses; the numeric
match itself has no row loop.

The four initial recall certificates failed before in 2.37 seconds. The
combined recall/table after run passed seven tests in 2.71 seconds. The
checkpoint run also covers an empty reading axis, both lanes, omitted
orders, old-shape migration and the full-window read count. In the latter,
eight window rows are eligible, the three newest equal matches are rows
7, 6 and 5, exactly three records are read, and the ensuing cued read
reuses those frames with zero additional record reads.

The selectors in `test_item18_finish.py` are
`test_recall_keeps_each_order_and_intersects_without_carrying` (15 and 18),
`test_recall_product_keeps_lanes_and_reads_only_selected_records` (16),
`test_episode_selection_preserves_both_lanes_at_every_order` and
`test_old_episode_shape_loads_at_order_zero_with_warning` (construction,
selection and migration). These are the four before failures above.

### 14.4 Two tables: D16, certificates 19 to 21

`LTM0.cued_rows` now reads only explicit references and rows returned by
episodic recall. Its role match and scope filters remain. It no longer
adds the newest window or every relation row. `LTM1.store_order` and its
candidate-order reader are removed; its old checkpoint key is accepted
only to warn and discard it. No live column or reader remains.

`ThoughtLTMCapability.relation_evidence` uses indexed `find` with both
ends, and indexed `rows_referencing` with one. It does not walk the table.
Tagged references preserve the distinction between a symbol and a nested
relation with the same integer address. The equality reading retains its
equality marker. `LTM1.add` updates `_pairs`, `_parts`, `_wholes` and
`_references` immediately, including an operator reference; a read after
that write does not rebuild them. A forgetting pass may rebuild once.

The three initial table certificates failed before in 2.28 seconds. They
passed in the seven-test recall/table run. The additional warmed-reader
cache audit passed four tests in 2.92 seconds and is included in the
checkpoint verification. Both-end relation evidence reads one relation
record. Conditional evidence reads that relation and its two endpoint
records, three records in total. These counts replace the former
table-scan accounting.

The tests whose candidate reach changed are listed below. Explicit
addresses retain assertions about matching, scope, roles, trust, history
and absence of implicit writes; they do not restore broad candidates.

| Test (under `test/`) | Row and changed reason |
|---|---|
| `test_accessible_mind.py::test_cue_fan_is_charged_and_scope_filters_before_ranking` | Former automatic candidates 0 through 11, including expected rows 2 and 3, now need explicit references; all 12 are given to test the same ranking and scope contract. |
| `test_accessible_mind.py::test_query_retrieves_every_old_cued_frame_without_an_implicit_write` | Old row 0 and newer row 21 were reached through the window. The renamed `test_query_retrieves_an_addressed_old_frame_without_an_implicit_write` addresses row 0; the automatic-window contract is retired by D16. |
| `test_accessible_mind.py::test_thought_retrieval_keeps_history_and_restores_episode_knowing` | The oldest row, 0, is explicitly addressed instead of being found by the removed window fallback. |
| `test_accessible_mind.py::test_nested_writes_are_shared_across_batch_positions` | The written parent row is supplied explicitly to both stream reads; stream-sharing assertions remain. |
| `test_item25_repair.py::test_small_cued_window_preserves_match_order_filters_and_explicit_references` | Given row 5 is the only initial candidate; former automatic rows 0, 3, 11, 4, 6, 10 and 7 disappear. A separate read explicitly gives all 12 rows to preserve the matching/filter assertions. |
| `test_item25_repair.py::test_two_thousand_rows_retrieve_the_newest_window_and_charge_only_returns` | Unaddressed row 1777 is no longer reached through the newest 1024 rows. Renamed `test_two_thousand_rows_read_only_given_references_and_charge_returns`: explicit 1777 costs one read; explicit 20 and 1777 reach both. Capacity remains 2000. The automatic-window contract is retired by D16. |
| `test_modal_timelines.py::test_retained_idea_keeps_its_subject_reference_after_token_expiry` | The retained idea row is explicitly addressed after token expiry; expired-timeline and retained-reference assertions remain. |
| `test_query_executors.py::test_content_query_keeps_all_roles_and_conflicting_fact_sources` | The first occurrence, row 0, is explicit; rows 0 and 1 no longer arrive from the window. The role, source, trust and no-write assertions remain. |
| `test_layout_removal.py::test_close_placement_updates_meaning_and_content_retrieval_finds_idea` | Row 0 has neither an episodic entry nor a supplied reference and is no longer implicitly found. The renamed addressed-read test supplies its occurrence and keeps the placement/meaning assertions. |
| `test_grammatical_query_vps.py::test_unary_query_keeps_full_description_through_an_existing_occurrence` | An unaddressed description no longer finds row 0. Binding creates row 1, whose zero-trust evidence is read without reaching row 0; the separately addressed original still reads row 0 at trust .75. Description, role, scope and count assertions remain. |
| `test_query_contract_boundaries.py::test_query_limits_native_content_reads_and_reports_incompleteness` | Former window candidates 1 and 0 are explicit operands, in that order. Read limits, ranking, trust .2, incompleteness and maximum-return assertions remain. |
| `test_query_contract_boundaries.py::test_unified_query_returns_complete_record_and_does_not_admit_observation` | Row 0 is explicitly addressed; it is no longer obtained from the window. Complete-record assertions remain. |
| `test_query_work_budget.py::test_fact_reads_stop_at_shared_budget_and_preserve_partial_evidence` | Rows 1 and 0 are explicit operands instead of window candidates. The shared-budget and partial-support assertions remain. |
| `test_taxonomy_policy.py::test_unrelated_true_episode_cannot_establish_the_next_parent_relation` | The true occurrence, row 0, is explicitly referenced by the first fact. The subsequent unrelated relation still cannot use its evidence. |

Other ports cover the per-order episodic shape, retired LTM1 percept and
order columns, relation writes' removed `when`/`where` arguments, renamed
entry-time reads, and the fourth `_materialize_answer_idea` result now
carrying references. The generation-policy fixture now supplies the
representable answer meaning whose occupancy that test exercises. The
thought-credit test measures the same policy input the native draw used,
including its relative time, instead of a separately assembled absolute-
time input. Its probability-movement and learned-chain assertions remain.

**Cache audit.** Each store declares one `caches` list; cached buffers are
nonpersistent, and old saved cache entries warn and are discarded.

| Store and declared cache | Source and reader confirmed |
|---|---|
| LTM1 `_pairs`, `_parts`, `_wholes`, `_references`, `_index_version` | Tagged endpoint/operator references; `find`, `parts`, `wholes`, `rows_referencing` and relation evidence. `_index_version` is invalidation bookkeeping. |
| LTM1 `relation_order`, `_order_revision` | Nested relation dependencies; `_sort` and `refresh_effective`. |
| LTM1 `effective_plus`, `effective_minus`, `_effective_source`, `_primitive_evidence` | Stored evidence and authority/operator gates; `evidence` and dependent relation refresh. |
| LTM0 `_index_occurrences` | `address_keys`; occurrence-to-row location. |
| LTM0 `_native_row_index`, `_native_reference_token` | Retained semantic references; native row and reference lookups. |
| LTM0 `_slot_index`, `_slot_index_source` | Retained rows' semantic slots; `slot_of_reference` and semantic reference reads. |
| LTM0 `_forward_postings`, `_forward_by_address`, `_forward_index_source` | Retained forward-reference bindings; `_forward_addresses`. |
| LTM0 `_direct_kinds` | Declared `KINDS` and their document digests; direct row-kind recognition. |
| LTM0 `_retention_columns` | Retained row evidence and metadata; forgetting's retention values. |
| LTM0 `_cue_roles`, `_cue_role_mask`, `_cue_available` | Scratch populated from addressed rows by the cued role match. |
| LTM0 `_observation_ids`, `_observation_revision`, `_observation_count` | Current occurrence rows; observation enumeration. |
| LTM0 `addressing.recurrence_weights` | Sum of evidence on rows assigned to each recurring reading; recurrence bookkeeping. |

**Exception to §3.14's proposed first table:** `recurrence_codes` is
auxiliary state, not a cache. `MemoryConsolidation.recurrence_groups`
retains a promoted reading's prototype after all of its occurrence rows
have gone and uses it for later angular admission. Those surviving
prototypes cannot be reconstructed from the retained two tables. Deleting
them as cache would change a read after epoch restoration.

The `auxiliary` declarations retain the proposed second table with uses:
both stores' access counts and surprise for forgetting; LTM1's sentence
reference for age, operator reference and operation seat for authority;
LTM0's semantic rows for roles/references/scope/bindings; source keys,
sentence positions, source authorities, fingerprints, required-sidecar
flags, texts, document sources and covers for provenance; stream for
stream membership; recurrence assignments/symbols/document counts and
expectation rows for recurrence and expectation. The declarations also
name LTM1 gates and their operation catalog, epoch handles/baseline and
vacant handles; LTM0 recurrence prototypes, allocated reading seats and
first identities, supplied assertions, epoch baseline, cover RNG state,
model clock and forgetting schedule/threshold/pending state. Each use is
written next to its field in the store's declaration.

At the largest constructed LTM0 configuration,
`data/BasicModel_corpus.xml`, capacity is 1,048,576, row width 1256,
recurrence capacity 262,144, stream capacity 28, episodic slots per order
four, and order count six. The stored row columns increase from **5241
to 5481 bytes per row**. The LTM0 persistent tensor payload increases from
**6,852,711,429 to 7,103,321,093 bytes**. The delta is 251,658,240 bytes
for the five additional order slices minus 1,048,576 bytes for the
derived recurrence-weight cache. This measures stored tensor payload,
not Python sidecars, LTM1, process memory or serialization framing.

### 14.5 Time: D13, certificate 17

| Retired entry-time name | Current name |
|---|---|
| `EntryRead.entry_when` | `EntryRead.entry_time` |
| `RowTime.stamp` | `RowTime.next_time` |
| `RowTime.read_stamp` | `RowTime.read_time` |
| `RowTime.write_stamp` | `RowTime.write_time` |
| `LTM0.when` | `LTM0.time` |
| `append_meaning(..., when=...)`, `append_idea(..., when=...)` | `time=...` |
| Row/part-row record's `when` key for entry time | `time` |
| `SituationFrame.when` | `SituationFrame.time` |
| `PartClosing.when` | `PartClosing.time` |
| `EpisodicMemory.anchor(..., when=...)` | `time=...` |
| Entry-time local `original_stamp` | `original_time` |
| Forgetting locals `stamp`, `stamps`, `stamped` | `time`, `times`, `timed` |

`SituationFrame.from_row` was a missed reader: after the record key was
renamed it silently read no time. It now reads and detaches `time`.
`when_start` still names the percept-band layout index. Percepts' and
chunks' `.when`, MTM time arguments, and the produced percept time stay
as such. Historical quoted retired API names in this decision record do
not denote live readers.

The initial time certificate failed before in 2.14 seconds. The completed
certificate, including the situation-frame reader, passes in the
checkpoint run. The frame-time/API port run passed 27 tests with one
warning in 26.51 seconds.

### 14.6 Finished-source certificates and measurements

The turnover- and checkpoint-fixture ports are the only source-file
changes from the measurement snapshot below to the returned snapshot. Production
code, configuration and the measured proof/lesson code are identical.
Sections 14.8 and 14.9 give their replays and focused checks; §14.7 gives the
default sweep on the resulting source. The existing proof and lesson
receipts remain attributed to the snapshot on which they ran.

On source `3dcad5952b641ebfc59305ed518c1c611fbd3a4c54e4268d7a11e8133a28b649`,
the final focused run passed **103 tests in 90.64 seconds**, including
both stipulated stopping gates, numeric XOR, completion repairs,
reference-carried generation, perceptual scalar bounds, recall,
tables/caches and entry-time names. This explicitly selected proof run
uses `RUN_SLOW=1` for its existing marked numeric proof; the default sweep
unsets it as requested. Neither stopping gate has failed in this completion.
The kept verbal cover,
`test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`,
passed its **four-of-four** word-multiset assertion after a fresh unseeded
training: **one passed in 763.49 seconds** (guard 766.944 seconds, peak
4,648,457,472 bytes). Its historical selector is retained; its required
bar is four, not the percentage in that name. Certificate 11's numeric
and verbal proofs pass. Section 14.7 records the earlier sweep failure,
its captured replay and the repair; it is not hidden by this passing run.
The final sweep and 160-sentence lesson are reported below.

**Ten numeric XOR trainings.** All ten unseeded `XOR_exact.xml` runs
completed before the post-sweep generation and text-field repairs. Each
had MSE **0.00042077835**, promoted row **2**, and the same
four answers and three field-reader weights below. Fixed native axes and
reader initialization explain the identical results; no seed was chosen.
Answers are in input order 00, 01, 10, 11. These ran beside the separate
verbal/lesson jobs, within their process memory guards.

| Run | Four answers | Three weights | Promoted row | Seconds |
|---|---|---|---|---|
| 1 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.464 |
| 2 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.359 |
| 3 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.406 |
| 4 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.406 |
| 5 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.313 |
| 6 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.328 |
| 7 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.275 |
| 8 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.381 |
| 9 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.252 |
| 10 | -0.0187577084, 1.0230909586, 1.0230909586, 0.0162750408 | 2.5449185371, 2.5449185371, -3.7416553497 | 2 | 10.286 |

**Certificate 13, as amended by §3.10.** The native scalar lesson completed
24 presentations in **17.774 seconds**: six each of `one` with 1, `two`
with 2, `three` with 3, and `unfelt` with no number. The first three had
distinct order-zero concepts (148, 150, 152), each with six views and
definition evidence **0.8646647334**. Each collected its reading's native
basic percept; `unfelt` had no scalar object. All six omission checks
passed through `understand` and `reconstructModalities`:

| Omitted modality | Given input | Restored result |
|---|---|---|
| Number | `one` | 1.0000000123994952 |
| Number | `two` | 2.0000002496485365 |
| Number | `three` | 2.999999343408167 |
| Word | 1 | `one` |
| Word | 2 | `two` |
| Word | 3 | `three` |

The shared perceptual reader assigned 1, 2 and 3 to those respective
concepts; .5, 1.5, 2.5, 3.5 and 10 belonged to none. These are membership
probes, not emitted question answers. Native question recognition is
excluded by Alec's latest §3.10 amendment. The named retirement is
`test_scalar_perception_binds_event_and_class`: its native recognition
claim is deferred, and its retained percept/concept reference, field,
order and no-number assertions are now
`test_scalar_perception_keeps_percept_and_concept_references`.

The tentative `test_what_episode_memory.py` configuration port was
withdrawn: its original `MM_xor.xml` fixture already exercises the memory
contract. All seven original tests pass, with their original assertions
and capacities. No vector-pipeline assumptions were added to it.

**Whole-path lesson arm.** The unchanged `explore-1.xml` configuration,
`lesson-13` curriculum and eager native driver completed **160 of 160
sentences**. The sentence intervals total **995.179 seconds**;
the guarded command took **1057.777 seconds**, with peak process-tree
memory **1,804,290,472 bytes**. The lesson ran on CPU alongside the
default sweep and the separate verbal proof. There were **599 generate
calls, 320 thought episodes and 320 paths**. The repaired observer
captured every kept path's field before trial restoration: **160 of 160**
comparisons are available. **54** were exactly equal; the largest absolute
coordinate difference was **1**, and the coordinate-weighted mean absolute
difference was **0.0548051895**, over **54,636** coordinates. This is
a measurement of agreement, not a passing threshold.

The exempt gates `interpret`, `sign`, `part`, `whole`, `equal`, `implies`,
`generic`, `lift` and `bind` were open at sentence **0**. Every other
declared gate remained closed throughout this run.
The accumulated compound kernel-test count was **114,318**. No missing
observer value is reported as a zero difference.

Per-sentence numbers follow. Calls counts all generate invocations; paths
counts the sentence alternatives. Differences compare the kept walk's
field with stimulation, after aligning concept references and lanes.

| Sentence | Generate calls | Thought episodes | Paths | Seconds | Maximum field difference | Mean field difference |
|---|---|---|---|---|---|---|
| 1 | 4 | 2 | 2 | 2.017 | 0.000000 | 0.000000 |
| 2 | 4 | 2 | 2 | 3.174 | 0.000000 | 0.000000 |
| 3 | 3 | 2 | 2 | 1.880 | 0.000000 | 0.000000 |
| 4 | 3 | 2 | 2 | 2.498 | 0.147619 | 0.003767 |
| 5 | 3 | 2 | 2 | 1.627 | 0.000000 | 0.000000 |
| 6 | 4 | 2 | 2 | 3.808 | 1.000000 | 0.051668 |
| 7 | 3 | 2 | 2 | 1.905 | 0.000000 | 0.000000 |
| 8 | 4 | 2 | 2 | 3.771 | 0.342267 | 0.008596 |
| 9 | 4 | 2 | 2 | 2.675 | 0.000000 | 0.000000 |
| 10 | 3 | 2 | 2 | 3.527 | 0.147619 | 0.003468 |
| 11 | 4 | 2 | 2 | 3.231 | 0.000000 | 0.000000 |
| 12 | 4 | 2 | 2 | 3.771 | 1.000000 | 0.025536 |
| 13 | 4 | 2 | 2 | 3.576 | 0.000000 | 0.000000 |
| 14 | 3 | 2 | 2 | 2.931 | 0.999999 | 0.030696 |
| 15 | 4 | 2 | 2 | 3.352 | 0.000000 | 0.000000 |
| 16 | 4 | 2 | 2 | 3.744 | 0.999999 | 0.039622 |
| 17 | 4 | 2 | 2 | 2.321 | 0.000000 | 0.000000 |
| 18 | 4 | 2 | 2 | 2.860 | 0.999999 | 0.049902 |
| 19 | 3 | 2 | 2 | 1.983 | 0.000000 | 0.000000 |
| 20 | 4 | 2 | 2 | 5.153 | 0.000000 | 0.000000 |
| 21 | 4 | 2 | 2 | 3.538 | 0.000000 | 0.000000 |
| 22 | 4 | 2 | 2 | 4.872 | 0.314911 | 0.007562 |
| 23 | 4 | 2 | 2 | 4.894 | 0.000000 | 0.000000 |
| 24 | 4 | 2 | 2 | 6.474 | 0.999999 | 0.037771 |
| 25 | 3 | 2 | 2 | 3.365 | 0.000000 | 0.000000 |
| 26 | 4 | 2 | 2 | 8.674 | 1.000000 | 0.040838 |
| 27 | 3 | 2 | 2 | 4.346 | 0.000000 | 0.000000 |
| 28 | 4 | 2 | 2 | 5.821 | 0.000000 | 0.000000 |
| 29 | 4 | 2 | 2 | 6.064 | 0.000000 | 0.000000 |
| 30 | 4 | 2 | 2 | 10.582 | 1.000000 | 0.035619 |
| 31 | 3 | 2 | 2 | 10.049 | 0.000000 | 0.000000 |
| 32 | 3 | 2 | 2 | 9.175 | 1.000000 | 0.046043 |
| 33 | 4 | 2 | 2 | 6.455 | 0.000000 | 0.000000 |
| 34 | 4 | 2 | 2 | 5.153 | 1.000000 | 0.051884 |
| 35 | 4 | 2 | 2 | 7.191 | 0.000000 | 0.000000 |
| 36 | 4 | 2 | 2 | 8.119 | 0.000000 | 0.000000 |
| 37 | 3 | 2 | 2 | 6.098 | 0.000000 | 0.000000 |
| 38 | 4 | 2 | 2 | 9.564 | 1.000000 | 0.038217 |
| 39 | 4 | 2 | 2 | 9.788 | 0.000000 | 0.000000 |
| 40 | 4 | 2 | 2 | 10.621 | 1.000000 | 0.050591 |
| 41 | 4 | 2 | 2 | 12.098 | 0.000000 | 0.000000 |
| 42 | 3 | 2 | 2 | 9.730 | 1.000000 | 0.047518 |
| 43 | 4 | 2 | 2 | 14.640 | 0.000000 | 0.000000 |
| 44 | 4 | 2 | 2 | 11.786 | 1.000000 | 0.053281 |
| 45 | 4 | 2 | 2 | 16.236 | 0.000000 | 0.000000 |
| 46 | 3 | 2 | 2 | 12.859 | 1.000000 | 0.042447 |
| 47 | 3 | 2 | 2 | 13.454 | 0.000000 | 0.000000 |
| 48 | 4 | 2 | 2 | 21.182 | 1.000000 | 0.053202 |
| 49 | 4 | 2 | 2 | 5.853 | 0.000000 | 0.000000 |
| 50 | 4 | 2 | 2 | 5.074 | 1.000000 | 0.055042 |
| 51 | 4 | 2 | 2 | 4.892 | 0.000000 | 0.000000 |
| 52 | 3 | 2 | 2 | 3.271 | 0.000000 | 0.000000 |
| 53 | 3 | 2 | 2 | 2.560 | 0.000000 | 0.000000 |
| 54 | 4 | 2 | 2 | 4.136 | 1.000000 | 0.056812 |
| 55 | 4 | 2 | 2 | 4.412 | 0.000000 | 0.000000 |
| 56 | 4 | 2 | 2 | 4.275 | 1.000000 | 0.060760 |
| 57 | 3 | 2 | 2 | 2.977 | 0.000000 | 0.000000 |
| 58 | 4 | 2 | 2 | 4.920 | 1.000000 | 0.059130 |
| 59 | 4 | 2 | 2 | 3.082 | 0.000000 | 0.000000 |
| 60 | 4 | 2 | 2 | 3.897 | 0.000000 | 0.000000 |
| 61 | 4 | 2 | 2 | 3.362 | 0.000000 | 0.000000 |
| 62 | 4 | 2 | 2 | 3.742 | 1.000000 | 0.058768 |
| 63 | 3 | 2 | 2 | 2.764 | 0.000000 | 0.000000 |
| 64 | 3 | 2 | 2 | 3.172 | 1.000000 | 0.064557 |
| 65 | 4 | 2 | 2 | 6.148 | 0.337704 | 0.013559 |
| 66 | 4 | 2 | 2 | 3.446 | 0.000000 | 0.000000 |
| 67 | 4 | 2 | 2 | 5.731 | 1.000000 | 0.074405 |
| 68 | 4 | 2 | 2 | 5.598 | 1.000000 | 0.042742 |
| 69 | 4 | 2 | 2 | 6.630 | 0.998348 | 0.073177 |
| 70 | 4 | 2 | 2 | 4.370 | 1.000000 | 0.088436 |
| 71 | 4 | 2 | 2 | 6.504 | 1.000000 | 0.127727 |
| 72 | 4 | 2 | 2 | 5.488 | 0.000000 | 0.000000 |
| 73 | 4 | 2 | 2 | 8.376 | 1.000000 | 0.128800 |
| 74 | 4 | 2 | 2 | 7.508 | 1.000000 | 0.133833 |
| 75 | 4 | 2 | 2 | 6.140 | 0.861445 | 0.060579 |
| 76 | 4 | 2 | 2 | 7.999 | 1.000000 | 0.138944 |
| 77 | 4 | 2 | 2 | 7.980 | 1.000000 | 0.126326 |
| 78 | 3 | 2 | 2 | 4.182 | 0.000000 | 0.000000 |
| 79 | 4 | 2 | 2 | 10.097 | 1.000000 | 0.122881 |
| 80 | 4 | 2 | 2 | 6.516 | 1.000000 | 0.133108 |
| 81 | 4 | 2 | 2 | 8.547 | 0.550097 | 0.023804 |
| 82 | 4 | 2 | 2 | 8.792 | 1.000000 | 0.130566 |
| 83 | 4 | 2 | 2 | 10.975 | 1.000000 | 0.131506 |
| 84 | 4 | 2 | 2 | 5.443 | 0.000000 | 0.000000 |
| 85 | 4 | 2 | 2 | 12.910 | 1.000000 | 0.128019 |
| 86 | 4 | 2 | 2 | 8.636 | 1.000000 | 0.120658 |
| 87 | 4 | 2 | 2 | 5.668 | 0.646345 | 0.045893 |
| 88 | 4 | 2 | 2 | 6.326 | 1.000000 | 0.100353 |
| 89 | 4 | 2 | 2 | 6.290 | 1.000000 | 0.094936 |
| 90 | 4 | 2 | 2 | 5.766 | 0.000000 | 0.000000 |
| 91 | 3 | 2 | 2 | 6.372 | 1.000000 | 0.126362 |
| 92 | 4 | 2 | 2 | 5.730 | 1.000000 | 0.099704 |
| 93 | 4 | 2 | 2 | 5.210 | 0.999999 | 0.056472 |
| 94 | 3 | 2 | 2 | 6.228 | 1.000000 | 0.115720 |
| 95 | 4 | 2 | 2 | 5.936 | 1.000000 | 0.094279 |
| 96 | 4 | 2 | 2 | 4.621 | 0.000000 | 0.000000 |
| 97 | 3 | 2 | 2 | 5.328 | 1.000000 | 0.095311 |
| 98 | 4 | 2 | 2 | 5.779 | 1.000000 | 0.097652 |
| 99 | 4 | 2 | 2 | 5.317 | 0.999999 | 0.057376 |
| 100 | 4 | 2 | 2 | 5.959 | 1.000000 | 0.099145 |
| 101 | 4 | 2 | 2 | 12.538 | 1.000000 | 0.092442 |
| 102 | 4 | 2 | 2 | 6.190 | 0.000000 | 0.000000 |
| 103 | 4 | 2 | 2 | 7.458 | 1.000000 | 0.094939 |
| 104 | 4 | 2 | 2 | 6.336 | 1.000000 | 0.099338 |
| 105 | 3 | 2 | 2 | 5.106 | 0.999999 | 0.056591 |
| 106 | 4 | 2 | 2 | 5.878 | 1.000000 | 0.099040 |
| 107 | 3 | 2 | 2 | 5.939 | 1.000000 | 0.094002 |
| 108 | 4 | 2 | 2 | 5.572 | 0.000000 | 0.000000 |
| 109 | 3 | 2 | 2 | 7.743 | 1.000000 | 0.109985 |
| 110 | 4 | 2 | 2 | 7.553 | 1.000000 | 0.109958 |
| 111 | 4 | 2 | 2 | 5.898 | 0.999999 | 0.054040 |
| 112 | 3 | 2 | 2 | 4.933 | 1.000000 | 0.099166 |
| 113 | 4 | 2 | 2 | 6.217 | 1.000000 | 0.110504 |
| 114 | 4 | 2 | 2 | 5.123 | 0.000000 | 0.000000 |
| 115 | 4 | 2 | 2 | 6.971 | 1.000000 | 0.109144 |
| 116 | 4 | 2 | 2 | 5.678 | 1.000000 | 0.097822 |
| 117 | 3 | 2 | 2 | 5.525 | 1.000000 | 0.069412 |
| 118 | 3 | 2 | 2 | 4.982 | 1.000000 | 0.097195 |
| 119 | 4 | 2 | 2 | 6.018 | 1.000000 | 0.092423 |
| 120 | 4 | 2 | 2 | 4.786 | 0.000000 | 0.000000 |
| 121 | 3 | 2 | 2 | 5.813 | 1.000000 | 0.092796 |
| 122 | 4 | 2 | 2 | 7.058 | 1.000000 | 0.096192 |
| 123 | 3 | 2 | 2 | 4.833 | 0.999999 | 0.062566 |
| 124 | 4 | 2 | 2 | 5.915 | 1.000000 | 0.098177 |
| 125 | 3 | 2 | 2 | 7.028 | 1.000000 | 0.094131 |
| 126 | 4 | 2 | 2 | 5.104 | 0.000000 | 0.000000 |
| 127 | 4 | 2 | 2 | 8.472 | 1.000000 | 0.092944 |
| 128 | 3 | 2 | 2 | 6.975 | 0.965936 | 0.092929 |
| 129 | 3 | 2 | 2 | 5.584 | 0.999999 | 0.059913 |
| 130 | 4 | 2 | 2 | 6.569 | 1.000000 | 0.097038 |
| 131 | 4 | 2 | 2 | 7.581 | 1.000000 | 0.093300 |
| 132 | 4 | 2 | 2 | 4.876 | 0.000000 | 0.000000 |
| 133 | 4 | 2 | 2 | 8.537 | 1.000000 | 0.092447 |
| 134 | 4 | 2 | 2 | 7.311 | 1.000000 | 0.095757 |
| 135 | 3 | 2 | 2 | 5.141 | 0.999999 | 0.058186 |
| 136 | 4 | 2 | 2 | 7.788 | 1.000000 | 0.097719 |
| 137 | 4 | 2 | 2 | 6.754 | 0.996400 | 0.092685 |
| 138 | 4 | 2 | 2 | 5.817 | 0.000000 | 0.000000 |
| 139 | 3 | 2 | 2 | 5.520 | 1.000000 | 0.092124 |
| 140 | 4 | 2 | 2 | 6.543 | 0.976708 | 0.094903 |
| 141 | 4 | 2 | 2 | 5.862 | 0.859348 | 0.065478 |
| 142 | 4 | 2 | 2 | 6.234 | 1.000000 | 0.098319 |
| 143 | 4 | 2 | 2 | 6.899 | 1.000000 | 0.092455 |
| 144 | 4 | 2 | 2 | 5.561 | 0.000000 | 0.000000 |
| 145 | 4 | 2 | 2 | 6.754 | 1.000000 | 0.094292 |
| 146 | 3 | 2 | 2 | 5.987 | 1.000000 | 0.097877 |
| 147 | 4 | 2 | 2 | 5.898 | 1.000000 | 0.074503 |
| 148 | 4 | 2 | 2 | 6.645 | 1.000000 | 0.096971 |
| 149 | 4 | 2 | 2 | 7.535 | 1.000000 | 0.092933 |
| 150 | 4 | 2 | 2 | 5.364 | 0.000000 | 0.000000 |
| 151 | 3 | 2 | 2 | 5.920 | 1.000000 | 0.091851 |
| 152 | 4 | 2 | 2 | 6.901 | 1.000000 | 0.098683 |
| 153 | 4 | 2 | 2 | 6.241 | 1.000000 | 0.080199 |
| 154 | 3 | 2 | 2 | 5.602 | 1.000000 | 0.099110 |
| 155 | 4 | 2 | 2 | 6.684 | 1.000000 | 0.093453 |
| 156 | 4 | 2 | 2 | 5.604 | 0.000000 | 0.000000 |
| 157 | 3 | 2 | 2 | 6.218 | 1.000000 | 0.093487 |
| 158 | 4 | 2 | 2 | 6.718 | 1.000000 | 0.097908 |
| 159 | 4 | 2 | 2 | 6.187 | 1.000000 | 0.076891 |
| 160 | 4 | 2 | 2 | 7.137 | 1.000000 | 0.098093 |

### 14.7 Sweep and remaining failures

**Final default sweep, on returned source `a8c9adced8e2`.**
`env -u RUN_TRAINED -u RUN_SLOW make test` completed **8,141 of 8,141
selected cases**, with nothing deselected: **7,815 passed, 323 skipped,
one non-strict XPASS and two failed**. The runner took **2327.417 seconds**
(wrapper **2327.573 seconds**) and reached peak aggregate process-tree
memory **16,316,975,728 bytes**. The result is `test_failure`, test-runner
exit 1 and `make` exit 2. There were **zero compile-cache failures**.
The frozen source remained unchanged throughout the run. The raw hook
reports contain three additional passing reports for already-counted
nodes; the totals above count distinct cases.

After the final return edits, the document-link check passed **1,014 tests
in 121.52 seconds**. The final whitespace/diff check is clean.

Both stipulated stopping gates passed. The sweep's fresh verbal XOR
training passed its **four-of-four** word-multiset assertion: **one passed
in 571.15 seconds**, worker **574.295 seconds**, peak **4,659,221,880 bytes**.
The turnover and checkpoint cases also passed, in **7.116** and **1.759
seconds**, respectively. The non-strict XPASS is
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.

The remaining failures are both earlier failures, left unrepaired as
instructed:

| Failed test under `test/` | Producer and disposition |
|---|---|
| `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` | The inventory expects `PartSpace._radix_part_events`; the existing reader is `PartSpace._radix_code_events`. The same older mismatch remains. The newly renamed entry-time readers were ported separately. |
| `test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row` | The native definition writes its basic-percept collection at feature column `{0}`; the older assertion expects `{4, 402}`. The earlier-source reproduction establishes the unchanged producer. |

The intermittent reasoning-query case passes in **1.250 seconds** in this
final sweep. Its earlier captured failure and producer remain recorded
below; this pass does not retract that attribution.

The first sweep after Alec's turnover approval, on `a056559b9c26`, was
interrupted before applying the checkpoint pipeline port: **5,632 of
8,141 selected cases completed**, with one failure,
`test_checkpoint_restores_optimizer_counters_and_rng`. The runner took
**1206.734 seconds** (wrapper **1206.860 seconds**); peak aggregate memory
was **15,291,663,256 bytes**. The turnover case passed in **3.299 seconds**.
No stopping gate failed. This is a partial diagnostic, not the final sweep.

**Complete sweep before the approved turnover-fixture port.**
`env -u RUN_TRAINED -u RUN_SLOW make test` completed **8,141 of 8,141
selected cases**, with nothing deselected: **7,814 passed, 323 skipped,
one non-strict XPASS and three failed**. The runner took **2374.691
seconds** (wrapper **2374.858 seconds**) and reached peak aggregate
process-tree memory **16,672,983,616 bytes**. Its result is `test_failure`,
test-runner exit 1 and `make` exit 2. There were no compile-cache failures.
The source snapshot remained exactly
`3dcad5952b641ebfc59305ed518c1c611fbd3a4c54e4268d7a11e8133a28b649`
throughout the run. The raw hook reports contain three additional passing
reports for already-counted nodes; the counts above are distinct cases.

Both stipulated stopping gates passed. The sweep's fresh verbal XOR
training also passed its four-of-four assertion: **one passed in 506.35
seconds**, worker **509.333 seconds**, peak **4,306,392,128 bytes**.
The non-strict XPASS is
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
The three failures and their producers are:

| Failed test under `test/` | Producer and disposition |
|---|---|
| `test_forgetting.py::test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads` | Four valid records per closing plus a cover require 33 rows before the document boundary, against the fixture's fixed 32. `ThoughtStream.write` rejects the final absence. Section 14.8 gives the replay and the subsequently approved contract port; this result precedes that port. |
| `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` | The older inventory still expects `PartSpace._radix_part_events`; the reader is `PartSpace._radix_code_events`. Unchanged older mismatch. |
| `test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row` | The older assertion expects feature columns `{4, 402}`, while the native basic-percept collection writes `{0}`. Reproduced on the earlier source; unchanged. |

The preceding sweep attempts and repairs follow, so the final result
does not conceal an earlier failed or interrupted run.

The first sandboxed invocation stopped before collection because the
runner could not query physical memory. The next invocation was terminated
at **1,615 of 8,140 completed cases**: the relative path in the diagnostic
stopping-hook option broke nested pytest runs in temporary projects. It
reported **27 distinct failed nodes**, including **13 harness failures**.
No stopping gate failed. The hook path is now absolute; the runner's own
checks pass **42, with three skips, in 64.10 seconds**. This partial attempt
is not counted as the required complete sweep, and its failures are not
hidden as model failures.

Its other 14 failures identified three missed test ports, all now checked:

* `test_basicmodel.py::TestDataTextStorage::test_train_input_strings_for_text`
  used the now-numeric built-in XOR loader. It gives the original four
  text examples and labels to the inline text loader, retaining both
  storage assertions (§3.9).
* The 12 `test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver`
  parameters supplied anonymous order-zero antecedents to an explicitly
  order-one request. They also wrote their mentions on stream `None` but
  read them on stream zero. The fixture now gives both antecedents native
  order-one identities and records its closes on stream zero. Binding,
  forward-reference, inverse and distractor assertions remain; the runtime
  order and accessibility checks are unchanged. This is not an LTM
  candidate-window port: the candidates were already in the supplied
  situation frames.
* `test_concept_memberships.py::test_fractional_feature_weights_use_per_pole_min_and_max_extent_union`
  still expected the old unqualified against pole. Under §3.2 its .75
  against value at weight .5 becomes `1 - (1 - .75)**.5 = .5`, so the
  greatest against pole is .5, not .36. All native percept and extent
  assertions remain.

The three files' focused run passed **58 tests in 23.29 seconds**. These
were the only source-input differences from the preceding certificate and
lesson runs at that stage: **no production file, data file or capacity
changed**. The first complete sweep's source hash is
`77d4556bcf39176c9ea495e39b81742c731e82732e4ad282f3051f77b20ce8ba`.

The complete default sweep selected and completed **8,140 cases**, with
nothing deselected, in **2,459.933 seconds** (wrapper 2,460.086 seconds).
Its distinct outcomes are **7,768 passed, 323 skipped, one non-strict
unexpected pass and 48 failed**. The duplicate flag case produced three
additional pass reports but does not add selected cases. Peak aggregate
memory was **14,664,025,488 bytes**. Both stipulated stopping gates passed.
The unexpected pass is
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
The native runner's result is `test_failure`; the make invocation exits two.

The sweep's 48 failures divide into 44 cases whose fixtures still called
retired interfaces or relied on retired contracts, three older failures,
and certificate 11. The 24 test-input ports preserve the surviving
assertions. Five candidate-reach ports are itemized in §14.4. The other
ports are:

| Test file(s), under `test/` | Change and retained contract |
|---|---|
| `test_item20_round34_repair.py` | The separate compose-credit fixture declares `compose_policy` as its owner; unrelated chooser and reader parameters still receive no gradient. |
| `test_item20_round6_reverse.py` | Text-only basic-percept mocks implement `is_value=False`; byte events and the 8192-position capacity assertion stay. |
| `test_item55_mechanisms.py::test_clock_independence` | Read `recurrence_weights` as the declared cache, separately from persistent state. Its shape, finiteness and clock-independence assertions stay. D16 retires its checkpoint membership. |
| `test_ltm0_location.py`, `test_selected_nested_meaning.py`, `fixtures/when-readers-round4a0.json` | Entry `.when` readers become `.time`. Only the three renamed entry-time readers leave the percept-time inventory; the older radix-method mismatch remains. |
| `test_model_time_when.py`, `test_reset_at_batch_boundary.py`, `test_review11_contracts.py`, `test_understanding_reconstruct.py`, the numeric fixture of `test_what_training.py` | Restore `MM_xor.xml` for callers that supply numeric XOR. Time, reset, no-retrieval, inverse ownership and answer-target separation assertions stay. `test_understand_returns_one_explicit_understanding` now requires the numeric inverse's `field` carrier; its old `terminal_idea` and text-mask carrier requirements are retired by §3.9. |
| `test_objective_ownership.py::test_linear_field_reader_and_definitions_have_their_declared_writers` | Use the original text-definition fixture, `XOR_symbols.xml`, with field operators declared explicitly under §3.1. Definitions remain reconstruction-owned and the linear reader output-owned. No dimension changes. |
| `test_meaning_kernels.py` | The two surface inverse cases give the left marker and recover the right content under D9; exact recovery, unchanged form and gradient assertions stay. |
| `test_part_expectation.py` | Assert the nine persistent relation columns, omitting the retired percept/order columns and the effective-value cache under D16; both closing-audit round trips stay. |
| `test_reference_occurrence_views.py` | Give the read the same stream zero as the supplied occurrences; all four participant and immutable-bank cases stay. |
| `test_serial_concepts.py` | The store mock declares `episodic_orders=len(owner._order_caps())`; all three native-read and inactive-stream cases stay. |
| `test_serve_thinking.py::test_thinking_payload_is_attached_for_a_thinking_model` | Retire the assumption that zero thought operations implies zero journal slots: an empty episode may have `begin`/`finish` boundaries. Zero iterations, no steps and the single-inference-call assertion stay. |
| `test_tetralemma_leaf.py::test_not_exchange_and_non_withdrawal` | D7 retires the retained against lane of `non`; all lanes must now be zero. |
| `test_thought_answer_adapters.py` | Generated-result mocks supply the new optional `realization_field=None`; concept, set and subgoal adapter assertions stay. |
| `test_word_admission.py::test_29_each_reading_publishes_only_its_words[MM_text.xml]` | Declare the former parallel field path at the original dimensions under §3.1. The absent-word evidence and distinct-reading assertions stay. This exposed the field-read defect below. |

The first verification of the 23 Python files gave **325 passed, six
skipped and three failed in 271.42 seconds**. Two residual fixture errors
were the numeric carrier assertion and the undeclared text-definition
pipeline above. The third exposed a production defect: `_sparse_active`
still suppressed the order-zero read whenever the old symbolic-order
counter was zero. A field-only grammar therefore reached its linear output
reader with no field. The read now follows declared field operations;
the order budget still limits higher folds. The same word-admission case
fails before and passes after. Its three files and both stopping gates
pass **44 tests in 29.57 seconds**. No capacity was changed.

A boundary audit then caught an overbroad version of that read check:
the default dense numeric grammar declares field names alongside its
`pi`/`sigma` substrate. Section 6 excludes that pipeline. The order-zero
read now uses the same exclusion of substrate operators as the existing
output and numeric-percept dispatch. The extended
`test_grammar_declares_three_independent_kinds` fails before this boundary
correction in **2.41 seconds** and passes afterward; its field, word-read,
dispatch and stopping-gate group gives **54 passed in 24.65 seconds**.
The in-progress sweep on the overbroad version was interrupted before
editing source: **2,088 of 8,141 cases completed, no failures, 294.146
seconds** (wrapper 294.261 seconds). It is a partial diagnostic, not the
final sweep. Neither stopping gate failed.

**Additional certificate 11 cause.** The sweep's recorded random streams
reproduce its failed availability assertion in **618.92 seconds**. The
generated words have correct forms, but the third sentence's two meaning
blocks are zero. Both the kept path and a fresh inverse choose disjunction:
pair search can fit a zero parent by erasing both candidate meanings, then
calls those children readable because their forms remain. The common
field decoder correctly has no named-concept evidence from which to say
them. This is not an observer-restoration error.

The repair is at pair construction in
`LanguageSpace._bounded_binary_reconstruction`: a free candidate with
defined meaning must retain a nonempty reading in its recovered lanes.
Already-empty candidates and explicitly supplied operands remain legal.
No word is selected by semantic resemblance to repair the output. A
disjoint conjunction keeps its compatible nominal operands; the erased
disjunction pair is unavailable. The new
`test_free_inverse_cannot_erase_both_named_operands_to_fit_an_empty_meaning`
fails before in **2.09 seconds** and passes after in a **77-pass run,
19.06 seconds**, including both stopping gates. Loading the captured
failing model, without another optimizer step, changes its output from
three available word multisets to **four of four**, with no NULs and no
truncated walk. This is a replay of the failure, not a selected passing
initialization. Section 14.6 gives the new unseeded training and lesson
reruns on the unchanged production code.

The three older failures seen in the first complete sweep are not repaired.
The first two also fail in the `3dcad5952b64` sweep; the third passes there:

* `test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers`:
  the inventory names `_radix_part_events`, while the live reader is
  `_radix_code_events`. That mismatch predates this round. The three
  newly renamed entry-time readers have been ported separately.
* `test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`:
  the old assertion expects native feature columns `{4, 402}`; the
  basic-percept collection supplies `{0}`. The same expectation fails on
  the earlier source. The producer is the native definition's basic-percept
  feature write, not a missing feature in this completion.
* `test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully`:
  on the second query, omission leaves leading NUL bytes. In
  `_stage_reading_word_concepts`, `bytes(values).split(b'\0', 1)[0]`
  treats the first omitted byte as the end, admits no grammar-object rows,
  and `_stage_mixing_reconstruction_bank` returns without a bank. The
  validator then raises. All seven relevant query/staging functions are
  unchanged from `9f7923d67f`; replaying the captured omitted input through
  those producers gives the same failure on that source and this one.
  The full old-source test stops earlier at its old reference-namespace
  guard, so it is the captured producer replay, not that full test, which
  establishes the attribution. The current failing-stream replay fails in
  1.29 seconds; both stems without the intervening full-query state pass.
  This corrects §11.2's narrower passing observation: it did not exercise
  this omission state. In the `3dcad5952b64` sweep this test passes in **0.815
  seconds**, and in the final `a8c9adced8e2` sweep in **1.250 seconds**;
  neither pass retires the captured failure.

The original §11.5 failure lacks the captured child needed for exact
historical attribution; §14.2 distinguishes that limit from the
demonstrated and repaired empty-read producer. The independently
reproduced, older self-owned MTM-token order-change rejection is also
described there. The determiner, time result of recall, preposition kernel
head, reduced memory and operator extensions remain outside this completion.

### 14.8 Approved turnover-test port

The `3dcad5952b64` sweep newly fails
`test_forgetting.py::test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads`.
Its fixed capacity is 32. With the §3.7-retired STOP mask removed, this
captured native path keeps one cover and, for each of its eight sentences,
an embedded unresolved occurrence, the observation, its estimate, and a
conceived absence. These are four distinct retained records per sentence;
the eighth absence would be row 33. `ThoughtStream.write` correctly raises
at 32, before the document boundary where forgetting is allowed.

The failing random-stream replay fails in **9.97 seconds**. The original
`9f7923d67f` test passes with its old STOP mask. Replaying the captured
matching initialized parameters into that earlier model, with the retired
mask removed, reproduces the same eight boundaries (occupancies 1, 5, 9,
13, 17, 21, 25, 29), the same final omission confidence
**0.6334125399589539**, and the same overflow, in **8.21 seconds**.
The parameter replay copies 116 matching named tensors; twelve old
parameters have no current counterpart and five have changed shape, and
are left at their recorded old initialization. This is evidence about
the earlier unmasked writer, not a claim that the unchanged old test
failed. `ThoughtClosing.absence` is unchanged; the occurrence, estimate,
absence and capacity producers already existed. This round changes the
test's exposure by removing its mask. There is no duplicate record to
remove, and no capacity, guard or production rule has been changed to
make this test pass.

**Alec's decision, 2026-11-12:** "Port the isolated turnover test
(recommended)." The original and restored fixture models now use the
existing `expectationOmissionThreshold=1` setting. This excludes conceived
absences from the turnover fixture while preserving all **16 assertions**,
the original **32-row capacity**, its eight- and two-sentence documents,
the high-water threshold of eight and the low-water target of six. The
assertion ASTs before and after are identical. No production file changed;
the full grammar and all native writing and capacity guards remain.

The captured failing random stream now passes: **one passed, one warning,
9.32 seconds**. All **130 initialized parameter tensors** are identical
between the failing and repaired replays. A fresh focused run of the
entire forgetting file, the native omission writer, the capacity preflight
and both stopping gates
gives **37 passed, one warning, 15.50 seconds**. The warning reports that
protected rows prevent reaching the low-water target; the original
assertions explicitly check this permitted outcome. No seed was selected
for passing. The case also passes in the final default sweep in **7.116
seconds**; §14.7 records that sweep's complete result.

### 14.9 Checkpoint fixture's pipeline port

The post-approval sweep exposed a missed §3.1 port in
`test_fineweb_preflight.py::test_checkpoint_restores_optimizer_counters_and_rng`.
Renaming its original `MM_xor.xml` text configuration to `MM_text.xml`
preserved the dimensions but changed the execution path: the original
fixture used the parallel field body, while the renamed grammar's vector
operations now select the word and composition walks. The checkpoint,
optimizer and RNG restoration assertions pass; the failure comes later
when resumed training tries to consolidate a further concept.

The fixture declares eight learned concept seats. With its operator
reservations, the vector configuration has 34 constructed rows: 26
reserved, five learned after the first partial epoch, and all 34 occupied
at the failed consolidation. `MemoryConsolidation._preflight` correctly
rejects the next plan. Both it and `consolidate` are unchanged from
`9f7923d67f`. That earlier test uses the parallel body and passes, including
when its matching initialized parameters are taken from the captured
failure; the additional vector-path workload is this round's fixture
change, not an older test failure.

The test now declares its original parallel field path through the existing
`field_text_config(examine=False)` fixture, at the original configured
dimensions and learned capacities. All **12 assertions**, the wrong-batch-size
rejection, both training runs and the cursor-resume check remain. The
before/after assertion ASTs are identical. The old native path and the
ported field path both learn three seats in the first partial epoch and
four after resume. The different operator reservations follow their
declared grammar; no configured capacity is increased and no writer or
capacity guard changes.

The recorded failing random stream gives **one failed, one warning, 33.63
seconds** before the port and **one passed in 2.24 seconds** after it. The
combined fresh checks for both fixture ports, omission and capacity, and
both stopping gates give **44 passed, one skipped in 15.72 seconds**.
The skip is the existing opt-in real FineWeb step. The final sweep is
completed on this source, with nothing deselected; this case passes in
**1.759 seconds**. Section 14.7 records the complete result.

## 15. Review of the return of §14 (Claude, 2026-11-13)

This reviews the work §13 and §13.1 assigned, and only that. Item 1.8 is
not complete: §3.16 (the memory reduction), §3.17 (extend and the
determiners), §3.18 (the two networks, symbols as points, placement by the
field) and the deferred recognition of a value were decided after that
text was given and have no Codex text yet.

**Accepted.** The return does what §13 and §13.1 asked, the measurements
are reported in full, including the failed and interrupted runs, and nothing
was found that should stop it landing.

### 15.1 What was checked

- *Source.* The working tree's `bounded_tests.source_snapshot` hash is
  `a8c9adced8e2a323038e01f768d36daeef4d6454e405f5ae78aaa8712b5a6290` over
  1,073 inputs, as returned. Git HEAD is still `9f7923d67f`; nothing is
  committed.
- *Certificates.* The four certificate files of this round
  (`test_item18_completion.py`, `test_item18_scalar.py`,
  `test_item18_finish.py`, `test_item18_perceptual_scalar.py`), with
  `test_item18_fields.py`, `test_concept_binding_operators.py` and both
  stopping gates: 80 passed in 26.84 seconds on the returned source.
- *Retired names.* `entry_when`, `read_stamp`, `write_stamp`,
  `field_memberships`, `_episodic_index` and `rebuild_index` are in no file
  of `bin/`; `scalar_state` and `store_order` remain only where an old
  checkpoint is read to be warned of and discarded.
- *Read in the code.* Saying a carried reference by its one word
  (`OutputOccurrences.generated`); recall as one product over the window,
  lane with lane and order with order, intersected through `FieldLogic`
  (`EpisodicMemory.matches`, `relevance`, `query`); `LTM0.cued_rows`
  reading only given and recalled rows; `relation_evidence` reading by
  `find` and `rows_referencing` and walking no table. Each is as §14
  describes it.
- *Not rerun.* The verbal XOR proof, the ten numeric trainings, the lesson
  arm and the default sweep. Their reported numbers are taken as returned.

### 15.2 Notes

1. *A relation question with neither end given now finds nothing.* It
   walked the table before. This follows from D16 and is accepted.
2. *A bare value inside two intervals is said by the word first in the
   inventory.* Two words over one interval have one code, and when no
   reference is carried the choice between them is the inventory's order.
   Where the answer's reference is carried, as it now is, the right word
   is said.
3. *No gate but the exempt ones opened in the 160-sentence lesson.* In
   the diagnostic of §11.3, `non` opened at sentence 45 and `ask` at 66.
   No cause is given. The gates are replaced in the next round by the
   amplitude of the operator's concept, so this does not hold the landing;
   it means the lesson exercised only the exempt operators.
4. *The kept walk's field and the stimulated field draw apart as the
   lesson goes on.* Equal for the first three sentences; from about the
   thirtieth the largest difference is one in most sentences and the mean
   rises to about one tenth. It is reported as a measurement, as it should
   be. Under §3.0 and §3.17 the walk's field is the formula's selection,
   and this comparison is replaced in the next round.
5. *The stored row grows.* At the largest configuration from 5,241 to
   5,481 bytes, for the order axis of the episodic entry. The reduction of
   §3.16 removes far more.
6. *`recurrence_codes` is not a cache* (§14.4). Correct; it is deleted
   with the recurrence table in the next round.
7. *Production changes beyond the text.* `_sparse_active`,
   `LanguageSpace._bounded_binary_reconstruction`,
   `ReferenceClose._prepare`, `ThoughtClosing.closing_question` and the
   integration of an empty `query` or `ask` were repaired where their
   causes arose, each with a case that fails before and passes after.
   Accepted.
8. *Two failures remain, both older* (§14.7), and the intermittent
   reasoning-query case is attributed to an unchanged producer. Accepted
   as listed.

### 15.3 At the landing

- Codex edited seven documents on `main` (Architecture, FutureWork,
  GradientFlow, Orders, Params, Philosophy, STM). The branch `testing`
  restructured the same documents and deleted the plans. The landing
  merges these edits into `testing`'s documents, folds this specification
  into them, and deletes it.
- Still waiting for Alec: the preposition's kernel head (D9's report).
- The next round is §3.16, §3.17 and §3.18, whose Codex text is written
  on this source once it has landed.

## 16. Codex text: commit and push the checkpoint (Claude, 2026-11-13; for Alec to paste)

Alec, 2026-11-13: "please emit text that instructs codex to commit what was
done and push to github."

~~~
Commit the work of §14 and push it (Alec, 2026-11-13: "commit what was done
and push to github"). It is reviewed and accepted in §15 of
doc/plans/2026-11-11-item1-8-last-architecture-round.md. This is a
checkpoint of item 1.8, not its landing: §3.16, §3.17 and §3.18 are not
built.

1. Before anything is staged, in the main checkout:
   a. the branch is main, and after git fetch origin both HEAD and
      origin/main are 9f7923d67f8921ef5c14afdf2fe18c536802018b;
   b. the bounded_tests.source_snapshot hash is
      a8c9adced8e2a323038e01f768d36daeef4d6454e405f5ae78aaa8712b5a6290.
   If either differs, stop and report what differs. Change nothing to
   make them agree.
2. Stage the whole working tree: 261 modified files, 23 new and 2 deleted
   (bin/IdeaEvaluation.py and test/test_idea_evaluation.py). The new files
   are source, configuration and tests only. Stage nothing that git
   ignores: no receipt, nothing under output/, no cache. If the counts
   differ, list the difference in your report and go on only if every
   extra file is one you wrote for this round.
3. Make one commit on main. Its subject says it is a checkpoint of item
   1.8. Its body gives, in a few lines: that §15 accepted the return of
   §14; the source hash above; the default sweep's result on it (7,815
   passed, 323 skipped, one non-strict unexpected pass, two failed, both
   older than the round, by name); and that §3.16 to §3.18 remain.
4. Push it: git push origin main. No force, no tag, no other branch. If
   the push is refused, stop and report the refusal; do not rebase, merge
   or force.
5. Do not run the sweep again; it was measured on this source. Do not
   touch the branch testing, and commit nothing in the parent WikiOracle
   repository.

Return in chat: the commit's hash, that origin/main now equals it, and
that the working tree is clean. Write nothing more into the
specification for this.
~~~

## 17. The next round of 1.8, as it is to be built (Claude, 2026-11-13; from §3.16 and §3.17)

§3.16 and §3.17 record their decisions in the order they were made, with
what each later one withdrew. This section states the result once. A rule
is Alec's unless it is marked *default*, which is Claude's choice where he
has not spoken and which he may overrule. §3.18 is the round after this
one and is not in it.

The parts are built in this order, because an answer must be readable by
extend before the columns it is read from today are removed.

### 17.1 Extend

1. *Two methods on every operator.* Each operator of a grammar file has a
   `compose_extend` and a `generate_extend` beside its `compose` and
   `generate` ("We might want to to define compose_extend() and
   generate_extend() methods on all operators"; "every operator needs an
   extend()"). A file with an operator that lacks either fails at load. No
   section and no rule of a grammar file is added for them.
2. *`compose_extend`* takes its operands' sets of active concepts and
   gives one set. A set is concept references with their trust on the two
   lanes. It is called once for each operator of the tree the kept path's
   compose left in STM, in compose's order. The leaves' sets are the
   conceptual activations the sentence already has, every order of them
   (`_sentence_open_fields`), or the strongest `extendConcepts` of them;
   no vector is matched against the concept codes to make them. An
   operator tells its operands' activations apart by their places
   (*default*).
3. *`generate_extend`* is its inverse, for an idea that has no sentence in
   STM: one recalled from `LTM0`, or one a thought concluded. The idea's
   generate walk gives the tree; `compose_extend` gives each node's set
   from the leaves, a known concept's set being its stored extension; and
   `generate_extend` gives, for each operand, the part of its parent's set
   that is that operand's (*default* for the form of the method).
4. *What `compose_extend` does, by operator.*

   | operator | `compose_extend` |
   |---|---|
   | `intersection`, `compound`, `compound_unary`, `adverb`, `verb` | narrowing: the `intersection` (§3.2) of its operands' sets |
   | `union`, `disjunction` | joining: their `union` |
   | `not` | the two lanes swapped |
   | `non` | both lanes withheld |
   | `definite`, `definite_unary` ("the") | the member of the kind's extension that MTM holds as an identity, at its trust; nothing where MTM holds none |
   | `lower`, `lower_unary` ("a") | one generic member: every subconcept of the kind at an equal share of one, so that the set's key (`FieldProjection.key`) is their centroid (*default* for this way of holding "perhaps it can be a centroid") |
   | `universal` ("every") | the entire extension, each member whole |
   | `some`, a rule added to `complete.grammar` and composed as `universal` is (*default*) | the strongest members of the extension, more than one and fewer than all |
   | `conjunction` | `intersection` (*default*; open in §3.17) |
   | `preposition`, `preposition_unary`, `surface`, `surface_unary`, `sign`, `interpret`, `bind`, `lift`, `transpose`, `generic`, `ask`, `part`, `whole`, `equal`, `implies`, `gloss`, `divide`, `descend`, `sum`, `product` | its content operand's set, unchanged; `interpret` gives its object's set, and the operand `ask` leaves open gives nothing (*default*; open in §3.17) |

   A kind's extension is its stored members (`ConceptHierarchy.members`,
   the definition store's edges).
5. *The field of understanding* is the set the last call leaves. It
   replaces the leaves' fields set side by side in
   `OutputOccurrences.generated`.
6. *Compose's own use of a determiner is not changed*: `determiner_mode`
   and the reference proposals stay as built (*default*).

### 17.2 Recall and answers

7. *The question's side* is its field of understanding, where §3.12 took
   `stimulate` of its roles.
8. *A result over the rows.* An operand's set gives one result over the
   window, the relevance of §3.12 (`EpisodicMemory.matches`, `relevance`),
   each stored reference counted at its concept's order.
9. *The question's operators combine the results*, each as its
   `compose_extend` combines sets: a narrowing intersects its operands'
   results, a joining unions them, `not` and `non` act on the lanes, a
   determiner leaves one row, some or all. The operand `ask` leaves open
   selects nothing. This replaces the fixed intersection of every order.
10. *Reading the answer.* A recalled row's idea is walked by `generate`,
    and `generate_extend` gives each operand's part of its field. The
    answer is the part at the operand that stands, under the same
    operators from the root, where the question's open operand stands;
    where the two trees do not agree so far, no answer is read from that
    row (*default*). The match by role of `LTM0.cued_rows` goes.
11. *A question with no open operand* takes its evidence from the
    relevance of its best row, lane with lane (*default*).

### 17.3 The reduction

12. *An `LTM0` row* is a kept sentence's or a kept thought's idea: the
    vector (`slots`, whole, its trust its length, its `time` in its band);
    its entry, a flat list of `episodicConcepts` references with their
    signed trust (`episodic_concepts`, `episodic_values`, no order axis),
    filled by the strongest activations of the field of understanding
    whatever their orders; and its `activity`.
13. *An `LTM1` row* is its ordered pair (`refs`, `ref_kinds`), its trust
    (`truth_plus`, `truth_minus`) and its `activity`.
14. *Activity* is one number from zero to one for a row of either store.
    - A write sets it from the sentence's surprise, or to
      `activityDefault` where that is unknown.
    - An access adds `activityStep`: a row recall returns, a relation
      read as evidence, a pair written again.
    - A pass multiplies every value that is not one by `activityDecay`,
      and when the store is over `forgetHighWater` drops the rows of
      least activity down to `forgetLowWater`. A value of one is never
      lowered and its row never dropped.
    - An `LTM0` row not yet promoted whose activity reaches
      `activityPromotion` is promoted: its idea becomes a concept, by the
      minting `MemoryConsolidation.consolidate` does now; its activity is
      set to one; and it is not considered again.
    - *Default* values: `activityDefault` 0.5, `activityStep` 0.1,
      `activityDecay` 0.9, `activityPromotion` 0.9.
    It replaces the access counts, the surprise columns, the retention
    value and `RetentionColumns` (§3.11, certificate 12), and the
    parameters `forgetTrustWeight`, `forgetUtilityWeight`,
    `forgetLuminosityWeight`, `forgetAgeWeight`, `forgetHalfLife`,
    `forgetSimilarity`, `forgetCoverage` and `forgetProtectOrigins`.
15. *A row is named by its time.* A relation's end that is a sentence row
    holds that row's time. A row's time is the model's present, or one
    more than the last row's time where the present is not later
    (*default*), so times rise strictly and a row is found by its time.
    `address_keys` and its index go.
16. *One history.* `stream` goes. The rows of a batch of several streams
    enter the one history in the order of their streams (*default*).
17. *No relation carries an operator.* The relation kind `'operator'`
    goes, with `LTM1.operator_reference`
    (`ClauseRow.predicate_relation`, `ClauseJournal`, `PartClosing`, and
    the start `operator_argout1` of the grammar files); a sentence whose
    verb takes a completed relation closes as an idea row.
18. *The gate is the operator's amplitude.* `TrustGates`, `closedTrust`,
    its threshold, `LTM1.operation_seat` and `operation_catalog` go. The
    trust radius of the operator's concept (`ConceptTrust.positive_length`
    of its `OperationSeats` concept) scales the trust a row is written
    with (*default* for where the amplitude acts).
19. *Gone from `LTM1`:* `sentence_reference`, `access_count`, `surprise`.
20. *Gone from `LTM0`:* the evidence pair (`truth_plus`, `truth_minus`),
    `access_count`, `surprise`; `_semantic_rows`, `metadata_required`,
    `semantic_fingerprint`; `document_keys`, `sentence_content_keys`,
    `sentence_index`, `source_documents`, `source_authorities`, `_texts`,
    `_document_sources`; the covers (`DocumentCover`, `_cover_symbols`,
    `cover_rng_state`); the recurrence table and the grouping of
    `MemoryConsolidation` (`recurrenceAngle`, `recurrenceCapacity`); and
    estimate rows with `_expectation_rows`, `expectation_pair` and their
    writers.
21. *What goes with them, as Alec accepted.* A row has no origin, so
    `set_origin`, `clear_origin`, `withdraw_origin`, the match on
    resubmission, the fact and observation kinds and the source text in a
    contradiction message go. A row has no kind (`kind_of`, `KINDS`). No
    read is scoped to a document. A sentence read again is a new row.
22. *What was in the bindings.* What is pending, an open question's slots,
    a forward reference, a name awaiting its referent, is held in MTM
    while it is pending. What a row denotes is read from its entry. Which
    thought produced a row is not kept.

### 17.4 Certificates of the round

34. Every operator of every grammar file has both methods, and a file with
    an operator lacking one fails at load.
35. On a sentence of a modifier, a noun and a verb, `compose_extend` is
    called once for each operator of the kept tree in compose's order, the
    set left is the intersection the narrowings give, and `stimulate` is
    not called on the way.
36. With a kind of three subconcepts: `universal` leaves the three whole;
    `lower` leaves each at one third and a key at their centroid;
    `definite` leaves the one MTM holds, and nothing when it holds none;
    `some` leaves two.
37. An idea recalled from `LTM0` with no sentence in STM is walked and
    extended to the set its sentence left when it was written.
38. A question whose tree joins two kinds recalls the rows of either; one
    that narrows recalls only the rows of both; one under `not` leaves
    them out.
39. With "I ate rice" stored, the question with its object open is
    answered by the part at that operand, and `cued_rows` has no match by
    role.
40. A question with no open operand has, lane with lane, the relevance of
    its best row as its evidence.
41. The saved columns of an `LTM0` row are exactly `slots`,
    `episodic_concepts`, `episodic_values` and `activity`, and of an
    `LTM1` row `refs`, `ref_kinds`, `truth_plus`, `truth_minus` and
    `activity`. Every name of points 14 and 17 to 20 is in no source file.
42. Activity: set at a write, raised by an access, lowered by a pass; the
    least are dropped between the two water marks; a row at one is neither
    lowered nor dropped; a row reaching `activityPromotion` is promoted
    once and set to one.
43. Times rise strictly, in a batch of several streams also; a relation
    finds the sentence row it names by its time, before and after a pass
    has dropped other rows.
44. A sentence read twice leaves two rows.
45. A verb over a completed relation leaves an idea row and no relation of
    kind `'operator'`; an operator of half the amplitude writes a relation
    of half the trust.
46. An open question's slots and a forward reference are read from MTM,
    and from no row.
47. A checkpoint of the checkpoint commit's layout loads: each row keeps
    its idea, its time and the strongest of its entry; a relation's end is
    renamed by its row's time; every other column is discarded with a
    warning.
48. The stored bytes of one `LTM0` row and of the store at the largest
    constructed capacity, before and after.
49. Certificate 11, certificate 13 as §3.10 amends it, and both stopping
    gates hold.

### 17.5 Not in this round

§3.18 (the two networks, symbols as points, placement by the field).
Recognition of a value, which needs the field §3.18 builds (*default*;
§3.10 had it wait for `compose_extend`). The compose use of the
determiners. The preposition's kernel head. The symbol space's taxonomy,
the chain rows and the uncalled writers of §3.18's trace, on which Alec
has not decided. The time result of recall. The restructured documents of
the branch `testing`.

## 18. Codex text: the next round of 1.8 (Claude, 2026-11-13; for Alec to paste)

~~~
Build the next round of item 1.8
(doc/plans/2026-11-11-item1-8-last-architecture-round.md). Its rules are
§17, which states §3.16 and §3.17 once; where §17 and those sections
differ, §17 stands. Start from the checkpoint commit of §16, once it is
pushed. Commit and push nothing in this round until Alec says so.

A rule of §17 marked "default" is Claude's choice where Alec has not
spoken. Build it as written and list each default you built in your
return, with anything you found against it.

1. Extend (§17.1). Give every operator of every grammar file a
   compose_extend and a generate_extend, as points 1 to 6 and the table of
   point 4 say. Add the rule some to complete.grammar. Replace the side by
   side leaves' fields of OutputOccurrences.generated by the set the
   operators leave. Certificates 34 to 37.
2. Recall and answers (§17.2). The question's side is its field of
   understanding; its operators combine the results over the rows; the
   answer is read by generate_extend at the open operand; a question with
   no open operand takes its best row's relevance. Remove the match by
   role of LTM0.cued_rows. List every thought face and every other reader
   whose read changes, with what it read before. Certificates 38 to 40.
3. The reduction (§17.3), after items 1 and 2 pass their certificates and
   both stopping gates.
   a. Activity, in both stores, with its four parameters in model.xsd;
      the pass on activity in place of the retention value; promotion at
      activityPromotion (point 14).
   b. Rows named by time, and one history (points 15 and 16).
   c. No relation of kind 'operator', and the operator's amplitude as the
      gate (points 17 and 18).
   d. The columns, sidecars and modules of points 19 and 20, with every
      reader and writer of each; what point 21 says goes with them; and
      what was pending in the bindings moved into MTM (point 22).
   e. The entry: a flat list, filled from the field of understanding
      (point 12). This removes the order axis you built in §14.3.
   For each thing removed, find its readers yourself and say where each
   went: to the entry, to MTM, to the relation table, or nowhere, with the
   point of §17 that retires it. Something that still has a reader which
   §17 does not retire is a design question: stop and write it for Alec
   with your recommendation. Certificates 41 to 48.
4. Tests. Port every test whose contract still holds, keeping its
   assertions. Retire by name, with the point of §17 that retires it,
   every test of a contract the round removes (origins and supplied
   truths, covers, streams, recurrence, expectation rows, row kinds,
   re-witnessing, the retention value, the gates). Give the count of each.
5. Then, on the finished source: certificates 11 and 13 as amended
   (certificate 49), the whole-path lesson arm (the explore-1.xml
   configuration of your return, 160 sentences), now comparing the field
   compose_extend leaves for each sentence with the field of its generated
   idea, and one default sweep,
   env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected. A failure
   that predates the round is listed, not repaired.

Rules: §3, §7 and the rules of §10's text. Find each cause yourself and
repair it where it arises; a mechanism added only to pass a gate is not a
repair. No seed is chosen so that an assertion passes; no fixture is
resized; what the round retires is deleted, not kept behind a switch; no
language is named in code. Continue without stopping and record each
decision you take for Alec; stop only for a design question, written for
Alec with your recommendation. A failed
test/test_mm_xor.py::TestMMXorConvergence::test_convergence or
test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class
stops the round. Keep document edits to what this round changes.

Not in the round: §17.5.

Return as §19: each item with its causes and its certificates, failing
before and passing after; the defaults you built and what you found
against any; the readers of item 2 and of item 3; the ports and the
retirements by name; the stored sizes; the lesson's and the proofs'
numbers written out (receipts stay local and are not linked); the sweep's
result; the failures that remain, with their producers; and the returned
source hash.
~~~
