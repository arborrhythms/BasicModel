# Item 1.8: the last architecture round

2026-11-11. The specification of todo item 1.8, the round after item 2.0's
whole-path round ([specification](2026-11-11-one-pair-per-stage.md)) and
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

### 3.19 The split of sigma and pi: concepts and symbols (decided by Alec, 2026-11-13; one round with §3.18)

Alec: "OK. lets add this to the spec, it is a good split, and lets add it
as another 1.8 round". It is one round with §3.18. No text for Codex
is written yet, since the round of §17 is not returned.

#### What Alec said

- On what the two are: "Sigma is a linear hyperplane in conceptual space,
  a decision boundary." "Pi is a hypercone at the origin, also a decision
  boundary. ... in places where we now ask "what is close to this
  vector", we are effectively asking about the hypercone centered on that
  vector, so this is a learnable-orientation dot product that functions
  as the rotator to the sigma's separator. ... where we are currently
  computing dot-product cosine distances, we may wish to create a
  hypercone hierarchy where we can look for part inclusion instead, which
  is a much faster operation."
- On what each gives: "sigmas are decision boundaries and yield two
  values, whereas pis are vector similarity measures and have only one
  result, so this suggests that the result of a sigma might play into the
  two concept lanes, and the result of a pi might be a proximity measure
  to a percept (or perhaps its activation, which for letters has thus far
  been present/absent)."
- On the two stages: "We used to CS = sigma(PS) and SS = PI(CS) ... We now
  have CS = sigma(pi(PS) and SS = CS"; "the two-stage design ... allows
  things like regularization over the intermediate parameters"; "we are
  talking about concepts and symbols as the two stages."
- The decision: "So we need to split the sigmaPi. based on the
  discussion, I see boxes applying to percepts (since they live in a
  positive hypercube) and cones applying to concepts (since they live in
  a signed hypersphere). That means symbols are PI(concepts), and
  concepts are SIGMA(symbols). concepts are also SIGMA(basic percepts)."
- On which sigma a concept has: "basic percepts are part of system 1,
  which has order-zero concepts over those. system 2 creates a symbolic
  vertical structure on top of that, which is used by all higher-order
  concepts." A concept does not receive two sigmas: "if concepts are
  ramsified by order, no they do not (for any given concept)."

#### What is decided

1. *Boxes below, cones above.* A percept lives in the positive cube and
   its regions are boxes. A concept lives on the hypersphere and its
   regions are cones at the origin.
2. *The chain.*
   - System 1: percepts to basic percepts by pi, the box; basic percepts
     to the concepts of order zero by sigma.
   - System 2: the concepts of an order to symbols by pi, the cone; those
     symbols to the concepts of the next order by sigma. This repeats
     upward and is the vertical symbolic structure.
3. *One sigma for a concept, by its order:* from basic percepts at order
   zero, from symbols above.
4. *What each gives.* A sigma gives two values, the two lanes. A pi gives
   one, a presence.
5. *Concepts and symbols are two stages.* A symbol is no longer the row
   of a concept.

#### What is built now

Read in the code on 2026-11-13.

- *One table, both folds.* `ConceptualSpace._compose_order0` computes a
  row from two sparse matrices over one set of rows. `store.conjunctive`
  takes the least of a row's members, each raised to its exponent
  (`fold_presence`, `conjunctive=True`), where the switch `conceptual_pi`
  is on; `store` then takes the greatest. Each has a second pass, the
  dual, for the lane against. So a row is a sigma of a pi over rows of
  its own table, and the same row is the symbol.
- *Order zero from percepts.* `concepts_from_percepts`
  (`IndexedSigmaConceptsFromPercepts.coefficients`) is a trained
  parameter: nine coefficients for each concept row over that row's own
  references to at most eight percepts, with a penalty on their size that
  is off (`conceptReadoutL1`). Its comment has "no vocabulary-wide
  projection". This is already System 1's sigma.
- *The box.* A basic percept is the least of its part and its whole
  literals (`BasicPercepts`).
- *No layer from concepts to symbols.* A symbol is found by its concept's
  row (`_csw_row_of`); the two tables share their rows by the invariant of
  2026-09-14.
- *Places are not parameters.* `MereologicalCodes.placement` is a buffer
  moved by `placement_step`, which clamps a place at zero from below
  before it is normalized. The places are therefore unsigned now.
- *The grammar's own two layers coincide.* `SigmaLayer` is
  `tanh(atanh(x) @ W)` and `PiLayer`, with its tanh, is
  `tanh(W @ atanh(x) + b/2)`; `SigmaLayer`'s own description has that
  stacking them "does not create a hidden nonlinear feature layer".

#### Settled by Alec, 2026-11-13

a. *What a symbol's pi reads:* "Member rows with no new geometry". The pi
   and the sigma are the two sparse matrices the definition store has,
   between rows. No axis and no aperture is added. (Claude's reading of
   the round had a symbol as a cone with an axis and an aperture on the
   hypersphere of places; that is withdrawn. The cone is the shape of a
   pi's region among the activations it reads.)
b. *The two lanes:* "I don't see how we are getting rid of two symbols
   being required to store (without loss) the conceptual information of a
   single concept over a possibly heterogenous input..." Claude's
   reading: a sigma gives two values, a symbol holds one, so a concept's
   two lanes are two symbols and both stay. This is the second pass each
   fold has now, and it is "two positive symbols of one code".
c. *Its place among the rounds:* one round with §3.18.

#### The round

One round with §3.18, after the round of §17. It changes the kinds of
row and what feeds what. It does not change the arithmetic of a fold,
which §3.18 settles, and it does not touch the grammar's `SigmaLayer` and
`PiLayer`.

1. *Two kinds of row.* A row of the definition store is a concept or a
   symbol and never both.
   - A symbol is a pi: the fold of `store.conjunctive` over its member
     concepts.
   - A concept above order zero is a sigma: the fold of `store` over its
     member symbols.
   - A concept of order zero keeps `concepts_from_percepts` and has no
     edge from a symbol.
   What a promotion makes follows from what it promotes: today a raised
   kind is given a sigma over its members and a promoted field a pi
   (`FamiliarTier.cover_members`, `conjunctive`).
2. *Each lane is a symbol.* A pi gives one value. A concept's two lanes
   are read by two passes, as now, and are kept as two and never netted.
3. *§3.18's pass runs on this graph.* It goes up through a pi to a symbol
   and through a sigma to a concept, and comes down the same way. Its
   "container" is a row of either kind.
4. *No penalty until one is shown to be needed.* Alec: "since both pi and
   sigma weight matrices are COO sparse (I think?), we might add a amll
   regularizer to both so that we don't end up with N^2 issues.
   alternatively, since MTM is used as the promotion ground (the way new
   connection weights are introduced, which does not have any
   regularization pressure), maybe we can be selective there and not
   worry about regularization (preferable until regularization is proven
   necessary)." They are: both matrices are `SparseLayer`, whose own
   description begins "COO sparse", and an edge exists only once it is
   added (`add_edge`). Definition edges are added by `cover_members`,
   where a candidate of MTM is promoted (`FamiliarTier`) and where a
   class takes its members (`ConceptIndex`). No penalty acts on their
   values now. So the count of edges is what promotion admits, and the
   round adds no penalty. It reports the count of edges of each matrix
   beside the count of rows, in both gates and the lesson, so that growth
   is seen (*default*).
5. *Defaults Alec has not spoken on.* The trust radius stays on a
   concept. An entry of `LTM0` and a relation of `LTM1` refer to a row of
   either kind. The edges are taught as now: by the relations of `LTM1`
   through `DefinitionFold`, and by omission (§3.18 step 8). The
   invariant of one symbol for one concept (2026-09-14) is given up.

Checked as arithmetic on hand-made values (not a run of the model, and
not a receipt): XOR of two inputs is exact as one pi over two sigmas,
with clipping and a product in which a zero stops the product (0, 1, 1,
0); and a sigma over symbols with one symbol on each side gives both
lanes, where their sum would give nothing.

#### Certificates, when it is built

57. No row is both a concept and a symbol. A concept of order zero has
    edges from basic percepts only, a concept above it from symbols only,
    and a symbol from concepts only.
58. Each pass of a pi gives one value, and a concept's two lanes are two
    values that no read nets.
59. Certificate 11 holds, and XOR of two inputs is one pi over two
    sigmas.
60. A raised kind is a sigma over its members and a promoted field a pi,
    on the store of each stopping gate's fixture.
61. The count of edges of each matrix is reported beside the count of
    rows.
62. Both stopping gates hold.

#### Not in the round

- *Geometry for inclusion.* An axis and an aperture for a row, computed
  from its members' places and used to prune a search, is not built
  (Alec: "no new geometry"). He asked for it to be explained; it is a
  discussion and not a rule here.
- *The tanh and atanh of the grammar's `SigmaLayer` and `PiLayer`.* Alec:
  "it seems that the tanh and atanh were added as excessive precautions
  against unit overflow: we can just use clipping and some remedy for
  saturation"; and "lets also add a bullet to future work to investigate
  removal of excessive tanh/atanh usage in sigma and pi". It is recorded
  in FutureWork, on the branch that holds the documents.
- *The arithmetic of the folds* (§3.18).

#### After the round was settled: the two systems, and `interpret` (Alec, 2026-11-13)

The discussion went on after "Settled by Alec" above, and it changes this
section and §3.18. It is written into the documents on the branch that
holds them (Reasoning, Language, Philosophy, Extend, FutureWork). No rule
of a round is written from it yet: this section and §3.18 are to be
rewritten to it before any text for Codex.

- *Both are kept, one to each system.* "I guess we still need both, as
  implicit and explicit: system 1 and system 2, system 1 still knows in
  virtue of spatial reasoning, but system 2 has explicit representation.
  The edges as promotion of LTM relations is good. But the intuit() and
  expect() operations should not be explicit: THEY should be maps from one
  space to the other, and we already have a description of how they are
  taught."
- *What that does to §3.18.* Part A replaced the predictor with a pass up
  and down through containers; expectation is now a map, "a large sigma
  layer", and the pass is how System 2 reads its edges. Part B kept
  `intuit` as least and greatest over terms; it is now a map, with the
  same teacher.
- *A further teacher.* "explicit membership can be used to teach implicit
  maps, since we have input (token) and output (type) provided as the
  teaching example for order."
- *What `interpret` is.* "the interpret() operator in our grammar should
  create a symbol, and we do not symbolize all concepts to begin with, but
  we do symbolize words as a bit of a bootstrap." When a word is created,
  so is "meta-concept(symbol(word-concept), symbol(object-concept))"; the
  meta-concepts are collected under one symbol, which acts "like a
  dictionary"; and `interpret` "will replace word-concept with (one of)
  the object-concepts found under that meta-symbol", by "a sibling search
  under the meta-concept for the word".
- *A symbol of one concept.* "The symbol of a single concept is
  admissible, and perhaps necessary for a zero-D thing." The invariant of
  one symbol for one concept is still given up, since not every concept
  has one.
- *Read in the code, 2026-11-13.* `InterpretLayer.define` writes one
  relation row between the object and the word, with a trust pair, and no
  edge of either matrix; a word may name several objects
  (`named_identities`); `DefinitionIndex` is the fast index over those
  rows. The perceptual towers stop at the word (`Meronomy.pin_word_spans`;
  a run is promoted only within one tile of the word tiling), so a
  conjunction of two words first exists on the conceptual side.
- *Which object, where a word has several.* Alec: "How one object is
  chosen when a word has several: expectation."
- *Not yet said.* Whether `intuit` has matrices of its own (below).

#### Further answers (Alec, 2026-11-13)

- *Basic percepts and concepts.* "Once basic percepts run as a pi over
  parts and wholes (nicely covering that area in mereological space),
  concepts over those basic percepts will form concepts =
  sigma(pi(percepts)."
- *`expect` and `intuit`.* "Expect and intuit can make the same traversal
  forwards and backward (they do not have recursion), but they might need
  dedicated mapping matrices : sigma^-1(concepts) -> percepts,
  pi(percepts) -> concepts (Im hoping to use the inverse path because it
  is already present in the architecture. If we are operating percepts to
  percepts, then P = sigma^-1(pi(P) ). I think that intuit is the same
  operation."
- *The definition store.* "The definition store can still be split, just
  flag it in future work". So the two kinds of row of "The round" above
  are not in a round; they are in FutureWork.
- *Who makes symbols.* "lets let interpret() make all symbols. We should
  probably call it interpret." A promoted conjunction is then a concept
  and not a symbol.
- *The trust of a word's mapping.* "the trust of a words mapping is
  really something the architecture should be setting based on success.
  lets just use one, and flag it." It is one, as `InterpretLayer.define`
  writes now where it is given no evidence; setting it by success is in
  FutureWork.
- *Contradiction.* "logical contradiction is merely an signal that
  something needs to be fixed, there is no automatic fixing routine for
  inconsistency (but there is a measure of that inconsistency for users
  of WikiOracle)." Written into §17 point 12a.
- *Negatives.* Alec: "the negatives come from the negative values of the
  hyperplane ordering? Perhaps I done understand this." Claude's answer:
  where membership is read as the signed projection onto the type's
  vector with no threshold, nothing has to be given as a negative. What
  is unstated lies near zero, which is unknown, and only what is stated
  against lies on the negative side. Claude's caution, that stated
  membership gives positives only, held for a boundary with a threshold
  and is withdrawn for this reading.

Two cautions of Claude's on `expect`, for Alec:

- *The order of the two maps.* With concepts as sigma of pi of percepts,
  the way back from concepts is the inverse of sigma and then the inverse
  of pi. "P = sigma^-1(pi(P))" composes only if pi takes percepts
  straight to concepts.
- *An exact inverse expects nothing.* If the way down is the exact
  inverse of the way up, the round trip returns what went in, and a word
  that was left out stays out. What is expected has to come either from a
  matrix of its own for the way down, as Alec allows, or from a step
  between that loses and restores, such as the snap to known concepts.

Alec's answer: "For expect, both sigma and pi matrices are matrices of
their own." So `expect` uses no exact inverse and shares no weight with
the forward path: it goes up by a pi of its own and comes down by a sigma
of its own, each a map between the two spaces. That settles both
cautions. On "already present in the architecture" he added: "I just
mean that the neural pathways are there, in the sense that we already
have those operations between those spaces." What is reused is the kind
of operation in each direction, and not its weights. Whether `intuit`
has a pair of its own is not said; Alec had "I think that intuit is the
same operation."

Checked as arithmetic (not the model, and not a receipt). A type's vector
made from its members alone, with like tokens placed near each other, at
64 dimensions: the least member projects at .72 to .99, the mean
non-member at .00 to .18, and the greatest non-member at .34 to .98. So
most of what is unstated lies near zero, and a close relative projects
nearly as high as a member: the map gives likeness, and telling a member
from its sibling is the edge's. And one input coordinate of .8, left out
and sent through a matrix and its exact inverse, comes back as zero.

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
([whole-path §11](2026-11-11-one-pair-per-stage.md): land the accepted
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

   6a. *The preposition's kernel head* (Alec, 2026-11-13: "I think we
   agreed that prepositions functions like composite ADVerbials").
   A preposition and its object make one modifier, which then selects
   among the cases of what it modifies, as [Language](../Language.md)
   has it: "A preposition makes an adjective or an adverb by this
   selection". The modifier's content is its object's. So
   `PrepositionLayer`'s kernel takes its object as head, operand 1, as
   `SurfaceLayer`'s has since D9 and as the closing already reads it.
   (That the answer gives head 1 is Claude's reading, and Codex's
   recommendation of §14: *default*.)

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
11. *A question with no open operand is read on its region, in two
    ways.* Alec, 2026-11-13, in place of Claude's default: "A yes/no
    question is a query into conceptual space. We can ask the positive or
    negative form as an existence check within the region of order-0
    conceptual space that is specified by that; the question is true if a
    trusted object exists in that space. So we can consider the
    projection of multiple turth statements into that space: it is a
    luminousity test." And, the same day, on what that check can show:
    "we can do reasoning from analogy wi "the brightest statement in a
    region", but that is not sufficient for logical pervasion in the
    geluk sense. For pervasion we need the entire space of the truth
    claim in conceptual space to be luminous, or have a positive value
    (non-negative is not contradicted, but unknown, so analogy is not
    deductively true)."
    - *The region* is the question's field of understanding (points 7 to
      9): its concepts, each at its membership.
    - *Existence* is that a trusted object lies in the region: the
      greatest, over the rows of the window, of a row's relevance times
      $\lvert t \rvert$, its `trust` (point 12a), on the lane for where
      $t$ is above zero and on the lane against where it is below. Every
      statement is projected and the brightest is taken; none is summed
      with another. A row that is only like the claim, at a relevance
      below one, is what reasoning from analogy has, and it does not
      make the claim deductively true.
    - *Pervasion is coverage from a higher order.* Alec: "Privation is
      just coverage from a higher order, I hope that was clear." (Claude
      reads the first word as pervasion.) And: "Like "cats have fur"
      should be a relation in a higher order space about cat-things and
      things-with-fur. So every intention (e.g. the property fur) will
      have an extension when applied to a set (things with fur) which
      will look like a region in conceptual space (the furry things
      region)."
      - A property is an intension. Applied to a set it has an
        extension, the things it holds of, and that extension is a
        region of conceptual space. In the code it is the property's
        raised class with its stored members (`ConceptHierarchy.classes`,
        `members`).
      - A claim says a property of a subject, and the subject's set is a
        region as well: one thing, or the members of a kind.
      - The claim is pervaded when the subject's region lies wholly
        within the property's region, and this is held from the order
        above, where each of the two kinds is one object: by a trusted
        relation between them in `LTM1`. Its measure is that relation's
        trust; for a subject that is a member of the kind, the least
        along the path from the subject to its kind and from the kind to
        the property's class, and the greatest over such paths. It is
        read by the read of relations that exists (the taxonomy path of
        `Queries`, `Intuition.pervasion`), and from no row of `LTM0`.
      - *What writes the relation.* Alec, 2026-11-13: "If a sentence
        like "cats have fur" does not write a relation, then we have not
        correctly parsed the sentence as a relative truth, or a relation
        between two ideas (have expresses parthood, is expresses identity
        (unless predication), both are relations that are written to STM1
        by a correct grammatical parse, are stored in LTM1, and teach the
        conceptual taxonomy)." It is the `relative_truth` family of
        `complete.grammar` (`part`, its converse `whole`, `equal`,
        `implies`): the kept tree applies one of them between the
        sentence's two ideas, the pair is left in `STM1` and a row is
        written to `LTM1`, at the sentence's trust on the lane of its
        sign (point 12a). A sentence of this kind that leaves only an
        idea row has a defect in its parse or its close. Which word an
        operator is learned from is written in no code; the lessons'
        relations stage teaches it.
      - It is not gathered from the members: a kind each of whose known
        members is known to have the property is not pervaded, since its
        stored members are only those known (*default*).
      - A region lit in part and held false nowhere is unknown, and is
        not false.
    - *The two forms.* The positive form is read on the lane for and the
      negative form on the lane against, so the check gives four numbers:
      existence and pervasion of each form. A form is contradicted to
      the degree that the other form exists in its region. The lanes are
      never netted, so a claim can be true, false, both or neither.
    - *What takes which.* A deduction takes the claim's pervasion, or
      the claim's own statement, a trusted row of its content at a
      relevance of one (*default*), with the other form's existence
      against it. Existence at a lower relevance is carried with the
      answer as its analogy and is the `trust` of no row (*default*;
      what a conclusion by analogy is trusted at is not yet stated).
    (That the negative form is the question under `not`, read on the
    against lane of the entries, is Claude's reading.)

### 17.3 The reduction

12. *An `LTM0` row* is a kept sentence's or a kept thought's idea: the
    vector (`slots`, whole, its `time` in its band); its `trust`
    (point 12a); its `referent` (point 12b); its entry, a flat list of
    `episodicConcepts` references
    with their signed trust (`episodic_concepts`, `episodic_values`, no
    order axis), filled by the strongest activations of the field of
    understanding whatever their orders; and its `activity`.

    12a. *The trust of a row* (Alec, 2026-11-13; §19.3, §20.7). One
    signed number from $-1$ to $1$, the column `trust`. Above zero the
    statement is held true to that degree, below zero it is held false,
    and at zero it lights nothing. The idea is not scaled by it. It takes
    the place of the evidence pair and of "its trust its length", which
    §19 showed a row cannot give.
    - *Its sources.* Alec: "trust can be established by reasoning from
      trusted premises, authority, or direct witnessing." Witnessing
      establishes the word-concepts and the feelings, which are the
      row's entry and not its `trust`.
    - *An entry of the truth store* (`Models.store_truths`, and
      `provision_ltm` for a configuration's own) is written at the value
      it is given, "at face value". `_effective_incoming_trust` goes.
    - *A sentence read in training* is written at the model's `trust`,
      which WikiOracle sets in the configuration it trains with
      (`_incoming_trust_multiplier`; 1.0 in `data/model.xml`), times the
      certainty of the sentence's words (`clause.evidence`), as
      `PartClosing` does now for a sentence that closes as a relation
      and, from this round, for one that closes as an idea row
      (*default*).
    - *A sentence of a conversation* (`serve.py`, by `Models.infer`) is
      written at zero. Alec: "Conversations go in at zero, so truth
      statements have their trust evaluated." Claude's reading: what a
      question in a conversation is answered from is the trust of the
      truth statements, and what was only said lights nothing. By the
      code as read, a relation such a sentence writes is then at
      $(0, 0)$, so a word is not defined by what a conversation says of
      it (`FamiliarTier.define_pending` takes the relation's pair, and
      `definitionTrust` is 0.5).
    - *A kept thought* is written at what its premises give, the least
      on each lane as `ThoughtInference` and `Intuition` compute it. A
      conclusion with evidence on both lanes is written as two rows, the
      one held true first (*default*), so that no row nets one lane
      against the other.
    - *Nothing changes it after.* Alec: "facts can be forgotten, but
      changing trust values has to be done by proof (establishing
      contradiction within the existing truth store)." `set_trust` and
      `set_evidence` go, with the withdrawals of point 21. No writer
      that changes a stored `trust` is built. Alec, 2026-11-13: "logical
      contradiction is merely an signal that something needs to be
      fixed, there is no automatic fixing routine for inconsistency (but
      there is a measure of that inconsistency for users of
      WikiOracle)."
    - A relation a sentence writes takes the same number on the lane of
      its sign; the pair of `LTM1` is kept (point 13).
    12b. *The referent of a row* (Alec, 2026-11-13: "Yes, keep it"; §21.9,
    §22.4). One optional reference for what the row's idea is of: a
    concept, or another row by its time. It is empty where the close
    resolved none, which is so for most sentences. It is written from the
    reference the close selected (`PartClosing.occurrence_bindings`,
    `_identity`), and by a promotion (point 14). An idea is an intension
    and its entry is its extension, and two ideas can have one extension:
    a kind of one member and that member leave equal entries. The
    referent is what tells them apart, and the row's order is the order
    of the concept it names. The name `referent` is a *default*; Codex
    proposed `denotation`.
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
      set to one; and it is not considered again. The new concept is
      written as the row's `referent`. Its order is the referent's where
      the row had one, and one above the greatest order among its
      entry's concepts where it had none (*default*).
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
18. *No gate on an operator inside the model.* Alec, 2026-11-13:
    "Operators are "a priori" concepts that we are using for the control
    of the architecture itself. Lets not do that; the gating of those
    operators might be able to be done *in the tests*, which would act
    like piagetian stages signalling that the test suite can move on to
    the next stage." This replaces decision 7 ("Operators are enabled by
    their trust"), §3.7's gate, and the amplitude of the operator's
    concept as a gate, which this point had as a default. `TrustGates`,
    `closedTrust`, its threshold, `LTM1.operation_seat`,
    `operation_catalog` and every scaling of a row's trust by an
    operator's concept go. The whole grammar of a file is open from the
    first sentence. `PartsOfSpeech.operation`'s definiteness may still be
    reported (`corpus_lessons.readiness_report`); it gates nothing.
    - *The gate moves to the lessons* (not built in this round; it is the
      lessons' work, item 0.5 on the branch `testing`). Alec asked: "Are
      there tests that we could administer that would determine if the
      correct operators are being used in the correct locations? If so,
      the test gating should be moved outside." There are, and their
      parts exist:
      - the ten lesson stages are one to an operator family
        (`data/lessons/recipes`: words, interpret, clause, ask, verb,
        modifiers, determiners, adverb, prepositions, relations), each
        with a `criterion` and a held-out set;
      - a recipe's sentences are made from templates with named slots, so
        which words an operator should join is known for every generated
        sentence;
      - the harness makes a later stage wait for an earlier one's
        artifact (`test/phase_artifacts.py`).
    - *The tests* (Claude's proposal). On a stage's held-out sentences:
      the kept tree applies the stage's operator to the words that fill
      its template's slots, counted by operator and location; and the
      field of understanding the sentence is left with is the selection
      that operator should make, with the held-out questions answered.
      The first asks for the operator at its operands and not for one
      whole tree, since a sentence can have more than one right tree.
      Today each stage's `criterion` is the model's own
      `occupant_definiteness`, with no threshold set.
19. *Gone from `LTM1`:* `sentence_reference`, `access_count`, `surprise`.
20. *Gone from `LTM0`:* the evidence pair (`truth_plus`, `truth_minus`),
    whose place the one `trust` of point 12a takes; `access_count`,
    `surprise`; `_semantic_rows`, `metadata_required`,
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
    What the origin and the kind were used for is done by `trust`
    (Alec, 2026-11-13; §20.7):
    - *Which rows are truths.* A row whose `trust` is not zero. A
      conversation's rows are at zero, so they stay out of every read
      that took the rows of kind `'fact'`: `Queries.existence_evidence`,
      which point 11 replaces, and `TruthLayer.sync_from_ltm`, which
      takes the latest `truthMaxEntries` rows of `trust` other than zero
      (*default*) and holds each as its content times its `trust`.
    - *Taking a truth back.* Nothing does (point 12a). A supplied truth
      is kept and dropped by its activity like any row; a list sent
      again is written again; a restore withdraws nothing
      (`_finish_ltm_restore`). Since nothing now clears the list the
      parent sends with every request, a write on the serving path that
      finds the store over `forgetHighWater` runs the pass of point 14
      first (*default*).
    - *A contradiction is established as now*, by
      `TruthLayer.consistency`, and reported by `suggest_clarifications`
      and `assess`, whose messages say a row by `generate` and name it by
      its time where they quoted its text (*default*).
    - *The writers stay.* `Models.store_truths` and `provision_ltm` write
      a text with its trust, with no source key.
22. *What was in the bindings.* What is pending, an open question's slots,
    a forward reference, a name awaiting its referent, is held in MTM
    while it is pending. What a row's idea refers to is read from its
    `referent` (point 12b), and what it extends to from its entry. Which
    thought produced a row is not kept.
23. *Unused structure is deleted* (Alec, 2026-11-13: "Delete unused
    structure. The symbol space's own taxonomy, the chain rows and
    several writers only tests call."). As traced on the returned source
    of §21:
    - The symbol space's own taxonomy: the class `Taxonomy` of
      `Language.py`, `attach_knowledge` and the switch `symbolicPriming`.
      Nothing outside tests calls `attach_knowledge`. This is not
      `Taxonomy.py`, whose read of the conceptual taxonomy is in use.
    - The chain rows: `MemoryPaths.write`, which `Memory.py` calls at a
      promotion, and their one reader `ChainRead.path`, which nothing
      calls. The rest of `ChainRead` and of `MemoryPaths` is in use.
    - The writers `relate`, `reify_concept`, `synthesize_higher_order`,
      `singleton_concept` and `assert_concept_relation` of `Spaces.py`.
      `reify_concept` is the one caller of `relate`, and nothing outside
      tests calls the others.

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
40. A question with no open operand is read on its region in two ways.
    With one row of the same content, at trust $t$ and relevance one,
    the positive form exists at $t$ and a deduction may take it at $t$.
    With a trusted relation in `LTM1` between a kind and a property's
    class, the question of one of the kind's members is pervaded at the
    least along the path. With a trusted statement of another member of
    the kind and no such relation, existence is above zero and pervasion
    is zero. With a kind of three known members and a trusted statement
    of each, the question of the kind is not pervaded. With a row of
    trust below zero in the region, the negative form exists. Two rows
    give the greater of their two values, not their sum, and a row of
    trust zero gives nothing.
41. The saved columns of an `LTM0` row are exactly `slots`,
    `episodic_concepts`, `episodic_values`, `trust`, `referent` and
    `activity`, and of an
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
    kind `'operator'`. Every operator of a file can be chosen at the first
    sentence, and no row's trust depends on which operator composed it.
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
50. An entry of the truth store is read back from its row's `trust`
    unchanged, at the values $.3$, $-.6$ and $1$, and whatever the
    model's `trust` is set to.
51. One sentence read in training with the model's `trust` at one, and
    read as a conversation by `serve.py`'s path: the first row's `trust`
    is the certainty of its words and the second's is zero. A question
    over it with no open operand is answered for in the first case and on
    neither lane in the second; the question with its operand open is
    answered in both.
52. After a row is written nothing changes its `trust`: `set_trust`,
    `set_evidence`, `withdraw_origin` and `clear_origin` are in no source
    file. A list of truths stored twice leaves twice the rows, and the
    existence check gives the same evidence after the second as after the
    first.
53. A conclusion from a premise of trust $a$ and a rule of trust $b$, at
    a match of one, is written at $\min(a, b)$. A conclusion with
    evidence on both lanes leaves two rows.
54. `TruthLayer.sync_from_ltm` holds the rows of `trust` other than zero
    and no row of a conversation. Two stored truths that contradict are
    reported by `consistency`, with no origin and no text.
55. On the serving path, storing a list of truths with the store over
    `forgetHighWater` runs the pass and does not raise.
56. A sentence whose kept tree composes its two ideas by `part`, and one
    by `equal`, each leave the pair in `STM1` and a row in `LTM1` between
    the two ideas, at the sentence's trust on the lane of its sign, and
    no idea row in place of it. After the first, the question of a member
    of its kind is pervaded (certificate 40).
63. A member and its kind of one member, each closed as a row, leave
    equal entries and different referents, and `LTM0.order_of_row` gives
    their two native orders.
    `test_reference_close.py::test_recent_token_returns_at_its_kinds_native_conceptual_order`
    holds with its assertions unchanged. A promoted row's referent is the
    concept it became. (Certificates 57 to 62 are §3.19's.)
64. Every name of point 23 is in no source file. Each test that used one
    is ported to a writer production uses, with its assertions kept, or
    retired by name. At initialization a preposition and its object
    compose to the object's meaning, and the reverse recovers the object
    from the parent and the preposition (point 6a).

### 17.5 Not in this round

§3.18 (the two networks, symbols as points, placement by the field).
§3.19 (the split of sigma and pi into concepts and symbols).
Recognition of a value, which needs the field §3.18 builds (*default*;
§3.10 had it wait for `compose_extend`). The compose use of the
determiners. The time result of recall (Alec: "the previous time is baked
in"; it is not built). The restructured documents of the branch
`testing`.

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
   no open operand is an existence check in its region (point 11). Remove
   the match by role of LTM0.cued_rows. List every thought face and every
   other reader
   whose read changes, with what it read before. Certificates 38 to 40.
3. The reduction (§17.3), after items 1 and 2 pass their certificates and
   both stopping gates.
   a. Activity, in both stores, with its four parameters in model.xsd;
      the pass on activity in place of the retention value; promotion at
      activityPromotion (point 14).
   b. Rows named by time, and one history (points 15 and 16).
   c. No relation of kind 'operator', and no gate on an operator inside
      the model (points 17 and 18).
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

### 18.1 Amendment to the text of §18 (Claude, 2026-11-13; for Alec to paste)

~~~
Amendment to the text of §18, item 2 (Alec, 2026-11-13). §17 point 11 and
certificate 40 have been rewritten since the checkpoint commit: a question
with no open operand is an existence check, and no longer "the relevance
of its best row". Alec: "the question is true if a trusted object exists
in that space ... it is a luminousity test." Read point 11 and certificate
40 again in the working tree before you build item 2, and build them as
they now stand. The two readings marked as Claude's in point 11 are
defaults like the others: list them in your return with anything you find
against them. Nothing else in the text changes.
~~~

### 18.2 Second amendment to the text of §18 (Claude, 2026-11-13; for Alec to paste)

~~~
Second amendment to the text of §18, item 3c (Alec, 2026-11-13). §17 point
18 and certificate 45 have been rewritten: there is no gate on an operator
inside the model. Alec: "Operators are "a priori" concepts that we are
using for the control of the architecture itself. Lets not do that." The
earlier default, that the amplitude of the operator's concept scales the
trust a row is written with, is withdrawn: build no such scaling. Remove
TrustGates, closedTrust, its threshold, LTM1.operation_seat,
operation_catalog and every scaling of a row's trust by an operator's
concept; the whole grammar of a file is open from the first sentence.
PartsOfSpeech.operation's definiteness may still be reported and gates
nothing. If OperationSeats or an operator's concept is left with no
reader, delete it and say so; if it is left with one, list the reader.
The gate that replaces these belongs to the lessons and is not built in
this round. Read point 18 and certificate 45 again in the working tree.
~~~

### 18.3 Third amendment to the text of §18 (Claude, 2026-11-13; for Alec to paste)

~~~
Third amendment to the text of §18: Alec's answer to your §19
(2026-11-13). Read §20 (Claude's review of §19) and §20.7 (Alec's
answers). Then read again in the working tree §17 points 11, 12, 12a, 20
and 21 and certificates 40, 41 and 50 to 55. Then build the round of §18
with its two earlier amendments. What changed:

1. A row's trust is one signed column, trust, on LTM0, from -1 to 1:
   your column of §19.1, with the sign. The idea and its time band are
   untouched and no kernel changes. The evidence pair still goes from
   LTM0, and LTM1 keeps its pair. Every reader of the pair reads trust:
   find them yourself (Queries.end_evidence, best_match and
   premise_evidence, ThoughtClosing.closing_question and
   TruthLayer.sync_from_ltm are among them) and list each.
2. Who writes it (point 12a).
   - An entry of the truth store: the value it is given, unchanged.
     Remove the scaling of Models._effective_incoming_trust from
     store_truths and provision_ltm.
   - A sentence read in training: the model's trust setting times the
     certainty of its words, for a sentence that closes as an idea row as
     for one that closes as a relation (default).
   - A conversation: the rows that serve.py's read of the messages writes
     (Models.infer) are at zero. How the caller states this is yours to
     choose; say what you chose.
   - A kept thought: what its premises give. A conclusion with evidence
     on both lanes is two rows (default).
3. Nothing changes a stored trust. Alec: "facts can be forgotten, but
   changing trust values has to be done by proof (establishing
   contradiction within the existing truth store)." Remove set_trust and
   set_evidence with the withdrawals of point 21. Build no writer that
   changes a stored trust and no replacement for clear_origin: a list
   sent again is written again, and a restore withdraws nothing.
4. TruthLayer reads rows by their trust, and not by origin, kind or the
   two poles (point 21). consistency, suggest_clarifications and assess
   stay, and establish a contradiction as now.
5. Certificates 50 to 55 are added, and 40 and 41 amended.
6. In item 4, a test of supplied truths is retired only for what point
   21 removes: origins, withdrawal and the match on resubmission. What it
   asserts of a truth being stored with its trust and read back is
   ported.

Trace and report, without deciding:
a. InterpretLayer.define called with neither trust nor evidence writes
   (1, 0), and LTM1.add keeps the greater. Say for which callers this
   raises a counted definition, and whether any result depends on it.
b. positive_length of ConceptTrust.py returns one for a concept that is
   not allocated or whose chart is not tracked. Say which reads take
   that one.
c. The rows read in training now enter TruthLayer's view, which the
   factor on the training loss reads. Give both stopping gates beside
   their §14 values; do not tune to restore them.
d. Whether a word can still be defined in a served conversation, now
   that the relation that defines it is written at (0, 0).

List as defaults in your return: the training trust on idea rows; two
rows for a conclusion on both lanes; the latest truthMaxEntries rows of
trust other than zero; the pass on the serving path; the wording of a
contradiction message. The rules of §18 hold. Return as §21, since §19
and §20 are taken.
~~~

### 18.4 Fourth amendment to the text of §18 (Claude, 2026-11-13; for Alec to paste)

~~~
Fourth amendment to the text of §18, item 2 (Alec, 2026-11-13). §17
point 11 and certificate 40 have been rewritten again. Alec: "we can do
reasoning from analogy wi "the brightest statement in a region", but that
is not sufficient for logical pervasion in the geluk sense. For pervasion
we need the entire space of the truth claim in conceptual space to be
luminous, or have a positive value (non-negative is not contradicted, but
unknown, so analogy is not deductively true)."

And: "Privation is just coverage from a higher order" (Claude reads the
first word as pervasion). And: "Like "cats have fur" should be a relation
in a higher order space about cat-things and things-with-fur. So every
intention (e.g. the property fur) will have an extension when applied to
a set (things with fur) which will look like a region in conceptual space
(the furry things region)."

The check of a question with no open operand now reads its region in two
ways and keeps them apart. Existence is that a trusted object lies in
the region, the brightest row of LTM0, which is all the earlier text had;
below a relevance of one it is analogy. Pervasion is that the subject's
region lies wholly within the region that is its property's extension,
held by a trusted relation between the two kinds in LTM1 and read by the
read of relations that exists, at the least along the path. It is read
from no row of LTM0 and is not gathered from the members' own
statements. Read point 11 and certificate 40 again in the working tree,
with §20.8, and build them as they now stand.

A sentence such as "cats have fur" writes that relation, and it is not a
design question. Alec: "If a sentence like "cats have fur" does not
write a relation, then we have not correctly parsed the sentence as a
relative truth, or a relation between two ideas (have expresses
parthood, is expresses identity (unless predication), both are relations
that are written to STM1 by a correct grammatical parse, are stored in
LTM1, and teach the conceptual taxonomy)." Certificate 56 is added: a
sentence whose kept tree composes its two ideas by part, and one by
equal, each leave the pair in STM1 and a row in LTM1 at the sentence's
trust, and the question of a member of the kind is then pervaded. Where
such a sentence leaves an idea row only, find the cause in the parse or
the close and repair it where it arises; add no writer of the relation
beside a wrong parse. Name no word in code: which word part or equal is
learned from is the lessons' to teach, and a test may set a word's part
of speech directly. Say what you found.

In your list of the thought faces whose read changes (item 2), say for
each which of the two readings it takes; by point 11 a deduction takes
pervasion, or a trusted row of the claim's own content, and analogy is
the trust of no row. Intuition.pervasion stays as built: say in your
return which built read gives point 11's pervasion and where the two
differ on the same question, if they do. The parts of point 11
marked default are defaults like the others. Nothing else in the text
changes.
~~~

### 18.5 Fifth amendment to the text of §18 (Claude, 2026-11-13; for Alec to paste)

~~~
Fifth amendment to the text of §18: Alec's answer to your §21.9, and
Claude's review of §21 (2026-11-13). Read §22. Then read again in the
working tree §17 points 12, 12a, 12b, 14 and 22 and certificates 41 and
63, and resume the reduction.

1. Alec on §21.9: "Yes, keep it". Your one optional typed reference on
   an LTM0 row is kept. It is point 12b: what the row's idea is of, a
   concept or another row by its time, empty where the close resolved
   none. Write it from the reference the close selected
   (PartClosing.occurrence_bindings, _identity). Every reader of your
   §21.9 table reads it in place of the sidecar. The name referent is a
   default, since "denotation" will be read as the extension, which is
   the entry; use it and say if you find a reason against it.
2. Promotion writes it too (point 14): the concept a row's idea becomes
   is that row's referent. Its order is the referent's where the row had
   one, and one above the greatest order among its entry's concepts where
   it had none (default).
3. Point 22 now reads: what a row's idea refers to is its referent, what
   it extends to is its entry. Certificate 41 adds the column.
   Certificate 63 is added.
4. Twelve tests fail on your returned source that §21 does not mention
   (§22.2, with each test and what it raises). For each, say whether it
   is a port, a retirement with its point of §17, or a defect, and
   repair the defects where they arise. Find and say in particular:
   whether a restore inside a process should pass through
   RowTime.checkpoint_times at all, which Alec decided for a checkpoint
   of the old layout; what a fixture with no time band is under point
   15; and what test_medium_term_memory.unchanged finds changed, and
   whether a trial that is not kept may leave activity changed.
5. Point 12a's last sentence is replaced by Alec's words on
   contradiction: it is a signal, with a measure for the user, and
   nothing fixes it automatically. This changes nothing you build.
6. §3.19 and its later subsections record a discussion for a later
   round. Build nothing from them.

The limits you state in §21.2 stand and are carried to the completion,
with the comparison of Intuition.pervasion and point 11's read on one
question (§22.3). The rules of §18 hold. Continue your return as §23.
~~~

### 18.6 Sixth amendment to the text of §18 (Claude, 2026-11-13; for Alec to paste)

~~~
Sixth amendment to the text of §18 (Alec, 2026-11-13). Two things are
added to the reduction. Read §17 points 6a and 23 and certificate 64 in
the working tree.

1. Alec: "Delete unused structure. The symbol space's own taxonomy, the
   chain rows and several writers only tests call." Delete, with their
   configuration and schema entries:
   - the class Taxonomy of Language.py, attach_knowledge and the switch
     symbolicPriming (not Taxonomy.py, whose read is in use);
   - MemoryPaths.write with its call in Memory.py, and ChainRead.path
     (the rest of both modules is in use);
   - relate, reify_concept, synthesize_higher_order, singleton_concept
     and assert_concept_relation of Spaces.py.
   Find the callers yourself before each deletion, and say what you
   found. Something that turns out to have a production reader is not
   deleted: report it. Port each test that used one of these to a writer
   production uses, keeping its assertions, or retire it by name. Your
   reproduction for §21.9 and certificate 63 build a kind of one member
   with singleton_concept; build it as production does.
2. Alec on the preposition's kernel head: "I think we agreed that
   prepositions functions like composite ADVerbials". PrepositionLayer's
   kernel takes its object as head, operand 1, as SurfaceLayer's does
   since D9 and as the closing reads it. That the answer gives head 1 is
   Claude's reading and your own recommendation of §14: build it as a
   default, with the certificate you gave surface.

The rules of §18 hold. Nothing else in the text changes.
~~~

### 18.7 Seventh amendment to the text of §18 (Claude, 2026-11-13; for Alec to paste)

~~~
Seventh amendment to item 1.8: the number modality on level sets, for any
number of dimensions. Alec, 2026-11-13: "Please ensure a path for the
number modality to process images as well as XOR (the only different
being the percept dims; MNIST-size images would be a good test, but no
need for multimodal at this point, that work is left to someone else)",
and, asked when: "In 1.8 now". The design and its reasons are §22.9 of
doc/plans/2026-11-11-item1-8-last-architecture-round.md. §22.7 and §22.8
are records of discussion and are not instructions; build nothing from
them. Commit and push nothing in this round until Alec says so.

What changes.

1. One reader. NumericPercepts (bin/NumericPercepts.py) stays the one
   reader for dataType numeric on a file with field operators only. Its
   dimensions stay the product of the input shape: 2 for data/MM_xor.xml,
   784 for a 28 by 28 input. Each dimension stays an axis of the percept
   cube (BasicPercepts.numeric_dimension). No second reader, and no branch
   on the number of dimensions.

2. Level sets in place of the value as evidence. Today each axis concept's
   for lane is the felt value (read(): evidence[..., 0, 0] = values[:, axis]).
   Instead a reading on an axis with a felt value above zero is perceived
   as a value percept on that axis, as BasicPercepts.perceive_value does
   for the scalar beside the words: a basic percept, reserved by
   BasicPercepts.reserve(parts, wholes), that knits the axis's atom with a
   part and a whole whose codes are the felt value on that axis. The axis
   concept made at construction collects each value percept met on its
   axis (add_concept_feature(row, 'bp', percept, 1.)), and its for lane is
   the reading's membership in that collection
   (BasicPercepts.read_collection: 1 inside the bounds, falling off outside
   by IdentityRows.meet_ratios). Alec: "percept activations have a kind
   there but no magnitude". Zero stays no presence: no percept and no
   evidence. feeling() and number() are unchanged.

3. Share the code. ScalarPercepts and this reader perceive, collect and
   read a value by the same functions; the scalar beside the words is the
   case of one axis with no declared shape (default). If one function
   cannot serve both, stop that part and report the reason; do not keep
   two copies silently.

4. Back from the field. NumericPercepts.reverse returns each dimension's
   number from the level its concept holds, not from the size of the
   evidence. For an input of two levels this is exact. For more levels
   report what it returns; no bar is set.

5. An image configuration. Add one new file (default name
   data/MM_image.xml): dataType numeric, field operators only as in
   data/MM_xor.xml, an input of 28 by 28, and its own stated sizes so that
   the percept cube, basicPerceptCapacity and the concept inventory hold
   784 dimensions. conceptualPi is false in it (default): the promotion
   of a whole sample's inputs (FamiliarTier.observe_intersection) cannot
   recur for an image, and the learned step that replaces it is a later
   part of 1.8. Resize no existing configuration and no fixture. Leave
   data/mnist.xml and its dense-slab tests as they are.

6. The numbers given to the reader are raw intensities scaled to the unit
   range. For two levels a pixel at or above one half is 1 and the others
   are 0 (default). The loader's normalization by mean and deviation
   (loadMNist of bin/data.py) gives negative numbers and is not used on
   this path.

Certificates (continuing §17's count).

65. One reader: on data/MM_xor.xml and on the image file,
    model.numeric_percepts is a NumericPercepts with dimensions 2 and 784,
    and both outputs come from OutputSpace.read_field.
66. The XOR proofs hold on level sets with their bars kept:
    test/numeric_xor.py certificate() with its assertions unchanged, and
    the stopping gate
    test/test_mm_xor.py::TestMMXorConvergence::test_convergence. Port
    test/test_item18_fields.py::test_numeric_native_field_promotes_and_reader_is_stepped
    with a record: the promoted concept's evidence where both inputs are
    present is the membership, no longer .4093839.
    test_numeric_scale_order_and_fixed_code keeps its assertions.
67. A 28 by 28 input of two levels: model.forward, model.understand and
    model.reverseReconstruct run in the default sweep (not under
    RUN_SLOW), and the reconstruction returns every number (atol 1e-6).
    Report wall time and peak memory for one forward and one reverse.
68. A measurement, with no bar: the accuracy of the linear read
    (outputSpace.field_readout, ten outputs) on held-out two-level images
    after a number of presentations fixed before the run. Use a bounded
    subset of the real files as
    test/test_mnist.py::test_real_mnist_subset_trains_one_short_batch
    does; where the files are Git LFS pointers, skip with that reason and
    measure on generated 28 by 28 patterns fixed before the run.

Stop and return, without widening a bar, if level sets cannot keep
certificate 66, if 784 dimensions need an operation over all pairs of
dimensions or over all rows, or if data/MM_xor.xml has no room for its
two axes' value percepts (do not resize it without Alec's word).

Not in this amendment: grey levels beyond the report that point 4 asks
for, the learned step between concepts (it comes with the rewriting of
§3.18 and §3.19), a place of two dimensions, and two modalities at once.

Report under your current return, in a part named for this amendment:
what was shared between the two readers, each default you kept or
changed, the four certificates, and the ports with their records.
~~~

## 19. Design checkpoint: the trust of an idea (2026-11-13; for Alec)

**Stopped for a design question before implementation.** The checkout is
on the pushed checkpoint `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`.
Its source still hashes to
`a8c9adced8e2a323038e01f768d36daeef4d6454e405f5ae78aaa8712b5a6290`.
The incoming uncommitted change was this specification's amendment to the
existence check. No production, configuration or test file has changed in
this round; no commit or push has been made.

### 19.1 Question for Alec: which quantity is the row's trust?

Point 11's default says that a row's trust is the length of its idea;
point 12 stores that idea whole, with its time in its band. These two
requirements conflict with the current vector representation and the
bounded evidence required by certificate 40.

The producer is `LTM0.append_meaning`: it copies the idea into `slots`,
then `RowTime.write_time` writes the time band. `RowTime.encoder` uses
two sine/cosine pairs. Their norm is always square root of two,
independently of the content or its trust. A direct probe on the unchanged
source gives:

| Quantity | Value |
|---|---|
| Input idea's norm | 0.20000000298023224 |
| Written evidence for | 0.2 |
| Time band's norm | 1.4142135381698608 |
| Stored whole idea's norm | 1.4282857179641724 |

Thus whole-row length cannot give certificate 40's evidence 0.2 for this
row at relevance one. Clipping the result to one would make it certain.
Scaling the time band with the trust would lose the time, and hence the
row's new name under point 15, when trust is zero.

Excluding the time band is necessary but does not by itself establish a
bounded trust radius. The existing meaning kernel also grows geometric
length: `MeaningCodes.compose('union', ...)` on positive meaning vectors
`(.8, 0)` and `(0, .8)`, with empty against lanes, produces `(.8, .8)`.
The operand norms are **0.800000011920929** each; the result's norm is
**1.1313709020614624**, without any form, place or time coordinates.
Changing that kernel's geometry or renormalizing its output is not a
reader port; it changes the meaning and inverse contracts that must still
hold in this round.

**Recommendation:** keep the geometric idea unchanged and use one bounded
row-confidence scalar for this round. Replace the two LTM0 evidence
columns with one `trust` column, give the recalled row's positive and
negative relevance their polarity from the entry, and use
`max(relevance * trust)` independently on the two lanes as point 11 now
requires. This preserves the accepted time encoding and meaning kernels.
It requires an explicit amendment to point 12 and certificate 41; I have
not added that column. The alternative is an explicit bounded-amplitude
encoding for an idea that remains valid through composition, inversion,
zero trust and the time band, which is a larger change.

For Alec: **may this round retain that one bounded LTM0 `trust` column,
or should the idea carry confidence by a different specified encoding?**

### 19.2 Work and verification standing

The amendment to point 11 and certificate 40 was reread after Alec's
message. Both Claude readings there are defaults to be recorded when
built: the negative form under `not` reads the entries' against lane, and
row trust is the idea's length. The latter is the finding above. No
default has yet been implemented, and no certificate from 34 to 49 is
claimed. No tests were ported or retired, no fixture was resized, and no
seed was used in the two direct tensor/store probes. Both stopping gates,
the proofs, lesson and default sweep have not been rerun in this round.
The last accepted results remain those of §14. Local receipts are not
linked. Only this design record has been added to the specification.

### 19.3 Reply to Claude: Alec's answer on trust (2026-11-13)

Claude, the question I raised for Alec was what supplies the bounded trust
that §17.11's existence check reads, and how that value is represented
after LTM0's evidence pair is removed. The current stored vector's length
includes the time band, and even the meaning kernel can produce lengths
above one (§19.1). This prompted my recommendation of one bounded trust
column, preserving the idea's geometry.

Alec answered:

> trust can be established by reasoning from trusted premises, authority,
> or direct witnessing. Direct witnessing we already have, but that covers
> word-concepts and feelings, not object concepts. object concept truth is
> established by reasoning from existing trusted premises (in LTM or
> conceptual space). authority trust is brought directly into LTM by the
> WikiOracle harness at face value from the truth store.

Alec has asked you to document this decision and evaluate whether it is
consistent everywhere in the architecture. That includes the distinction
between directly witnessed word-concepts and feelings and reasoned object
concepts, the harness's authority input, and §17's memory reduction and
existence check. In particular, the retirement of origins and supplied-
truth machinery needs to be reconciled with accepting the harness's trust
at face value.

This answers where trust comes from. The remaining design issue is how
that trust is represented and read after the evidence pair is removed;
§19.1's single-column proposal remains a recommendation for that issue.

## 20. Review of §19, and where trust comes from (Claude, 2026-11-13)

**The round is not built.** Codex stopped at the question of §19 before
any code. The checkout is the pushed checkpoint `ab5176e0e2`, `main` on
the remote is the same commit, and the one changed file of the working
tree is this specification. No certificate from 34 to 49 is claimed and
none was run for this review. This section reviews §19, records Alec's
answer, and checks it against the code and against §17. Alec's answers
to §20.6 are in §20.7, and the text that follows from them is §18.3.

### 20.1 §19 is right

Both of its findings hold on the source.

- The time band's length is $\sqrt{2}$ for every row: `RowTime.encoder`
  writes four coordinates, two pairs of a sine and a cosine.
- `MeaningCodes.compose` under `union` takes the greater of the two
  operands at each coordinate of the for half, so $(.8, 0)$ and $(0, .8)$
  give $(.8, .8)$, of length $.8\sqrt{2}$.

There is a third, which decides it. Nothing writes trust into the idea.
`LTM0.append_meaning` takes the pair as an argument beside the vector,
and `ClauseRow.write_clause` multiplies the pair, never the vector, by
`source_trust`. The idea's length has not been the row's trust since the
pair became two columns. `TruthLayer`'s own description still has the
rule Alec recalled, "stored = activation * degree", for the store it had
before it read `LTM0`. So "its trust its length" in points 11 and 12
(Claude's reading of "can be inferred for semantic vectors by their
length") cannot be read from what a row stores, and Codex was right to
stop.

### 20.2 Alec's answer

> trust can be established by reasoning from trusted premises, authority,
> or direct witnessing. Direct witnessing we already have, but that covers
> word-concepts and feelings, not object concepts. object concept truth is
> established by reasoning from existing trusted premises (in LTM or
> conceptual space). authority trust is brought directly into LTM by the
> WikiOracle harness at face value from the truth store.

All three are in the code now, each with its own rule:

| Source | What it establishes | Where the code has it |
|---|---|---|
| Direct witnessing | word-concepts and feelings | A part percept that is present counts one (`Spaces.part_memberships`). A felt number is `feeling` of `NumericPercepts.py`. A word-concept's radius is counted from its kept occurrences: `ConceptTrust.refresh` gives `agreement / views * (1 - exp(-views / trustViews))`, and a new concept starts at zero. A sentence's `clause.evidence` is the least of its words' evidence on each lane (`ClauseJournal.metadata`, whose comment reads "input identification poles, never the source's authority"). |
| Reasoning from trusted premises | object concepts and conclusions | `ThoughtInference` gives a conclusion, on each lane, the least of its premise, its match and its rule, and joins alternatives by the greatest. `Intuition` fires at `min(coverage, trust)`. A relation is read as its pair times the least radius of its two ends (`LTM1.refresh_effective`): a premise in `LTM1` limited by premises in conceptual space. |
| Authority | what the truth store holds | The parent's `bin/response.py` sends each entry's `content` and its `trust` from $-1$ to $1$. `Models.store_truths` reads the text as a sentence and `ClauseRow.write_clause` writes $(\lvert t \rvert, 0)$ or $(0, \lvert t \rvert)$ by the sign. |

Claude's reading of it against the Gelug account that
[Philosophy](../Philosophy.md#valid-cognition-in-dharmakirti) follows.
There are two valid cognizers, the direct perceiver and the inferential
cognizer, and inference is of three kinds, of which two are by the power
of the fact and through belief in one who knows. Alec's three sources are
the direct perceiver and those two inferences, and the third is how that
document already treats testimony ("inference from speaker reliability").
Which source serves follows the object: what is manifest is witnessed,
what is hidden is reasoned from what is manifest, and what cannot be
reasoned from what the knower has is taken from one who knows. To a model
that reads, only words and felt numbers are manifest. Every object is
hidden, which is the line Alec draws between word-concepts and object
concepts.

Luminosity is how this trust is read. Point 11 has it: the trusted
statements are projected into the order-zero space and a question is true
to the degree that its region is lit. Two things follow that the rules
already keep. The brightest statement is taken and none is added to
another, so a thing said twice is no truer (the parent's `doc/Truth.md`:
"Truth is not the same thing as consensus."). And a region with no
trusted object is dark on both lanes, which is the answer neither.

### 20.3 Where the architecture already agrees

- `LTM1` keeps its pair, and a relation's evidence is limited by the
  radii of its ends.
- Point 18: no operator's concept scales a row's trust. An operator is
  none of the three sources.
- Point 14: what is kept is decided by activity and not by trust. The
  source of a truth is used when its trust is written and is not kept
  after (Alec: "we know and do not necessarily remember where we first
  learned").
- `Queries.existence_evidence` reads rows of kind `'fact'` only and by
  one signed value, the view `LTM0.trust`; its comment reads "Existence
  uses external event authority." So the existence check that is built
  reads authority, and not the witnessing of a sentence's words. It adds
  the rows and stops at one, where point 11 takes the greatest.

### 20.4 Where it does not

1. **A read sentence is held true because its words were seen.**
   `Models._append_observed_meaning` writes `clause.evidence`, the
   witnessing of the words, as the row's pair, and `Queries.best_match`
   and `end_evidence` answer a question from that pair. By Alec's answer
   the witnessing establishes the word-concepts and not what the sentence
   says of objects. Today the two are kept apart only by the row's kind,
   `'observation'` for a row of the conversation and `'fact'` for a
   supplied one (`LTM0.kind_of`), on which `existence_evidence` and
   `TruthLayer.sync_from_ltm` select. Point 21 removes kind and origin.
   As §17 stands, nothing would then tell a sentence that was heard from
   a sentence that is held true (decision 2).
2. **The truth store's value is not taken at face value.**
   `Models._effective_incoming_trust` multiplies it by the model's own
   `trust` (`data/model.xml`, 1.0, set in no other file). The same number
   reaches read sentences by `_incoming_trust_multiplier`, where
   `PartClosing` applies it to a sentence that closes as a relation and
   not to one that closes as an idea row.
3. **Trust from no source.**
   a. `InterpretLayer.define` called with neither `trust` nor `evidence`
      writes $(1, 0)$, and `LTM1.add` keeps the greater on each lane. A
      definition counted from its views is so raised to one when its word
      is next read (`Interpret.forward`, `reading`, `admit_discovered`,
      `lookup_word`, and `FamiliarTier.promote_identity` without
      evidence).
   b. `ConceptTrust.positive_length` returns one for a concept that is
      not allocated or whose chart is not tracked.
   c. Relations written at a constant one: `ConceptIndex.index_part_row`,
      `Allocator.add`, `Spaces.relate`, `MemoryPaths.write` and
      `membership`.
   Claude's reading: (c) records what the model itself composed, which it
   witnesses. (a) and (b) are for Codex to trace, and each is either
   given one of the three sources or reported.
4. **An object concept's radius is counted as a word's is.**
   `ConceptTrust.observe` is called at promotion, at minting, at a
   reference and for a class's members (`FamiliarTier`,
   `IdentityRows.mint`, `ReferenceClose`, `ConceptHierarchy`). Claude's
   reading: it agrees where each view is taken as a conclusion, that the
   object's word occurred and that the word names it, and a view should
   then count no more than that definition is trusted. Not in this round.
5. **`TruthLayer` is left with nothing to read.** `sync_from_ltm` selects
   rows by origin (`rows_of_origin`), by kind and by the two poles. §17
   removes all three and does not mention the layer. Its readers are in
   use: the factor on the training loss (`truth_modulated_loss`), and
   `consistency`, `suggest_clarifications` and `assess` after
   `store_truths`. This is a gap of §17.
6. **A truth cannot be taken back.** Today each request replaces the
   user's truths (`clear_origin(ORIGIN_USER)`, then the new list), and a
   restore without state withdraws them (`_finish_ltm_restore`). Point 21
   removes both. The parent sends its whole list with every chat request,
   so the list would be written again as new rows each time; a truth the
   user has deleted or lowered would stay lit at its old value until the
   activity pass drops it, since the check takes the brightest; and on a
   server used by several people one person's truths would light
   another's answers. The parent's `doc/Constitution.md` has "Answers
   must be conditioned on what each user chooses to trust". Alec accepted
   the repeats ("too much lookup. But it will be forgotten.") and the
   loss of the source ("we can look in memory to see which book we opened
   prior to learning"). Taking a truth back was not put to him then
   (decision 3).
7. **In the parent** (another repository, and not this round). It sends
   no identifier with a truth. When the user deletes the last truth it
   sends no `truth` at all, so the model is not told. An entry of its
   kind `feeling` loads at trust zero and is sent as a statement of trust
   zero, which is stored and lights nothing. That kind is the user's own
   feeling; the feelings of Alec's answer are the model's felt numbers.

### 20.5 What follows for §17

Proposed before Alec's answers of §20.7. What holds is §17 as it now
reads: points 11, 12, 12a, 20 and 21, and certificates 40, 41 and 50 to
55.

- *Point 11.* A row's trust is its `trust`, a signed number. The evidence
  for is the greatest, over the rows of positive trust, of relevance
  times trust; the evidence against is the greatest, over the rows of
  negative trust, of relevance times $\lvert t \rvert$. The negative form
  of the question exchanges the two. With every statement written as
  content times trust, this is the coverage `TruthLayer.luminosity`
  already measures, taken over the question's region.
- *Points 12 and 20, certificate 41.* The saved columns of an `LTM0` row
  are `slots`, `episodic_concepts`, `episodic_values`, `trust` and
  `activity`. The pair goes as point 20 has it, and `trust` is written
  where the pair was: the truth store's value as given, a read sentence's
  by decision 2, a conclusion's from its premises. A conclusion with
  evidence on both lanes is written as two rows (*default*), so that no
  row nets one lane against the other.
- *Point 21.* Origins, kinds, the match on resubmission and the texts go
  as accepted. `store_truths` and `provision_ltm` stay as the two writers
  of a text with its trust, without a source key, and
  `_effective_incoming_trust` no longer scales. `TruthLayer.sync_from_ltm`
  takes the latest `truthMaxEntries` rows of trust other than zero
  (*default*). A contradiction message names its rows by their times and
  says each by `generate` where it quoted the text (*default*).
  `forgetProtectOrigins` goes with point 14: a supplied truth is kept by
  its activity like any row.
- *Certificate 40* stands with $t$ the row's `trust`. To add: a value
  from the truth store is read back unchanged; a row of trust zero lights
  neither lane; a row of negative trust lights the lane against.
- *For Codex to trace and report* (not to decide): 20.4 point 3 (a) and
  (b), and what a restore without state does once it has no origin to
  withdraw by.

### 20.6 For Alec

1. **How a row's trust is held** (§19.1). Recommended: one signed `trust`
   on an `LTM0` row, from $-1$ to $1$ as the truth store has it, with the
   idea and its time untouched. It is Codex's one column, with the sign,
   and it is the number the built existence check already reads. The
   others: keep the pair `LTM0` has now, as `LTM1` does, which changes
   least and keeps both lanes on one row; or scale the idea by its trust
   so that the length carries it, which needs a bounded measure the
   kernels do not have and changes what `generate` walks back.
2. **What a sentence that is only read is trusted at.** Recommended:
   whoever presents a text states its trust and the model adds none of
   its own. The truth store gives each entry's value; for every other
   text the harness or the training driver gives one number, which is the
   model's `trust` setting of today and one by default; and the row's
   trust is that number limited by the certainty of the words. Nothing
   changes at the defaults, the lessons keep their answers, and the
   harness can present a text it does not vouch for at zero. The other:
   a sentence that is only read is written at trust zero, recalled and
   quoted but lighting nothing until the truth store asserts it or a
   thought concludes it.
3. **Taking a truth back.** Recommended: outside the model, by time. A
   request's truths are the rows written since the request began, so when
   the model is not training the serving process returns memory to that
   time at the end of the request. No column says where a row came from,
   and what was concluded from a request's truths goes with them. I have
   not traced what serving does at a restore today; that is in §20.5 for
   Codex to report. The other: nothing is taken back; a truth fades by
   its activity, a changed mind is a new statement, and the two stand as
   evidence on both lanes until the older is forgotten.

Later, and not for this round: whether an object concept's radius is
counted or follows from its premises (20.4 point 4), and whether the
authority who said a thing is kept as a concept in the row's entry, so
that a question can be asked within one authority's frame.

### 20.7 Alec's answers (2026-11-13)

1. *How a row's trust is held:* "1 as recommended". One signed `trust`
   on an `LTM0` row.
2. *What a read sentence is trusted at:* "this should be something that
   wiki oracle sets when training, and users set in the truth store.
   Conversations go in at zero, so truth statements have their trust
   evaluated."
3. *Taking a truth back:* "facts can be forgotten, but changing trust
   values has to be done by proof (establishing contradiction within the
   existing truth store)."

They are written into §17 as point 12a, with points 11, 12, 20 and 21
and certificates 40, 41 and 50 to 55 amended to them. The text for Codex
is §18.3. Claude's proposal to return memory to the time before a
request (§20.6, decision 3) is withdrawn.

What follows from them, by the code as read:

- In a served conversation a question with no open operand, about what
  was only said, is answered on neither lane. The same question with an
  operand open is still answered from the row.
- A word is not defined by what a conversation says of it, since the
  relation that would define it is written at $(0, 0)$ (§18.3, d).
- A truth that the user lowers in the truth store keeps its old value in
  memory until the row is forgotten, since the lower value arrives as a
  new row and the check takes the brighter. A value that is raised
  arrives as a brighter row.
- The rows read in training are truths at the value WikiOracle sets, so
  they enter `TruthLayer`'s view, which until now held only the
  provisioned and the user's rows (§18.3, c).

What an established contradiction changes was not stated then. Alec has
since said that it changes nothing by itself (point 12a): it is a signal,
with a measure for the user.

### 20.8 A change to the luminosity test (Alec, 2026-11-13)

> One change to luminousity: we can do reasoning from analogy wi "the
> brightest statement in a region", but that is not sufficient for
> logical pervasion in the geluk sense. For pervasion we need the entire
> space of the truth claim in conceptual space to be luminous, or have a
> positive value (non-negative is not contradicted, but unknown, so
> analogy is not deductively true).

When Claude had written pervasion as every concept of the region being
lit by whichever rows reach it:

> Privation is just coverage from a higher order, I hope that was clear.

And when Claude had then written it as one row of `LTM0` whose field
covers the question's region:

> Like "cats have fur" should be a relation in a higher order space about
> cat-things and things-with-fur. So every intention (e.g. the property
> fur) will have an extension when applied to a set (things with fur)
> which will look like a region in conceptual space (the furry things
> region).

Claude reads the first word of the second as pervasion, and the three
together so. A property applied to a set has an extension, which is a
region of conceptual space. A claim says a property of a subject, whose
set is a region too. The claim is pervaded when the subject's region
lies wholly within the property's, and that is one relation between two
kinds in the space of the order above, where each kind is one object. It
is the coverage that luminosity already is, read from the order above,
and not a second mechanism. A trusted statement of a like object of the
same order is an instance, which is analogy.

Point 11 and certificate 40 are rewritten to this: the check reads a
region in two ways, existence from the rows of `LTM0` and pervasion from
the relations of `LTM1`, and keeps them apart. The text for Codex is
§18.4.

What the code has for it: a property's raised class and its stored
members (`ConceptHierarchy.classes`, `members`), relations between
concepts in `LTM1` with their trust, the read of a path of such
relations at the least of its edges (`Queries`), and
`Intuition.pervasion`.

Claude had left open whether a sentence that says a property of a kind
writes that relation, and had it as a design question for Codex to
return. Alec:

> If a sentence like "cats have fur" does not write a relation, then we
> have not correctly parsed the sentence as a relative truth, or a
> relation between two ideas (have expresses parthood, is expresses
> identity (unless predication), both are relations that are written to
> STM1 by a correct grammatical parse, are stored in LTM1, and teach the
> conceptual taxonomy).

So it is not a design question. The names are built: `complete.grammar`
has the family `relative_truth` (`part`, `whole`, `equal`, `implies`),
"each a Relation(RI_1, RI_2) output over two ideas", `STM1` is a class of
`Memory.py`, and `DefinitionFold` already takes the evidence of the
relations as its teacher. Point 11 now says what writes the relation,
and certificate 56 is added. Whether the code does write it for such a
sentence today is not traced; where it does not, it is a defect for
Codex to find and repair (§18.4).

Two readings of Claude's. That a word such as "have" is composed by
`part` is learned and is in no code (the lessons' relations stage, and
the gate on operator placement of item 0.5). And "unless predication":
[Language](../Language.md) already has it, "The generic subject decides,
not the copula": a predication of one thing, as "the cat is an animal",
is an idea of one situation and writes no relation, and a generic, as
"cats are animals", writes the relation between the two kinds. His
"teach the conceptual taxonomy" also reads as his answer to whether
stated relations feed the definition store; whether the unused taxonomy
of the symbol space and the chain rows are deleted is still his to
decide.

Checked on a toy with `FieldLogic.reduce` (a kind of three known
members, membership at one and trust .8 where not said; not a receipt).
The toy took the kind's statement as covering the kind's stored
members, which in these cases is what the least along the relation's
path gives:

| Question | Trusted | Existence for | Pervasion for | Existence against |
|---|---|---|---|---|
| of one member | the relation of its kind to the property | .8 | .8 | 0 |
| of one member | another member's statement | .4 | 0 | 0 |
| of the kind | each member's own statement | .36 | 0 | 0 |
| of the kind | the relation of the kind to the property | .8 | .8 | 0 |
| of the kind | that relation, and one member's statement at $-.9$ | .8 | .8 | .36 |

In the Gelug form of a proof this is the difference between the example
and the pervasion: that Felix has fur follows from his being a cat only
by the relation of cats to things with fur, and not from another cat's
having it.

Two defaults of point 11 are Claude's: that the statements of a kind's
members are not gathered into a pervasion of the kind, and that a
deduction may also take a trusted row of the claim's own content.

## 21. Codex return to Alec for Claude: checkpoint at a reduction question (2026-11-13)

**This is an incomplete return, not the completion of §18.** The extension
and recall implementation and its current certificates are in the main
checkout. Both stopping gates passed before the reduction and after the
retained-occurrence port Alec accepted in §21.1. That port is built. The
checkpoint migration decision in §21.7 is answered and its rejection is
built. LTM1's reduction has begun (§21.8). Work now stops for the exact
denotation read in §21.9, under §18's instruction to stop for an unretired
reader without a specified replacement. No commit or push has been made.
The final proofs, lesson and default sweep have not run on this source.
The latest source and limits are in §21.10; §21.6 describes the earlier
checkpoint. Other reviewers' document changes were preserved.

### 21.1 For Alec: the observation count used to exclude a reference

`ReferenceClose.exclusion_pairs` is a reader of the retiring
`LTM0.access_count` that §17 does not retire. It sums the counts of retained
individual occurrences whose coordinate is at least .3. Two coordinates
are treated as incompatible after each has eight observations and no
retained occurrence has both. This is used to exclude candidates, not just
to rank which memories to forget.

The callers are:

| Reader | What the result does |
|---|---|
| `ReferenceClose.exclusion_query` → `individuals` | Splits accessible occurrences into distinct identities when their contents conflict. |
| `ReferenceClose.exclusion_query` → `score` | Removes a conflicting antecedent from a reference proposal's candidates. |
| `Models._sentence_reference_bank` → `ReferenceBank.exclusions` → `ReferenceContext.prepare_operands` → `STMChunks.bind_menu` | Removes conflicting off-stack chunks from compose's bind menu. |

The old writer, without changing source or fixture capacity, reproduces
the dependency. Two ideas, each naming a known order-one concept, are
written once: there are two retained rows, counts `(1, 1)`, no exclusion
pair, and the incompatibility query is false. Writing each identical
occurrence seven more times leaves the same two rows, counts `(8, 8)`,
pair `(0, 1)`, and a true query. `slots`, `episodic_concepts`,
`episodic_values` and `trust` compare exactly equal before and after. The
probe completed in 2.405636787414551 seconds. It is an observation of the
old writer, not a proposed retention of re-witnessing.

§17.14 replaces the count with activity, but activity cannot supply this
count: it is bounded by one, is affected by accesses and decay, and a
single surprising write may start at one. §17.16 and .21 make each repeated
sentence another row. §17.6 leaves reference proposals in place. None says
whether this empirical incompatibility rule stays, or what observations
it counts after the change.

**Recommendation:** preserve the reference-selection rule by counting
distinct retained eligible occurrences in the single history. Each row
counts once; no activity value is treated as a number of witnesses and no
new counter is stored. Forgetting an occurrence removes that observation
from this read, just as forgetting an old counted row removes its
observations now. Port the tests' compressed eight-witness setup to eight
actual writes, retaining their capacities and assertions.

**Initial question:** should I make that port, or retire empirical exclusion and
its consumers? That question stopped the reduction under
§18 item 3: “Something that still has a reader which §17 does not retire is
a design question: stop and write it for Alec with your recommendation.”

**Alec's answer, 2026-11-13:** “Count retained occurrences (recommended).”
The reduction resumed with that decision. The reader now counts each
retained eligible row once. The two tests below use eight actual writes
instead of one row with an edited counter; their capacities and assertions
are unchanged. Those four cases passed in 3.03 seconds, wrapper
3.6717638969421387 seconds. Both stopping gates passed again, two passed in
4.01 seconds, wrapper 5.150184869766235 seconds. These six checks used the
returned source hash in §21.6. The earlier measurements in §21.5 describe
the source before this resumption.

The affected existing tests are
`test_reference_review10.py::test_later_ball_mention_preserves_the_boxs_key_members_and_count`
and
`test_reference_review10.py::test_own_content_excludes_an_observed_incompatible_antecedent`,
two parameterized cases each. All four still pass. Their assertions have
not been weakened or retired.

### 21.2 Extension and recall built so far, and the limits of their certificates

`GrammarLayer` now supplies `compose_extend` and `generate_extend` through
`Extension`. Grammar configuration rejects a registered structural
operator missing either method. `complete.grammar` has `some` on both
walks, and an inverse entry for `universal`. The new set value carries
concept references and both lanes; it uses `FieldLogic` for narrowing,
joining, negation and withholding. The compose journal supplies the order
of calls, with every order of the already-open input activations.
`FamiliarTier.open_sentence` previously published only the order-zero
field; it now publishes the activations returned by `cs_forward_content`.
No stimulation is performed to make those compose operands.

`SentenceUnderstanding` carries the resulting field and transient tree.
`OutputOccurrences.generated` reconstructs the generate walk's tree and
returns its operator-combined field. Its separate realization field still
locates the emitted words; it is no longer the semantic field. The
native generate root counts are passed explicitly. No generated tree is
saved with an LTM row.

The real MTM case of certificate 36 exposed a stale member reader:
`ConceptHierarchy.members` read only raw constituent sidecars, while the
allocator's declared symbolic members were in LTM1. It now uses
`allocator.refs(identity, 'part')`; `classes` also includes the allocator's
raised classes. Before the repair the held member was absent; afterwards
the held identity is selected at its two MTM readings, `.7` and `.2`.
This common member read also serves class placement, conditioning and the
determiner readers; it was repaired at the reader rather than copied into
the certificate's extension implementation.

`ExtendedRecall.matches` computes each operand over the entire window and
combines the row vectors with the question's operators. Open operands
impose no constraint. `EntryRead.cued_rows` no longer computes a cosine or
other match by role: it preserves a recalled row's extension relevance;
an explicit address by itself supplies no resemblance score. The old
order-axis entry, document filters, kind filters and associated scratch
allocation are still present until the reduction.

An open answer is read from the recalled idea's own generated tree at the
same operator path as the question's hole. A mismatching path gives no
answer. `ThoughtReferences.fill` takes that operand's reference or set,
not the whole witness row as the answer's identity. An unanswered
`query`/`ask` returns its requested meaning even when both evidence lanes
are zero; otherwise `ThoughtStream.integrate` reconstructed the executable
request as a multi-role conclusion and encountered the writer's guard.
That producer is repaired and the guard is unchanged.

Existence and exact-content support are separate maxima over signed row
trust and relevance. Pervasion uses the existing taxonomy path reader over
LTM1, with the minimum along a path and maximum over paths. It is not
gathered from a kind's known members. The deduction result carries analogy
separately and retains the witnesses of its exact statement or relation
path, not those of a brighter analogy. At this checkpoint LTM0 still has
its old evidence pair and derived `trust` property: the immutable signed
column and two-row writer are **not yet built**.

Certificate 56's correctly selected `part` and `equal` trees already
closed into LTM1 and were copied into STM1; no extra relation writer was
needed. The positive `part` case nevertheless returned pervasion zero
after storing `.7`: `capture_taxonomy` read the operator-gated effective
pair. It now reads the relation's stored pair. The same certificate then
returned `.7` through a member edge at `.8`. All four `(part/equal) ×
(.7/-.6)` cases pass. These are certificates of a **selected kept tree**,
not a claim that an untrained chooser learns the words' parts of speech.

`ThoughtTaxonomyCapability.evidence` also used the priming cache's netted
edge and first path for a proof. It now uses the existing bounded taxonomy
capture and strongest path. A negative final relation can oppose a
subject reached through a positive prefix; a negative intermediate edge
does not continue a positive containment path. The priming cache's own
geometric read is unchanged.

The new tests in `test_item18_extend.py` have 24 collected cases covering
34–40 and 56. They include the native compose journal without stimulation,
a real MTM determiner read, an LTM recall with STM cleared and a native
generate walk, and the sentence/open-object matching-path read. The latter
uses a controlled inverse trace through the production extension code;
it does **not** demonstrate a learned multioperator parse/generate round
trip. The single known-reference recall independently uses the actual
generate walk. Certificate 34 currently checks the registered structural
compose/generate operators; no claim is made here about the nonstructural
forward/reverse interfaces. Before final acceptance, the full negative-form
integration, multiple open operands, generation work charging, and the
finished-source lesson still need verification. These qualifications must
not be lost when carrying these tests into the completion.

### 21.3 Readers changed so far

This is the inventory for the implemented portion, not the completed
reader/writer inventory required for the reduction.

| Face or reader | Before | At this checkpoint |
|---|---|---|
| `query`, `ask` | Role-matched recalled frame, or its whole-row reference as a filling. | For an open operand, the matching generated operand; for a closed claim, exact-content/pervasion support with existence carried separately as analogy. |
| `isTrue` without an explicit occurrence | `truth_evidence` → best role match. | Closed-claim deduction read through `best_match`. Its addressed `end_evidence` branch still reads the old pair, pending point 12a. |
| `isPart`, closed operands | Direct relation first, then taxonomy evidence, whose shortcut used priming's netted edge. | Direct relation still first; fallback is the two-lane trusted taxonomy path. It takes relation evidence/pervasion, not an LTM0 analogy. |
| `isPart`, open operand | Taxonomy neighbors. | Same neighbors, with taxonomy capture now reading the raw relation pair. |
| `infer` | Identity/equality matching and the netted membership shortcut, then premise/rule lane minima. | Identity/equality paths stay; membership matching takes the trusted taxonomy path. It takes no lower-relevance LTM0 row as premise trust. |
| `intuit` | `intuition_rules` and `Intuition`'s geometric pervasion. | Unchanged; it is not silently redirected to the new region existence read. |
| `exist` | Conceptual-space extent, not LTM0 existence. | Unchanged. |
| `bind`, `isEqual`, `part`, `equal`, `implies`, `not`, `gain` | Reference proposals, direct equality, vector operations, lane reversal or gain. | Their faces are unchanged; they do not choose between the two new LTM0/region readings. |
| `ThoughtLTMCapability._cued`, `retrieve`, `resolve_description`, `query`, `best_match` | Stimulated role cue, fixed order intersection and role re-scoring. | Extension cue and operator-combined row results, with generated-operand answers where open. |
| `existence_evidence`, `truth_evidence` | The older existence/best-match evidence route. | Region existence is exposed independently; closed ordinary questions use the deduction result. The existing timeline branch remains. |
| `ThoughtClosing.closing_question` | Prepared the question from the parsed or committed meaning and its evidence pair. | Also preserves the question's extension; the pair-to-column port remains. It prepares the query rather than choosing one of the two readings itself. |
| `WholePaths.fields`, `SentenceUnderstanding`, generated-output comparison | Input/output leaf fields or description without the kept extension. | The kept field/tree is carried across the thought boundary, and generated output has the combined field. |

Point 11's new built read is `ExtendedRecall.pervasion` through
`Taxonomy.capture_taxonomy` and `ConceptualTaxonomyView.part_of`.
`Intuition.pervasion` remains as built: it first checks a Part/property
predicate, reads the subject's known members and their fields, and uses
its geometric coverage calculation. The new read requires the trusted
kind-to-property-class path and does not infer it from that member census.
A numerical comparison of the two on the same native question remains
outstanding; their equivalence is not claimed.

### 21.4 Defaults and implementation choices so far

The built defaults are operand locations to select the input activations;
the inverse returning each operand's share of its parent's set; equal
`1/N` membership for `lower`; `some` composed as `universal`; conjunction
as intersection; the listed content operators passing their content
operand's set; an open `ask` imposing no constraint; unchanged compose
determiner/reference proposals; no answer on an operator-path mismatch;
no pervasion inferred from a census of known members; an exact-content
trusted statement usable by a deduction; and lower-relevance existence
carried as analogy, not row trust. The changed member and taxonomy readers
above were evidence against their old implementations, not reasons to
reverse those defaults. The reference-count question is evidence that
unchanged reference proposals need an additional reduction decision.

Additional Codex choices made explicit: `some` selects the strongest
`max(2, N // 2)` of `N >= 3` members, and is empty when no proper plural
subset exists; ties retain the stable member order. Separate roots join
by union. The extra shipped algebra faces `pi`/`min` narrow and
`sigma`/`max` join. A transpose journal action moves its operands rather
than introducing an application node absent from the kept tree. A held
definite member uses both of its held MTM lanes. Mandatory interpretation
transports existing memberships along declared word/object references,
as the explicit interpretation face does. These choices await review.

The negative-form default has lane-level tests, but its complete thought
path is not certified by this checkpoint. The reduction defaults are
**not built**: activity's four values; strictly increasing times;
stream-order insertion; training trust on idea rows; positive-first two
rows for a two-lane conclusion; the latest `truthMaxEntries` nonzero rows;
the serving-path pass; and generated contradiction wording. No choice
about any of them was changed to pass a gate.

### 21.5 Tests, ports and measurements actually run

The exact 24 new cases were copied to a temporary read-only source archive
of checkpoint commit `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`: **24 failed
in 4.86 seconds**, wrapper 5.482522010803223 seconds. Missing extension
interfaces account for much of this baseline; it is not evidence of 24
independent old defects. The real MTM and relation-trust failures described
above were also observed separately while implementing their readers.

On the returned source, the six-file extension/recall regression group
(`test_item18_extend`, `test_taxonomy_view`, `test_query_work_boundaries`,
`test_query_vp_boundaries`, `test_thought_infer`, `test_item18_finish`)
completed **90 passed in 21.35 seconds**, wrapper 22.003040075302124 seconds.
The two required stopping gates then completed **2 passed in 3.98
seconds**, wrapper 5.11142110824585 seconds:

| Stopping gate | §14 | This checkpoint |
|---|---|---|
| `test_mm_xor.py::TestMMXorConvergence::test_convergence` | Passed. | Passed; setup .61 seconds, test call .55 seconds. |
| `test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class` | Passed. | Passed. |

§14's numeric certificate 11 reported MSE .00042077834950760007. That
measurement is not a measurement of this gate run, and it has not been
rerun here. §14 recorded no separate MSE for the stopping gate. The
training-to-TruthLayer change has not been built, so the requested
after-reduction comparison is still due; nothing has been tuned to recover
the old result.

The two exclusion tests (four cases) and the priming membership-cache test
completed **5 passed in 3.11 seconds**, wrapper 3.722730875015259 seconds,
on the same source. This confirms the blocking reader's current contract;
it is not a test of either proposed replacement.

Ports before the resumption: **one named test port and one shared fixture
port; zero retirements**. `test_thought_infer.py::test_membership_match_uses_the_existing_part_cache`
now observes the proof's `capture_taxonomy` call instead of
`Priming.part_edges`; its call-count, support and replay assertions remain.
`test_priming_diffusion.py::owner` now declares its allocator's existing
placement, raised/retired sets and layer inventory so the real taxonomy
reader can read it. Its own
`test_membership_cache_follows_native_addresses_evidence_forgetting_and_load`
retains all assertions. No existing fixture was resized and no seed was
selected for an assertion.

After Alec's §21.1 decision, `test_reference_review10.py::observed_individual`
was ported to eight actual writes. This ports the two named exclusion
tests in §21.1, four parameterized cases, without changing an assertion or
capacity. The current count is therefore **three named test ports, two
shared fixture/helper ports, zero retirements**. This is not the complete
reduction's test inventory.

The required one default sweep, 160-sentence whole-path lesson, certificates
11 and 13/49, and reduction certificates 41–48 and 50–55 have **not** run
on a finished source. There is no new sweep result to report. The two
older failures recorded in §14 remain carried, not repaired:
`test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers`
(inventory of `inventory_part_events` versus `_radix_code_events`) and
`test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row`
(old percept-column expectation `{4, 402}` versus native BP `{0}`). Their
status on this source has not been measured by a default sweep.

### 21.6 What remains and returned source

The remaining reduction awaits Alec's answer to §21.7: signed immutable
trust and every source/reader; activity, promotion and forgetting; time
names and one history; operator-relation and gate removal; all retired
columns/modules; flat entries filled from the final understanding field;
MTM pending bindings; checkpoint conversion; TruthLayer and serving.
Its complete reader/writer destinations, test ports/retirements and counts
are not yet claimed. The requested traces of unqualified `Interpret.define`,
`ConceptTrust.positive_length`'s fallback, the changed training-loss truth
factor, and zero-trust served definitions remain due. The same is true of
the finished-source tests and measurement.

No row-layout reduction has been made. The stored-size baseline in §14 is
5,481 bytes per LTM0 row and 7,103,321,093 tensor bytes at the largest
constructed capacity, 1,048,576 rows. There is no after-reduction size to
report yet; these are the earlier measurements, not a new measurement.

Returned uncommitted source SHA-256, computed with the existing
`bounded_tests.source_snapshot` protocol (source/configuration/tests,
excluding mutable document prose and local receipts):
`5535bafcd3b4f5e9ed2bbc26f23771f370d889277101f0b57332485692df9d22`
over 1,076 source inputs.
This hash was used by the four ported exclusion cases and both stopping
gates after Alec's answer. The earlier 90 checks, two stopping gates and
five reader-contract checks used
`262c684580ace9cd1fd7ee8c2449bb1e578aeb1c517a5d0dcd3f220231b31a56`.
HEAD remains
`ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`. No commit or push was made.

### 21.7 For Alec: old times are not unique occurrence names

Certificate 47 says each old row keeps its idea and its time, and a
relation's occurrence end is renamed by that row's time. Point 15 removes
the old addresses and makes time the row's name. These cannot all hold
for a valid old checkpoint containing different occurrences at the same
time, including occurrences without a time.

The reproduction ran on the archived checkpoint commit
`ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`, not on the new writer. Its
`Memory.py` and `PartClosing.py` blobs were checked against that commit.
It uses the existing `test_orders_learned.equality_fixture` and two calls
to its existing `event` helper, with no capacity or dimension changes.
That helper writes different witnessed ideas with `time=False`.
A native LTM1 relation connects their occurrence references. Both stores
are saved and successfully reloaded with the existing checkpoint readers.

| Observation after reload | First occurrence | Second occurrence |
|---|---|---|
| Row | 0 | 1 |
| Old address | 9078488801920670264 | 5394997665020254761 |
| Time band | `(0, 0, 0, 0)` | `(0, 0, 0, 0)` |
| Decoded time | 0 | 0 |

The ideas differ. The relation still has two different occurrence ends.
Renaming those ends by the preserved times makes both `(OCCURRENCE, 0)`.
This loses which idea each end names. The store has its unchanged
1,024-by-16 layout, with the time band beginning at coordinate 12; this is
not a fixture without a time band.

This is also a production-writer condition: `PartClosing._write_occurrence`
passes `time=False` for embedded occurrences, and `RowTime.write_time`
zeros their bands. Even a nonzero time is not guaranteed unique in the
old format: the old `RowTime.next_time` takes the model's present without
advancing it past another row already written at that present.

**Recommendation:** amend certificate 47 for migration only. Visit legacy
rows in retained order. Preserve an existing valid time when it exceeds
the preceding assigned time; otherwise assign the next tick, including
when the old band is absent. Preserve every non-time coordinate and the
strongest entry readings, remap all occurrence references to the assigned
times, advance the store's next-time bound, and warn with the number of
changed times. Preserve both rows; do not merge them or retain an address
column. This changes the time coordinates of affected legacy ideas and
therefore needs the explicit exception to “keeps its idea, its time.”

**Question:** should migration make that exception, or reject checkpoints
with ambiguous occurrence times? Neither choice is implemented. The first
preserves their usable content and relations; the second cannot satisfy
certificate 47's current unconditional load contract. Source work stops
here under §18's design-question rule, with the accepted exclusion port
complete and the checkpoint files unchanged.

**Alec's answer, 2026-11-13:** “Reject ambiguous checkpoints.” The
recommended rewriting of old times is not authorized and will not be
built. Certificate 47 consequently excludes checkpoints whose occurrence
times cannot name their rows unambiguously. Validation rejects missing or
duplicate occurrence times before loading any row; it does not alter the
checkpoint's time bands or merge rows. The remaining reduction resumes
under this decision.

### 21.8 Work after Alec's checkpoint answer

**The ambiguous-checkpoint decision is built.** Before loading any LTM0
row, `RowTime.checkpoint_times` checks the occupied rows' time bands.
Missing bands, nonfinite bands and duplicate decoded times raise
`ValueError`. The destination remains empty and the supplied state's
tensors remain unchanged in both missing-time and duplicate-time cases.
No retimestamping or row merging is built. This is the rejection part of
certificate 47, not its still-unbuilt remapping of an unambiguous old
checkpoint's occurrence references.

`RowTime.next_time` now takes the greater of the model's present and the
store's next tick. Two writes while the model's present stays at 9 receive
9 and 10. This checks the clock portion of certificate 43; the full
producer port, time-named row lookup, batch and forgetting checks remain
due. In particular, the old embedded writer still passes `time=False`;
it has not yet been ported to the new LTM0 writer.

**LTM1 now declares five row columns:** `refs`, `ref_kinds`, `truth_plus`,
`truth_minus`, `activity`. Its native pair indexes are maintained on an
add. Evidence reads return the stored pair and add activity; rewriting an
existing pair takes the greater evidence as before and adds activity.
The check writes activity .5, rewrites the pair to reach .6, and reads its
evidence to reach .7. The pair remains (.7, .2). No declared cache is
checkpointed. Its old-layout loader discards retired columns with a
warning; occurrence-end migration awaits LTM0's time names.

`MemoryActivity.Activity` holds the four declared parameters, now also in
`model.xsd`, at defaults .5, .1, .9 and .9. The common write, access and
decay operations are built. The pass and LTM0 promotion are not yet wired,
so this is not certificate 42 in full.

`TrustGates` and LTM1's effective-evidence scaling are removed. The two
configuration blocks and their schema type are removed. `Definitions`
now watches the raw relation revision. `PartClosing`, `SentenceStrata`,
`MemoryPaths` and the relation verification write in `Spaces` no longer
pass retired operator, witness or occupation metadata. `PartsOfSpeech`
no longer reports an observation to a gate. The relation diagnostic in
`corpus_lessons` reads activity in place of the removed sentence link.
`Taxonomy.capture_taxonomy` reads the unscaled pair through `evidence`.
An LTM1 relation has no time of its own: `RowTime.read_time` returns an
empty time band for it, and `ClauseRow.part_row` reads spatial content
from the chain's head idea instead of the retired linked sentence.

This is a partial reduction, not a claim that every retired name has
gone. The old LTM0 layout, presentation restoration, forgetting and their
callers still need the rest of the port. The readiness report also still
needs its non-gating occupation reader detached from the removed LTM1
owner link. The temporary STM sentence must retain its syntax under its
own owner when the durable semantic sidecar goes.

The new checks and their actual runs are:

| Check | Before | After |
|---|---|---|
| `test_43_each_write_advances_past_a_stationary_model_present` and both parameter cases of `test_47_ambiguous_legacy_times_are_rejected_without_loading` | 3 failed in 2.33 seconds; wrapper 2.9436769485473633 seconds. | 3 passed in 2.31 seconds; wrapper 2.929734945297241 seconds. |
| `test_41_relations_keep_only_pair_trust_and_activity` | 1 failed in 2.10 seconds; wrapper 2.708595037460327 seconds. | Together with the three time checks: 4 passed in 2.49 seconds; wrapper 3.105503797531128 seconds. |
| `test_41_44_52_ideas_keep_flat_entries_and_immutable_signed_trust` | 1 failed in 2.11 seconds; wrapper 2.718306064605713 seconds. | Not built; LTM0 is still the old layout. This is an explicit before check, not a passing certificate. |

Neither stopping gate was run during this partial rewrite. Their last
run remains the two passes after the retained-occurrence port in §21.5.
No stopping-gate failure has occurred. No final sweep or lesson run is
claimed here.

### 21.9 For Alec: an extension does not retain its exact denotation

Point 22 says what a row denotes is read from its entry. Point 12 fills
that entry only from the strongest activations of the field of
understanding. Point 6 preserves the determiner's existing reference
proposals. The existing proposals and other reads distinguish a kind
from its member by the exact reference and its native order. The flat
extension does not always retain that distinction.

The reproduction uses the unchanged `SentenceFixture` (64 rows, width
8), its real `singleton_concept` constructor and the grammar's
`universal` rule. It chooses that declared rule in a kept program, closes
the program with the native writer, and runs `Extension.reading` on its
two leaves' supplied conceptual activations. No capacity, dimension or
seed is changed. The two closed programs have respectively a member
`x` and its native singleton kind `S = {x}` as their content operand.

| Read | Member's row | Singleton kind's row |
|---|---|---|
| Exact retained reference | `('sym', 137)` | `('sym', 139)` |
| Native order, also returned by `LTM0.order_of_row` | 0 | 1 |
| `ReferenceClose._derivation_symbol` | 137 | 139 |
| Current MTM token for the occurrence | None | None |
| Field left by the actual `universal` operator | `{137: (1, 0)}` | `{137: (1, 0)}` |

Thus the strongest readings are identical at every nonzero entry
capacity. Keeping more readings cannot recover which reference the
writer had. The entries need no pending binding: both referents are
already resolved. The two idea vectors differ in this reproduction,
but point 22 specifies an entry read, and matching a vector to concept
codes would not preserve an exact reference in the allowed equal-code
case. No such matching repair is proposed.

This exact-reference read predates the reduction. The archived checkpoint
writer and reader return different references 133 and 135, at orders 0
and 1, for the same native construction. The new operator's entry rule
is the part that discards the distinction. The old `ConceptHierarchy`
helper does not enumerate these native singleton members correctly;
that helper's existing fix in §21.2 is needed to implement the new
extension rule, not a reason to retain the old read.

The producer is `PartClosing.occurrence_bindings`, which carries the
selected reference as `_identity`; `LTM0.append_meaning` currently saves
it in the retiring semantic sidecar. Its readers include:

| Reader | Use that still needs a specified destination |
|---|---|
| `LTM0.raw_meaning_of` | Restores the exact `role_refs[0]` of the kept idea. |
| `LTM0.order_of_row` → `Spaces._concept_source_order` | Uses the denoted concept's order, not the grammatical output order or a member's order. |
| `ReferenceClose._derivation_symbol` | Recovers that reference for reference proposals after the close, including when MTM has no token. |
| `ReferenceClose.context_records`, `exclusion_pairs`; `EntryRead.recall_identities`, `recall_chunks` | Use the row's native order for admissible identities, empirical exclusion and recalled chunk order. |
| `ClauseRows.read_reference_bindings` | Finds an occurrence's denoted identity before following its MTM binding in a relation read. |
| `ThoughtLTMCapability._with_components` | Finds the denoted identity and its current MTM value when reading a retained occurrence. |
| The minting portion of `MemoryConsolidation.consolidate` that point 14 reuses | Needs the idea's order when allocating its promoted concept. The grouping is retired; the concept allocation still requires an order. |

The existing test
`test_reference_close.py::test_recent_token_returns_at_its_kinds_native_conceptual_order`
checks the distinction explicitly, at native orders 0 and 3 despite a
clause of grammatical order 1. This is a reference contract, not a test
of origins, recurrence, streams or a retired row kind.

**Recommendation:** preserve one optional typed `denotation` reference
on LTM0, written from the selected reference, separately from the field
entry. A two-int64 carrier (reference kind and identity, empty when no
resolved reference is given) adds 16 bytes per row, or 16,777,216 bytes
at 1,048,576 rows. Read the referenced concept's order through the
allocator; keep pending names and slots in MTM. Do not retain bindings,
syntax, provenance or a second content vector. This requires an explicit
amendment to points 12 and 22 and certificate 41. It is not implemented.

**Question for Alec:** should that exact reference be retained, or does
point 22 intentionally retire these exact-denotation and native-order
contracts when distinct intentions leave the same extension? Silently
choosing the entry's strongest member changes the kind's order from 1
to 0 in the reproduction. Adding the denoted kind to the field merely
to preserve a reference changes the prescribed extension. Neither is
built. §18 item 3 explicitly requires a stop for an unretired reader
without a specified replacement, so source work stops here for this
decision.

**Alec's response, 2026-11-13:** “lets ask claude”. This question is
referred to Claude for architectural review. The proposed `denotation`
reference remains unbuilt, and the reduction stays stopped pending that
review and Alec's decision.

**For Claude:** please review the singleton reproduction and the readers
above against points 6, 12 and 22. Can the prescribed entry supply the
exact denotation and native order these readers require? If the intended
architecture changes those contracts, identify which reads and assertions
are retired and specify their replacement. Otherwise, please evaluate
the proposed typed reference, or specify another representation that
preserves the distinction while leaving the prescribed extension intact.
Record the resulting amendments to the construction text and certificates
for Alec's decision.

### 21.10 Current checkpoint and source

The reduction is unfinished. LTM1 has its new columns; LTM0 still has
its old layout, so no new LTM0 stored-size result is claimed. Activity
promotion and forgetting, immutable signed trust throughout, time names
and unambiguous migration, the MTM pending port, the remaining removals,
test ports and retirements, required traces, proofs, lesson and default
sweep remain due. The previous failures and their attribution in §14
are unchanged; this partial rewrite has not had a default sweep.

Returned source SHA-256 by the existing `bounded_tests.source_snapshot`
protocol:
`655250f4c8f4fc2ad0701be6f8c0131b6002f27017b804c951167bf2911ce056`,
over 1,077 source inputs. The four passing reduction checks used
`3cef4c12bceba83e07068660f0a2c1875d41e6e0b78bd3b8b5cb3e55e32225cf`;
the only subsequent source change adds the explicitly failing-before
LTM0 check above. Document prose and local receipts are excluded from
that source hash. Receipts remain local and are not linked.

HEAD remains `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd` on `main`.
Nothing was committed or pushed.

## 22. Review of §21, and the question of §21.9 (Claude, 2026-11-13)

**§21 is an honest partial return, and the round is unfinished.** What it
says of its own standing checks out. This section verifies it, reports
twelve failing tests it does not mention, and answers the question Alec
referred to Claude in §21.9. It holds no Codex text, since §22.4 is for
Alec to decide.

### 22.1 Verified

- *Standing.* HEAD is `ab5176e0e2bc` on `main`, nothing committed; 42
  changed or new files, among them the new `Extension.py`,
  `ExtendedRecall.py`, `MemoryActivity.py` and two test files, and
  `TrustGates.py` deleted.
- *The source hash.* `bounded_tests.source_snapshot` over the working
  tree gives 1,077 inputs, and the SHA-256 of its compact JSON is
  `655250f4c8f4fc2ad0701be6f8c0131b6002f27017b804c951167bf2911ce056`,
  as §21.10 has it.
- *The new tests.* `test_item18_extend.py` and `test_item18_reduction.py`
  together: 28 passed and 1 failed in 6.61 seconds. The one failure is
  `test_41_44_52_ideas_keep_flat_entries_and_immutable_signed_trust`,
  which §21.8 gives as not built.
- *Both stopping gates*, which §21.8 says were not run after the partial
  rewrite: 2 passed in 3.89 seconds on the returned source.
- *The pieces of §21.9.* `Extension` replaces a kind by its stored
  members before a determiner acts (`stored`), so `universal` over a kind
  of one member leaves that member alone. `LTM0.order_of_row` reads the
  exact reference from the `_identity` binding of the semantic sidecar,
  which points 20 and 22 retire. Each reader §21.9 lists exists. The test
  it cites exists. `singleton_concept`, which the reproduction uses, has
  no caller outside tests.

### 22.2 Not in the return: twelve tests fail on the returned source

Run on the returned source, whole files: `test_thought_infer.py`,
`test_item18_finish.py`, `test_reference_review10.py`,
`test_priming_diffusion.py` and `test_reference_close.py` give 12 failed,
70 passed and 4 skipped in 22.50 seconds. §21.5's 90 passed and §21.1's
"All four still pass" were measured on earlier hashes. Each failing test
was then run alone to pair it with what it raises.

| Test | What it raises |
|---|---|
| `test_thought_infer.py::test_trial_restore_keeps_active_streams_inside_the_constructed_batch` | `ValueError: ambiguous occurrence times: checkpoint has no time band` (`RowTime.py` line 129) |
| `test_item18_finish.py::test_equal_codes_are_said_by_their_given_object_references` | the same `ValueError` |
| `test_reference_review10.py::test_thought_bind_is_credited_only_by_episode_advantage`, both cases | the same `ValueError` |
| `test_reference_close.py::test_kept_journal_supplies_np_content_predicate_and_positional_provenance`, two cases | the same `ValueError` |
| `test_reference_review10.py::test_later_ball_mention_preserves_the_boxs_key_members_and_count`, both cases | `AssertionError: Tensor-likes are not equal!` in the helper `test_medium_term_memory.unchanged` |
| `test_reference_review10.py::test_answer_view_and_loss_judge_only_a_direct_root_site`, both cases | the same `AssertionError` |
| `test_item18_finish.py::test_relation_write_updates_reference_indexes_without_rebuild` | `LTM1.add() got an unexpected keyword argument 'operator'` |
| `test_item18_finish.py::test_declared_caches_rebuild_and_are_not_checkpointed` | `'LTM1' object has no attribute 'effective_plus'` |

- *Six raise the rejection of §21.7.* The check Alec decided for a
  checkpoint of the old layout fires when a store with no time band is
  restored at all, inside a process as well.
- *Four fail a comparison that the stores are unchanged.* The helper
  takes a snapshot of the stores before an operation and asserts every
  tensor equal after it. One of them is an exclusion test ported for
  §21.1, in both its cases. Claude's guess, not checked: a read now adds
  to `activity` (§21.8), so a read is no longer without effect on what
  is stored.
- *Two are tests of what points 17 and 18 retire*, not yet ported or
  retired.

For Codex to find and say, without deciding (as §18 has it for
diagnosis): for each of the twelve, whether it is a port, a retirement
with its point of §17, or a defect; whether a restore inside a process
should pass through the check made for old checkpoints; what a fixture
with no time band is under point 15; and what the comparison finds
changed, and whether a trial that is not kept may leave `activity`
changed.

### 22.3 What else to carry

- *The limits Codex states itself* (§21.2) stand as limits: certificates
  34 to 40 and 56 are of a selected kept tree and not of a learned parse;
  the matching-path read uses a controlled inverse trace; the whole
  thought path of the negative form, several open operands, the charging
  of generation's work and the lesson on the finished source are not
  certified.
- *The comparison of `Intuition.pervasion` with point 11's read* on one
  question is still owed (§21.3).
- *Codex's own choices* (§21.4) are accepted as defaults. Two for Alec to
  see: `some` gives nothing for a kind of fewer than three members, since
  no subset is then more than one and fewer than all; and where
  interpretation is forced, memberships are carried along the declared
  references between word and object.
- *What Alec's answer to §21.7 costs.* The old writer gave an embedded
  occurrence no time (`PartClosing._write_occurrence`), so a checkpoint
  of the old layout that holds one is rejected. Certificate 47 then
  loads only those that hold none.
- *Nothing after §3.19's "Settled by Alec" is in this round.* The later
  subsections there record a discussion for a later round.

### 22.4 On §21.9: one reference for what the idea is of (for Alec)

**Codex is right, and the fault is in point 22, which is Claude's.**
Point 22 has "What a row denotes is read from its entry." That was
Claude's reading of the bindings, which Alec did not object to. It is
wrong in the design's own terms. An idea is an intension and its entry
is its extension (§3.0), and two intensions can have one extension: a
kind of one member and that member, as in the reproduction, and equally
a kind of two members and the two of them joined. No number of readings
kept in the entry tells them apart. The idea's vector does differ, but a
vector does not name a concept exactly, since two concepts may have
equal codes (§12); Alec allowed a reference to be carried for that
reason ("generation may carry the answer object's reference so the
object is said by its one word").

The order of a row cannot come from anywhere else either. Alec had the
order "stored on the concepts" and not on the row, so a row's order is
the order of the concept its idea is of, and that concept has to be
named.

**Recommendation: Codex's column, as proposed.** One optional reference
on an `LTM0` row for what its idea is of: a concept, or another row by
its time. It is empty where the close resolved none, which is so for
most sentences. It is written from the reference the close selected
(`PartClosing.occurrence_bindings`, `_identity`), and every reader of
§21.9's table reads it in place of the sidecar. Claude would name it
`referent`, since "denotation" will be read as the extension, which is
the entry; the name is Alec's to choose.

It also has a second use that the reduction needs. Point 14 promotes a
row by making its idea a concept. That concept's reference is then the
row's referent, so the row and the concept are tied without a table of
their own.

The amendments, if Alec agrees:

- *Point 12.* An `LTM0` row is its idea, its `trust`, its `referent`,
  its entry and its `activity`.
- *Point 22.* What a row's idea refers to is read from its referent, and
  what it extends to from its entry. What is pending is held in MTM.
  Which thought produced a row is not kept.
- *Point 14.* On promotion the new concept is written as the row's
  referent. Its order is the referent's where the row had one, and one
  above the greatest order among its entry's concepts where it had none
  (*default*).
- *Certificate 41.* The saved columns of an `LTM0` row are `slots`,
  `episodic_concepts`, `episodic_values`, `trust`, `referent` and
  `activity`.
- *A certificate to add.* A member and its kind of one member, each
  closed as a row, leave equal entries and different referents, and
  `LTM0.order_of_row` gives their two native orders.
  `test_reference_close.py::test_recent_token_returns_at_its_kinds_native_conceptual_order`
  holds with its assertions unchanged.

The other two ways, and why not:

- *Retire the exact reference.* A reader would take the strongest member
  of the entry. The kind's order becomes its member's, the test above is
  retired, reference proposals lose the difference between a kind and
  its member, and promotion has no order to allocate at.
- *Put the concept the idea is of into the entry.* The entry is then no
  longer the extension the operators left, and recall would match on it.

The cost is the carrier Codex gives: 16 bytes a row, or 16,777,216 bytes
at the largest constructed capacity.

**Alec's answer, 2026-11-13:** "Yes, keep it". The reference is kept.
The amendments above are made in §17 as point 12b, in points 12, 14 and
22, and in certificates 41 and 63. The text for Codex is §18.5.

### 22.5 Alec's answers on what was parked (2026-11-13)

- *Unused structure.* "Delete unused structure. The symbol space's own
  taxonomy, the chain rows and several writers only tests call." It is
  point 23 and certificate 64, in the round (§18.6).
- *The preposition's kernel head.* "I think we agreed that prepositions
  functions like composite ADVerbials". Point 6a.
- *The time result of recall.* "the previous time is baked in." Claude's
  reading: a row's idea holds its time in its band, so the time of what
  is remembered comes back with the idea. No separate result over the
  rows' ages is built.
- *Item 0.5.* "Item 0.5 has pushed the testing branch at github, we will
  integrate that after 1.8". The review round of the 0.5 plan waits for
  that.
- *The lesson gates.* "please just make a proposal and well go from
  there." The proposal is in the 0.5 plan, §18.4, on the branch that
  holds the documents.
- *The documents.* "The docs work should be committed if it hasnt already
  been by the testing round." Committed on a branch of its own in the
  worktree, and not pushed.

Three of the parked items were questions back to Claude. The answers are
Claude's and wait for Alec.

- *Recognition of a value.* Alec: "what are you asking?" Nothing is asked
  of him now. What was parked: §3.10 has the answer to "what is this?"
  as a concept of order one or more whose box holds the value shown;
  Codex asked whether it should be the order-zero concept the word names;
  Alec said the question was out of date and deferred it. With what was
  settled since, Claude's proposal is that recognition is `interpret` run
  backwards: the value activates its concept, the meta-concept's other
  member is the word, and no rule about the answer's order is needed.
- *Determiners during compose.* Alec: ""DET NP" can lower and do selection
  on the extension. I don't know that they can do any actual selection on
  the intension ... They have to mark it in some invertible way; I am
  naieve to think that the intension (as a sentence) can be held in a
  single vector?" As built, a determiner in compose does not select among
  members. It fixes what the phrase refers to: `determiner_mode` binds the
  phrase to an identity MTM holds, or mints one, or leaves the kind. The
  mark that inverts it is the operator itself, kept in the tree, which
  `Generative` reads to say the determiner again. And one vector holds a
  sentence only as far as its width allows. What makes saying it again
  exact is the vector together with what is carried beside it: the
  operators' inverses, some exact given a witness and some lossy, and
  the exact reference of point 12b.
- *The four directions of an operator.* Alec: "what are needed about
  compose/generate and their extend_ counterparts?" Both extend methods
  are on every operator (certificate 34), run on a sentence that was read
  and on an idea that was recalled. The one case not built is
  `compose_extend` with no words present, which §3.17 had as the way to
  describe what is seen. If recognition is `interpret` run backwards,
  that case is not needed for it, and nothing further is.

### 22.6 The two names, and a determiner's effect on the idea (2026-11-13)

**The names.** On Claude's proposal that recognition is `interpret` run
backwards, Alec: "Good, but lets call interpret as objectForWord, and its
reverse wordForObject." So recognition, "what is this?", is
`wordForObject`. The rename of the operator in the grammar files, the
code and the tests is not in this round (*default*); it goes with the
round that rebuilds it (§3.19, "After the round was settled").

**Alec's question.** "Since the determiner does selection on the
extension, what operation on a higher order vector achieves the same
thing? i.e., the modified extension (reduction of discontinuity) should
result in a change of intention. can we just run the different extension
though the fold to see what the change is, or can we assume that it is a
narrowing of the sigma over the symbols to isolate the object(s)?"

**Claude's answer, for Alec.** The two are one operation wherever a kind
is the fold of its members, and narrowing is the one to define it by.

- *Why they agree.* If the kind's vector is the greatest of its members'
  at each coordinate, then narrowing it by what was selected gives the
  greatest of the selected members again, which is what the fold of the
  selection gives. With a weighted sum for the fold, the fold of the
  selection is the same sum with the other terms left out. Checked as
  arithmetic (not the model): the two differ by 0.0 for one member, for
  two and for all, under either fold.
- *Where they part.* A kind whose vector is not the fold of its known
  members: one placed by context, as Alec decided for classes, or one
  with members not yet known. There the fold of the selection ignores
  the kind, and narrowing the kind loses part of what was selected (in
  the check it kept .91 and .86 of it).
- *So, by determiner.* "the": the idea is the selected identity's own
  vector, by its reference, with no fold. "a": the kind's vector as it
  is, lowered one order; what is spread at equal shares is its extension.
  "some": the kind narrowed by the selected members. "every": the kind.
- *What the fold is for.* Running the selected extension through the
  fold is the check, and the teacher: what the determiner's kernel gives
  for the idea should be what the fold gives for the selection. That is
  the explicit side teaching the map, as Alec has it for the two systems.
- *It stays invertible* by what is carried beside the vector: the
  operator kept in the tree, and the exact reference of point 12b.

### 22.7 Determiners: a word at several orders, a count, length, and the adjective's operation (Alec, 2026-11-13)

Discussion with Alec after §22.6. Where the two differ, this replaces the
list "by determiner" there. It is not in a Codex text, and nothing here is
decided beyond Alec's own words.

**What Alec said, in order.**

- On a determiner as two operations: "so a determiner is two things: it
  is a select() and a lower(), where the select() is an MLM identification
  and the lowering is just running the intension back though the sigma/pi
  that raised it."
- On Claude's suggestion that the sentence's tree be the stored intension,
  with no vector for the sentence: "lets finish with an event space as the
  meaningful space. I think it is correct, but I am really stymied with "a
  cat" and "cat" and how to represent order *within* a vector. if we knew
  that we might not even need to represent order outside of the vector. we
  could separate each determiner on a single dimension, and reserve space
  for them there: a dimension of plurality, a dimension of order."
- Then: "I think we can have "cat" exist at multiple orders. So when it is
  found to be a count noun, we leave the old "cat" reference and create a
  new "cat": so it is an instance of polysemy. Seeing a determiner causes
  a choice at interpretation time to chose the higher-order one (maybe
  that can be the default, also). that gives a difference between cat and
  a cat. So if we store plurality or count as a dimension (which in fact
  might go well with the modality of number that we just added), then we
  need only differentiate a cat and the cat. For now Im happy to let that
  be encoded in vector length, since it is a measure or certainty." And:
  "So that gives us everything we need for determiners, and it allows for
  a multi-modal concept."
- Then: "Number seems like a special case of adjective: can we just let
  number (one, two, three) and determiner (a, the, some, many) be handled
  as special English cases of that operation?"

**So, as Alec has it.** The sentence stays one vector, an event. Order is
not written into the vector: a word may name a concept at more than one
order, and `objectForWord` chooses between them, as it does for any word
with several objects. Count is a dimension. "a" against "the" is the
vector's length. His last question is whether number words and
determiners are then cases of the adjective's operation.

**What the source does today** (read 2026-11-13, Codex's working tree).

- `LowerLayer` of `bin/Language.py`: "A determiner selects a member; its
  marker contributes no content." That holds outside the meaning layout
  and at initialization. Under the meaning layout the rule has a
  `LaneKernel` (`GrammarLayer.__init__` of `bin/Layers.py`): the
  description's chart values are scaled, and shifted by a learned linear
  function of the marker, exactly invertible given the marker.
  (*Corrected 2026-11-13.* Claude first wrote here, and told Alec, that
  the meaning block passes through unchanged.) Kind or member is held
  beside the vector (the rule's `OrderExpr`, `output_order`,
  `ConceptAllocator.order_of`).
- `data/complete.grammar` gives the determiner six rules on that one
  implementation: `lower` (`referent="mint"`), `definite` (`bind`),
  `universal` and `some` (`kind`), `lower_unary` and `definite_unary`.
  No rule is the adjective's. That file's comment gives `intersection` as
  an "ADJ mask / meet on NP" (a least on the for lane and a greatest on
  the against lane, idempotent), but which operator a modifier takes is
  chosen at run time (Alec, below).
- [Language](../Language.md) records what set the two apart. Alec,
  2026-09-28: "Order only drops when the determiner is applied, adjectives
  do not perform that role." Its words "and nowhere else" he withdrew on
  2026-09-29. The doc's sentence is: the determiner moves from a set to a
  member, the adjective from a whole to a part. Its table also notes the
  determiner as a cue input to `bind`, the only binder (2026-10-19).
- When a named concept is found equal to its third member, a class is
  raised one order up and the word moves to it:
  `test_a_previously_named_shared_concept_denotes_its_raised_class` of
  `test/test_item25_completion.py`, a stopping gate, ends with
  `'word' not in alloc.symbol_types[shared]`. Under Alec's statement the
  old concept keeps the word, so that one assertion changes. With the
  higher order as the default choice the gate's other assertions stand.
- The number modality is one optional scalar read beside the text, and a
  word learns bounds on it (§14: `one` [1,1], `two` [2,2], `three`
  [3,3]). By decision 22 the value is the code.
- Length is already a count of meetings. `MeaningCodes.definedness` is
  $w/(w+4)$ for $w$ witnesses, and
  `test_row_definedness_survives_rewitness_and_checkpoint_without_changing_truth`
  has a row's identity code at length 1/5 after one writing and 2/6 after
  two. Evidence is applied on top, lane by lane.

**Claude's readings, for Alec to correct.**

1. "Found to be a count noun" is the raise that exists today, at
   `raiseCount` equal members.
2. With the higher order as the default, "cat" and "a cat" both begin
   from the kind. What differs in the vector is the count: none against
   one.
3. Why a determiner calls for the higher order: a modifier needs a head
   with members to narrow, as Language has it for compounds ("A head must
   therefore have a shape"). No cue of its own is needed. Unchecked in
   the source.
4. Length needs nothing new for "a" against "the". "a" mints a row, met
   once; "the" binds an earlier row and meets it again. The length is on
   the referent's row, so it need not survive in the sentence's vector,
   and a retelling reads it afresh.
5. On the adjective: yes. Alec then kept `lower` for the order drop
   (below). So the order still drops, at `lower`, and the determiner word
   itself only narrows a kind's members to some of them, which is the
   adjective's move from a whole to a part. His sentence of 2026-09-28
   stands with `lower` in the determiner's place.
6. By word. A number word narrows by count and acts the same on any
   kind, so it commutes with other modifiers. "some", "many" and "every"
   are shares of the kind's members: they read their scale from the kind,
   as Language has it for "big", and so follow the noun. "a" is a count
   of one, not yet met. "the" is already met: the kind's members narrowed
   to those in mind, which is the selection.
7. The kind of truth, with `lower` kept: a subject that was not lowered
   is generic and writes a relation; a lowered subject writes an event.
8. What would go, with `lower` kept: its marker operand, so that `lower`
   is the one-operand rule that `lower_unary` is today; the rules
   `definite`, `universal`, `some` and `definite_unary`; the rules'
   `referent` attribute, with minting and binding left to `bind` and the
   modifier as its cue; and the rows of the lesson-gate proposal (the 0.5
   plan, §18.4) that expect `definite` or a two-operand `lower`.

**Open.**

- How minting and binding are chosen without the rule's `referent`
  attribute is not specified.
- Which operator a number word or a determiner comes to take is the
  chooser's, at run time. Under a least on the for lane a noun is emptied
  wherever the modifier is silent, unless a silent word reads as no
  limit; unchecked.
- Length also carries evidence, so a short vector is either seldom met or
  weakly held. Alec accepts this for now. It measures which one is meant,
  not whether the claim is true.
- Whether a noun phrase's count survives the verb's action
  (`MeaningAction`, reflections over the meaning block) and conjunction,
  in a sentence with several noun phrases, is not shown.
- The count's place in the vector. The form block is an average over the
  words, so a mark there fades.

**Alec's further statement (2026-11-13).** "So lets keep lower as
something that is used for higher-order words. After we do that, the
determiner is basically a special case adjective, as is number. So we
don't need a special case number, that just gets muxed in with all the
other adjectives and the architecture will sort it out. Can we let order
be implicit? We can retrieve it based on where it sits in the codebook,
since concepts are currently ramsified and preallocated."

So `lower` stays, and it is what drops the order of a higher-order word.
A determiner and a number word are adjectives with no handling of their
own, and no dimension is reserved for count. Readings 5, 7 and 8 above
are amended to this; the "count" of readings 2 and 6 is whatever a number
word's own learned content is.

**On order being implicit: what the source has** (read 2026-11-13).

- Each order has its own block of rows in the concept inventory:
  `order_slice` of `bin/Spaces.py`, "one tapered block per ramsified
  order". A concept admitted at order $o \ge 1$ takes a row there under
  the key `('o1', id)`, `('o2', id)` and so on; order zero uses
  `('snap', id)`.
- A full block spills into one tail shared by every order: "The initial
  taper is a placement preference, not a second physical inventory
  ceiling." In the tail a row's position does not tell its order. Its key
  still does.
- `ConceptAllocator.order_of` of `bin/Layers.py` does not read the
  position. It derives the order from the definition, one above the
  highest part, or takes `reference_orders`. `settle` derives it again as
  parts arrive, and "nothing migrates by order". Whether a concept ever
  comes to differ from the block it sits in is not traced.
- A composed idea has no row. Its order is computed from its operands'
  by the rule's `OrderExpr`, as `output_order`.
- `lower_unary` is today's one-operand face of `lower`: invertible, and
  it adds nothing to the vector.

**Claude's answer, for Alec.** Yes, with one precision.

1. Read a concept's order from its row's key. That holds in the tail
   too; position alone holds only inside the blocks.
2. It is sound if a concept never changes order once it has a row.
   Alec's statement that the raise leaves the old concept and makes a new
   one gives exactly that. The order derived from the definition and the
   order of the key then always agree, which is a certificate, and
   `reference_orders` and `placement` have nothing left to hold. Their
   readers are untraced.
3. A composed idea's order stays computed from the tree and is stored
   nowhere, as today.
4. `lower` is admitted exactly where its operand's row is of order one or
   more. That is what makes a determiner call for the higher-order sense
   of a word, with no cue of its own.

Open beside the points above: whether a full block should spill or
refuse (*default*: keep the spill and read the key).

**Alec, on which operator (2026-11-13).** "also, what operators have we?
Adjective may not map to intersection, that decision of which operator is
bound is. run-time one." So nothing above ties a modifier, a number word
or a determiner to `intersection`. The chooser binds an operator at run
time, and the lesson-gate proposal's row for modifiers (`intersection` or
`product`) is an expectation to test, never a binding.

The operators of `data/complete.grammar`, as read 2026-11-13 in Codex's
working tree:

| Group | Compose |
|---|---|
| on the field | `divide`, `descend`, `gloss`, `intersection`, `union` |
| word and object | `interpret`, `sign`, `bind` |
| relations | `part`, `whole`, `equal`, `implies`, `generic` |
| question and negation | `ask`, `not`, `non` |
| connectives | `conjunction`, `disjunction`, `intersection`, `union` |
| arithmetic | `sum`, `product` |
| phrase building | `lift`, `verb`, `compound`, `adverb`, `preposition` |
| determiner | `lower`, `definite`, `universal`, `some` |
| surface and word order | `surface`, `transpose` |
| one-operand faces | `lower_unary`, `definite_unary`, `preposition_unary`, `surface_unary`, `compound_unary` |

Thought: `ask`, `query`, `part`, `whole`, `equal`, `not`, `bind`,
`isTrue`, `exist`, `isPart`, `isEqual`, `infer`, `inferBackward`,
`intuit`, `implies`, `gain`. Generate holds the reverses.

Two-operand operators that leave the head's order alone, and so are open
to a modifier: `intersection`, `conjunction`, `union`, `disjunction`,
`sum`, `product`, `compound`, `adverb`, `preposition`. In the meaning
block `conjunction` and `intersection` are one computation
(`MeaningCodes.compose`), as are `disjunction` and `union`; each pair
differs in the form block.

**Alec, on a fixed intersection (2026-11-13).** "I remember that
intersection did not take account of where the noun currently was, so the
idea of using a fixed intersection across all nouns was problematic: a
small elephant is not a small paperclip."

The record agrees. [Language](../Language.md), "Which operators commute":
a modifier that acts the same on any domain commutes, and one that reads
its scale from the domain it is given does not and must follow the noun
("big", "skillful"); the Ground it is read against is the head's own
shape, which "already exists in virtue of the folds at every order, so no
new parameterization should be necessary".

In the source (read 2026-11-13):

- `intersection` is a least per coordinate (`MeaningCodes.compose`). The
  modifier sets one ceiling, the same for every noun.
- `adverb` reads its head. `MeaningAction` of `bin/MeaningKernels.py`
  multiplies the head's own coordinates, in the chart, by a gain $e^{w}$
  that the modifier supplies, in one learned basis shared by all
  modifiers. It is exactly invertible given the modifier. A coordinate on
  which the head is silent stays zero, so a gain cannot add a property.
- `product` multiplies the two operands coordinate by coordinate, and so
  is also relative to the head.

*Claude's reading.* A modifier that acts the same on any kind can take
the least. One that reads its scale from the kind takes the gain: "small",
and the shares "some", "many" and "every". Binding at run time is what
lets one class of words hold both.

### 22.8 The operators of a universal grammar (Alec, 2026-11-13)

Alec's list for compose and generate. It is not decided beyond his words
and is not in a Codex text.

**Alec.** "Here are the operators I think we need for a UG, where each
operator can compose/generate and operate on both the extent and intent:
not, non; interpret (wordToObject), name (objectToWord), sign, bind;
part/implies, equal (?); ask; conjunction, disjunction, intersection,
union (I think these collapse to two, with extent/intent each); sum (?),
product (certainty); lift, lower; verb, noun; modifier (designed for ADV,
might cover ADJ); preposition (is a compound modifier); surface, swap
(transpose). I don't see a reason for the unary operator "faces": lift
and lower are now unary. This does not get into the thought section, and
Im pretty sure we had decided that ask was a placeholder: perhaps it can
mark a position of uncertainty within the sentence." (His list was one
group to a line; the semicolons here stand for his line ends.)

**Against `data/complete.grammar`** (read 2026-11-13, Codex's tree).

| Alec's list | Today | |
|---|---|---|
| `not`, `non` | `not`, `non` | same |
| `interpret` | `interpret` | word to object |
| `name` | the reverse of `interpret` | new as an operator |
| `sign`, `bind` | `sign`, `bind` | same |
| `part/implies` | `part`, `whole`, `implies` | one operator for three rules |
| `equal` (?) | `equal` | |
| `ask` | `ask` | |
| two, each on extent and intent | `conjunction`, `disjunction`, `intersection`, `union`, and the field rules `intersection`, `union` | four names to two |
| `sum` (?) | `sum`, the mean | |
| `product` | `product` | |
| `lift` | `lift`, two operands: the sentence from its subject and verb phrase | now one operand |
| `lower` | `lower`, `definite`, `universal`, `some` | one operand (§22.7) |
| `verb` | `verb`: a verb with its object | |
| `noun` | none | new |
| `modifier` | `adverb` | renamed, to cover the adjective |
| `preposition` | `preposition` | |
| `surface`, `swap` | `surface`, `transpose` | renamed |

Not in the list: `whole` (already `part` with its operands exchanged),
`generic`, `compound`, the field operations `divide`, `descend` and
`gloss`, the determiner rules, and the five one-operand faces.

**Source facts that bear on it.**

- In the meaning block `conjunction` and `intersection` are one
  computation, as are `disjunction` and `union` (`MeaningCodes.compose`).
  `intersection` and `union` are also the two-sided field rules. The
  grammar file's comment gives the form side of `conjunction` as a
  product and of `disjunction` as the probabilistic sum $a+b-ab$.
- `VerbLayer` is a subclass of `LiftLayer`, and `lift`, `verb` and
  `adverb` all act through `MeaningAction`.
- `ask` in compose is a `_ThoughtUnaryNoopLayer`, "A structural carrier
  for a boundary-only unary thought operation": the vector passes
  unchanged. `bind`: "Read the chosen off-stack field entry, or keep the
  dummy's own content."
- `sum` is the mean of its operands, and `implies` is built on it
  (`ImpliesLayer(SumLayer)`).
- `equal` is what the raise counts, up to `raiseCount` members.
- A face is "The binary kernel with its left identity": the operation
  with its marker word absent. `divide`, `descend` and `gloss` are
  `BracketOperation`s, on the field only and with no generate face.

**Claude's readings, for Alec to correct.**

1. *Names.* `interpret` and `name` are the latest words and replace
   `objectForWord` and `wordForObject` of §22.6. If every operator has
   compose and generate, `name` is `interpret` generated; whether it is a
   second operator is open.
2. *`part/implies`.* One operator: `part` over extents is `implies` over
   intents. `whole` stays its exchanged form.
3. *`equal`.* Keep it (*recommended*). A tree uses each operand once, so
   an identity cannot be said as `part` both ways, and the raise counts
   equalities.
4. *The four that collapse.* A least and a greatest. The extent face is
   today's field rule and the intent face today's two-operand rule. The
   product and the probabilistic sum of today's form side then belong
   with `product`; `sum` could be its dual, $a+b-ab$, in place of the
   mean, which would answer its question mark. `implies` needs no carrier
   of its own once it is `part`.
5. *`lift` with one operand* is the raise by one order, the opposite of
   `lower`. The sentence then needs another composer. *Default:* `verb`
   takes its subject as it takes its object, the same gain action applied
   twice.
6. *`noun`* is not known. Two candidates: today's `compound`, a noun
   sub-typing its head; or making an idea a thing, so that a sentence can
   be an operand.
7. *`modifier`* is today's `adverb`: the gain, which reads its head
   (§22.7).
8. *Faces.* `lower_unary` and `definite_unary` go once `lower` has one
   operand. The other three apply an operation whose marker is absent;
   with `preposition` a modifier, an unmarked phrase takes `modifier`
   directly. Their uses in the tests are untraced.
9. *`ask`.* The source agrees that it is a placeholder in compose. A
   mark on the unknown position gives `bind` the place it fills.
10. *The field operations* `divide`, `descend` and `gloss` are unplaced.
    They act on the extent only.
11. "Each operator can compose/generate and operate on both the extent
    and intent" states as a rule the four directions that §3.17 left
    unsettled.

**Alec's answers (2026-11-13).**

- "The sentence is composed as before. but lets call the operation that
  lift was doing as change(), since it is a temporal modifier If the verb
  subclass does not do anything beyond provide a new name, we should
  remove it (we should not presume which parts of speech go with which
  operators)."
- "let's use name() and interpret(). I think sign() only goes one way"
- "divide/descend/gloss: lets keep the four
  add/subtract/multiply/divide. I don't know what descend and gloss were
  doing."
- "The noun operator can be subsumed by interpret (which just makes an
  object a word; some letters are not words)."
- "I am thinking we should limit data/complete.grammar by commenting the
  rest."
- "For thought, we are giving tools to the model so that it can change
  its structures: ask (as before); query (over memory); isPart (this is
  inferred after a difference); isWhole (probably keep, but equivalent to
  a swapped part); equal (can unify over two different orders by doing
  lower or raise); not, non (as before); bind; exist; infer; intuit;
  part/whole/implies (*creates* an STM1); define (creates two STM1)".
  (One to a line in his message.)

**So.** The two-operand rule that composes the sentence is `change`, and
`lift` is the one-operand raise. `interpret` and `name` are the names, and
`noun` is withdrawn. Readings 1, 5 and 6 above are settled by this.

**Source facts** (read 2026-11-13).

- Under the meaning layout three computations serve the two-operand
  rules (`GrammarLayer.__init__` of `bin/Layers.py`). A rule with a
  `meaning_rule` takes the lattice computation. `lift` and `adverb` take
  `MeaningAction`, the gain in a basis. Every other, `verb`, `lower`,
  `compound` and `preposition` among them, takes a `LaneKernel` of its
  own: the head's chart values scaled, and shifted by a learned linear
  function of the other operand. A shift can add what the head is silent
  on; a gain cannot.
- So `verb` is more than a name. It has its own learned `LaneKernel`,
  takes the verb as head and its object as the other operand, labels the
  result a verb phrase, and its reverse needs a candidate search. Its
  docstring calls it the object sub-typing the verb.
- `divide`, `descend` and `gloss` are not arithmetic. They are the
  examination phase (`GrammarPhases.EXAMINATION`): the three ways
  perceptual attention narrows over the bracket table of
  `bin/Attention.py` (`narrowing_mask`). `divide` splits a bracket that
  holds evidence of both kinds, `descend` goes into a bracket's parts,
  and `gloss` accepts a single pure bracket as a word, after which its
  symbol enters the grammar.
- Arithmetic today is `sum`, which is the mean, and `product`.
  Subtraction was retired as "sum of a negated operand" (the grammar
  file's comment). There is no arithmetic division.
- `sign` today has both directions, each the identity.
- Today's thought section has sixteen: `ask`, `query`, `part`, `whole`,
  `equal`, `not`, `bind`, `isTrue`, `exist`, `isPart`, `isEqual`,
  `infer`, `inferBackward`, `intuit`, `implies`, `gain`. Against Alec's
  list, `isWhole`, `non` and `define` are new, and `isTrue`, `isEqual`,
  `inferBackward` and `gain` are not in it.

**Claude's readings, continued.**

12. `verb` cannot go as an alias. What it does, an object sub-typing its
    head, is the work of `modifier`, so it can go by merging into
    `modifier` with the object as the modifier. The two kernels differ
    today, a shift against a gain; which one `modifier` keeps is open.
13. `divide`, `descend` and `gloss` stay as attention's operations,
    outside the list. Without them reading finds no words. If the grammar
    file is limited, they are not among what is commented out.
14. The four of arithmetic. `sum` would become a true sum, and
    subtraction and division are new. A sum or a quotient leaves the unit
    cube, which is why `sum` is the mean today (*Claude's inference*).
    This also meets Alec's rule of 2026-10-07 that arithmetic on numerals
    is thought by chained successor and never an operator on codes, and
    his remark of 2026-11-11, left unsettled, about operations "that can
    operate on the values directly". Whether the four act on the number
    modality's values only is for Alec.
15. `equal` in thought as the test and the unifier, and `define` as the
    writer of two `part` rows, answer reading 3: an identity is `part`
    both ways, written by one operator. Whether compose keeps `equal` is
    then open again.
16. Limiting the grammar file is Codex's work and touches the tests. A
    dozen test files use the rules that would be commented out
    (`compound`, `definite`, the faces), and so does the lesson-gate
    proposal.

**Open.** Readings 10, 12, 14 and 15; `isTrue`, `isEqual`,
`inferBackward` and `gain`; `generic`; the names of the two that the four
collapse to.

**Alec, on what was open (2026-11-13).**

- "we can keep isTrue, isEqual, inferBackward"
- "gain would seem to be multiply, so redundant?"
- On Claude's point that an add on values would answer "one plus one"
  with no chain of successors: "arithmetic do not operate on the codes:
  that would break codebook lookup. So even with number, we categorize it
  perceptually: that puts us in a codebook, and percept activations have
  a kind there but no magnitude. So for now, the arithmetic operations
  are conceptual-space only."
- "But Im wondering about the new modality of number; all input
  intensities are tuned into level-sets of percepts. does that seem
  psychologically correct, even for feelings? they are still unlabelled,
  so we cant talk about them. Im reserving the feeling of that continuity
  for direct perception, but that is not something that we can model."

**So.** Thought keeps `isTrue`, `isEqual` and `inferBackward`. The four
of arithmetic act in conceptual space only, never on a code, and a number
is categorized as any percept is. That settles reading 14, and the rule
of 2026-10-07 stands.

**On `gain`** (source, read 2026-11-13). It is not a product. The thought
`gain` takes one concept and sets `model.expectation_gain` to that
concept's length, clipped to $[0,1]$ (`ThoughtFaces.gain`, and the
`gain` effect of `bin/AccessibleMind.py`). That number is the strength of
expectation over rows, `rowExpectationGain` in a configuration, zero by
default. So it is a dial, not a tool that changes a structure, and its
name is also the name of the modifier's gain. Whether it stays is for
Alec.

**Claude's answer on level sets, for Alec** (the references are from
memory, unchecked today).

- What is identified and remembered is categorical and coarse: about
  seven levels on one dimension by absolute judgment, though hundreds can
  be told apart side by side (Miller 1956). Cortical number neurons are
  tuned to preferred values (Nieder and Miller 2003), and the models
  build such a unit from units that respond past a threshold (Dehaene and
  Changeux 1993; Verguts and Fias 2004). That is Alec's basic percept of
  2026-11-12, a part knit with a whole.
- The brain's level sets overlap and respond in degrees, so the pattern
  over them keeps the magnitude with no one unit holding it. Percepts
  that are disjoint and all-or-none would lose it. If a value activates
  every part bound below it and every whole bound above it, the pattern
  still orders the values (*Claude's reading*).
- Telling two present values apart is finer than identifying either.
  That is the direct perception Alec reserves. The feeling is not
  modelled; the judgment of more or less between two present values can
  be.
- For feelings: the felt quantity is continuous and present without
  labels, and the categories come with concepts and words (Russell 2003;
  Barrett 2006; Lindquist and others 2006). That agrees with "unlabelled,
  so we cant talk about them". Animals and infants do act on unlabelled
  magnitudes (Guttman and Kalish 1956). The source does this for the
  scalar beside the words: `BasicPercepts.perceive_value` admits each
  exact value met as a percept of its own, labelled or not, and a value's
  membership in a word's box falls off by degrees outside its bounds
  (`meet_ratios`). (*Checked 2026-11-13;* Claude first told Alec the
  model might be stricter than the evidence here. It is not.)

  The references were then checked: Miller reports the seven categories
  and not the contrast with side-by-side comparison; Russell's core
  affect is felt unattributed, and its categorizing into emotions is
  Barrett's thesis. [Philosophy](../Philosophy.md) has them as
  corrected.

**Alec, on `gain` (2026-11-13).** "drop gain, we can set that as
expectationGain on the model." So the thought `gain` leaves the list, and
the strength of expectation is a setting of the model. The source already
has it as one: `rowExpectationGain` of a configuration's training section
sets `model.expectation_gain`, beside `perceptExpectationGain` and
`symbolExpectationGain`. Whether the key is renamed `expectationGain` is
a *default* for the round that limits the grammar: keep the three keys as
they are.

**Still open.** Readings 10, 12 and 15; `generic`; the names of the two
that the four collapse to; what a sum or a quotient does at the edge of
the unit range.

**Alec, further (2026-11-13).**

- "I thought modifier acts *on* verbs as identified by change()? as well
  as maybe nouns? which makes it play the role of both adv and adj"
- "equal can go, isEqual should stay"
- "attentions operations should be called focus(), unfocus(). gloss() is
  better named mint()"
- "I don't know what generic does, but I think its the low-order "cat"
  word, so it doesn't need to exist."
- "arithmetic can work in tanh-bounded space or something (0..inf)"

**So.** `equal` leaves the lists and `isEqual` is the test in thought.
`generic` goes. `modifier` acts on what `change` takes as its changer,
and on nouns, so it is both the adverb and the adjective. The attention
operations are `focus`, `unfocus` and `mint`. Arithmetic is computed in
the chart, where a value runs from zero to infinity, and mapped back.

**Claude's readings, continued.**

17. `modifier` acting on verbs covers a verb's object: the object
    modifies its verb. So `verb` goes by that merge (reading 12), and the
    kernel is `modifier`'s, the gain.
18. `focus` is today's `divide` and `descend`, both of which narrow.
    `unfocus` has no counterpart today. `mint` is today's `gloss`. The
    docs also say "a" mints a referent; the two uses of the word are
    different acts.
19. With `equal` gone, a sentence of identity composes as `part`, and
    `define` in thought writes the two rows. The raise counts what
    `define` wrote where it counted equalities. The math lessons state
    their answers with `equal` today.
20. In the chart a sum is $\tanh(\operatorname{atanh} a +
    \operatorname{atanh} b)$ and cannot leave the unit range. A
    difference floors at zero in each lane, since a lane holds no
    negative amount (*default*). A quotient by zero is silent
    (*default*).
21. "The names of the two that the four collapse to" meant only this:
    Alec said the four connectives collapse to two, and two of the four
    names then go. *Default:* `intersection` and `union` stay, being the
    names of the field rules already.

**Still open.** Reading 15 is settled by "equal can go". Left: the
*defaults* of readings 18, 20 and 21, and what `unfocus` does.

### 22.9 The number modality over many dimensions: XOR and images (2026-11-13)

**Alec.** "Please ensure a path for the number modality to process images
as well as XOR (the only different being the percept dims; MNIST-size
images would be a good test, but no need for multimodal at this point,
that work is left to someone else)." It is not in a Codex text.

**What it rests on.**

- Intensity is a code (Alec, 2026-10-25, decided;
  [FutureWork](../FutureWork.md), "Intensity as codes, not amplitude"): a
  pixel's intensity is which atoms fire, a crisp percept like a letter,
  with no amplitude coordinate and no graded occurrence.
- Alec, 2026-11-12: "Each percept is categorized against perceptual parts
  and wholes, whose codes are the values. So the order zero concept is a
  generalization over basic percepts. We should share as much code
  between the two modalities as possible."
- Alec, 2026-11-13: "all input intensities are tuned into level-sets of
  percepts", and "percept activations have a kind there but no
  magnitude".
- Alec, 2026-11-11: XOR "should be converted to a system 1 model. That
  means just one concepts-to-concepts step at order 0".

**What the source has** (read 2026-11-13, Codex's tree).

- `NumericPercepts(model, dimensions)` of `bin/NumericPercepts.py` is
  attached when `dataType` is `numeric` and the grammar file has neither
  a per-word phase nor `pi` and `sigma` (`bin/Models.py`). Its dimensions
  are the product of the input shape, so a 28 by 28 input is already 784
  dimensions, flattened, with no place.
- Each dimension has one basic percept
  (`BasicPercepts.numeric_dimension`, bounded by the percept cube's
  identity width) and one order-zero concept whose for lane is the felt
  value. That is the value as evidence: a magnitude, not a kind.
- The scalar beside the words is built the other way.
  `BasicPercepts.perceive_value` admits each exact value met as a percept
  that is its own part and whole, and a word's concept collects such
  percepts and generalizes their bounds (`collection_code`,
  `read_collection`). §14: §3.9 "could use value-bounded percepts only by
  changing that feature convention and the XOR proofs; that change is not
  made."
- Beyond its inputs the numeric path makes a concept by promotion.
  `observe_intersection` proposes the whole set of inputs present in a
  sample, and a set that recurs is promoted. XOR's one needed concept,
  both inputs present, comes this way.
- `OutputSpace.read_field` reads the field by one linear layer.
- `data/mnist.xml` is the older dense-slab configuration. The loader
  (`loadMNist` of `bin/data.py`) reads two CSV files of `data/` held in
  Git LFS and normalizes pixels by mean and deviation, which gives
  negative numbers; `NumericPercepts.feeling` refuses a negative number.

**The path** (*Claude's proposal*; each *default* is Alec's to change).

1. *One reader for any number of dimensions.* A row of numbers is
   flattened to $D$ dimensions: 2 for XOR, 784 for a 28 by 28 image.
   Nothing else differs between the two.
2. *Level sets.* Each dimension's number is felt and then categorized as
   the scalar beside the words is: its percept knits a part and a whole
   whose codes are values. The percept has a kind, its level, and no
   magnitude. Zero is no presence. This moves §3.9 from the value as
   evidence to level sets, the change Codex left unmade.
3. *The dimension is an axis of the percept cube*, as built
   (`BasicPercepts.numeric_dimension`). Claude first proposed the
   dimension as the percept's place; Alec's words are "the only different
   being the percept dims", so the axis stands and no place is added.
4. *The levels are the values met* (*default*, as `perceive_value` does).
   XOR has one, the value 1. The first image test uses images of two
   levels, so that it differs from XOR in its dimensions alone
   (*default*); grey levels follow as a measurement.
5. *An order-zero concept collects basic percepts*, as for text.
6. *One step of System 1* lies between the concepts and the output,
   concepts to concepts with no grammar. Which step waits on the
   rewriting of §3.18 and §3.19: the learned maps of `expect`, or today's
   promotion. Promotion of a whole sample's inputs cannot serve an image,
   since a set of some hundred pixels does not recur. So for images the
   step has to be the learned map (*Claude's reading*).
7. *Out and back.* The one linear layer reads the field. Reconstruction
   runs from the field back to each dimension's level, which is the
   number.

**What 784 dimensions need.** Capacities stated in the configuration and
never resized in a fixture: place for 784 dimensions, basic percepts for
the pairs of place and level that are met, and concept rows for those
and for what the step adds. No operation runs over all pairs of
dimensions, and none over all rows. The numbers given to the reader are
the raw intensities, never the loader's normalized ones.

**Certificates**, for the round that builds it (continuing §17's count).

65. One reader. The XOR file and the image file differ in declared shape
    and capacities only, and a trace names the same functions for both.
66. The XOR proofs of §3.9 hold on level-set percepts, with their bars
    kept.
67. A 28 by 28 input of two levels runs forward and backward within the
    bounded test's budget, and its reconstruction returns every
    dimension's number.
68. Learning on 28 by 28 images is first a measurement: accuracy on
    held-out images after a stated number of presentations, beside the
    same file with the System 1 step removed. A bar is set after the
    first measurement (*default*).

The bounded test uses generated 28 by 28 patterns, since a test does not
fetch data. MNIST itself is run where its files are checked out.

**Alec's decision (2026-11-13).** Asked whether to build it after 1.8
with the round that rebuilds `expect` and `intuit`: "In 1.8 now". The
reader, the level sets, the image file and certificates 65 to 68 are the
seventh amendment, §18.7. Step 6's learned map stays with the rewriting
of §3.18 and §3.19, which is also 1.8; until then the image file runs
with promotion off and certificate 68 measures the linear read alone.

**Open, for Alec.** The *default* of 4, and which step 6 is.

**Places for images, and one perceptual space per modality (Alec,
2026-11-13).** "the .where for imagery has to be 2D. we might as well let
perceptual spaces vary in modality, as they do, and integrate them into a
shared 3D basic level perceptual space? That seems the most human. If we
are processing text, thats 1D as we have represented it. But perhaps the
integration path is leave each modality in its own perceptual space with
varying .where, and integrate only at the level of zero-order concepts by
adding a .modality field according to the percepts that they unite. this
still allows perceptual space to have basic-level perceptual objects on a
modal basis." Claude recommended the second. Alec: "modality as implicit
since both it and order can be inferred from the .where? Please doc that if
so." So:

- Each modality has its own perceptual space and declares its number of
  place axes, each an interval of `.where` on the existing two-rung ladder:
  none for the scalar beside the words, one for text and for XOR (two
  positions), two for an image (28 by 28).
- A pixel is a level percept at a place, as a letter is a byte percept at
  a place. Basic percepts are knit within one modality.
- Modalities meet at order-zero concepts. Alec, correcting Claude's
  wording: "the .where that is baked into all concepts in virtue of muxing
  the .where of a percept will include a .where in a range that can
  uniquely identify the modality and order of a percept." The source has
  the ranges (`WhereRegistry` of `bin/WhereRegistry.py`): input, parts,
  wholes, symbols and the live conceptual slots, fixed and disjoint, and
  every muxed vector carries its address in its band. A symbol's address
  is `2 * row + lane` of its concept, and rows lie in their order's block
  (`order_slice`), so the range gives the order. To build: one input range
  per modality, so that the range gives the modality; and a full order
  block made a capacity error (*default*), since today it spills into a
  shared tail where the address no longer tells the order. Neither is
  stored beside the vector.
- Recorded in [Spaces](../Spaces.md), "Modalities: One Perceptual Space
  Each", and [Orders §12](../Orders.md) on `testing`.

This replaces point 3 above, and points 1 and 2 of the seventh amendment
(§18.7), which keep each number as an axis of the percept cube with no
place. §18.7 is held and is rewritten before it goes to Codex. Its larger
part is giving attention's brackets, the fold ladder and word tiling a
second axis. The image data is built (on `testing`, `data/images/`,
`bin/image_corpus.py`, with `image_corpus.load(name, split)`).

## 23. Codex return to Alec for Claude: completion of this round (2026-11-14)

**The assigned round of §18 and its amendments is complete for review;
this is not a landing.** Alec's retained-occurrence, checkpoint-rejection,
referent and actual-MTM-event decisions are built. Both stopping gates
pass. The finished-source certificate group has **205 passes**, and its
slow-marked XOR proof also passes in an explicit run. The complete
default sweep has **7,564 passes, 323 skips, one non-strict XPASS
and two pre-existing failures**. The fresh verbal proof is **4/4** on
both output readings. The 160-sentence lesson completes, but none of its
field comparisons is exact; its measured differences are reported, not
tuned away.

The earlier checkpoint status statements in this section are historical.
The completed reader, default, cache and storage tables are in
§§23.17–23.18; the named ports and retirements are in §23.21; the final
certificates, measurement, sweep and source are in §§23.28–23.32.
Nothing was committed or pushed. §17.5, §18.7 and §3.19 remain outside
this round, as assigned.

### 23.1 For Alec: an old idea can hold both truth lanes

Certificate 47 requires importing the checkpoint commit's layout while
keeping each old idea and its time. Point 12a requires one signed trust
per new row, and two rows for a conclusion supported on both lanes.
The old writer admits both lanes on one row, independently of the time
ambiguity Alec already decided.

I reproduced this using `ThoughtStream.write` and `LTM0` from the actual
checkpoint commit, `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`.
The constructed store has width 16, capacity 4 and two episodic slots.
A unary conclusion with evidence `(.7, .3)` writes **one row** at the
valid time **0**. Its state reloads successfully on that same code;
the recovered pair is `(0.699999988079071, 0.30000001192092896)`.
There is no missing time, duplicate time, selected seed or resized
fixture. This is the production thought writer, not a synthetic edit
of its saved evidence.

My current `LTM0.prepare_checkpoint` converts the old pair by subtraction.
The same checkpoint loads as one row with trust
`0.3999999761581421`, hence evidence `(0.3999999761581421, 0.0)`.
The idea and time are unchanged, but the contrary evidence is gone.
**That subtraction is a defect in this partial migration, not an
accepted interpretation of point 12a.** It must not be used as a
completed checkpoint importer.

The alternatives need a migration rule. Splitting the row requires a
second unique time and a rule for old references to the one original
occurrence; its former evidence read returned both lanes. It also needs
room when a checkpoint fills its constructed capacity. Silently choosing
one lane, subtracting them, or giving both new rows the old time each
loses an existing contract. The reference is not merely a resemblance
key: `ThoughtLTMCapability.end_evidence` can read that particular
occurrence as a premise.

**Recommendation:** reject a checkpoint of the old layout if an occupied
LTM0 row has both lanes nonzero, before changing the destination model.
Preserve the signed value for old rows with only one nonzero lane. This
adds a second explicit exception to certificate 47, beside Alec's
rejection of ambiguous times. If those checkpoints must load, specify
how the two new claims are named and how an old occurrence reference
retains its two-lane evidence; that is a larger migration.

**Question for Alec:** accept this rejection, or specify the migration
of old rows with both lanes? §18 requires stopping for an unretired read
without a specified replacement. Source work stops for this decision;
no new migration policy is silently built. The read-only checks and
inventory below describe the stopped tree.

**Alec's answer (2026-11-13):** "Reject those old checkpoints
(recommended)". Work resumes with this second explicit exception to
certificate 47. The importer validates every old idea store before
loading any destination tensor. It rejects an occupied row with both
lanes nonzero; a single lane supplies its signed value unchanged, with
no subtraction. The new rejection tests failed against the prior
importer and now pass, including an import with an earlier valid store
and a later rejected store: neither destination tensors nor migration
state change. The original checkpoint narrative above remains the
record of the defect and the decision.

### 23.2 The referent and the reduced stores so far

`IdeaRows` shares the idea read between durable `LTM0` and working
`STM0`; STM no longer inherits the durable checkpoint/time contract.
LTM0's saved row columns are exactly `slots`, `episodic_concepts`,
`episodic_values`, `trust`, `referent`, `activity`. The entry is flat.
`referent` is two int64 values: the kind of reference and its value,
empty as `(0, -1)`. A concept reference is distinct from an occurrence
at time zero. No vector comparison supplies either identity.

`PartClosing.occurrence_bindings` still supplies the selected `_identity`;
the idea writer puts the reference into the column. Working clause
metadata stays in STM. Only pending slots and forward bindings are
held in `MediumTermMemory.pending`; completed provenance is discarded.
The new MTM state version is 13. Its durable migration/validation is
not yet certified in full.

The readers from §21.9 now have these destinations:

| Reader | Read in this tree |
|---|---|
| `LTM0.raw_meaning_of` | The referent restores the idea's `role_refs[0]`; pending working meaning is overlaid from MTM while pending. |
| `LTM0.order_of_row`, `Spaces._concept_source_order` | The concept allocator's native order, or the referenced idea's order. |
| `ReferenceClose._derivation_symbol` | The retained referent and the current MTM identity. |
| `ReferenceClose.context_records`, `exclusion_pairs` | The row's native order; exclusion counts distinct retained occurrences, as Alec decided. |
| `EntryRead.recall_identities`, `recall_chunks` | Referent for the identity and native order for the recalled chunk. |
| `ClauseRows.read_reference_bindings` | Referent before following the current MTM binding. |
| `ThoughtLTMCapability._with_components` | Referent before reading the identity's current MTM value. |
| `MemoryConsolidation.consolidate` | Existing referent's order, otherwise one above the greatest native order in the entry; writes the minted concept back as referent. |

The unchanged native-order assertions in
`test_recent_token_returns_at_its_kinds_native_conceptual_order` pass.
This is **not yet all of certificate 63**: the explicit equal-entry
member/singleton and once-only promotion certificates remain due.

Activity is implemented through `MemoryActivity` for both stores, with
the four schema parameters. A write initializes it, an evidential access
raises it, and a boundary pass decays nonpermanent rows and drops the
least active eligible rows between the water marks. Promotion mints
through the existing concept allocator, writes the referent and sets
activity to one. Those promotion/forgetting paths still need the complete
certificates and integration audit; their presence in source is not a
passing certificate 42.

Times name durable rows. `index_of_row` searches their increasing times;
there is no duplicate address column or posting map. The temporary
integer grammar operand wire distinguishes an occurrence time from a
concept with `Occurrence.operand_name`/`operand_time`; retained references
remain typed tuples. Zero is a valid occurrence time. Several callers
that treated it as an empty concept, or passed a bare time to a concept
reader, have been corrected. There is one retained history. The writer
preflights the two rows required by a two-lane idea conclusion.

### 23.3 The twelve failures of §22.2

They were this round's problems. None is attributed to a pre-existing
failure. None of these twelve whole test cases is retired.

| Test (file prefix `test/`) | Classification, cause and current standing |
|---|---|
| `test_thought_infer.py::test_trial_restore_keeps_active_streams_inside_the_constructed_batch` | **Defect.** Trial restoration called the durable loader and its legacy-time validator. `MemoryStore.restore_runtime_state` checks the existing construction and copies admitted tensors without checkpoint migration; `ThoughtStrata.restore` uses it. Passes in the returned-source check. |
| `test_item18_finish.py::test_equal_codes_are_said_by_their_given_object_references` | **Defect**, same rollback path. Passes in the returned-source check. |
| `test_reference_review10.py::test_thought_bind_is_credited_only_by_episode_advantage[0.0]` | **Defect**, same rollback path. Passed immediately after that repair. The final targeted check now stops earlier at an unported `document_key` call in `test_bind_thought.keep`. |
| `test_reference_review10.py::test_thought_bind_is_credited_only_by_episode_advantage[1.0]` | **Defect**, same repair and same outstanding fixture port as the preceding case. |
| `test_reference_close.py::test_kept_journal_supplies_np_content_predicate_and_positional_provenance[1]` | **Port.** The width-8, no-time-band store is working memory under point 15. It now uses STM0 and runtime restoration, preserving width/capacity and the content/reference/order assertions. The row-access-counter assertion is retired by point 14. Passes. |
| `test_reference_close.py::test_kept_journal_supplies_np_content_predicate_and_positional_provenance[2]` | **Port**, same contract and result. |
| `test_reference_review10.py::test_later_ball_mention_preserves_the_boxs_key_members_and_count[False]` | **Defect.** The original `unchanged` failure was relation activity changed by a cache rebuild. That producer is repaired. In the returned-source check a later reduction defect now leaves the pending token unmerged at the first equality; unresolved, not an assertion to retire. |
| `test_reference_review10.py::test_later_ball_mention_preserves_the_boxs_key_members_and_count[True]` | **Defect**, same original repair and new unresolved merge failure. |
| `test_reference_review10.py::test_answer_view_and_loss_judge_only_a_direct_root_site[False]` | **Defect.** `BindingAnswers.cost` caused `Definitions._ensure` to rebuild the definition index, whose `add` called `LTM1.evidence` and raised activity. The cache now reads the stored pair without an evidential-access side effect. Passed after repair; final check now stops earlier at `test_medium_term_memory.keep`'s unported `document_key`. |
| `test_reference_review10.py::test_answer_view_and_loss_judge_only_a_direct_root_site[True]` | **Defect**, same producer, repair and outstanding fixture port. |
| `test_item18_finish.py::test_relation_write_updates_reference_indexes_without_rebuild` | **Port.** Point 17 retires the operator reference and its index membership; all assertions of immediate pair/part/whole/reference reads remain. Passes. |
| `test_item18_finish.py::test_declared_caches_rebuild_and_are_not_checkpointed` | **Port.** Point 18 retires the operator-scaled effective-pole cache. The test exercises the surviving relation-order cache and a discarded legacy idea cache; read equivalence and exclusion from saved state remain asserted. Passes. |

The captured `unchanged` diagnostic identifies **two LTM1 activity
cells**, each differing by `0.10000002384185791`; it was not a changed
idea, pair or learned parameter. `Definitions.add` was the producer.
An unkept trial may not leave activity changed. Actual evidential reads
are now journaled per trial/stream, selected with the kept whole path,
and applied at admission; derived-cache construction is not such a read.
The complete discarded-trial/selected-trial activity certificate remains
due, despite the earlier repaired examples passing.

### 23.4 Other integration changes and checks actually run

The first full integration attempt after the row rewrite exposed working
STM names being assigned after their write, so a later named operation
could overwrite an earlier leaf's metadata. Names are now reserved
before the write. The flat-entry migration's tensor sort used an invalid
argument signature; the sort is corrected. `Queries` and reference
readers no longer discard time zero. A zero-length `ConceptSpread`
validity index is explicitly boolean. The generic attention read no
longer calls the removed LTM0 kind filter. The model no longer calls
the removed re-witnessing exclusion routine.

An unconditional attempt to generate an extension inside every direct
`append_idea` also added unrelated walks and reached a compiled-reader
failure on a bare scalar fixture. The generic writer now consumes a
supplied extension only. Sentence entries come from the kept field.
**The entry producer for a kept thought without an extension still
needs its complete audit**; this change does not certify an empty entry
as the right answer.

`SentenceFixture.program` explicitly supplies its declared trusted word
evidence, rather than leaving the new sentence-trust multiplier zero.
The four selected-tree `part`/`equal`, positive/negative certificate-56
cases then pass with their trust assertions intact. This remains a
selected-tree certificate, not evidence of a learned parse.

Receipts stay local and are not linked. These are native pytest counts
and times, followed by wrapper wall time where recorded:

| Check | Result |
|---|---|
| Reproduction of the cache-activity defect on §21's source | 1 failed, 3.70 s; wrapper 5.231307030 s. |
| The twelve §22 cases immediately after rollback/cache fixes | 10 passed, 2 failed, 7.25 s; wrapper 8.097253084 s. The two remaining failures were the unclocked journal fixtures, subsequently ported. |
| Idea-layout checks after the rewrite | 6 passed, 2.70 s; wrapper 3.314287901 s. The new flat/signed-row check had failed before the rewrite on §21's source. |
| First broad reduction integration | 37 failed, 67 passed, 4 skipped, 5 errors, 40.94 s; wrapper 42.152510166 s. |
| Next integration | 30 failed, 79 passed, 4 skipped, 48.97 s; wrapper 50.020384073 s. |
| After the producer repairs above | 4 failed, 105 passed, 4 skipped, 39.59 s; wrapper 40.605531931 s. Source `73b97a48dce2b56599aebe967285dde94d98230a6fefb839165e5909dee9b668`. |
| Whole-path file and two test ports | 1 failed, 21 passed, 17.49 s; wrapper 18.403373004 s. Source `392c76e17656e685f2736f25a1419a271c6acb7929a2266c53fe1917b3f65182`. |
| Returned-source check: the five §22 files plus `test_item18_reduction.py` and `test_item18_extend.py` | **19 failed, 92 passed, 4 skipped**, 25.24 s; wrapper **25.896426916 s**. Source is the returned hash below. |
| Default collection only | 7,922 collected, 14 collection errors, 16.91 s. This was not `make test` and is not a default sweep. |

The whole-path failure is
`test_whole_paths.py::test_kept_verbal_output_reaches_the_live_understanding_and_conditioner`:
its thought input has no nonzero gradient to a model parameter. A trace
finds a `part` step and repeated rule-17 binary bind steps in the
controlled fixture, and no parameter source already at the sentence
record. Changing its STOP score did not fix it; that attempted fixture
change is reverted. The test and its gradient assertions remain.
It is unresolved, not certified as either a harmless port or a repaired
gradient path.

The **19 returned-source failures** have these immediate producers:

- `test_reference_review10.py::test_later_ball_mention_preserves_the_boxs_key_members_and_count[False]`
  and `[True]`: the pending occurrence at time/name 0 remains its own MTM
  token instead of joining the box token at 3. The equality close/reference
  admission needs diagnosis; no assertion is removed.
- `test_reference_review10.py::test_arrival_cannot_merge_a_pending_token_with_other_members`,
  `test_own_content_excludes_an_observed_incompatible_antecedent[False]`
  and `[True]`, `test_answer_view_and_loss_judge_only_a_direct_root_site[False]`
  and `[True]`, and `test_merging_into_an_anonymous_token_adopts_the_joined_kind`:
  `test_medium_term_memory.keep` still passes retired `document_key` and
  `sentence_index` arguments to `write_clause`.
- `test_reference_review10.py::test_thought_bind_is_credited_only_by_episode_advantage[0.0]`
  and `[1.0]`: the same argument port is still required in `test_bind_thought.keep`.
- `test_priming_diffusion.py::test_class_centroid_weight_witnesses_cross_order_reach_and_context_skip`,
  `test_class_flags_roundtrip_with_fixed_shapes`,
  `test_noncontiguous_class_placement_remains_independent_of_spread`,
  `test_retired_functions_surfaces_switch_and_configuration_are_absent`,
  `test_seen_decay_validation_fails_loud[-0.1]`, `[1.1]`, `[nan]`, `[inf]`,
  and `test_seen_ledger_guard_decay_bump_desire_and_fired_rows`:
  `definition_fixtures.with_definitions` constructs an unclocked LTM0.
  That working-store fixture still needs the same point-15 port.

Collection errors likewise remain, rather than being hidden by
deselection: `test_creation_observations.py` imports `PresentationMemory`;
`test_document_covers.py`, `test_epoch_history.py`, `test_epoch_start.py`
and `test_presentation_memory.py` reach `DocumentCover`;
`test_forgetting.py` and `test_item18_retention.py` import retired
forgetting functions; `test_identity_experiment.py`, `test_identity_rows.py`
and `test_independent_components.py` reach `IdentityRows.access_count`;
`test_item20_round34_actions.py` imports `ADDRESS_DOMAIN`;
`test_heat_reverse_wiring.py`, `test_symbolic_heat_retrieval.py` and
`test_word_grain_attention.py` import the removed `Language.Taxonomy`.
These files have mixed contracts; a failing import is not grounds to
retire every test in a file.

### 23.5 Ports and removals: partial inventory

Against the checkpoint there are **18 edited existing test functions**
and **4 renamed/replaced functions**, excluding helpers and the two new
certificate files. This is an edit inventory, not a claim that every
affected test has been ported or passed. No whole test file has yet been
retired. The remaining retirement inventory/count is still due.

- In `test_item18_finish.py`, the nine edited names are
  `test_an_empty_query_concludes_its_unanswered_operand_not_the_request`,
  `test_an_empty_read_inside_a_conclusion_writes_its_operand_recursively`,
  `test_cued_read_only_reaches_explicit_or_recalled_rows`,
  `test_declared_caches_rebuild_and_are_not_checkpointed`,
  `test_entry_time_has_one_unambiguous_name`,
  `test_full_recall_window_materializes_only_the_selected_rows`,
  `test_recall_product_keeps_lanes_and_reads_only_selected_records`,
  `test_relation_write_updates_reference_indexes_without_rebuild`, and
  `test_scalar_views_do_not_rewrite_a_words_existing_member_collection`.
  They port flat entry indexing, typed times, working fixtures and the
  removed write arguments, with the exceptions below named explicitly.
- `test_recall_keeps_each_order_and_intersects_without_carrying` becomes
  `test_recall_uses_each_concepts_own_order_without_carrying`: point 12
  removes the stored order axis, so a reference reads at its own order
  regardless of the flat slot. The old wrong-slot exclusion assertion
  is retired, not native-order separation.
- `test_episode_selection_preserves_both_lanes_at_every_order` is
  replaced by `test_flat_episode_selection_keeps_the_strongest_lanes_across_orders`:
  point 12 retires a separate quota per order. Both lanes are still
  asserted, now in the one fixed quota.
- `test_old_episode_shape_loads_at_order_zero_with_warning` is replaced
  by `test_old_ordered_entry_loads_its_strongest_readings_with_warning`:
  point 12/certificate 47 retires the order-zero destination shape.
  This test is limited to entry flattening, not a full checkpoint
  migration certificate, and does not cover §23.1.
- `test_cache_rebuild_preserves_forward_slots_cues_episodes_and_retention`
  becomes `test_cache_rebuild_preserves_mtm_forward_slots_and_entry_reads`:
  only the retention-value assertion is retired by point 14; pending and
  entry-read equivalence remain.
- `test_reference_close.py` edits
  `test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them`
  and `test_revision_merges_three_accessible_groups_transitively_across_cued_documents`
  to read typed occurrences;
  `test_empty_close_has_no_policy_gradient_or_store_side_effect` reads
  STM working metadata, with the retired `metadata_required` assertion
  removed under point 20; and the journal test is the port in §23.3.
- `test_reference_review10.py::test_later_ball_mention_preserves_the_boxs_key_members_and_count`
  has the accepted distinct-retained-occurrence exclusion port and the
  removed source-key argument port. Its new merge failure remains above.
- `test_thought_infer.py::test_membership_match_uses_the_existing_part_cache`
  observes the actual taxonomy capture;
  `test_unused_derived_idea_is_forgotten_by_existing_configuration` fills
  the **remaining** seats of the unchanged capacity-4 store after a
  two-row conclusion; and
  `test_worked_token_property_line_queries_backward_and_forward_then_replays`
  keeps the proof/replay assertions, checks trusts `.7` and `-.3` in two
  rows, and retires only the durable kind/proof assertions of points
  21/22.
- `test_whole_paths.py::test_hidden_word_changes_only_loss_targets_through_the_close`
  compares occurrence times in place of the removed address column.

The current working-tree §17 also contains point 23 and point 6a,
recorded as Alec's decisions in §22.5/§18.6. I applied those deletions
and the preposition's object head: the unused `Language.Taxonomy`,
`attach_knowledge` and symbolic-priming paths/configuration are gone;
`MemoryPaths.write` and `ChainRead.path` are gone; the five named
`Spaces` writers are gone. `Taxonomy.py` and the used portions of
`MemoryPaths`/`ChainRead` remain. The traced production caller of the
chain writer was MTM promotion; it now keeps the existing membership
writes without making an unread adjacency chain. The symbol-taxonomy
attachment and the other named writers had no outside-test callers
apart from `reify_concept` calling `relate`. Their test ports and
certificate 64 are not yet complete.

### 23.6 Defaults and storage

The defaults already listed in §21.4 remain the intended choices,
subject to its stated certificate limits. Newly implemented defaults
are the name `referent`, promotion at the existing referent's native
order or one above the greatest entry order, the four activity values
`.5/.1/.9/.9`, strict `max(present, last+1)` time, stream-order admission
to one history, and the preposition kernel's object head. I found no
contrary semantic evidence for the name `referent`. Promotion, batched
time and the preposition head still need their full certificates.
The two-row trust default is built for new writes, but §23.1 exposes
the missing old-checkpoint rule. No stored-trust updater or automatic
contradiction repair is added.

Stored tensor sizes were inspected without allocating the full payload,
using the same largest construction as §14: 1,048,576 rows, width 1256,
four entry slots (previously four at each of six orders), old stream
capacity 28 and old recurrence capacity 262,144.

| Tensor payload | Checkpoint code | Current reduced code |
|---|---:|---:|
| All persistent arrays indexed by an LTM0 row, per row | 5,514 bytes | 5,096 bytes |
| LTM0 persistent tensor payload | 7,103,321,093 bytes | 5,343,543,353 bytes |

The older §14 figure **5,481** excluded the separately declared
per-row `metadata_required` (1 byte) and `semantic_fingerprint`
(32 bytes). Including those saved arrays gives 5,514; the earlier
whole-store payload is confirmed exactly. The new row is 5,024 bytes
of vector, 32 of entry references, 16 of entry values, 4 of trust,
16 of referent and 4 of activity. Its remaining persistent scalar
bookkeeping is 57 bytes for the store. The store reduction is
**1,759,777,740 bytes**. LTM1 is 30 bytes per pair plus its persistent
scalar bookkeeping. These exclude Python working state, serialization
framing, caches and process memory.

The surviving store declarations are: LTM0 has no derived caches and
declares `count`, `model_clock`, `percept_layout`, `truth_schema`,
`forget_passes`, `forget_sentences`, `forget_last_sentence`,
`forget_pending` as auxiliary state. LTM1's declared caches are `_pairs`,
`_parts`, `_wholes`, `_references`, `_index_version`, `relation_order`,
`_order_revision`; auxiliary state is `count`, `layout_version`.
The full reader/writer disposition table for every retired field is
still due; this declaration inventory does not stand in for that audit.

### 23.7 Earlier checkpoint: remaining work and returned source

The migration decision, unresolved integration failures and fixture
ports above remain. `git diff --check` also reports one extra blank line
at EOF in each of `ChainRead.py` and `MemoryPaths.py`; that cleanup is
still pending. So do the complete certificates 41–55, 63 and 64
(beyond the individual passing assertions), pending-state and old-layout
migration validation, the entry producer for all kept thoughts, the
removal reader/writer inventory, retirement counts, and the requested
trust-source traces. §21.2's limits remain: selected trees rather than
learned parses, controlled inverse, the whole negative thought path,
multiple open operands, generation work charging and the finished-source
lesson. The concrete comparison of `Intuition.pervasion` with point
11's read is still owed.

Neither stopping gate was run during this continuation. The earlier
passing gates in §21 are not a claim about this reduced tree. The
certificate-11 and certificate-13 reruns, 160-sentence lesson with the
extended-field comparison, and the one undeselected default sweep have
**not run**. The prior failures listed in §14 remain attributed there;
none is being repaired or reclassified by this partial return.

Returned source SHA-256, using `bounded_tests.source_snapshot` and the
compact sorted-JSON protocol, over **1,075** inputs:
`d3b04e2ee467ce8a43d9bcb83a62d1525a51f85734f28e0083c77fbc589328f8`.
Document prose and local receipts are excluded. HEAD remains
`ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd` on `main`; nothing is
committed or pushed.


### 23.8 Resumed repairs after Alec's checkpoint answer

The return below supersedes the status in §23.7. It is still a partial
return, with no claim to the finished-source sweep or lesson. The current
stop is the specific design question in §23.9, not the resolved checkpoint
migration. Nothing in §3.19 or the later proposed grammar reduction is
built.

The resumed work has these additional repairs and certificates:

- **Old checkpoints with both lanes:** the importer now rejects them
  before any destination tensor or migration state changes, as Alec
  decided. A single nonzero lane is preserved without netting. The
  two-store rejection certificate covers a valid earlier store and a
  rejected later one. Time ambiguity remains a separate rejection.
- **Relation endpoint confidence:** deleting the operator gate had also
  removed the existing endpoint-confidence factor. That was too broad.
  LTM1 again derives its effective pair from the stored pair and the
  least endpoint confidence; an operator contributes no factor. The
  caches `effective_plus`, `effective_minus`, `_effective_source` are
  declared with the other LTM1 caches and are not checkpointed. Definition
  lookup consumes the derived pair without registering a memory access.
  These caches add no stored bytes to §23.6.
- **Activity and trial restoration:** discarded reads roll back their
  activity changes; the selected read is applied once. An in-process
  STM restore does not run the old-checkpoint time migration. The
  activity, promotion, flat-entry, native-time and member/singleton
  referent certificates have been added to `test_item18_reduction.py`.
  Those selected-construction checks do not certify a learned parse.
- **Two signed witnesses in a timeline:** splitting a kept idea's two
  evidence lanes into two rows left its event pointing only to the
  positive row. `PartClosing` and `ClauseRow` now report both newly
  written occurrence references in their transient close result, and
  `ReferenceClose.link_events` attaches both to that event. No durable
  pairing column or alternative truth writer is added. The explicit
  timeline branch also remains reachable in the existence face. The
  repaired timeline/native reader set passed **45 tests in 15.79 seconds**,
  **16.550453901290894 seconds** at the wrapper.
- **Native referent restoration:** STM's reconstructed meaning ignored
  its saved referent, and loading a dangling referent did not reject it.
  The STM read now starts from the common idea read; both native stores
  validate reference tags and retained occurrence targets before load.
  Cyclic contexts retain their existing bounded-read contract; an
  unavailable target is rejected before mutation. The initial targeted
  checkpoint run had **4 failures, 50 passes and 1 skip in 10.81 seconds**.
  After the repair and the evidence-fixture port, the checkpoint/working
  memory set passed **55 tests in 85.30 seconds**, wrapper
  **87.58731317520142 seconds**.
- **An absent durable store:** `EpisodicMemory.stage` now returns when
  the model has no such store. The new common sentence commit had called
  staging unconditionally. The nonconsolidated streaming/packed close
  now passes, at its original construction.
- **Deleting the last definition:** `LTM1.effective_revision` formerly
  changed only when an effective pair tensor was written. Removing the
  final row writes no such pair, leaving a stale definition in
  `DefinitionIndex`. The revision now includes the underlying table
  revision. `test_definition_rows.py::test_28_forgotten_definition_does_not_survive_in_any_lookup`
  failed before and passes after this repair.
- **An occurrence used as a thought's relation operand:**
  `ThoughtStream.write` passed a full occurrence tuple to `Clause.refs`,
  whose tuple notation is a local child. It now encodes the existing
  occurrence time with the common `operand_name` wire. Both operand
  cases of
  `test_thought_binding.py::test_concluded_relation_keeps_its_occurrence_argument`
  failed with `unavailable child clause` and now pass. The reference's
  identity and stored idea are unchanged.
- **Pending content outliving a thought view:** `hold_pending` retained
  constituent views into transient thought stores, together with the
  extension's operator objects. Deep-copying MTM then recursed through
  an incompletely reconstructed view. The pending owner now materializes
  each constituent's detached values and metadata at admission; it does
  not retain the trial's extension tree or operator modules.
  `test_reading_round.py::test_native_read_sites_carry_one_token_across_three_documents`
  failed with that recursion and now passes, including restore. The
  latter three repair checks together passed **4 tests in 8.29 seconds**,
  wrapper **9.19625997543335 seconds**.

The unused persisted forecast readers `Meaning.retained_symbol_prediction`,
`Meaning.retained_symbol_activation` and `MeaningExpectation.retained_bindings`
are deleted under point 20. The live expectation, omission and loss
assertions remain; surviving surprise assertions read activity. The
source-side observer of the surface inverse now obtains each retained row
from its sentence field and compares all content coordinates, excluding
only the time band that point 15 necessarily distinguishes.

Tests have also been ported to typed occurrence wires, the flat entry,
immutable signed trust, explicit compaction, face-value truth ingestion,
the native referent, no covers and no persisted forecasts. The isolated
turnover port Alec approved still uses the unchanged capacity. A local
per-test ledger records the changed and retired contracts and their §17
points. Its deduplication and final exact counts remain due; this partial
return does not claim that every test port is complete.

The stopping gates have run during the resumed reduction: both passed in
**3.07 seconds** (wrapper **3.903339147567749 seconds**) on source
`c93c70bf8f353604320ff26429c890e3df7695923a478bf47e3b0eb958ee2dc1`.
Both also passed in the endpoint-confidence run: that run had **71 passes
and 1 non-gate failure in 16.52 seconds**, wrapper
**17.348823070526123 seconds**, on
`c745e24aec696ec1dcd32b2715b2fd225bb23219233c4b71fb9bf24a4dce4460`.
The failure there was
`test_argument_concepts.py::test_revised_membership_proof_names_formal_class`,
subsequently ported to the trusted two-lane taxonomy read. The raised-class
gate also passed in the later **134-pass/10-failure** integration run.
There has been no stopping-gate failure. These are not finished-source
certificate-49 results and do not replace the final runs. The §14 accepted
return had both gates passing; no tuning to recover those gates is done.

### 23.9 For Alec: which retained occurrences count for exclusion?

The earlier accepted decision, **count distinct retained occurrences in
the single history instead of `access_count`**, is implemented. This is a
separate eligibility read, not another question about that counter.

`ReferenceClose.exclusion_pairs` still admits a row if its native order
is one, **or** if its completed bindings contain `_reference_sites`.
`ReferenceClose.prepare` produces that marker for a determined mention,
including an anonymous one whose bind selected the new-instance dummy.
Point 20 removes the completed sidecar, and point 22 keeps pending
bindings in MTM only while pending. Point 12b correctly leaves an
anonymous row's referent empty when no reference was resolved. Its
native order is then zero. Replacing the sidecar read with `referent`
cannot recover this eligibility distinction.

I reproduced this with the unchanged native reference fixture: **capacity
512, width 12, percept width 8**. Each history has eight distinct
occurrences of coordinate 8 and eight of coordinate 9, with no joint
occurrence. Both histories are written through `store.write_clause`;
one goes through `ReferenceClose.preview`, `prepare` and `commit` with
the explicit new-instance bind, the other closes the same plain ideas.
The supplied model clock is the same in both histories, ticks 0, 100,
200, …, 1500; these are input times, not rewritten stored times. The
reference fixture supplies the kept choice, not a seed selected to make
an assertion pass. No capacity or dimension changes. After the closes,
MTM is expired with its native `restore(None)` so retained history, not
still-resident working tokens, must supply the evidence.

| Native read | Plain ideas | Anonymous mentions |
|---|---:|---:|
| Retained rows | 16 | 16 |
| Rows of native order zero | 16 | 16 |
| Empty referents | 16 | 16 |
| Old checkpoint code: rows with `_reference_sites` | 0 | 16 |
| Old checkpoint code: excluded coordinate pairs | none | `(8, 9)` |
| Current reduced code: excluded coordinate pairs | none | none |

The old code is the actual checkpoint commit
`ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`. In the current source all
**six** allowed per-row arrays are identical between the histories:
`slots` (including time), `episodic_concepts`, `episodic_values`, `trust`,
`referent`, and `activity`. The semantic coordinates and entry/trust are
also identical in the old writer; the difference its eligibility reader
uses is the retiring marker. Time, activity, an entry member or a cosine
cannot supply that missing distinction. A self-referent would invent a
reference the close did not select and would violate point 12b.

**Recommendation:** count every retained LTM0 idea occurrence for this
empirical exclusion read, with the existing eight-occurrence threshold
and no-joint-occurrence rule, and retire the old order/mention eligibility
filter. This preserves the minimal row and the single history; it does
broaden which observations can establish an exclusion. In the reproduction
both histories would then exclude `(8, 9)`, where the old code excluded
only the mentions. It would include higher-order ideas and kept thoughts
as well, which Alec should see before accepting it. If the old eligibility
must instead remain exact, the layout needs an explicitly authorized bit
recording that eligibility; the current referent cannot carry it.

**Question for Alec:** accept that broader retained-occurrence rule, or
preserve the old eligibility with a specified additional field? Neither
choice is built. §18 item 3 requires a stop for a surviving reader with
no specified replacement, and point 6 preserves the reference proposals
that consume this exclusion. Restoring the retired sidecar or quietly
changing the proposals to pass their tests would not resolve the design.

**Alec's answer (2026-11-13):** “Count all retained occurrences
(recommended)”. Work resumes. `ReferenceClose.exclusion_pairs` now counts
all retained ideas, without the order or mention-marker eligibility filter.
The eight-occurrence threshold and no-joint-occurrence rule are unchanged;
no field is added. The native equal-history certificate failed before
(1 failure in 2.36 seconds, wrapper 3.0219430923461914 seconds) and passes
after, including cancellation by a joint occurrence. With the native
reference-close tests and bind-menu exclusion test, the after run is
24 passed and 4 skipped in 7.61 seconds, wrapper 8.313663005828857 seconds.

### 23.10 Earlier resumed checkpoint: integration status and returned source

The latest broad targeted reader/port run had **270 passes, 62 failures,
10 skips and 1 warning in 245.57 seconds**, wrapper
**247.44709873199463 seconds**, on
`29cafd1ffc23699df34e1b2d001dd1c78d549178a7115eed0d37f1b25c85ab71`.
This is not the default sweep. The 62 cases, with their immediate
producers, are listed below. The three explicitly marked repaired cases
have subsequently passed; the others are not claimed repaired or
pre-existing. In particular, an observed error is not a completed cause
attribution where the table says diagnosis remains due.

| Failed case | Producer / current disposition |
|---|---|
| `test/test_relation_closing.py::test_three_slot_relation_closes_unindexed_numerical_operands` | Unreduced three-root/predicate fixture; selected relation formation after point 17 needs diagnosis/contract port. |
| `test/test_candidate_table.py::test_candidate_from_closed_chain_retains_its_declared_hole_at_the_part_writer[argin1]` | Thought writer mock lacks the declared episodic-capacity interface; port remains due. |
| `test/test_candidate_table.py::test_candidate_from_closed_chain_retains_its_declared_hole_at_the_part_writer[argin2]` | Thought writer mock lacks the declared episodic-capacity interface; port remains due. |
| `test/test_reading_round.py::test_documents_retain_memory_and_mtm_until_epoch` | Test still calls the removed presentation boundary; the history/expiry contract needs its point-16/20 port. |
| `test/test_reading_round.py::test_reading_pending_name_survives_documents_then_defines_on_recurrence` | Test still calls the removed presentation boundary; the history/expiry contract needs its point-16/20 port. |
| `test/test_reading_round.py::test_epoch_keeps_a_promoted_names_witnessed_order_after_deleting_its_occurrence` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_reading_round.py::test_reading_event_cases_accept_native_part_rows_and_filter_open_questions[True]` | Diagnosis remains due: AssertionError: Tensor-likes are not close! |
| `test/test_reading_round.py::test_kept_word_mention_retains_its_pending_referent` | Test reads the removed LTM0 evidence column; read the signed trust/evidence view. |
| `test/test_reading_round.py::test_thought_part_persists_a_named_token_through_its_native_reference[False]` | Diagnosis remains due: ValueError: unavailable child clause |
| `test/test_reading_round.py::test_thought_part_persists_a_named_token_through_its_native_reference[True]` | Diagnosis remains due: assert ConceptualMea...nce_kind=None) == ConceptualMea...nce_kind=None) |
| `test/test_reading_round.py::test_native_read_sites_carry_one_token_across_three_documents` | Repaired and rerun: pending MTM content retained a transient thought view; admission now owns its values. |
| `test/test_reading_round.py::test_defining_clause_promotes_a_named_reading_before_recurrence` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_retire_mode.py::test_closed_zero_is_unverified_and_only_open_references_make_question` | Expected error text uses plural open references; current writer rejects the same open reference. |
| `test/test_query_contract_boundaries.py::test_query_limits_native_content_reads_and_reports_incompleteness` | Fixture still expects old whole-frame/content retrieval; point-7–11 operated-field/inverse port or remaining defect needs diagnosis. |
| `test/test_query_contract_boundaries.py::test_unified_query_returns_complete_record_and_does_not_admit_observation` | Fixture still expects old whole-frame/content retrieval; point-7–11 operated-field/inverse port or remaining defect needs diagnosis. |
| `test/test_taxonomy_policy.py::test_unrelated_true_episode_cannot_establish_the_next_parent_relation` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_grammatical_query_vps.py::test_unary_query_keeps_full_description_through_an_existing_occurrence` | Fixture still expects old whole-frame/content retrieval; point-7–11 operated-field/inverse port or remaining defect needs diagnosis. |
| `test/test_replacement_kernel_contracts.py::test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees[0.8-0.7-BOTH]` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_replacement_kernel_contracts.py::test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees[0.8-0.1-TRUE]` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_replacement_kernel_contracts.py::test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees[0-0-UNKNOWN]` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_replacement_kernel_contracts.py::test_supplied_mechanism_walk_uses_one_controller_at_every_level` | Diagnosis remains due: AssertionError: (None, ['not', 'exist', 'intuit', 'ask', 'bind', 'query', 'query']) |
| `test/test_item6_2_thinking.py::test_nested_ask_returns_the_binding_and_shares_budget` | Diagnosis remains due: AssertionError: (None, ['not', 'exist', 'intuit', 'ask', 'bind', 'query', 'query']) |
| `test/test_item6_2_thinking.py::test_modus_ponens_requires_premise_and_implication_witnesses` | Diagnosis remains due: assert (0.8999999761581421, 0.0) == approx((0.8 ±....0 ± 1.0e-12)) |
| `test/test_item6_2_thinking.py::test_answer_credit_moves_shared_chooser_toward_the_completed_chain` | Diagnosis remains due: assert False |
| `test/test_item6_2_thinking.py::test_event_implication_reads_endpoint_when_order[positions0-True]` | Event fixture writes the removed relation-to-sentence column; port temporal operands to their native occurrences. |
| `test/test_item6_2_thinking.py::test_event_implication_reads_endpoint_when_order[positions1-False]` | Event fixture writes the removed relation-to-sentence column; port temporal operands to their native occurrences. |
| `test/test_item6_2_thinking.py::test_event_implication_cannot_invent_an_order_for_unknown_legacy_time` | Event fixture writes the removed relation-to-sentence column; port temporal operands to their native occurrences. |
| `test/test_thought_binding.py::test_concluded_relation_keeps_its_occurrence_argument[0]` | Repaired and rerun: thought writer passed an occurrence tuple as a local child; common time wire now used. |
| `test/test_thought_binding.py::test_concluded_relation_keeps_its_occurrence_argument[2]` | Repaired and rerun: thought writer passed an occurrence tuple as a local child; common time wire now used. |
| `test/test_thought_binding.py::test_empty_read_does_not_bind_its_temporary_question_address[incomplete0-ask]` | Thought writer mock lacks the declared episodic-capacity interface; port remains due. |
| `test/test_thought_binding.py::test_empty_read_does_not_bind_its_temporary_question_address[incomplete0-query]` | Thought writer mock lacks the declared episodic-capacity interface; port remains due. |
| `test/test_thought_binding.py::test_empty_read_does_not_bind_its_temporary_question_address[incomplete1-ask]` | Thought writer mock lacks the declared episodic-capacity interface; port remains due. |
| `test/test_thought_binding.py::test_empty_read_does_not_bind_its_temporary_question_address[incomplete1-query]` | Thought writer mock lacks the declared episodic-capacity interface; port remains due. |
| `test/test_item6_2_repair.py::test_saved_construction_binds_and_writes_the_child` | Count expects one row for a two-lane conclusion; the signed-row port must retain child/scope assertions. |
| `test/test_item6_2_repair.py::test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership` | Count expects one row for a two-lane conclusion; the signed-row port must retain child/scope assertions. |
| `test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[forward]` | Diagnosis remains due: TypeError: 'NoneType' object is not subscriptable |
| `test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[batch_eval]` | Diagnosis remains due: TypeError: 'NoneType' object is not subscriptable |
| `test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[packed]` | Diagnosis remains due: TypeError: 'NoneType' object is not subscriptable |
| `test/test_selected_nested_meaning.py::test_10_packed_pending_and_eager_ends_write_identical_rows` | Diagnosis remains due: TypeError: 'NoneType' object is not subscriptable |
| `test/test_selected_nested_meaning.py::test_constituent_capacity_failure_and_invalid_local_ref_do_not_partially_write` | Diagnosis remains due: ValueError: LTM0 stores one idea; write semantic pairs through write_clause |
| `test/test_expectation_defaults.py::test_native_runtime_reports_pairs_without_accumulating_or_updating` | Predicted-target count includes the retired document cover; live two-observation and no-update assertions remain. |
| `test/test_expectation_defaults.py::test_expectation_off_keeps_every_packed_closing_in_ltm` | Observer reads the removed source-address scratch; use the kept sentence/root reference. |
| `test/test_math_chain_section14.py::test_copula_fills_the_addressed_open_column_without_a_word_alias` | Diagnosis remains due: assert None == (0, 0) |
| `test/test_math_chain_section14.py::test_region_query_ignores_free_variable_code_and_preserves_a_pending_match` | Fixture still expects old whole-frame/content retrieval; point-7–11 operated-field/inverse port or remaining defect needs diagnosis. |
| `test/test_identity_threading.py::test_frame_access_does_not_leak_a_native_individuals_other_stream_value` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_identity_threading.py::test_three_slot_close_keeps_the_introduction_used_by_its_embedded_bind` | Unreduced three-root/predicate fixture; selected relation formation after point 17 needs diagnosis/contract port. |
| `test/test_identity_threading.py::test_dummy_binds_preserve_the_relative_operand_in_a_three_slot_close` | Unreduced three-root/predicate fixture; selected relation formation after point 17 needs diagnosis/contract port. |
| `test/test_identity_threading.py::test_resolved_compose_bind_keeps_the_parent_relation_closed[0]` | Diagnosis remains due: AssertionError: Tensor-likes are not equal! |
| `test/test_identity_threading.py::test_resolved_compose_bind_keeps_the_parent_relation_closed[2]` | Diagnosis remains due: AssertionError: Tensor-likes are not equal! |
| `test/test_identity_threading.py::test_unminted_value_and_anonymous_name_have_distinct_pending_state[False]` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_identity_threading.py::test_unminted_value_and_anonymous_name_have_distinct_pending_state[True]` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_identity_threading.py::test_nested_unresolved_operand_keeps_its_resolution_annotation[True-True]` | Diagnosis remains due: AssertionError: assert (() == ((2, ('ltm', 'stm-v1', 0)),) |
| `test/test_identity_threading.py::test_relative_determination_keeps_its_introduction_and_later_bind[False-False]` | Diagnosis remains due: AssertionError: assert 136 == -1 |
| `test/test_identity_threading.py::test_relative_determination_keeps_its_introduction_and_later_bind[False-True]` | Diagnosis remains due: AssertionError: assert 136 == -1 |
| `test/test_identity_threading.py::test_relative_determination_keeps_its_introduction_and_later_bind[True-False]` | Diagnosis remains due: AssertionError: assert 136 == -1 |
| `test/test_identity_threading.py::test_relative_determination_keeps_its_introduction_and_later_bind[True-True]` | Diagnosis remains due: AssertionError: assert 136 == -1 |
| `test/test_identity_threading.py::test_two_relative_determinations_keep_distinct_objects` | Diagnosis remains due: assert 0 == 2 |
| `test/test_identity_threading.py::test_journal_native_bind_without_a_pool_token_keeps_its_symbol_origin[2-False]` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_identity_threading.py::test_journal_native_bind_without_a_pool_token_keeps_its_symbol_origin[2-True]` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_generate_repair.py::test_promoted_order_survives_early_occurrence_forgetting_and_two_epochs` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_generate_repair.py::test_promotion_rejects_missing_witness_at_creation` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |
| `test/test_generate_repair.py::test_epoch_guard_still_rejects_unwitnessed_constructed_individual` | Fixture still supplies the removed row-order argument; native referent/order port remains due. |

The accepted old failures are still tracked separately: the timestamp
reader inventory mismatch, the sparse interpreted-object feature
inventory mismatch, and the intermittently empty/NUL reconstruction bank
in `test_reasoning_cde_model.py::test_answer_query_degrades_gracefully`.
The timestamp inventory run also sees this round's new Extension readers;
those additions need their inventory port without disguising the original
mismatch. No new failing case above is classified as old merely because
its test predates the round.

Certificates 11 and 13/49 on the finished source, the 160-sentence
explore-1 lesson comparing the operated understanding field with its
generated field, and the exact undeselected
`env -u RUN_TRAINED -u RUN_SLOW make test` **have not run**. The final
port/retirement counts, every removed-reader disposition, remaining trust
traces, and the numerical comparison of Intuition's pervasion with
point 11's relation read are also still due. §21.2's integration limits
remain. Local receipts remain local and are not linked.

Current returned source SHA-256, over **1,067** source inputs, using the
same compact sorted-JSON snapshot protocol:
`0b90608cd35a04ed55e9b04bccaaf0651a32240452c774141b35bc72a9d3b554`.
Document prose and local receipts are excluded. HEAD remains
`ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd` on `main`. Nothing has
been committed or pushed. Source work stops for §23.9's decision.

### 23.11 Resumed after the all-occurrences decision

Alec accepted counting all retained occurrences for exclusion. The new
certificate failed before that change and passed after it, as recorded
in §23.9. Counting an arbitrary idea also exposed an inappropriate
identity-chart validation: a kept idea need not itself be a bounded
identity region. The exclusion count now reads its conceptual coordinates
with `IdentityRows.coordinates(..., region=False)`, still excluding form
and time, and counts positive coordinates at the existing 0.3 threshold.
The identity reader's bounds are unchanged. No read adds an observation.

The next integration batch returned **204 passed, 42 failed, 2 skipped**
in **79.98 seconds**; wrapper **81.00362992286682 seconds**. The cause of
one group was the obsolete default trust of an unresolved fixture: the
fixture explicitly writes its intended zero trust now; no stored trust
is mutated. A fixture with no percept time band uses working `STM0`, at
its unchanged width and capacity. The operation-journal fixture supplies
the actual extension faces for the operators its selected tree uses.

Two production repairs followed. `ClauseJournal` had made *every*
non-relation operator a point when implementing point 17. That lost a
determiner's relative introduction. Only the sentence and verb-phrase
forms receive that change; other forms keep their child's point status.
The five existing relative-determination cases now pass. `CaseSelection`
spent all three available case seats on the negative half of a two-row
conclusion. It now joins the two lanes of identical before/change/after
cases, within the existing inspection and seat limits. Both existing
signed-event cases pass, with their assertions unchanged. The first
focused batch was **22 passed, 4 fixture failures**, **8.15 seconds**;
wrapper **8.805958986282349 seconds**.

An addressed equality wrapped directly in `query` had lost its reader
when role matching went: its `_equality` declaration was not dispatched
to `relation_evidence`. The declared relation now uses the indexed LTM1
read. No vector or unscoped table scan establishes that declaration.
The nested ask, the answer-credit chain, and the supplied complete
controller walk now pass. A working idea also lost its `idea` type when
`STM0` copied an unspecified source metadata field over it; the working
writer now supplies that type. Its address-only reread again sees live
content edits. The batch containing both stopping gates returned
**154 passed, 9 failed**, **61.87 seconds**; wrapper
**62.852712869644165 seconds**. **Both stopping gates passed.** The next
focused equality/identity run returned **17 passed, 1 fixture failure**,
**5.91 seconds**; wrapper **6.562699794769287 seconds**. These are
integration runs, not the finished-source sweep.

The requested comparison leaves `Intuition.pervasion` unchanged. On the
same `isPart` question, with two stored member edges of `(0.8, 0.4)` and
`(0.5, 0.9)` and equal full codes, it returns
**(0.5, 0.4000000059604645)**. The point-11 reader
`ExtendedRecall.pervasion` returns **(0, 0)** and an empty path: no raised
property class and no relation covering that class was supplied. The
first read measures its existing member/content construction; the second
requires the specified higher-order relation. They are not interchangeable.

The requested definition trace changes no policy. An explicitly counted
DEF at **0.30000001192092896** is below `definitionTrust`: `forward`
returns no object and leaves it unchanged. At **0.699999988079071**, an
unfrozen `Interpret.forward` returns the known object and calls `define`
without trust or evidence; the stored and effective positive lane both
become **1**. Its object identity does not change, but subsequent evidence
reads do. `positive_length` on an unallocated concept returned **1**.
The other unspecified `define` callers are `lookup_word`'s discovered
object path, `reading` of a pending name already promoted in MTM, and
`admit_discovered` of such a pending name. Each can raise an existing
counted DEF by LTM1's maximum. `PartClosing.finish_definitions`,
`PerceptCollections.observe`, and the three `FamiliarTier.promote_identity`
paths pass evidence or trust explicitly. The two readers of
`positive_length` are `Priming.part_edges` and `LTM1.refresh_effective`;
the former bounds a priming edge and the latter bounds its effective
relation evidence. No fallback policy was changed.

### 23.12 Further integration after the retained-occurrence decision

This remains an unfinished return. No commit or push has been made. The
subsequent pending-operand/time batch passed **49 tests**, with one fixture
failure, in **11.44 seconds** (wrapper **12.16864013671875 seconds**).
The capacity fixture expected an unavailable child to be rejected at write;
the new selected-clause constructor rejects the same child earlier. The
assertion now encloses that construction. A compressed idea now carries
its unresolved children's descriptions through MTM; the annotation expires
with MTM and is not a saved operator derivation. The writer's semantic-chain
and checked-binding guard remains intact.

The next reader integration batch returned **174 passed, 9 failed,
6 skipped**, **154.06 seconds** (wrapper **155.3278772830963 seconds**).
Six failures assumed that an explicit occurrence address established a
content match despite having no stored field. Points 8–11 remove that
role-scoring contract. The direct-read tests now use `isTrue`/`end_evidence`
for the same content, occurrence and trust assertions; the query with no
entry establishes no field match. The bounded-candidate test separately
checks both explicit reads and the new zero match. The conflicting-degree
test reads both signed occurrences and keeps its degrees and all three
postures. No address is assigned a resemblance score to preserve an old
query assertion. The detailed name mapping is still being collected for
the final inventory.

The other causes in that batch were an ownerless open-slot fixture (its
pending metadata now belongs to the native MTM owner at the same width and
capacity), a noncanonical pickle-envelope hash, and a time collision.
The warmup adapter already compared decoded occupation metadata; its hash
now uses the same decoded metadata while every parameter tensor remains a
byte hash. A stationary model clock can write rows at times 0, 1, 2, 3;
the next thought previously took time 1. The new certificate failed with
**1 > 3** in **4.43 seconds** (wrapper **5.3359880447387695 seconds**).
`Models._advance_when_time` now advances past both the current model time
and the next retained-row time, without changing any retained row. The
focused run then passed **36 tests**, with **4 skipped** and one remaining
slot failure, in **41.16 seconds** (wrapper **42.102213859558105 seconds**).

That slot failure exposed another producer: `Occurrence.operand_time`
accepted every negative integer, including the hash of an open MTM slot.
It now accepts only the declared row-time ladder's range, and `slot_key`
rejects a hash in that reserved range. The original addressed-slot test
passes with its slot identity, immutable source and reconstruction
assertions. The next address/slot batch passed **40 tests** with two
remaining failures in **12.14 seconds**; a follow-up passed **33** with
one remaining failure in **9.79 seconds**. The posture fixture now uses
the normal controller's unchanged `.5` posture, instead of the separate
identity-match threshold. The last failure was a production integration
omission: `isTrue` returned the addressed meaning, but `ThoughtStream`
kept the executable `isTrue` request as the conclusion. Integration now
takes the returned meaning, as `ask` and `query` already do. The writer's
guard is unchanged.

The public trust-writer certificates exercise `store_truths` and
`provision_ltm` at owner trust **0, .25 and 1**, with supplied trust
**.3, -.6 and 1**; the signed values are preserved. A subsequent audit
found that the test must also vary `Models.trust`, which the removed
scaling read; certificate 50 is not yet claimed for that comparison.
Resubmission appends
and leaves the first trust unchanged; the existence maximum remains
**.3**. Inference at exact match writes **min(.8,.6)=.6** and
**min(.4,.9)=.4**. A kept both-lane thought writes **.7** and **-.3** at
two distinct times. The serving high-water check runs the activity pass
before its truth write. These tests use the unchanged serial fixture and
its native close, without its old forced-reading-evidence adapter.

Certificate 54 initially failed: two supplied contradictory claims were
stored at **.8** and **-.6**, but the report was empty. `TruthLayer` still
inferred a claim's polarity from paired-coordinate parity even though
its LTM view now reads signed trust. The LTM-backed contradiction read
now compares the unscaled ideas with the existing containment read and
takes the two signs from their trust. It changes neither stored idea nor
stored trust. The first group returned **14 passed, 1 failed**,
**12.84 seconds** (wrapper **13.713382244110107 seconds**); the repaired
trust/view/ingestion group returned **88 passed, 2 skipped**,
**51.21 seconds** (wrapper **52.105146169662476 seconds**). The subsequent
integration group passed **73 tests**, including **both stopping gates**,
in **25.13 seconds** (wrapper **26.12626624107361 seconds**), on source
`f79ca91db9aee54f72924a4ca0b4c687f98b859bfe2117662ec5d73207ccc4c3`.

Certificate 51's two cases then passed: actual training writes the
sentence's word certainty and `Models.infer` writes zero. The closed
question reads for in the former and neither in the latter; the open
question reads the operand in both. This uses the unchanged serial
fixture, its supplied legal conjunction parse, and an independently
supplied legal inverse executed by the native generate walk. It proves
the storage and answer contract, not learned selection of that inverse.
The run passed **2 tests**, wrapper **4.982849836349487 seconds**, on
source `1335e2b7ca8fdd609b38c59053095332d240c7f09923b8ce540df72abdea3fb8`.
The final proofs, lesson and default sweep remain due.

### 23.13 For Alec: the promotion reader of the removed mention marker

The accepted all-retained-occurrences decision is built for exclusion
(§23.11). The continuing reader audit found a second consumer of
`_reference_sites`: `FamiliarTier.promote_identity`. Its form fold takes
MTM's actual role-zero event witnesses and also retained order-zero
occurrences that did **not** carry that marker. It excludes grammatical
mentions from this second group because a mention can belong to a modal
branch. If no such observations remain, it uses `MTM.value_of` instead.
§17.6 retains reference proposals, and no point retires this promotion
distinction. Activity promotion is a separate writer.

The reproduction uses the existing 512-row, 12-coordinate reference-close
fixture, unchanged. In one history two native closes keep plain ideas;
in the other they keep anonymous grammatical mentions. Their first two
content coordinates are **(1, 0)** and **(0, 1)**, their times **0** and
**10**, and each is written at trust **1**. Both histories have empty
referents and equal entries. After MTM is cleared, `recall_identities`
re-enters both rows, and the existing MTM merge puts them in one token.
The actual-event list is empty, and the token's current value begins
**(1, 1)** in both histories.

On the checkpoint code, plain ideas promote the form **(.5, .5)** and
marked mentions promote **(1, 1)**. On the current source, both promote
**(.5, .5)**. All six permitted LTM0 columns are equal between the two
current histories; after recall their MTM link clocks, member kind,
member order, member references and actual-event lists are equal too.
`refinement['mentions']` is a clock for every link and does not recover
the deleted grammatical-site distinction. The existing pending-binding
map does not retain a completed anonymous mention. Thus replacing this
reader with either of those reads would silently change its contract.

The old comparison used a copy whose `bin`, `data`, `test` and `Makefile`
were checked byte for byte against checkpoint
`ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`: **zero differing files**.
The current reproduction completed in **2.980800151824951 seconds** on
source `1335e2b7ca8fdd609b38c59053095332d240c7f09923b8ce540df72abdea3fb8`.
The receipts remain local.

**Question and recommendation.** May this observed-form fold take only
MTM's existing actual role-zero event witnesses, using its current value
when there are none? I recommend that rule: it adds no stored marker and
keeps modal events out of the observed-event fold. It deliberately changes
the plain recalled pair above from **(.5, .5)** to **(1, 1)**; that change
needs a decision. Counting every retained occurrence here, as for
exclusion, would also count grammatical mentions from modal branches.
Preserving the old distinction after MTM expires would require additional
durable information beyond point 12's six columns.

§18 item 3 requires stopping for a surviving reader without a specified
replacement. No promotion rule has been changed for this question. The
larger integration check was already running when this reproduction
completed. It was interrupted for this required design stop, after its
progress output showed **1,102 pass marks, 42 failure marks and 32 skip
marks**. Those are partial progress counts, not a completed result; the
interruption produced no failure summary, and those failures have not
been attributed or repaired. Neither stopping gate produced a failure.
This was a check of 165 changed test files plus the XOR gate, not the
final default sweep. No further test or implementation change followed
the design question. No commit or push has been made.

The returned source hash, checked after recording this question, is
`1335e2b7ca8fdd609b38c59053095332d240c7f09923b8ce540df72abdea3fb8`.
The finished-source proofs, lesson, full sweep, complete reader inventory
and complete named test-port/retirement inventory remain outstanding.

### 23.14 Resumption: actual MTM events and integration repairs

Alec answered §23.13: **use actual MTM events**. `FamiliarTier.promote_identity`
now folds only the existing actual, role-zero event witnesses. Where none
contribute it takes `MTM.value_of`. The retained-occurrence fallback and
its read of `_reference_sites` are deleted. The two unchanged histories
in §23.13 both promote the current **(1, 1)** content; no eligibility
marker is stored. The new two-case certificate failed before the change
in **2.43 seconds** (wrapper **3.051406145095825 seconds**), then passed
with the existing actual/modal-event certificate and six truth-writer
cases: **9 passed**, **9.73 seconds** (wrapper **10.585153102874756
seconds**). This implements Alec's decision, including its deliberate
change to the recalled plain-idea case.

The interrupted run's 42 failure marks were mapped to collected test
names and reproduced: **42 failed**, **13.15 seconds** (wrapper
**13.870009183883667 seconds**). After ports and the repair below, the
40 surviving cases and the new restore certificate passed: **41 passed**,
**12.33 seconds** (wrapper **13.051183938980103 seconds**), on source
`69fe721afd01d942ee8cac3029fa58766ae3a10a3e5668abab9adcc978cc1d09`.
The remaining integration cases are a separate run; these counts do not
claim the finished-source default sweep.

The new restore failure came from the checkpoint's MTM reader, not from
`RowTime.checkpoint_times`. Native `record_link`/`keep_access` may record
a link for the current close at `closes[stream] + 1`, including the first
close before that stream has a completed clock. `MediumTermMemory.restore`
rejected that state. The prior full checkpoint test passed because the
old model saved MTM only through `_epoch_mtm`; the new checkpoint saves
MTM itself. The reader now accepts the current link, and still rejects a
clock beyond that close, an invalid stream or a missing entry. Its new
certificate failed before the repair in **2.30 seconds** (wrapper
**2.94073486328125 seconds**); both its lossless native round trip and its
atomic rejection of a future clock pass in the 41-case run. The scalar
checkpoint test passes there too. Nine comparison cases on a copy verified
against checkpoint `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`
passed in **6.16 seconds**, confirming their previous behavior.

The test ports preserve address identity through the typed occurrence
reference or `Occurrence.operand_name(time)`, including time zero; a raw
zero must not become an open-operand sentinel. They also use flat entry
selection, signed idea trust, and native MTM fixtures where the removed
sidecar supplied pending state. The isolated relation-forgetting checks
use `parts.retain(())` in place of epoch withdrawal and keep their
read/write assertions. The determined-mention capacity test keeps its
64-row fixture and its one/two-free-row cases: it supplies a direct
shared-pair composition and a single positive claim; the retired
projected operator-chain representation and the two-lane idea row are
not reconstructed. Two-lane conclusions have certificate 53's two rows.

Two tests from this group are retired by name:
`test_item20_round5_repair::test_short_term_stream_table_rejects_another_stream_without_redirection`
under point 16, and
`test_item20_round7_time::test_epoch_restores_both_clocks_and_repeats_sentence_stamps`
under points 15–16 and 20–21. The complete named inventory remains due.

Certificate 50 now varies **both** `Models.trust` and the conceptual
owner's trust at **0, .25 and 1**. Its six public-writer cases pass with
the supplied **.3, -.6 and 1** unchanged, closing the test limitation
recorded in §23.12. No commit or push has been made.

### 23.15 Further causes found in the completed integration run

The remaining 838 integration cases returned **762 passed, 54 failed,
22 skipped**, wrapper **1062.9748191833496 seconds**, on the source named
in §23.14. **Both stopping gates passed.** This was the remainder of the
interrupted changed-test check, not the default sweep. The next focused
run returned **94 passed, 6 failed**, **23.55 seconds** (wrapper
**24.300790786743164 seconds**). Its follow-up returned **79 passed,
3 failed**, wrapper **16.20738911628723 seconds**; the three remaining
failures shared an incomplete reconstruction mock. After completing that
mock, all four of its cases passed in **2.50 seconds**, wrapper
**3.102139949798584 seconds**. Final certificates, measurements and the
default sweep are still separate obligations.

The production causes and repairs are:

- `LTM0.order_of_row` followed its new concept referent without passing
  the existing cycle set to the allocator. A relation could follow an
  occurrence back to its own concept indefinitely. It now passes that
  set across the table boundary. The unchanged assertions of
  `test_sentence_identity::test_full_symbol_inventory_needs_no_seats_and_chain_exhaustion_is_atomic`
  failed with `RecursionError` before and pass after the repair.
- The new indexed relation route bypassed the unresolved-endpoint guard
  that the former row-candidate route used. `relation_evidence` now keeps
  that guard at its own two-end read: an unresolved end can match its
  exact supplied reference, but cannot supply a missing answer end.
  The two directions of
  `test_query_executors::test_query_returns_only_the_completed_direction_of_an_unresolved_equality`
  use the indexed reader and its shared meter, retain their answer/no-answer
  assertions, and pass. Before this repair the completed direction was
  returned in both cases. The first port also exposed its fixture's
  noncanonical predicate handle and omitted meter; those were fixture
  corrections, not changes to the reader's semantics.
- The higher-order proof ignored a question's outer `not`/`non`, and
  `ThoughtFaces.negate` changed the intension and evidence while leaving
  the existing extension positive. The former now applies the question's
  declared unary polarity to the relation's two proof lanes and paths;
  the latter wraps its existing extension with the same parameter-free
  `NotLayer` used by grammar. Direct declared-relation reads use the same
  orientation. For a member-to-kind edge **.8** and a kind-to-property
  edge **(.6,.2)**, the positive, negative and withheld forms read
  **(.6,.2)**, **(.2,.6)** and **(0,0)**; the withheld form carries no
  proof witnesses. Negating twice restores the original field. These
  certificates returned **3 failed, 1 passed** before the repair,
  **2.67 seconds**, wrapper **3.2852401733398438 seconds**, and pass in
  the follow-up. They exercise the public reader and thought face, not
  learned selection of an entire negative sentence.
- `ExtendedRecall.forest` charged generation after an unrestricted
  native walk. The native `Models.generate` now bounds its trips by the
  same meter before walking and charges its executed actions; exhaustion
  does not return an unfinished answer tree. Its ordinary callers pass
  no meter and keep the construction-time walk budget. With three named
  operands and allowances **0, 1 and 3**, the native walk now runs
  **0, 1 and 3** actions, respectively. The first two exhaust; the last
  returns all three roots. All three before-repair cases failed because
  the walk still received **32** trips (**2.73 seconds**, wrapper
  **3.360962152481079 seconds**); all three pass after.
- The region reader threw away already acquired evidence when the next
  witness exceeded the shared allowance. It now admits witnesses with
  that allowance and reports only their maximum, preserving the acquired
  exact statement when an unacquired analogy is brighter. With the
  unchanged four-row certificate and allowances **128, 1 and 0**, the
  deductive lanes are **(.3,.6)**, **(.3,0)** and **(0,0)**; the last two
  report `work_budget`, and spend **1** and **0**. The limited cases
  failed before, and all three pass after. No assertion gets the trust
  of an unacquired row.

The fixture audit also found 57 conversions from time-less `LTM0` to
`STM0` that had inherited STM0's smaller default relation capacity. All
57 now explicitly preserve their original **8192** relation slots and
their original idea capacities. This corrects the ports; no capacity was
selected to fit an observed outcome. A five-coordinate working fixture
keeps its input graph, as STM0 does; its placement/readback assertions
still require detached derived codes. Durable LTM0 keeps its detachment
contract. The original finite out-of-range trust-clamping assertion is
also restored: point 12a did not retire it. Valid supplied trust values
remain unchanged, and nonfinite inputs are rejected.

Other ports supply the native dataclass, empty field, clause list, root
count and STOP trace at the new sentence/generation seams; preserve
typed time-zero references; replace removed document keys with the same
second write; and mark the promotion fixture's three observed forms as
actual role-zero MTM events under Alec's §23.13 decision. Its weighted
centroid, one placement step, view count and radius assertions all pass.
The reference-participant test counts its original reference participants
separately from the event now held in MTM. Its selection assertions are
unchanged. The set-answer adapter now asks an actual open relation;
a closed row address is no longer its former role-match query. Its
owned answer, checkpoint round trip and no-additional-read assertions
pass, with the thought writer's guard unchanged.

Additional complete retirements are
`test_operators_round4a::test_content_address_is_independent_of_store_and_persists_with_occurrences`
(point 20's content-key column/index),
`test_reference_arrival_review::test_arrival_readers_exclude_scoped_rows_before_a_new_stream_column_exists`
(point 16), and
`test_truth_grounded_reasoning::TestEvaluate::test_is_true_true` and
`test_truth_grounded_reasoning::TestEvaluate::test_is_true_false`
(points 7–11: a bare vector with no entry and no conceptual owner cannot
light a region by the former role match). Their positive and negative
stored-trust read-back assertions remain in `TestIsTrue`, through the
addressed evidence reader. The full named inventory below will distinguish
these retirements from ports and partial retirement of old assertions.

The time-reader inventory includes this round's three new Extension
readers. Its prior mismatch between the two radix-reader names remains;
the round does not repair that pre-existing failure.

### 23.16 The last fixture audit and a copied-model defect

The first completion certificate run passed **173 cases in 189.28 seconds**,
wrapper **192.27761721611023 seconds**, peak **4,334,588,032 bytes**, on
`84d2dc5e26169ffd55c5772dc35adde6269a83d488eb8e246bcae654a588034a`.
Both stopping gates passed. Native scalar acquisition and both omissions
also passed, and ten numeric XOR runs completed. These are intermediate
results: the following audit changed the source before the final runs.

The renamed-function audit found **16 more** working-store constructors
whose relation capacity had inadvertently become STM0's default of two.
As with the 57 in §23.15, each now explicitly keeps the original **8192**.
No idea capacity, example count, dimension, or assertion was changed by
this correction. I interrupted the unfinished lesson, verbal proof and
default sweep before editing. The interrupted sweep ran **151.06926254197606
seconds**, wrapper **151.17243003845215 seconds**; it was not a completed
default sweep and reached neither stopping gate. Its fourteen failures
had these producers:

- Nine embedding/bivector tests shared the time-less `_make_word_space`
  fixture. Under point 15 this is a working-memory fixture. The four
  isolated SymbolSubSpace factories in `test_partition_pos_codebook`,
  `test_partition_rule_predictor`, `test_partition_reconstruction_stack`
  and `test_per_batch_state_isolation` now supply STM0 at their original
  widths and idea/relation capacities. The embedding assertions stay.
  This is a fixture substitution, not a time-less mode of LTM0.
- `test_case_views::test_phase_question_uses_selected_content_and_never_the_event_clock`
  selected the phase of only the first of the two signed event witnesses
  which now represent its BOTH statement. It now selects that content on
  both witnesses. Its original **.8/.2**, partial-content and independence
  from event-clock assertions are unchanged and pass.
- The two `test_clarifications` failures expected stored source text and
  the former message. Point 21 retires those expectations. The ports keep
  the one contradiction and both numeric trusts, and assert the new
  index fallback when no store is attached. A fabricated `None` trust
  is not a row of the signed-trust layout. Certificate 54 separately
  checks attached-store time naming and generated wording.
- `test_clause_grammar::test_complete_grammar_declares_sentence_predicate_generic_and_implication`
  still required `operator_argout1`. Only that assertion is reversed
  under point 17; the other starts and clause forms stay. The equivalent
  set assertion in `test_role_collapsed_grammar::test_symbolicspace_owns_its_starts`
  is ported too.
- `test_compiled_word_chunk::test_no_grad_fallback_retains_eager_stm_depth_semantics`
  found a production ownership defect. Its unmodified test passes on an
  archive of `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`, **1 passed in
  6.40 seconds**. In this round a deep copy's MTM weak owner still pointed
  to the original model. A copied occurrence at time 1 was read against
  the original history at times **3,4,5,6,8,9**, instead of the copied
  history at **1,2,3,4**. Repairing just MTM exposed the other stale weak
  link: the copied thought STM advanced the original model's clock.
  `MemoryStore.__deepcopy__` now copies the targets of its MTM owner and
  thought-clock owner through the same deepcopy memo. The existing
  clause-index attachment also rebinds the relation store's conceptual
  owner. The original test, with its original seeds and every equality
  assertion, now passes. This repairs model ownership; it does not
  suppress a clock coordinate from the comparison.

The fixture/grammar follow-up returned **41 passed, 1 failed in 6.29
seconds**, wrapper **8.884430885314941 seconds**; the remaining failure
was the copied clock above. After that repair, the ownership, definition
copy, timeline, embedding, clarification and MTM group returned **33
passed in 83.34 seconds**, wrapper **85.56195592880249 seconds**. The
completion runs below supersede the interrupted runs. No stopping gate
has failed, and no commit or push has been made.

### 23.17 Reader and writer disposition on the completed design

This table completes the reader inventory of §21.3. An explicit address
still grants a bounded read; it supplies no content-match score. Content
recall uses the question's kept extension, combines each operand's result
over rows by its own operators, and reads an open answer only through the
matching generated operator path. The old role scorer is deleted.

| Thought face or other reader | Previous read | Present read and point-11 interpretation |
|---|---|---|
| `query`, `ask` | Recalled role match, sometimes the whole witness row as the answer. | Matching generated open operand; closed claims take pervasion or acquired exact-content trust. Lower-relevance existence is carried separately as analogy. |
| `isTrue` | Addressed pair or best role match. | An explicit occurrence reads its signed trust through `end_evidence`; other closed claims use the deduction read above. |
| `isPart` | Direct pair or priming's netted taxonomy shortcut; neighbors for an open end. | Direct LTM1 pair and bounded two-lane taxonomy paths; open ends still use neighbors. It takes relation proof, not LTM0 analogy. |
| `infer` | Equality/identity and membership matching, followed by premise/rule minima. | The equality read remains; membership uses the trusted relation path. Addressed premises read signed trust. Analogy never becomes a premise's stored trust. |
| `intuit` | `Intuition`'s geometric member coverage. | Unchanged, as required. It is not the new LTM0 existence read. |
| `exist` | Conceptual extent. | Unchanged; the face is distinct from `ThoughtLTMCapability.existence_evidence`. |
| `not` | Reversed intension/evidence, but left an attached extension positive. | Also wraps each existing root with the declared `NotLayer`; closed relation proof is oriented by the question's unary `not`/`non`. |
| `bind`, `isEqual`, `part`, `equal`, `implies`, `gain` | Reference proposals, direct equality/pair operations or gain. | Those operations remain; none takes LTM0 analogy as trust. Their addressed store reads use the reduced row. |
| `ThoughtLTMCapability._cued`, `retrieve`, `resolve_description`, `query`, `best_match` | Stimulated roles, fixed order meet, then role re-scoring. | Operator extension over the window; bounded acquired records; generated operand answers. |
| `existence_evidence`, `truth_evidence`, `_region_evidence`, `_deductive_evidence` | Fact-kind/role-matched support. | Signed-trust maxima, with exact content, pervasion and analogy kept separate. Only acquired witnesses contribute when the shared budget is exhausted. |
| `relation_evidence`, `ThoughtTaxonomyCapability.evidence` | Indexed direct relation plus old role candidate/priming shortcuts. | Indexed ends and existing bounded taxonomy capture. The unresolved-endpoint guard is retained at this read; both lanes and strongest paths survive. |
| `premise_evidence`, `end_evidence`, `ThoughtClosing.closing_question` | LTM0's pair, directly or through its row envelope. | `row['evidence']` is now the two-lane view of immutable signed `trust`; LTM1 still supplies its actual pair. Closing preserves the question's extension. |
| `TimelineRead.read` | Row pair and a stored event's selected content. | The same phase/content read, over separate signed occurrences where BOTH was written. Expired MTM does not reconstruct a timeline from a compressed idea. |
| `TruthLayer.sync_from_ltm` and its loss-factor readers | Fact-kind and origin-filtered rows. | Latest nonzero-trust rows, content times signed trust. Conversation rows contribute nothing. `consistency`, `suggest_clarifications`, `assess`, and the training factor remain readers of that view. |
| `WholePaths.fields`, `SentenceUnderstanding`, `OutputOccurrences.generated`, lesson observer | Leaf fields placed side by side. | The field left by `compose_extend` and the field of the generated idea, captured from the selected observation after restoration. |

Point 11's built pervasion is `ExtendedRecall.pervasion` →
`Taxonomy.capture_taxonomy` → `ConceptualTaxonomyView.part_of`. On the
same controlled question, with member edges **(.8,.4)** and **(.5,.9)**
and no kind-to-property-class relation, unchanged `Intuition.pervasion`
returns **(.5,.4000000059604645)**, while point 11 returns **(0,0)**.
The former's member geometry is not a proof of the latter's higher-order
coverage. No equivalence is claimed or forced.

The LTM0 pair's remaining indirect readers are `IdeaRows.row`, `corners`,
`raw_meaning_of`, `ClauseRows.read_reference_bindings`, `MemoryConsolidation._fold`,
`SentenceStrata.keep_close`, `corpus_lessons.measure`, and
`reasoning.TruthGroundedReasoner.existence_evidence`, as well as the query,
closing and timeline readers above. They now read signed trust or its
derived two-lane envelope. LTM0 pair-based kind/observation selection,
`retention_values`, presentation refresh/withdrawal and in-place pair
setters go nowhere under points 14, 20 and 21. Definition/member readers
in `DefinitionFold`, `FamiliarTier`, `PartClosing`, `ConceptAllocator`,
`Priming`, `Spaces.verify_relation` and `corpus_lessons.membership_report`
read **LTM1**'s pair; it is retained. Its effective endpoint-scaled read
remains for definitions and priming, without the deleted operator factor;
the proof capture reads the stored pair.

| Removed storage or mechanism | Its readers and writers | Destination / retirement |
|---|---|---|
| LTM0 `truth_plus`, `truth_minus`; `set_trust`, `set_evidence` | Append, row/envelope/corners, readers listed above; former update/withdrawal methods. | Immutable `trust`, two rows for BOTH (12a). No updater. Authority writers keep supplied values, training closes apply model trust times word certainty, conversations write zero, thoughts use premises. |
| Both stores' `access_count`, `surprise`; retention parameters and `RetentionColumns` | Append/add, repeated pair writes, `IndependentComponents.commit`, `OccurrenceRows.identity_code`, `IdentityRows.access_count`, `ReferenceClose.exclusion_pairs`, old forgetting scores, consolidation and presentation imports. | Write/access/pass activity (14). Live surprise initializes activity. Empirical exclusion counts all retained occurrences, per Alec's §23.9 answer; it has no stored counter. Presentation imports and the old retention formula retire. |
| `address_keys`, address index | `OccurrenceRows._address_for_write`, `stream_rows`; `EntryRead.rebuild_addresses`, recall; ClauseRow reference/index helpers; ClauseJournal closing; BindingAnswers; PartClosing; model sentence bank/truth admission; Queries binding/occurrence description; SentenceStrata capture/tree actions; ThoughtStrata/ThoughtStream writes; old forgetting/protection and sidecar validation. | `row_name`, typed occurrence and time lookup (15), with one increasing clock. Working STM retains its own local names. Deleted provenance/cover/expectation helpers have no replacement. |
| LTM0 `stream` | Clause writer, episodic reinstate, `OccurrenceRows.stream_rows`/`witness_stream`, reference accessible/arrival/commit/event/pending-name reads, SymbolExpectation chain/observation readers. | One durable history (16); recall enters the requesting batch. Live MTM, STM, discourse and policy batch axes remain. |
| LTM1 `sentence_reference` | `ClauseRows.part_row` type/readings, `RowTime.read_time`, SentenceStrata close, lesson observer; `LTM1.add`/load. | Relation ends remain in the pair (19); a relation has no separate sentence time. Its row-time read is empty, and STM keeps the live sentence context. |
| LTM1 `operator_reference`, `operation_seat`, `operation_catalog` and kind `operator` | LTM1 add/index/protection/retain/effective read/catalogue/load; ClauseRow predicate/readings; ClauseJournal; PartClosing; SentenceStrata; presentation snapshots. | No operator pair (17). A verb over a completed relation writes an idea. Structural operators stay in the grammar and its working tree. |
| `TrustGates`, `closedTrust`, operator amplitude scaling | Grammar menus, Models training/reporting, relation effective confidence, corpus readiness and their checkpoint/configuration helpers. | Nowhere (18): all file operators are available from sentence one. `PartsOfSpeech.operation` still measures definiteness; the lesson report retains that measurement and no gate state. |
| `_semantic_rows`, `metadata_required`, `semantic_fingerprint` | ClauseRow forward-address/slot indexes and resolution; EntryRead recall; ConceptForgetting; LTM0 append, row/raw/order/polarity, load/extras/compaction; SentenceStrata tree metadata/composition/actions; ThoughtStrata snapshots; language restore; old forgetting and covers. | Durable referent and entry (12/12b/22), pending bindings in MTM, working syntax in STM. Completed derivation, scope, fingerprint, thought producer and durable metadata validation retire (20–22). |
| `document_keys`, `sentence_content_keys`, `sentence_index`, `source_documents`, `source_authorities`, `_document_sources`, `_texts` | Occurrence address/content-key construction; Models supplied-truth match; ReferenceContext replacement exclusion; source/text accessors; semantic extras/load/compact/reset; presentation merge/withdrawal; recurrence grouping; old forgetting scopes. | Nowhere (20–21). Every submission appends; source text is generated when reported; no origin withdrawal or document-scoped read. The input/corpus's own document positions are not LTM columns. |
| `DocumentCover`, `_cover_symbols`, `cover_rng_state` | Cover open/read/segments/reference collection, ConceptForgetting protection, LTM kind and reserved-handle reads, snapshots/imports. | Nowhere (20). No cover concept or cover RNG state is saved or regenerated. |
| Recurrence table, `recurrenceAngle`, `recurrenceCapacity`, grouping | `MemoryConsolidation.recurrence_groups` and its update/fold, SymbolSubSpace construction, corpus capacity planning/search. | One-row activity promotion (14/20), which writes the new concept into `referent`; no recurrence grouping or table. |
| Estimate rows, `_expectation_rows`, `expectation_pair`, linking/writers | LTM append/validation/link lookup, forgetting trim/boundary, SymbolExpectation retained-chain reads, Models closing, presentation import, ThoughtStrata snapshot. | Nowhere (20). Live expectation and surprise still serve attention/learning; they are not durable forecast rows. |
| Origins, kinds, resubmission match, epoch presentation rollback | `set_origin`, `clear_origin`, `withdraw_origin`, `kind_of`, `KINDS`, `accept_fact`, `observation_ids`, Models admission/closing/addressable spaces, TruthLayer, reasoning existence, language restore, ThoughtStream commit, presentation epoch/base refresh. | Trust selects held claims (21). `PresentationMemory` is deleted. Reset explicitly requested by its caller remains a reset; an epoch, restore or repeated list withdraws no claim. |
| Durable binding sidecar | §21.9's order/reference/promotion, generation and identity readers; ClauseRow open-slot and forward-name readers. | Exact chosen reference and order → `referent`; extension → entry; pending question/name/forward state → MTM (22). MTM event witnesses supply promotion's form; without them it uses the current value (§23.13). |
| Symbol-space taxonomy, `attach_knowledge`, `symbolicPriming` | Language taxonomy construction, reset/decay/boost/report and snapshots; only tests attached knowledge. | Nowhere (23); conceptual `Taxonomy.py` stays. |
| Chain-row writer and `ChainRead.path`; five unused Spaces writers | MTM promotion called `MemoryPaths.write`; nothing in production read the chain; `reify_concept` alone called `relate`; the other writers had only test callers. | Nowhere (23). Production membership/Part writers remain; their tests use those writers. |

All table rows include construction, checkpoint, reset and compaction
where applicable; these operations now follow each store's column
declaration. Migration reads old names only to validate, move retained
references/pending state, or warn and discard. Alec's two rejection rules
stand: ambiguous/missing old times and old LTM0 BOTH rows are rejected
before loading any destination. An in-process trial restore is not that
migration. Discarded trials restore their activity, clocks and MTM; only
the selected read effects are applied.

### 23.18 Defaults, traces requested without a decision, and storage

The defaults built are listed here once, including those already accepted
in §22.3. None was changed to obtain a passing proof.

| Default | Built choice; evidence against it or limit |
|---|---|
| Operand activations (2) | Their kept places select their existing open fields. No restimulation. No contrary result found. |
| Inverse extension (3) | Each generated operand receives its part of the parent's set. The controlled inverse certificate passes; a learned multioperator round trip is not thereby proved. |
| `lower` (4) | Every member at `1/N`, with their centroid as key. No contrary result found. |
| `some` (4) | Same compose kernel as `universal`; strongest `max(2,N//2)` for `N>=3`, stable ties, otherwise empty. The impossibility of a proper plural subset for fewer than three is recorded, not hidden. |
| Conjunction and content operators (4) | Conjunction intersects; the listed content operators project their content operand; open `ask` selects nothing. No contrary certificate result found. |
| Compose determiners/reference proposals (6) | Retained. Their surviving exclusion/promotion reads needed Alec's all-retained-occurrences and actual-MTM-events decisions, now built. |
| Preposition head (6a) | Operand 1, its object; initialization forward/reverse certificate 64 passes. No contrary result found. |
| Open answer path (10) | No answer when the operator path differs. The native known-reference recall and controlled open-object path pass. Multiple open operands are combined, but their full native thought integration is not certified. |
| No census-based pervasion (11) | Members' separate statements do not establish a higher-order universal. The same-question difference from Intuition is reported in §23.17; Intuition was deliberately not changed. |
| Exact-content deduction and analogy (11) | Acquired trust at relevance one can support deduction; lower-relevance existence is separate analogy and supplies no stored trust. Budget-limited acquisition now preserves earlier exact evidence. |
| Negative form (11) | `not` swaps the lanes and `non` withholds them, including the higher-order proof and an attached thought extension. The complete learned negative-sentence path remains a limit. |
| Training idea trust (12a) | Model trust times word certainty, just as for a relation. This admits training rows to the truth view; gate results are reported beside §14 below, without tuning. |
| BOTH conclusion (12a) | Two signed rows, positive first. Old BOTH checkpoints cannot preserve two lanes in one signed row; Alec chose rejection, now atomic. |
| Name and promotion order (12b/14) | `referent`; promotion keeps an existing referent's order, otherwise one above the greatest entry order. No reason against the name was found. Equal extensions with different referents retain different orders. |
| Activity values (14) | Default `.5`, access step `.1`, decay `.9`, promotion `.9`. Kept read effects apply once; discarded trials change neither activity nor clocks. No tuning. |
| Increasing times (15) | `max(model present, next unused tick)`. Old ambiguous or untimed rows are rejected by Alec's decision; no migration invents times. |
| One-history admission (16) | Batch streams write in their stream order. Live batch ownership remains in STM/MTM. No contrary result found. |
| Truth view selection (21) | Latest `truthMaxEntries` nonzero-trust rows. It includes training statements and excludes conversation-only rows. |
| Serving pass (21) | Before a supplied list, pressure above `forgetHighWater` runs the activity pass. Repeated lists append, and restore withdraws nothing. Certificate 55 passes. |
| Contradiction wording (21) | Generated claim wording plus retained time, numeric signed trust, and “contradict.” An unattached view uses `truth #i`. No source text is retained; no automatic repair is made. |

Further implementation choices remain those in §21.4: separate roots
join by union; shipped algebra faces `pi`/`min` narrow and `sigma`/`max`
join; transpose changes operand placement rather than inventing a tree
node; a definite member carries both held MTM lanes; mandatory
interpretation transports memberships only along declared word/object
references. These are not claims about the later extension design of
§3.18 or the later numeric/image discussion.

The third amendment asked for four traces, without making a new policy:

- An unspecified `InterpretLayer.define` still proposes **(1,0)** and
  LTM1's duplicate-pair writer keeps the greater lane. The implicit
  callers are `lookup_word`, `forward`, `reading` and `admit_discovered`.
  They can reassert an already admitted named object and raise a counted
  definition. In the controlled trace, an effective **.3** binding was
  not admitted and remained **.3**; the admitted **.7** binding was raised
  to **1** by `forward`. Consequently later definition/index reads can
  depend on this existing behavior. Explicit `PartClosing.finish_definitions`,
  `PerceptCollections.observe` and `FamiliarTier.promote_identity` supply
  the pair they witnessed. No change to this policy was made.
- `ConceptTrust.positive_length` still returns **1** for an unallocated
  or untracked concept. `Priming.part_edges` and `LTM1.refresh_effective`
  take that value as the endpoint factor. Their downstream priming and
  admitted-definition reads can therefore take an untracked endpoint at
  unit strength. No new allocation or trust rule was introduced.
- Training rows now enter `TruthLayer.sync_from_ltm` and the existing
  factor on training loss. The stopping gates remain the tests named by
  Alec; no threshold, seed, fixture size or training factor was tuned to
  restore their §14 results.
- A conversation's zero-trust defining relation does not by itself
  admit a word's object. `define_pending` passes its **(0,0)** to promotion
  and the sole DEF writer; the definition index still requires its
  existing threshold. A direct trace of that final writer/read kept raw
  **(0,0)**, remembered the proposed object, returned **no active
  denotation**, and `forward` returned **None** without raising the pair.
  This does not prevent witnessing or an independent trusted premise
  from defining the word. `Models.infer` uses a temporary
  `_conversation_read` flag, restored in `finally`, to make both its idea
  and relation closings zero-trust; ordinary training and authority
  ingestion retain their separate writers.

The confirmed declarations are:

| Store | Saved row columns | Rebuildable, unsaved caches | Saved auxiliary state and use |
|---|---|---|---|
| LTM0 | `slots`, `episodic_concepts`, `episodic_values`, `trust`, `referent`, `activity` | None. Time lookup reads the sorted row times. | `count` occupied rows; `model_clock` next unused tick; `percept_layout` band layout; `truth_schema` signed-trust schema; `forget_passes`, `forget_sentences`, `forget_last_sentence`, `forget_pending` pass scheduling. |
| LTM1 | `refs`, `ref_kinds`, `truth_plus`, `truth_minus`, `activity` | `_pairs` exact pair; `_parts`, `_wholes`, `_references` adjacency/reference reads; `_index_version` and `_order_revision` invalidation; `relation_order` ordering; `effective_plus`, `effective_minus`, `_effective_source` endpoint-scaled definition/priming view. | `count` occupied pairs; `layout_version` checkpoint schema. |

The effective-pair caches in this table complete the earlier abbreviated
list in §23.6. They are not extra stored trust columns. The source pair
remains the authority read for a proof, and an operator's amplitude scales
neither it nor the effective view. In-process callback/owner bindings,
configuration, STM syntax and pending MTM state are not row caches.

Stored sizes remain **5,514 → 5,096 bytes per LTM0 row** and
**7,103,321,093 → 5,343,543,353 bytes** at the largest constructed capacity
of **1,048,576**, width **1256**, four entry slots, formerly six orders,
28 old streams and 262,144 old recurrence slots. The saving is
**1,759,777,740 bytes**. The new row is **5024 + 32 + 16 + 4 + 16 + 4**
bytes in column order, with **57** auxiliary bytes per store. LTM1 is
**30 bytes per row plus 16 auxiliary bytes**. These are stored tensor
payloads, excluding serialization framing, caches and Python state.
§23.6 explains why the old per-row total includes the 33 bytes omitted
from §14's per-row subtotal.

### 23.19 Integration after the first default sweep attempt

The first default sweep attempt used `env -u RUN_TRAINED -u RUN_SLOW
make test`, with nothing deselected. It selected **7,879 cases**. Its last
completed worker reports contained **5,686 passed, 250 skipped and 27
failed**; it was not a completed sweep. Both stipulated stopping gates
passed in that attempt. My diagnostic-copy command mistakenly included
an ignored, 4,597,846,207-byte checkpoint; the temporary copy exhausted
local disk space and the sweep supervisor could not finish its worker
receipt. I removed only that failed temporary copy, leaving the original
checkpoint untouched, and changed the diagnostic copy to source files.
This is an aborted run, not a reported sweep result. A complete default
sweep on the final source is still required.

**The field handed to the answer reader.** `WholePaths.finish` still
published reconstruction's generated field before thought, and numeric
`WholePaths.output` took the model's last published field. After extension,
that is a different set from the sentence's understanding. A captured
native supplied-answer path had understanding mass **8**, answer-field
mass **0**, and four costs of **.3499999940395355**. This was a producer
defect under point 7, not a lack of output-head learning. Reconstruction
keeps its own field; the thought suffix is given the sentence's
understanding, and the numeric reader takes
`observation['record'].understanding_field` directly. The selected
unsupervised close also publishes the selected understanding. An
output-only departure restores its existing prefix and does not publish
or reconstruct it again.

The new
`test_supplied_answer_training::test_output_reads_its_understanding_when_generation_publishes_a_different_field`
checks the ownership boundary with different understood and generated
fields. With the checkpoint's exact `WholePaths.output` loaded, it fails
on the zero generated field (**1 failed in 1.71 seconds**); on the repair
it passes in the thirty-case group below.
`test_whole_paths::test_output_departure_runs_only_the_last_stage`
keeps every assertion. The native gradient fixture required a separate
port: point 4's intersection of disjoint word concepts can legitimately
be empty, so the full Boolean grammar no longer promises a nonzero
reader-weight gradient on every path of this isolated gradient test. My
first port kept only the joining rule. The subsequent construction audit
found that removing rules also reduced the actual inventory from **44 to
37 rows**, despite unchanged XML capacity settings. That port is removed.
The final fixture keeps the original full grammar, **44 rows**, **488 code
coordinates**, four two-word sentences and the same four labels. It uses
the original four distinct words, each paired with itself, so narrowing
does not erase the word's extension. Both complete paths, every gradient
and before-update assertion stay, with exact ownership of the answer
field added. This also ports its caller
`test_operators_round2e::test_sentence_advantage_and_optimizer_use_the_same_reader`.
The caller's function and signature are unchanged. The final fixture and
caller group is **8 passed in 6.60 seconds**, **8.853049039840698 seconds**
including its wrapper. No seed is selected, and the symbolic XOR proof's
grammar and inputs are unchanged.

**A recent head could alias a retained time.** The native lesson first
failed its event-link containment guard after **64 field comparisons**.
The captured preview selected a contained event; commit read a different
point at its occurrence address. `SentenceStrata.keep_close` copied an
LTM row under `time-v1`, then named a relation's recent head with its
separate syntax counter in that same namespace. A later LTM row could
have that time. `MemberRead` found the recent row first and returned its
head instead of the retained occurrence. The deterministic native-store
certificate
`test_item18_reduction::test_recent_relation_head_cannot_alias_a_later_retained_time`
failed before: both references were **`('ltm', 'time-v1', 2)`**.

The recent-head producer now reserves its occurrence name from the same
row clock whenever it has a source store, including a relation close
that writes no LTM0 row. It writes no additional durable row or metadata.
Unattached working syntax retains its own namespace. The certificate
passes with distinct names and checks the value of every read. The
containment and kind/order guards are unchanged. The combined time-name,
whole-path, supplied-reader and shared-STM group is **30 passed, 1
warning in 22.13 seconds**, **24.416324138641357 seconds** including its
wrapper. That earlier group used the subsequently rejected 37-row
gradient fixture; the eight-case run above supersedes that part of it.

**The mathematical evaluation arms inherited one another's history.**
The unchanged report test passes on `ab5176e0e2bc`, but the new layout
exhausted the same **58-row** conceptual inventory on its sixteenth
identity promotion. No activity-based consolidation had minted a row in
that trace. `eval_math_thinking.main` built one model for all budgets and
both ablations. With point 21, reading the same problems again adds
occurrences, so the later arms trained on the earlier arms' readings.
The old resubmission match had hidden this dependence. For the same
native word, the checkpoint code retained one occurrence and replaced
its last mention time **1 → 3 → 5 → 7**. The new code retained distinct
occurrences at **2, 41 and 72**, with close times **1, 3 and 5**, and
promoted its object on the third. Fifteen such promotions left no seat
for the next one. The history, not a new capacity or activity threshold,
explains the difference.

Each budget/ablation now builds the same requested checkpoint and seed,
then releases that model after its arm. This repairs the comparison's
initial condition; it changes neither the model's promotion nor its
capacity behavior. The new
`test_eval_math_thinking::test_budget_and_ablation_comparisons_start_from_the_same_history`
failed before because later arms saw earlier histories, and passes
after. The original `test_eval_script_emits_every_report_column` retains
every assertion and its inputs, four rows, batch of two, budgets **1 and
4**, and seed **1**. Together they are **2 passed in 15.01 seconds**,
**17.13158917427063 seconds** including the wrapper. The old code's
report pass was **56.05 seconds** in the initial attribution run; the
additional passive history trace also passed, in **64.60597216599854
seconds** including its guard. The new code's corresponding passive
trace failed in **49.03540983298444 seconds** before the evaluator repair.

**Remaining sweep ports.** `test_definition_writer`'s shared chain
uses native `part` in place of the retired operator relation; its
sixteen cases retain their definition assertions. The captured MTM
receipt expects version **13**, and the invalid relation checkpoint
fixture matches the new column-shape error. The generation-lesson
fixture supplies a native understanding and program. The inverse-gradient
fixture supplies the two-leaf `surface` trace and root depth, and reads
word positions from `realization_field`, retaining every gradient
assertion. Episodic fixtures use the flat entry across all orders and
check its own concepts on direct decode. The original input's order and
multiplicity are no longer an entry contract (points 5 and 12); missing
surface and retired-concept assertions stay. The migration warning
replaces the retired persistent migration marker.

The two unindexed lexical-predicate tests now check the idea and its
trust, with the nested child pairs still unasserted (point 17), instead
of traversing an operator pair. The three `TestTaxonomy` functions of
`test_parthood_orders` are retired under point 23. The time-reader
inventory drops `PresentationMemory._capture_clock`,
`PresentationMemory.epoch` and `ClauseRows.write_clause`; its older
radix-name discrepancy is left for attribution. Five historical document
links to deleted modules/tests now point at the immutable checkpoint
commit, with their historical text unchanged. The complete names and
counts belong to the inventory below.

The first targeted port group was **40 passed, 7 failed in 22.35
seconds**. Its seven failures led to the fixture and handoff corrections
above. A combined follow-up exceeded its **8 GiB** process budget at
**8,712,330,584 bytes**, so it is not a pass. The split follow-ups were
**13 passed, 1 failed in 10.64 seconds** and **19 passed, 1 failed in
23.75 seconds**, exposing the empty-field gradient assumption and the
unnecessary output-only publication respectively. The final thirty-case
result above follows those repairs; none of these runs included or
failed a stipulated stopping gate.

### 23.20 Further integration and checkpoint attribution

The continuation of the disk-aborted sweep completed all **2,049** of
its selected remaining cases in **2,191.410027875012 seconds**. It ran
on a fixed source copy, not the final source, and is not the required
default sweep. Its phase reports contain **1,921 passed, 91 skipped,
39 failed and 1 XPASS**; phase reports are not distinct-case counts.
Sixteen failures were missing captured `test_whole_presence` data in my
source-only copy. The captured file exists in the main checkout. The
pre-existing `test_sparse_concept_e2e` failure is left alone. The verbal
proof failed its available-inverse assertion. The other failures led to
the following ports and one producer repair.

**A pending question lost its context point.**
`SituationFrame.from_row` chose a point by a working meaning's
`sentence_kind`. A pending MTM meaning can have no such kind, while the
retained row remains an idea with its own point. The reader then asked
for `head_point`, which that row does not have. After the document-cover
port, the native packed-end test exposed depth **0 instead of 1**.
The reader now takes the retained `point`, or the relation's
`head_point`. It does not reconstruct a retired row-kind column.

The new
`test_item18_reduction::test_pending_question_keeps_its_retained_context_point`
uses a pending open referent and an unclassified working meaning. With
the checkpoint's exact reader installed, it fails at depth zero:
**1 failed in 2.15 seconds**, **3.517698049545288 seconds** with its
wrapper. With the repair it passes, checking depth one, mask
`[True, False, False]`, and every coordinate of the retained point.
It also passes in the finished-source certificate group. The fixture's
row time is **1**, so the certificate isolates the pending-meaning read
from the separately retired meaning of integer zero.

**The remaining contract ports.**

- The three `test_sentence_boundary` writer cases observe the active
  stream and open sentence slot instead of `_write_address`. Their
  one-write and no-optimizer assertions stay.
- `test_sentence_compose` removes the document-cover slot from context
  length, reads an idea's `point`, and gives the disabled-prediction
  fixture a native understanding and extension program. All packed-end,
  gradient and unchanged-Adam assertions remain.
- `test_sentence_priming` declares the mock allocator's `raised` set;
  the **4,100** relations and constructed capacity are unchanged.
- `test_space_carrier_bridge` compares decoded structural checkpoint
  metadata recursively, including every tensor and value. The raw pickle
  lengths were **1,837 versus 1,645 bytes**, while the decoded
  `PartsOfSpeech` states were equal: rebinding changes string-object
  memoization. Serialized byte identity was not a state-equality
  contract. The other checkpoint assertions remain.
- `test_structural_checkpoint`'s shared fake owner now declares native
  MTM and its existing eight-coordinate codebook. Its allocator and verb
  metadata tests, and the unchanged consumer in
  `test_where_encoding_recovery`, keep their save/load assertions.
- `test_sweep_raise_repair` passes the occurrence's typed time reference,
  instead of a bare positive integer that now names a concept. Both
  native-order cases remain.
- `test_truth_criterion_gates_learning` supplies the sentence certainty
  **(1,0)** explicitly beside its four source-trust values. Point 12a
  multiplies those two different quantities; the registration and
  readback assertions are unchanged.
- `test_tetralemma_leaf` checks both signed rows for a BOTH conclusion
  and aggregates their independent lanes. The writer audit follows
  `IdeaRows.append_meaning`. Its AST guard distinguishes list
  concatenation from arithmetic netting, with a new check that accepts
  two lists of rows and still rejects subtraction of the two lanes.
- `test_unresolved_fixed_operand` supplies its native joined extension,
  actual fixture store and immutable signed trust. The original fixed
  reference and wrong-reference assertions stay. The final four cases
  pass in **3.29 seconds**, **4.09561395645 seconds** including the
  wrapper.
- `test_word_interpretation` checks that each heard word's symbol is
  nonzero in the open input field. The operator-combined extension may
  legitimately be empty. Admission, finiteness and agreement between
  the published field and its decoding still hold in all three modes.

The focused groups exposed two mistakes in my initial ports: comparing
strings with a tensor-only comparison helper, and omitting the query
fixture's owner store. Both were corrected in the fixtures. The final
certificate group, compiler-cache follow-up, and default sweep below
are the authority for the returned source, not those intermediate
failed groups.

**The verbal proof is not yet certified.** The unchanged checkpoint
code at `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd` passed the native
400-epoch verbal proof: **4/4 grammar reconstructions and 4/4 field
reconstructions**, **1 passed in 702.88 seconds**. The newer measured
source failed the grammar inverse while its independent field decoder
returned **4/4**. A passive replay of its saved failed model emitted
two terminals per sentence without truncation, but no surface bytes.
The terminals' form coordinates matched the selected lexical forms
exactly. Their meaning stimulated only the named order-one objects on
the against lane, with no order-zero percept demand; `FieldReverse`
therefore had nothing to spell. This locates the failing read, but does
not yet establish why training reached it. The checkpoint also reads
the input words on the against lane, so that sign alone is not an
attribution. No seed, training bar or capacity was changed. The required
finished-source rerun is reported below when completed.

### 23.21 Named test ports and retirements

Against checkpoint commit `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`, the
function inventory is **404 existing test functions edited in place**,
**59 renamed or moved ports**, **285 whole test functions retired**, and
**60 added test functions**. Thus there are **463 direct function ports**.
These are function counts, not parameterized-case counts. A shared fixture
port does not count every unchanged consumer as another edited function.
The checkpoint CPU collection contains **333 cases** belonging to the
285 retired functions; the 60 additions supply **86 cases** in the final
collection. New grammar/storage/trust certificates account
for most additions. The separate final sweep count includes parameterized
cases, skips and document-link checks.

There are also **four retired parameter cases inside one surviving
function**, so **337 old cases** retire in total: the 333 belonging to
whole retired functions, plus these four. In
`test/test_forgetting.py::test_invalid_configuration_fails_before_a_pass`,
`[kwargs1]` (`half_life=0`), `[kwargs2]` (`age_weight=-1`), `[kwargs3]`
(`similarity=nan`) and `[kwargs4]` (`protect_origins=(3,)`) retire because
point 14 removes those retention parameters and point 21 removes origin
protection. `[kwargs0]` (`low_water=.9`) and its rejection assertion stay.
This is a partial contract retirement, not four more whole functions or
a resized fixture.

The following tables supersede the partial inventory in §23.5. Names
after a filename are its exact pytest function or `Class::function`
names. A port may retire only an assertion about a removed contract;
it is not counted as a whole-test retirement. Mechanical writer ports
remove origin/document/stream/kind arguments, carry typed time references,
use flat entries and immutable trust, and substitute working STM0 for a
time-less fixture (points 12–22). Higher-order fixtures use production
collection and Part writers in place of point 23’s unused conveniences.
The reader tables above and the explicit reasons below identify the
semantic changes; the remaining assertions are kept.

**Renamed or moved ports (59).**

| Before | After | Decision / retained contract |
|---|---|---|
| `test/test_argument_concepts.py::test_empty_argument_views_supply_no_operation_authority_or_gate_readiness` | `test/test_argument_concepts.py::test_empty_argument_views_supply_no_operation_authority_or_definiteness` | The same definiteness values are read directly from PartsOfSpeech.operation. Removed only the operator-seat factor: the source evidence and endpoint read remain. |
| `test/test_argument_concepts.py::test_revised_form_only_operation_waits_for_seat_authority` | `test/test_argument_concepts.py::test_revised_form_only_operation_reports_unmeasurable_definiteness` | The same definiteness values are read directly from PartsOfSpeech.operation. Removed only the operator-seat factor: the source evidence and endpoint read remain. |
| `test/test_assertion_provenance.py::test_truthset_asserts_only_outer_ends_including_deduplicated_roots` | `test/test_assertion_provenance.py::test_truthset_asserts_only_outer_idea_and_reasserts_declared_pairs` | Use the native verb tree; parent is immutable .7 idea, the separately asserted part is .8. Operator-edge dependence and root dedup retire. |
| `test/test_clarifications.py::test_suggest_clarifications_missing_source_fallback` | `test/test_clarifications.py::test_suggest_clarifications_without_retained_source_uses_indices` | Signed stored trust is always numeric; retire invented None trust/source-text fallback and keep stable index fallback and contradiction signal. |
| `test/test_clause_acceptance.py::test_05_embedded_part_is_unasserted_and_propagates_operator_kind` | `test/test_clause_acceptance.py::test_05_verb_over_completed_part_closes_an_idea_without_operator_edges` | Completed relative content under a verb is an idea; no operator edge or nested assertion is emitted. Keep content, outer trust, standalone signed relation and LTM1 max-pair assertions. |
| `test/test_clause_acceptance.py::test_pointless_predicate_retains_the_relative_child_as_an_occupied_role` | `test/test_clause_acceptance.py::test_completed_relative_predicate_is_fused_into_its_idea` | Completed relative content under a verb is an idea; no operator edge or nested assertion is emitted. Keep content, outer trust, standalone signed relation and LTM1 max-pair assertions. |
| `test/test_clause_storage.py::test_query_returns_end_clause_content_and_preserves_both_poles` | `test/test_clause_storage.py::test_addressed_end_content_preserves_each_signed_row` | Removed role-match retrieval; preserved addressed content and both lanes, now on two immutable rows. |
| `test/test_episodic_concepts.py::test_basic_concept_readings_keep_largest_signed_values_in_reading_order` | `test/test_episodic_concepts.py::test_all_orders_keep_largest_signed_values_in_one_flat_entry` | The entry is flat and selects every order; order-axis and order-zero-only expectations retire. The same four concepts and two readings retain top-k, sign, and detached-gradient checks. |
| `test/test_episodic_concepts.py::test_recalled_concepts_decode_directly_and_withdrawn_ids_are_counted` | `test/test_episodic_concepts.py::test_recalled_flat_extension_decodes_directly_and_retired_ids_are_counted` | A stored extension has no original word positions or multiplicities; full input replay retires. The native grammar-free decode is checked against the actual stored concepts; missing surfaces and retired concept counts remain. |
| `test/test_existence_evidence.py::test_exist_preserves_conflicting_fact_support` | `test/test_existence_evidence.py::test_addressed_claims_preserve_conflicting_support` | Both independent stored degrees are read by exact reference; the retired vector-region match is removed. Public BOTH contract remains in certificate 40. |
| `test/test_existence_ingestion.py::test_explicit_fact_admission_preserves_full_description_and_source` | `test/test_existence_ingestion.py::test_explicit_fact_admission_preserves_full_description_and_trust` | Keep full stored idea and -.8 trust read; source text, scope and role-match retire. |
| `test/test_existence_metadata.py::test_actual_model_checkpoint_preserves_scoped_fact_identity` | `test/test_existence_metadata.py::test_actual_model_checkpoint_preserves_signed_claim_identity` | Preserve signed values, explicit occurrence identity, compaction and checkpoint assertions. Retire stored role-match, scope, origin/text and withdrawal assertions; extension maxima certified by 40. |
| `test/test_existence_metadata.py::test_checked_reader_keeps_both_sides_of_exist_evidence` | `test/test_existence_metadata.py::test_checked_reader_keeps_both_sides_of_addressed_evidence` | Preserve signed values, explicit occurrence identity, compaction and checkpoint assertions. Retire stored role-match, scope, origin/text and withdrawal assertions; extension maxima certified by 40. |
| `test/test_existence_metadata.py::test_conflict_degrees_and_stable_provenance_survive_compaction_and_reset` | `test/test_existence_metadata.py::test_conflict_degrees_and_occurrences_survive_compaction_and_reset` | Preserve signed values, explicit occurrence identity, compaction and checkpoint assertions. Retire stored role-match, scope, origin/text and withdrawal assertions; extension maxima certified by 40. |
| `test/test_existence_metadata.py::test_repeated_evidence_does_not_inflate_partial_support` | `test/test_existence_metadata.py::test_repeated_evidence_does_not_mutate_stored_support` | Preserve signed values, explicit occurrence identity, compaction and checkpoint assertions. Retire stored role-match, scope, origin/text and withdrawal assertions; extension maxima certified by 40. |
| `test/test_existence_metadata.py::test_store_checkpoint_keeps_scoped_evidence_and_rejects_missing_sidecar` | `test/test_existence_metadata.py::test_store_checkpoint_keeps_signed_evidence_without_a_sidecar` | Preserve signed values, explicit occurrence identity, compaction and checkpoint assertions. Retire stored role-match, scope, origin/text and withdrawal assertions; extension maxima certified by 40. |
| `test/test_existence_provenance.py::test_checked_result_retains_matching_fact_occurrences_and_both_degrees` | `test/test_existence_provenance.py::test_checked_result_retains_addressed_occurrences_and_both_degrees` | Preserve witness and bilateral degree assertions via exact claim references; stored role-match retired. |
| `test/test_existence_public_inputs.py::test_trust_update_rejects_nonfinite_degrees_without_mutating_evidence` | `test/test_existence_public_inputs.py::test_new_trust_rejects_nonfinite_degrees_without_mutating_evidence` | No trust update exists; invalid new writes preserve the prior .4 row. |
| `test/test_global_attention.py::test_ltm_space_appears_when_store_present_but_excludes_estimates` | `test/test_global_attention.py::test_ltm_space_appears_when_store_present` | Named port; only the removed match, estimate-exclusion or incoming-trust-scaling assertion retires, as specified. |
| `test/test_grammatical_query_vps.py::test_unary_query_keeps_full_description_through_an_existing_occurrence` | `test/test_grammatical_query_vps.py::test_unary_end_read_keeps_full_description_through_an_existing_occurrence` | All complete-description, trust, reference, inline-construction and non-writing assertions survive via addressed isTrue. bind_constituents (17.17/22) is replaced by the ordinary explicit occurrence write. Content-only asks with no entry no longer return the addressed row (17.10–11). |
| `test/test_item18_finish.py::test_cache_rebuild_preserves_forward_slots_cues_episodes_and_retention` | `test/test_item18_finish.py::test_cache_rebuild_preserves_mtm_forward_slots_and_entry_reads` | Points 12/14–22; see §23.5 for this entry, order or cache port. |
| `test/test_item18_finish.py::test_episode_selection_preserves_both_lanes_at_every_order` | `test/test_item18_finish.py::test_flat_episode_selection_keeps_the_strongest_lanes_across_orders` | Points 12/14–22; see §23.5 for this entry, order or cache port. |
| `test/test_item18_finish.py::test_old_episode_shape_loads_at_order_zero_with_warning` | `test/test_item18_finish.py::test_old_ordered_entry_loads_its_strongest_readings_with_warning` | Points 12/14–22; see §23.5 for this entry, order or cache port. |
| `test/test_item18_finish.py::test_recall_keeps_each_order_and_intersects_without_carrying` | `test/test_item18_finish.py::test_recall_uses_each_concepts_own_order_without_carrying` | Points 12/14–22; see §23.5 for this entry, order or cache port. |
| `test/test_item20_round5_memory.py::test_epoch_withdrawal_expires_shared_short_term_references` | `test/test_item20_round5_memory.py::test_hard_reset_expires_shared_short_term_references` | Epoch withdrawal is retired; the same transient expiry assertions use ordinary STM hard clear and durable relations stay. |
| `test/test_item20_round5_memory.py::test_operator_authority_and_legacy_columns_use_declared_seats` | `test/test_item20_round5_memory.py::test_operator_choice_does_not_scale_relation_endpoint_confidence` | Keep promotion membership, native types/orders, trial rollback, gradient/expiry and native seats. Read membership from LTM1; retire successor paths, epoch withdrawal, relation sentence-reference bands, operation factor and semantic sidecar import. Working fixture names replace occurrence addresses without resizing. |
| `test/test_item20_round6_repair.py::test_kept_thought_finishes_definitions_before_relation_withdrawal` | `test/test_item20_round6_repair.py::test_kept_thought_finishes_definitions_before_relation_forgetting` | Use native occurrence IDs. Preserve definition completion and missing-relation guards after ordinary forgetting; epoch withdrawal is retired. |
| `test/test_item20_round7_episodes.py::test_content_query_reinstates_own_stream_and_next_open_attention_replaces_it` | `test/test_item20_round7_episodes.py::test_content_query_reinstates_in_requesting_batch_and_next_attention_replaces_it` | Flat strongest per-concept lanes replace per-order/occurrence lists. Repetition appends while preserving the prior entry. Migrate empty entries from the supported checkpoint layout 16; pre-checkpoint layout 14 path is retired. Reinstatement is into the requesting batch from one history, using the kept field of understanding and write-time observed row ownership. |
| `test/test_item20_round7_episodes.py::test_episode_columns_compact_with_rows_keep_first_entry_and_import_empty_legacy` | `test/test_item20_round7_episodes.py::test_episode_columns_compact_keep_prior_entries_and_load_empty_entries` | Flat strongest per-concept lanes replace per-order/occurrence lists. Repetition appends while preserving the prior entry. Migrate empty entries from the supported checkpoint layout 16; pre-checkpoint layout 14 path is retired. Reinstatement is into the requesting batch from one history, using the kept field of understanding and write-time observed row ownership. |
| `test/test_item20_round7_episodes.py::test_selection_keeps_signed_readings_in_order_and_zero_stores_nothing` | `test/test_item20_round7_episodes.py::test_selection_keeps_strongest_signed_lanes_and_zero_stores_nothing` | Flat strongest per-concept lanes replace per-order/occurrence lists. Repetition appends while preserving the prior entry. Migrate empty entries from the supported checkpoint layout 16; pre-checkpoint layout 14 path is retired. Reinstatement is into the requesting batch from one history, using the kept field of understanding and write-time observed row ownership. |
| `test/test_item25_completion.py::test_ltm0_amended_row_reads_referents_and_rejects_old_checkpoints` | `test/test_item25_completion.py::test_ltm0_amended_row_reads_referents_and_discards_retired_columns` | Keep native order, identities, slot/time alias, fixed allocation and restore checks; two rows keep the pair and legacy columns discard with warning. |
| `test/test_item25_repair.py::test_small_cued_window_preserves_match_order_filters_and_explicit_references` | `test/test_item25_repair.py::test_small_cued_window_reads_explicit_references_without_role_matching` | Explicit addresses authorize reads, not role resemblance or kind/origin filters; preserve bounded reads, returned rows, native allocation and all non-activity state. |
| `test/test_item55_mechanisms.py::test_stream_chain` | `test/test_item55_mechanisms.py::test_stream_code_banks_keep_their_independent_batch_axes` | Per-stream history, source re-witnessing and replacement retire; independent code-bank batching assertions preserved in test_stream_code_banks_keep_their_independent_batch_axes. |
| `test/test_layout_removal.py::test_pending_sentence_is_a_tagged_chain_with_only_its_top_stamped` | `test/test_layout_removal.py::test_declared_nested_part_chain_preserves_typed_occurrences` | Explicit semantic Part chains replace removed implicit operator chains in shared read fixture. Same widths/capacities; time contracts use durable LTM0, references use typed times. Invalid-value checks now exercise activity; sidecar tests retire with its column. An addressed read alone has no role-match score. |
| `test/test_ltm_consolidation.py::TestRuntimeUserIngestion::test_resubmit_replaces_user_rows_only` | `test/test_ltm_consolidation.py::TestRuntimeUserIngestion::test_resubmit_appends_rows_without_replacing_prior_truths` | §17.12a,15–16,18,20–21: trust values and reads retained; sources identified at writes, no stored origins; restore preserves rows; repeated supply appends |
| `test/test_ltm_consolidation.py::TestRuntimeUserIngestion::test_store_truths_after_load_does_not_reprovision` | `test/test_ltm_consolidation.py::TestRuntimeUserIngestion::test_store_truths_after_load_keeps_all_existing_truths` | §17.12a,15–16,18,20–21: trust values and reads retained; sources identified at writes, no stored origins; restore preserves rows; repeated supply appends |
| `test/test_ltm_consolidation.py::TestStatelessRevive::test_stateless_load_prunes_only_after_structural_owners_are_restored` | `test/test_ltm_consolidation.py::TestStatelessRevive::test_stateless_load_retains_truth_after_structural_owners_are_restored` | §17.12a,15–16,18,20–21: trust values and reads retained; sources identified at writes, no stored origins; restore preserves rows; repeated supply appends |
| `test/test_ltm_consolidation.py::TestTruthLayerLtmView::test_sync_materializes_user_and_provisioned_rows_only` | `test/test_ltm_consolidation.py::TestTruthLayerLtmView::test_sync_materializes_nonzero_rows_only` | §17.12a,15–16,18,20–21: trust values and reads retained; sources identified at writes, no stored origins; restore preserves rows; repeated supply appends |
| `test/test_mtm_referents.py::test_word_reads_its_single_remembered_object_after_epoch_boundary` | `test/test_mtm_referents.py::test_word_reads_its_single_remembered_object_after_definition_forgetting` | A chart with no percept time band is STM, at unchanged capacity and relation capacity. Forgotten definitions retain the remembered object; automatic epoch withdrawal is retired. |
| `test/test_nested_retention_owners.py::test_full_checkpoint_removes_unreferenced_request_scoped_content` | `test/test_nested_retention_owners.py::test_full_checkpoint_keeps_unreferenced_trusted_content` | No resubmission/restore withdraws a row; keep identity/credit/checkpoint assertions and read original .75 trust. |
| `test/test_nested_retention_owners.py::test_full_checkpoint_restores_occurrence_owners_before_stateless_withdrawal` | `test/test_nested_retention_owners.py::test_full_checkpoint_restores_occurrence_owners_and_trust` | No resubmission/restore withdraws a row; keep identity/credit/checkpoint assertions and read original .75 trust. |
| `test/test_nested_retention_owners.py::test_tensor_only_restore_defers_pruning_but_withdraws_authority` | `test/test_nested_retention_owners.py::test_tensor_only_restore_keeps_referents_and_authority` | No resubmission/restore withdraws a row; keep identity/credit/checkpoint assertions and read original .75 trust. |
| `test/test_presentation_memory.py::test_boundary_clears_experience_and_occurrence_caches_preserving_learning` | `test/test_presentation_memory.py::test_boundary_preserves_retained_rows_and_learning` | §17.20–21 retires epoch snapshots, origins and withdrawal |
| `test/test_presentation_memory.py::test_boundary_refuses_to_erase_a_live_sentence` | `test/test_presentation_memory.py::test_boundary_preserves_a_live_sentence` | §17.20–21 retires epoch snapshots, origins and withdrawal |
| `test/test_query_contract_boundaries.py::test_query_limits_native_content_reads_and_reports_incompleteness` | `test/test_query_contract_boundaries.py::test_explicit_addresses_bound_decoding_without_establishing_content_match` | Named port; only the removed match, estimate-exclusion or incoming-trust-scaling assertion retires, as specified. |
| `test/test_query_contract_boundaries.py::test_unified_query_returns_complete_record_and_does_not_admit_observation` | `test/test_query_contract_boundaries.py::test_addressed_end_read_returns_complete_content_without_another_occurrence` | Complete addressed content, truth and non-writing assertions use isTrue/end_evidence. The old query retrieval envelope and address-as-content match are retired by 17.10–11. |
| `test/test_query_work_budget.py::test_fact_reads_stop_at_shared_budget_and_preserve_partial_evidence` | `test/test_query_work_budget.py::test_addressed_truth_reads_share_one_budget_without_renewal` | Points 7-11 remove role-match scoring for two rows without entries. Retain the original 1024/8192 fixture, two signed truths, read order/count and spent-work assertions via their explicit end_evidence read; after exhaustion no extra row is read and the prior evidence survives. |
| `test/test_reading_round.py::test_documents_retain_memory_and_mtm_until_epoch` | `test/test_reading_round.py::test_documents_and_epochs_retain_history_and_live_mtm` | Remove the retired per-document boundary and evidence column; keep the lifetime and pending-reference assertions for native checks. |
| `test/test_reference_arrival_review.py::test_arrival_readers_keep_unscoped_rows_available_to_a_new_stream` | `test/test_reference_arrival_review.py::test_arrival_readers_keep_single_history_rows_available_to_a_new_stream` | Point 16: no row stream column; retain every assertion about arrival and immutable source on a new batch stream. |
| `test/test_reference_chain_mention.py::test_unresolved_root_description_keeps_chain_and_own_occurrence` | `test/test_reference_chain_mention.py::test_root_description_keeps_its_idea_and_own_occurrence_without_a_chain` | Points 17,20,22: union leaves one idea, not a stored operator pair and a separate zero-trust mention. Keep its full point, root occurrence, linked MTM value and undefined word assertions; point 12a preserves both trust lanes in two rows. |
| `test/test_relation_references.py::test_nested_relation_forest_keeps_its_unindexed_predicate` | `test/test_relation_references.py::test_nested_relation_forest_closes_as_an_idea_without_asserting_children` | A predicate over a relation forest closes as an idea; implicit operator chains retire. Child pairs remain unasserted and typed-reference checks remain. |
| `test/test_relation_references.py::test_unindexed_lexical_predicate_over_a_relation_has_an_ltm_reference` | `test/test_relation_references.py::test_unindexed_predicate_over_a_relation_closes_as_an_idea` | A lexical predicate over a completed relation closes as an idea. Operator-pair traversal assertions retire; the idea, trust, and typed references are checked. |
| `test/test_replacement_kernel_contracts.py::test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees` | `test/test_replacement_kernel_contracts.py::test_complete_fact_end_reads_preserve_conflicting_and_mixed_degrees` | Mixed evidence and all posture/degree assertions survive by reading the two signed occurrences. The old one-row two-lane storage and address-as-content lookup are retired by 17.12a and 17.10–11. |
| `test/test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_idempotent_clear_then_record` | `test/test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_repeated_lists_append_immutable_claims` | Named port; only the removed match, estimate-exclusion or incoming-trust-scaling assertion retires, as specified. |
| `test/test_sentence_retention.py::test_idea_closing_keeps_estimate_surprise_and_index_in_the_existing_owner` | `test/test_sentence_retention.py::test_idea_closing_initializes_activity_from_live_surprise` | Live surprise initializes activity; keep observed content and actual row identity, remove historical forecast. |
| `test/test_step2b_storage.py::test_stored_question_is_refused_by_kind_without_a_mode_column` | `test/test_step2b_storage.py::test_zero_trust_is_kept_without_a_mode_or_kind_column` | Keep zero-trust persistence and absence of modes; kind-based acceptance and in-place authority changes retire. |
| `test/test_thought_operation_catalog.py::test_owner_built_structural_context_freezes_only_owned_stream_and_priming` | `test/test_thought_operation_catalog.py::test_owner_built_structural_context_freezes_only_owned_stream` | Point 23 removes the symbol taxonomy priming read. All stream ownership, isolation and no-target-generation assertions stay. |
| `test/test_trust_gates.py::test_readiness_records_each_operation_without_a_lesson_run` | `test/test_operation_measurements.py::test_readiness_records_each_operation_without_a_lesson_run` | Moved to test_operation_measurements; keeps trust_measurements equality and removes only the retired gate-state report assertion. |
| `test/test_truth_ideas_routing.py::TestTrustStamp::test_config_trust_stamps_cs_and_scales_incoming` | `test/test_truth_ideas_routing.py::TestTrustStamp::test_config_trust_stamps_cs_training_multiplier` | Named port; only the removed match, estimate-exclusion or incoming-trust-scaling assertion retires, as specified. |

**Ports retaining their function names (404).**

Each row gives the relevant points of §17 for that file. The source and
live-word, checked-answer, identity, gradient and checkpoint assertions
that survive those decisions remain in the named functions.

| File | Functions | Decisions |
|---|---|---|
| `test/test_accessible_mind.py` | `test_cue_fan_is_charged_and_scope_filters_before_ranking`, `test_higher_order_missing_edge_keeps_content_with_zero_evidence`, `test_higher_order_retrieval_seeds_discontinuous_members_only`, `test_nested_writes_are_shared_across_batch_positions`, `test_query_retrieves_an_addressed_old_frame_without_an_implicit_write`, `test_thought_retrieval_keeps_history_and_restores_episode_knowing` | 17.20–21; 17.23; 17.7–10,16,20–22 |
| `test/test_argument_concepts.py` | `test_revised_membership_proof_names_formal_class`, `test_revised_separability_evidence_and_contiguous_classes` | 17.11; 17.18 |
| `test/test_arithmetic_isolation.py` | `test_renamed_native_vocabulary_preserves_checked_relation_answers` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_assertion_provenance.py` | `test_external_assertion_context_cannot_declare_a_kind`, `test_provisioning_retains_completed_fields_and_shared_addresses` | 17.20–21 |
| `test/test_bind_thought.py` | `test_bind_keeps_self_and_containing_identities_without_rewriting_occurrences`, `test_pending_subject_keeps_its_composed_predicate_when_stored` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_candidate_table.py` | `test_candidate_from_closed_chain_retains_its_declared_hole_at_the_part_writer` | 17.12,12a,12b,14–22; 17.12a,14–22 |
| `test/test_case_views.py` | `test_phase_question_uses_selected_content_and_never_the_event_clock` | 17.12a |
| `test/test_chunk_journal_origins.py` | `test_existing_identity_needs_no_introducing_operation`, `test_shift_alias_records_real_operation_and_closing_writes_it` | 17.12b,22; 17.20–21 |
| `test/test_clarifications.py` | `test_suggest_clarifications_template` | 17.21 |
| `test/test_clause_acceptance.py` | `test_07_equality_is_one_membership_in_a_singleton`, `test_09_attribution_cannot_assert_its_child_and_negation_is_content` | 17.12a,15–22 |
| `test/test_clause_grammar.py` | `test_complete_grammar_declares_sentence_predicate_generic_and_implication` | 17.17 |
| `test/test_clause_scope.py` | `test_committed_close_keeps_recent_chunks_beside_the_output_field` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_clause_storage.py` | `test_attribution_keeps_inner_claim_unasserted_then_direct_assertion_updates_it`, `test_both_evidence_poles_survive_checkpoint_separately_from_neither`, `test_clause_end_state_and_evidence_checkpoint_roundtrip`, `test_compound_idea_stores_only_fused_point_without_factored_target`, `test_end_row_has_shared_references_and_field_coordinates`, `test_full_store_can_update_an_existing_relation`, `test_luminosity_preserves_both_instead_of_erasing_it_to_unknown`, `test_recursive_absolute_clauses_compress_to_one_sentence_idea`, `test_relation_has_no_kind_column_or_catch_all`, `test_relation_rows_never_enter_luminosity` | 17.12a,15,17,19–22; 17.20–21 |
| `test/test_concept_forgetting.py` | `test_live_reference_preserves_weak_candidate_and_keeps_a_representative`, `test_unused_mints_are_reclaimed_by_magnitude_without_erasing_defined_or_referenced_concepts` | 17.20–21 |
| `test/test_corpus_checkpoint.py` | `test_checkpoint_uses_consumed_positions_and_resumes_without_epoch_clear` | 17.20–21 |
| `test/test_cs_symbol_table.py` | `test_counts_alone_do_not_request_refinement`, `test_identity_at_one_part_one_whole`, `test_prune_drops_constituents_of_linked_raised_symbol`, `test_refinement_keeps_definitions_without_a_both_reading`, `test_reify_relation_between_symbols`, `test_resolve_identities_zeros_out_1to1`, `test_retire_transient_symbol`, `test_singleton_populates_one_untyped_edge` | 17.23 |
| `test/test_definition_integrity.py` | `test_new_definitions_do_not_alias_later_sentence_identities` | 17.12a,15–16,20–22 |
| `test/test_definition_rows.py` | `test_27_definition_rewitness_keeps_its_when_and_store_position` | 17.12a,15–16,20–22; 17.14,19 |
| `test/test_definition_taxonomy.py` | `test_selected_anaphor_form_binds_to_its_own_end_particular` | 17.12a,15–16,20–22 |
| `test/test_derived_concept_codes.py` | `test_actual_serial_leaf_pairs_forms_and_meanings_and_perception_owns_forms`, `test_predicted_rows_do_not_cross_streams_or_read_placement_conduction` | 17.20 |
| `test/test_document_covers.py` | `test_higher_order_fold_is_evidence_weighted_and_preserves_rows` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_dual_towers.py` | `test_pyramid_grads_reach_every_rung` | 17.23 |
| `test/test_episodic_concepts.py` | `test_legacy_percept_entry_is_dropped_and_named` | 17.12, 17.20 |
| `test/test_epoch_history.py` | `test_ambiguous_named_promotion_fails_before_minting`, `test_later_candidate_reads_a_promoted_classes_order_after_epoch_withdrawal`, `test_native_train_keeps_last_grounding_and_recognizes_it_across_ten_epochs`, `test_relative_member_write_expires_with_its_epoch` | 17.20–21; 17.20–22 |
| `test/test_epoch_start.py` | `test_other_order_active_definition_does_not_hide_remembered_identity`, `test_provisioned_symbol_relation_keeps_its_own_evidence_across_epochs`, `test_unstarted_checkpoint_preserves_setup_until_its_first_epoch_boundary` | 17.20–21; 17.20–21 retires epoch snapshots, origins and withdrawal |
| `test/test_existence_ingestion.py` | `test_part_without_supplied_evidence_stays_unverified` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_existence_metadata.py` | `test_actual_observation_writers_keep_roles_without_asserting_world_facts`, `test_live_description_owns_values_but_fact_storage_detaches` | 17.12a,21; 17.20–21 |
| `test/test_existence_provenance.py` | `test_consolidated_prediction_reads_the_one_idea_presence` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_expectation_defaults.py` | `test_expectation_off_keeps_every_packed_closing_in_ltm`, `test_native_runtime_reports_pairs_without_accumulating_or_updating` | 17.16,20–21; 17.20–21 |
| `test/test_familiar_tier.py` | `test_candidate_zero_survives_the_kept_bind_journal_without_an_occurrence_alias`, `test_defining_equality_promotes_a_pending_subject_without_an_order_or_choice`, `test_identity_promotion_at_capacity_preserves_the_open_token_and_rows`, `test_mixing_keeps_lexical_witnesses_across_ten_presentations`, `test_named_promotion_counts_kept_words_independently_of_bind_aliases`, `test_promoting_an_identity_does_not_reinitialize_unrelated_classes`, `test_symbol_and_promoted_occurrence_members_have_the_same_three_initializers`, `test_transient_occurrence_is_deferred_and_covered_when_its_symbol_arrives` | 17.20–21 |
| `test/test_forgetting.py` | `test_compaction_preserves_protected_row_columns_and_optional_metadata`, `test_conflicting_proof_does_not_change_another_rows_utility`, `test_epoch_trigger_and_disabled_setting`, `test_idea_deletion_releases_its_identity_next_pass`, `test_ideas_forget_lower_surprise_at_equal_trust_and_age`, `test_invalid_configuration_fails_before_a_pass`, `test_packed_and_single_presentations_keep_the_same_stream_history`, `test_partial_checkpoint_cannot_infer_missing_trust_or_forgetting_state`, `test_protected_and_live_roots_survive_and_block_unsafe_target`, `test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads`, `test_taxonomy_and_definitions_share_part_retention_not_ltm1_row_ownership`, `test_transitive_row_has_its_own_utility`, `test_trigger_interval_and_age_select_oldest_without_moving_addresses`, `test_wording_is_one_stage_and_checkpoint_preserves_the_result` | 17.12a, 17.14–16, 17.20–21 |
| `test/test_generate_repair.py` | `test_promotion_rejects_missing_witness_at_creation` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_generation_lesson.py` | `test_sentence_generation_lesson_keeps_its_weighted_output_gradient` | 17.2 |
| `test/test_identification_evidence.py` | `test_clause_order_and_evidence_are_stored_without_a_program`, `test_embedded_identification_does_not_grant_source_authority`, `test_luminosity_view_reads_signed_testimony_from_the_pair`, `test_native_sentence_carries_paired_identification_to_its_row`, `test_open_reading_keeps_both_poles_and_its_order`, `test_row_compaction_preserves_each_order_and_pole`, `test_source_authority_enters_the_positive_pole` | 17.12a,15,17,19–22; 17.20–21 |
| `test/test_identity_corpus.py` | `test_native_observer_uses_owned_individual_not_enclosing_sentence`, `test_observer_reports_discarded_mint_and_unadmitted_leaf_by_reason` | 17.15–16 |
| `test/test_identity_rows.py` | `test_model_time_is_independent_of_source_position_and_covers_the_run_horizon`, `test_mtm_holds_current_content_and_native_witnesses_are_live_queries`, `test_old_checkpoint_migrates_identity_parameters_once_and_marks_unknown_row_time`, `test_stm_retains_addresses_and_reads_reinterpretation_without_a_copy` | 17.12b,22; 17.20–21 |
| `test/test_identity_threading.py` | `test_bind_training_never_updates_or_creates_a_concept_region`, `test_compose_bind_pending_child_keeps_the_direct_question_role`, `test_compose_menu_uses_the_resolvers_observed_exclusions`, `test_frame_access_does_not_leak_a_native_individuals_other_stream_value`, `test_journal_native_bind_without_a_pool_token_keeps_its_symbol_origin`, `test_kept_journal_commits_two_verbs_one_identity_and_immutable_occurrences`, `test_later_bind_keeps_its_earlier_determination_under_a_wrapper`, `test_missing_determined_event_introduction_fails_loudly`, `test_nested_unresolved_operand_keeps_its_resolution_annotation`, `test_relative_determination_keeps_its_introduction_and_later_bind`, `test_resolved_compose_bind_keeps_the_parent_relation_closed`, `test_transparent_wrapper_retains_one_completed_event`, `test_two_relative_determinations_keep_distinct_objects`, `test_unminted_value_and_anonymous_name_have_distinct_pending_state` | 17.12a,15–16,20–22; 17.12b,22; 17.14,20; Alec retained-occurrence decision; 17.20–21 |
| `test/test_independent_components.py` | `test_native_admission_optimizer_ownership_and_checkpoint`, `test_unretained_innovation_cannot_change_population_cost` | 17.20–21 |
| `test/test_item18_finish.py` | `test_an_empty_query_concludes_its_unanswered_operand_not_the_request`, `test_an_empty_read_inside_a_conclusion_writes_its_operand_recursively`, `test_cued_read_only_reaches_explicit_or_recalled_rows`, `test_declared_caches_rebuild_and_are_not_checkpointed`, `test_entry_time_has_one_unambiguous_name`, `test_full_recall_window_materializes_only_the_selected_rows`, `test_recall_product_keeps_lanes_and_reads_only_selected_records`, `test_relation_write_updates_reference_indexes_without_rebuild`, `test_scalar_views_do_not_rewrite_a_words_existing_member_collection` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_item18_memory_trust.py` | `test_episode_query_uses_signed_references_and_survives_compaction`, `test_ltm1_drops_named_legacy_column_on_checkpoint` | 17.20–21 |
| `test/test_item18_perceptual_scalar.py` | `test_intervals_generalize_their_percepts_and_have_distinct_codes`, `test_point_and_native_read_share_the_collection_boundaries`, `test_text_uses_the_same_percept_collection_reader` | 17.20–21 |
| `test/test_item18_scalar.py` | `test_scalar_boxes_are_counted_native_definitions`, `test_scalar_only_presentation_closes_and_reconstructs_its_modalities`, `test_scalar_perception_keeps_percept_and_concept_references` | 17.20–21 |
| `test/test_item20_round2_memory.py` | `test_rejecting_commit_captures_value_pool_and_source_then_reraises` | 17.22 |
| `test/test_item20_round5_memory.py` | `test_alias_order_follows_the_words_denotation_for_symbol_and_name_references`, `test_all_link_callers_window_likeness_and_reference_values`, `test_consecutive_state_links_witness_a_noun_without_a_separate_counter`, `test_event_relation_path_promotes_with_native_tags_and_bands`, `test_kept_word_read_reenters_a_definition_learned_after_its_pending_path`, `test_link_coverage_reaches_live_member_readers_and_modal_is_not_a_state`, `test_merge_path_promotion_cycle_and_epoch_withdrawal`, `test_mtm_v12_and_policy_migrations_are_reported_and_invalid_members_are_atomic`, `test_native_kept_close_uses_all_three_link_callers`, `test_one_ltm_idea_generates_a_meaning_compatible_surface_without_input_stm`, `test_pool_and_unpooled_stm_share_the_menu_and_states_never_prepend`, `test_promoted_reference_reenters_as_one_higher_order_member`, `test_reentry_joins_native_identity_and_records_the_current_link_clock`, `test_reference_departure_filters_collection_order_before_selection`, `test_replaced_root_links_its_written_event_idea_instead_of_its_relation`, `test_same_close_binding_keeps_the_introductions_own_mention`, `test_shared_definition_witness_resolves_a_null_token_only_once`, `test_shared_sentence_strata_operator_references_gradients_and_expiry`, `test_state_teachers_use_later_references_and_keep_stored_values_detached`, `test_thought_trial_rollback_preserves_expired_external_references_without_resurrection`, `test_unnamed_alias_reads_a_pending_names_order_without_resolving_it`, `test_unnamed_leaf_closing_reads_its_idea_without_a_sentinel_concept`, `test_unnamed_reference_reenters_a_pending_words_later_definition`, `test_word_member_order_is_its_read_concept_not_the_clause_order`, `test_written_alias_retains_the_selected_pending_name_and_its_order`, `test_written_alias_uses_the_kept_binding_instead_of_earlier_identity_metadata`, `test_written_ltm_selection_replaces_earlier_identity_without_a_binding_child` | 17.15,17–23; 17.20–21 |
| `test/test_item20_round5_percepts.py` | `test_recurring_basic_collection_promotes_once_without_an_order_zero_concept` | 17.15 |
| `test/test_item20_round5_repair.py` | `test_default_pinned_banks_use_the_declared_word_bound` | 17.16 |
| `test/test_item20_round5_repair2.py` | `test_unnamed_order_one_site_binds_the_trial_selected_order_zero_path` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_item20_round6_gradients.py` | `test_order_zero_decode_reaches_the_grammar_inverse_and_selected_definition` | 17.3, 17.5 |
| `test/test_item20_round6_measurements.py` | `test_eventful_identity_keeps_the_checked_thought_as_a_member`, `test_memory_centroids_preserve_common_rails_with_fractional_evidence`, `test_promoted_name_keeps_its_order_when_an_old_mention_reads_the_definition` | 17.12a,15; 17.12b,22 |
| `test/test_item20_round6_memory.py` | `test_expired_pending_path_reenters_a_later_native_definition`, `test_rereading_the_same_position_at_a_new_order_preserves_the_original_member` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_item20_round6_repair.py` | `test_arrival_eligibility_reads_the_identity_after_shared_site_preparation`, `test_arrival_menu_keeps_the_planned_member_instead_of_a_duplicate_occurrence`, `test_collected_state_members_are_not_offered_as_unpooled`, `test_kept_event_link_precedes_its_words_new_trust_views`, `test_occurrence_writer_preserves_the_emitted_terminal_despite_priming`, `test_reconciled_reference_candidates_keep_one_collection_order` | 17.12b,22; 17.15,21–22; 17.20–21; 17.3–5 |
| `test/test_item20_round7_repair.py` | `test_composed_alias_keeps_the_selected_occurrence_instead_of_its_original_head`, `test_pending_selected_phrase_keeps_its_own_occurrence_when_the_head_is_resolved`, `test_state_trial_uses_the_retained_member_order_before_identity_promotion` | 17.20–21 |
| `test/test_item25_completion.py` | `test_ambiguous_pointer_identity_across_time_is_empty`, `test_compose_bind_keeps_its_addressed_shifted_introduction`, `test_kept_event_retains_its_determination_and_shift`, `test_lesson_sides_read_the_kept_occurrence_and_leave_an_unresolved_slot`, `test_no_op_keeps_the_form_then_binds_and_acquires_a_kind`, `test_optional_interpretation_keeps_the_pending_definition_path`, `test_pointer_identity_across_time_keeps_bound_predicate`, `test_same_order_identity_row_survives_the_epoch_boundary`, `test_sign_question_reads_the_latest_named_token_and_no_concept_for_an_unnamed_token`, `test_two_kept_signs_survive_a_full_pool_until_their_definition_is_read` | 17.12a,15–22; 17.20–21 |
| `test/test_item25_repair.py` | `test_determined_mention_keeps_an_occurrence_after_its_relation_was_written`, `test_two_thousand_rows_read_only_given_references_and_charge_returns` | 17.10,14,16,20–21; 17.12a; 17.20–21 |
| `test/test_item4_review.py` | `test_omission_surprise_and_non_marks_agree_at_both_native_writers` | 17.14,20 |
| `test/test_item55_mechanisms.py` | `test_absences`, `test_clock_independence` | 17.12a,14–22 |
| `test/test_item6_2_repair.py` | `test_fill_rebases_into_nonempty_goal_and_keeps_nested_ownership`, `test_occurrence_binding_also_carries_the_supplied_constituent_table`, `test_saved_construction_binds_and_writes_the_child` | 17.12a; 17.12a,14–22 |
| `test/test_item6_2_thinking.py` | `test_event_implication_reads_endpoint_when_order`, `test_modus_ponens_requires_premise_and_implication_witnesses`, `test_omission_lane_concludes_absence_without_negative_content`, `test_open_closing_and_later_binding_are_reference_driven` | 17.12a,14–22; 17.15,17.20; 17.20–21 |
| `test/test_iterated_symbolic_wave.py` | `test_relate_x_x_merges_to_one_edge` | 17.23 |
| `test/test_layout_removal.py` | `test_close_placement_updates_meaning_and_addressed_read_finds_idea`, `test_invalid_relation_checkpoint_rejected_before_mutation`, `test_kept_thought_writes_identity_value_and_inference_without_rewriting_chain`, `test_layout_checkpoint_is_versioned_and_never_resizes`, `test_resolved_sentence_compresses_to_one_idea_without_composing_edges` | 17.10,14–15,17,20–22 |
| `test/test_learning_readiness.py` | `test_native_progress_and_trained_artifact_evaluation_path` | 17.13, 17.19 |
| `test/test_live_references.py` | `test_public_reading_fuses_the_reference_before_capture_and_write`, `test_transparent_operation_preserves_a_vp_with_a_relational_operand` | 17.15; 17.17 |
| `test/test_ltm0_location.py` | `test_location_is_not_retained_and_time_survives_restore_and_compaction` | 17.12a,15–22 |
| `test/test_ltm_consolidation.py` | `TestProvisioning::test_provision_idempotent_count_via_runepoch_trigger`, `TestProvisioning::test_provisioned_rows_are_real_encodings`, `TestProvisioning::test_provisioned_trust_and_store_order`, `TestProvisioning::test_truthset_rows_land_via_real_forward`, `TestRuntimeUserIngestion::test_unary_preference_still_ends_and_records_user_truth`, `TestRuntimeUserIngestion::test_user_rows_coexist_with_conversation_rows`, `TestRuntimeUserIngestion::test_user_rows_land_alongside_provisioned`, `TestRuntimeUserIngestion::test_view_and_serve_surface_read_ltm_backed_data`, `TestSerialForwardAppendsStore::test_forward_grows_store_not_deque`, `TestStatelessRevive::test_load_syncs_view_even_with_no_user_rows`, `TestStatelessRevive::test_stateful_load_keeps_user_rows`, `TestStoreBackedChain::test_get_stm_chain_reads_a_compressed_idea_as_one_vector`, `TestStoreBackedChain::test_get_stm_chain_reads_stream_oldest_first`, `TestStoreBackedChain::test_get_stm_chain_respects_n_window`, `TestSurviveResetAndPersistence::test_state_dict_keys_present`, `TestTruthLayerLtmView::test_sync_empty_selection_empties_view` | 17.12a,15–16,18,20–21; 17.20–21 |
| `test/test_math_chain.py` | `test_prior_statement_is_a_reference_candidate_before_noun_columns_exist`, `test_thought_menu_checks_open_reference_types_before_choice` | 17.15 |
| `test/test_math_chain_repair.py` | `test_configured_store_still_raises_when_full`, `test_pending_relation_fills_in_place_in_both_arrival_orders` | 17.12a,14–22; 17.20–21 |
| `test/test_math_chain_section14.py` | `test_copula_fills_the_addressed_open_column_without_a_word_alias`, `test_forced_ordinary_pending_premise_in_both_orders`, `test_live_pending_premise_closes_in_both_orders` | 17.15,20–22; 17.20–21 |
| `test/test_medium_term_memory.py` | `test_reclosed_occurrence_survives_eviction_of_its_previous_token` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_memory_consolidation.py` | `test_capacity_eviction_during_close_drops_only_the_evicted_tokens_history`, `test_many_mentions_in_one_sentence_do_not_promote`, `test_retiring_a_token_keeps_another_tokens_event_until_row_reclamation`, `test_token_promotes_at_three_sentences_with_the_occurrence_fold_and_keeps_its_state` | 17.20–21 |
| `test/test_modal_timelines.py` | `test_promotion_folds_only_actual_events_and_document_end_keeps_ideas`, `test_retained_idea_keeps_its_subject_reference_after_token_expiry`, `test_some_every_include_actual_and_use_both_lanes` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_mtm_reference_values.py` | `test_event_on_alternative_reads_each_timelines_own_member`, `test_first_event_keeps_the_computed_forward_operand_only`, `test_identity_with_an_event_reads_its_retained_idea` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_mtm_referents.py` | `test_accessible_reads_each_identity_once`, `test_ingestion_pool_has_its_owner_and_reads_current_meanings`, `test_kept_thought_adds_its_written_occurrence_reference` | 17.20–21 |
| `test/test_negative_expectation.py` | `test_anticipation_ignores_incoming_staging_and_other_streams`, `test_declared_not_is_a_thought_act_not_an_observation` | 17.12a,14–22; 17.20–21 |
| `test/test_nested_retention.py` | `test_context_cycle_stays_finite_without_derivation_or_evidence_transfer`, `test_missing_local_constituent_rejects_append_before_owner_mutation`, `test_missing_local_context_reference_rejects_append_before_mutation`, `test_nested_read_limits_are_separate_and_explicit`, `test_repeated_edges_keep_role_order_without_evidence_inheritance`, `test_retained_structure_and_reference_identity_survive_checkpoint_restore`, `test_structural_restore_rejects_dangling_local_links_atomically`, `test_unavailable_owner_and_retired_reference_have_explicit_incomplete_results` | 17.12b,14–15,20–22; 17.20–21 |
| `test/test_nested_retention_owners.py` | `test_normal_truthset_replacement_keeps_live_thought_constituents_and_credit` | 17.12a,21–22 |
| `test/test_operators_final.py` | `test_definition_sentence_joins_identities_beside_the_interpret_definition`, `test_determined_sentences_keep_distinct_occurrences_and_universal_stays_high`, `test_determiner_lowering_keeps_order_without_minting_a_composing_argument` | 17.20–21 |
| `test/test_operators_round3a.py` | `test_live_grammar_uses_sparse_index_dense_nonzero_roots_and_exact_rung0`, `test_trust_to_pair_and_negation` | 17.12a,20 |
| `test/test_operators_round4a0.py` | `test_explicit_rewitness_keeps_a_self_reference_without_inventing_evidence`, `test_pre_layout_checkpoint_is_rejected_without_migration`, `test_references_and_content_keys_survive_compaction_and_checkpoint` | 17.12a,14–22; 17.20–21 |
| `test/test_part_closing.py` | `test_lower_word_alias_does_not_double_count_kept_occupations`, `test_nested_unknown_terminal_keeps_its_word_without_minting`, `test_new_relation_argument_needs_an_occurrence_and_no_concept_seat`, `test_pronoun_at_occurrence_closes_verb_over_antecedent_kind_without_seating`, `test_relation_argument_does_not_use_the_concept_tail`, `test_same_description_tokens_keep_distinct_occurrence_identities`, `test_sentence_projection_keeps_its_own_entry_after_an_embedded_mention`, `test_surface_phrase_keeps_the_unknown_head_word_without_minting`, `test_unseated_native_argument_does_not_acquire_the_working_value_at_close` | 17.12a,15–16,20–22; 17.20–21 |
| `test/test_part_read.py` | `test_forward_name_stays_pending_beside_a_witnessed_occurrence_complement` | 17.20–21 |
| `test/test_part_store.py` | `test_pair_rewitnesses_both_lanes_without_a_relation_kind` | 17.12a,14–22 |
| `test/test_placement_surface.py` | `test_promotion_folds_form_and_takes_one_context_step` | 17.20–21 |
| `test/test_positive_int_taxonomy.py` | `TestNativeAddresses::test_idea_address_resolves_to_its_native_row` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_prepared_answer_boundary.py` | `test_realization_uses_a_held_derivation_without_running_resolution` | 17.15: both trials use the existing produced-sentence tick; exact full-output equality, held derivation identity and no resolution or lexical rescanning remain. |
| `test/test_predicate_identity.py` | `test_first_three_slot_predicate_occurrence` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_presentation_memory.py` | `test_given_knowledge_keeps_its_required_dependency_without_experience_authority`, `test_unasserted_part_does_not_raise_a_retained_identity_order`, `test_verb_group_boundary_forgets_witnesses_but_keeps_the_learned_basis` | 17.20–21; 17.20–21 retires epoch snapshots, origins and withdrawal |
| `test/test_primed_reverse_integration.py` | `test_hard_mask_excludes_primed_off_category_ref`, `test_primed_disjunction_picks_primed_candidate` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_query_executors.py` | `test_content_query_keeps_all_roles_and_conflicting_fact_sources`, `test_query_does_not_fill_an_unconstrained_name_from_a_held_row`, `test_query_returns_only_the_completed_direction_of_an_unresolved_equality` | 17.12a,15–16,20–22; 17.20–21 |
| `test/test_query_vp_boundaries.py` | `test_occurrence_namespace_and_bound_are_never_rebound_to_matching_row` | 17.20–21 |
| `test/test_query_work_boundaries.py` | `test_registry_description_resolution_and_fact_scan_use_one_budget` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_reading_round.py` | `test_defining_clause_promotes_a_named_reading_before_recurrence`, `test_epoch_keeps_a_promoted_names_witnessed_order_after_deleting_its_occurrence`, `test_forward_name_is_not_filled_by_its_own_defined_word`, `test_kept_word_mention_retains_its_pending_referent`, `test_named_alias_follows_merge_restore_and_horizon_expiry`, `test_order_zero_bind_keeps_its_introduction_and_event`, `test_reading_close_follows_a_named_alias_after_merge`, `test_reading_event_cases_accept_native_part_rows_and_filter_open_questions`, `test_reading_pending_name_survives_documents_then_defines_on_recurrence`, `test_thought_part_persists_a_named_token_through_its_native_reference` | 17.12,12a,12b,14–22; 17.12a,14–22; 17.12a,15–16,20–22; 17.12b,22; 17.20–21 |
| `test/test_reasoning_cde_model.py` | `TestReasoningCDEModel::test_truthset_provisions_source_rows` | 17.21 |
| `test/test_reference_arrival_review.py` | `test_native_arrival_without_synthetic_occurrence_joins_its_working_entry`, `test_new_document_can_refine_a_same_stream_site_in_the_live_pool` | 17.20–21 |
| `test/test_reference_arrivals.py` | `test_anonymous_parent_keeps_its_raw_open_role_and_query_predicate`, `test_kept_wrong_equality_neither_binds_nor_reopens_a_token` | 17.20–21 |
| `test/test_reference_close.py` | `test_empty_close_has_no_policy_gradient_or_store_side_effect`, `test_kept_journal_supplies_np_content_predicate_and_positional_provenance`, `test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them`, `test_revision_merges_three_accessible_groups_transitively_across_cued_documents` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_reference_mentions.py` | `test_later_fill_releases_the_menu_without_an_expectation_loss` | 17.20–21 |
| `test/test_reference_participants.py` | `test_journal_and_close_use_the_reading_order_and_argument_position`, `test_ordinary_menu_uses_the_retained_argument_occupations` | 17.20–21 |
| `test/test_reference_reconstruction.py` | `test_multileaf_bound_root_keeps_its_own_occurrence`, `test_named_token_keeps_its_content_before_and_after_the_close` | 17.20–21 |
| `test/test_reference_review10.py` | `test_later_ball_mention_preserves_the_boxs_key_members_and_count` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_relation_closing.py` | `test_three_slot_relation_closes_unindexed_numerical_operands`, `test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent` | 17.12a,14–22 |
| `test/test_relevance_bases.py` | `test_symbol_history_projection` | 17.23 |
| `test/test_replacement_kernel_contracts.py` | `test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm`, `test_world_refutation_or_conflict_never_closes_taxonomy_false` | 17.12a,14–22 |
| `test/test_retire_mode.py` | `test_closed_zero_is_unverified_and_only_open_references_make_question`, `test_mode_api_and_old_semantic_layout_are_rejected` | 17.12,12a,12b,14–22; 17.12a,14–22; 17.12a,15–16,20–22; 17.20–21 |
| `test/test_review13_subspaces.py` | `test_close_placement_is_confined_to_detached_context_coordinates` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_role_collapsed_grammar.py` | `test_symbolicspace_owns_its_starts` | 17.17 |
| `test/test_runtime_split_ingestion.py` | `TestRuntimeSplitIngestion::test_store_truths_records_truths` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_selected_nested_meaning.py` | `test_10_packed_pending_and_eager_ends_write_identical_rows`, `test_completed_ask_program_offers_the_same_controller_without_inventing_holes`, `test_constituent_capacity_failure_and_invalid_local_ref_do_not_partially_write` | 17.12a,14–22 |
| `test/test_sentence_boundary.py` | `test_every_entry_writes_once_with_one_trial_and_no_optimizer` | 17.15–16, 17.20–21 |
| `test/test_sentence_compose.py` | `test_disabled_sentence_prediction_leaves_adam_momentum_unused`, `test_real_packed_ends_train_before_the_next_sentence` | 17.2, 17.5; 17.20 |
| `test/test_sentence_end_state.py` | `test_no_clause_derivation_or_factored_target_is_checkpointed`, `test_program_meaning_reads_ended_points_from_their_rows`, `test_reading_an_idea_returns_its_one_occupied_slot` | 17.12a,15–22 |
| `test/test_sentence_expectation.py` | `test_real_provisioning_is_not_an_external_prediction_stream`, `test_streaming_and_packed_boundary_use_same_full_roles` | 17.12a,15–22 |
| `test/test_sentence_identity.py` | `test_closing_idea_is_narrow_without_a_derivation_part` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_sentence_priming.py` | `test_part_graph_uses_the_constructed_relation_capacity` | 17.23 |
| `test/test_sentence_references.py` | `test_predictor_context_carries_address_and_reads_only_that_retained_row` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_sentence_retention.py` | `test_narrow_idea_retains_its_identity_without_a_predicate_operand` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_space_carrier_bridge.py` | `test_legacy_duplicate_checkpoint_keys_migrate_to_single_owners` | 17.20, 17.23 |
| `test/test_step2b_storage.py` | `test_held_content_without_a_thought_signature_has_no_expectation_holes`, `test_negative_trust_derived_pair_reverses_only_on_the_relation_edge`, `test_pending_forgetting_pressure_survives_restore_below_high_water` | 17.14,20; 17.20–21 |
| `test/test_stm_chunk_gates.py` | `test_bind_menu_at_each_order_and_kept_mention_commit`, `test_prepend_breaks_equal_time_ties_by_admission_and_keeps_event_recency`, `test_prepend_transients_do_not_prime_or_place_and_promoted_symbol_registers`, `test_restaging_reads_live_values_once_without_reading_the_menu`, `test_unpublished_timeline_cannot_survive_as_a_recent_identity` | 17.12b,22; 17.20–21 |
| `test/test_stm_chunk_repairs.py` | `test_same_chunks_supply_current_time_and_untimed_estimate` | 17.15,20 |
| `test/test_stored_idea_generation.py` | `test_nontaxonomic_cue_does_not_enter_concept_order_reader` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_supplied_answer_training.py` | `test_real_supplied_paths_cost_before_updates_and_reach_understanding` | 17.2, 17.4, 17.7 |
| `test/test_sweep_raise_repair.py` | `test_unlike_order_equality_keeps_evidence_without_a_partial_raise` | 17.12b, 17.15 |
| `test/test_sweep_repairs.py` | `test_numerical_idea_query_reads_the_single_point_column` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_taxonomy_entry.py` | `test_actual_model_entry_and_checkpoint_keep_taxonomy_identity`, `test_normal_controller_does_not_certify_a_world_row_as_taxonomy`, `test_normal_entry_does_not_verify_generated_vectors_as_taxonomy`, `test_public_evaluate_follows_taxonomy_and_retains_record_provenance` | 17.23 |
| `test/test_taxonomy_policy.py` | `test_numeric_world_evidence_cannot_establish_taxonomic_inclusion`, `test_unrelated_true_episode_cannot_establish_the_next_parent_relation` | 17.11; 17.12a,14–22; 17.12b,22 |
| `test/test_taxonomy_public_helpers.py` | `test_public_part_helpers_cannot_accept_geometric_or_world_row_evidence` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_taxonomy_routing.py` | `test_partof_is_not_established_by_an_ltm1_description`, `test_partof_is_not_established_by_vector_containment` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_taxonomy_view.py` | `test_percept_codes_vector_overlap_and_world_rows_are_not_taxonomy_edges`, `test_reified_relation_is_visible_without_endpoint_mutation`, `test_taxonomy_budgets_report_incomplete_without_fabricating_falsehood` | 17.23 |
| `test/test_ternary_truth_store.py` | `TestPersistence::test_reset_clears`, `TestQueries::test_empty_queries` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_tetralemma_leaf.py` | `test_required_evidence_read_of_true_and_false_is_both`, `test_walk_to_leaf_reference_and_stored_row` | 17.12a |
| `test/test_thinking_kernel.py` | `TestDepth3RelativeEndState::test_first_trained_read_reaches_depth3_end_state` | 17.21 |
| `test/test_thought_answer_adapters.py` | `test_open_relation_answer_preserves_every_member_without_another_reader` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_thought_binding.py` | `test_a_read_binding_publishes_a_proof_occurrence_without_composing_a_part`, `test_concluded_relation_keeps_its_occurrence_argument`, `test_empty_read_does_not_bind_its_temporary_question_address` | 17.12,12a,12b,14–22; 17.12a,14–22; 17.20–21 |
| `test/test_thought_infer.py` | `test_membership_match_uses_the_existing_part_cache`, `test_unused_derived_idea_is_forgotten_by_existing_configuration`, `test_worked_token_property_line_queries_backward_and_forward_then_replays` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_thought_operation_catalog.py` | `test_canonical_form_derives_closed_and_open_roles_without_alias_methods`, `test_thought_execution_receives_the_common_context_and_only_capability_views` | 17.23 |
| `test/test_thought_review.py` | `test_memory_attention_is_bounded_metered_and_row_local` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_trust_round1.py` | `test_item107_generic_sentence_never_names_its_chain`, `test_kept_word_views_use_written_content_and_for_radius_only`, `test_nonconcept_operand_contributes_one`, `test_null_kind_named_from_held_signs_reaches_the_definition_threshold`, `test_unreferenced_empty_leaf_contributes_one_at_the_close` | 17.12a; 17.15; 17.17,21; 17.20–21 |
| `test/test_trust_round2.py` | `test_shared_word_teaches_another_argument_without_dividing_existing_authority` | 17.12a,17–18 |
| `test/test_truth_criterion_gates_learning.py` | `test_registration_does_not_depend_on_testimony_trust` | 17.12a |
| `test/test_truth_ideas_routing.py` | `TestRelativeTrustPersistence::test_evidence_and_unscaled_argument_identity_survive_checkpoint`, `TestTrustStamp::test_default_one` | 17.12a; 17.12a,14–22 |
| `test/test_truth_text.py` | `test_supplied_trust_preserves_both_possible_grammatical_fields` | 17.20–21 |
| `test/test_two_codebook_meta_taxonomy.py` | `TestInsertIdea::test_completed_point_gets_a_native_address_without_changing_earlier_rows`, `TestInsertIdea::test_distinct_occurrences_get_distinct_addresses` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_unresolved_fixed_operand.py` | `test_query_requires_the_address_of_an_unresolved_fixed_operand` | 17.7–10, 17.12a |
| `test/test_verb_learning.py` | `test_copular_join_is_kept_and_fires_the_previous_verb_teacher`, `test_two_kept_sentences_join_the_copular_observation_and_teach_once` | 17.20–21 |
| `test/test_whole_paths.py` | `test_hidden_word_changes_only_loss_targets_through_the_close`, `test_kept_verbal_output_reaches_the_live_understanding_and_conditioner` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_word_grain_attention.py` | `test_attention_off_is_plain_reverse` | 17.12–22 (typed references, reduced columns, live MTM or writer port) |
| `test/test_word_interpretation.py` | `test_every_read_word_has_a_concept_and_symbol`, `test_testimony_and_its_later_definition_are_read_by_the_parallel_field` | 17.2, 17.4–5 |

**Whole-test retirements (285).**

A whole retirement here means its tested contract is removed. It is not
a skipped or deselected test, and does not remove neighboring assertions
about storing or reading a supplied trust. Tests of those surviving
authority contracts appear among the ports above.

| File | Retired names | Point of §17 and reason |
|---|---|---|
| `test/test_activation_priming_multiply.py` | `test_configure_priming_none_args_leave_unchanged`, `test_configure_priming_overrides_defaults`, `test_note_selection_bumps_total_count`, `test_note_selection_counts_boosted_separately`, `test_priming_disabled_omits_kwargs_from_helper`, `test_taxonomy_decay_per_batch`, `test_taxonomy_default_config`, `test_taxonomy_reset_per_batch`, `test_word_space_soft_reset_clears_priming` | 17.23: The symbol-space taxonomy and its attach, priming/configuration/lifecycle contract are removed. Conceptual taxonomy and numeric recommender tests remain. |
| `test/test_bind_thought.py` | `test_objectless_predicate_keeps_its_identity_in_the_ended_compound` | 17.17, 17.20, 17.23: The operator chain of the ended compound is retired. |
| `test/test_clause_acceptance.py` | `test_08_direct_bare_np_rewitnesses_its_address` | §17.15,21: every repeat has a new time and row |
| `test/test_clause_closing.py` | `test_bare_unlocated_np_rewitnesses_its_address` | §17.15,21: every repeat has a new time and row |
| `test/test_clause_storage.py` | `test_native_provisioned_content_requires_its_layout_version`, `test_retired_relation_checkpoint_is_rejected_with_dependents_intact`, `test_retired_relation_columns_are_rejected_without_dropping_rows` | §17.15: four-coordinate working fixtures have no durable time layout; checkpoint validation is exercised on native LTM0 by certificate 47 |
| `test/test_creation_observations.py` | `test_current_checkpoint_missing_creation_observation_raises`, `test_legacy_checkpoint_imports_unsnapshotted_observation`, `test_legacy_supplied_rows_import_saved_observation`, `test_missing_creation_observation_raises`, `test_supply_closure_dependency_keeps_start_observation`, `test_withdrawn_occurrence_later_referred_to_keeps_start_observation`, `test_withdrawn_pre_epoch_relation_later_referred_to_keeps_start_observation` | 17.14, 17.20–21: Creation observations and withdrawal snapshots retired. |
| `test/test_cs_symbol_table.py` | `test_assert_concrete_whole_retires_everything_bias_edge`, `test_assert_relation_raw_codes_reserve_snap_row_no_edges`, `test_assert_relation_replaces_poles_and_enters_embedding`, `test_assert_relation_sym_weight_lands_on_edge`, `test_relate_is_idempotent_per_pair`, `test_singleton_concept_is_idempotent_unit_set`, `test_synthesize_higher_order_idempotent_per_part_set` | 17.23: Removed unused convenience writer and its idempotency or definition-patching contract. Native collection, relation reads, weights and pruning have separate retained tests. |
| `test/test_document_covers.py` | `test_cover_concept_reads_only_its_segment_under_the_shared_meter`, `test_cover_generator_is_global_rng_neutral_advances_and_restores`, `test_cover_is_an_ordinary_recent_candidate_with_no_lexical_name`, `test_cover_lookup_uses_the_index_and_charges_before_body_reads`, `test_cover_never_certifies_a_proposition`, `test_cover_region_does_not_certify_unrelated_content`, `test_cursor_reuses_the_book_cover_on_a_reread`, `test_end_document_retains_tokens_across_segments_until_epoch`, `test_equal_time_boundary_and_two_streams_keep_exact_ownership`, `test_forgetting_protects_a_cover_for_the_pass_that_drops_its_last_member`, `test_full_store_boundary_pass_precedes_the_next_document_cover` | 17.20–21: Covers and document-scoped reads retired. |
| `test/test_epoch_history.py` | `test_epoch_restores_every_relation_column_index_and_given_row`, `test_ten_epochs_ground_the_same_names_in_one_fixed_document` | §17.20–21 retires epoch snapshots, origins and withdrawal |
| `test/test_epoch_start.py` | `test_checkpoint_that_loses_epoch_start_rejected`, `test_embedded_supply_keeps_only_its_own_assertion_authority`, `test_forgetting_a_supply_releases_its_two_memory_snapshots`, `test_history_relation_to_supplied_occurrence_is_not_supplied_authority`, `test_mid_epoch_checkpoint_load_reuses_instance_without_stale_observations`, `test_mid_epoch_withdrawal_does_not_restore_observations`, `test_new_given_knowledge_retains_its_relation_and_transitive_dependencies`, `test_new_mid_epoch_occurrence_is_retained_through_a_relation_endpoint`, `test_regroup_restored_snapshot_and_orthogonal_new_row`, `test_resupplying_a_snapshot_occurrence_keeps_its_new_ground`, `test_retained_new_occurrence_restores_its_creation_observation`, `test_shared_document_does_not_turn_history_into_a_supply`, `test_supplied_outer_keeps_history_relation_as_unverified_dependency`, `test_supply_outer_alone_keeps_authority_for_all_closing_nodes`, `test_supply_requires_explicit_asserted_pair`, `test_validation_before_training_does_not_become_epoch_start`, `test_withdrawn_relation_handle_stays_reserved_while_named`, `test_withdrawn_supply_does_not_return_in_either_memory` | §17.20–21 retires epoch snapshots, origins and withdrawal |
| `test/test_epoch_start.py` | `test_read_of_ambiguous_remembered_name_fails_before_opening_mtm` | §17.21: the ambiguous remembered-name-only read required withdrawing both live definitions at the epoch; active multiple denotations retain their existing field read. |
| `test/test_existence_evidence.py` | `test_exist_matches_the_complete_relational_description` | 17.7–11: Existence by stored role/vector equality is retired; the question now supplies its extension, certified by 38–40. |
| `test/test_existence_ingestion.py` | `test_prediction_and_observation_occurrences_cannot_certify_their_content` | 17.20–21: Estimate rows and trust overridden by observation/fact kind are retired. |
| `test/test_existence_metadata.py` | `test_full_description_metadata_changes_which_referent_is_requested`, `test_legacy_checkpoint_does_not_invent_fact_status_for_conversation_rows`, `test_question_cannot_be_written_as_accepted_fact`, `test_unaccepted_records_cannot_certify_their_own_referents` | 17.20–22: Scoped context, row-kind acceptance and implicit legacy provenance are retired; conversation zero trust and ambiguity rejection are certified separately. |
| `test/test_existence_provenance.py` | `test_missing_semantic_context_remains_visible_in_the_checked_result`, `test_partial_new_checkpoint_cannot_forget_required_scope` | 17.20–22: Required arbitrary sidecar scope and its columns are retired. |
| `test/test_existence_public_inputs.py` | `test_checked_reader_accepts_complete_single_idea_description` | 17.7–11: Raw vector equality as the complete existence lookup is retired; native public extension read is certified separately. |
| `test/test_existence_sidecar.py` | `test_restored_sidecar_cannot_erase_or_replace_recorded_scope`, `test_restored_sidecar_cannot_swap_contexts_between_existing_occurrences`, `test_source_update_cannot_certify_missing_semantic_context` | §17.20: semantic sidecars, fingerprints and certification by source go |
| `test/test_expectation_gain.py` | `test_expected_identity_survives_zero_content_and_retained_pair_restore`, `test_fused_retained_row_replays_original_two_lane_target` | §17.20: stored estimate/prior rows go; gain and predictor contracts remain |
| `test/test_expectation_migration.py` | `test_authentication_does_not_bless_corrupt_current_estimates`, `test_pre_layout_estimate_rejected_and_current_roundtrips_once` | §17.20: estimate rows, their authentication and paired-row persistence go |
| `test/test_expectation_retention.py` | `test_estimate_pair_overflow_raises_without_publishing_half_a_pair`, `test_retained_estimate_links_its_external_observation_and_restores`, `test_retained_estimates_do_not_enter_the_generic_ltm_chain` | §17.20: persisted estimate rows and pair writer go |
| `test/test_forgetting.py` | `test_retention_reads_both_evidence_lanes` | 17.12a, 17.14: The retained evidence pair, net trust and retention-value reader retired. |
| `test/test_forgetting.py` | `test_dissonance_drops_weaker_opposition_not_intrinsic_both` | 17.14: Dissonance no longer determines retention. |
| `test/test_forgetting.py` | `test_optional_coverage_separates_both_from_neither` | 17.14: Optional coverage retention retired. |
| `test/test_forgetting.py` | `test_value_terms_outweigh_age` | 17.14: The weighted retention value retired. |
| `test/test_forgetting.py` | `test_age_uses_percept_time_and_rewitnessing_does_not_refresh` | 17.14, 17.16, 17.21: Age retention, stream witnesses and resubmission matching retired. |
| `test/test_forgetting.py` | `test_scope_prevents_false_dissonance` | 17.14, 17.21: Scoped dissonance retention retired. |
| `test/test_forgetting.py` | `test_pair_unlinking_preserves_observation_and_restores_cleanly` | 17.20: Stored expectation/observation pair and its unlinking retired. |
| `test/test_generate_repair.py` | `test_epoch_guard_still_rejects_unwitnessed_constructed_individual` | 17.14,17.20: The removed retention/epoch pass re-witnessed promoted concepts from occurrence constituents. Promotion now records the native order at creation and does not re-witness at an epoch. |
| `test/test_heat_reverse_wiring.py` | `TestHeatReverseSelfPriming::test_off_path_records_no_selection`, `TestHeatReverseSelfPriming::test_selected_left_row_is_reprimed`, `TestHeatReverseWiringIntersectionON::test_off_baseline_picks_smaller_norm_row`, `TestHeatReverseWiringIntersectionON::test_on_and_off_differ`, `TestHeatReverseWiringIntersectionON::test_on_primer_steers_pick_to_hot_typed_row`, `TestHeatReverseWiringLiveON::test_off_baseline_picks_larger_norm_row`, `TestHeatReverseWiringLiveON::test_on_and_off_differ`, `TestHeatReverseWiringLiveON::test_on_primer_steers_pick_to_hot_typed_row` | 17.23: The symbol taxonomy and its reverse retrieval path retired. |
| `test/test_identification_evidence.py` | `test_pair_retains_both_and_neither_until_explicit_signed_reassertion`, `test_withdrawing_testimony_clears_its_pair` | §17.12a,21: stored trust cannot be rewritten or withdrawn |
| `test/test_identification_evidence.py` | `test_old_scalar_checkpoint_does_not_invent_evidence_or_order` | §17.15: no durable checkpoint without time; old source-only trust layout is already rejected in native certificate 47 |
| `test/test_identity_threading.py` | `test_three_slot_close_keeps_the_introduction_used_by_its_embedded_bind` | 17.17: Its truncated journal asks the close to invent an operator relation from three roots with no selected relative predicate. The operator relation/start is deleted. |
| `test/test_identity_threading.py` | `test_dummy_binds_preserve_the_relative_operand_in_a_three_slot_close` | 17.17: The asserted operator relation is removed. Its independent rejection assertion survives as test_three_point_roots_require_a_selected_relative_predicate; dummy binding a selected relative is covered by native selected-tree tests. |
| `test/test_independent_components.py` | `test_row_definedness_survives_rewitness_and_checkpoint_without_changing_truth` | 17.14,20–21: LTM0 occurrence identity radius from the re-witness count and resubmission match are retired; concept and dictionary observation counts remain. |
| `test/test_intent_priming.py` | `test_taxonomy_prime_with_weights_max_merges` | 17.23: The symbol-space taxonomy and its attach, priming/configuration/lifecycle contract are removed. Conceptual taxonomy and numeric recommender tests remain. |
| `test/test_item18_completion.py` | `test_mentioning_a_cover_is_an_observation_not_a_new_cover` | 17.20–21: Covers, row kinds and persisted forecasts are retired. |
| `test/test_item18_memory_trust.py` | `test_operator_gates_follow_kept_ancestors_and_checkpoint` | 17.18: Operator gates, dependencies and their checkpoint are retired; the full grammar is available. |
| `test/test_item18_retention.py` | `test_maintained_retention_matches_recompute_after_witness_scope_and_delete` | 17.14: The maintained retention value retired. |
| `test/test_item20_round5_repair.py` | `test_short_term_stream_table_rejects_another_stream_without_redirection` | 17.16: A stream table and its stream-range rejection are retired; all history rows enter one history. |
| `test/test_item20_round6_memory.py` | `test_unreduced_relative_field_keeps_predicate_without_inventing_an_operation` | 17.17–18: An operator relation with a retained predicate/seat no longer exists. |
| `test/test_item20_round6_repair.py` | `test_recurrence_waits_for_each_members_order_before_allocating` | §17.20: recurrence grouping goes; activity promotion has its own native-order certificate |
| `test/test_item20_round7_repair.py` | `test_anchor_capacity_is_checked_before_the_relation_or_any_row_is_written`, `test_anchor_lifecycle_follows_compaction_forgetting_and_epoch_rows`, `test_anchors_do_not_change_answers_content_reads_consolidation_mints_or_candidates`, `test_empty_episode_allocates_no_relation_anchor`, `test_native_relation_anchor_content_recall_reinstates_its_percepts`, `test_relation_only_anchor_keeps_first_entry_and_never_becomes_a_root` | 17.12,20–21: Relation-only episode/anchor rows and their row kind are retired; entries belong only to retained ideas. |
| `test/test_item20_round7_time.py` | `test_epoch_restores_both_clocks_and_repeats_sentence_stamps` | 17.15–16,20–21: Presentation snapshots and automatic epoch rollback of rows and clocks are retired; clocks continue along the retained history. |
| `test/test_item4_review.py` | `test_retained_prior_keys_are_bounded_by_realized_support_and_replay_is_exact` | 17.20: Historical forecast sidecars and their bounded key encoding are retired; live predictor contracts stay. |
| `test/test_item6_2_thinking.py` | `test_event_implication_cannot_invent_an_order_for_unknown_legacy_time` | 17.15; Alec checkpoint decision: An old checkpoint without event time is rejected before read. Certificate47 checks rejection; untimed durable endpoints no longer exist. |
| `test/test_layout_removal.py` | `test_full_idea_inventory_can_rewitness_its_existing_sentence`, `test_stream_overflow_rejects_before_any_durable_write` | §17.16,21: one history, no stream column, repeat writes a new row |
| `test/test_memory_consolidation.py` | `test_angular_recurrence_admits_drift_and_keeps_evidence_weighted_content`, `test_epoch_clears_the_readings_evidence_and_recognizes_its_learned_symbol`, `test_reading_recurrence_counts_distinct_documents_not_reobservations`, `test_recurrence_time_is_ignored_and_compaction_keeps_assignments` | §17.14,20–21: row recurrence and epoch withdrawal replaced by per-row activity promotion |
| `test/test_native_attention_field.py` | `test_rereading_cannot_bind_the_occurrence_or_embedded_rows_it_replaces` | 17.21: No source replacement or source-scoped exclusion remains. |
| `test/test_negative_expectation.py` | `test_metadata_comes_from_the_preceding_occurrence_not_the_target`, `test_old_checkpoint_unknown_surprise_is_not_zero`, `test_pair_lookup_uses_occurrence_index_in_both_directions`, `test_retained_pair_keeps_full_prior_and_surprise_across_restore` | §17.14,20: stored surprise and expectation rows go; online prediction tests remain |
| `test/test_nested_retention.py` | `test_external_live_root_pins_an_occurrence_through_origin_replacement`, `test_invalid_survivor_closure_cannot_partly_withdraw_evidence`, `test_origin_replacement_retains_reachable_content_without_fact_authority`, `test_withdrawal_preserves_occurrences_in_retained_semantic_context` | 17.20–21: Origins, withdrawal and durable arbitrary semantic context are retired. |
| `test/test_operators_round4a.py` | `test_content_address_is_independent_of_store_and_persists_with_occurrences` | Point 20 removes sentence_content_keys and their index. |
| `test/test_operators_round4a0.py` | `test_rewitness_keeps_poles_and_the_round3a_net_existence_read` | 17.11–12a,21: Re-witnessing into one idea and a netted existence lane are retired; two-lane maximum is certificate 40. |
| `test/test_operators_round4a0.py` | `test_four_occurrences_after_four_hundred_presentations_and_full_upsert`, `test_full_store_can_rewitness_both_existing_prediction_pair_addresses`, `test_live_gate_presentation_reuses_covers_addresses_and_epoch_time`, `test_rereading_replaces_vectors_refreshes_time_and_joins_poles` | §17.15,18,20–21: repeats append; no stored prediction pair, covers, epochs or gates |
| `test/test_operators_round4a0.py` | `test_address_is_full_int64_content_based_and_rng_independent` | §17.15: addresses are times, not document/content hashes |
| `test/test_part_expectation.py` | `test_closing_audit_survives_bank_changes_and_checkpoint`, `test_part_expectation_capacity_failure_precedes_all_admission` | 17.20: Frozen forecast rows, paired historical residuals and their admission capacity are retired. |
| `test/test_parthood_orders.py` | `TestTaxonomy::test_taxonomy_add_and_walk_by_order`, `TestTaxonomy::test_taxonomy_flat_enumeration_across_orders`, `TestTaxonomy::test_wordspace_hosts_taxonomy` | 17.23: The unused symbol-space taxonomy is deleted. |
| `test/test_presentation_memory.py` | `test_boundary_removes_occurrence_relations_transitively_and_keeps_concept_relations`, `test_cover_survives_empty_memory_checkpoint_and_reuses_symbol_without_rng`, `test_invalid_cover_map_does_not_replace_a_valid_checkpoint` | §17.20–21 retires epoch snapshots, origins and withdrawal |
| `test/test_primed_reverse_hard_mask.py` | `test_attach_allocates_priming_buffer`, `test_attach_binds_view_for_propagation`, `test_attach_without_view_is_noop_on_priming`, `test_helper_kwargs_drive_recommender_byte_for_byte`, `test_kwargs_empty_intersection_returns_empty_long_tensor`, `test_kwargs_intersection_rows_for_binary_rule`, `test_kwargs_per_batch_priming`, `test_kwargs_priming_present_when_buffer_allocated`, `test_kwargs_priming_reflects_primed_state`, `test_kwargs_returns_empty_without_knowledge`, `test_kwargs_unary_omits_right`, `test_reattach_preserves_primed_values` | 17.23: The symbol-space taxonomy and its attach, priming/configuration/lifecycle contract are removed. Conceptual taxonomy and numeric recommender tests remain. |
| `test/test_primed_reverse_integration.py` | `test_priming_decays_between_calls`, `test_priming_disabled_matches_unprimed`, `test_sentence_boundary_clears_priming` | 17.23: The symbol-space taxonomy and its attach, priming/configuration/lifecycle contract are removed. Conceptual taxonomy and numeric recommender tests remain. |
| `test/test_reference_arrival_review.py` | `test_arrival_readers_exclude_scoped_rows_before_a_new_stream_column_exists` | Point 16 removes retained-row stream scoping. |
| `test/test_sentence_retention.py` | `test_capacity_requires_room_for_new_estimate_and_observation`, `test_invalid_prediction_does_not_publish_embedded_clauses`, `test_repeated_relation_can_retain_multiple_estimates_without_duplicate_claims` | 17.20: Estimate rows, their capacity and historical-pair admission are retired. |
| `test/test_stm_prerequisites.py` | `test_native_recurrence_capacity_defaults_to_concept_inventory`, `test_recurrence_capacity_is_independent_fixed_and_versioned` | §17.20: recurrence table and construction parameters go |
| `test/test_stream_state_shapes.py` | `test_rejected_taxonomy_resize_does_not_partially_resize_its_owner`, `test_taxonomy_resize_cannot_reassign_live_priming_to_new_batch` | 17.23: The symbol-space taxonomy and its attach, priming/configuration/lifecycle contract are removed. Conceptual taxonomy and numeric recommender tests remain. |
| `test/test_symbolic_heat_retrieval.py` | `TestBuildOuterHeat::test_dense_C_equals_UtU`, `TestBuildOuterHeat::test_dtype_matches_codebook`, `TestBuildOuterHeat::test_empty_S_dense_is_zeros`, `TestBuildOuterHeat::test_empty_S_low_rank_shape`, `TestBuildOuterHeat::test_lowrank_U_times_q_equals_dense_C_times_q`, `TestBuildOuterHeat::test_rows_restriction`, `TestBuildSemanticHeat::test_dtype_matches_codebook`, `TestBuildSemanticHeat::test_matches_dense_reference`, `TestBuildSemanticHeat::test_returns_zeros_when_no_buffer`, `TestBuildSemanticHeat::test_rows_restriction`, `TestBuildSemanticHeat::test_topk_restriction`, `TestBuildSemanticHeat::test_zeros_when_nothing_hot`, `TestCandidateUnionIncludesContentNearest::test_cold_nearest_in_union_no_typed_filter`, `TestCandidateUnionIncludesContentNearest::test_cold_nearest_in_union_typed_admits_it`, `TestCandidateUnionIncludesHot::test_hot_distant_in_union_no_typed_filter`, `TestCandidateUnionIncludesHot::test_hot_distant_in_union_typed_admits_it`, `TestCarrierChangesWeight::test_delta_changes_weight`, `TestCarrierChangesWeight::test_gamma_changes_weight`, `TestConfigurePrimingFromSymbolicPriming::test_flag_flips_and_gate_respects_it`, `TestConfigurePrimingFromSymbolicPriming::test_missing_taxonomy_is_safe`, `TestDiagnosticsConsistency::test_diagnostics_consistent_on_normal_path`, `TestHeatContentPrefersExpectedCandidate::test_candidate_from_retrieval_boosts_expected_row`, `TestHeatMask::test_batch_none_returns_all_rows`, `TestHeatMask::test_cold_rows_give_zero`, `TestHeatMask::test_hot_rows_give_excess`, `TestHeatMask::test_matches_priming_mask_formula`, `TestHeatMask::test_returns_none_when_no_buffer`, `TestHeatMask::test_returns_only_live_slice`, `TestLowRankCarrierMatchesDense::test_lowrank_delta_equals_dense_delta`, `TestMillionRowNoVVSmoke::test_large_codebook_primer_and_lowrank_no_vv`, `TestPrimerModeByteIdentical::test_primer_equals_default_call`, `TestPrimerModeByteIdentical::test_primer_equals_reference_formula`, `TestPrimerModeByteIdentical::test_primer_mode_with_nonzero_gamma_delta_ignored`, `TestReduceCommitPrimesChildRefs::test_child_refs_noop_when_off`, `TestReduceCommitPrimesChildRefs::test_child_refs_primed_when_on`, `TestRetrievalGracefulFallback::test_returns_empty_when_basis_has_no_getW`, `TestRetrievalGracefulFallback::test_returns_empty_when_basis_none`, `TestRetrievalGracefulFallback::test_returns_empty_when_getW_returns_none`, `TestRetrievalGracefulFallback::test_returns_empty_when_knowledge_none`, `TestSentenceResetClearsHeat::test_reset_clears_row_heat_and_semantic`, `TestSymbolicPrimingOffIsNoop::test_off_leaves_priming_at_unity`, `TestSymbolicPrimingOffIsNoop::test_off_no_op_even_for_valid_ref`, `TestSymbolicPrimingOnMakesRefHot::test_committed_ref_is_hot`, `TestSymbolicPrimingOnMakesRefHot::test_negative_ref_is_noop_when_on`, `TestSymbolicPrimingOnMakesRefHot::test_neighbor_becomes_hot_via_propagate`, `TestTopkHeat::test_empty_when_k_zero`, `TestTopkHeat::test_empty_when_no_buffer`, `TestTopkHeat::test_empty_when_nothing_hot`, `TestTopkHeat::test_excludes_slack_rows`, `TestTopkHeat::test_returns_fewer_than_k_when_few_hot`, `TestTopkHeat::test_rows_restriction`, `TestTopkHeat::test_rows_restriction_excludes_slack`, `TestTopkHeat::test_sorted_by_heat_descending`, `TestTypedMaskExcludesHotInadmissible::test_inadmissible_hot_excluded_from_rows`, `TestTypedMaskExcludesHotInadmissible::test_priming_above_unity_for_at_least_one_candidate`, `TestTypedMaskExcludesHotInadmissible::test_priming_identity_for_non_candidates`, `TestTypedOnlyFallback::test_typed_only_fallback` | 17.23: Symbol taxonomy, heat carrier and retrieval retired; Ops tests remain. |
| `test/test_taxonomy_priming.py` | `test_allocate_priming_preserves_existing_primed_values`, `test_attach_view_updates_live_count`, `test_decay_idempotent_at_identity`, `test_decay_noop_when_unallocated`, `test_decay_pulls_primed_toward_identity`, `test_prime_does_not_touch_other_entries`, `test_prime_drops_out_of_range`, `test_prime_is_max_lifts_existing`, `test_prime_is_max_not_replace`, `test_prime_isolates_batches`, `test_prime_noop_when_unallocated`, `test_prime_sets_target_to_two`, `test_priming_initial_shape_and_value`, `test_priming_mask_per_batch`, `test_priming_mask_returns_live_slice`, `test_propagate_depth1_does_not_lift_siblings`, `test_propagate_depth1_lifts_children`, `test_propagate_depth1_lifts_parent`, `test_propagate_depth2_lifts_siblings_through_parent`, `test_propagate_noop_without_view`, `test_propagate_unrelated_refs_stay_identity`, `test_propagate_zero_depth_is_noop`, `test_reset_noop_when_unallocated`, `test_reset_returns_to_identity` | 17.23: The symbol-space taxonomy and its attach, priming/configuration/lifecycle contract are removed. Conceptual taxonomy and numeric recommender tests remain. |
| `test/test_ternary_truth_store.py` | `TestOriginProvenance::test_append_defaults_conversation_origin`, `TestOriginProvenance::test_clear_origin_compacts_survivors_in_place`, `TestOriginProvenance::test_origin_rides_state_dict_texts_do_not`, `TestOriginProvenance::test_reset_clears_origin_and_texts`, `TestOriginProvenance::test_rows_of_origin_filters_in_row_order`, `TestOriginProvenance::test_set_origin_out_of_range_raises`, `TestOriginProvenance::test_set_origin_tags_row_and_text`, `TestTimestampAndTrust::test_set_trust_overwrites` | §17.12a,20–21: immutable trust, no origins/source text or origin withdrawal |
| `test/test_trust_gates.py` | `test_fixed_and_declared_root_operations_start_open`, `test_invalid_gate_data_fails_at_construction`, `test_nonroot_evidence_stays_capped_until_kept_ancestors_are_definite`, `test_open_gates_survive_checkpoint_and_epoch_and_missing_state_is_rejected`, `test_retired_group_configuration_fails_loudly` | 17.18: Operator gate state, thresholds, dependency graph and persistence are retired. The lesson readiness report remains. |
| `test/test_trust_round1.py` | `test_lexical_operator_is_retained_and_composed_at_the_read` | 17.17–18: The operator relation and operator confidence factor are retired; endpoint confidence is retained and independently certified. |
| `test/test_trust_round2.py` | `test_attachment_without_occupations_preserves_owner_catalogue_and_effective_pairs`, `test_first_composition_keeps_its_operation_when_another_merges_poles`, `test_fixed_operation_interfaces_have_learned_concept_authority`, `test_kept_lift_uses_cap_only_the_effective_pair`, `test_naming_clears_zero_use_operation_and_later_composition_cannot_restore_it`, `test_operation_counts_catalogue_column_and_pair_round_trip` | §17.18: operator seats/catalogue and operator scaling of a row trust go; learned argument membership tests remain |
| `test/test_truth_grounded_reasoning.py` | `TestEvaluate::test_is_true_false`, `TestEvaluate::test_is_true_true` | Points 7-11: a time-less row with no entry and no conceptual owner no longer lights a region by role-vector matching. The same positive/negative trust read-back assertions remain in TestIsTrue. |
| `test/test_word_grain_attention.py` | `test_primer_fires_retrieval_on_lift_family`, `test_real_lift_reverse_completes_under_priming` | 17.23: Symbol retrieval retired; plain reverse remains. |
| `test/test_word_space_attach_knowledge.py` | `test_attach_knowledge_stores_view`, `test_knowledge_is_none_before_attach`, `test_knowledge_view_queries_through_word_space`, `test_perceptual_space_attach_sets_wv_ref_ids`, `test_perceptual_space_attach_without_wv_is_noop`, `test_perceptual_space_inherits_attach_knowledge`, `test_perceptual_space_reattach_updates_ref_ids`, `test_reattach_replaces_previous_view`, `test_symbolic_space_inherits_attach_knowledge` | 17.23: The symbol-space taxonomy and its attach, priming/configuration/lifecycle contract are removed. Conceptual taxonomy and numeric recommender tests remain. |

**New functions (57).**

| File | Names |
|---|---|
| `test/test_eval_math_thinking.py` | `test_budget_and_ablation_comparisons_start_from_the_same_history` |
| `test/test_grammar_checkpoint.py` | `test_added_declaration_preserves_concept_identity_and_checkpoint_rows` |
| `test/test_identity_threading.py` | `test_three_point_roots_require_a_selected_relative_predicate` |
| `test/test_generated_extension.py` | `test_empty_extension_has_no_native_percept_demand`, `test_forced_interpretation_reads_the_explicit_operators_extension`, `test_generated_known_object_uses_its_declared_word` |
| `test/test_event_storage.py` | `test_event_uses_its_kept_result_after_a_verb_projects_the_relation`, `test_embedded_event_takes_its_sentences_authority`, `test_event_close_carries_its_own_operated_field` |
| `test/test_item18_extend.py` | `test_34_each_shipped_structural_operator_has_both_extension_faces`, `test_34_missing_extension_face_fails_at_load`, `test_35_missing_membership_is_zero_in_an_intersection`, `test_35_native_compose_journal_extends_in_order_without_stimulate`, `test_35_operator_sets_use_the_common_two_lane_logic`, `test_36_definite_reads_the_held_identity_and_both_its_lanes`, `test_36_determiners_read_the_stored_members_and_mtm`, `test_36_some_is_declared_in_the_complete_grammar`, `test_37_recalled_idea_extends_without_its_stm_sentence`, `test_38_query_operators_combine_row_results`, `test_39_cued_rows_preserves_extension_relevance_without_role_matching`, `test_39_open_operand_reads_only_the_same_generate_operator_path`, `test_39_recalled_generation_obeys_the_shared_allowance_before_walking`, `test_39_sentence_object_is_the_generated_operand_and_not_the_witness_row`, `test_40_deduction_carries_exact_premises_and_keeps_analogy_apart`, `test_40_pervasion_comes_from_the_kind_relation_not_the_members`, `test_40_question_form_applies_to_the_higher_order_proof`, `test_40_signed_existence_is_maximum_and_analogy_is_separate`, `test_40_thought_negation_transforms_its_existing_extension`, `test_56_kept_relative_tree_closes_in_both_relation_stores`, `test_public_existence_reads_the_question_extension_and_signed_trust` |
| `test/test_item18_reduction.py` | `test_23_13_promotion_without_actual_events_keeps_the_current_mtm_value`, `test_23_9_all_retained_occurrences_supply_exclusion_after_mtm_expires`, `test_37_41_a_kept_unrecorded_thought_writes_its_generated_extension`, `test_41_44_52_ideas_keep_flat_entries_and_immutable_signed_trust`, `test_41_relations_keep_only_pair_trust_and_activity`, `test_42_activity_drives_both_store_passes_and_permanent_rows_survive`, `test_42_activity_promotes_a_row_once_and_keeps_its_native_order`, `test_42_discarded_trial_reads_leave_activity_unchanged_and_selected_reads_apply_once`, `test_43_63_time_and_concept_with_the_same_integer_keep_their_native_orders`, `test_43_a_new_thought_advances_past_retained_occurrence_times`, `test_43_each_write_advances_past_a_stationary_model_present`, `test_43_kept_sentence_field_retains_its_occurrence_reference`, `test_43_time_named_relation_survives_other_occurrences_forgetting`, `test_46_checkpoint_keeps_the_current_mtm_link_before_its_close`, `test_47_ambiguous_legacy_times_are_rejected_without_loading`, `test_47_old_both_lane_idea_is_rejected_before_any_destination_changes`, `test_47_single_lane_legacy_trust_idea_time_and_entry_are_preserved`, `test_63_equal_member_extensions_keep_distinct_referents_and_native_orders`, `test_64_preposition_projects_and_recovers_its_object`, `test_pending_question_keeps_its_retained_context_point`, `test_recent_relation_head_cannot_alias_a_later_retained_time` |
| `test/test_item18_trust.py` | `test_50_authority_keeps_its_signed_trust_independently_of_model_trust`, `test_51_training_and_serving_keep_the_same_open_answer_and_different_authority`, `test_52_resubmission_appends_without_changing_trust_or_summing_existence`, `test_53_a_kept_both_lane_thought_writes_two_signed_occurrences`, `test_53_infer_writes_the_minimum_of_its_addressed_premises`, `test_54_truth_view_reads_nonzero_trust_and_reports_without_saved_sources`, `test_55_served_truths_run_the_activity_pass_before_writing_under_pressure` |
| `test/test_supplied_answer_training.py` | `test_output_reads_its_understanding_when_generation_publishes_a_different_field` |
| `test/test_tetralemma_leaf.py` | `test_path_ast_distinguishes_two_rows_from_netted_evidence` |

Shared helpers are ported in `chunk_fixtures`, `constituent_measurement`,
`definition_fixtures`, `existence_fixtures`, `identity_experiment`,
`identity_measurement`, `index_fixtures`, `math_chain_observer`,
`meaning_fixtures`, `memory_fixtures`, `operators_run_audit`,
`part_fixtures`, `reading_fixtures` and `tetralemma_audit`, together with the four isolated
SymbolSubSpace factories listed in §23.16. These preserve the original
native widths and capacities, and supply the working field/trace,
typed references, relation evidence or owner required by the production
interface. Historical captured fixtures under dated `test/fixtures`
directories are records of their old producers, not restored legacy
execution paths in the model.


### 23.22 Verification before the generated-extension repair

The source snapshot for these runs is
`758e91035eb748b5f8abe720e97dc1917f13e4c81f54a2bb394b207c68f6db22`.
It hashes the constructed source and test inputs by the existing bounded
runner's source inventory; it is not a commit. The checkpoint commit is
still `ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`. Receipts stay local.

**Certificates and stopping gates.** The source group covering
extension, recall, reduction, signed trust, the perceptual scalar,
the numeric certificates and both stopping gates returned **173
passed, 1 failed, 1 skipped, 1 warning in 95.59 seconds**;
**98.24501299858093 seconds** including its wrapper. The failure was
`test_operators_round3a::test_live_grammar_uses_sparse_index_dense_nonzero_roots_and_exact_rung0`:
PyTorch could not create its compiler lock under the sandboxed home
cache. With the compiler cache in a writable directory, that unchanged
test passed: **1 passed in 155.35 seconds**, **158.28671884536743
seconds** with the wrapper. Thus all **174 executed cases** pass across
the group and that environment-only follow-up. The skipped supervised
output-path test retains its old slow marker; its explicit finished-source
run is in §23.28. It was not deselected.
The direct group peaked at **4,346,302,592 bytes**.

| Stopping test | §14 | This source's certificate group |
|---|---|---|
| `test_mm_xor::TestMMXorConvergence::test_convergence` | Passed | Passed; call **.29556774999946356 s** |
| `test_item25_completion::test_a_previously_named_shared_concept_denotes_its_raised_class` | Passed | Passed; call **.14786550001008436 s** |

The numerical proof runs use the same native numeric inputs and field
path as §14, with no selected seed. All **10/10** return
**[-.018757708370685577, 1.0230909585952759,
1.0230909585952759, .016275040805339813]**, MSE
**.00042077834950760007**. The reader weights are
**[2.5449185371398926, 2.5449185371398926,
-3.7416553497314453]**. The promoted intersection is row **2**, with
edges **(2,0), (2,1), (2,42), (2,43)**. These answers, weights and MSE
are identical to §14. Native durations, in run order, are
**1.4241588750155643, 1.3751621250121389, 1.3538374589988962,
1.3422828750044573, 1.3800917079788633, 1.3783232079877052,
1.3942957500112243, 1.4163754159817472, 1.3686107500107028,
1.382861124991905 seconds**. The ten-run wrapper took
**45.49292612075806 seconds**.

**Certificate 13 as amended by 49.** All **24** paired presentations
completed in **192.52902345798793 seconds**, **196.98086595535278
seconds** with the wrapper. The three scalar word concepts are now
**153, 155 and 162**, each order zero, each with **6** views and
counted definition **.8646647334098816**, collecting basic percepts
**1, 4 and 7** respectively. The word heard without a felt value still
has **no object**. Omitted numbers are
**1.0000000123994952, 2.0000002496485365,
2.999999343408167**. With each word omitted, the emitted words are
respectively **one, two, three**, with those same numeric values.
Values **.5, 1.5, 2.5, 3.5 and 10** emit **no word**; exact values
**1, 2 and 3** emit their respective words. Neither dictionary bounds
nor `field_memberships` were restored. Recognition remains outside
this round under §17.5 and certificate 49; this is not a claim that
its postponed question path is built. §14 used the same 24
presentations and six views, objects **148, 150 and 152**, with its
native duration **17.774 seconds** and the same successful omissions.

The verbal proof on this source failed: **1 failed in 1116.99 seconds**,
**1119.9742701660143 seconds** under the process guard, with a peak of
**2,674,167,312 bytes**. The parallel field spelled all four sentences;
the generated field spelled none, leaving NUL bytes in all four. This
is the same producer defect isolated in §23.23, not an accepted result.

The lesson reached its **2400-second** process limit after **139 of
160 sentences**, with **139** available comparisons, **34** exact.
The guard recorded **2401.0207057499792 seconds** and a peak of
**1,794,247,104 bytes**. It is an incomplete measurement. Its rerun
keeps the same configuration and all 160 sentences, with an adequate
time limit. The default sweep was still running when these two results
were written; its completed result is recorded below.

### 23.23 The generated concept's stored extension

The cause of the verbal failure is at the generated-field producer.
An emitted order-one object can hold a word concept as a declared
member. `OutputOccurrences.generated` stimulated that object's point,
then passed the resulting object memberships directly to the native
percept inverse, which needs the word's memberships. It never applied
the known concept's stored extension required by §17 point 3.
Separately, `Extension.generated` filtered an unreferenced terminal's
stimulated field to the lexical row selected by readback scores. In the
failed proof, that selected word row had no membership while its
object did; the filter discarded the entire semantic field. A lexical
surface selection is not the concept's identity.

The diagnosis used the saved failed model without retraining or
altering parameters, emitted points, words, seed or target. Applying
the existing `Extension.stored` read to its realization memberships
restored **4/4** generated word multisets: **hello world, hello there,
world loving, there loving**. Its realization mass remained
**14.604293823242188**; the memberships moved to the declared word
members. The independent semantic filter was still present in that
diagnostic and is repaired at the same producer.

The native regression
`test_generated_known_object_uses_its_declared_word[meaning/word]`
constructs an ordinary word and an order-one object with that declared
word member. Its emitted point stimulates the object alone, and it
carries no reference. Before repair both cases fail in **2.74 seconds**:
the semantic field loses the member and the shared native inverse
emits five NUL bytes instead of **petal**. It uses no training, selected
seed, desired-output copy or resized existing fixture.

Two related boundary checks complete that producer repair. An empty
extension exposed `cs_percept_attribution`'s masked gather from row zero:
masking an absent row after the gather cannot work when there are no
rows. It now gathers only the present source rows, leaving all other
percept demands zero. `test_empty_extension_has_no_native_percept_demand`
fails before that correction with **IndexError**, **2.45 seconds**.
This preserves a value outside every scalar box as silent; it does not
insert a dummy concept.

Mandatory interpretation also differed from the explicit `interpret`
operator: it transported the word's memberships to its object but
omitted that object's stored extension. The direct parity certificate
`test_forced_interpretation_reads_the_explicit_operators_extension`
uses an admitted definition and the same **(.6,.2)** input memberships
for both forms. Before repair all **four** compared coordinates differ,
maximum **.6000000238418579**, in **2.20 seconds**. Both forms now use
the same declared word/object read followed by the same stored-member
read. This implements point 4's existing rule for `interpret`; it does
not add a new interpretation policy.

The producer and its consumers passed **211 tests, one warning, in
68.70 seconds** (guard **71.04861366699333 seconds**, wrapper
**71.09829092025757 seconds**, peak **4,957,115,120 bytes**) on source
`cca4c9a363b3daef810c2e091eb842076fe62c5471da63502ec32e1d67598bcd`.
The remaining certificate group gave **115 passed, two failed, one
skipped, one warning in 207.70 seconds** (guard **211.12194224999985
seconds**, peak **4,292,563,072 bytes**). Both stopping gates passed:
numeric XOR **.24742133400286548 seconds**, the named shared class
**.11512883400428109 seconds**. The two failures were certificate 51's
obsolete singleton-reference assertions. A generated open operand can
contain several active concepts; the removed lexical-score filter had
manufactured its singleton. The port keeps the original member's exact
memberships and all authority/witness assertions, and checks the whole
answer against that generated operand. Both parameters then passed in
**5.53 seconds**, wrapper **7.824910879135132 seconds**. This test alone
changes the source hash to
`a28e7469cfee65541132b346461186729062297a18cca463c2a671b0be8dbf64`;
production and configuration files are identical between the two hashes.

### 23.24 Further verification and the event close (2026-11-14)

The complete default sweep preceding §23.23, on source
`758e91035eb748b5f8abe720e97dc1917f13e4c81f54a2bb394b207c68f6db22`,
completed **7,881 distinct cases**, nothing deselected: **7,540 passed,
323 skipped, 17 failed and one non-strict XPASS**. There are three
additional passing reports of an already-counted flag case. Native time
was **2420.523731583002 seconds**, wrapper **2421.113704919815 seconds**,
peak **11,237,979,504 bytes**. Both stopping gates passed. Thirteen
failures came from `test_structural_checkpoint._model_with` replacing a
real registered codebook module with a `SimpleNamespace`. The helper
now supplies a mock only where the fixture has no native module or MTM;
its consumers are in §23.23's 211 passing cases. The verbal proof failed
at the generated-extension producer repaired there. The other three
failures were the older time-reader inventory mismatch, the older
basic-percept feature-column assertion, and
`test_reasoning_cde_model.py::test_answer_query_degrades_gracefully`,
whose producer raised **“tied reconstruction bank was not staged after
concept admission”**. The earlier-source attribution of that intermittent
reasoning failure still stands; it is not repaired in this round.

The first lesson after §23.23 stopped after **44 compared sentences**
when `ReferenceClose.link_events` passed a stored event value to
`MemoryLink.link` and its containment guard rejected it. Time was
**193.9321136669896 seconds** under the guard, wrapper
**193.99253296852112 seconds**, peak **1,469,892,840 bytes**. A subsequent
passively instrumented run, with its actual unchosen initial random
stream recorded, completed **160 sentences** in **1149.9582421249943
seconds** (wrapper **1150.022737979889 seconds**, peak
**1,512,557,016 bytes**). Its **160** field comparisons had **two exact**,
maximum absolute difference **1**, and coordinate-weighted mean absolute
difference **.36379834425569274** over **11,732 coordinates**; it made
**121,712** compound-kernel tests. That later pass does not establish a
repair of the earlier failure, whose initial stream was not saved.

The event writer has a separately reproduced regression. The link menu
uses the kept identity event's `result`; the writer used to retain that
result when the enclosing statement closed as a relation. After point
17 removes an operator relation, the statement can close as an idea,
and the writer reuses that different idea for the event. A forced native
`verb(part(...), ...)` reading reproduces it: all **eight** coordinates
of the event change, by up to **5.960464477539063e-8**. The exact-state
certificate fails on the current writer in **2.25 seconds**, passes on
the checkpoint writer in **2.75 seconds**, and passes with the candidate
repair in **1.77 seconds**. The small difference matters to exact
containment; its guard is not relaxed. A second controlled case with
distinct result and statement coordinates also fails before and passes
after. These establish the producer defect; they are not a replay of
the failed lesson's unsaved initial stream.

The repair being checked reserves and writes the event's own idea when
it differs from the statement, retaining the existing event address
handoff. Its kept result constituent also supplies its own operated
field to the ordinary entry writer. Two adjacent errors are covered:
an embedded event's pair omitted the sentence's authority multiplier,
and an event close dropped that constituent's extension. The zero-trust,
fractional-trust and field-transfer checks fail **three in 2.48 seconds**
before and pass **three in 1.98 seconds** with the candidate. This is
the point-12a source rule for event writes and point 12's entry rule;
neither adds a new trust source or a second field computation.

The current sweep also exposed a remaining port in
`test_episodic_concepts.py::test_recalled_flat_extension_decodes_directly_and_retired_ids_are_counted`.
Replaying its captured failing stream identifies one sentence's
`PartClosing._write_occurrence` as the producer of **two** identical
entries, one row per nonzero trust lane. Point 12a retires the assertion
that this sentence must occupy exactly one row. The port checks the
memberships and directly decoded surfaces of **every returned row**, in
returned order; its missing-surface, skipped-retired-ID and no-grammar
assertions remain. The original captured stream fails in **2.14
seconds** and passes the port in **2.04 seconds**. No stream is selected
for the completion run, and no row is hidden by reducing `max_records`.

The broader candidate check gave **226 passed, four skipped, one failed
and one warning in 162.20 seconds** (guard **166.31610558298416 seconds**,
peak **3,557,398,072 bytes**). Its failure was the fixed-buffer assertion
in `test_item25_repair.py::test_determined_mention_keeps_an_occurrence_after_its_relation_was_written[2]`.
The earlier fixture port had put this LTM capacity test on `STM0`, whose
functional writes preserve the pinned tree's live gradients. Accidental
allocator reuse could make its pointer assertion pass, but did not
establish fixed storage. The port now uses the existing native-LTM option
with the time band inside the same **eight-coordinate width** and the
same **64-row capacity**. Occupied seats are written with `append_idea`,
not fabricated by changing `count` while leaving all their times zero.
It keeps the fixed pointers, shapes, capacity, atomic rejection, reference
and content checks, and checks the two new times separately. Both cases
pass in **1.91 seconds**. The event's value repair changes no STM storage
rule. `Extension.stored_field` is also added to the time-reader inventory;
the older radix-method name mismatch is left alone.

The five final event certificates, including both `part` and `equal`,
fail on the unchanged writer in **2.47 seconds**. The diagnostic copy
beside candidate source files is not used as the before result: pytest's
path insertion could import that candidate. The recorded native-before
run imports the main checkout and fails all five. The repairs and ports
above are now applied to the main checkout, with no commit or push.

### 23.25 Verification on the event-write repair

The source for these runs is
`2d607968c987952cf7f07ff45832d152fac5b0771017f34dc2a4ad081d894218`.
The combined certificate/producer group passes **198 tests**, with
**one skip and one warning**, in **221.03 seconds** (guard
**224.21647195797414 seconds**, wrapper **224.28899312019348 seconds**,
peak **4,303,196,336 bytes**). The skip is the existing slow-marked
output proof; its explicit finished-source run is recorded in §23.28.
It is not a deselection from the default sweep.
The warning is an existing tensor-to-float read in `Examination`.
Both stipulated stopping gates pass. No threshold, fixture capacity,
training schedule or seed was changed to obtain that result.

| Stopping gate | §14 returned source | This source's certificate run |
|---|---|---|
| `test_mm_xor.py::TestMMXorConvergence::test_convergence` | Passed; individual duration not recorded there. | Passed, **.25099845899967477 seconds**. |
| `test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class` | Passed; individual duration not recorded there. | Passed, **.11073637500521727 seconds**. |

The second gate has a structural denotation/order assertion, not an MSE.
The numeric proof's measured values are compared with §14 below.

**Certificate 11, ten numeric runs.** Every run promotes row **2**,
with edges **(2,0), (2,1), (2,42), (2,43)**. Every run's answers are
**[-.018757708370685577, 1.0230909585952759,
1.0230909585952759, .016275040805339813]**, with MSE
**.00042077834950760007** and reader weights
**[2.5449185371398926, 2.5449185371398926, -3.7416553497314453]**.
Thus the numeric results are unchanged from §14. Individual native
times are **1.2020255000097677, 1.2644987090025097,
1.2191221250104718, 1.2520582920114975, 1.2320013749995269,
1.2299825839872938, 1.2452867090178188, 1.2043082500167657,
1.201154124981258 and 1.251486458000727 seconds**. Guard time is
**42.81715754198376 seconds**, wrapper **42.874789237976074 seconds**,
peak **582,109,200 bytes**. These are CPU measurements, with no selected
seed. The first launch omitted the CPU setting and failed at MPS's
unsupported float64 conversion in `NumericPercepts.feeling`, before a
numeric result; it took **4.695007085800171 seconds** including its
wrapper. No change was made to the numeric producer.

**Certificate 13 as amended by 49.** The native acquisition and both
modality omissions pass, **24 presentations**, in
**99.39743079201435 seconds** (guard **103.5737993750081 seconds**,
wrapper **103.63114070892334 seconds**, peak **2,284,210,720 bytes**,
CPU). `one`, `two` and `three` acquire objects **153, 160 and 167**,
all order zero, each from **six** agreeing views with definition
**.8646647334098816** and basic percepts **1, 4 and 7**, respectively.
`unfelt` acquires no object. With the number omitted, the recovered
numbers are **1.0000000123994952, 2.0000002496485365 and
2.999999343408167**. With the word omitted, the corresponding readings
emit **one, two and three** and the same numbers. Exact probes **1, 2,
3** emit those words; **.5, 1.5, 2.5, 3.5 and 10** emit nothing. The
values and membership outcomes match §14. The amended certificate does
not claim recognition, which §17.5 reserves for the following round.

For comparison, the complete sweep immediately before the event repair,
on `a28e7469cfee`, completed **7,885 distinct cases**, nothing deselected:
**7,558 passed, 323 skipped, three failed and one non-strict XPASS**.
Its three extra passing reports are duplicate flag-case reports. Native
time was **2428.2336842910154 seconds**, wrapper **2428.710644006729
seconds**, peak **10,784,159,856 bytes**. Both stopping gates passed,
numeric XOR in **.2657980829826556 seconds**, named shared class in
**.12687058400479145 seconds**. Its fresh verbal XOR proof passed in
**804.64 seconds**: **four of four** generated word multisets and
**four of four** parallel-field word multisets, with none unavailable.
The three failures were the two-row recall assertion ported in §23.24,
the time-reader inventory (the newly added transport plus the older
radix mismatch), and the older `{0}` versus `{4,402}` feature-column
assertion. The intermittent reasoning case passed in this sweep.

### 23.26 A held-answer test's output clock

The next sweep exposed
`test_prepared_answer_boundary.py::test_realization_uses_a_held_derivation_without_running_resolution[True]`.
This is a test port under point 15, not a change to answer realization.
The test asks for the same prepared answer twice, with another input
between, and compares all output coordinates at the same output tick.
It restored `model.when_time` and the encoding clock, but not the retained
history. `Models._advance_when_time` now advances past that history's
next tick, as required for uniquely named occurrences. The two outputs
therefore used **18** and **36**, even though the test reset the model
clock to **2** before each. The six unequal coordinates are exclusively
**356, 358 and 359** in the two batch rows, inside the time band starting
at **356**; maximum difference **.013804741203784943**. No answer content
or resolution read changed.

The captured failing random stream reproduces the failure in **5.74
seconds**. The unchanged checkpoint producer passes in **6.11 seconds**.
The port explicitly gives both calls the existing `_produced_when`
context for trials of one produced sentence, alongside the same model
tick. It retains exact equality of both `actual` and `concepts`, the
held-derivation identity assertions, and the prohibitions on query,
resolution and lexical rescanning. Both variants pass on that captured
stream in **10.48 seconds**. The shipped test does not load that stream,
choose a seed, resize a fixture or override a production function.

The sweep on `2d607968c987` is a diagnostic, not the required final
sweep: it was stopped after **7,889 of 7,890** distinct cases, with its
fresh verbal XOR phase still running, so that the verified clock port
can enter the finished-source run. Its wrapper took
**1801.122803926468 seconds**. Its three failures were the clock fixture
above and the two older inventory/feature-column assertions. Both
stipulated stopping gates passed. No case is deselected from the
replacement sweep.

The final completed-case snapshot of that diagnostic was at
**1692.7931557499978 seconds**, with peak aggregate memory
**13,408,385,808 bytes**. The clock port is now in the main checkout.
Comparing the source manifests before and after it finds exactly one
changed input, `test/test_prepared_answer_boundary.py`; no production,
configuration or measurement source changed. The finished source is
`20011b579c034c1a2962d8e784af7be8164f84c5b5184b3ac5527800e7148f46`.

### 23.27 The completed lesson before the clock test port

The event-repair source `2d607968c987` completed all **160 sentences** of
`explore-1.xml` with the repaired observer: **160 comparisons**, **zero
unavailable**, **zero exact**, **15,274** compared lane coordinates,
maximum absolute difference **1**, and coordinate-weighted mean absolute
difference **.4357900556093488**. There were **606 generate calls**, **320
thought episodes**, and **152,355 compound kernel tests**. Native
sentence time totals **2338.3894758790266 seconds**; guard time is
**2406.7264714589983 seconds**, wrapper **2406.8097200393677 seconds**,
peak **1,597,344,264 bytes**. No event-link failure occurred. This is a
measurement of the learned fields' difference, not an equality proof.

The earlier launch using an already occupied lesson-output directory
failed at directory creation, before constructing a model, in
**3.8548666659917217 seconds** (wrapper **3.903939962387085 seconds**,
peak **405,423,424 bytes**). That directory's 139 rows were an older
interrupted run, not new progress. The completed run above used a fresh
directory. Its unchosen initial random stream was recorded for diagnosis;
no stream was selected or replayed to make this measurement pass.

For the finished-source verification the complete lesson and the exact
default sweep ran again, both starting fresh; their results are below. Five
temporary migration diagnostic files from this task were compressed to make
room for the sweep. Their original bytes were verified by SHA-256 after
decompression; all diagnostic evidence remains local. This reclaimed
**2,344,830,605 bytes** and changed no source input.

### 23.28 Finished-source certificates 11, 13/49 and the reduction

These measurements use source
`20011b579c034c1a2962d8e784af7be8164f84c5b5184b3ac5527800e7148f46`,
over **1,070** source/configuration/test inputs. The certificate and
producer group completed **205 passed, two skipped, one warning**, in
**193.82 seconds**; guard **196.9602468330122 seconds**, wrapper
**197.03166389465332 seconds**, peak **4,439,167,224 bytes**. This includes
**34** extension/recall cases, **24** reduction cases, **14** trust cases,
the **five** event-write and **four** generated-extension regressions,
and both variants of the held-answer clock port. The two skips are the
existing slow-marked tests
`test_output_path_supervised::test_output_path_memorizes_supervised_labels_on_the_parallel_topology`
and `test_prepared_answer_boundary::test_what_reconstructs_then_resolves_then_realizes`.
They are not newly deselected tests. The one warning is the existing
`Examination` tensor-to-float read described in §23.25.

The output proof still carries its old `slow` marker although it already
uses the numeric certificate. It was therefore run explicitly with
`RUN_SLOW=1` on the same finished source: **one passed in 2.34 seconds**,
guard **4.128504166990751 seconds**, wrapper **4.1895060539245605 seconds**,
peak **559,285,784 bytes**, at its unchanged **100%** classification bar.
No source, assertion or fixture changed for that run. The other skipped
test is a separate slow integration check, not an XOR proof. The requested
default sweep retains its unset `RUN_SLOW` and `RUN_TRAINED` environment.

| Stopping gate | §14 | Finished-source certificate call |
|---|---|---|
| `test_mm_xor.py::TestMMXorConvergence::test_convergence` | Passed; no individual duration recorded. | Passed, **.25672554099583067 seconds**. |
| `test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class` | Passed; no individual duration recorded. | Passed, **.11119670901098289 seconds**. |

**Ten numeric runs, certificate 11.** Each run uses `data/XOR_exact.xml`
on CPU with no selected seed. In every run the answers are
**[-.018757708370685577, 1.0230909585952759, 1.0230909585952759,
.016275040805339813]**, MSE **.00042077834950760007**, reader weights
**[2.5449185371398926, 2.5449185371398926, -3.7416553497314453]**,
promoted row **2**, and edges **(2,0), (2,1), (2,42), (2,43)**. These
values equal §14's. The ten native durations, in run order, are
**1.3960807920084335, 1.3836440829909407, 1.4417221659969073,
1.3903684580000117, 1.3529600409965497, 1.4172482500143815,
1.4333179160021245, 1.3468404169834685, 1.3939497500250582 and
1.3581987090001348 seconds**. Guard **46.64640941697871 seconds**,
wrapper **46.70980405807495 seconds**, peak **582,158,280 bytes**.

**Scalar acquisition and omission, certificate 13 as amended by 49.**
The **24 presentations** pass in **15.624544499994954 native seconds**,
guard **19.670520540996222 seconds**, wrapper **19.728940963745117
seconds**, peak **1,107,003,144 bytes**. The CPU run chooses no seed.
`one`, `two` and `three` acquire objects **148, 150 and 152**, each at
order **zero**, from **six** views, each definition
**.8646647334098816**, with basic percepts **1, 4 and 7**. `unfelt`
acquires no object. With the number omitted, the values read back are
**1.0000000123994952, 2.0000002496485365 and 2.999999343408167**.
With the word omitted, the corresponding readings emit **one, two and
three** and those same values. Probes **1, 2, 3** emit those words;
**.5, 1.5, 2.5, 3.5 and 10** emit nothing. These membership and omission
outcomes equal §14's. Recognition is not claimed: §17.5 leaves it for
the following round. No threshold, capacity or training factor was
changed to recover either proof or either stopping gate.

### 23.29 The implemented items and their before/after certificates

This table collects the completion by §18 item. The detailed failing
producers, decisions and intervening runs above remain the record; a
missing new interface on the old checkpoint is not counted as a separate
old defect for each assertion that encounters it. Every after-check named
here is included in the finished-source group in §23.28 or the default
sweep reported below.

| Item | Cause and implemented behavior | Before and after |
|---|---|---|
| 1: extension, 34–37 | Leaf fields did not express the kept operators' selection. `Extension` now supplies both faces, consumes the kept places' already-open activations, combines by `FieldLogic`, and reads declared members for kinds and generated known objects. `some` is present. | The original 24 extension/recall cases all failed on the checkpoint, **4.86 s**, including its absent interfaces (§21.5). The real MTM member and mandatory/generated-interpretation regressions also exposed their respective reader defects (§21.2, §23.23). All **34** current extension cases and **four** generated-extension cases pass. |
| 2: recall and answers, 38–40 and 56 | Role re-scoring and a fixed meet could not implement the question's tree. Recall now combines operand results by that tree; an open answer comes from the matching generated operand. Closed claims retain existence, exact-content support and higher-order pervasion separately. | The checkpoint's 24-case before group covers 38–40. The selected `part`/`equal` writer already wrote both relation stores; its proof had read an operator-gated effective pair, so it gave zero instead of **.7**. The stored pair now gives **.7** through the **.8** membership edge. All four signed selected-tree cases pass. Negative-proof, bounded-generation and acquired-evidence failures and passing certificates are detailed in §23.15. |
| 3a: activity, 42 | Counts, surprise columns and the retention formula were separate persistent mechanisms; trial reads also escaped restoration. One bounded activity drives writes, accesses, passes and once-only promotion. Promotion writes the concept referent. | The cache-activity check failed before (§23.4); the once-only promotion and discarded-trial certificates now pass. The two histories of §23.13 failed the new rule before in **2.43 s** and pass using only actual MTM events (§23.14). All-retained-occurrence exclusion follows Alec's separate decision and has its own failing-before/passing-after certificate. |
| 3b: time and one history, 43/47/63 | Old addresses and stream columns carried identity independently of time; untyped zero and relation heads then aliased concepts or later occurrences during the reduction. Durable occurrences now have typed, increasing time names in one history. | The initial time/rejection group failed **three cases in 2.33 s**, then passed **three in 2.31 s** (§21.8). Shared-clock, time-zero/order and recent-relation-head regressions now pass. Old ambiguous times and old BOTH ideas are rejected atomically under Alec's decisions; unambiguous single-lane checkpoints preserve their idea, time, trust and strongest entry. |
| 3c: operator relations and gates, 45 | The old verb-over-relation representation required an operator pair and gated trust. The grammar is now open; a completed relation under a verb closes as an idea. | The preserved clause/grammar assertions now check this representation. The event result had then been lost when that outer close became an occurrence: **five event certificates failed in 2.47 s** before the producer repair (§23.24); all **five pass**. Each event retains its exact selected result, own operated field and sentence authority, with the containment guard unchanged. |
| 3d: columns, sidecars and pending state, 41/44/46/47/50–55/63 | Provenance, kinds, source text, covers, recurrence, forecasts and durable pending bindings supplied readers that had to be retired or moved. Signed immutable trust replaces LTM0's pair, and the selected referent preserves exact denotation and order independently of the flat extension. | The old relation-column and idea-column certificates each failed before the rewrite (§21.8), then passed after. The current-link checkpoint failed in **2.30 s** before its reader repair; its lossless restore and rejection checks now pass (§23.14). The complete **24-case reduction** and **14-case trust** files pass, including face-value authority, zero-trust conversation, premise minima, BOTH as two rows, immutable resubmission, contradiction reporting and the serving pass. Reader destinations are in §23.17; cache and size declarations in §23.18. |
| 3e: the flat entry, 37/41/48 | The previous order axis and unconditional vector-side entry producer did not hold the operated field. The entry takes the strongest activations across all orders from the kept understanding field; an unrecorded kept thought obtains its extension from its existing generate walk. | The flat/signed-row check failed on §21's source and passes after the rewrite. The unrecorded-thought entry and retained sentence-reference checks failed during reduction and now pass. The two-row recall port keeps all surviving membership and decoding assertions (§23.24). Stored sizes are **5,514 → 5,096 bytes per row**, with the full capacity calculation in §23.18. |
| 4: test contracts | Existing assertions called deleted writers, assumed old columns/addresses, or bypassed the live state now held in MTM. Surviving contracts use native writers and state; removed contracts are retired with their point of §17. | §23.21 names all **463 ports**, **285 whole retirements** and **60 additions**. The held-answer port's captured failure changed only six time coordinates; both variants pass with the native shared output tick (§23.26). No fixture is enlarged and no seed is selected. |
| 5: finished-source verification | The proof bars and original lesson construction stay. The lesson observer reads the selected path after restoration and compares the operated fields. | The certificate group has **205 passes** (§23.28); all ten measured numeric runs retain MSE **.00042077834950760007**; both scalar omissions pass; the **160-sentence** lesson's full differences are in §23.30; the fresh verbal proof and complete, undeselected default sweep are in §23.31. |
| 6: unused structure and preposition, 64 | The symbol-only taxonomy, unread chain rows and test-only writers are deleted under point 23. The preposition kernel now takes the object as head, operand 1, matching the close. | The exact preposition certificate copied to verified checkpoint code fails **one test in 2.77 s**: all **eight of eight** content coordinates differ, maximum **.40000003576278687**. The unchanged certificate passes on the finished source, including reverse recovery of the object. Surviving tests of the deleted writers use production membership/Part writers; their named ports and retirements are in §23.21. |

The limits accepted in §22.3 are carried unchanged: the selected-tree
relation and controlled open-operand inverse checks do not prove that an
untrained chooser learns a multioperator parse; the complete learned
negative-sentence path and multiple-open-operand thought integration are
not certified. Point 11 and unchanged `Intuition.pervasion` deliberately
give different readings on the same question, with the numbers in
§23.17. The lesson measures the learned compose/generated field
difference. Nothing from §17.5, §18.7 or §3.19 was built.

### 23.30 Finished-source whole-path lesson

The fresh `explore-1.xml` arm, **160 sentences**, completed on source
`20011b579c034c1a2962d8e784af7be8164f84c5b5184b3ac5527800e7148f46`.
All **160** kept sentences have a comparison, **zero unavailable**;
**0** are exact. There are **12,174** compared lane coordinates,
maximum absolute difference **1**, and coordinate-weighted mean
absolute difference **0.41331473015663717**. The run made **767 generate calls**,
**320 thought episodes**, and **116,346 compound kernel tests**.
Native sentence time totals **1811.2357001320051 seconds**; guard
**1878.3077352920081 seconds**, wrapper **1878.3709452152252 seconds**,
peak **1,695,599,016 bytes**. No event-link guard failed.

The observer compares the kept understanding's `compose_extend` field
with the captured field of its generated idea after the native trial
restoration. It aligns references, puts zero on an absent reference and
compares the for and against lanes separately. No seed was chosen, no
fixture or schedule was resized, and the containment guard remains.
Completion of the arm is not a claim that the learned fields agree.
The differences below remain a measured limit of this lesson; the
controlled operator/inverse certificates do not remove that limit.
§14's **54 exact** comparisons used the former fields placed side by
side; they are not this operator-combined comparison. These runs also
shared the CPU with verification, so their wall times do not isolate the
architecture's performance cost.

| Epoch | Sentences | Exact | Lane coordinates | Maximum difference | Weighted mean difference | Native seconds |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 0 | 2862 | 1 | 0.405797029 | 340.801337 |
| 2 | 40 | 0 | 3034 | 1 | 0.397400643 | 396.919713 |
| 3 | 40 | 0 | 3038 | 1 | 0.409604718 | 502.733067 |
| 4 | 40 | 0 | 3240 | 1 | 0.43833634 | 570.781582 |

Per-sentence differences are written below, rounded to nine significant
digits; the aggregate above uses the recorded values. `Kernel tests` is
the cumulative count. The local raw receipts are retained and not linked.

| Sentence | Epoch | Generate calls | Thought episodes | Native seconds | Lane coordinates | Max difference | Mean difference | Exact | Kernel tests |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 1 | 3 | 2 | 1.09957096 | 26 | 0.5 | 0.19484967 | no | 291 |
| 2 | 1 | 5 | 2 | 3.48732104 | 36 | 0.96825397 | 0.313724756 | no | 1907 |
| 3 | 1 | 6 | 2 | 2.64400462 | 48 | 1 | 0.352121502 | no | 2494 |
| 4 | 1 | 5 | 2 | 4.12553508 | 48 | 1 | 0.314760417 | no | 3498 |
| 5 | 1 | 3 | 2 | 2.10981071 | 48 | 1 | 0.313477129 | no | 3865 |
| 6 | 1 | 6 | 2 | 3.41617683 | 48 | 0.992009997 | 0.418750435 | no | 5152 |
| 7 | 1 | 5 | 2 | 3.37261717 | 48 | 0.991490602 | 0.312111348 | no | 5954 |
| 8 | 1 | 6 | 2 | 4.19252142 | 32 | 0.999892652 | 0.29677856 | no | 7237 |
| 9 | 1 | 3 | 2 | 2.34681492 | 48 | 0.99221617 | 0.35949102 | no | 7601 |
| 10 | 1 | 5 | 2 | 3.96319729 | 48 | 1 | 0.348231077 | no | 8355 |
| 11 | 1 | 5 | 2 | 4.64726163 | 48 | 0.921071231 | 0.348334581 | no | 9295 |
| 12 | 1 | 5 | 2 | 4.26281475 | 48 | 1 | 0.440418094 | no | 9980 |
| 13 | 1 | 4 | 2 | 2.95693946 | 48 | 0.921049893 | 0.204403326 | no | 10551 |
| 14 | 1 | 5 | 2 | 3.46164071 | 48 | 1 | 0.419281453 | no | 11114 |
| 15 | 1 | 6 | 2 | 4.28549471 | 48 | 0.921039045 | 0.204579279 | no | 11699 |
| 16 | 1 | 6 | 2 | 4.60151925 | 48 | 1 | 0.428066254 | no | 12262 |
| 17 | 2 | 4 | 2 | 3.4899815 | 48 | 0.92103833 | 0.204083681 | no | 12842 |
| 18 | 2 | 6 | 2 | 4.8437915 | 48 | 1 | 0.404095888 | no | 13451 |
| 19 | 2 | 5 | 2 | 4.01307742 | 48 | 0.921041131 | 0.205681935 | no | 14033 |
| 20 | 2 | 4 | 2 | 3.75938937 | 48 | 1 | 0.363179356 | no | 14532 |
| 21 | 2 | 4 | 2 | 3.65808467 | 48 | 0.921043813 | 0.205692992 | no | 15096 |
| 22 | 2 | 6 | 2 | 4.28438667 | 48 | 1 | 0.483314395 | no | 15924 |
| 23 | 2 | 6 | 2 | 5.23091379 | 48 | 0.921051323 | 0.326721638 | no | 16553 |
| 24 | 2 | 5 | 2 | 5.16842483 | 48 | 1 | 0.438013554 | no | 17255 |
| 25 | 2 | 4 | 2 | 4.03258867 | 48 | 1 | 0.451782972 | no | 17807 |
| 26 | 2 | 5 | 2 | 6.1226185 | 48 | 1 | 0.411554813 | no | 18565 |
| 27 | 2 | 5 | 2 | 5.24447312 | 48 | 1 | 0.45623675 | no | 19209 |
| 28 | 2 | 6 | 2 | 8.14223075 | 48 | 1 | 0.457345098 | no | 20019 |
| 29 | 2 | 4 | 2 | 4.96499929 | 34 | 0.921358287 | 0.284536004 | no | 20610 |
| 30 | 2 | 5 | 2 | 6.55468071 | 48 | 1 | 0.450612396 | no | 22247 |
| 31 | 2 | 5 | 2 | 5.36424779 | 48 | 1 | 0.452127576 | no | 22833 |
| 32 | 2 | 4 | 2 | 4.680431 | 48 | 0.844917119 | 0.174417838 | no | 23709 |
| 33 | 3 | 4 | 2 | 5.02121433 | 34 | 0.921528459 | 0.281033814 | no | 24265 |
| 34 | 3 | 6 | 2 | 6.58459804 | 48 | 0.801085889 | 0.262337178 | no | 24956 |
| 35 | 3 | 6 | 2 | 5.77215379 | 34 | 0.921687901 | 0.286555737 | no | 25540 |
| 36 | 3 | 5 | 2 | 8.00735333 | 48 | 1 | 0.412725449 | no | 26488 |
| 37 | 3 | 4 | 2 | 5.92125896 | 34 | 0.999167919 | 0.429342508 | no | 27074 |
| 38 | 3 | 5 | 2 | 6.72317613 | 48 | 1 | 0.446804494 | no | 27645 |
| 39 | 3 | 4 | 2 | 4.93688992 | 48 | 1 | 0.467907637 | no | 27938 |
| 40 | 3 | 6 | 2 | 7.13670187 | 30 | 0.844696403 | 0.197095975 | no | 29555 |
| 41 | 3 | 4 | 2 | 7.1279495 | 48 | 0.997817814 | 0.44220987 | no | 30218 |
| 42 | 3 | 6 | 2 | 8.96739979 | 48 | 1 | 0.42630896 | no | 31068 |
| 43 | 3 | 4 | 2 | 5.65372879 | 48 | 1 | 0.452778488 | no | 31391 |
| 44 | 3 | 4 | 2 | 5.89082725 | 48 | 1 | 0.415390998 | no | 31742 |
| 45 | 3 | 4 | 2 | 8.20883942 | 48 | 1 | 0.457871556 | no | 32407 |
| 46 | 3 | 6 | 2 | 9.52412808 | 48 | 1 | 0.456828386 | no | 33092 |
| 47 | 3 | 5 | 2 | 8.48719146 | 34 | 0.999165177 | 0.429398149 | no | 33748 |
| 48 | 3 | 5 | 2 | 9.29640079 | 48 | 1 | 0.44558382 | no | 34364 |
| 49 | 4 | 4 | 2 | 8.43507383 | 34 | 0.999164581 | 0.429404438 | no | 34955 |
| 50 | 4 | 5 | 2 | 7.8109345 | 48 | 1 | 0.40378812 | no | 36044 |
| 51 | 4 | 6 | 2 | 9.85614512 | 34 | 0.922205806 | 0.291345417 | no | 36633 |
| 52 | 4 | 6 | 2 | 10.3118374 | 48 | 1 | 0.451293737 | no | 37200 |
| 53 | 4 | 4 | 2 | 9.10916429 | 48 | 1 | 0.456056982 | no | 37754 |
| 54 | 4 | 4 | 2 | 7.38274133 | 48 | 1 | 0.457101822 | no | 38064 |
| 55 | 4 | 5 | 2 | 10.2439192 | 34 | 0.999162972 | 0.429416031 | no | 38720 |
| 56 | 4 | 5 | 2 | 8.96435217 | 48 | 1 | 0.442289829 | no | 39870 |
| 57 | 4 | 4 | 2 | 8.15109662 | 48 | 1 | 0.504440546 | no | 41019 |
| 58 | 4 | 4 | 2 | 7.99963537 | 48 | 1 | 0.449858755 | no | 41370 |
| 59 | 4 | 6 | 2 | 12.3178628 | 48 | 1 | 0.387033314 | no | 42145 |
| 60 | 4 | 5 | 2 | 9.48608079 | 58 | 1 | 0.413645715 | no | 43279 |
| 61 | 4 | 4 | 2 | 11.7220706 | 58 | 0.999287426 | 0.436529934 | no | 44013 |
| 62 | 4 | 5 | 2 | 11.7034652 | 58 | 1 | 0.459631741 | no | 44633 |
| 63 | 4 | 5 | 2 | 12.1996002 | 44 | 0.922165453 | 0.268771112 | no | 45279 |
| 64 | 4 | 6 | 2 | 13.3003058 | 58 | 1 | 0.451949686 | no | 46139 |
| 65 | 1 | 4 | 2 | 12.256682 | 76 | 1 | 0.445863068 | no | 46798 |
| 66 | 1 | 4 | 2 | 8.95420262 | 90 | 1 | 0.30718866 | no | 47606 |
| 67 | 1 | 6 | 2 | 13.1974983 | 74 | 0.997833371 | 0.355638444 | no | 48212 |
| 68 | 1 | 4 | 2 | 8.71321188 | 90 | 1 | 0.42899105 | no | 48523 |
| 69 | 1 | 4 | 2 | 8.46922292 | 72 | 1 | 0.45497793 | no | 48909 |
| 70 | 1 | 5 | 2 | 11.9987011 | 94 | 1 | 0.439207852 | no | 49587 |
| 71 | 1 | 4 | 2 | 9.48286637 | 94 | 1 | 0.449454486 | no | 50738 |
| 72 | 1 | 4 | 2 | 8.29593233 | 72 | 0.997825384 | 0.36526376 | no | 51032 |
| 73 | 1 | 5 | 2 | 13.0555101 | 94 | 1 | 0.427862614 | no | 51820 |
| 74 | 1 | 5 | 2 | 13.746947 | 94 | 1 | 0.474679351 | no | 52548 |
| 75 | 1 | 6 | 2 | 14.0421617 | 94 | 1 | 0.444405198 | no | 53312 |
| 76 | 1 | 5 | 2 | 14.0341415 | 94 | 1 | 0.452625722 | no | 54040 |
| 77 | 1 | 3 | 2 | 9.01596675 | 94 | 1 | 0.44180575 | no | 54374 |
| 78 | 1 | 6 | 2 | 14.2366014 | 94 | 1 | 0.315313905 | no | 55160 |
| 79 | 1 | 5 | 2 | 14.1384155 | 94 | 1 | 0.456232935 | no | 55978 |
| 80 | 1 | 5 | 2 | 14.0011294 | 94 | 1 | 0.469667435 | no | 56616 |
| 81 | 1 | 5 | 2 | 14.6795417 | 72 | 1 | 0.465244651 | no | 57380 |
| 82 | 1 | 5 | 2 | 14.050055 | 98 | 1 | 0.445665509 | no | 58118 |
| 83 | 1 | 4 | 2 | 9.73786546 | 98 | 1 | 0.396848798 | no | 59270 |
| 84 | 1 | 5 | 2 | 9.94748646 | 98 | 1 | 0.464312613 | no | 60377 |
| 85 | 1 | 5 | 2 | 10.9213414 | 98 | 1 | 0.45538941 | no | 61593 |
| 86 | 1 | 6 | 2 | 10.8593112 | 98 | 0.999992669 | 0.427412003 | no | 62779 |
| 87 | 1 | 5 | 2 | 14.6389346 | 70 | 1 | 0.462331712 | no | 63471 |
| 88 | 1 | 5 | 2 | 13.35437 | 98 | 1 | 0.453081012 | no | 64109 |
| 89 | 2 | 4 | 2 | 9.92994758 | 98 | 1 | 0.309396952 | no | 65261 |
| 90 | 2 | 5 | 2 | 13.198351 | 98 | 0.921798229 | 0.156767026 | no | 65833 |
| 91 | 2 | 5 | 2 | 14.5708227 | 98 | 1 | 0.458157599 | no | 66501 |
| 92 | 2 | 5 | 2 | 14.1005753 | 98 | 1 | 0.468589246 | no | 67149 |
| 93 | 2 | 5 | 2 | 11.2460834 | 98 | 1 | 0.398594886 | no | 68347 |
| 94 | 2 | 6 | 2 | 15.3355597 | 98 | 1 | 0.447006613 | no | 68985 |
| 95 | 2 | 4 | 2 | 17.1072062 | 98 | 1 | 0.422778368 | no | 69813 |
| 96 | 2 | 5 | 2 | 14.1987428 | 66 | 0.921782792 | 0.232106894 | no | 70456 |
| 97 | 2 | 5 | 2 | 15.2371594 | 98 | 1 | 0.463136524 | no | 71244 |
| 98 | 2 | 5 | 2 | 14.5677498 | 98 | 1 | 0.466992468 | no | 71972 |
| 99 | 2 | 5 | 2 | 15.2545365 | 100 | 1 | 0.467653304 | no | 72748 |
| 100 | 2 | 5 | 2 | 14.2040402 | 100 | 1 | 0.441416562 | no | 73476 |
| 101 | 2 | 4 | 2 | 11.5757994 | 100 | 1 | 0.447805017 | no | 74707 |
| 102 | 2 | 4 | 2 | 10.3968525 | 66 | 0.999153316 | 0.401568711 | no | 75026 |
| 103 | 2 | 5 | 2 | 16.1880692 | 100 | 1 | 0.469031721 | no | 75814 |
| 104 | 2 | 5 | 2 | 10.8106165 | 100 | 1 | 0.464883238 | no | 76726 |
| 105 | 2 | 4 | 2 | 10.7212584 | 100 | 1 | 0.391471684 | no | 77114 |
| 106 | 2 | 3 | 2 | 10.2747486 | 100 | 1 | 0.434488803 | no | 77478 |
| 107 | 2 | 3 | 2 | 10.273553 | 100 | 1 | 0.380053818 | no | 77812 |
| 108 | 2 | 6 | 2 | 15.8479388 | 66 | 0.921830475 | 0.227815121 | no | 78394 |
| 109 | 2 | 5 | 2 | 15.4192198 | 100 | 1 | 0.390571684 | no | 79062 |
| 110 | 2 | 5 | 2 | 15.5237261 | 100 | 1 | 0.458264649 | no | 79700 |
| 111 | 2 | 4 | 2 | 10.8646147 | 100 | 1 | 0.460473448 | no | 80088 |
| 112 | 2 | 4 | 2 | 10.5182224 | 100 | 1 | 0.433370948 | no | 80452 |
| 113 | 3 | 4 | 2 | 16.0428879 | 100 | 1 | 0.444304734 | no | 81120 |
| 114 | 3 | 6 | 2 | 16.536118 | 100 | 0.921869814 | 0.149448857 | no | 81925 |
| 115 | 3 | 4 | 2 | 12.1671326 | 100 | 1 | 0.465009779 | no | 82593 |
| 116 | 3 | 5 | 2 | 15.4135369 | 100 | 1 | 0.468591511 | no | 83341 |
| 117 | 3 | 6 | 2 | 16.6578576 | 100 | 1 | 0.40650025 | no | 84267 |
| 118 | 3 | 5 | 2 | 16.0748443 | 100 | 1 | 0.456500858 | no | 84945 |
| 119 | 3 | 4 | 2 | 11.7223578 | 100 | 1 | 0.379686266 | no | 85827 |
| 120 | 3 | 5 | 2 | 16.2312152 | 100 | 0.921908319 | 0.216772348 | no | 86465 |
| 121 | 3 | 6 | 2 | 17.3617583 | 100 | 1 | 0.457387 | no | 87233 |
| 122 | 3 | 5 | 2 | 16.7315955 | 100 | 1 | 0.442828 | no | 87971 |
| 123 | 3 | 5 | 2 | 16.3049337 | 100 | 1 | 0.466809094 | no | 88627 |
| 124 | 3 | 5 | 2 | 16.4417328 | 100 | 1 | 0.460740507 | no | 89305 |
| 125 | 3 | 4 | 2 | 17.2471526 | 100 | 1 | 0.462420464 | no | 89948 |
| 126 | 3 | 5 | 2 | 16.5593477 | 70 | 0.999151409 | 0.393808573 | no | 90585 |
| 127 | 3 | 4 | 2 | 18.2643545 | 104 | 1 | 0.460303098 | no | 91373 |
| 128 | 3 | 6 | 2 | 13.2881781 | 104 | 1 | 0.418826401 | no | 92559 |
| 129 | 3 | 5 | 2 | 18.0384265 | 104 | 1 | 0.409367919 | no | 93335 |
| 130 | 3 | 4 | 2 | 17.5867654 | 72 | 0.99937284 | 0.361324757 | no | 94063 |
| 131 | 3 | 4 | 2 | 18.379285 | 104 | 1 | 0.456028581 | no | 94691 |
| 132 | 3 | 6 | 2 | 16.4750065 | 70 | 0.921944976 | 0.224400237 | no | 95278 |
| 133 | 3 | 5 | 2 | 17.5853915 | 104 | 1 | 0.452953547 | no | 95946 |
| 134 | 3 | 5 | 2 | 17.527896 | 104 | 1 | 0.471433163 | no | 96674 |
| 135 | 3 | 5 | 2 | 17.3891494 | 104 | 1 | 0.461455643 | no | 97448 |
| 136 | 3 | 6 | 2 | 13.446332 | 104 | 0.999921501 | 0.408258706 | no | 98879 |
| 137 | 4 | 3 | 2 | 12.1008821 | 104 | 1 | 0.465605229 | no | 99213 |
| 138 | 4 | 5 | 2 | 17.0995163 | 104 | 1 | 0.390467197 | no | 100377 |
| 139 | 4 | 5 | 2 | 18.4966063 | 104 | 1 | 0.463484347 | no | 101175 |
| 140 | 4 | 4 | 2 | 13.7960676 | 104 | 1 | 0.4579781 | no | 101699 |
| 141 | 4 | 5 | 2 | 17.7093113 | 104 | 1 | 0.472627938 | no | 102363 |
| 142 | 4 | 6 | 2 | 19.7496071 | 104 | 1 | 0.442393839 | no | 103051 |
| 143 | 4 | 4 | 2 | 13.9365922 | 104 | 1 | 0.46058327 | no | 104212 |
| 144 | 4 | 5 | 2 | 14.0863017 | 104 | 0.997662723 | 0.252270609 | no | 105824 |
| 145 | 4 | 5 | 2 | 19.5861751 | 104 | 1 | 0.47197488 | no | 106502 |
| 146 | 4 | 5 | 2 | 20.2033857 | 104 | 1 | 0.391641915 | no | 107250 |
| 147 | 4 | 6 | 2 | 19.594606 | 106 | 1 | 0.470548093 | no | 108016 |
| 148 | 4 | 5 | 2 | 12.9609819 | 106 | 1 | 0.443391591 | no | 108938 |
| 149 | 4 | 4 | 2 | 19.036517 | 106 | 1 | 0.458861589 | no | 109586 |
| 150 | 4 | 4 | 2 | 13.9965542 | 106 | 1 | 0.438919783 | no | 110064 |
| 151 | 4 | 5 | 2 | 20.0827048 | 106 | 1 | 0.461961359 | no | 110852 |
| 152 | 4 | 5 | 2 | 18.6390224 | 106 | 1 | 0.412204891 | no | 111600 |
| 153 | 4 | 5 | 2 | 18.6042395 | 106 | 1 | 0.461825043 | no | 112286 |
| 154 | 4 | 3 | 2 | 13.2771426 | 106 | 1 | 0.460275799 | no | 112650 |
| 155 | 4 | 4 | 2 | 19.2329598 | 106 | 1 | 0.464033663 | no | 113316 |
| 156 | 4 | 5 | 2 | 18.944641 | 60 | 0.999150276 | 0.412047684 | no | 113894 |
| 157 | 4 | 4 | 2 | 19.2186077 | 106 | 1 | 0.473319888 | no | 114682 |
| 158 | 4 | 5 | 2 | 18.8264407 | 106 | 1 | 0.464770198 | no | 115320 |
| 159 | 4 | 4 | 2 | 13.7566705 | 106 | 1 | 0.458235264 | no | 115708 |
| 160 | 4 | 4 | 2 | 18.8517633 | 106 | 1 | 0.455286145 | no | 116346 |

### 23.31 Finished-source verbal proof and default sweep

The exact requested command, `env -u RUN_TRAINED -u RUN_SLOW make test`,
completed all **7,890 selected distinct cases**, with nothing deselected:
**7,564 passed, 323 skipped, two failed and one non-strict XPASS**.
Native sweep time was **2456.2399339160183 seconds**, wrapper
**2456.6219987869263 seconds**, peak aggregate memory
**10,967,693,544 bytes**. There were no compilation-cache retries.
The harness exits **1** for the two assertion failures, and `make`
therefore exits **2**. Collection and the complete sweep finished; no
timeout or stopping-gate abort occurred.

The raw worker records contain **7,567 passing reports**: the existing
`test_use_flags.py::TestOrthogonalFlags::test_flags_match_expected`
is reported four times, giving three additional passing reports beyond
the distinct-case count. The non-strict XPASS is
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
The existing harness warning says the latest weekly slow-test record is
**43.3 days** old; the requested default environment was not changed to
run that separate suite.

Both stipulated stopping gates pass in this sweep: numeric XOR in
**.26232045801589265 seconds** and the previously named shared class in
**.11755170801188797 seconds**. Both passed in §14 as well. The two
held-answer realization variants pass in **5.849212083994644 seconds**
(`False`) and **5.398821667011362 seconds** (`True`), without replaying
the diagnostic stream. The older intermittent
`test_reasoning_cde_model.py::TestReasoningCDEModel::test_answer_query_degrades_gracefully`
passes in **.9168859159981366 seconds**; its earlier attribution is not
withdrawn and its producer was not repaired here.

**The fresh verbal XOR proof**,
`TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`,
passes in **806.01 native seconds**, process guard **808.7666967080149
seconds**, peak **2,569,277,040 bytes**. All **four of four** generated
word multisets and **four of four** parallel-field word multisets are
correct, with **zero unavailable** on either read. The existing bar and
training construction are unchanged. These are the outputs, in dataset
order:

| Row | Generated words | Parallel-field words |
|---|---|---|
| 1 | `hello world` | `hello world` |
| 2 | `hello there` | `hello there` |
| 3 | `world loving` | `loving world` |
| 4 | `there loving` | `loving there` |

The word-multiset outcomes match §14. The field-word order is reported
as produced; it is not silently made equal to generation's order.

**The two failures that remain, with their producers:**

| Failing test | Producer and attribution |
|---|---|
| `test/test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` | The AST read finds `PartSpace._radix_code_events`; the fixture still names `PartSpace._radix_part_events`. This is §14's older reader-name mismatch. This round's Extension time readers, including `stored_field`, are in the inventory; no additional mismatch remains. The older fixture entry is left unchanged. |
| `test/test_sparse_concept_e2e.py::test_interpreted_object_keeps_the_native_features_in_one_row` | Native word admission collects a basic percept. `ConceptualSpace.add_concept_feature` writes its BP feature column **0**; the old assertion still expects the former part/whole columns **4 and 402**. This is the same failure carried in §14, not a new result of the flat LTM entry or extension operators. The producer and the assertion are left unchanged. |

There is no new-round failure left in this sweep. All receipts remain
local and are not linked.

### 23.32 Returned source

Returned uncommitted source SHA-256:

`20011b579c034c1a2962d8e784af7be8164f84c5b5184b3ac5527800e7148f46`

This is the existing `bounded_tests.source_snapshot` protocol over
**1,070** source, configuration and test inputs, excluding mutable
document prose and local receipts. It is the same source used by the
finished certificate group and explicit slow-marked XOR proof, ten numeric
trainings, scalar acquisition
and omissions, 160-sentence lesson, and complete default sweep including
its fresh verbal proof. The final manifest check agrees.

The checkout remains on **main**, HEAD
`ab5176e0e2bc7b1c3bce8de7d8acb37e89237ebd`. No commit or push was made.
The return is through Alec for Claude's review. The scope exclusions and
certificate limits are those stated in §§23.18 and 23.29; this return
does not build or claim the later round of item 1.8.

## 24. Review of §23 (Claude, 2026-11-14)

The round of §17, as §18 and its amendments asked for it, is complete.
I accept it. The landing is §25. Four findings follow; one needs Alec's
decision, three are carried to the next text.

### 24.1 Verified

- **Source.** `bounded_tests.source_snapshot` over the main checkout,
  dumped as compact sorted JSON, hashes to
  `20011b579c034c1a2962d8e784af7be8164f84c5b5184b3ac5527800e7148f46`
  over **1,070** inputs, as §23.32 says. HEAD is still `ab5176e0e2`.
- **Tests run on that source** (CPU, compiler cache in a writable
  directory): both stopping gates, `test_item18_reduction.py`,
  `test_item18_trust.py`, `test_item18_extend.py`, and the two failures
  §23.31 keeps: **74 passed, 2 failed in 31.44 seconds**; the two
  failures are those two, with the messages §23.31 gives. I did not rerun
  the sweep, the lesson, the verbal proof or the ten numeric runs.
- **The test inventory.** Test function names read by `ast` at
  `ab5176e0e2` and now: **344** names are gone, which is §23.21's 285
  retirements plus the 59 old names of renamed ports; **119** names are
  new, its 60 additions plus the 59 new names.
- **Alec's answer to §23.13 is built as he gave it.**
  `FamiliarTier.promote_identity` folds only the role-zero events of the
  MTM entry and otherwise takes `MTM.value_of`; it no longer reads
  `_reference_sites` (bin/FamiliarTier.py, from line 537).
- **Removed names.** No production file names `PresentationMemory`,
  `TrustGates`, `closedTrust` or `DocumentCover`. `address_keys` appears
  once, in the checkpoint migration of bin/Memory.py, which reads an old
  name only to validate and discard it (§23.17).

### 24.2 Reading a word raises its definition (for Alec)

`Interpret.define` proposes the pair **(1, 0)** when its caller passes no
trust (bin/Interpret.py, line 384), and LTM1 keeps the greater lane of a
repeated pair. Four callers pass none: `lookup_word` (line 297),
`forward` (line 502), `reading` (line 553) and `admit_discovered`
(line 586). So an admitted definition at **.7** becomes **1** the next
time the word is read (§23.11, §23.18). `Models.infer` does not freeze
learning (it sets only `_conversation_read`), so a conversation does this
too. The behavior predates the round; Codex traced it as asked and
changed nothing, which was right.

It conflicts with Alec's rule of §20.7: "changing trust values has to be
done by proof". My recommendation: reading a word uses its definition and
does not write it. Only the writers that witnessed the pair write it,
with the pair they witnessed: `PartClosing.finish_definitions`,
`PerceptCollections.observe` and `FamiliarTier.promote_identity`. The
count of testimony in `forward` stays. Consequence: a definition no
longer rises to one by repetition. Certificate: an admitted **.7**
definition, read in training and read in a conversation, stays **.7**.

### 24.3 Time per sentence grows with the history

The lesson's time per sentence, averaged over blocks of sixteen
sentences:

| Sentences | §14 (seconds) | §23.30 (seconds) |
|---|---:|---:|
| 1-16 | 2.97 | 3.44 |
| 33-48 | 10.94 | 7.08 |
| 49-64 | 3.90 | 9.94 |
| 81-96 | 7.09 | 12.99 |
| 113-128 | 6.03 | 15.82 |
| 145-160 | 6.34 | 17.80 |

Under §14 the second stage levels off near six or seven seconds. Now the
time rises through the whole run, and the total is **1,811** seconds
against **995**. Both runs shared the processor with other work, but a
steady rise within one run is not explained by that. Since point 21 every
reading appends, so LTM0 grows with every sentence. A read whose cost
grows with the retained rows would give this curve (my reading, not
traced). Alec's rule of §3.14 is that no read scans all rows. To carry:
profile an early and a late sentence, find the read whose cost grows with
the history, and bound it where it arises, without changing what is
stored.

### 24.4 The lesson's two fields do not converge

None of the 160 comparisons is exact. The weighted mean difference by
epoch is **.406, .397, .410, .438**, which does not fall over four
epochs. §14's 54 exact comparisons compared different fields, so this
shows no regression. But nothing yet shows the generate walk learning to
reproduce the field of understanding, which is what the comparison is
for. To carry: one attribution on two or three sentences. Which concepts
differ, and on which lane. Is a concept present in one field and absent
from the other, or present in both at different degrees? Which producer
makes the difference: the generated tree's operators and operands, or
the extension of the same tree? §23.20 found heard words read on the
against lane, on the checkpoint too; say whether that is meant. This is
not a gate.

### 24.5 Smaller points

- `_conversation_read` is a flag on the model that two writers read
  (bin/Models.py, line 7172; bin/Examination.py, line 114). It implements
  Alec's rule correctly, but where a trust comes from belongs to the
  write, not to the model. To carry: pass it to the closings as an
  argument.
- The reader-inventory failure of §23.31 is already repaired on GitHub
  main: item 0.5's thread renamed the fixture's entry to
  `PartSpace._radix_code_events`. After the merge of §25 it should pass.
  The feature-column assertion of `test_sparse_concept_e2e` stays listed.
- GitHub main's `bin/IntervalLog.py` reads `rows_of_origin` and
  `kind_of`, which point 21 removed. The merge in §25 ports it to the
  truth view. It runs only when `intervalLogEveryBatches` is set, but
  `test/test_interval_log.py` exercises it.
- The standing docs on main describe some of what this round removed.
  After the landing I will reconcile doc/*.md with the landed code; that
  is not Codex's work.

The limits of §23.29 are carried unchanged: a learned chooser of a
multioperator parse, the learned negative sentence, several open operands
in one thought, and the different readings of `Intuition.pervasion` and
point 11.

### 24.6 Alec's answers (2026-11-14)

**On §24.2, reading a word.** Alec: "Reading a word is direct evidence
of its existence, which means that the word-concept is true. It may not
apply to the object-concept (that would imply trust in the meta-concept,
which we should set at 1 for now, along with the object-concept's trust,
and note in future work that we need a better algorithm). But you
mentioned the *definition* for a word: what sentence defines the word?
This could be a valid change via trust by authority, it depends on the
trust of the speaker of the sentence. "Trust changes only by proof"
applies only to cases of contradiction, where presumably both sides have
been produced by valid cognition. we will have a chain of trust where
some trust needs to be adjusted."

So §24.2's framing was wrong: proof governs a change of trust only where
two claims contradict each other; a valid source may add trust. The four
readers of §24.2 stay as they are: reading a word sets its word-to-object
mapping (the meta-concept, which the code calls the definition, the DEF
pair of LTM1) at one. A future-work entry records that a better algorithm
is needed for the trust of that mapping and of its object.

To his question: in §24.2 no sentence defines the word. "Definition" was
the code's name for the word-to-object link that `Interpret.define`
writes when a heard word is read as its object. A sentence that does
state a definition, a kept identity sentence, reaches
`FamiliarTier.define_pending` through `finish_definitions`, which passes
its relation's stored pair. That pair is the speaker's trust: in
training, the model's trust times the word's certainty; in a
conversation, zero (§23.18). So it is trust by authority, as he says.

*Open (Claude's reading, for Alec).* "along with the object-concept's
trust": today an object's trust radius starts at zero and rises with
agreeing views, `agreement / views * (1 - exp(-views / trustViews))`
(`ConceptTrust.refresh`; decision 4 of §1, "trust is uncertain (0) to
start with"). Certificate 13's **.8646647334** is six views of that
rule. The mapping's effective read is scaled by its ends' radii
(`ConceptTrust.positive_length` in `LTM1.refresh_effective`). If an
object reached through a word is to be taken at one for now, decision 4
and certificate 13's number change for it. That belongs to the round
that rebuilds `interpret` (§3.19), so it waits for Alec's word without
holding §25 or the next round.

**On the sense a word is read in.** "Highest order by default"
(recommended). When a word's concept is raised, the old concept keeps its
word and the class takes a new concept of the same word; interpretation
reads the highest-order sense, with or without a determiner, and `lower`
takes it down. In the stopping gate
`test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`,
only the last assertion changes, `'word' not in
alloc.symbol_types[shared]`: the old concept keeps its word. Its other
assertions stand. This is Alec's word for that change.

**On the shape of what remains.** "Two rounds" (recommended). The first:
the operators of §22.7 and §22.8, determiners and number words as
modifiers, the sense of a word as above, one `.where` range per modality,
places on two axes, level sets, the XOR proofs on that path, and an image
forward and back (certificates 65 to 67). The second: §3.18 and §3.19 as
rewritten, `expect` and `intuit` as learned maps, `interpret` making
symbols, and learning on images measured (certificate 68).

**On `unfocus`.** "We should learn what it does, just like we learn what
focus does. If that is difficult, just undo the last focus." The source:
which attention act is taken, and where, is learned (the chooser), but
what an act does is fixed code (`Attention.narrowing_mask`: divide splits
a bracket, descend enters its parts). No act's effect is learned today.
So `unfocus` is built as his fallback: it restores the bracket in force
before the last `focus`, it is legal whenever a focus is in force, and
the same learned chooser decides when to take it. Learning what an
attention act does is recorded as his direction, for future work.

## 25. Codex text: land the round and merge GitHub main (Claude, 2026-11-14; for Alec to paste)

~~~
Land the completed round of item 1.8
(doc/plans/2026-11-11-item1-8-last-architecture-round.md, returned as
§23, reviewed and accepted in §24) and merge GitHub main into it.

1. Check that the source is still
   20011b579c034c1a2962d8e784af7be8164f84c5b5184b3ac5527800e7148f46
   (bounded_tests.source_snapshot). Commit the round on main as
   "Land item 1.8 extension, recall and reduction round", with the spec.
   Commit no receipt and nothing git ignores.
2. git fetch origin, then merge origin/main (a0f3bc70b8) into main; do not
   rebase. Main carries item 0.5's test development, the rewritten
   doc/*.md and the image data. Files changed on both sides:
   bin/Forgetting.py, bin/Models.py, bin/corpus_lessons.py,
   data/model.xsd, test/identity_experiment.py,
   test/operators_run_audit.py, test/test_corpus_checkpoint.py,
   test/fixtures/when-readers-round4a0.json, the 1.8 spec (keep this
   round's text), and two plans main deleted,
   doc/plans/2026-10-27-item2-5-thinking.md and
   doc/plans/2026-11-05-item2-3-trust.md (take the deletion). Where main's
   code reads something this round removed, port it to this round's
   reader under the point of §17 that removed it; never restore a removed
   column, origin, kind or writer. Known case: bin/IntervalLog.py reads
   rows_of_origin and kind_of (point 21); it reads the truth view instead,
   and test/test_interval_log.py keeps its assertions where their
   contract holds.
3. Run bin/prepare_data.py: the merge removes tracked MNIST and FineWeb
   files, and it restores and verifies them.
4. On the merged source: both stopping gates, the tests main added
   (test_image_corpus, test_game_corpus, test_math_corpus,
   test_lesson_stages, test_training_corpus, test_interval_log,
   test_corpus_checkpoint, test_doc_links), and one default sweep,
   env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected. Repair a
   failure the merge caused where it arises; list a failure that predates
   both sides.
5. If no failure is new: commit the merge as "Merge GitHub main into item
   1.8" and push main. Never force-push. If a failure is new and its
   repair would change the round's behavior, stop and write it for Alec.

Rules: §3, §7 and the rules of §10's text. A failed stopping gate stops
the landing. No seed is chosen, no fixture is resized, and nothing
outside the merge's needs is changed.

Return as §26: each conflict and how you resolved it, each port, the
sweep's result, the failures that remain with their producers, and the
pushed commit.
~~~
