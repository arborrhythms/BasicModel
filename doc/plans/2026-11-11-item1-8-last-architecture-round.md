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
- *What stays:* the model clock (`when_time`, `RowTime.stamp`), the where
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
