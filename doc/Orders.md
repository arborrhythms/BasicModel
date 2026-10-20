# Orders: sets, membership and the two ladders

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.


*Alec, 2026-10-15, in dialogue with Claude; recorded as decided theory.
Technique lives in the code; this document is the framing it points to.
Earlier statements it extends: [a concept is sigma over pi](Philosophy.md#where-is-context-particulars-universals-and-the-two-axes-2026-09-22),
[symbolization establishes an order-1 individual](Mereology.md),
[independent components](specs/2026-09-26-independent-components.md),
and the item 6 plan [§6](plans/2026-10-10-item-6-stored-idea-generativity.md#6-parts-of-speech-from-operand-participation-alec-2026-10-10)
and [§12](plans/2026-10-10-item-6-stored-idea-generativity.md#12-orders-sets-and-membership-alec-2026-10-15-dialogue-with-claude).*

## 1. Orders are sets of symbols

An object at order *n* ≥ 1 is a set of order-(*n*−1) symbols, with graded
(fuzzy) membership, exactly as in set theory. Order increases are always
over individuals: a set collects symbols, never field regions directly.

Order 0 is the field: percepts, located in `.where`/`.when`, and the
Boolean and mereological operations among them — union, intersection,
complement, parts and wholes. There is no "same" at order 0, only overlap
and containment; two identical percepts are not one thing until a symbol
collects their occurrences. Truth at order 0 is coverage of an extent (the
catuṣkoṭi's for/against lanes).

## 2. Intension lives at order 0; orders above 0 are extensional

The pi (intersection) of a set's members is a field object, so every set
at every order carries an order-0 **type** as its characterization. The pi
is computed from the members, not stored as a second object. Orders above
0 are purely extensional; what differs between them is only the
*collecting relation*.

Consequences:

* **Mass terms** (water) are a pi over water-bearing events with no
  individuals: order 0. That is why they behave as properties — no
  plural, "much" not "many" — and why "wholes are types" covers them.
* **Count kinds** (cats) are sigma sets. Second order over individuals
  (these cats), or first order directly over occurrences — the mass
  reading of a count noun ("there's cat all over the sofa", the universal
  grinder) is the first-order quotient. Count/mass is which quotient the
  sentence licenses, not a lexical fact.
* A kind has two faces: its set (extension, who belongs and how much) and
  its pi (intension, the shared signature). Membership degree is computed
  from the pi: a symbol belongs to the extent the shared signature
  explains it.

## 3. Membership is by evidence, learned per set

The collecting relation is never fixed per order; it is evidence, and the
behaviour of each order follows from its evidence, not from set-hood.

| Order | Collects | Evidence | What follows |
| --- | --- | --- | --- |
| 1 | occurrences → an **individual** | continuity (`.where`/`.when` trajectory); witnesses are the degrees | identity and counting begin; persistence and change — verbs, tense, reference (names, pronouns, definites) apply here |
| 1 | tokens → a **word** | form sameness (exact; typos and variants are the fuzzy fringe) | the symbol table's META rows |
| 2 | individuals → a **kind** | explanation by the kind's pi | inclusion (cats ⊂ animals), quantification, generic truth; kinds do not change, their members do |
| 2 | individuals → a **collection** | adjacency / plurality | location (the sum of members) without an identity criterion beyond membership |
| 2 | words → a **part of speech** | participation in operand positions | the learned grammar's classes |

Identity is per order: a generalization over occurrences (an individual,
which keeps its history) is never identified with a generalization over
individuals (a kind). Same point, different order stamp ⇒ different thing.

## 4. What an order increase abstracts

"Abstracting temporal information" is two different things:

* occurrences → individual abstracts the *moment* but keeps the
  *history* (the individual is a temporal object with a trajectory —
  object permanence is ordered persistence, not timelessness);
* individuals → kind abstracts the *index* (which one, where, when) but
  not temporal *content*: a kind of change (a B column) keeps its shape of
  change at any order; "sunset" is a kind defined by temporal structure.
  [5.5](specs/2026-09-30-occurrence-tense-aspect.md)'s split between the
  situation stamp and tense/aspect as content is this line.

Even a kind keeps a statistical trace of situation — cats occur in houses
— as an expectation ([4.5](specs/2026-10-08-expectation-at-every-level.md)),
not as identity. The independence term excludes the `.where`/`.when`
bands ([6.5 §4.8](specs/2026-09-26-independent-components.md)) for this
reason.

## 5. Two ladders, one fold

Forms and referents are parallel ladders with the same fold:

* **forms**: tokens (0) → words (1) → parts of speech (2) → grammar rules
  (3, relations over classes);
* **referents**: percepts (0) → individuals (1) → kinds and collections
  (2) → generic propositions (3).

Reference crosses between them at any order: a word (1) can refer to a
type (0), an individual (1) or a kind (2). Counting relations as type
theory does — a relation over order-*n* things is order *n*+1 — a sentence
about individuals is order 2, a generic sentence and a grammar rule are
order 3, and a document of generics or a reasoning chain over rules is
order 4. Nothing in the fold stops at any order; the only ceilings are
where the grammar stops reading and what the reference table declares
(today `kind: 2`, to be retired with the operand names, plan §6).

Every order has the same two faces — a fuzzy set of symbols one order down,
and a pi at order 0 — and the orders differ only in the collecting
evidence: form (words), continuity (individuals), explanation (kinds),
participation (word classes), co-occurrence in a row (relations).

## 6. Distributional for words, semantic for objects

A part of speech is a kind-of-word: an order-2 set formed by the same fold,
with participation as the evidence. The class vector in position space
([plan §6.1](plans/2026-10-10-item-6-stored-idea-generativity.md#61-the-learning-rule-alec-2026-10-10-claudes-reading-of-it))
*is* the pi of the word-kind, and the softmax share at a position is
membership by explanation. A kind of thing is an order-2 set formed by the
same fold with explanation by its pi over meaning coordinates as the
evidence. Same machinery, two signature spaces.

Two definitions of "noun" caused the trouble in the literature —
thing-word versus NP-filler. They belong to different ladders: *thing* is
the referent's order and family (A/B/property), *noun* is the word's
participation, and their correlation is learned. The exceptions
(*thunder*, *arrival*) are where the ladders legitimately disagree, which
is what makes the correlation a fact rather than a definition.

**Pinker and Gleitman.** Semantic bootstrapping (things → nouns by innate
linking rules) and syntactic bootstrapping (a verb's meaning read off the
frames it appears in) were not a dispute about adult categories; both
agree those are distributional. They disputed the *bootstrap* — which
direction runs first when the child knows neither categories nor meanings,
with a UG stake in whether linking rules are innate. For this project the
curriculum settles the data order: stage 2 presents words *with* their
objects, so the first referents are given by the teacher, not inferred by
a linking rule; distribution then has grounded credit to work from.
Gleitman's direction survives as the one cross-ladder flow that is
evidence rather than rule: the scene underdetermines a change, so a verb's
B column is constrained by the frames its word fills.

## 7. Grammatical operations are order operations

Determiners lower (choose a member); bare plurals and `every` stay at
order 2 (`every` lowers to the extension and chooses no member); a *bare*
mass term stays at 0, while a determiner lowers a mass description to a
portion — *the water* is an individual collected by continuity like any
other (2026-10-16); count/mass, singular/plural and proper/common are the
grammar reading the order and the collecting relation. The determiner is
the name of the quotient. A definite with nothing accessible to bind to
writes its occurrence and stays **unbound** until a referent appears
(presupposition accommodation); it never asserts novelty.

*Worked example (Alec, 2026-10-16): "The water froze. It became ice," ice
never perceived.* `the water` lowers a mass description to a portion and
stays unbound; `it` binds to it; `ice` is a bare NP at order 2 with a
symbol but no content. The temporal difference on the portion is `+ice`
as a context coordinate; it teaches `froze` ("→ ice"), makes `became` the
copular identity once `froze` predicts ice, and gives the type *ice* its
first content from this context. The requirement on a VP's operation is
therefore "definable at order 0, however the coordinates were learned";
"over percepts" is its special case. The portion stays the same individual
while its kind changes — kinds do not change, members do.

## 8. Consequences for item 6 part 2

* The identity lessons (order 1: the same individual across sentences;
  mint on recurrence of the individual, bind on re-mention; binding and
  row-pair metrics) are the first test of the dictionary. The admission
  gate's "recurrence" is membership by evidence at order 1 and stays in
  that form.
* Factorial-versus-confounded recovery of kinds and properties is an
  order-2 question — the pi over individuals' signatures, `.when`
  excluded — measured only once individuals exist. Whether that pi is a
  second dictionary or the taxonomy's computation is open.
* Parts of speech are learned at order 2 over words by participation
  (plan §6), never credited to phrase heads (the toy: heads merge pronouns
  into NP), with the referent's order and family admissible only as extra
  signature coordinates, weak enough that actual use overrides them.

## 9. Per-order conceptual spaces, ramsified over one symbol tree; the stratified priming surface (Alec, 2026-10-20; decided)

Alec's summary: the perceptual tree up to the zeroth-order symbol is
mereological (the word form at the centroid of its letters and wholes in
the percept cube); zeroth-order concepts are distributed relative to each
other; first-order symbols are again mereological but a centroid may be
meaningless for a discontinuous type; first-order concepts are distributed
relative to each other — and the question whether all concepts share one
space or are distributed per order. Decided: **per order, ramsified; all
symbols in the one mereological tree.** In his words: "ramsify over
conceptual spaces, but keep all symbols in the mereological space."

- **Ramsify over conceptual spaces.** A concept at order *n* is defined by
  its relations to the symbols of order *n−1* — a kind by which symbols
  are its members, an individual by which occurrences are its, with the
  truth lanes as degree (§3). Those relations are its coordinates: order
  *n*'s conceptual space is the cube over the order-(*n−1*) symbol index,
  a concept there is its characteristic vector, and similarity at order
  *n* is overlap of relation sets — the meet of two sparse membership
  vectors. This is §2 made geometric: intension at order 0, extension
  above. It is never meaningless: a discontinuous type is a sparse vector
  over disjoint members. Order 0 is the base case, its concepts placed by
  context in a dense similarity space — a compression of the same kind of
  relation set, the symbol's contexts ([FutureWork, placement](FutureWork.md#placement-from-the-registered-priming-surface-one-metric-for-sbow-diffusion-and-bind-alec-2026-10-20-proposed-with-the-objective-to-decide)).
  Nothing is stored for the orders above 0 beyond LTM2's `Part(member,
  whole)` edges: the edges *are* the vectors.
- **Keep all symbols in the mereological space.** Every symbol of every
  order lives in one structure, the symbol index with its Part tree
  (two truths §1, "LTM2, one relation"): a word form the centroid of its
  letters and wholes in the percept cube; a kind tied to its members, an
  individual to its occurrences, a sentence to its constituents, by Part
  edges. One tree, all orders, one relation; two carriers — geometric up
  to order 0 (forms), recorded above it. The **centroid** of an order
  above 0 is a derived cache over its Part edges (the intersection model
  of identity), kept only while it passes the test that its meet with
  each member stays high; where that fails the type is discontinuous and
  the set is the only representation (the union model). The two identity
  models of [Philosophy](Philosophy.md) are these two representations.
- **"Is Felix in the same space as *cat*?" (Alec, 2026-10-20) — not as
  a point beside him; the three things that could mean are each something
  else.** Felix and the *kind* cat: **membership**, a Part edge, Felix one
  coordinate of the kind's vector over order-1 symbols. Felix and a
  *typical* cat: **typicality**, the meet of Felix's order-1 vector with
  the kind's centroid cache — which lives **one order down, in the
  members' space**, because it is the fold over members like him; so
  that comparison does happen in one space, Felix's own. Felix and the
  *word* "cat": the word is order 0 and is an **axis** of Felix's
  representation (he was described as one), not a neighbour; the name
  "Felix" is another axis, the proper noun whose definition is the
  individual. A kind therefore has two representations in two spaces —
  its member set above its members, its centroid among them, kept while
  the type is continuous — and Felix belongs to the first and meets the
  second.
- **Cross-order comparison is membership, not distance.** An individual
  against a kind is a membership read; an idea against a kind a membership
  read of its participants; `bind` compares candidates at the request's
  own order. Distance between orders was never a meaningful quantity, and
  one shared space would buy only that, at the price of crowding and
  meaningless centroids.
- **The priming surface is the same object, read stratified.** It is one
  vector of energy over the codebook rows with each row's order known
  (`priming_row_orders`). Restricted to the symbols of order *n−1* it is a
  fuzzy set of them — a point in order *n*'s conceptual space: the
  activation over words is the current context as a point in order 1's
  space, over individuals the current scene in order 2's. One vector, one
  point in every order's space at once. **The diffusion kernel per
  stratum:** across orders, along the Part edges (member to whole, whole
  to its other members — the existing two-hop); within an order above 0,
  the same two-hop through shared members, so no code-derived channel
  exists there; within order 0 only, the geometric overlap in the dense
  similarity space — the one code-derived channel, which placement must
  not read. **Read points per consumer:** placement reads the order-0
  stratum after the seen bumps and the edge diffusion, before the
  order-0 geometric channel; `bind` reads the request's order as its
  expectation term and the Part edges for the verb's participants;
  attention and the chooser read all strata; thought's desire channel
  bumps the stratum it names. Decay, bumps, energy conservation and the
  capped read are unchanged ([Architecture, priming](Architecture.md)).
- **Where:** todo item 3, with placement; the stratified surface and the
  per-order kernel land with the LTM2 store (3.5 step 1b carries the
  edges; item 3 carries the strata and placement).
