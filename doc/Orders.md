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
