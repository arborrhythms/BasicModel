# Orders: sets, membership and the two ladders

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](specs/2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](specs/2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](specs/2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; the document and the stream are **context wholes** in WholeSpace, so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.
> - **One conceptual space (Alec, 2026-10-22; §10):** §9's per-order spaces and stratified read are reversed; one surface places every concept at every order; strata stay in LTM.
> - **Orders are learned (Alec, 2026-10-21; [§10](#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3)).** No ordered grammar is written and parts of speech stay order-free. Definitions raise, through equivalence classes; NP + VP lowers, returning to S0 (amended 2026-10-22 and 2026-10-23: the determiner selects and the modal withholds the lowering). Definitions are kept both as LTM1 rows (explicit) and as folds (implicit), which supersedes "folds, not rows" above; the rows train the fold, and their match or mismatch is judged. Built in todo item 3.


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

*Amended 2026-10-22 ([§10](#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3)): the per-order conceptual spaces and
the stratified read are reversed; the conceptual space is one. The symbol
tree, the Part edges as the record of membership, membership read by edge
and the cross-order diffusion stand.*

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

*Centroids (Alec, 2026-10-24; [context-wholes spec §2b](plans/2026-10-24-context-wholes.md)):*
an order-0 concept's form is the center of the space of its parts and its
wholes, each averaged over its count; a higher-order concept has no parts
and wholes, so its content is the pure centroid of its members' content,
weighted by their positive and negative evidence. The caches of §9 and the
promotion fold follow this rule, not a max or a min.

## 10. Orders are learned: definitions raise, three compositions lower (Alec, 2026-10-21; decided, built in item 3)

Alec: "we need not create an ordered grammar, because we have the rule
that raises order." Orders live in the concepts, not in the grammar's
categories. Parts of speech stay order-free and come only from the
grammar's positions and occupations
([reconstruction spec §10](plans/2026-10-21-reconstruction-reads-the-bind.md#10-decided-alec-2026-10-21-later-parts-of-speech-only-from-the-grammar)).
The full design record, with the verification against the code, is in
todo item 3.

**Definitions raise, by equivalence class.** Alec: "'Is' does not cross
order by itself, but when used with multiple identical RHS or LHS, it
creates an equivalence class." The class sits one order above its
members; this is Frege's definition by abstraction, as a direction is the
class of lines parallel to a given line. The learning sequence (Alec):

- "This is Felix, that is Felix": 0 to 1. Demonstratives point at events,
  at order 0. Two different events that are both Felix make Felix the
  class of his occurrences, an individual. The same rule therefore
  creates individuals in the first place.
- "Felix is a cat, Ciao is a cat": 1 to 2. Two different individuals make
  cat a kind.
- "Paris is the capital of France" is identity: both sides pick out one
  individual, and nothing is raised. "y is three. x is three." raises
  nothing either, since x and y are both bound to three itself.

Conditions (Claude's reading, accepted into item 3): "the same side" is
the symbol the mentions read to; the other sides must be distinct, as
events at different places or times or as different identities in MTM;
and the word is given the raised concept as its object. Each definition
is an LTM1 row; the raise fires when a second distinct member joins the
singleton an equality already marks.

**Three compositions lower, returning to S0.** Alec: "two things lower
the order: a determiner and the combination of NP and VP", and "vp mp for
modal". Each lowers one order and stops at 0:

- The **determiner** picks a member out of a class: "the cat".
- **NP + VP**: a noun phrase names a class, the invariant across its
  members, and a verb is the step between two of them (the occurrence
  spec's first difference). The verb holds at some members, not at the
  class, so the sentence lands among them: "Felix runs" is an occurrence
  of Felix at order 0; "dodos died out" is a change across the kind's
  members, at order 1. A mass noun's instances are already order 0, so
  "water flows" stays there.
- **VP + MP**: the modal ranges over possible worlds, the second
  difference; a verb phrase under it holds in some of them, one order
  down.

  *Reversed (Alec, 2026-10-23, "Good correction"): the modal does not lower;
  it withholds the lowering. A modal clause is the ordinary composition
  kept above order 0, an abstraction of a possible event that extends
  an alternative timeline of the one identity, never an order-0
  occurrence ([iteration spec §4](plans/2026-10-23-verb-and-modal.md)).
  Only NP + VP lowers.*

IS raises by collecting members, and these three lower by landing among
them. This supersedes §5's count for predications, under which a
sentence about individuals was order 2.

**Amended (Alec, 2026-10-22): determiners select; only NP + VP and VP + MP
lower** *(and 2026-10-23: only NP + VP; the modal withholds lowering)*. "Determiners might not even be dropping the order: they might just
be selecting one of a plurality, and leaving the order untouched. Which
means only NP VP and VP MP combinations are dropping the order." A
determiner picks one of a plurality, and the phrase's order is the order
of what it picks, read by lookup: a member of a kind is an individual, a
shaped portion of a mass is an individual. So "the water" is an individual
at order 1, as §7 has it: "It is a count noun as an elision of 'the glass
of water', and a mass noun that needs a shape or form in 'the cup of
water'." The §7 question below is closed. And "raises and lowers had been
sigmas and pi's within the context of the grammar, and we can leave them
calling those operations, but order assignment is by lookup these days
(which is set by equivalence classes, not calling these methods)": `lift`
and `lower` keep their numerical folds, and a constituent's order is read
from the concept it denotes, whose order the equivalence classes set. The
two lowerings are facts about what the compositions denote, an occurrence
and a difference, one order below their subject, which the lookup returns.
Credit: "binds should just be adding to the MTM pool, which is input to
output prediction, so binds should get credit assignment through the
derivative"; raises carry no price of their own either.

**Modals (Alec, 2026-10-22; for 3.5 step 5).** "Verbs and nouns are both
of the same order ... So modals may also be of that order. But MPs are
differences of VPs just as VPs are differences of NPs (not really
differences, but diagonals within quadratics). Running and walking are
both possibilities for Joe's movement. So they should be treated
analogously to VPs." The step-5 spec builds the modal as the second
difference in that form: a group over a noun phrase's alternative verb
phrases, a diagonal per modal in a shared basis, as step 4 builds the verb
over an identity's occurrences.

**Explicit and implicit definitions, judged together.** Alec: "multiple
DEF in LTM1 can be used to train the fold being used in the symbolic
order loop, and that's really all we need for integration. We can keep
the explicit and implicit definitions, and their match or mismatch is a
judgement process, just as in human cognition." The fold raise collects
two rows that occur in matching contexts under a parent one order up;
definition rows supply its targets. A definition is evidence for
membership and the fold's reading is evidence too; agreement fills one
truth lane and a mismatch fills both, the corner that prompts refinement
(2026-09-23). The lanes are never netted.

**Two meanings of a concept (Alec, 2026-10-22).** "The contextual words
provide a subsymbolic definition. We have also definitions in LTM1, and
somehow those two are the explicit and implicit meanings of concepts, for
now." The implicit meaning is the concept's place in conceptual space, set
by the registered priming surface: the words that occur with it. The
explicit meaning is its definitions, LTM1 rows that give its members and
its classes. They are the same pair as the two learning routes above: the
fold reads the implicit meaning and definitions supply the explicit one,
the rows train the fold, and their agreement or mismatch is judged in the
two truth lanes. Each has what the other lacks. The implicit meaning
places a concept before anything is said about it, and grades similarity.
The explicit meaning is exact, can be stated and recalled, and raises
order. A concept with only an implicit meaning is a placed order-0 type.
A concept with only an explicit meaning has not yet been met in use.
The basis of a class's identity, the intersection that is its invariant
or the union that is its extension, is kept open in
[FutureWork](FutureWork.md#intersection-or-union-as-the-basis-of-identity-alec-2026-10-21)
(Alec, 2026-10-22: important for kinds and for the Buddhist reading).

**The surface places the concepts (Alec, 2026-10-22).** "We are doing
explicit spreading over the priming surface, and then using the activation
of the priming surface to distribute vectors in conceptual space, including
the order-zero concepts. That said, the priming is across orders, and the
concepts each have their own space, so there is still some disconnect."
Agreed, with §9's resolution of the disconnect (reversed in the next
paragraph): the spread crosses orders
along the Part edges and the shared occurrences, but the read is per
stratum. Restricted to the symbols of order n-1, the surface is a point in
order n's space, so each concept is placed by the one surface read at its
own stratum. One energy, many strata, no shared space. In the code the
spread crosses orders (`Priming.diffuse`) and each row's order is known
(`priming_row_orders`), but no reader yet restricts the surface to a
stratum; that read, with placement, is item 3's. Placement itself is
unchanged: the same meet-raising objective at every order, over the stratum
the concept's order names.

**Decided (Alec, 2026-10-22, later): one conceptual space, not strata.**
"I am not convinced we should sort by stratum, which is good for LTM. Here
we are connecting priming across orders (good), and the more I think about
it, the more I am in favor of a single conceptual space (so concrete and
abstract concepts all intermingled)." This reverses §9's per-order
conceptual spaces and the stratified read of the surface. Strata remain
the structure of LTM: LTM0 occurrences, LTM1 relations, the derived LTM2.
In conceptual space, one surface places every concept, at every order,
with the same objective: raise its meet with what the surface activates,
lower it against negatives. A kind is then placed among the words,
individuals and events it occurs with, and an abstract concept sits beside
the concrete ones it is used with. The code already has this shape: one
`similarity_codebook` holds every order's rows, and `context_similarities`
compares them in one space. What §9 keeps: all symbols in the one
mereological tree, the Part edges as the record of membership, membership
read by edge rather than by distance, and the cross-order diffusion.
Item 2.9 specifies cross-order diffusion by location in the one space, class
rows at their members' centroid. What
goes: the per-order cubes, the "point in every order's space at once"
read, and `priming_row_orders` as a read restriction. §9's crowding
objection is answered by the objective's negatives; the collapse
diagnostic measures it.

**Every symbol's order.** Each concept has an order: a recorded one
first, else one above its highest symbolic part, else 0. Item 3 closes
four gaps: 0 doubles as "unknown"; a recorded order overrides structure
unchecked; a relation row's handle always reads 0; and a test-only
formation API keeps an outdated ladder.

**Learning parts of speech** follows the same order: proper nouns first,
then mass nouns, then count nouns where the language has them. Until item
3 lands, 3.5 uses two stand-ins: a determiner always outputs order 1, and
unit tests force a name's order 1 (Alec: "unit tests can force parts of
speech").

**Open, decided with item 3.**

- *A determined mass or event noun.* Closed above (2026-10-22): the
  determiner selects, and "the water" is an individual at order 1.
- *Where a definition is stored.* Step 2 reads a relation's stratum from
  its argument tags, so both "this is Felix" and "Felix is a cat" are
  LTM1 rows. Alec's reading would store each in the stratum of the class
  it builds, putting the second in LTM2. The stratum is derived when read,
  so this changes reading, not storage.
