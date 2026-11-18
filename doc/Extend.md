# Extend

How a sentence's operators turn the concepts its words activated into the
field of understanding. Stated by Alec on 2026-11-12; quotations are his.

## Terms

- **Token and type, whole and part.** "For kinds, a token implies a type.
  For mereology, a whole implies a part."
- **The sentence-concept** is what a sentence's words compose to. "The
  sentence words can be transformed into a single vector in conceptual
  space, which does not (necessarily) exist as a single concept in
  conceptual space." Having no row of its own in the concept inventory, it
  cannot be referred to; it is held as a vector.
- **The idea** "is the intension of the sentence-concept": that vector.
- **The field** "is the extension of the sentence-concept": a set of known
  concepts, each with its trust on the two lanes, for and against.
  It is the symbolic activation at every order, tokens, their types and
  the types of those, and holds no grammatical operation
  ([Reasoning](Reasoning.md#symbols-the-field-and-the-one-hypersphere)).
- **The field of understanding** is the field a sentence is left with when
  every operator of its tree has been extended. "It is non-conceptual
  understanding in that not all order-zero concepts are necessarily
  symbolized (although in the current architecture all of them are)": a
  member of the field need not have a word.

## The four uses of an operator

"compose, think, generate, extend".

In the first three an operator takes intensions and gives an intension: "In
a linguistic context operators select the best match, they compose and
generate intentions." Compose builds the idea from the words, thought works
on ideas, and generate walks an idea back to words.

In the fourth it takes a set of active concepts and gives a set: "We need
those operators to generate an extension from an intension, which is easy
for known concepts, and then we can perform a selection. So the result of
the operation is the selected field, not the sentence idea or its
constituent word ideas."

## The operation

"So *every* operator needs an extend(): we start with the set of concepts
in the extension of the noun phrase, and then extend() it until we have a
new set, which is the field of understanding."

1. **Two methods of every operator.** `compose_extend` and
   `generate_extend` stand beside `compose` and `generate` on the
   operator. A grammar file whose operator lacks either fails at load. The
   file gains no section and no rule for them, and nothing is parsed a
   second time.
2. **It starts from what is active.** "extend should operate on the
   existing conceptual activations, or a top-k select thereof." The
   starting set is the extension of the noun phrase: the concepts its
   words activated, with their trust. A leaf's set is the sentence's
   existing activations at every order (`_sentence_open_fields`). The extension of a known concept is
   stored, its members in the definition store, so nothing is found by
   matching a vector against the code of every concept.
3. **It follows the tree compose left.** "if we have the STM tree, we can
   just iterate over its existing compose(), calling extend() on each
   operator to process the set of conceptual activations." One call of
   `compose_extend` for each operator of the kept tree, in the order
   compose applied them.
4. **A set in, a set out.** A set is concept references, each with its
   trust on the two lanes (`Extension.ConceptSet`). An operator of two
   operands takes the sets of both and tells them apart by their places in
   the sentence.
5. **The last call leaves the field of understanding.**
6. **An operation on the field, not a lowering.** "it really is an
   operation on the field of active concepts and not necessarily lowering
   the order of a concept." Only `lower` goes from a kind to its members.

An idea that is met without its sentence, one recalled from long-term
memory or one a thought concluded, has no tree in short-term memory. "We
already have a mechanism for mapping words to a sentence concept
bidirectionally. So if we walk the idea to the sentence (a WFF), the wff
can select the extension precisely." Its generate walk gives the tree, and
`compose_extend` gives each node's set from the leaves, a known concept's
set being its stored extension. `generate_extend` is the inverse: it gives
each operand its part of its parent's set, which is how an operand's part
of a recalled idea's field is read.

## What each operator's extend does

Three rules cover most operators. Narrowing one operand by another is
`intersection`. Joining kinds is `union`. Negation is `not`, which swaps
the two lanes, or `non`, which withholds both.

| operator | when composing | extend |
|---|---|---|
| `intersection` | the least, on a field or of two operands | the intersection of its operands' sets |
| `union` | the greatest, on a field or of two operands | their union |
| `not` | swaps the two lanes | swaps the two lanes of the set |
| `non` | withholds both lanes | withholds both lanes of the set |
| `compound` | a modifier selects among its noun's cases | narrowing: the noun's set intersected with the modifier's |
| `modifier` | a gain applied to its head | narrowing |
| `verb` | gives a verb phrase its basis from its object | narrowing |
| `lower` | drops a word of order one or more by one order | one generic member (below) |
| `interpret` | a word to its object | its object's set in place of the word's |
| `ask` | marks a position of uncertainty | its open operand gives nothing |
| `change`, `preposition`, `surface`, `sign`, `bind`, `swap`, `part`, `whole`, `implies`, `focus`, `mint` | | its content operand's set, unchanged |

### Determiners

"Determiners do lowering/selection on their arguments, to either one or
multiple vectors." A determiner takes a kind to its extension and selects
there.

| word | selects |
|---|---|
| "the" | "a known identity from MTM" |
| "a" | "a single subconcept": "'A cat' should select a generic 'cat', presumably; perhaps it can be a centroid?" One generic member, the centroid of the kind's subconcepts |
| "some" | "some subconcepts" |
| "every" | "the entire extension" |

No determiner has an operator of its own. A determiner is a modifier, and
which operator its word takes is chosen at run time
([Language](Language.md#determiners-number-words-and-the-modifier-alec-2026-11-13)).
`lower` drops a word of order one or more by one order, and its extend
gives one generic member: every subconcept of the kind at an equal share
of one, so that the set's key (`FieldProjection.key`) is their centroid.
Over the rows of memory it leaves one row.

## What the field of understanding is for

- **Memory.** A row of long-term memory holds both sides of one
  sentence-concept: its idea, and references to the strongest members of
  its field of understanding with their signed trust. The row has one
  signed trust of its own, from $-1$ to $1$, which is how far the
  statement is held true or false; the idea is not scaled by it. "A
  semantic vector
  comes from the associated episodic concepts, which may or may not be
  enough to reconstitute the sentence verbatim." Two ideas can have one
  extension: a kind of one member and that member leave equal entries. A
  row's optional referent, what its idea is of, tells them apart.
- **Recall.** Extend is the mechanism of recall. A question is extended
  over its own tree like any sentence, and its field of understanding is
  the side it is matched on. One term's result over the rows of memory is
  one product, matched order by order and never reduced across orders
  (`EpisodicMemory.matches`, `relevance`); each operator of the question's
  tree then combines its operands' results as its extend says
  (`ExtendedRecall.matches`): narrowing intersects, joining unions, `not`
  and `non` act on the lanes, and `lower` selects one row. The operand a
  question leaves open selects nothing. A recalled row's idea is walked
  back to its sentence and extended, and the part of its field at the
  operand that stands, under the same operators from the root, where the
  question's open operand stands is the answer (`ExtendedRecall.answer`);
  where the two trees do not agree that far, no answer is read from that
  row. A row keeps only the strongest members of its field, so the
  products find the candidates and the walk gives their whole fields. A
  question with no open operand is read on its region instead, by
  existence and by pervasion
  ([Logic](Logic.md#existence-and-pervasion-alec-2026-11-13)).
- **Reference.** What "the" selected is in the field. Resolving a
  reference is part of extend, not a search made beside it.
- **Saying.** A member that has a word is said by that word. A member
  without one cannot be said, which is the sense of "non-conceptual". Not
  every concept has a symbol: "we do not symbolize all concepts to begin
  with", and `interpret` is what creates one
  ([Language](Language.md#interpret-word-concept-to-object-concept-2026-09-25)).

## Regions (Alec, 2026-11-13)

"Like "cats have fur" should be a relation in a higher order space about
cat-things and things-with-fur. So every intention (e.g. the property
fur) will have an extension when applied to a set (things with fur) which
will look like a region in conceptual space (the furry things region)."

- A property is an intension. Applied to a set it has an extension, the
  things it holds of, and that extension is a region of conceptual space.
- A subject's set is a region as well: one thing, or the members of a
  kind.
- A sentence that says a property of a kind is a relation between the two
  kinds, in the space of the order above, where each kind is one object.
  It says that the first region lies wholly within the second.
- A claim is **pervaded** when its subject's region lies wholly within
  its property's region by such a relation, and it has an **instance**
  when a trusted object lies in both. Only the first carries a deduction
  ([Logic](Logic.md#existence-and-pervasion-alec-2026-11-13)).

## In the code

- The field is a `ConceptualField` (`bin/Understanding.py`): concept
  references, evidence on two lanes, and each reading's place and time.
- The two lanes are combined by `FieldLogic.reduce` (`bin/FieldLogic.py`),
  which defines `intersection`, `union`, `not` and `non` once.
- The operators are the `GrammarLayer` classes of `bin/Language.py`;
  `compose_extend` and `generate_extend` are their methods beside
  `compose` and `generate` (`bin/Layers.py`), carried out by
  `bin/Extension.py`. `Extension.reading` extends the kept STM tree once,
  in its compose order.
- Recall by extension is `bin/ExtendedRecall.py`.
- The centroid of a field is `FieldProjection.key`
  (`bin/FieldProjection.py`).
- `CompoundLayer` narrows inside its compose as well, through
  `CaseSelection.select_cases`.

## Open

- Whether what "a" selects is entered in MTM, so that a later "the" finds
  it: proposed, yes.
